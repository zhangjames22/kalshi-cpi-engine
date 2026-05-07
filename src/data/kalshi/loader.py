"""Real Kalshi historical loader.

Replaces `data.stub.InMemoryCatalog` and the synthetic snapshot stream with
a loader that reads the same shape of data Kalshi's REST API returns
(per notebook 02). The catalog and the snapshot iterator both implement the
Protocols defined in `src/core/`, so nothing in `core/` or `backtest/` needs
to know we've swapped from synthetic data to real.

Two ways to populate the loader:

1. **From cached parquet** — `load_catalog_from_parquet(path)` reads the
   markets DataFrame written by notebook 02 (or any equivalent puller) and
   builds the catalog. `iter_orderbook_snapshots(df)` produces an
   OrderbookSnapshot stream from the same DataFrame's pre-computed
   top-of-book columns.

2. **From the live API** — `KalshiClient` wraps the public Kalshi
   trade-api/v2 endpoints. `KalshiClient.get_markets(series_ticker)` returns
   the markets list as dicts; the same `load_catalog_from_dataframe` builds
   the catalog. Candlesticks fetching is wired up too for denser historical
   replay, but the v1 default path uses the simpler single-snapshot pulls.

Utilities ported from the earlier surprise-engine reference:

- `parse_custom_strike(x)` — Kalshi's `custom_strike` field is sometimes a
  dict, sometimes a JSON-ish string, sometimes a scalar. One function to
  rule them all.
- `build_buckets(mk)` — derives (floor, cap) per bucket from adjacent
  strikes within an event, with -inf at the bottom and +inf at the top.
  Returns a tidy DataFrame the catalog builder consumes.
- `normalize_market_probs(p_mkt)` — auto-detects cents vs dollars and
  normalizes to [0, 1].
- `EVENT_CODE_REGEX` — the `KXECONSTATCPIYOY-<code>-...` regex used to
  extract event codes from tickers.

These belong here, not in `core/` (Kalshi-specific quirks) and not in
`strategies/` (data-loading concerns are loader concerns).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import requests

from core.market import (
    BucketLadder,
    BucketMarket,
    Event,
    EventId,
    Market,
    MarketCatalog,
    MarketId,
    SeriesId,
)
from core.state import OrderbookSnapshot


EVENT_CODE_REGEX = r"KXECONSTATCPIYOY-([^-]+)-"


# ---------------------------------------------------------------------------
# Utilities — ported from the surprise-engine reference. Kalshi-specific
# parsing quirks belong here, not in core or strategies.
# ---------------------------------------------------------------------------

def parse_custom_strike(x: Any) -> float:
    """Parse Kalshi's `custom_strike` field.

    Real-world shapes seen:
      - dict: {"Value": "3.5"}
      - str:  "{'Value': '3.5'}"        (single-quoted Python repr)
      - str:  '{"Value": "3.5"}'        (proper JSON)
      - str:  "3.5"                     (bare scalar)
      - float/int: 3.5                  (already parsed)

    Returns NaN for unparseable inputs rather than raising — the caller
    drops NaN rows in `build_buckets`.
    """
    if isinstance(x, dict):
        val = x.get("Value")
    elif isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and "Value" in s:
            try:
                val = json.loads(s.replace("'", '"')).get("Value")
            except Exception:
                return float("nan")
        else:
            val = s
    else:
        val = x
    return pd.to_numeric(val, errors="coerce")


def build_buckets(mk: pd.DataFrame) -> pd.DataFrame:
    """Add (strike_value, floor, cap) columns to a Kalshi markets DataFrame.

    Per-event:
      - sort by strike_value ascending
      - floor[i] = strike_value[i]
      - cap[i]   = strike_value[i+1]
      - first floor overridden to -inf, last cap overridden to +inf

    Drops rows where `strike_type` is not "custom" (binary markets and other
    non-numeric-ladder series) and rows where `custom_strike` is unparseable.
    Returns a fresh DataFrame; does not mutate `mk`.
    """
    required = ["ticker", "event_ticker", "custom_strike", "strike_type"]
    missing = [c for c in required if c not in mk.columns]
    if missing:
        raise KeyError(f"Missing required market columns: {missing}")

    out = mk[mk["strike_type"].astype(str).str.lower() == "custom"].copy()
    out["strike_value"] = out["custom_strike"].apply(parse_custom_strike)
    out = out.dropna(subset=["strike_value"]).copy()
    out = (
        out.sort_values(by=["event_ticker", "strike_value", "ticker"])
        .reset_index(drop=True)
    )

    # Cast to float64 up-front so int-strike inputs don't raise on the
    # -inf/+inf assignment below (pandas now warns/errors on dtype mismatch).
    out["floor"] = out["strike_value"].astype("float64")
    out["cap"] = out.groupby("event_ticker")["strike_value"].shift(-1).astype("float64")

    first_idx = out.groupby("event_ticker").head(1).index
    last_idx = out.groupby("event_ticker").tail(1).index
    out.loc[first_idx, "floor"] = -np.inf
    out.loc[last_idx, "cap"] = np.inf

    return out


def normalize_market_probs(p_mkt: pd.DataFrame) -> pd.DataFrame:
    """Normalize a market-implied-probability DataFrame.

    - Convert `p_market` to numeric.
    - If the bulk of values exceed 1, divide by 100 (cents -> dollars).
    - Dedupe to the latest snapshot per ticker.

    Returns a fresh DataFrame with columns ['ticker', 'p_market'].
    """
    if "ticker" not in p_mkt.columns or "p_market" not in p_mkt.columns:
        raise KeyError("Expected ['ticker', 'p_market'] columns.")

    out = p_mkt[["ticker", "p_market"]].copy()
    out["p_market"] = pd.to_numeric(out["p_market"], errors="coerce")

    if out["p_market"].dropna().gt(1).mean() > 0.5:
        out["p_market"] = out["p_market"] / 100.0

    return out.drop_duplicates(subset=["ticker"], keep="last")


def extract_event_code(ticker: str) -> Optional[str]:
    """Extract the YYMMM event code (e.g. '26NOV') from a CPI YoY ticker."""
    if not isinstance(ticker, str):
        return None
    m = re.search(EVENT_CODE_REGEX, ticker)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Kalshi HTTP client — small, focused, no auth required for read endpoints.
# ---------------------------------------------------------------------------

class KalshiClient:
    """Thin wrapper around the public Kalshi trade-api/v2 endpoints we use.

    Stateless except for the requests Session. Constructed once per script.
    Re-uses notebook 02's pagination pattern verbatim.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, base_url: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url or self.BASE_URL
        self.session = session or requests.Session()
        self.session.headers.update({"accept": "application/json"})

    def get_markets(
        self,
        series_ticker: str,
        status: str = "open",
        limit: int = 1000,
        max_pages: int = 50,
        timeout: int = 30,
    ) -> list[dict]:
        """Paginated GET /markets for a series. Returns a list of market dicts."""
        out: list[dict] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            params: dict[str, Any] = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            r = self.session.get(f"{self.base_url}/markets", params=params, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            out.extend(j.get("markets", []))
            cursor = j.get("cursor") or None
            if not cursor:
                break
        return out

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_minutes: int = 60,
        timeout: int = 30,
    ) -> list[dict]:
        """GET /series/{series}/markets/{ticker}/candlesticks.

        `start_ts`/`end_ts` are Unix epoch seconds; `period_minutes` ∈ {1, 60, 1440}.
        Returns the `candlesticks` array, each candle carrying `yes_bid` and
        `yes_ask` OHLC sub-objects we can convert to OrderbookSnapshot.
        """
        url = (
            f"{self.base_url}/series/{series_ticker}"
            f"/markets/{ticker}/candlesticks"
        )
        params = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_minutes),
        }
        r = self.session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json().get("candlesticks", [])


# ---------------------------------------------------------------------------
# KalshiCatalog — implements core.MarketCatalog Protocol.
# ---------------------------------------------------------------------------

@dataclass
class KalshiCatalog:
    """In-memory catalog of Kalshi markets/events/ladders.

    Implements `core.MarketCatalog` Protocol. Construct via
    `load_catalog_from_dataframe` or `load_catalog_from_parquet` rather than
    instantiating directly — the from-* helpers do the bucket math.
    """
    markets_by_id: dict[MarketId, Market]
    events_by_id: dict[EventId, Event]
    ladders_by_event: dict[EventId, BucketLadder]

    def markets_for_series(self, series_id: SeriesId, t: datetime) -> Iterable[Market]:
        return [
            m for m in self.markets_by_id.values()
            if m.series_id == series_id
            and m.open_time <= t < m.close_time
        ]

    def market(self, market_id: MarketId) -> Market:
        return self.markets_by_id[market_id]

    def event(self, event_id: EventId) -> Event:
        return self.events_by_id[event_id]

    def ladder(self, event_id: EventId) -> Optional[BucketLadder]:
        return self.ladders_by_event.get(event_id)

    def events_resolving_between(
        self, t_start: datetime, t_end: datetime
    ) -> Iterable[Event]:
        return [
            e for e in self.events_by_id.values()
            if t_start <= e.expiration_time < t_end
        ]


# ---------------------------------------------------------------------------
# Catalog builders — DataFrame -> KalshiCatalog.
# ---------------------------------------------------------------------------

def _to_utc(value: Any) -> datetime:
    """Coerce ISO-string / pandas Timestamp / datetime to a tz-aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, pd.Timestamp):
        if value.tz is None:
            value = value.tz_localize("UTC")
        return value.to_pydatetime()
    return pd.Timestamp(value, tz="UTC").to_pydatetime()


def _series_id_for(ticker: str, event_ticker: str) -> SeriesId:
    """Derive series_ticker from market ticker / event_ticker.

    Kalshi tickers are <SERIES>-<EVENT_CODE>-<STRIKE>. Event tickers are
    <SERIES>-<EVENT_CODE>. So the series is the prefix before the first '-'.
    """
    src = event_ticker or ticker
    return src.split("-", 1)[0]


def load_catalog_from_dataframe(markets_df: pd.DataFrame) -> KalshiCatalog:
    """Build a KalshiCatalog from a markets DataFrame.

    Expected columns (notebook 02 schema):
        ticker, event_ticker, title, close_time, expiration_time,
        strike_type, custom_strike

    Markets without `strike_type == "custom"` are skipped (binary single-market
    events would require `BinaryMarket` instead — wire that up when we add a
    binary series).
    """
    bucketed = build_buckets(markets_df)

    open_time_default = datetime(1970, 1, 1, tzinfo=timezone.utc)

    markets_by_id: dict[MarketId, Market] = {}
    events_by_id: dict[EventId, Event] = {}
    buckets_by_event: dict[EventId, list[BucketMarket]] = {}

    for row in bucketed.itertuples(index=False):
        ticker = str(row.ticker)
        event_ticker = str(row.event_ticker)
        series_id = _series_id_for(ticker, event_ticker)

        close_time = _to_utc(row.close_time)
        expiration_time = _to_utc(row.expiration_time)
        title = str(getattr(row, "title", "") or "")

        floor_v = float(row.floor)
        cap_v = float(row.cap)
        # `np.inf`/`-np.inf` from numpy convert cleanly to math.inf / -math.inf
        # when crossed through float(); BucketMarket's __post_init__ accepts both.
        strike_v = float(row.strike_value)

        market = BucketMarket(
            market_id=ticker,
            event_id=event_ticker,
            series_id=series_id,
            open_time=open_time_default,
            close_time=close_time,
            expiration_time=expiration_time,
            title=title,
            floor=floor_v,
            cap=cap_v,
            strike_value=strike_v,
        )
        markets_by_id[ticker] = market
        buckets_by_event.setdefault(event_ticker, []).append(market)

        if event_ticker not in events_by_id:
            events_by_id[event_ticker] = Event(
                event_id=event_ticker,
                series_id=series_id,
                close_time=close_time,
                expiration_time=expiration_time,
                title=title,
            )

    ladders_by_event: dict[EventId, BucketLadder] = {}
    for ev_id, buckets in buckets_by_event.items():
        # build_buckets already sorted by strike_value ascending within event,
        # so the floor/cap chain is contiguous. BucketLadder will validate.
        ladders_by_event[ev_id] = BucketLadder(
            event_id=ev_id, buckets=tuple(buckets),
        )

    return KalshiCatalog(
        markets_by_id=markets_by_id,
        events_by_id=events_by_id,
        ladders_by_event=ladders_by_event,
    )


def load_catalog_from_parquet(path: str | Path) -> KalshiCatalog:
    """Convenience: read the markets parquet and build the catalog."""
    df = pd.read_parquet(path)
    return load_catalog_from_dataframe(df)


# ---------------------------------------------------------------------------
# Orderbook snapshot iterator — DataFrame -> OrderbookSnapshot stream.
# ---------------------------------------------------------------------------

def _coerce_price(v: Any) -> Optional[float]:
    """Parse a price field; return None for null/empty/unparseable."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def iter_orderbook_snapshots(
    markets_df: pd.DataFrame,
    *,
    ts_col: str = "pulled_at_utc",
    default_size: int = 100,
) -> Iterator[OrderbookSnapshot]:
    """Yield OrderbookSnapshot from a markets DataFrame, time-sorted.

    Uses pre-computed top-of-book columns:
        yes_bid_dollars, yes_ask_dollars, no_bid_dollars, no_ask_dollars

    The notebook-02 markets parquet doesn't carry bid/ask sizes, so we
    populate `*_size` with `default_size`. This is fine for backtesting
    a no-trade strategy (CPI-Normal). Strategies that actually trade should
    read the deeper orderbook parquet instead.
    """
    needed = [
        "ticker", ts_col,
        "yes_bid_dollars", "yes_ask_dollars",
        "no_bid_dollars", "no_ask_dollars",
    ]
    missing = [c for c in needed if c not in markets_df.columns]
    if missing:
        raise KeyError(f"iter_orderbook_snapshots missing columns: {missing}")

    df = markets_df[needed].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(by=[ts_col, "ticker"]).reset_index(drop=True)

    for row in df.itertuples(index=False):
        ts = row[1]
        ts_dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)

        yes_bid = _coerce_price(row.yes_bid_dollars)
        yes_ask = _coerce_price(row.yes_ask_dollars)
        no_bid = _coerce_price(row.no_bid_dollars)
        no_ask = _coerce_price(row.no_ask_dollars)

        yield OrderbookSnapshot(
            market_id=str(row.ticker),
            ts=ts_dt,
            yes_bid_price=yes_bid,
            yes_bid_size=default_size if yes_bid is not None else 0,
            yes_ask_price=yes_ask,
            yes_ask_size=default_size if yes_ask is not None else 0,
            no_bid_price=no_bid,
            no_bid_size=default_size if no_bid is not None else 0,
            no_ask_price=no_ask,
            no_ask_size=default_size if no_ask is not None else 0,
        )
