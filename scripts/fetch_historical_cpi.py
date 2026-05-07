"""Fetch historical Kalshi CPI orderbook data for resolved events.

Pulls every settled market in the KXECONSTATCPIYOY series, then walks each
market's daily candlestick history (period_interval=1440) from open_time
through close_time. The candles carry yes_bid / yes_ask OHLC; we use the
close_dollars of each as the end-of-day top-of-book.

Writes three parquets under `tests/fixtures/historical_cpi/`:

  - markets.parquet       — same schema as `data/kalshi_cpi_markets_open.parquet`,
                            consumed by `load_catalog_from_parquet`.
  - snapshots.parquet     — daily orderbook snapshots, schema compatible with
                            `iter_orderbook_snapshots`. One row per (market, day).
  - fred_core_cpi.parquet — multi-year CPILFESL panel covering the strategy's
                            warmup window plus the resolution months. Required
                            for both forecasting and settlement.

Re-runnable: overwrites the parquets each invocation. Network-bound;
takes ~30-60 seconds for the current 32-market universe.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
load_dotenv(_REPO_ROOT / ".env")

from data.features.fred import FRED_SERIES_META, FredClient
from data.kalshi.loader import KalshiClient


SERIES = "KXECONSTATCPIYOY"
FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "historical_cpi"
DAILY_PERIOD_MIN = 1440


def _iso_to_epoch(s: str) -> int:
    return int(pd.Timestamp(s).tz_convert("UTC").timestamp() if pd.Timestamp(s).tz is not None
               else pd.Timestamp(s, tz="UTC").timestamp())


def fetch_markets(client: KalshiClient) -> list[dict]:
    """Pull every finalized market in SERIES.

    `status='settled'` is the API's term for finalized markets; the markets
    endpoint refuses other statuses for this series, so this is the only way
    to enumerate resolved events.
    """
    print(f"[markets] fetching settled markets for {SERIES}...")
    mkts = client.get_markets(SERIES, status="settled", limit=1000)
    print(f"[markets] got {len(mkts)} settled markets")
    events = sorted({m["event_ticker"] for m in mkts})
    print(f"[markets] events: {events}")
    return mkts


def to_markets_df(markets: list[dict]) -> pd.DataFrame:
    """Project the API response into the parquet schema the loader consumes."""
    rows = []
    pulled_at = datetime.now(tz=timezone.utc).isoformat()
    for m in markets:
        rows.append({
            "ticker": m["ticker"],
            "event_ticker": m["event_ticker"],
            "title": m.get("title", "") or m.get("yes_sub_title", ""),
            "close_time": m["close_time"],
            "expected_expiration_time": m.get("expected_expiration_time"),
            "expiration_time": m["expiration_time"],
            "strike_type": m["strike_type"],
            "custom_strike": m.get("custom_strike"),
            # Top-of-book at the moment we pulled — only used as a fallback;
            # the snapshots parquet is the real time-series.
            "yes_bid_dollars": _last_yes_bid(m),
            "yes_ask_dollars": _last_yes_ask(m),
            "no_bid_dollars": _last_no_bid(m),
            "no_ask_dollars": _last_no_ask(m),
            "pulled_series": SERIES,
            "pulled_at_utc": pulled_at,
        })
    return pd.DataFrame(rows)


def _last_yes_bid(m: dict) -> float | None:
    v = m.get("yes_bid")
    return v / 100.0 if isinstance(v, (int, float)) else None


def _last_yes_ask(m: dict) -> float | None:
    v = m.get("yes_ask")
    return v / 100.0 if isinstance(v, (int, float)) else None


def _last_no_bid(m: dict) -> float | None:
    v = m.get("no_bid")
    return v / 100.0 if isinstance(v, (int, float)) else None


def _last_no_ask(m: dict) -> float | None:
    v = m.get("no_ask")
    return v / 100.0 if isinstance(v, (int, float)) else None


def fetch_candles(
    client: KalshiClient, ticker: str, start_ts: int, end_ts: int,
    *, max_attempts: int = 5,
) -> list[dict]:
    """Daily candles for a single market. Retries on 429 with exponential backoff."""
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return client.get_candlesticks(
                series_ticker=SERIES,
                ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_minutes=DAILY_PERIOD_MIN,
            )
        except Exception as exc:
            msg = str(exc)
            if "429" in msg and attempt < max_attempts:
                print(f"[candles] {ticker}: 429, sleeping {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"[candles] {ticker}: ERROR {exc}")
            return []
    return []


def candles_to_snapshots(ticker: str, candles: list[dict]) -> list[dict]:
    """Convert the candlestick OHLC into OrderbookSnapshot-shaped rows.

    The snapshot timestamp is the candle's end_period_ts; bids/asks are the
    close_dollars of the yes_bid / yes_ask blocks. The NO side is derived
    from the YES side (no_bid = 1 - yes_ask, no_ask = 1 - yes_bid). Candles
    where bid==0 and ask==1 are kept — they represent untraded days where
    the strategy still ticks; the strategy can choose to ignore them via
    snapshot.yes_mid being None / extreme.
    """
    rows = []
    for c in candles:
        end_ts = c.get("end_period_ts")
        yb = c.get("yes_bid", {}) or {}
        ya = c.get("yes_ask", {}) or {}
        try:
            yes_bid = float(yb.get("close_dollars", "nan"))
            yes_ask = float(ya.get("close_dollars", "nan"))
        except (TypeError, ValueError):
            continue

        # Convert "no quotes" candles (bid=0, ask=1) into NaN so the strategy
        # treats them as missing rather than as a 0/1 spread.
        if yes_bid == 0.0 and yes_ask == 0.0:
            yes_bid = float("nan")
            yes_ask = float("nan")

        if yes_ask is not None and yes_ask > 0 and yes_ask <= 1:
            no_bid = 1.0 - yes_ask
        else:
            no_bid = float("nan")
        if yes_bid is not None and yes_bid >= 0 and yes_bid <= 1:
            no_ask = 1.0 - yes_bid
        else:
            no_ask = float("nan")

        rows.append({
            "ticker": ticker,
            "pulled_at_utc": pd.Timestamp(end_ts, unit="s", tz="UTC").isoformat(),
            "yes_bid_dollars": yes_bid,
            "yes_ask_dollars": yes_ask,
            "no_bid_dollars": no_bid,
            "no_ask_dollars": no_ask,
        })
    return rows


def fetch_all_snapshots(
    client: KalshiClient, markets: list[dict]
) -> pd.DataFrame:
    out: list[dict] = []
    for i, m in enumerate(markets, 1):
        ticker = m["ticker"]
        start_ts = _iso_to_epoch(m["open_time"])
        end_ts = _iso_to_epoch(m["close_time"])
        candles = fetch_candles(client, ticker, start_ts, end_ts)
        rows = candles_to_snapshots(ticker, candles)
        out.extend(rows)
        print(f"[candles] {i}/{len(markets)} {ticker}: {len(candles)} candles, {len(rows)} usable")
        # Be polite to the API — Kalshi rate-limits ~10 req/s for unauthenticated reads.
        time.sleep(0.2)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["pulled_at_utc"] = pd.to_datetime(df["pulled_at_utc"], utc=True)
    return df.sort_values(["pulled_at_utc", "ticker"]).reset_index(drop=True)


def fetch_fred_panel() -> pd.DataFrame:
    """Multi-year CPILFESL series — enough warmup for YoY std-dev fitting."""
    print("[fred] fetching CPILFESL...")
    client = FredClient()
    series = client.fetch_series(
        FRED_SERIES_META["core_cpi"], start_date="2010-01-01"
    )
    df = pd.DataFrame({"core_cpi": series})
    df.index.name = "date"
    print(f"[fred] {len(df)} obs, {df.index.min()} -> {df.index.max()}")
    return df


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    kc = KalshiClient()
    markets = fetch_markets(kc)

    markets_df = to_markets_df(markets)
    markets_path = FIXTURE_DIR / "markets.parquet"
    markets_df.to_parquet(markets_path, index=False)
    print(f"[write] markets -> {markets_path} ({len(markets_df)} rows)")

    snaps_df = fetch_all_snapshots(kc, markets)
    snaps_path = FIXTURE_DIR / "snapshots.parquet"
    snaps_df.to_parquet(snaps_path, index=False)
    print(f"[write] snapshots -> {snaps_path} ({len(snaps_df)} rows)")

    fred_df = fetch_fred_panel()
    fred_path = FIXTURE_DIR / "fred_core_cpi.parquet"
    fred_df.to_parquet(fred_path)
    print(f"[write] fred -> {fred_path} ({len(fred_df)} rows)")


if __name__ == "__main__":
    main()
