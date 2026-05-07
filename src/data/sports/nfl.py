"""NFL game data adapter — BinaryMarket-based catalog, settlement, snapshots.

The platform's binary-market path is exercised here. One NFL game maps to
one `BinaryMarket`: the YES side is "home team wins". A game is its own
event (no bucket ladder), so `NflCatalog.ladder(event_id)` returns None
and the engine's settlement code routes through the binary fallback.

Data source
-----------
Two parquets under `tests/fixtures/historical_nfl/`:

  - games.parquet     — one row per game with kickoff time, scores, and
                        sportsbook closing moneylines. Powers the catalog
                        and the settlement loader.
  - snapshots.parquet — one orderbook snapshot per game timestamped just
                        before close_time. The yes_bid / yes_ask are
                        derived from the closing home moneyline (with a
                        synthetic spread); these stand in for what we'd
                        otherwise pull from a Kalshi NFL series.

The adapter exposes the same shape as `data/kalshi/loader.py`:
`load_catalog_from_parquet`, `iter_orderbook_snapshots`-equivalent
(`iter_game_snapshots`), and a `MarketCatalog`-implementing class.
Nothing in `core/`, `backtest/`, `data/features/`, or `data/kalshi/`
needs to know NFL exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

from core.market import (
    BinaryMarket,
    BinaryResolution,
    BucketLadder,
    Event,
    EventId,
    Market,
    MarketCatalog,
    MarketId,
    Outcome,
    SeriesId,
)
from core.state import OrderbookSnapshot


SERIES_ID: SeriesId = "KXNFLGAME"

#: Synthetic ticker prefix. Real Kalshi NFL game tickers also follow a
#: KX*-style scheme; we use a constant prefix so loaders relying on the
#: `<SERIES>-<EVENT>` split (`series_id_for`) work the same way.
TICKER_PREFIX = SERIES_ID + "-"


# ---------------------------------------------------------------------------
# Moneyline conversion.
# ---------------------------------------------------------------------------

def moneyline_to_prob(ml: float | int) -> float:
    """American odds -> implied probability.

    Positive: prob = 100 / (ml + 100). Negative: prob = -ml / (-ml + 100).
    Returns NaN for None/NaN input. Includes the bookmaker's vig — pair
    home and away to de-vig if needed.
    """
    if ml is None or pd.isna(ml):
        return float("nan")
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


def devig(p_home: float, p_away: float) -> tuple[float, float]:
    """Strip vig by normalizing the two implied probs to sum to 1."""
    total = p_home + p_away
    if total <= 0 or pd.isna(total):
        return float("nan"), float("nan")
    return p_home / total, p_away / total


# ---------------------------------------------------------------------------
# NflCatalog — implements core.MarketCatalog Protocol.
# ---------------------------------------------------------------------------

@dataclass
class NflCatalog:
    """In-memory NFL catalog. Each game is one BinaryMarket / one Event.

    `ladder(event_id)` always returns None — these are binary events, the
    engine's settlement fallback (`event_market_ids = [event_id]`) handles
    them correctly. We deliberately keep no `ladders_by_event` attribute
    so that any code path that mistakenly iterates ladders on this catalog
    fails loudly.
    """
    markets_by_id: dict[MarketId, Market]
    events_by_id: dict[EventId, Event]

    def markets_for_series(
        self, series_id: SeriesId, t: datetime
    ) -> Iterable[Market]:
        return [
            m for m in self.markets_by_id.values()
            if m.series_id == series_id and m.open_time <= t < m.close_time
        ]

    def market(self, market_id: MarketId) -> Market:
        return self.markets_by_id[market_id]

    def event(self, event_id: EventId) -> Event:
        return self.events_by_id[event_id]

    def ladder(self, event_id: EventId) -> Optional[BucketLadder]:
        return None

    def events_resolving_between(
        self, t_start: datetime, t_end: datetime
    ) -> Iterable[Event]:
        return [
            e for e in self.events_by_id.values()
            if t_start <= e.expiration_time < t_end
        ]


# ---------------------------------------------------------------------------
# Catalog builders — DataFrame -> NflCatalog.
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


def load_catalog_from_dataframe(games_df: pd.DataFrame) -> NflCatalog:
    """Build an NflCatalog from a games DataFrame.

    Required columns:
        ticker, event_ticker, title,
        open_time, close_time, expiration_time

    The fetcher (`scripts/fetch_historical_nfl.py`) writes this schema.
    Each row becomes one BinaryMarket and one Event. Markets that share a
    ticker raise — the fetcher should have deduped already.
    """
    required = ["ticker", "event_ticker", "title", "open_time", "close_time", "expiration_time"]
    missing = [c for c in required if c not in games_df.columns]
    if missing:
        raise KeyError(f"Missing required NFL game columns: {missing}")

    markets_by_id: dict[MarketId, Market] = {}
    events_by_id: dict[EventId, Event] = {}

    for row in games_df.itertuples(index=False):
        ticker = str(row.ticker)
        event_ticker = str(row.event_ticker)

        if ticker in markets_by_id:
            raise ValueError(f"duplicate market ticker {ticker!r}")

        open_time = _to_utc(row.open_time)
        close_time = _to_utc(row.close_time)
        expiration_time = _to_utc(row.expiration_time)
        title = str(getattr(row, "title", "") or "")

        market = BinaryMarket(
            market_id=ticker,
            event_id=event_ticker,
            series_id=SERIES_ID,
            open_time=open_time,
            close_time=close_time,
            expiration_time=expiration_time,
            title=title,
        )
        markets_by_id[ticker] = market

        events_by_id[event_ticker] = Event(
            event_id=event_ticker,
            series_id=SERIES_ID,
            close_time=close_time,
            expiration_time=expiration_time,
            title=title,
        )

    return NflCatalog(markets_by_id=markets_by_id, events_by_id=events_by_id)


def load_catalog_from_parquet(path: str | Path) -> NflCatalog:
    return load_catalog_from_dataframe(pd.read_parquet(path))


# ---------------------------------------------------------------------------
# Snapshot iterator — DataFrame -> OrderbookSnapshot stream.
# ---------------------------------------------------------------------------

def iter_game_snapshots(
    snaps_df: pd.DataFrame,
    *,
    ts_col: str = "pulled_at_utc",
    default_size: int = 100,
) -> Iterator[OrderbookSnapshot]:
    """Yield OrderbookSnapshot rows in time order.

    Schema mirrors the Kalshi loader's iter_orderbook_snapshots so the
    backtest engine consumes both transparently:

        ticker, pulled_at_utc,
        yes_bid_dollars, yes_ask_dollars,
        no_bid_dollars,  no_ask_dollars
    """
    needed = [
        "ticker", ts_col,
        "yes_bid_dollars", "yes_ask_dollars",
        "no_bid_dollars", "no_ask_dollars",
    ]
    missing = [c for c in needed if c not in snaps_df.columns]
    if missing:
        raise KeyError(f"iter_game_snapshots missing columns: {missing}")

    df = snaps_df[needed].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(by=[ts_col, "ticker"]).reset_index(drop=True)

    for row in df.itertuples(index=False):
        ts = row[1]
        ts_dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)

        yes_bid = _coerce(row.yes_bid_dollars)
        yes_ask = _coerce(row.yes_ask_dollars)
        no_bid = _coerce(row.no_bid_dollars)
        no_ask = _coerce(row.no_ask_dollars)

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


def _coerce(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


# ---------------------------------------------------------------------------
# NflSettlementLoader — implements core.SettlementLoader Protocol.
# ---------------------------------------------------------------------------

@dataclass
class NflSettlementLoader:
    """Resolve NFL game outcomes from the games DataFrame.

    YES = home team wins. Tie games (rare in NFL but possible) resolve NO
    on the home line — same convention as Kalshi's "team to win" markets.

    The DataFrame must carry `ticker`, `home_score`, `away_score`,
    `expiration_time`. Games with missing scores are treated as unresolved.
    """

    games_df: pd.DataFrame

    def __post_init__(self) -> None:
        required = ["ticker", "home_score", "away_score", "expiration_time"]
        missing = [c for c in required if c not in self.games_df.columns]
        if missing:
            raise KeyError(f"NflSettlementLoader missing columns: {missing}")
        self._by_id = self.games_df.set_index("ticker")

    def outcome(self, event_id: EventId) -> Outcome | None:
        if event_id not in self._by_id.index:
            return None
        row = self._by_id.loc[event_id]
        h = row["home_score"]
        a = row["away_score"]
        if pd.isna(h) or pd.isna(a):
            return None

        # YES = home win. Use BinaryResolution to record the resolution
        # cleanly; the engine's payout logic compares market_id to
        # winning_market_id, which is None for a binary-NO outcome.
        if float(h) > float(a):
            resolution = BinaryResolution.YES
            winning_market_id: MarketId | None = event_id
        else:
            resolution = BinaryResolution.NO
            winning_market_id = None

        return Outcome(
            event_id=event_id,
            winning_market_id=winning_market_id,
            expiration_value=float(h) - float(a),
            resolved_at=_to_utc(row["expiration_time"]),
            binary_resolution=resolution,
        )
