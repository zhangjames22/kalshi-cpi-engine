"""Stub data layer — pure-Python in-memory implementations of the catalog,
settlement loader, feature view, and snapshot stream.

Used by the end-to-end pipeline test to validate that the engine wires up
correctly. Real Kalshi / FRED loaders will live in sibling modules
(`data/kalshi/`, `data/features/`) and conform to the same Protocols, so
nothing in `core/` or `backtest/` cares which one is plugged in.

CPI-shaped scenario builder
---------------------------
`build_cpi_scenario` constructs everything you need to simulate a single
CPI-YoY release event with a Kalshi-style bucket ladder and a synthetic
FRED-like feature history.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from core.market import (
    BucketLadder,
    BucketMarket,
    Event,
    EventId,
    Market,
    MarketCatalog,
    MarketId,
    Outcome,
    SeriesId,
    SettlementLoader,
)
from core.state import OrderbookSnapshot
from core.strategy import FeatureView


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@dataclass
class InMemoryCatalog:
    """Trivial MarketCatalog backed by dicts. Constructed once at the top of
    a test/scenario; immutable thereafter by convention."""

    markets_by_id: dict[MarketId, Market]
    events_by_id: dict[EventId, Event]
    ladders_by_event: dict[EventId, BucketLadder]

    def markets_for_series(
        self, series_id: SeriesId, t: datetime
    ) -> Iterable[Market]:
        return [
            m for m in self.markets_by_id.values()
            if m.series_id == series_id
            and m.open_time <= t < m.close_time
        ]

    def market(self, market_id: MarketId) -> Market:
        return self.markets_by_id[market_id]

    def event(self, event_id: EventId) -> Event:
        return self.events_by_id[event_id]

    def ladder(self, event_id: EventId) -> BucketLadder | None:
        return self.ladders_by_event.get(event_id)

    def events_resolving_between(
        self, t_start: datetime, t_end: datetime
    ) -> Iterable[Event]:
        return [
            e for e in self.events_by_id.values()
            if t_start <= e.expiration_time < t_end
        ]


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

@dataclass
class InMemorySettlementLoader:
    """SettlementLoader backed by a dict. Pre-populate with the outcomes
    you want; the engine will ask for them as events resolve."""

    outcomes_by_event: dict[EventId, Outcome]

    def outcome(self, event_id: EventId) -> Outcome | None:
        return self.outcomes_by_event.get(event_id)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class InMemoryFeatureView:
    """FeatureView backed by a {feature_name: [(available_at, value), ...]} map.

    Series are kept sorted by `available_at`. `get` returns the latest value
    with available_at <= t; `get_series` returns the truncated list of values.
    """

    series: dict[str, list[tuple[datetime, float]]]

    def get(self, name: str, t: datetime) -> Any:
        s = self.series.get(name, [])
        latest: float | None = None
        for available_at, value in s:
            if available_at <= t:
                latest = value
            else:
                break
        return latest

    def get_series(self, name: str, t: datetime) -> list[float]:
        s = self.series.get(name, [])
        return [v for available_at, v in s if available_at <= t]


# ---------------------------------------------------------------------------
# Synthetic snapshot generator
# ---------------------------------------------------------------------------

def make_uniform_snapshot(
    market_id: MarketId,
    ts: datetime,
    p_yes: float,
    spread: float = 0.04,
    size: int = 100,
) -> OrderbookSnapshot:
    """Build a snapshot whose YES top-of-book brackets `p_yes` symmetrically.

    Useful for tests where you want the market to "show" a particular
    implied probability. NO side is set to (1 - p_yes) symmetrically.
    """
    half = spread / 2.0
    yes_bid = max(0.01, p_yes - half)
    yes_ask = min(0.99, p_yes + half)
    no_bid = max(0.01, (1 - p_yes) - half)
    no_ask = min(0.99, (1 - p_yes) + half)
    return OrderbookSnapshot(
        market_id=market_id,
        ts=ts,
        yes_bid_price=yes_bid, yes_bid_size=size,
        yes_ask_price=yes_ask, yes_ask_size=size,
        no_bid_price=no_bid, no_bid_size=size,
        no_ask_price=no_ask, no_ask_size=size,
    )


# ---------------------------------------------------------------------------
# CPI scenario builder
# ---------------------------------------------------------------------------

@dataclass
class CpiScenario:
    """Everything you need to run a CPI-shaped backtest end-to-end."""

    catalog: InMemoryCatalog
    settlement: InMemorySettlementLoader
    feature_view: InMemoryFeatureView
    snapshots: list[OrderbookSnapshot]
    event_id: EventId
    series_id: SeriesId
    realized_value: float


def build_cpi_scenario(
    *,
    series_id: SeriesId = "SYNCPI",
    event_id: EventId = "SYNCPI-26NOV",
    edges: list[float] | None = None,
    open_time: datetime | None = None,
    close_time: datetime | None = None,
    expiration_time: datetime | None = None,
    snapshot_interval: timedelta = timedelta(days=1),
    realized_value: float = 3.27,
    market_implied_mean: float = 3.0,
    market_implied_sigma: float = 0.4,
    feature_history_len: int = 240,
    feature_mean: float = 3.0,
    feature_sigma: float = 0.5,
    seed: int = 7,
) -> CpiScenario:
    """Build one CPI-YoY event with a bucket ladder, a sequence of orderbook
    snapshots, a synthetic core_cpi_yoy feature history, and a settlement.

    The ladder edges default to [2.5, 3.0, 3.5, 4.0] — five buckets, mass
    concentrated near 3%. Snapshots are issued for every market on every
    `snapshot_interval` between open_time and close_time. Market-implied
    probabilities follow Normal(market_implied_mean, market_implied_sigma).
    """
    rng = random.Random(seed)

    if edges is None:
        edges = [2.5, 3.0, 3.5, 4.0]
    if open_time is None:
        open_time = datetime(2026, 10, 1, tzinfo=timezone.utc)
    if close_time is None:
        close_time = datetime(2026, 11, 10, tzinfo=timezone.utc)
    if expiration_time is None:
        expiration_time = datetime(2026, 11, 15, tzinfo=timezone.utc)

    # ---- catalog: ladder + buckets ----
    bounds = [-math.inf, *edges, math.inf]
    buckets: list[BucketMarket] = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        strike = hi if hi != math.inf else lo
        # Use bucket index in market_id so unbounded ends don't collide with
        # the adjacent finite bucket (both might pick the same edge as strike).
        buckets.append(
            BucketMarket(
                market_id=f"{event_id}-B{i:02d}",
                event_id=event_id,
                series_id=series_id,
                open_time=open_time,
                close_time=close_time,
                expiration_time=expiration_time,
                title=f"CPI YoY in [{lo}, {hi})",
                floor=lo,
                cap=hi,
                strike_value=strike,
            )
        )
    ladder = BucketLadder(event_id=event_id, buckets=tuple(buckets))
    event = Event(
        event_id=event_id,
        series_id=series_id,
        close_time=close_time,
        expiration_time=expiration_time,
        title="Synthetic CPI YoY release",
    )
    catalog = InMemoryCatalog(
        markets_by_id={b.market_id: b for b in buckets},
        events_by_id={event_id: event},
        ladders_by_event={event_id: ladder},
    )

    # ---- snapshots: market-implied normal projected onto the ladder ----
    market_cdf = _normal_cdf(market_implied_mean, market_implied_sigma)
    base_probs = ladder.project(market_cdf)
    snapshots: list[OrderbookSnapshot] = []
    t = open_time
    while t <= close_time:
        for b in buckets:
            p = base_probs[b.market_id]
            # Add small per-tick noise so spreads aren't pathologically constant.
            p_noisy = max(0.01, min(0.99, p + rng.gauss(0, 0.005)))
            snapshots.append(make_uniform_snapshot(b.market_id, t, p_noisy))
        t += snapshot_interval
    snapshots.sort(key=lambda s: s.ts)

    # ---- features: synthetic core_cpi_yoy series anchored at feature_mean ----
    feature_series: list[tuple[datetime, float]] = []
    monthly = open_time - timedelta(days=feature_history_len * 30)
    val = feature_mean
    for _ in range(feature_history_len):
        val = val + rng.gauss(0, feature_sigma * 0.05)
        feature_series.append((monthly, val))
        monthly += timedelta(days=30)
    feature_view = InMemoryFeatureView(series={"core_cpi_yoy": feature_series})

    # ---- settlement: realized_value lands in some bucket ----
    winning_market_id = ladder.winning_market(realized_value)
    outcome = Outcome(
        event_id=event_id,
        winning_market_id=winning_market_id,
        expiration_value=realized_value,
        resolved_at=expiration_time,
    )
    settlement = InMemorySettlementLoader(outcomes_by_event={event_id: outcome})

    return CpiScenario(
        catalog=catalog,
        settlement=settlement,
        feature_view=feature_view,
        snapshots=snapshots,
        event_id=event_id,
        series_id=series_id,
        realized_value=realized_value,
    )


# ---------------------------------------------------------------------------
# Internal — normal CDF without depending on core.forecast.
# ---------------------------------------------------------------------------

def _normal_cdf(mu: float, sigma: float):
    sqrt2 = math.sqrt(2.0)
    return lambda x: 0.5 * (1.0 + math.erf((x - mu) / (sigma * sqrt2)))
