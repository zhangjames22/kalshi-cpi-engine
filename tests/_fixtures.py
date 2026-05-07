"""Shared test fixtures — small Kalshi-shaped fakes."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from core.market import (
    BinaryMarket,
    BucketLadder,
    BucketMarket,
    Event,
)


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T_CLOSE = datetime(2026, 12, 10, tzinfo=timezone.utc)
T_EXPIRY = datetime(2027, 3, 11, tzinfo=timezone.utc)


def make_bucket(
    market_id: str,
    floor: float,
    cap: float,
    strike_value: float | None = None,
    event_id: str = "EVT-26NOV",
    series_id: str = "SERIES",
) -> BucketMarket:
    return BucketMarket(
        market_id=market_id,
        event_id=event_id,
        series_id=series_id,
        open_time=T0,
        close_time=T_CLOSE,
        expiration_time=T_EXPIRY,
        floor=floor,
        cap=cap,
        strike_value=strike_value if strike_value is not None else cap,
    )


def make_cpi_like_ladder(
    edges: list[float],
    event_id: str = "EVT-26NOV",
) -> BucketLadder:
    """Build a ladder from interior edges. e.g. edges=[2.0, 2.5, 3.0] yields
    buckets (-inf, 2.0), [2.0, 2.5), [2.5, 3.0), [3.0, +inf)."""
    bounds = [-math.inf, *edges, math.inf]
    buckets = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        buckets.append(
            make_bucket(
                market_id=f"{event_id}-T{i}",
                floor=lo,
                cap=hi,
                strike_value=hi if hi != math.inf else lo,
                event_id=event_id,
            )
        )
    return BucketLadder(event_id=event_id, buckets=tuple(buckets))


def make_binary(market_id: str = "MOON-LANDING-2027") -> BinaryMarket:
    return BinaryMarket(
        market_id=market_id,
        event_id=market_id,
        series_id="MOON",
        open_time=T0,
        close_time=T_CLOSE,
        expiration_time=T_EXPIRY,
        title="Will humans land on the Moon in 2027?",
    )


def make_event(event_id: str = "EVT-26NOV", series_id: str = "SERIES") -> Event:
    return Event(
        event_id=event_id,
        series_id=series_id,
        close_time=T_CLOSE,
        expiration_time=T_EXPIRY,
    )
