"""Tests for data.kalshi.loader against real Kalshi data fixtures.

Fixtures captured from a real notebook-02 pull:
  - tests/fixtures/kalshi_markets_26JUN.parquet : 21 rows, one CPI event,
    real `custom_strike` shapes (dict-with-Value), real bid/ask dollars.

These tests guard the parsing-and-bucket-construction pipeline against
the actual edge cases the live API surfaces.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.market import BucketLadder, BucketMarket
from core.state import OrderbookSnapshot
from data.kalshi.loader import (
    EVENT_CODE_REGEX,
    KalshiCatalog,
    build_buckets,
    extract_event_code,
    iter_orderbook_snapshots,
    load_catalog_from_dataframe,
    normalize_market_probs,
    parse_custom_strike,
)


FIXTURES = Path(__file__).parent / "fixtures"
MARKETS_FIXTURE = FIXTURES / "kalshi_markets_26JUN.parquet"


@pytest.fixture(scope="module")
def real_markets() -> pd.DataFrame:
    return pd.read_parquet(MARKETS_FIXTURE)


# ---------------------------------------------------------------------------
# parse_custom_strike — handles every shape Kalshi has been seen to return.
# ---------------------------------------------------------------------------

def test_parse_custom_strike_dict_form():
    assert parse_custom_strike({"Value": "3.5"}) == 3.5


def test_parse_custom_strike_python_repr_str_form():
    # Kalshi historically dumped these as Python-repr strings with single quotes.
    assert parse_custom_strike("{'Value': '3.5'}") == 3.5


def test_parse_custom_strike_proper_json_str_form():
    assert parse_custom_strike('{"Value": "3.5"}') == 3.5


def test_parse_custom_strike_bare_scalar_str():
    assert parse_custom_strike("3.5") == 3.5


def test_parse_custom_strike_numeric_passthrough():
    assert parse_custom_strike(3.5) == 3.5
    assert parse_custom_strike(4) == 4.0


def test_parse_custom_strike_unparseable_returns_nan():
    assert pd.isna(parse_custom_strike("not a number"))
    assert pd.isna(parse_custom_strike("{garbage"))


# ---------------------------------------------------------------------------
# build_buckets — adjacency, -inf/+inf at edges.
# ---------------------------------------------------------------------------

def test_build_buckets_real_data_is_contiguous_with_inf_edges(real_markets):
    bucketed = build_buckets(real_markets)
    assert len(bucketed) > 0

    # Per event_ticker, sorted by strike_value, contiguous, first floor=-inf, last cap=+inf.
    for ev, group in bucketed.groupby("event_ticker"):
        group = group.reset_index(drop=True)
        assert math.isinf(group["floor"].iloc[0]) and group["floor"].iloc[0] < 0, ev
        assert math.isinf(group["cap"].iloc[-1]) and group["cap"].iloc[-1] > 0, ev
        # Contiguity: cap[i] == floor[i+1] for interior pairs.
        for i in range(len(group) - 1):
            assert group["cap"].iloc[i] == group["floor"].iloc[i + 1], (ev, i)


def test_build_buckets_drops_non_custom_rows():
    df = pd.DataFrame([
        {"ticker": "A-T1", "event_ticker": "EV", "strike_type": "custom", "custom_strike": {"Value": "1"}},
        {"ticker": "A-T2", "event_ticker": "EV", "strike_type": "custom", "custom_strike": {"Value": "2"}},
        {"ticker": "B-BIN", "event_ticker": "EV2", "strike_type": "binary", "custom_strike": None},
    ])
    out = build_buckets(df)
    assert "B-BIN" not in out["ticker"].values


def test_build_buckets_raises_on_missing_columns():
    df = pd.DataFrame([{"ticker": "X"}])
    with pytest.raises(KeyError, match="Missing required market columns"):
        build_buckets(df)


# ---------------------------------------------------------------------------
# normalize_market_probs — cents-vs-dollars detection.
# ---------------------------------------------------------------------------

def test_normalize_market_probs_dollars_passthrough():
    df = pd.DataFrame({"ticker": ["A", "B"], "p_market": [0.4, 0.6]})
    out = normalize_market_probs(df)
    assert out["p_market"].tolist() == [0.4, 0.6]


def test_normalize_market_probs_cents_to_dollars():
    df = pd.DataFrame({"ticker": ["A", "B", "C"], "p_market": [40, 60, 55]})
    out = normalize_market_probs(df)
    assert all(p <= 1.0 for p in out["p_market"])
    assert out["p_market"].tolist() == [0.40, 0.60, 0.55]


def test_normalize_market_probs_dedupes_to_latest_per_ticker():
    df = pd.DataFrame({"ticker": ["A", "A", "B"], "p_market": [0.3, 0.5, 0.2]})
    out = normalize_market_probs(df)
    assert len(out) == 2
    a_row = out[out["ticker"] == "A"]
    assert float(a_row["p_market"].iloc[0]) == 0.5  # last value wins


# ---------------------------------------------------------------------------
# extract_event_code — regex behavior on real ticker shapes.
# ---------------------------------------------------------------------------

def test_extract_event_code_real_tickers(real_markets):
    codes = real_markets["ticker"].apply(extract_event_code).dropna().unique()
    assert len(codes) >= 1
    # All tickers in the fixture are 26JUN.
    assert set(codes) == {"26JUN"}


def test_extract_event_code_returns_none_for_unknown_shape():
    assert extract_event_code("RANDOM-TICKER") is None
    assert extract_event_code(None) is None
    assert extract_event_code(42) is None


# ---------------------------------------------------------------------------
# load_catalog_from_dataframe — end-to-end on real data.
# ---------------------------------------------------------------------------

def test_load_catalog_from_real_markets_dataframe(real_markets):
    catalog = load_catalog_from_dataframe(real_markets)
    assert isinstance(catalog, KalshiCatalog)
    assert len(catalog.markets_by_id) == len(real_markets)

    # Exactly one event in the fixture.
    assert set(catalog.events_by_id) == {"KXECONSTATCPIYOY-26JUN"}
    assert set(catalog.ladders_by_event) == {"KXECONSTATCPIYOY-26JUN"}

    ladder = catalog.ladders_by_event["KXECONSTATCPIYOY-26JUN"]
    assert isinstance(ladder, BucketLadder)
    assert len(ladder.buckets) == len(real_markets)


def test_load_catalog_ladder_validates_contiguity(real_markets):
    """BucketLadder's __post_init__ enforces contiguity + -inf/+inf edges.
    If load_catalog_from_dataframe returns a real BucketLadder, those
    invariants are guaranteed to hold."""
    catalog = load_catalog_from_dataframe(real_markets)
    ladder = catalog.ladders_by_event["KXECONSTATCPIYOY-26JUN"]
    # Spot-check: the ladder covers all of R.
    assert ladder.buckets[0].floor == -math.inf
    assert ladder.buckets[-1].cap == math.inf


def test_catalog_markets_for_series_filters_by_active_window(real_markets):
    catalog = load_catalog_from_dataframe(real_markets)
    series_id = "KXECONSTATCPIYOY"

    # Pre-close: every market active.
    t_before = datetime(2026, 5, 1, tzinfo=timezone.utc)
    active = list(catalog.markets_for_series(series_id, t_before))
    assert len(active) == len(real_markets)

    # Post-close: no markets active.
    t_after = datetime(2027, 1, 1, tzinfo=timezone.utc)
    active_after = list(catalog.markets_for_series(series_id, t_after))
    assert len(active_after) == 0


def test_catalog_events_resolving_window(real_markets):
    catalog = load_catalog_from_dataframe(real_markets)
    # The 26JUN event resolves in 2026 (expiration_time around Oct 2026).
    in_window = list(catalog.events_resolving_between(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
    ))
    assert len(in_window) == 1
    assert in_window[0].event_id == "KXECONSTATCPIYOY-26JUN"


# ---------------------------------------------------------------------------
# iter_orderbook_snapshots — yields parsed snapshots from real top-of-book.
# ---------------------------------------------------------------------------

def test_iter_orderbook_snapshots_yields_one_per_row(real_markets):
    snapshots = list(iter_orderbook_snapshots(real_markets))
    assert len(snapshots) == len(real_markets)


def test_iter_orderbook_snapshots_prices_in_unit_interval(real_markets):
    for snap in iter_orderbook_snapshots(real_markets):
        for price in [
            snap.yes_bid_price, snap.yes_ask_price,
            snap.no_bid_price, snap.no_ask_price,
        ]:
            if price is not None:
                assert 0.0 <= price <= 1.0


def test_iter_orderbook_snapshots_default_size_when_quoted(real_markets):
    """A side with a price gets default_size; an unquoted side gets size=0."""
    snaps = list(iter_orderbook_snapshots(real_markets, default_size=42))
    for s in snaps:
        if s.yes_bid_price is not None:
            assert s.yes_bid_size == 42
        else:
            assert s.yes_bid_size == 0


def test_iter_orderbook_snapshots_time_sorted(real_markets):
    snaps = list(iter_orderbook_snapshots(real_markets))
    times = [s.ts for s in snaps]
    assert times == sorted(times)


def test_iter_orderbook_snapshots_raises_on_missing_columns():
    df = pd.DataFrame({"ticker": ["A"]})
    with pytest.raises(KeyError, match="missing columns"):
        list(iter_orderbook_snapshots(df))
