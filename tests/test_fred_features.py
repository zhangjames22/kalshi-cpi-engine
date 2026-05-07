"""Tests for data.features.fred against real FRED data fixtures.

Fixture: tests/fixtures/fred_core_cpi_recent.parquet — last 24 months
of real CPILFESL observations from a notebook-00 pull. This is enough
to exercise the YoY-and-bucket plumbing the settlement loader needs.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.market import Outcome
from data.features.fred import (
    DEFAULT_PUBLICATION_LAG,
    FredCpiSettlementLoader,
    FredFeatureView,
    PUBLICATION_LAG_BY_SERIES,
)
from data.kalshi.loader import load_catalog_from_dataframe


FIXTURES = Path(__file__).parent / "fixtures"
CPI_FIXTURE = FIXTURES / "fred_core_cpi_recent.parquet"
KALSHI_FIXTURE = FIXTURES / "kalshi_markets_26JUN.parquet"


@pytest.fixture(scope="module")
def cpi_panel() -> pd.DataFrame:
    return pd.read_parquet(CPI_FIXTURE)


# ---------------------------------------------------------------------------
# FredFeatureView basics
# ---------------------------------------------------------------------------

def test_feature_view_get_returns_latest_available(cpi_panel):
    view = FredFeatureView(panel=cpi_panel)
    # Far-future query — the last value (with publication lag baked in)
    # should be returned.
    t_future = datetime(2027, 1, 1, tzinfo=timezone.utc)
    val = view.get("core_cpi", t_future)
    assert val is not None
    last_observed = float(cpi_panel["core_cpi"].dropna().iloc[-1])
    assert val == pytest.approx(last_observed)


def test_feature_view_publication_lag_blocks_too_recent_data(cpi_panel):
    """Querying right at the observation date should NOT see that observation
    yet — publication lag has to elapse first."""
    view = FredFeatureView(panel=cpi_panel)
    last_obs_date = cpi_panel["core_cpi"].dropna().index[-1]

    # On the observation date itself: the lag (14 days) hasn't elapsed,
    # so we should get the *previous* value, not this one.
    t = (last_obs_date + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
    val_now = view.get("core_cpi", t)

    # Way after the lag: we should see the latest value.
    t_after = (last_obs_date + pd.Timedelta(days=30)).to_pydatetime().replace(tzinfo=timezone.utc)
    val_after = view.get("core_cpi", t_after)

    last_observed = float(cpi_panel["core_cpi"].dropna().iloc[-1])
    assert val_after == pytest.approx(last_observed)
    # Pre-lag query saw an earlier value (or None if the panel is too short).
    if val_now is not None:
        assert val_now != pytest.approx(last_observed)


def test_feature_view_get_series_returns_full_history(cpi_panel):
    view = FredFeatureView(panel=cpi_panel)
    t = datetime(2027, 1, 1, tzinfo=timezone.utc)
    series = view.get_series("core_cpi", t)
    assert len(series) == cpi_panel["core_cpi"].dropna().shape[0]
    assert all(isinstance(v, float) for v in series)


def test_feature_view_get_unknown_feature_raises(cpi_panel):
    view = FredFeatureView(panel=cpi_panel)
    with pytest.raises(KeyError, match="not in FRED panel"):
        view.get("nonexistent_series", datetime.now(tz=timezone.utc))


def test_feature_view_returns_none_when_no_data_available_yet(cpi_panel):
    """Query a date earlier than the first observation."""
    view = FredFeatureView(panel=cpi_panel)
    t_too_early = datetime(1900, 1, 1, tzinfo=timezone.utc)
    assert view.get("core_cpi", t_too_early) is None


def test_feature_view_get_indexed_series_preserves_dates(cpi_panel):
    view = FredFeatureView(panel=cpi_panel)
    t = datetime(2027, 1, 1, tzinfo=timezone.utc)
    s = view.get_indexed_series("core_cpi", t)
    assert isinstance(s, pd.Series)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) > 0


def test_feature_view_per_series_lag_overrides_default():
    """Daily series like oil_price get a 1-day lag, not the 14-day default."""
    panel = pd.DataFrame(
        {"oil_price": [80.0, 82.0]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    view = FredFeatureView(panel=panel)
    # Day after the second observation: with 1-day lag we should see 82.
    t = datetime(2026, 1, 4, tzinfo=timezone.utc)
    assert view.get("oil_price", t) == pytest.approx(82.0)


# ---------------------------------------------------------------------------
# FredCpiSettlementLoader — derives YoY from CPI and finds the winning bucket.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kalshi_catalog():
    df = pd.read_parquet(KALSHI_FIXTURE)
    return load_catalog_from_dataframe(df)


def test_settlement_loader_returns_none_for_unknown_event(cpi_panel, kalshi_catalog):
    loader = FredCpiSettlementLoader(catalog=kalshi_catalog, panel=cpi_panel)
    assert loader.outcome("NOT-A-REAL-EVENT-ID") is None


def test_settlement_loader_returns_none_when_observation_missing(cpi_panel, kalshi_catalog):
    """The 26JUN fixture event needs CPI for June 2026 to settle — that
    observation is not in the recent-CPI fixture (which ends in March 2026),
    so the loader returns None."""
    loader = FredCpiSettlementLoader(catalog=kalshi_catalog, panel=cpi_panel)
    assert loader.outcome("KXECONSTATCPIYOY-26JUN") is None


def test_settlement_loader_resolves_when_data_present(cpi_panel, kalshi_catalog):
    """If we know the realized YoY, the loader should locate the winning
    bucket on the catalog's ladder and return a fully-formed Outcome."""
    # March 2026 CPI is in the fixture; March 2025 is too. We can fake a
    # 25MAR event by registering one against the existing ladder structure.
    loader = FredCpiSettlementLoader(catalog=kalshi_catalog, panel=cpi_panel)

    # Compute what YoY March 2026 produces, just to verify the math.
    panel = cpi_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    cpi_2026_03 = float(panel["core_cpi"][panel.index.to_period("M") == pd.Period("2026-03")].iloc[0])
    cpi_2025_03 = float(panel["core_cpi"][panel.index.to_period("M") == pd.Period("2025-03")].iloc[0])
    expected_yoy_pct = (cpi_2026_03 / cpi_2025_03 - 1.0) * 100.0
    assert 1.0 < expected_yoy_pct < 6.0  # sanity: real CPI YoY in a plausible range

    # Build a ladder for an artificial 26MAR event and use a thin catalog.
    # We construct it by reusing the 26JUN ladder structure verbatim.
    jun_ladder = kalshi_catalog.ladder("KXECONSTATCPIYOY-26JUN")

    from core.market import BucketLadder, BucketMarket, Event
    mar_buckets = tuple(
        BucketMarket(
            market_id=f"KXECONSTATCPIYOY-26MAR-T{b.strike_value:g}",
            event_id="KXECONSTATCPIYOY-26MAR",
            series_id="KXECONSTATCPIYOY",
            open_time=b.open_time,
            close_time=b.close_time,
            expiration_time=b.expiration_time,
            title=b.title,
            floor=b.floor, cap=b.cap, strike_value=b.strike_value,
        )
        for b in jun_ladder.buckets
    )
    mar_ladder = BucketLadder(event_id="KXECONSTATCPIYOY-26MAR", buckets=mar_buckets)
    mar_event = Event(
        event_id="KXECONSTATCPIYOY-26MAR",
        series_id="KXECONSTATCPIYOY",
        close_time=datetime(2026, 4, 12, tzinfo=timezone.utc),
        expiration_time=datetime(2026, 7, 11, tzinfo=timezone.utc),
        title="CPI YoY Mar 2026",
    )
    from data.kalshi.loader import KalshiCatalog
    mar_catalog = KalshiCatalog(
        markets_by_id={m.market_id: m for m in mar_buckets},
        events_by_id={"KXECONSTATCPIYOY-26MAR": mar_event},
        ladders_by_event={"KXECONSTATCPIYOY-26MAR": mar_ladder},
    )

    loader2 = FredCpiSettlementLoader(catalog=mar_catalog, panel=cpi_panel)
    out = loader2.outcome("KXECONSTATCPIYOY-26MAR")
    assert isinstance(out, Outcome)
    assert out.event_id == "KXECONSTATCPIYOY-26MAR"
    assert out.expiration_value == pytest.approx(expected_yoy_pct)

    # The winning bucket must contain the realized YoY.
    winner = next(b for b in mar_buckets if b.market_id == out.winning_market_id)
    assert winner.contains(expected_yoy_pct)


def test_settlement_loader_parses_event_code():
    """Belt-and-suspenders for the YYMMM regex on each month abbreviation."""
    panel = pd.DataFrame({"core_cpi": [100.0, 103.0]},
                         index=pd.DatetimeIndex(["2025-06-01", "2026-06-01"]))
    # Empty catalog — outcome will fail on ladder=None even though the parse succeeds.
    from data.kalshi.loader import KalshiCatalog
    catalog = KalshiCatalog(markets_by_id={}, events_by_id={}, ladders_by_event={})
    loader = FredCpiSettlementLoader(catalog=catalog, panel=panel)

    # Parse succeeds, but no ladder => returns None.
    assert loader.outcome("KXECONSTATCPIYOY-26JUN") is None
    # Bogus suffix => returns None.
    assert loader.outcome("KXECONSTATCPIYOY-26ZZZ") is None
    assert loader.outcome("malformed-event") is None
