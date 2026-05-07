"""End-to-end pipeline test against real Kalshi + FRED data fixtures.

Uses the small fixtures captured from real API pulls (not the full panels)
so the test runs in <1s and is reproducible without network access. The
goal is to lock in that the *real* data flows through every layer of the
framework — loader, feature view, settlement loader, engine, report —
without touching the synthetic stub.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features.fred import FredCpiSettlementLoader, FredFeatureView
from data.kalshi.loader import (
    iter_orderbook_snapshots,
    load_catalog_from_parquet,
)
from strategies.cpi_normal import CpiNormalStrategy


FIXTURES = Path(__file__).parent / "fixtures"
KALSHI_FIXTURE = FIXTURES / "kalshi_markets_26JUN.parquet"
FRED_FIXTURE = FIXTURES / "fred_core_cpi_recent.parquet"


@pytest.fixture
def real_pipeline():
    """Build the full pipeline from real-data fixtures and run it once."""
    catalog = load_catalog_from_parquet(KALSHI_FIXTURE)
    macro_panel = pd.read_parquet(FRED_FIXTURE)
    feature_view = FredFeatureView(panel=macro_panel)
    settlement = FredCpiSettlementLoader(catalog=catalog, panel=macro_panel)

    snapshots = list(iter_orderbook_snapshots(pd.read_parquet(KALSHI_FIXTURE)))

    engine = BacktestEngine(
        catalog=catalog,
        settlement=settlement,
        snapshot_stream=snapshots,
        feature_view=feature_view,
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(CpiNormalStrategy(series_id="KXECONSTATCPIYOY"))
    report = engine.run()
    return catalog, snapshots, report


# ---------------------------------------------------------------------------
# Catalog shape from real data
# ---------------------------------------------------------------------------

def test_real_catalog_has_one_event_with_real_buckets(real_pipeline):
    catalog, _snapshots, _report = real_pipeline
    assert set(catalog.events_by_id) == {"KXECONSTATCPIYOY-26JUN"}
    ladder = catalog.ladders_by_event["KXECONSTATCPIYOY-26JUN"]
    # Real fixture has 21 markets in this event.
    assert len(ladder.buckets) == 21
    # Ladder covers all of R.
    assert math.isinf(ladder.buckets[0].floor) and ladder.buckets[0].floor < 0
    assert math.isinf(ladder.buckets[-1].cap) and ladder.buckets[-1].cap > 0


# ---------------------------------------------------------------------------
# Snapshot stream from real top-of-book columns
# ---------------------------------------------------------------------------

def test_real_snapshot_stream_yields_one_per_market(real_pipeline):
    _catalog, snapshots, _report = real_pipeline
    assert len(snapshots) == 21  # 21 markets in the fixture, one snapshot each


def test_real_snapshot_prices_are_in_unit_interval(real_pipeline):
    _catalog, snapshots, _report = real_pipeline
    for s in snapshots:
        for p in [s.yes_bid_price, s.yes_ask_price, s.no_bid_price, s.no_ask_price]:
            if p is not None:
                assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Forecast generation against real data
# ---------------------------------------------------------------------------

def test_real_pipeline_produces_forecasts(real_pipeline):
    """Each tick: strategy fits a Normal from real CPI history, projects
    onto the real bucket ladder. With 21 markets / 1 event / 21 ticks
    we expect 21*21 = 441 forecast records."""
    _catalog, _snapshots, report = real_pipeline
    assert report.metrics["n_forecasts"] == 441


def test_real_forecasts_sum_to_one_per_emission(real_pipeline):
    """A proper CDF projected onto a contiguous ladder must integrate to 1.

    The fixture has all 21 snapshots at the same wall-clock timestamp, so
    grouping by (ts, event_id) collapses 21 separate emissions; instead we
    chunk the per-strategy/per-event ledger into ladder-sized groups, each
    representing one DistributionForecast emission, and verify each sums to 1.
    """
    catalog, _snapshots, report = real_pipeline
    ladder_size = len(catalog.ladders_by_event["KXECONSTATCPIYOY-26JUN"].buckets)
    n_records = len(report.forecasts)
    assert n_records % ladder_size == 0, (
        f"forecast records {n_records} not divisible by ladder size {ladder_size}"
    )

    # Records are appended in tick order; within a tick, in ladder-bucket order.
    for i in range(0, n_records, ladder_size):
        chunk = report.forecasts[i : i + ladder_size]
        total = sum(f.p_yes for f in chunk)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_real_forecasts_are_valid_probabilities(real_pipeline):
    _catalog, _snapshots, report = real_pipeline
    for f in report.forecasts:
        assert 0.0 <= f.p_yes <= 1.0


# ---------------------------------------------------------------------------
# Settlement against real data
# ---------------------------------------------------------------------------

def test_real_pipeline_unresolved_event_yields_no_scoring(real_pipeline):
    """The 26JUN event hasn't been observed yet (FRED fixture ends in
    March 2026), so settlement returns None and nothing is scored."""
    _catalog, _snapshots, report = real_pipeline
    assert report.metrics["n_scored_forecasts"] == 0
    assert report.metrics["brier"] is None
    assert report.metrics["log_loss"] is None


def test_real_pipeline_zero_pnl_with_no_orders(real_pipeline):
    _catalog, _snapshots, report = real_pipeline
    assert report.metrics["n_fills"] == 0
    assert report.total_return() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Round-trip through parquet
# ---------------------------------------------------------------------------

def test_real_pipeline_writes_real_data_ledgers(real_pipeline, tmp_path):
    _catalog, _snapshots, report = real_pipeline
    report.write_to(str(tmp_path))

    forecasts = pd.read_parquet(tmp_path / "forecasts.parquet")
    assert len(forecasts) == 441
    assert set(forecasts.columns) >= {"ts", "strategy", "market_id", "event_id", "p_yes"}
    # Real Kalshi tickers should be preserved verbatim.
    assert all(mid.startswith("KXECONSTATCPIYOY-26JUN") for mid in forecasts["market_id"])
