"""Reproducible benchmark: CpiNormalStrategy on the historical fixture.

Asserts that the BacktestEngine produces a non-trivial scored-forecast set
and that the strategy's Brier score stays in a sane range. Tightening the
range is left to future work — for now the assertion is "non-zero scored
forecasts and Brier within [0, 0.25]" so a regression that breaks scoring
will fail the test.

The fixture under tests/fixtures/historical_cpi/ is committed; this test
takes ~3 seconds and does no network I/O.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features.fred import FredCpiSettlementLoader, FredFeatureView
from data.kalshi.loader import (
    iter_orderbook_snapshots,
    load_catalog_from_parquet,
)
from strategies.cpi_normal import CpiNormalStrategy


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "historical_cpi"
SERIES_ID = "KXECONSTATCPIYOY"


def _build_engine() -> BacktestEngine:
    catalog = load_catalog_from_parquet(FIXTURE_DIR / "markets.parquet")
    panel = pd.read_parquet(FIXTURE_DIR / "fred_core_cpi.parquet")
    snaps_df = pd.read_parquet(FIXTURE_DIR / "snapshots.parquet")
    snapshots = list(iter_orderbook_snapshots(snaps_df))

    engine = BacktestEngine(
        catalog=catalog,
        settlement=FredCpiSettlementLoader(catalog=catalog, panel=panel),
        snapshot_stream=snapshots,
        feature_view=FredFeatureView(panel=panel),
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(CpiNormalStrategy(series_id=SERIES_ID))
    return engine


def test_historical_backtest_produces_scored_forecasts():
    report = _build_engine().run()
    m = report.metrics

    assert m["n_forecasts"] > 0, "strategy emitted no forecasts"
    assert m["n_scored_forecasts"] > 0, "no forecasts could be scored against outcomes"
    # n_scored should equal n_forecasts here because every CPI event in the
    # fixture is resolvable from the FRED panel.
    assert m["n_scored_forecasts"] == m["n_forecasts"]

    assert 0.0 < m["brier"] < 0.25, f"brier={m['brier']!r} outside sane range"
    assert m["log_loss"] > 0, f"log_loss={m['log_loss']!r}"

    # Every event in the catalog must resolve.
    for ev_id in {f.event_id for f in report.forecasts}:
        loader = FredCpiSettlementLoader(
            catalog=load_catalog_from_parquet(FIXTURE_DIR / "markets.parquet"),
            panel=pd.read_parquet(FIXTURE_DIR / "fred_core_cpi.parquet"),
        )
        assert loader.outcome(ev_id) is not None, f"{ev_id} unresolved"


def test_historical_calibration_has_data():
    report = _build_engine().run()
    cal = report.metrics["calibration"]
    assert len(cal) == 10, "expected 10 calibration bins"
    populated = [n for _mp, _my, n in cal if n > 0]
    assert sum(populated) > 0, "no bin has any forecasts"


def test_historical_per_strategy_metrics_present():
    report = _build_engine().run()
    per = report.metrics["per_strategy"]
    assert "cpi_normal" in per
    s = per["cpi_normal"]
    assert s["n_scored_forecasts"] > 0
    assert s["brier"] is not None
    assert s["log_loss"] is not None
    # No orders are emitted in v1; cash flow and payouts must be exactly zero.
    assert s["trading_cash_flow"] == 0
    assert s["settlement_payout"] == 0
