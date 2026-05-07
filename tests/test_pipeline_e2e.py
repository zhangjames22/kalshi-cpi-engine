"""End-to-end pipeline validation.

Spins up a synthetic CPI scenario, runs the CpiNormalStrategy through the
full BacktestEngine + CrossSpreadFill + report pipeline, and asserts the
plumbing produced the expected shape of output.

This is the validation goal for the v1 backtest framework: the existing
notebook 03 logic can run, end-to-end, against the new abstractions.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from backtest import (
    BacktestConfig,
    BacktestEngine,
    CrossSpreadFill,
    MidFill,
)
from data.stub import build_cpi_scenario
from strategies.cpi_normal import CpiNormalStrategy


def _run(execution=None):
    scenario = build_cpi_scenario(realized_value=3.27)
    engine = BacktestEngine(
        catalog=scenario.catalog,
        settlement=scenario.settlement,
        snapshot_stream=scenario.snapshots,
        feature_view=scenario.feature_view,
        execution=execution or CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(
        CpiNormalStrategy(series_id=scenario.series_id)
    )
    return scenario, engine.run()


# ---------------------------------------------------------------------------
# Pipeline smoke
# ---------------------------------------------------------------------------

def test_pipeline_runs_end_to_end_and_produces_a_report():
    scenario, report = _run()
    # The strategy publishes one forecast per ladder bucket per tick.
    assert report.metrics["n_forecasts"] > 0
    # Every forecast was on the single resolved event, so all should be scored.
    assert report.metrics["n_scored_forecasts"] == report.metrics["n_forecasts"]
    # No orders → no fills, no settlement payouts (no positions to settle).
    assert report.metrics["n_fills"] == 0
    assert report.metrics["n_settlements"] == 0


def test_pipeline_brier_score_is_in_valid_range():
    _, report = _run()
    brier = report.metrics["brier"]
    # Brier on binary outcomes is in [0, 1]. A naive 1/n_buckets forecaster
    # would land near (n-1)/n^2; we just check the range.
    assert 0.0 <= brier <= 1.0


def test_pipeline_calibration_curve_returned():
    _, report = _run()
    cal = report.metrics["calibration"]
    assert isinstance(cal, list)
    assert len(cal) == 10
    # Bins are (mean_predicted, mean_realized, n) tuples.
    total_n = sum(n for _, _, n in cal)
    assert total_n == report.metrics["n_scored_forecasts"]


def test_pipeline_per_strategy_metrics_attribute_correctly():
    _, report = _run()
    per_strat = report.metrics["per_strategy"]
    assert set(per_strat.keys()) == {"cpi_normal"}
    cpi = per_strat["cpi_normal"]
    assert cpi["n_scored_forecasts"] == report.metrics["n_scored_forecasts"]
    assert cpi["net_pnl"] == 0.0  # no orders


def test_pipeline_zero_pnl_when_no_orders_emitted():
    _, report = _run()
    assert report.total_return() == pytest.approx(0.0)
    assert report.final_cash == pytest.approx(report.initial_cash)


def test_pipeline_winning_bucket_has_highest_realized_rate():
    """At resolution time, the bucket containing realized_value should be the
    one and only YES winner among the ladder's bucket markets."""
    scenario, report = _run()
    winning = scenario.settlement.outcome(scenario.event_id).winning_market_id
    yes_outcomes = [
        (f.market_id, y) for f, y in report.scored_forecasts
        if f.event_id == scenario.event_id
    ]
    yes_winners = {mid for mid, y in yes_outcomes if y == 1}
    assert yes_winners == {winning}


# ---------------------------------------------------------------------------
# Parquet serialization
# ---------------------------------------------------------------------------

def test_pipeline_writes_all_ledgers_to_parquet():
    _, report = _run()
    with tempfile.TemporaryDirectory() as tmpdir:
        report.write_to(tmpdir)
        for name in ["forecasts.parquet", "fills.parquet", "settlements.parquet", "equity.parquet"]:
            path = os.path.join(tmpdir, name)
            assert os.path.exists(path), f"missing {name}"
        with open(os.path.join(tmpdir, "summary.json")) as f:
            summary = json.load(f)
        assert "brier" in summary
        assert "per_strategy" in summary


def test_pipeline_forecasts_parquet_roundtrips_via_pandas():
    _, report = _run()
    pd = pytest.importorskip("pandas")
    with tempfile.TemporaryDirectory() as tmpdir:
        report.write_to(tmpdir)
        df = pd.read_parquet(os.path.join(tmpdir, "forecasts.parquet"))
        assert len(df) == report.metrics["n_forecasts"]
        assert set(df.columns) >= {"ts", "strategy", "market_id", "event_id", "p_yes"}
        assert df["p_yes"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Mid vs cross-spread parity (no-trade strategy)
# ---------------------------------------------------------------------------

def test_pipeline_execution_model_choice_is_independent_when_no_orders():
    """A no-trade strategy should produce identical forecast metrics under
    MidFill and CrossSpreadFill — proves the execution model isn't leaking
    into the forecast pipeline."""
    _, mid_report = _run(MidFill())
    _, cross_report = _run(CrossSpreadFill())
    assert mid_report.metrics["brier"] == cross_report.metrics["brier"]
    assert mid_report.metrics["n_forecasts"] == cross_report.metrics["n_forecasts"]
