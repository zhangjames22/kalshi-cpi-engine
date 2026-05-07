"""Reproducible benchmark: NflHomeFavoriteStrategy on the historical fixture.

Same shape as `test_historical_cpi_benchmark.py`. Fails if the platform's
binary-market path regresses — a regression that breaks scoring or
binary-event settlement here is the canary. The fixture under
tests/fixtures/historical_nfl/ is committed; this test takes ~3 seconds
and does no network I/O.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from core.market import BinaryResolution
from data.features import NullFeatureView
from data.sports.nfl import (
    SERIES_ID,
    NflSettlementLoader,
    iter_game_snapshots,
    load_catalog_from_parquet,
)
from strategies.nfl_home_favorite import NflHomeFavoriteStrategy


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "historical_nfl"


def _build_engine() -> BacktestEngine:
    games_df = pd.read_parquet(FIXTURE_DIR / "games.parquet")
    snaps_df = pd.read_parquet(FIXTURE_DIR / "snapshots.parquet")
    catalog = load_catalog_from_parquet(FIXTURE_DIR / "games.parquet")

    engine = BacktestEngine(
        catalog=catalog,
        settlement=NflSettlementLoader(games_df=games_df),
        snapshot_stream=list(iter_game_snapshots(snaps_df)),
        feature_view=NullFeatureView(),
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(NflHomeFavoriteStrategy(series_id=SERIES_ID))
    return engine


def test_nfl_backtest_produces_scored_forecasts():
    report = _build_engine().run()
    m = report.metrics

    assert m["n_forecasts"] > 0
    assert m["n_scored_forecasts"] > 0
    assert m["n_scored_forecasts"] == m["n_forecasts"], (
        "every NFL forecast should be resolvable from the games fixture"
    )
    # A constant-0.55 forecaster on a binary event with empirical home-win
    # rate ~0.54 should land near 0.25 Brier; bracket loosely so the test
    # tolerates fixture refreshes.
    assert 0.20 <= m["brier"] <= 0.30, f"unexpected brier={m['brier']!r}"
    assert m["log_loss"] > 0


def test_nfl_calibration_concentrated_in_constant_bin():
    """The home-favorite strategy emits a single constant probability, so
    exactly one calibration bin should contain all the mass. This guards
    against accidental DistributionForecast / ladder routing on a binary
    event."""
    report = _build_engine().run()
    cal = report.metrics["calibration"]
    populated = [(i, n) for i, (_mp, _my, n) in enumerate(cal) if n > 0]
    assert len(populated) == 1, f"expected 1 populated bin, got {populated}"
    bin_idx, n = populated[0]
    assert bin_idx == 5, f"home_advantage=0.55 should land in bin [0.5, 0.6); got {bin_idx}"
    assert n == report.metrics["n_scored_forecasts"]


def test_nfl_per_strategy_metrics_present():
    report = _build_engine().run()
    per = report.metrics["per_strategy"]
    assert "nfl_home_favorite" in per
    s = per["nfl_home_favorite"]
    assert s["n_scored_forecasts"] > 0
    assert s["brier"] is not None
    assert s["trading_cash_flow"] == 0
    assert s["settlement_payout"] == 0


def test_nfl_settlements_use_binary_path():
    """A BinaryMarket event has no BucketLadder. The engine must settle
    purely binary events even though no positions were opened. We verify
    by checking that the catalog's `ladder()` returns None for every
    event the strategy forecasted on — that path is what the engine's
    `event_market_ids = [event_id]` fallback exercises."""
    catalog = load_catalog_from_parquet(FIXTURE_DIR / "games.parquet")
    for ev_id in catalog.events_by_id:
        assert catalog.ladder(ev_id) is None, (
            f"NFL event {ev_id} unexpectedly carries a BucketLadder"
        )


def test_nfl_outcomes_use_binary_resolution_enum():
    """Loader must populate BinaryResolution and use None winner for NO
    resolutions — no string sentinels like '<event>#NO'."""
    games_df = pd.read_parquet(FIXTURE_DIR / "games.parquet")
    loader = NflSettlementLoader(games_df=games_df)

    saw_yes = saw_no = False
    for ticker in games_df["ticker"]:
        out = loader.outcome(ticker)
        assert out is not None
        assert out.binary_resolution in (BinaryResolution.YES, BinaryResolution.NO)
        if out.binary_resolution is BinaryResolution.YES:
            assert out.winning_market_id == ticker
            saw_yes = True
        else:
            assert out.winning_market_id is None
            saw_no = True
        # Old sentinel must never appear.
        assert out.winning_market_id != f"{ticker}#NO"

    assert saw_yes and saw_no, "fixture should contain both YES and NO outcomes"
