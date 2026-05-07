"""Run NflHomeFavoriteStrategy against the historical NFL fixture.

Mirrors `scripts/run_cpi_historical_backtest.py`: same engine, same report
shape (Brier, log-loss, calibration, per-game attribution, market baseline).
Only the data adapter and the strategy change — that's the architectural
test: a binary-market sport plugs in without touching `core/`, `backtest/`,
`data/features/`, or `data/kalshi/loader.py`.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features import NullFeatureView
from data.sports.nfl import (
    SERIES_ID,
    NflSettlementLoader,
    iter_game_snapshots,
    load_catalog_from_parquet,
)
from strategies.nfl_home_favorite import NflHomeFavoriteStrategy


FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "historical_nfl"


def main() -> None:
    print("=== NFL historical backtest ===")
    print(f"Fixture dir: {FIXTURE_DIR}")

    games_df = pd.read_parquet(FIXTURE_DIR / "games.parquet")
    snaps_df = pd.read_parquet(FIXTURE_DIR / "snapshots.parquet")
    catalog = load_catalog_from_parquet(FIXTURE_DIR / "games.parquet")

    print(
        f"Catalog: {len(catalog.markets_by_id)} binary markets, "
        f"{len(catalog.events_by_id)} events"
    )

    settlement = NflSettlementLoader(games_df=games_df)
    snapshots = list(iter_game_snapshots(snaps_df))

    print(
        f"Snapshots: {len(snapshots)} from {snapshots[0].ts} to {snapshots[-1].ts}"
    )

    n_resolved = sum(
        1 for ev_id in catalog.events_by_id
        if settlement.outcome(ev_id) is not None
    )
    print(f"Settlement probe: {n_resolved}/{len(catalog.events_by_id)} events resolvable")

    engine = BacktestEngine(
        catalog=catalog,
        settlement=settlement,
        snapshot_stream=snapshots,
        feature_view=NullFeatureView(),
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(NflHomeFavoriteStrategy(series_id=SERIES_ID))

    print("\nRunning backtest...")
    report = engine.run()

    _print_report(report)
    _print_market_baseline(games_df, snapshots, catalog, settlement)


def _print_report(report) -> None:
    m = report.metrics
    print()
    print("=" * 60)
    print("REPORT")
    print("=" * 60)
    print(repr(report))
    print()
    for k in ("n_forecasts", "n_scored_forecasts", "n_fills", "n_settlements",
              "brier", "log_loss"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print()
    print("Calibration (10 bins):")
    print(f"  {'bin':<14} {'mean_p':>8} {'mean_y':>8} {'n':>8}")
    for i, (mp, my, n) in enumerate(m.get("calibration") or []):
        lo, hi = i * 0.1, (i + 1) * 0.1
        if n == 0:
            print(f"  [{lo:.1f}, {hi:.1f})        --       --        0")
        else:
            print(f"  [{lo:.1f}, {hi:.1f})  {mp:>8.4f} {my:>8.4f} {n:>8d}")

    print()
    print("Per-game P&L attribution (terminal forecast vs realized):")
    _print_event_attribution(report)

    print()
    print("Per-strategy:")
    for name, ms in m.get("per_strategy", {}).items():
        print(f"  {name}:")
        for k, v in ms.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")


def _print_event_attribution(report) -> None:
    """Aggregate the strategy's last forecast per game into a brief
    summary: count of games where strategy was 'right' (p > 0.5 and home
    won, or p < 0.5 and away won), and Brier breakdown by home/away win.
    """
    if not report.scored_forecasts:
        print("  (no scored forecasts)")
        return

    last_per_event: dict[str, tuple] = {}
    for f, y in report.scored_forecasts:
        cur = last_per_event.get(f.event_id)
        if cur is None or f.ts > cur[0].ts:
            last_per_event[f.event_id] = (f, y)

    n_total = len(last_per_event)
    n_correct = sum(
        1 for f, y in last_per_event.values()
        if (f.p_yes > 0.5 and y == 1) or (f.p_yes < 0.5 and y == 0)
    )
    home_wins = [(f, y) for f, y in last_per_event.values() if y == 1]
    away_wins = [(f, y) for f, y in last_per_event.values() if y == 0]
    brier_home = sum((f.p_yes - 1) ** 2 for f, _ in home_wins) / max(1, len(home_wins))
    brier_away = sum((f.p_yes - 0) ** 2 for f, _ in away_wins) / max(1, len(away_wins))
    print(f"  n_games={n_total} | accuracy(threshold=0.5)={n_correct/n_total:.4f}")
    print(f"  home_wins={len(home_wins)} brier_on_home_wins={brier_home:.6f}")
    print(f"  away_wins={len(away_wins)} brier_on_away_wins={brier_away:.6f}")


def _print_market_baseline(games_df, snapshots, catalog, settlement) -> None:
    """Compute Brier of the orderbook's own implied probability for each
    game (yes_mid at the only snapshot we captured). Comparable apples-to-
    apples with the strategy's terminal Brier."""
    print()
    print("=" * 60)
    print("MARKET BASELINE (closing-line implied probabilities)")
    print("=" * 60)

    # Map each ticker to the (only) snapshot we recorded.
    snap_by_ticker = {s.market_id: s for s in snapshots}

    rows: list[tuple[str, float, int]] = []
    for _, g in games_df.iterrows():
        ticker = g["ticker"]
        snap = snap_by_ticker.get(ticker)
        if snap is None or snap.yes_mid is None:
            continue
        outcome = settlement.outcome(ticker)
        if outcome is None:
            continue
        y = 1 if outcome.winning_market_id == ticker else 0
        rows.append((ticker, snap.yes_mid, y))

    if not rows:
        print("  (no quoted snapshots)")
        return

    ps = [p for _, p, _ in rows]
    ys = [y for _, _, y in rows]
    market_brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
    n_correct = sum(
        1 for _, p, y in rows
        if (p > 0.5 and y == 1) or (p < 0.5 and y == 0)
    )
    print(f"  n={len(rows)} brier={market_brier:.6f} accuracy={n_correct/len(rows):.4f}")


if __name__ == "__main__":
    main()
