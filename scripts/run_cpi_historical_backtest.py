"""Run CpiNormalStrategy against the historical CPI fixture.

Reads `tests/fixtures/historical_cpi/{markets,snapshots,fred_core_cpi}.parquet`,
runs the BacktestEngine end-to-end, and prints a report with:

  - Overall Brier and log-loss
  - 10-bin calibration curve
  - Per-event P&L attribution (winning bucket vs. all losing buckets)
  - Per-strategy summary

Use this as the reproducible benchmark — the fixtures are committed, the FRED
panel inside is frozen at fetch time, and the strategy is deterministic given
those inputs.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features.fred import FredCpiSettlementLoader, FredFeatureView
from data.kalshi.loader import (
    iter_orderbook_snapshots,
    load_catalog_from_parquet,
)
from strategies.cpi_normal import CpiNormalStrategy


SERIES_ID = "KXECONSTATCPIYOY"
FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "historical_cpi"


def main() -> None:
    print(f"=== CPI historical backtest ===")
    print(f"Fixture dir: {FIXTURE_DIR}")

    catalog = load_catalog_from_parquet(FIXTURE_DIR / "markets.parquet")
    print(
        f"Catalog: {len(catalog.markets_by_id)} markets, "
        f"{len(catalog.events_by_id)} events, "
        f"{len(catalog.ladders_by_event)} ladders"
    )

    macro_panel = pd.read_parquet(FIXTURE_DIR / "fred_core_cpi.parquet")
    feature_view = FredFeatureView(panel=macro_panel)
    settlement = FredCpiSettlementLoader(catalog=catalog, panel=macro_panel)

    print()
    print("Settlement probe:")
    for ev_id in sorted(catalog.events_by_id):
        out = settlement.outcome(ev_id)
        if out is None:
            print(f"  {ev_id}: unresolved")
        else:
            print(
                f"  {ev_id}: realized YoY={out.expiration_value:.3f}%, "
                f"winning={out.winning_market_id}"
            )

    snaps_df = pd.read_parquet(FIXTURE_DIR / "snapshots.parquet")
    snapshots = list(iter_orderbook_snapshots(snaps_df))
    print(f"\nSnapshots: {len(snapshots)} from {snapshots[0].ts} to {snapshots[-1].ts}")

    engine = BacktestEngine(
        catalog=catalog,
        settlement=settlement,
        snapshot_stream=snapshots,
        feature_view=feature_view,
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(CpiNormalStrategy(series_id=SERIES_ID))

    print("\nRunning backtest...")
    report = engine.run()

    _print_report(report)
    _print_market_baseline(snapshots, settlement, catalog)


def _print_market_baseline(snapshots, settlement, catalog) -> None:
    """Compute the orderbook's own Brier as a baseline comparison.

    For each market, take the daily yes-mid (mean of yes_bid / yes_ask) on
    each snapshot, normalize per event (so the bucket probabilities sum to
    1), and score against the realized winner. Reports overall and per-event
    Brier so we can directly compare CpiNormal vs. market.
    """
    print()
    print("=" * 60)
    print("MARKET BASELINE (orderbook-implied probabilities)")
    print("=" * 60)

    # Build (event_id, ts) -> {market_id: yes_mid}
    by_event_ts: dict[tuple, dict[str, float]] = {}
    for s in snapshots:
        if s.yes_mid is None:
            continue
        market = catalog.market(s.market_id)
        key = (market.event_id, s.ts)
        by_event_ts.setdefault(key, {})[s.market_id] = s.yes_mid

    # Collect outcomes
    outcome_cache: dict[str, str] = {}
    rows: list[tuple[str, str, float, int]] = []  # (event_id, market_id, p, y)
    for (event_id, ts), market_to_p in by_event_ts.items():
        if event_id not in outcome_cache:
            o = settlement.outcome(event_id)
            if o is None:
                continue
            outcome_cache[event_id] = o.winning_market_id
        winner = outcome_cache[event_id]

        # Normalize so the per-event bucket probs sum to 1 (markets sometimes
        # quote >1 sum due to spread / arbitrage). Skip ts where we don't have
        # all buckets — partial coverage skews the normalization.
        ladder = catalog.ladder(event_id)
        if ladder is None:
            continue
        bucket_ids = [b.market_id for b in ladder.buckets]
        if not all(mid in market_to_p for mid in bucket_ids):
            continue
        total = sum(market_to_p[mid] for mid in bucket_ids)
        if total <= 0:
            continue
        for mid in bucket_ids:
            p_norm = market_to_p[mid] / total
            y = 1 if mid == winner else 0
            rows.append((event_id, mid, p_norm, y))

    if not rows:
        print("  (no market quotes scored)")
        return

    ps = [r[2] for r in rows]
    ys = [r[3] for r in rows]
    market_brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
    print(f"  market n_scored={len(rows)} brier={market_brier:.6f}")

    # Per-event terminal market Brier — the last fully-quoted ts per event
    last_ts_by_event: dict[str, object] = {}
    for (event_id, ts), _ in by_event_ts.items():
        cur = last_ts_by_event.get(event_id)
        if cur is None or ts > cur:
            last_ts_by_event[event_id] = ts

    for event_id, last_ts in sorted(last_ts_by_event.items()):
        bucket_ids = [b.market_id for b in catalog.ladder(event_id).buckets]
        market_to_p = by_event_ts.get((event_id, last_ts), {})
        if not all(mid in market_to_p for mid in bucket_ids):
            continue
        total = sum(market_to_p[mid] for mid in bucket_ids)
        if total <= 0:
            continue
        winner = outcome_cache.get(event_id)
        if winner is None:
            continue
        p_winner = market_to_p[winner] / total
        terminal_brier = sum(
            (market_to_p[mid] / total - (1 if mid == winner else 0)) ** 2
            for mid in bucket_ids
        ) / len(bucket_ids)
        print(
            f"  {event_id}: terminal market p_winner={p_winner:.4f} "
            f"terminal_brier={terminal_brier:.6f} ts={last_ts}"
        )


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
    cal = m.get("calibration") or []
    for i, (mp, my, n) in enumerate(cal):
        lo, hi = i * 0.1, (i + 1) * 0.1
        if n == 0:
            print(f"  [{lo:.1f}, {hi:.1f})        --       --        0")
        else:
            print(f"  [{lo:.1f}, {hi:.1f})  {mp:>8.4f} {my:>8.4f} {n:>8d}")

    print()
    print("Per-event P&L attribution (terminal forecast vs. realized outcome):")
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
    """For each event, show the strategy's last forecast vs. the realized
    winner. P&L isn't dollars (no trades emitted) — instead we report the
    Brier contribution and a 'edge_score' = sum over markets of
    (realized - forecast)^2 weighted by prior probability.
    """
    if not report.scored_forecasts:
        print("  (no scored forecasts)")
        return

    by_event: dict[str, dict] = defaultdict(lambda: {"records": []})
    for f, y in report.scored_forecasts:
        by_event[f.event_id]["records"].append((f, y))

    for ev_id in sorted(by_event):
        recs = by_event[ev_id]["records"]
        # Take the most recent forecast per market
        last_per_market: dict[str, tuple] = {}
        for f, y in recs:
            cur = last_per_market.get(f.market_id)
            if cur is None or f.ts > cur[0].ts:
                last_per_market[f.market_id] = (f, y)

        ps = [r[0].p_yes for r in last_per_market.values()]
        ys = [r[1] for r in last_per_market.values()]
        if not ps:
            continue
        brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
        prob_on_winner = next(
            (r[0].p_yes for r in last_per_market.values() if r[1] == 1), None
        )
        n_recs_total = len(recs)
        winner = next(
            (r[0].market_id for r in last_per_market.values() if r[1] == 1),
            "?",
        )
        prob_str = f"{prob_on_winner:.4f}" if prob_on_winner is not None else "  ----"
        print(
            f"  {ev_id}:"
            f" winner={winner}"
            f" p_winner={prob_str}"
            f" terminal_brier={brier:.6f}"
            f" n_markets={len(last_per_market)}"
            f" n_forecasts={n_recs_total}"
        )


if __name__ == "__main__":
    main()
