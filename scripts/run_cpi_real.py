"""End-to-end pipeline run on real Kalshi + FRED data.

Usage:
    python scripts/run_cpi_real.py

Reads cached parquet from `data/`:
    - data/kalshi_cpi_markets_open.parquet  (notebook 02 output)
    - data/macro_raw.parquet                (notebook 00 output)

Builds the catalog, snapshot stream, FRED feature view, and CPI settlement
loader; runs CpiNormalStrategy through the BacktestEngine; writes a
BacktestReport to `data/output/`.

This is the validation that the framework runs against real market data,
not the synthetic stub. No new strategies — just CpiNormalStrategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features.fred import FredCpiSettlementLoader, FredFeatureView
from data.kalshi.loader import (
    iter_orderbook_snapshots,
    load_catalog_from_parquet,
)
from strategies.cpi_normal import CpiNormalStrategy


SERIES_ID = "KXECONSTATCPIYOY"
DATA_DIR = _REPO_ROOT / "data"
MARKETS_PATH = DATA_DIR / "kalshi_cpi_markets_open.parquet"
MACRO_PATH = DATA_DIR / "macro_raw.parquet"
OUTPUT_DIR = DATA_DIR / "output_real"


def main() -> None:
    print(f"Loading Kalshi catalog from {MARKETS_PATH}...")
    catalog = load_catalog_from_parquet(MARKETS_PATH)
    print(
        f"  catalog: {len(catalog.markets_by_id)} markets, "
        f"{len(catalog.events_by_id)} events, "
        f"{len(catalog.ladders_by_event)} ladders"
    )
    for ev_id, ladder in catalog.ladders_by_event.items():
        print(f"    {ev_id}: {len(ladder.buckets)} buckets")

    print(f"\nLoading FRED macro panel from {MACRO_PATH}...")
    macro_panel = pd.read_parquet(MACRO_PATH)
    print(f"  macro panel: {len(macro_panel)} rows, "
          f"{macro_panel.index.min()} -> {macro_panel.index.max()}")

    feature_view = FredFeatureView(panel=macro_panel)
    settlement = FredCpiSettlementLoader(catalog=catalog, panel=macro_panel)

    # Probe each event for resolution status — gives us a feel for how many
    # of the open events the FRED panel can settle right now.
    print("\nSettlement probe:")
    for ev_id in sorted(catalog.events_by_id):
        out = settlement.outcome(ev_id)
        if out is None:
            print(f"  {ev_id}: unresolved (CPI not yet observed)")
        else:
            print(
                f"  {ev_id}: realized YoY={out.expiration_value:.3f}%, "
                f"winning={out.winning_market_id}"
            )

    print(f"\nBuilding orderbook stream from {MARKETS_PATH}...")
    markets_df = pd.read_parquet(MARKETS_PATH)
    snapshots = list(iter_orderbook_snapshots(markets_df))
    print(f"  {len(snapshots)} snapshots, "
          f"{snapshots[0].ts} -> {snapshots[-1].ts}")

    print("\nRunning BacktestEngine...")
    engine = BacktestEngine(
        catalog=catalog,
        settlement=settlement,
        snapshot_stream=snapshots,
        feature_view=feature_view,
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(CpiNormalStrategy(series_id=SERIES_ID))
    report = engine.run()

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(repr(report))
    print()
    for k, v in report.metrics.items():
        if k in ("calibration", "per_strategy"):
            continue
        print(f"  {k}: {v}")

    if report.metrics.get("calibration"):
        print("\nCalibration:")
        for i, (mp, mr, n) in enumerate(report.metrics["calibration"]):
            if n == 0:
                continue
            lo, hi = i * 0.1, (i + 1) * 0.1
            print(f"  [{lo:.1f}, {hi:.1f}): predicted={mp:.3f} "
                  f"realized={mr:.3f} n={n}")

    print("\nPer-strategy:")
    for name, m in report.metrics["per_strategy"].items():
        print(f"  {name}: {m}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_to(str(OUTPUT_DIR))
    print(f"\nLedgers written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
