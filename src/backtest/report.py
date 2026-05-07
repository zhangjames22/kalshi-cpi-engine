"""BacktestReport — what comes back from `engine.run()`.

Three layers of information:

1. Forecast quality   — Brier, log-loss, calibration curve. Works even
                        when the strategy emits zero orders.
2. P&L                — total return, per-strategy return, equity curve,
                        per-fill cash flows, settlement payouts.
3. Tidy ledgers       — forecasts.parquet, fills.parquet, settlements.parquet,
                        equity.parquet — for downstream analysis in pandas
                        without re-running the backtest.

The BacktestReport object is a thin container over typed records plus
derived metrics; serialization to parquet is via `report.write_to(path)`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.market import MarketCatalog, MarketId, Outcome, SettlementLoader


# ---------------------------------------------------------------------------
# Records — one row per parquet ledger. Plain dataclasses; pandas will
# coerce them on write.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastRecord:
    ts: datetime
    strategy: str
    market_id: str
    event_id: str
    p_yes: float


@dataclass(frozen=True)
class FillRecord:
    ts: datetime
    strategy: str
    market_id: str
    side: str
    action: str
    qty: int
    price: float
    cash_delta: float


@dataclass(frozen=True)
class SettlementRecord:
    ts: datetime
    strategy: str
    event_id: str
    market_id: str
    # None when a binary event resolved NO — there is no winning YES market
    # in that case, by construction.
    winning_market_id: str | None
    yes_qty: int
    no_qty: int
    payout: float


@dataclass(frozen=True)
class EquitySample:
    t: datetime
    cash: float
    mark: float

    @property
    def total(self) -> float:
        return self.cash + self.mark


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def brier_score(p_yes: list[float], outcome_yes: list[int]) -> float:
    """Mean squared error between forecast probabilities and binary outcomes.

    Range [0, 1]; lower is better. 0.25 is the score of a constant 0.5 forecast.
    """
    if not p_yes:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(p_yes, outcome_yes)) / len(p_yes)


def log_loss(
    p_yes: list[float], outcome_yes: list[int], eps: float = 1e-12
) -> float:
    """Cross-entropy loss. Lower is better; clipped to avoid log(0)."""
    if not p_yes:
        return float("nan")
    s = 0.0
    for p, y in zip(p_yes, outcome_yes):
        p_clip = min(max(p, eps), 1.0 - eps)
        s += -(y * math.log(p_clip) + (1 - y) * math.log(1 - p_clip))
    return s / len(p_yes)


def calibration_curve(
    p_yes: list[float], outcome_yes: list[int], n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Reliability diagram data: (mean_predicted, mean_realized, n_in_bin)
    for each of `n_bins` equal-width probability bins. A perfectly calibrated
    forecaster has mean_predicted == mean_realized for every bin."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(p_yes, outcome_yes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    out: list[tuple[float, float, int]] = []
    for b in bins:
        if not b:
            out.append((float("nan"), float("nan"), 0))
            continue
        mean_p = sum(p for p, _ in b) / len(b)
        mean_y = sum(y for _, y in b) / len(b)
        out.append((mean_p, mean_y, len(b)))
    return out


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    """Container for everything the engine produced. Build via `.build`,
    serialize via `.write_to(path)`."""

    strategies: list[str]
    initial_cash: float
    final_cash: float
    forecasts: list[ForecastRecord]
    fills: list[FillRecord]
    settlements: list[SettlementRecord]
    equity: list[EquitySample]

    # Forecast scoring — only includes resolved forecasts.
    scored_forecasts: list[tuple[ForecastRecord, int]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        strategies: list[str],
        initial_cash: float,
        forecasts: list[ForecastRecord],
        fills: list[FillRecord],
        settlements: list[SettlementRecord],
        equity: list[EquitySample],
        settlement_loader: "SettlementLoader",
        catalog: "MarketCatalog",
    ) -> "BacktestReport":
        # Pair each forecast with its eventual binary outcome (1 if this
        # market won, 0 otherwise). Forecasts on unresolved events are dropped
        # from scoring but kept in the ledger. Cache by event_id; binary-NO
        # outcomes have winning_market_id=None, which is a legitimate cached
        # value (no market_id will ever match it -> all forecasts on that
        # event score 0, which is exactly the YES-side semantic).
        scored: list[tuple[ForecastRecord, int]] = []
        outcome_cache: dict[str, "Outcome | None"] = {}
        for f in forecasts:
            if f.event_id not in outcome_cache:
                outcome_cache[f.event_id] = settlement_loader.outcome(f.event_id)
            outcome = outcome_cache[f.event_id]
            if outcome is None:
                continue
            scored.append((f, 1 if f.market_id == outcome.winning_market_id else 0))

        ps = [f.p_yes for f, _ in scored]
        ys = [y for _, y in scored]

        metrics: dict = {
            "n_forecasts": len(forecasts),
            "n_scored_forecasts": len(scored),
            "n_fills": len(fills),
            "n_settlements": len(settlements),
            "brier": brier_score(ps, ys) if ps else None,
            "log_loss": log_loss(ps, ys) if ps else None,
            "calibration": calibration_curve(ps, ys) if ps else [],
            "per_strategy": _per_strategy_metrics(strategies, scored, fills, settlements, initial_cash, equity),
        }

        final_cash = equity[-1].cash if equity else initial_cash

        return cls(
            strategies=strategies,
            initial_cash=initial_cash,
            final_cash=final_cash,
            forecasts=forecasts,
            fills=fills,
            settlements=settlements,
            equity=equity,
            scored_forecasts=scored,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def write_to(self, path: str) -> None:
        """Write all four ledgers + a summary.json into `path` (created if
        it doesn't exist).

        Uses pandas + pyarrow (already in requirements.txt). Empty ledgers
        still produce empty parquet files so downstream readers have a
        consistent schema.
        """
        os.makedirs(path, exist_ok=True)
        import pandas as pd

        _to_parquet(
            pd.DataFrame([asdict(r) for r in self.forecasts]),
            os.path.join(path, "forecasts.parquet"),
            schema={"ts": "datetime64[ns, UTC]", "p_yes": "float64"},
        )
        _to_parquet(
            pd.DataFrame([asdict(r) for r in self.fills]),
            os.path.join(path, "fills.parquet"),
        )
        _to_parquet(
            pd.DataFrame([asdict(r) for r in self.settlements]),
            os.path.join(path, "settlements.parquet"),
        )
        _to_parquet(
            pd.DataFrame(
                [
                    {"t": e.t, "cash": e.cash, "mark": e.mark, "total": e.total}
                    for e in self.equity
                ]
            ),
            os.path.join(path, "equity.parquet"),
        )

        with open(os.path.join(path, "summary.json"), "w") as f:
            json.dump(_jsonable(self.metrics), f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        return self.final_cash - self.initial_cash

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestReport(strategies={self.strategies}, "
            f"n_forecasts={m.get('n_forecasts', 0)}, "
            f"n_scored={m.get('n_scored_forecasts', 0)}, "
            f"n_fills={m.get('n_fills', 0)}, "
            f"brier={m.get('brier')}, "
            f"return={self.total_return():+.2f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_strategy_metrics(
    strategies: list[str],
    scored: list[tuple[ForecastRecord, int]],
    fills: list[FillRecord],
    settlements: list[SettlementRecord],
    initial_cash: float,
    equity: list[EquitySample],
) -> dict:
    out: dict = {}
    for name in strategies:
        strat_scored = [(f, y) for f, y in scored if f.strategy == name]
        ps = [f.p_yes for f, _ in strat_scored]
        ys = [y for _, y in strat_scored]
        cash_in = sum(
            f.cash_delta for f in fills if f.strategy == name
        )
        settlement_payouts = sum(
            s.payout for s in settlements if s.strategy == name
        )
        out[name] = {
            "n_scored_forecasts": len(strat_scored),
            "brier": brier_score(ps, ys) if ps else None,
            "log_loss": log_loss(ps, ys) if ps else None,
            "trading_cash_flow": cash_in,
            "settlement_payout": settlement_payouts,
            "net_pnl": cash_in + settlement_payouts,
        }
    return out


def _to_parquet(df, path: str, schema: dict | None = None) -> None:
    if df.empty:
        # Write an empty file so consumers can rely on existence.
        df.to_parquet(path, index=False)
        return
    df.to_parquet(path, index=False)


def _jsonable(obj):
    """Convert numpy/pandas/dataclass nesting into stdlib-only types so
    json.dumps doesn't choke on the metrics dict."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
