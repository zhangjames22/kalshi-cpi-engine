"""Tests for GTC cancel support.

Three behaviors to lock in:
  - cancel-before-fill: removes the resting order from the book
  - cancel-after-fill:  no-op, not an error
  - cancel-of-unknown-client_id: no-op, not an error
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from core.market import MarketCatalog, MarketId
from core.order import CancelOrder, Order, OrderAction, Side, TimeInForce
from core.state import MarketState, OrderbookSnapshot, PortfolioView
from core.strategy import FeatureView, Strategy, StrategyOutput
from data.stub import (
    InMemoryCatalog,
    InMemoryFeatureView,
    InMemorySettlementLoader,
    make_uniform_snapshot,
)
from tests._fixtures import make_binary


# ---------------------------------------------------------------------------
# Test scaffolding — a tiny binary market and a scripted strategy.
# ---------------------------------------------------------------------------

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _binary_catalog():
    bm = make_binary("BIN-1")
    return InMemoryCatalog(
        markets_by_id={"BIN-1": bm},
        events_by_id={},   # no events scheduled to resolve in the run
        ladders_by_event={},
    )


def _snapshot_stream(prices: list[tuple[int, float, float]]) -> list[OrderbookSnapshot]:
    """prices = [(minute_offset, yes_bid, yes_ask), ...]. The opposite (NO)
    side is derived as 1 - yes_ask / 1 - yes_bid for symmetry."""
    out = []
    for minutes, yb, ya in prices:
        ts = T0 + timedelta(minutes=minutes)
        out.append(OrderbookSnapshot(
            market_id="BIN-1", ts=ts,
            yes_bid_price=yb, yes_bid_size=100,
            yes_ask_price=ya, yes_ask_size=100,
            no_bid_price=1 - ya, no_bid_size=100,
            no_ask_price=1 - yb, no_ask_size=100,
        ))
    return out


class _ScriptedStrategy(Strategy):
    """Emit one set of orders/cancels per tick from a scripted plan, keyed
    by the snapshot timestamp the engine ticks on."""

    def __init__(self, plan: dict[datetime, StrategyOutput]) -> None:
        self._plan = plan

    @property
    def name(self) -> str:
        return "scripted"

    def universe(self, t, catalog) -> Iterable[MarketId]:
        return ["BIN-1"]

    def on_tick(self, t, state, features, portfolio) -> StrategyOutput:
        return self._plan.get(t, StrategyOutput.empty())


def _engine_for(plan: dict[datetime, StrategyOutput], snapshots: list[OrderbookSnapshot]):
    catalog = _binary_catalog()
    engine = BacktestEngine(
        catalog=catalog,
        settlement=InMemorySettlementLoader(outcomes_by_event={}),
        snapshot_stream=snapshots,
        feature_view=InMemoryFeatureView(series={}),
        execution=CrossSpreadFill(),
        config=BacktestConfig(initial_cash=10_000.0),
    )
    engine.add_strategy(_ScriptedStrategy(plan))
    return engine


# ---------------------------------------------------------------------------
# Cancel-before-fill: removes from the resting book.
# ---------------------------------------------------------------------------

def test_cancel_before_fill_removes_resting_order():
    # Tick 0: rest a deep-OTM GTC buy at 0.10 — won't fill against ask=0.45.
    # Tick 1: cancel it.
    # Tick 2: market crosses (ask drops to 0.05) — would have filled if still resting.
    snapshots = _snapshot_stream([
        (0, 0.40, 0.45),
        (1, 0.40, 0.45),
        (2, 0.04, 0.05),  # if still resting, this would cross our 0.10 limit
    ])
    rest_order = Order(
        strategy="scripted", market_id="BIN-1",
        side=Side.YES, action=OrderAction.BUY,
        qty=10, limit_price=0.10, tif=TimeInForce.GTC,
        client_id="A1",
    )
    plan = {
        snapshots[0].ts: StrategyOutput(orders=(rest_order,)),
        snapshots[1].ts: StrategyOutput(cancels=(CancelOrder(strategy="scripted", client_id="A1"),)),
    }
    engine = _engine_for(plan, snapshots)
    report = engine.run()

    # No fills, no positions, cash unchanged.
    assert report.metrics["n_fills"] == 0
    assert report.total_return() == 0.0


# ---------------------------------------------------------------------------
# Cancel-after-fill: no-op, not an error.
# ---------------------------------------------------------------------------

def test_cancel_after_fill_is_noop():
    # Tick 0: rest a marketable GTC buy at 0.50 — fills immediately against ask=0.45.
    # Tick 1: cancel the same client_id. Order is gone; cancel is harmless.
    snapshots = _snapshot_stream([
        (0, 0.40, 0.45),
        (1, 0.40, 0.45),
    ])
    rest_order = Order(
        strategy="scripted", market_id="BIN-1",
        side=Side.YES, action=OrderAction.BUY,
        qty=10, limit_price=0.50, tif=TimeInForce.GTC,
        client_id="A1",
    )
    plan = {
        snapshots[0].ts: StrategyOutput(orders=(rest_order,)),
        snapshots[1].ts: StrategyOutput(cancels=(CancelOrder(strategy="scripted", client_id="A1"),)),
    }
    engine = _engine_for(plan, snapshots)
    report = engine.run()

    # The order filled at tick 0 (price=ask=0.45, qty=10) and was never resting
    # by the time the cancel arrived. Cancel is a no-op; no exception raised.
    assert report.metrics["n_fills"] == 1
    fill = report.fills[0]
    assert fill.qty == 10
    assert fill.price == 0.45


# ---------------------------------------------------------------------------
# Cancel-of-unknown-client_id: no-op, not an error.
# ---------------------------------------------------------------------------

def test_cancel_of_unknown_client_id_is_noop():
    snapshots = _snapshot_stream([(0, 0.40, 0.45)])
    plan = {
        snapshots[0].ts: StrategyOutput(
            cancels=(CancelOrder(strategy="scripted", client_id="DOES_NOT_EXIST"),)
        ),
    }
    engine = _engine_for(plan, snapshots)
    report = engine.run()  # should not raise

    assert report.metrics["n_fills"] == 0
    assert report.total_return() == 0.0


# ---------------------------------------------------------------------------
# Sanity: cancel-then-replace pattern works in a single tick.
# ---------------------------------------------------------------------------

def test_cancel_then_replace_in_same_tick():
    """A strategy can cancel a resting order and submit a new one in the
    same tick. Cancels are processed first so the new order doesn't fight
    the old one for any resting-book slot."""
    snapshots = _snapshot_stream([
        (0, 0.40, 0.45),
        (1, 0.40, 0.45),
    ])
    old = Order(
        strategy="scripted", market_id="BIN-1",
        side=Side.YES, action=OrderAction.BUY,
        qty=10, limit_price=0.10, tif=TimeInForce.GTC, client_id="A1",
    )
    new = Order(
        strategy="scripted", market_id="BIN-1",
        side=Side.YES, action=OrderAction.BUY,
        qty=5, limit_price=0.20, tif=TimeInForce.GTC, client_id="A2",
    )
    plan = {
        snapshots[0].ts: StrategyOutput(orders=(old,)),
        snapshots[1].ts: StrategyOutput(
            cancels=(CancelOrder(strategy="scripted", client_id="A1"),),
            orders=(new,),
        ),
    }
    engine = _engine_for(plan, snapshots)
    report = engine.run()

    # Neither order would have filled (both sub-ask). No fills expected.
    # The important assertion is that engine.run() completes without error
    # and the resting book ended with one order (the replacement).
    assert report.metrics["n_fills"] == 0
    assert len(engine._resting._by_market) == 1
    resting = list(engine._resting._by_market.values())[0]
    assert len(resting) == 1
    assert resting[0].order.client_id == "A2"


# ---------------------------------------------------------------------------
# Auto-generated client_ids are unique by default.
# ---------------------------------------------------------------------------

def test_orders_have_unique_auto_generated_client_ids_by_default():
    o1 = Order(
        strategy="s", market_id="M1",
        side=Side.YES, action=OrderAction.BUY, qty=1,
    )
    o2 = Order(
        strategy="s", market_id="M1",
        side=Side.YES, action=OrderAction.BUY, qty=1,
    )
    assert o1.client_id != o2.client_id
    assert len(o1.client_id) > 0
