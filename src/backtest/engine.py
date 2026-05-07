"""BacktestEngine — drives the strategy lifecycle against historical data.

Tick model
----------
The engine ticks **per orderbook snapshot**. For every snapshot in the
input stream (which must be time-sorted):

    1. Update the latest-snapshot-per-market map.
    2. Settle any events whose expiration_time <= snapshot.ts.
    3. Re-check resting GTC orders for the snapshot's market_id.
    4. For each strategy:
         a. Resolve `universe(t, catalog)`.
         b. Build a MarketState (latest snapshot per universe market).
         c. Call `on_tick`.
         d. Record forecasts.
         e. Submit orders (IOC: drop remainder; GTC: rest remainder).

Strategies that only care about daily updates can ignore intra-day ticks
by checking `state.staleness(market_id)` or by tracking their own last-acted
timestamp internally.

Multi-strategy
--------------
Multiple strategies share a single Portfolio (cash + position book), but
positions are partitioned by strategy name. The report aggregates per-strategy
metrics. Two strategies with the same `name` is an error.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from core.forecast import Forecast
from core.market import EventId, MarketCatalog, MarketId, Outcome, SettlementLoader
from core.order import CancelOrder, Fill, Order, OrderAction, Side, TimeInForce
from core.state import (
    MarketState,
    OrderbookSnapshot,
    Portfolio,
    Position,
)
from core.strategy import FeatureView, Strategy

from .execution import ExecutionModel
from .report import (
    BacktestReport,
    EquitySample,
    FillRecord,
    ForecastRecord,
    SettlementRecord,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """Knobs that don't belong to any one strategy.

    `max_staleness` filters which markets the engine will hand to a strategy:
    if no fresh snapshot for market M has arrived within this window, M is
    omitted from MarketState. Set to None to disable filtering — the strategy
    is then responsible for checking `state.staleness()` itself.
    """
    initial_cash: float = 10_000.0
    max_staleness: timedelta | None = None


# ---------------------------------------------------------------------------
# Resting-order book
# ---------------------------------------------------------------------------

@dataclass
class _RestingOrder:
    order: Order
    remaining_qty: int


class _RestingBook:
    """Per-(strategy, market) resting GTC orders.

    Indexed by market_id so a new snapshot only re-checks orders on that
    market — keeps the inner loop O(orders on this market) not O(all orders).
    """

    def __init__(self) -> None:
        self._by_market: dict[
            tuple[str, MarketId], list[_RestingOrder]
        ] = defaultdict(list)

    def add(self, order: Order, remaining_qty: int) -> None:
        self._by_market[(order.strategy, order.market_id)].append(
            _RestingOrder(order=order, remaining_qty=remaining_qty)
        )

    def orders_on(self, market_id: MarketId) -> list[tuple[str, _RestingOrder]]:
        out: list[tuple[str, _RestingOrder]] = []
        for (strategy, mid), orders in self._by_market.items():
            if mid == market_id:
                for ro in orders:
                    out.append((strategy, ro))
        return out

    def cancel(self, strategy: str, client_id: str) -> bool:
        """Remove the resting order with this (strategy, client_id), if any.

        Returns True if something was removed, False otherwise. Both outcomes
        are considered legitimate — see CancelOrder docstring for rationale.
        """
        for (strat, _mid), orders in list(self._by_market.items()):
            if strat != strategy:
                continue
            for i, ro in enumerate(orders):
                if ro.order.client_id == client_id:
                    del orders[i]
                    return True
        return False

    def prune_filled(self) -> None:
        """Drop any RestingOrder with remaining_qty <= 0, and clean up empty
        per-market lists."""
        for key in list(self._by_market.keys()):
            self._by_market[key] = [
                ro for ro in self._by_market[key] if ro.remaining_qty > 0
            ]
            if not self._by_market[key]:
                del self._by_market[key]

    def expire_after(self, t: datetime) -> None:
        """Drop orders whose markets have closed.

        Called sparingly — pruning is best-effort, the engine doesn't care
        about stale entries except for memory hygiene.
        """
        # The order itself doesn't carry close_time; the engine could enrich
        # this if needed. For v1 we rely on prune_filled and end-of-run cleanup.
        return


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Drive `Strategy`s through a stream of orderbook snapshots and produce
    a `BacktestReport`.

    Construction:
        engine = BacktestEngine(catalog, settlement, snapshots, features,
                                execution, config)
        engine.add_strategy(my_strategy)
        report = engine.run()
    """

    def __init__(
        self,
        catalog: MarketCatalog,
        settlement: SettlementLoader,
        snapshot_stream: Iterable[OrderbookSnapshot],
        feature_view: FeatureView,
        execution: ExecutionModel,
        config: BacktestConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.settlement = settlement
        self.snapshot_stream = snapshot_stream
        self.feature_view = feature_view
        self.execution = execution
        self.config = config or BacktestConfig()

        self.strategies: list[Strategy] = []
        self.portfolio = Portfolio(cash=self.config.initial_cash)

        self._latest: dict[MarketId, OrderbookSnapshot] = {}
        self._resting = _RestingBook()
        self._settled_events: set[EventId] = set()

        # Ledgers
        self._forecasts: list[ForecastRecord] = []
        self._fills: list[FillRecord] = []
        self._settlements: list[SettlementRecord] = []
        self._equity: list[EquitySample] = []

    def add_strategy(self, strategy: Strategy) -> None:
        if any(s.name == strategy.name for s in self.strategies):
            raise ValueError(
                f"Duplicate strategy name {strategy.name!r}; names must be unique"
            )
        self.strategies.append(strategy)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(self) -> BacktestReport:
        if not self.strategies:
            raise RuntimeError("No strategies registered; nothing to backtest.")

        last_t: datetime | None = None

        for snapshot in self.snapshot_stream:
            t = snapshot.ts
            self._latest[snapshot.market_id] = snapshot

            # 1) Settle anything that resolved between last_t and t.
            self._process_settlements(last_t, t)

            # 2) Re-check resting GTC orders for the new snapshot.
            self._fill_resting_against(snapshot, t)

            # 3) Tick each strategy.
            for strategy in self.strategies:
                state = self._build_market_state(strategy, t)
                output = strategy.on_tick(
                    t,
                    state,
                    self.feature_view,
                    self.portfolio.view_for(strategy.name),
                )
                self._record_forecasts(strategy.name, t, output.forecasts)
                # Cancels are processed before new orders so that a strategy
                # can replace a resting order in a single tick by emitting
                # (CancelOrder(old), Order(new)) without race semantics.
                self._process_cancels(strategy.name, output.cancels)
                self._submit_orders(strategy.name, t, output.orders)

            # 4) Snapshot equity (post-tick).
            self._equity.append(
                EquitySample(t=t, cash=self.portfolio.cash, mark=self._mark_to_market())
            )

            last_t = t

        # Final-pass settlement: events that resolve after the last snapshot.
        if last_t is not None:
            far_future = last_t + timedelta(days=10_000)
            self._process_settlements(last_t, far_future)

        return self._build_report()

    # -----------------------------------------------------------------
    # MarketState assembly
    # -----------------------------------------------------------------

    def _build_market_state(self, strategy: Strategy, t: datetime) -> MarketState:
        universe_ids = list(strategy.universe(t, self.catalog))
        snapshots: dict[MarketId, OrderbookSnapshot] = {}
        markets = {}
        for mid in universe_ids:
            snap = self._latest.get(mid)
            if snap is None:
                continue
            if self.config.max_staleness is not None:
                if t - snap.ts > self.config.max_staleness:
                    continue
            snapshots[mid] = snap
            markets[mid] = self.catalog.market(mid)
        return MarketState(t=t, snapshots=snapshots, markets=markets)

    # -----------------------------------------------------------------
    # Cancel processing
    # -----------------------------------------------------------------

    def _process_cancels(
        self, strategy_name: str, cancels: tuple[CancelOrder, ...]
    ) -> None:
        for cancel in cancels:
            if cancel.strategy != strategy_name:
                raise ValueError(
                    f"CancelOrder.strategy={cancel.strategy!r} does not match "
                    f"emitting strategy {strategy_name!r}"
                )
            # Both removal and not-found are legitimate outcomes. We don't log
            # the no-op case; strategies routinely emit cancels for orders
            # that have just filled, and that's not a bug.
            self._resting.cancel(cancel.strategy, cancel.client_id)

    # -----------------------------------------------------------------
    # Order submission
    # -----------------------------------------------------------------

    def _submit_orders(
        self, strategy_name: str, t: datetime, orders: tuple[Order, ...]
    ) -> None:
        for order in orders:
            if order.strategy != strategy_name:
                raise ValueError(
                    f"Order.strategy={order.strategy!r} does not match emitting "
                    f"strategy {strategy_name!r}"
                )
            snap = self._latest.get(order.market_id)
            if snap is None:
                log.warning(
                    "Dropping order on %s — no snapshot has been seen for it",
                    order.market_id,
                )
                continue

            fills = self.execution.attempt_fill(order, order.qty, snap, t)
            filled_qty = sum(f.qty for f in fills)
            self._apply_fills(fills)

            remaining = order.qty - filled_qty
            if remaining > 0 and order.tif == TimeInForce.GTC:
                self._resting.add(order, remaining)
            # IOC: drop remainder silently. Strategies can detect under-fills
            # by reading the fills ledger if they care.

    # -----------------------------------------------------------------
    # Resting-order processing
    # -----------------------------------------------------------------

    def _fill_resting_against(
        self, snapshot: OrderbookSnapshot, t: datetime
    ) -> None:
        for strategy_name, resting in self._resting.orders_on(snapshot.market_id):
            fills = self.execution.attempt_fill(
                resting.order, resting.remaining_qty, snapshot, t
            )
            filled_qty = sum(f.qty for f in fills)
            if filled_qty <= 0:
                continue
            self._apply_fills(fills)
            resting.remaining_qty -= filled_qty
        self._resting.prune_filled()

    # -----------------------------------------------------------------
    # Position + cash bookkeeping
    # -----------------------------------------------------------------

    def _apply_fills(self, fills: list[Fill]) -> None:
        for fill in fills:
            order = fill.order
            position = self.portfolio.position(order.strategy, order.market_id)

            if order.side == Side.YES:
                new_qty, new_avg = _update_avg(
                    cur_qty=position.yes_qty,
                    cur_avg=position.yes_avg_price,
                    fill_qty=fill.qty,
                    fill_price=fill.price,
                    is_buy=order.action == OrderAction.BUY,
                )
                new_pos = Position(
                    market_id=position.market_id,
                    yes_qty=new_qty, yes_avg_price=new_avg,
                    no_qty=position.no_qty, no_avg_price=position.no_avg_price,
                )
            else:
                new_qty, new_avg = _update_avg(
                    cur_qty=position.no_qty,
                    cur_avg=position.no_avg_price,
                    fill_qty=fill.qty,
                    fill_price=fill.price,
                    is_buy=order.action == OrderAction.BUY,
                )
                new_pos = Position(
                    market_id=position.market_id,
                    yes_qty=position.yes_qty, yes_avg_price=position.yes_avg_price,
                    no_qty=new_qty, no_avg_price=new_avg,
                )

            self.portfolio.set_position(order.strategy, new_pos)

            cash_delta = -fill.notional if order.action == OrderAction.BUY else fill.notional
            self.portfolio.cash += cash_delta

            self._fills.append(
                FillRecord(
                    ts=fill.ts,
                    strategy=order.strategy,
                    market_id=order.market_id,
                    side=order.side.value,
                    action=order.action.value,
                    qty=fill.qty,
                    price=fill.price,
                    cash_delta=cash_delta,
                )
            )

    # -----------------------------------------------------------------
    # Settlement
    # -----------------------------------------------------------------

    def _process_settlements(
        self, last_t: datetime | None, t: datetime
    ) -> None:
        if last_t is None:
            # First snapshot — only settle events that already resolved before t.
            window_start = datetime.min.replace(tzinfo=t.tzinfo) if t.tzinfo else datetime.min
        else:
            window_start = last_t

        events = list(self.catalog.events_resolving_between(window_start, t))
        for event in events:
            if event.event_id in self._settled_events:
                continue
            outcome = self.settlement.outcome(event.event_id)
            if outcome is None:
                continue
            self._settle_event(event.event_id, outcome)

    def _settle_event(self, event_id: EventId, outcome: Outcome) -> None:
        ladder = self.catalog.ladder(event_id)
        if ladder is not None:
            event_market_ids = [b.market_id for b in ladder.buckets]
        else:
            # Binary event: the event's single market shares the event_id by convention.
            event_market_ids = [event_id]

        for strategy in self.strategies:
            for mid in event_market_ids:
                position = self.portfolio.position(strategy.name, mid)
                if position.yes_qty == 0 and position.no_qty == 0:
                    continue
                payout = self._payout(position, mid, outcome)
                self.portfolio.cash += payout
                # Zero out the position post-settlement.
                self.portfolio.set_position(
                    strategy.name, Position(market_id=mid),
                )
                self._settlements.append(
                    SettlementRecord(
                        ts=outcome.resolved_at,
                        strategy=strategy.name,
                        event_id=event_id,
                        market_id=mid,
                        winning_market_id=outcome.winning_market_id,
                        yes_qty=position.yes_qty,
                        no_qty=position.no_qty,
                        payout=payout,
                    )
                )
            strategy.on_settlement(event_id, outcome)

        self._settled_events.add(event_id)

    @staticmethod
    def _payout(position: Position, market_id: MarketId, outcome: Outcome) -> float:
        """Cash paid into the portfolio when this market resolves.

        YES contracts pay $1 per contract iff this market_id won.
        NO contracts pay $1 per contract iff this market_id did NOT win.
        """
        won = market_id == outcome.winning_market_id
        if won:
            return float(position.yes_qty)
        else:
            return float(position.no_qty)

    # -----------------------------------------------------------------
    # Mark-to-market
    # -----------------------------------------------------------------

    def _mark_to_market(self) -> float:
        """Sum of (yes_qty * yes_mid + no_qty * no_mid) across all positions
        using the latest known snapshot. Returns 0 for positions whose market
        has no snapshot yet."""
        total = 0.0
        for (_strategy, mid), position in self.portfolio.positions.items():
            if position.yes_qty == 0 and position.no_qty == 0:
                continue
            snap = self._latest.get(mid)
            if snap is None:
                continue
            yes_mark = snap.yes_mid or 0.0
            no_mark = snap.no_mid or (1.0 - yes_mark if snap.yes_mid is not None else 0.0)
            total += position.yes_qty * yes_mark + position.no_qty * no_mark
        return total

    # -----------------------------------------------------------------
    # Forecast recording
    # -----------------------------------------------------------------

    def _record_forecasts(
        self, strategy_name: str, t: datetime, forecasts: tuple[Forecast, ...]
    ) -> None:
        for forecast in forecasts:
            probs = forecast.market_probabilities(self.catalog.ladder)
            for mid, p in probs.items():
                market = self.catalog.market(mid)
                self._forecasts.append(
                    ForecastRecord(
                        ts=t,
                        strategy=strategy_name,
                        market_id=mid,
                        event_id=market.event_id,
                        p_yes=p,
                    )
                )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    def _build_report(self) -> BacktestReport:
        return BacktestReport.build(
            strategies=[s.name for s in self.strategies],
            initial_cash=self.config.initial_cash,
            forecasts=self._forecasts,
            fills=self._fills,
            settlements=self._settlements,
            equity=self._equity,
            settlement_loader=self.settlement,
            catalog=self.catalog,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_avg(
    cur_qty: int, cur_avg: float, fill_qty: int, fill_price: float, is_buy: bool
) -> tuple[int, float]:
    """Update (qty, avg_price) given a new fill on the same leg.

    Buys grow qty and weight the avg price; sells reduce qty and leave the
    avg price unchanged on the remaining lot. Selling more than held flips
    sign — for v1 we forbid that and fail loudly.
    """
    if is_buy:
        new_qty = cur_qty + fill_qty
        if new_qty == 0:
            return 0, 0.0
        new_avg = (cur_qty * cur_avg + fill_qty * fill_price) / new_qty
        return new_qty, new_avg

    new_qty = cur_qty - fill_qty
    if new_qty < 0:
        raise ValueError(
            f"Sell of {fill_qty} exceeds current holding {cur_qty}; v1 doesn't "
            f"support shorting via SELL — to short on Kalshi, BUY the opposite side."
        )
    if new_qty == 0:
        return 0, 0.0
    return new_qty, cur_avg
