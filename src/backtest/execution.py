"""Execution models — convert (Order, OrderbookSnapshot) into Fills.

Two implementations ship in v1:

- MidFill        — optimistic. Fills at the mid price ignoring book depth.
                   Useful for "what's the upper bound on this strategy's edge?"
- CrossSpreadFill — realistic. Fills at the touch (yes_ask for BUY, yes_bid
                   for SELL) up to the top-of-book size. Anything past the
                   top of book is rejected (IOC) or rests (GTC).

Both implementations are stateless. The engine owns the resting-order book
and is responsible for re-invoking the model against each new snapshot for
GTC orders.

Fill semantics
--------------
A passive GTC limit at L being hit by an aggressor would, in real markets,
fill at L. In a snapshot-only backtest we approximate by filling at the
touch price (yes_ask for BUY, yes_bid for SELL). This is consistent for
IOC and GTC and slightly pessimistic for resting limits — a small,
principled bias we'd rather have than complex book-replay machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from core.order import Fill, Order, OrderAction, Side
from core.state import OrderbookSnapshot


class ExecutionModel(ABC):
    """Stateless rule for converting an Order intent into Fills against a
    snapshot. The same `attempt_fill` is called for IOC submission and for
    each GTC re-check; the engine handles the resting-book bookkeeping."""

    @abstractmethod
    def attempt_fill(
        self,
        order: Order,
        remaining_qty: int,
        snapshot: OrderbookSnapshot,
        ts: datetime,
    ) -> list[Fill]:
        """Try to fill `remaining_qty` of `order` against `snapshot` at time
        `ts`. Returns 0 or 1 Fills (depth >1 is not modeled in v1)."""


# ---------------------------------------------------------------------------
# Helpers — pick the right side of the book for the order's (side, action).
# ---------------------------------------------------------------------------

def _book_side(
    snapshot: OrderbookSnapshot, side: Side, action: OrderAction
) -> tuple[float | None, int, bool]:
    """Return (touch_price, capacity, is_buy_direction).

    A BUY consumes the opposite side's ask; a SELL hits the opposite side's
    bid. Same logic on YES and NO books.
    """
    is_buy = action == OrderAction.BUY
    if side == Side.YES:
        if is_buy:
            return snapshot.yes_ask_price, snapshot.yes_ask_size, True
        return snapshot.yes_bid_price, snapshot.yes_bid_size, False
    else:
        if is_buy:
            return snapshot.no_ask_price, snapshot.no_ask_size, True
        return snapshot.no_bid_price, snapshot.no_bid_size, False


def _limit_allows(price: float, limit_price: float | None, is_buy: bool) -> bool:
    """A buy limit fills when touch <= limit; a sell limit when touch >= limit.
    A None limit is treated as a market order (always allowed)."""
    if limit_price is None:
        return True
    return price <= limit_price if is_buy else price >= limit_price


# ---------------------------------------------------------------------------
# MidFill — optimistic execution.
# ---------------------------------------------------------------------------

class MidFill(ExecutionModel):
    """Fills at the mid price of the relevant book, ignoring size.

    This is a research-mode execution model — useful for "what's the upper
    bound P&L if I could trade at mid?" Not suitable for any strategy whose
    edge is comparable to the spread.
    """

    def attempt_fill(
        self,
        order: Order,
        remaining_qty: int,
        snapshot: OrderbookSnapshot,
        ts: datetime,
    ) -> list[Fill]:
        if remaining_qty <= 0:
            return []

        mid = snapshot.yes_mid if order.side == Side.YES else snapshot.no_mid
        if mid is None:
            return []

        is_buy = order.action == OrderAction.BUY
        if not _limit_allows(mid, order.limit_price, is_buy):
            return []

        return [Fill(order=order, qty=remaining_qty, price=mid, ts=ts)]


# ---------------------------------------------------------------------------
# CrossSpreadFill — realistic execution.
# ---------------------------------------------------------------------------

class CrossSpreadFill(ExecutionModel):
    """Fills at the touch up to the top-of-book size.

    This is the v1 default. A market order takes liquidity at the touch;
    a limit order fills only when the touch is at-or-better than the limit,
    again at the touch price (see module docstring for the GTC-vs-touch
    pessimism note).
    """

    def attempt_fill(
        self,
        order: Order,
        remaining_qty: int,
        snapshot: OrderbookSnapshot,
        ts: datetime,
    ) -> list[Fill]:
        if remaining_qty <= 0:
            return []

        touch, capacity, is_buy = _book_side(snapshot, order.side, order.action)
        if touch is None or capacity <= 0:
            return []
        if not _limit_allows(touch, order.limit_price, is_buy):
            return []

        fill_qty = min(remaining_qty, capacity)
        return [Fill(order=order, qty=fill_qty, price=touch, ts=ts)]
