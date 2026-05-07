"""Trade intents and fills.

Orders are emitted by Strategies and consumed by an ExecutionModel
(MidFill or CrossSpreadFill) that the backtest engine wires up. The Order /
Fill / CancelOrder types here are pure data — they don't know how they're
filled.

Conventions
-----------
- All prices are in dollars in [0, 1] (Kalshi quotes prices as the price of
  one $1-paying contract).
- Quantities are whole contracts; partial-contract sizing is not supported.
- Side selects which contract you trade (YES or NO). Action selects whether
  you're opening (BUY) or closing (SELL). This four-way (Side x Action)
  encoding maps cleanly onto Kalshi's API and makes per-leg P&L explicit.
- Every Order carries the strategy name so multi-strategy backtests can
  attribute fills back to their author.
- Every Order carries a `client_id` (auto-generated UUID by default).
  Strategies use this to reference orders for cancellation. Auto-generation
  means strategies that never cancel never have to think about IDs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .market import MarketId


class Side(str, Enum):
    """Which side of the contract you're trading."""
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    """Open vs close. Position-aware execution uses this to decide whether
    to debit cash (BUY) or credit cash and reduce inventory (SELL)."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time-in-force.

    - IOC (immediate-or-cancel): fill what you can right now, drop the rest.
    - GTC (good-till-cancelled): book a resting order. Persists in the
      engine's resting book until filled or until the strategy emits a
      CancelOrder for its client_id.
    """
    IOC = "ioc"
    GTC = "gtc"


def _new_client_id() -> str:
    """Generate a fresh client_id. Pulled out so tests can monkeypatch if
    they need deterministic IDs."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Order:
    """A trade intent emitted by a Strategy.

    `limit_price` of None means a market order — the execution model decides
    the fill price. With a limit price set, the execution model only fills
    at limit_price or better.

    `client_id` is the strategy's handle on this order for the purpose of
    cancellation. Auto-generated as a UUID4 if not supplied. The id is
    namespaced by `strategy` — two strategies submitting orders with the
    same client_id do not collide.
    """
    strategy: str
    market_id: MarketId
    side: Side
    action: OrderAction
    qty: int
    limit_price: float | None = None
    tif: TimeInForce = TimeInForce.IOC
    client_id: str = field(default_factory=_new_client_id)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.limit_price is not None and not 0.0 <= self.limit_price <= 1.0:
            raise ValueError(
                f"limit_price must be in [0,1], got {self.limit_price}"
            )
        if not self.strategy:
            raise ValueError("Order must carry a non-empty strategy name")
        if not self.client_id:
            raise ValueError("Order must carry a non-empty client_id")


@dataclass(frozen=True)
class CancelOrder:
    """Request to cancel a previously-submitted GTC order.

    Resolution rules (handled by the engine):
      - If the (strategy, client_id) is currently resting → remove it.
      - If it was already fully filled → no-op (not an error).
      - If it never existed → no-op (not an error).

    The "no-op on missing" semantics are deliberate: races between order
    fills and cancel emissions are normal in real trading, and we don't
    want the backtest to diverge from live behavior on that edge case.
    """
    strategy: str
    client_id: str

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("CancelOrder must carry a non-empty strategy name")
        if not self.client_id:
            raise ValueError("CancelOrder must carry a non-empty client_id")


@dataclass(frozen=True)
class Fill:
    """An execution of an Order. May be partial (qty < order.qty)."""
    order: Order
    qty: int
    price: float
    ts: datetime

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"fill qty must be positive, got {self.qty}")
        if self.qty > self.order.qty:
            raise ValueError(
                f"fill qty {self.qty} exceeds order qty {self.order.qty}"
            )
        if not 0.0 <= self.price <= 1.0:
            raise ValueError(f"fill price must be in [0,1], got {self.price}")

    @property
    def notional(self) -> float:
        """Cash flow in dollars. Always positive; the engine signs it
        based on order.action."""
        return self.qty * self.price
