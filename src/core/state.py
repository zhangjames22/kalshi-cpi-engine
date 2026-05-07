"""Snapshot views of orderbook and portfolio state.

`MarketState` is what a Strategy sees of the market world at a single tick;
`PortfolioView` is what it sees of its own holdings. Both are read-only —
the backtest engine owns the underlying mutable Portfolio and exposes
strategy-scoped views so multi-strategy runs stay separable.

Staleness
---------
The engine ticks per orderbook snapshot. Every tick has two timestamps:

  - `t`            — the engine clock at which on_tick is being called.
  - `snapshot.ts`  — when the orderbook snapshot for a given market was
                     captured. May be earlier than `t` if no fresh snapshot
                     for that market has arrived yet.

`MarketState.staleness(market_id)` and `MarketState.snapshot_t(market_id)`
surface this gap explicitly. Strategies that care about latency (anything
trading at finer than daily resolution) MUST consult staleness — silently
trading on a stale book is a real backtest hazard and we want it visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from .market import Market, MarketId


# ---------------------------------------------------------------------------
# Orderbook snapshots — what the engine reads from historical data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderbookSnapshot:
    """Top-of-book bid/ask for a single market at a single instant.

    Prices are in dollars [0, 1]. Sizes are integer contracts. A side with
    no resting orders has bid_price=None / ask_price=None and size=0.

    The v1 backtest only models top-of-book; deeper book levels can be
    added later without changing this surface (we'd extend with optional
    `levels` collections).
    """
    market_id: MarketId
    ts: datetime
    yes_bid_price: float | None
    yes_bid_size: int
    yes_ask_price: float | None
    yes_ask_size: int
    no_bid_price: float | None
    no_bid_size: int
    no_ask_price: float | None
    no_ask_size: int

    @property
    def yes_mid(self) -> float | None:
        """Midpoint of the YES book. None if the book is fully empty;
        falls back to the single quoted side if only one is present."""
        if self.yes_bid_price is not None and self.yes_ask_price is not None:
            return (self.yes_bid_price + self.yes_ask_price) / 2.0
        return self.yes_bid_price if self.yes_bid_price is not None else self.yes_ask_price

    @property
    def yes_spread(self) -> float | None:
        if self.yes_bid_price is None or self.yes_ask_price is None:
            return None
        return self.yes_ask_price - self.yes_bid_price

    @property
    def no_mid(self) -> float | None:
        if self.no_bid_price is not None and self.no_ask_price is not None:
            return (self.no_bid_price + self.no_ask_price) / 2.0
        return self.no_bid_price if self.no_bid_price is not None else self.no_ask_price


# ---------------------------------------------------------------------------
# Per-tick view passed to the Strategy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketState:
    """Snapshot of all markets in a strategy's universe at engine-time `t`.

    Attributes:
        t:         engine clock for this tick (the time on_tick was called)
        snapshots: latest orderbook snapshot per market in the universe
        markets:   Market metadata per market_id (event_id, ladder bounds,
                   close_time, etc). Lets strategies group by event without
                   round-tripping through the catalog.

    The (t, snapshots) split exists because tick time and snapshot time can
    diverge — see `staleness`.
    """
    t: datetime
    snapshots: Mapping[MarketId, OrderbookSnapshot]
    markets: Mapping[MarketId, Market] = field(default_factory=dict)

    # ---- snapshot lookup --------------------------------------------------

    def __getitem__(self, market_id: MarketId) -> OrderbookSnapshot:
        return self.snapshots[market_id]

    def __contains__(self, market_id: object) -> bool:
        return market_id in self.snapshots

    def __len__(self) -> int:
        return len(self.snapshots)

    def market_ids(self) -> Iterable[MarketId]:
        return self.snapshots.keys()

    def get(self, market_id: MarketId) -> OrderbookSnapshot | None:
        return self.snapshots.get(market_id)

    # ---- metadata + timing ------------------------------------------------

    def market(self, market_id: MarketId) -> Market:
        """Market metadata for `market_id`. Raises if not in the universe."""
        try:
            return self.markets[market_id]
        except KeyError as e:
            raise KeyError(
                f"market {market_id!r} is not in this MarketState's universe"
            ) from e

    def snapshot_t(self, market_id: MarketId) -> datetime:
        """Timestamp of the orderbook snapshot used for `market_id`."""
        return self.snapshots[market_id].ts

    def staleness(self, market_id: MarketId) -> timedelta:
        """How old is the snapshot for `market_id` relative to engine time?

        Always >= 0. Strategies that require fresh data should compare this
        to a threshold and skip stale markets explicitly — the engine never
        silently filters stale markets.
        """
        return self.t - self.snapshots[market_id].ts


# ---------------------------------------------------------------------------
# Portfolio.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """Holdings in a single market, owned by a single strategy.

    YES and NO are tracked separately because Kalshi doesn't net them — a
    strategy can be long YES and long NO on the same market simultaneously
    (which is a bet on resolution, not direction). The engine collapses
    same-side fills into the avg-price field; opposite-side opens stack.
    """
    market_id: MarketId
    yes_qty: int = 0
    no_qty: int = 0
    yes_avg_price: float = 0.0
    no_avg_price: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Total dollars paid to open the current position."""
        return self.yes_qty * self.yes_avg_price + self.no_qty * self.no_avg_price


@dataclass
class Portfolio:
    """Mutable portfolio. Owned by the backtest engine; strategies see it
    only through PortfolioView.

    Positions are keyed by (strategy_name, market_id) so a single backtest
    run with N strategies can attribute each fill to its author. Cash is
    shared across strategies in v1 — if we later want per-strategy capital
    budgets, that goes here as a `cash_by_strategy` mapping without changing
    the Strategy interface.
    """
    cash: float
    positions: dict[tuple[str, MarketId], Position] = field(default_factory=dict)

    def position(self, strategy: str, market_id: MarketId) -> Position:
        """Position for (strategy, market_id), or a zeroed Position if the
        strategy has never traded this market."""
        return self.positions.get(
            (strategy, market_id),
            Position(market_id=market_id),
        )

    def set_position(self, strategy: str, position: Position) -> None:
        self.positions[(strategy, position.market_id)] = position

    def view_for(self, strategy: str) -> "PortfolioView":
        """Return the read-only view a Strategy sees on each tick."""
        return PortfolioView(_portfolio=self, _strategy=strategy)


@dataclass(frozen=True)
class PortfolioView:
    """Read-only window into the shared Portfolio, scoped to one strategy.

    A strategy can only see its own positions — it cannot inspect what
    other strategies in a multi-strategy run are holding. Cash is reported
    as the shared pool; strategies are expected to size responsibly.
    """
    _portfolio: Portfolio
    _strategy: str

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    def position(self, market_id: MarketId) -> Position:
        return self._portfolio.position(self._strategy, market_id)

    def positions(self) -> list[Position]:
        return [
            p
            for (s, _), p in self._portfolio.positions.items()
            if s == self._strategy
        ]
