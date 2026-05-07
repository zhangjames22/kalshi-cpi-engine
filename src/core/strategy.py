"""The Strategy ABC — the user's plug-in surface.

A Strategy implementer fills in three methods (`name`, `universe`, `on_tick`)
and optionally overrides `on_settlement`. The backtest engine drives the
lifecycle:

    1. engine ticks per orderbook snapshot (default) or on a custom clock
    2. engine asks strategy.universe(t, catalog) for relevant market_ids
    3. engine assembles MarketState for those ids at t and calls
       strategy.on_tick(t, state, features, portfolio) -> StrategyOutput
    4. engine scores forecasts (Brier/log-loss) and routes orders through
       the configured ExecutionModel (MidFill or CrossSpreadFill)
    5. when an event resolves, engine calls strategy.on_settlement

Strategies are stateful Python objects; the engine instantiates each one
once per backtest run. State that should survive across runs (cached
features, fitted models) belongs on the strategy as instance attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Protocol, runtime_checkable

from .forecast import Forecast
from .market import EventId, MarketCatalog, MarketId, Outcome
from .order import CancelOrder, Order
from .state import MarketState, PortfolioView


# ---------------------------------------------------------------------------
# Feature access — strategies' only window into non-market data.
# ---------------------------------------------------------------------------

@runtime_checkable
class FeatureView(Protocol):
    """Point-in-time feature lookup.

    Implementations enforce no-lookahead: every feature value carries an
    `available_at` timestamp (e.g. the FRED publication time, not the
    observation date), and only values with available_at <= t are returned.

    `get` returns the latest value at time t. `get_series` returns the
    full history truncated to t — strategies that fit models each tick
    use this to pull their training window. The return type is intentionally
    `Any` because feature backends differ (pandas Series, numpy array, plain
    list); the framework doesn't constrain it.
    """

    def get(self, name: str, t: datetime) -> Any:
        """Latest value of feature `name` available at time `t`."""
        ...

    def get_series(self, name: str, t: datetime) -> Any:
        """Full time series of `name`, truncated to available_at <= t."""
        ...


# ---------------------------------------------------------------------------
# Output contract.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyOutput:
    """What a Strategy emits each tick. Any field may be empty.

    - forecasts: scored even with no orders (Brier / log-loss / calibration).
    - orders:    routed through the engine's ExecutionModel.
    - cancels:   processed by the engine BEFORE new orders, removing the
                 named (strategy, client_id) from the resting book.

    A research strategy can return only forecasts. A pure execution strategy
    (e.g. a market-maker) can return forecasts + orders + cancels.
    """
    forecasts: tuple[Forecast, ...] = ()
    orders: tuple[Order, ...] = ()
    cancels: tuple[CancelOrder, ...] = ()

    @classmethod
    def empty(cls) -> "StrategyOutput":
        return cls()


# ---------------------------------------------------------------------------
# The Strategy ABC.
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """User-implemented prediction-market strategy.

    Subclass this and implement `name`, `universe`, and `on_tick`.
    Override `on_settlement` if you need to react to event resolutions
    (e.g. retrain a model, update a state machine, log P&L attribution).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this strategy.

        Used as the partition key in multi-strategy portfolios and reports.
        Must be unique within a single backtest run; the engine raises if
        two strategies share a name.
        """

    @abstractmethod
    def universe(
        self,
        t: datetime,
        catalog: MarketCatalog,
    ) -> Iterable[MarketId]:
        """Which markets do I want orderbook snapshots for at time `t`?

        Called once per tick. Returning a different universe between ticks
        is fine — the engine re-resolves snapshots each call. Strategies
        that always trade the same universe can compute it once in __init__
        and return the cached list here.
        """

    @abstractmethod
    def on_tick(
        self,
        t: datetime,
        state: MarketState,
        features: FeatureView,
        portfolio: PortfolioView,
    ) -> StrategyOutput:
        """Emit forecasts and/or orders given current world state.

        The strategy MUST NOT mutate `state`, `features`, or `portfolio` —
        they are read-only views. Returning StrategyOutput.empty() is a
        valid no-op (the strategy is sitting out this tick).
        """

    def on_settlement(self, event_id: EventId, outcome: Outcome) -> None:
        """Optional hook called when an event resolves.

        Default is a no-op so simple strategies don't need to implement it.
        Stateful strategies use this to update internal book-keeping;
        learning strategies use it to incorporate the realized outcome
        into the next training window.
        """
        return None
