"""Tests for core.strategy — the user-facing plug-in surface.

A Strategy is mostly an ABC; the meaningful tests are:
  - it can't be instantiated bare
  - a minimal concrete subclass can implement the contract end-to-end
  - StrategyOutput.empty() and partial outputs work
  - on_settlement default no-op exists and accepts the engine's call shape
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pytest

from core.forecast import DiscreteForecast, normal_cdf
from core.market import MarketCatalog, MarketId, Outcome
from core.order import Order, OrderAction, Side
from core.state import MarketState, OrderbookSnapshot, Portfolio
from core.strategy import FeatureView, Strategy, StrategyOutput
from tests._fixtures import make_binary, make_event


T = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Strategy is abstract
# ---------------------------------------------------------------------------

def test_strategy_cannot_be_instantiated_bare():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_strategy_subclass_must_implement_required_methods():
    class Incomplete(Strategy):
        @property
        def name(self) -> str:
            return "incomplete"
        # missing universe + on_tick

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# A working minimal strategy.
# ---------------------------------------------------------------------------

class _AlwaysFiftyFifty(Strategy):
    """Forecasts 0.5 on every market in its universe; never trades."""

    def __init__(self, watched: list[MarketId]) -> None:
        self._watched = watched
        self.settlement_calls: list[tuple[str, Outcome]] = []

    @property
    def name(self) -> str:
        return "fifty-fifty"

    def universe(self, t, catalog):
        return self._watched

    def on_tick(self, t, state, features, portfolio):
        if not state.market_ids():
            return StrategyOutput.empty()
        return StrategyOutput(
            forecasts=(
                DiscreteForecast(
                    probabilities={mid: 0.5 for mid in state.market_ids()}
                ),
            ),
        )

    def on_settlement(self, event_id, outcome):
        self.settlement_calls.append((event_id, outcome))


class _NullFeatures:
    """Trivial FeatureView for tests."""
    def get(self, name: str, t):
        return None
    def get_series(self, name: str, t):
        return []


def test_minimal_strategy_can_run_one_tick():
    binary = make_binary("BIN")
    snap = OrderbookSnapshot(
        market_id="BIN", ts=T,
        yes_bid_price=0.4, yes_bid_size=100,
        yes_ask_price=0.5, yes_ask_size=100,
        no_bid_price=0.5,  no_bid_size=100,
        no_ask_price=0.6,  no_ask_size=100,
    )
    state = MarketState(t=T, snapshots={"BIN": snap})
    portfolio = Portfolio(cash=1000.0).view_for("fifty-fifty")

    strat = _AlwaysFiftyFifty(watched=["BIN"])
    out = strat.on_tick(T, state, _NullFeatures(), portfolio)

    assert len(out.forecasts) == 1
    assert out.orders == ()
    probs = out.forecasts[0].market_probabilities(lambda _eid: None)
    assert probs == {"BIN": 0.5}


def test_strategy_universe_can_be_dynamic():
    """universe() is called every tick — strategies can change what they
    watch over time."""
    class Switcher(Strategy):
        def __init__(self):
            self.calls = 0
        @property
        def name(self): return "switcher"
        def universe(self, t, catalog):
            self.calls += 1
            return ["A"] if self.calls == 1 else ["B", "C"]
        def on_tick(self, t, state, features, portfolio):
            return StrategyOutput.empty()

    s = Switcher()
    assert list(s.universe(T, None)) == ["A"]
    assert list(s.universe(T, None)) == ["B", "C"]


# ---------------------------------------------------------------------------
# StrategyOutput
# ---------------------------------------------------------------------------

def test_strategy_output_empty_helper():
    out = StrategyOutput.empty()
    assert out.forecasts == ()
    assert out.orders == ()


def test_strategy_output_can_carry_only_orders():
    o = Order(
        strategy="s", market_id="M1",
        side=Side.YES, action=OrderAction.BUY, qty=5,
    )
    out = StrategyOutput(orders=(o,))
    assert out.forecasts == ()
    assert out.orders == (o,)


def test_strategy_output_can_carry_both():
    o = Order(
        strategy="s", market_id="M1",
        side=Side.YES, action=OrderAction.BUY, qty=5,
    )
    f = DiscreteForecast(probabilities={"M1": 0.6})
    out = StrategyOutput(forecasts=(f,), orders=(o,))
    assert len(out.forecasts) == 1
    assert len(out.orders) == 1


# ---------------------------------------------------------------------------
# on_settlement default
# ---------------------------------------------------------------------------

def test_on_settlement_default_is_noop():
    """Subclasses that don't override on_settlement should still accept the
    engine's call shape without raising."""
    class Minimal(Strategy):
        @property
        def name(self): return "minimal"
        def universe(self, t, catalog): return []
        def on_tick(self, t, state, features, portfolio):
            return StrategyOutput.empty()

    s = Minimal()
    outcome = Outcome(
        event_id="E", winning_market_id="M", expiration_value=3.5, resolved_at=T,
    )
    # Should return None and not raise.
    assert s.on_settlement("E", outcome) is None


def test_overridden_on_settlement_records_calls():
    s = _AlwaysFiftyFifty(watched=[])
    outcome = Outcome(
        event_id="E", winning_market_id="M", expiration_value=None, resolved_at=T,
    )
    s.on_settlement("E", outcome)
    assert s.settlement_calls == [("E", outcome)]


# ---------------------------------------------------------------------------
# FeatureView is a structural Protocol
# ---------------------------------------------------------------------------

def test_null_features_satisfies_feature_view_protocol():
    nf = _NullFeatures()
    assert isinstance(nf, FeatureView)
