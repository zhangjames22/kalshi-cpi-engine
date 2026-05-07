"""Tests for core.state — orderbook snapshots, MarketState, Portfolio isolation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.state import (
    MarketState,
    OrderbookSnapshot,
    Portfolio,
    Position,
)
from tests._fixtures import make_binary


T = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _book(
    market_id: str = "M1",
    yes_bid: float | None = 0.40,
    yes_ask: float | None = 0.45,
    no_bid: float | None = 0.55,
    no_ask: float | None = 0.60,
) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        market_id=market_id,
        ts=T,
        yes_bid_price=yes_bid,
        yes_bid_size=100 if yes_bid is not None else 0,
        yes_ask_price=yes_ask,
        yes_ask_size=100 if yes_ask is not None else 0,
        no_bid_price=no_bid,
        no_bid_size=100 if no_bid is not None else 0,
        no_ask_price=no_ask,
        no_ask_size=100 if no_ask is not None else 0,
    )


# ---------------------------------------------------------------------------
# OrderbookSnapshot
# ---------------------------------------------------------------------------

def test_yes_mid_is_midpoint_when_two_sided():
    s = _book(yes_bid=0.40, yes_ask=0.50)
    assert s.yes_mid == 0.45
    assert s.yes_spread == pytest.approx(0.10)


def test_yes_mid_falls_back_to_quoted_side():
    assert _book(yes_bid=None, yes_ask=0.55).yes_mid == 0.55
    assert _book(yes_bid=0.30, yes_ask=None).yes_mid == 0.30


def test_yes_mid_is_none_when_book_empty():
    s = _book(yes_bid=None, yes_ask=None)
    assert s.yes_mid is None
    assert s.yes_spread is None


def test_yes_spread_is_none_when_one_sided():
    assert _book(yes_bid=None, yes_ask=0.55).yes_spread is None


def test_no_mid_works_symmetrically():
    s = _book(no_bid=0.50, no_ask=0.60)
    assert s.no_mid == 0.55


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------

def test_market_state_lookup():
    s = _book("M1")
    state = MarketState(t=T, snapshots={"M1": s})
    assert state["M1"] is s
    assert "M1" in state
    assert "M2" not in state
    assert len(state) == 1
    assert state.get("M2") is None


def test_market_state_missing_lookup_raises():
    state = MarketState(t=T, snapshots={})
    with pytest.raises(KeyError):
        _ = state["MISSING"]


def test_market_state_staleness_is_engine_minus_snapshot_time():
    snapshot_ts = T - timedelta(minutes=5)
    snap = OrderbookSnapshot(
        market_id="M1", ts=snapshot_ts,
        yes_bid_price=0.4, yes_bid_size=100,
        yes_ask_price=0.5, yes_ask_size=100,
        no_bid_price=0.5,  no_bid_size=100,
        no_ask_price=0.6,  no_ask_size=100,
    )
    state = MarketState(t=T, snapshots={"M1": snap})
    assert state.snapshot_t("M1") == snapshot_ts
    assert state.staleness("M1") == timedelta(minutes=5)


def test_market_state_zero_staleness_when_snapshot_matches_engine_t():
    snap = OrderbookSnapshot(
        market_id="M1", ts=T,
        yes_bid_price=0.4, yes_bid_size=100,
        yes_ask_price=0.5, yes_ask_size=100,
        no_bid_price=0.5,  no_bid_size=100,
        no_ask_price=0.6,  no_ask_size=100,
    )
    state = MarketState(t=T, snapshots={"M1": snap})
    assert state.staleness("M1") == timedelta(0)


def test_market_state_market_metadata_lookup():
    binary = make_binary("BIN")
    snap = _book("BIN")
    state = MarketState(t=T, snapshots={"BIN": snap}, markets={"BIN": binary})
    assert state.market("BIN") is binary


def test_market_state_market_metadata_missing_raises_with_clear_msg():
    state = MarketState(t=T, snapshots={"M1": _book("M1")}, markets={})
    with pytest.raises(KeyError, match="not in this MarketState"):
        state.market("M1")


# ---------------------------------------------------------------------------
# Portfolio multi-strategy isolation
# ---------------------------------------------------------------------------

def test_portfolio_returns_zero_position_for_unknown_market():
    p = Portfolio(cash=1000.0)
    pos = p.position("strat-A", "M1")
    assert pos.yes_qty == 0
    assert pos.no_qty == 0


def test_portfolio_isolates_positions_by_strategy():
    p = Portfolio(cash=1000.0)
    p.set_position("strat-A", Position(market_id="M1", yes_qty=10, yes_avg_price=0.4))
    p.set_position("strat-B", Position(market_id="M1", yes_qty=5, yes_avg_price=0.6))

    a = p.position("strat-A", "M1")
    b = p.position("strat-B", "M1")
    assert a.yes_qty == 10
    assert b.yes_qty == 5
    # Each strategy's view sees only its own positions.
    assert p.view_for("strat-A").position("M1").yes_qty == 10
    assert p.view_for("strat-B").position("M1").yes_qty == 5


def test_portfolio_view_lists_only_own_positions():
    p = Portfolio(cash=1000.0)
    p.set_position("strat-A", Position(market_id="M1", yes_qty=10))
    p.set_position("strat-A", Position(market_id="M2", yes_qty=20))
    p.set_position("strat-B", Position(market_id="M1", yes_qty=5))

    view_a = p.view_for("strat-A")
    a_markets = sorted(pos.market_id for pos in view_a.positions())
    assert a_markets == ["M1", "M2"]

    view_b = p.view_for("strat-B")
    b_markets = sorted(pos.market_id for pos in view_b.positions())
    assert b_markets == ["M1"]


def test_portfolio_view_shares_cash():
    p = Portfolio(cash=500.0)
    assert p.view_for("strat-A").cash == 500.0
    assert p.view_for("strat-B").cash == 500.0


def test_position_cost_basis_sums_both_sides():
    pos = Position(
        market_id="M1",
        yes_qty=10, yes_avg_price=0.4,
        no_qty=5,  no_avg_price=0.55,
    )
    assert pos.cost_basis == pytest.approx(10 * 0.4 + 5 * 0.55)


# ---------------------------------------------------------------------------
# PortfolioView immutability
# ---------------------------------------------------------------------------

def test_portfolio_view_blocks_attribute_assignment():
    """Strategies must not be able to swap out the underlying Portfolio
    or rebind the strategy scope by assignment."""
    p = Portfolio(cash=100.0)
    v = p.view_for("strat-A")

    with pytest.raises(AttributeError):
        v.cash = 999.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        v._portfolio = Portfolio(cash=0.0)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        v._strategy = "strat-B"  # type: ignore[misc]
    # __slots__ also blocks creating brand-new attributes.
    with pytest.raises(AttributeError):
        v.injected = "anything"  # type: ignore[misc]


def test_portfolio_view_blocks_attribute_deletion():
    p = Portfolio(cash=100.0)
    v = p.view_for("strat-A")
    with pytest.raises(AttributeError):
        del v.cash  # type: ignore[misc]


def test_portfolio_view_has_no_dict():
    """__slots__ is the mechanism. Confirm no per-instance __dict__ exists,
    which prevents Python's default attribute storage from being used."""
    p = Portfolio(cash=100.0)
    v = p.view_for("strat-A")
    assert not hasattr(v, "__dict__")


def test_portfolio_view_strategy_property_is_readable():
    """The scoped strategy name is exposed read-only for diagnostics."""
    p = Portfolio(cash=100.0)
    v = p.view_for("strat-A")
    assert v.strategy == "strat-A"
    with pytest.raises(AttributeError):
        v.strategy = "strat-B"  # type: ignore[misc]


def test_portfolio_view_reflects_portfolio_mutations():
    """Read-only here means 'the view itself is read-only'; the underlying
    Portfolio is still mutable by the engine. Cash/positions read through
    the view should reflect those engine-side updates."""
    p = Portfolio(cash=100.0)
    v = p.view_for("strat-A")
    assert v.cash == 100.0
    p.cash = 250.0
    assert v.cash == 250.0
    p.set_position("strat-A", Position(market_id="M1", yes_qty=7))
    assert v.position("M1").yes_qty == 7
