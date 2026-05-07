"""Tests for core.order — Order/Fill validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.order import Fill, Order, OrderAction, Side, TimeInForce


T = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _order(**kwargs) -> Order:
    base = dict(
        strategy="strat-A",
        market_id="M1",
        side=Side.YES,
        action=OrderAction.BUY,
        qty=10,
    )
    base.update(kwargs)
    return Order(**base)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

def test_order_defaults_to_ioc_market():
    o = _order()
    assert o.tif == TimeInForce.IOC
    assert o.limit_price is None


def test_order_rejects_zero_qty():
    with pytest.raises(ValueError, match="qty must be positive"):
        _order(qty=0)


def test_order_rejects_negative_qty():
    with pytest.raises(ValueError, match="qty must be positive"):
        _order(qty=-5)


def test_order_rejects_out_of_range_limit_price():
    with pytest.raises(ValueError, match="limit_price"):
        _order(limit_price=1.01)
    with pytest.raises(ValueError, match="limit_price"):
        _order(limit_price=-0.5)


def test_order_accepts_endpoints():
    _order(limit_price=0.0)
    _order(limit_price=1.0)


def test_order_requires_strategy_name():
    with pytest.raises(ValueError, match="strategy"):
        _order(strategy="")


def test_order_is_frozen():
    o = _order()
    with pytest.raises(Exception):  # FrozenInstanceError, but exact type varies
        o.qty = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def test_fill_partial_is_allowed():
    o = _order(qty=10)
    f = Fill(order=o, qty=4, price=0.42, ts=T)
    assert f.qty == 4
    assert f.notional == pytest.approx(4 * 0.42)


def test_fill_rejects_overfill():
    o = _order(qty=5)
    with pytest.raises(ValueError, match="exceeds order qty"):
        Fill(order=o, qty=6, price=0.50, ts=T)


def test_fill_rejects_zero_qty():
    o = _order(qty=5)
    with pytest.raises(ValueError, match="fill qty must be positive"):
        Fill(order=o, qty=0, price=0.50, ts=T)


def test_fill_rejects_out_of_range_price():
    o = _order(qty=5)
    with pytest.raises(ValueError, match="fill price"):
        Fill(order=o, qty=1, price=1.5, ts=T)
    with pytest.raises(ValueError, match="fill price"):
        Fill(order=o, qty=1, price=-0.1, ts=T)
