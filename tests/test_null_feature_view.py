"""Tests for data.features.null.NullFeatureView.

Confirms the structural Protocol fit and the documented sentinel returns.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.strategy import FeatureView
from data.features import NullFeatureView


T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_null_feature_view_satisfies_feature_view_protocol():
    """FeatureView is `@runtime_checkable`, so isinstance(obj, FeatureView)
    works structurally — we don't need explicit subclassing."""
    assert isinstance(NullFeatureView(), FeatureView)


def test_null_feature_view_get_returns_none():
    fv = NullFeatureView()
    assert fv.get("anything", T) is None
    assert fv.get("", T) is None


def test_null_feature_view_get_series_returns_empty_list():
    fv = NullFeatureView()
    assert fv.get_series("anything", T) == []
    # Returning a fresh list each call is fine, but the empty value must be
    # a list (not None) so callers can iterate without a None check.
    assert isinstance(fv.get_series("anything", T), list)
