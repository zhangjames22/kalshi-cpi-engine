"""NullFeatureView — a no-op FeatureView for strategies that don't need features.

Sport-style strategies (NflHomeFavorite, prop bets) typically operate on
market data alone and never call into a feature backend. Constructing a
FRED-backed `FredFeatureView` just to pass it to the engine is awkward and
forces a network or panel dependency the strategy doesn't actually have.

`NullFeatureView` satisfies the `FeatureView` Protocol structurally and
returns sentinel values: `None` from `get`, `[]` from `get_series`. Any
strategy that accidentally reads features through it will see "missing"
data and can branch on it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class NullFeatureView:
    """No-op implementation of `core.strategy.FeatureView`.

    Stateless and threadsafe. Construct once and pass into any backtest
    engine whose strategies don't read features.
    """

    def get(self, name: str, t: datetime) -> Any:
        return None

    def get_series(self, name: str, t: datetime) -> list:
        return []
