"""CPI-Normal strategy — notebook 03's logic packaged as a Strategy.

The strategy owns its CPI-specific feature engineering: derive YoY from a
raw CPI level series, then fit a Normal(mu, sigma) and emit one
DistributionForecast per active CPI event. The FeatureView only serves
raw FRED observations.

Each tick:
  1. Pull the raw CPI level series (default `core_cpi` from FRED).
  2. Compute monthly YoY = level[t] / level[t-12] - 1, in percent units.
  3. Fit Normal(mu, sigma) where:
        mu    = latest YoY value (in percent)
        sigma = std-dev of monthly YoY changes, annualized via sqrt(12),
                clamped to a minimum so bucket probabilities don't degenerate.
  4. Emit one DistributionForecast per distinct event in scope; the engine
     projects the CDF onto each event's bucket ladder.

No orders are emitted in v1. The strategy is purely a forecast publisher;
calibration and Brier are scored from the forecast ledger when events resolve.
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from core.forecast import DistributionForecast, normal_cdf
from core.market import MarketCatalog, MarketId, SeriesId
from core.state import MarketState, PortfolioView
from core.strategy import FeatureView, Strategy, StrategyOutput


class CpiNormalStrategy(Strategy):
    """Forecast Kalshi CPI YoY buckets via a fitted Normal distribution."""

    def __init__(
        self,
        series_id: SeriesId,
        feature_name: str = "core_cpi",
        min_sigma: float = 0.3,
    ) -> None:
        self._series_id = series_id
        self._feature_name = feature_name
        self._min_sigma = min_sigma

    @property
    def name(self) -> str:
        return "cpi_normal"

    # ------------------------------------------------------------------
    # Universe — every active market in the configured CPI series.
    # ------------------------------------------------------------------

    def universe(self, t, catalog: MarketCatalog) -> Iterable[MarketId]:
        return [m.market_id for m in catalog.markets_for_series(self._series_id, t)]

    # ------------------------------------------------------------------
    # Forecast — one DistributionForecast per distinct event in scope.
    # ------------------------------------------------------------------

    def on_tick(
        self,
        t,
        state: MarketState,
        features: FeatureView,
        portfolio: PortfolioView,
    ) -> StrategyOutput:
        if len(state) == 0:
            return StrategyOutput.empty()

        levels = self._get_level_series(features, t)
        if levels is None or len(levels) < 13:
            # Need at least 13 monthly observations to compute one YoY.
            return StrategyOutput.empty()

        yoy_pct = self._compute_yoy_pct(levels)
        if len(yoy_pct) < 2:
            return StrategyOutput.empty()

        mu, sigma = self._fit_normal(yoy_pct)

        events = sorted({state.market(mid).event_id for mid in state.market_ids()})

        forecasts = tuple(
            DistributionForecast(event_id=eid, cdf=normal_cdf(mu, sigma))
            for eid in events
        )
        return StrategyOutput(forecasts=forecasts)

    # ------------------------------------------------------------------
    # Feature plumbing.
    # ------------------------------------------------------------------

    def _get_level_series(self, features: FeatureView, t) -> pd.Series | None:
        """Try to pull a date-indexed level series; fall back to a list.

        FRED's FeatureView exposes a richer `get_indexed_series`; the
        synthetic stub view only exposes the Protocol's `get_series` (a
        plain list). Either is enough to compute a monthly YoY: with the
        list path we assume the values are already monthly observations.
        """
        getter = getattr(features, "get_indexed_series", None)
        if getter is not None:
            s = getter(self._feature_name, t)
            if s is None or s.empty:
                return None
            return self._collapse_to_monthly(s)

        history = features.get_series(self._feature_name, t)
        if not history:
            return None
        # Assume already monthly, evenly spaced; index doesn't matter for
        # YoY because we only ever do `.iloc[-1] / .iloc[-13]` style math.
        return pd.Series(history)

    @staticmethod
    def _collapse_to_monthly(s: pd.Series) -> pd.Series:
        """Reduce a daily/sparse series to one observation per month.

        FRED CPI is monthly already, but its index may be daily after
        forward-fill. Take the first non-null value per month.
        """
        s = s.dropna()
        if s.empty:
            return s
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s = pd.Series(s.values, index=idx)
        return s.groupby(s.index.to_period("M")).first()

    # ------------------------------------------------------------------
    # YoY derivation + Normal fit.
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_yoy_pct(levels: pd.Series) -> list[float]:
        """Compute YoY % from a monthly level series.

        yoy_pct[i] = (levels[i] / levels[i-12] - 1) * 100
        """
        if len(levels) < 13:
            return []
        arr = levels.values.astype(float)
        out = []
        for i in range(12, len(arr)):
            prior = arr[i - 12]
            if prior == 0 or pd.isna(prior) or pd.isna(arr[i]):
                continue
            out.append(float((arr[i] / prior - 1.0) * 100.0))
        return out

    def _fit_normal(self, yoy_pct: list[float]) -> tuple[float, float]:
        """Fit Normal(mu, sigma) to a YoY-in-percent series."""
        mu = yoy_pct[-1]

        diffs = [b - a for a, b in zip(yoy_pct, yoy_pct[1:])]
        if not diffs:
            return mu, self._min_sigma
        mean_d = sum(diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        sigma_monthly = math.sqrt(var)
        sigma = sigma_monthly * math.sqrt(12)
        return mu, max(sigma, self._min_sigma)
