"""CPI-Normal strategy — the existing notebook 03 logic packaged as a Strategy.

Each tick:
  1. Look up the latest core_cpi_yoy time series available at engine time t.
  2. Fit Normal(mu, sigma) where:
        mu    = latest YoY value (in percent units)
        sigma = std-dev of monthly YoY changes, annualized via sqrt(12),
                clamped to a minimum so the bucket probabilities don't
                degenerate to a delta function.
  3. For each event in the strategy's universe, emit one DistributionForecast
     keyed to that event_id. The engine projects the CDF onto the event's
     bucket ladder.

No orders are emitted in v1. The strategy is purely a research/forecast
publisher; calibration and Brier are scored from the forecast ledger when
events resolve.
"""

from __future__ import annotations

import math
from typing import Iterable

from core.forecast import DistributionForecast, normal_cdf
from core.market import MarketCatalog, MarketId, SeriesId
from core.state import MarketState, PortfolioView
from core.strategy import FeatureView, Strategy, StrategyOutput


class CpiNormalStrategy(Strategy):
    """Forecast Kalshi CPI YoY buckets via a fitted Normal distribution."""

    def __init__(
        self,
        series_id: SeriesId,
        feature_name: str = "core_cpi_yoy",
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

        history = features.get_series(self._feature_name, t)
        if not history or len(history) < 2:
            return StrategyOutput.empty()

        mu, sigma = self._fit_normal(history)

        # Distinct events present in the universe.
        events = sorted({state.market(mid).event_id for mid in state.market_ids()})

        forecasts = tuple(
            DistributionForecast(event_id=eid, cdf=normal_cdf(mu, sigma))
            for eid in events
        )
        return StrategyOutput(forecasts=forecasts)

    # ------------------------------------------------------------------
    # Internal — fit Normal(mu, sigma) over the provided history.
    # ------------------------------------------------------------------

    def _fit_normal(self, history: list[float]) -> tuple[float, float]:
        # Convert to percent units if the history looks decimal-scale (e.g.
        # 0.028 -> 2.8). Uses median magnitude so a few outliers don't flip
        # the unit detection.
        sorted_h = sorted(abs(x) for x in history)
        median = sorted_h[len(sorted_h) // 2]
        scale = 100.0 if median < 1.0 else 1.0
        rescaled = [x * scale for x in history]

        mu = rescaled[-1]

        diffs = [b - a for a, b in zip(rescaled, rescaled[1:])]
        if not diffs:
            return mu, self._min_sigma
        mean_d = sum(diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        sigma_monthly = math.sqrt(var)
        sigma = sigma_monthly * math.sqrt(12)
        return mu, max(sigma, self._min_sigma)
