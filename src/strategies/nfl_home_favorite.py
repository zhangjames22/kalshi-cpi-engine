"""NFL home-favorite strategy — fixed-prior P(home wins).

Deliberately the simplest possible binary-market strategy. Each tick the
strategy emits one DiscreteForecast per active NFL game with p_yes set to
a constant (the league-wide home-field advantage prior, ~0.55).

The point is not accuracy — it's to exercise the platform's binary-market
path:

  - DiscreteForecast (vs. CPI's DistributionForecast)
  - BinaryMarket / no BucketLadder
  - Binary settlement via the engine's `event_market_ids = [event_id]` fallback

Strategies that want to be smarter (closing-line model, ELO, neural net)
plug into the same surface.
"""

from __future__ import annotations

from typing import Iterable

from core.forecast import DiscreteForecast
from core.market import MarketCatalog, MarketId, SeriesId
from core.state import MarketState, PortfolioView
from core.strategy import FeatureView, Strategy, StrategyOutput


class NflHomeFavoriteStrategy(Strategy):
    """Forecast P(home wins) = `home_advantage` for every active game."""

    def __init__(
        self,
        series_id: SeriesId,
        home_advantage: float = 0.55,
    ) -> None:
        if not 0.0 < home_advantage < 1.0:
            raise ValueError(
                f"home_advantage must be in (0, 1); got {home_advantage}"
            )
        self._series_id = series_id
        self._home_advantage = home_advantage

    @property
    def name(self) -> str:
        return "nfl_home_favorite"

    def universe(self, t, catalog: MarketCatalog) -> Iterable[MarketId]:
        return [m.market_id for m in catalog.markets_for_series(self._series_id, t)]

    def on_tick(
        self,
        t,
        state: MarketState,
        features: FeatureView,
        portfolio: PortfolioView,
    ) -> StrategyOutput:
        if len(state) == 0:
            return StrategyOutput.empty()

        # One DiscreteForecast per active market — each binary game gets
        # the same fixed-prior p_yes. Combined into a single DiscreteForecast
        # (the dataclass accepts arbitrary {market_id: p_yes} mappings).
        probs = {mid: self._home_advantage for mid in state.market_ids()}
        return StrategyOutput(forecasts=(DiscreteForecast(probabilities=probs),))
