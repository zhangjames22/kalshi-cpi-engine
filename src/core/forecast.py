"""Forecasts — the probability statements a Strategy is willing to be scored on.

Two flavors, mirroring the two market types:

- DiscreteForecast: direct {market_id: p_yes} assignment. Use for binary markets
  (politics, sports moneylines), and for any strategy that wants to bypass
  distribution-fitting on a bucket ladder and just hand-set probabilities.

- DistributionForecast: a continuous CDF over an event's underlying value.
  The engine resolves this against the event's BucketLadder at scoring time
  via `BucketLadder.project(cdf)`, producing one p_yes per bucket. This is
  exactly the path the CPI engine takes today.

Forecasts are independent of orders. A pure-research strategy can publish
forecasts every tick and emit zero orders; the backtest report still scores
its calibration via Brier / log-loss.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

from .market import BucketLadder, EventId, MarketId


# Resolves an event_id to its BucketLadder, or None for binary events.
# Supplied by the engine from the active MarketCatalog.
LadderLookup = Callable[[EventId], "BucketLadder | None"]


class Forecast(ABC):
    """A probability statement keyed to one or more markets."""

    @abstractmethod
    def market_probabilities(
        self, ladder_lookup: LadderLookup
    ) -> dict[MarketId, float]:
        """Resolve to {market_id: p_yes} for scoring and order routing.

        The engine supplies `ladder_lookup`. DiscreteForecast ignores it;
        DistributionForecast needs it to know which ladder to project onto.
        """


@dataclass(frozen=True)
class DiscreteForecast(Forecast):
    """Direct per-market p_yes assignment.

    Validates that probabilities are in [0, 1] at construction. Does NOT
    enforce that probabilities sum to 1 across markets — different forecasts
    can target different events, and binary events have only one market.
    """
    probabilities: Mapping[MarketId, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("DiscreteForecast must have at least one probability")
        for mid, p in self.probabilities.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"DiscreteForecast: p_yes for {mid} must be in [0,1], got {p}"
                )

    def market_probabilities(
        self, ladder_lookup: LadderLookup
    ) -> dict[MarketId, float]:
        return dict(self.probabilities)


@dataclass(frozen=True)
class DistributionForecast(Forecast):
    """A continuous distribution over an event's underlying value, projected
    onto the event's BucketLadder at scoring time.

    `cdf` must be a monotone non-decreasing function R -> [0, 1] with
    cdf(-inf) = 0 and cdf(+inf) = 1. The framework treats it opaquely;
    convenience constructors like `normal_cdf` are provided below for the
    common shapes.
    """
    event_id: EventId
    cdf: Callable[[float], float]

    def market_probabilities(
        self, ladder_lookup: LadderLookup
    ) -> dict[MarketId, float]:
        ladder = ladder_lookup(self.event_id)
        if ladder is None:
            raise ValueError(
                f"DistributionForecast targets event {self.event_id} but no "
                f"BucketLadder is registered for it. Use DiscreteForecast for "
                f"binary events."
            )
        return ladder.project(self.cdf)


# ---------------------------------------------------------------------------
# Convenience CDF constructors.
# ---------------------------------------------------------------------------

def normal_cdf(mu: float, sigma: float) -> Callable[[float], float]:
    """Build a Normal(mu, sigma) CDF as a plain Python callable.

    Pure stdlib (math.erf); no scipy dependency. The CPI strategy uses this.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    sqrt2 = math.sqrt(2.0)
    return lambda x: 0.5 * (1.0 + math.erf((x - mu) / (sigma * sqrt2)))


def empirical_cdf(samples: "list[float] | tuple[float, ...]") -> Callable[[float], float]:
    """Step-function CDF over an empirical sample.

    Useful for monte-carlo or bootstrap-based strategies that don't want
    to commit to a parametric family.
    """
    if not samples:
        raise ValueError("empirical_cdf requires at least one sample")
    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    def cdf(x: float) -> float:
        # Count of samples <= x, divided by n. Linear scan; for hot paths
        # callers can swap in a bisect-based implementation.
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_samples[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / n

    return cdf
