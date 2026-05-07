"""Tests for core.forecast — discrete and distribution forecasts."""

from __future__ import annotations

import math

import pytest

from core.forecast import (
    DiscreteForecast,
    DistributionForecast,
    empirical_cdf,
    normal_cdf,
)
from tests._fixtures import make_cpi_like_ladder


# ---------------------------------------------------------------------------
# DiscreteForecast
# ---------------------------------------------------------------------------

def test_discrete_forecast_round_trips_probabilities():
    f = DiscreteForecast(probabilities={"M1": 0.6, "M2": 0.2})
    out = f.market_probabilities(ladder_lookup=lambda eid: None)
    assert out == {"M1": 0.6, "M2": 0.2}


def test_discrete_forecast_rejects_out_of_range_prob():
    with pytest.raises(ValueError, match="must be in"):
        DiscreteForecast(probabilities={"M1": 1.2})
    with pytest.raises(ValueError, match="must be in"):
        DiscreteForecast(probabilities={"M1": -0.01})


def test_discrete_forecast_allows_endpoints():
    DiscreteForecast(probabilities={"M1": 0.0, "M2": 1.0})  # no error


def test_discrete_forecast_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        DiscreteForecast(probabilities={})


def test_discrete_forecast_allows_non_summing_probs():
    """Different DiscreteForecasts can target different events; we don't
    enforce that probs sum to 1."""
    DiscreteForecast(probabilities={"M1": 0.3, "M2": 0.3})  # sums to 0.6, fine


# ---------------------------------------------------------------------------
# DistributionForecast
# ---------------------------------------------------------------------------

def test_distribution_forecast_projects_via_ladder():
    ladder = make_cpi_like_ladder([2.0, 3.0, 4.0])
    f = DistributionForecast(event_id=ladder.event_id, cdf=normal_cdf(3.0, 0.5))

    probs = f.market_probabilities(
        ladder_lookup=lambda eid: ladder if eid == ladder.event_id else None
    )

    assert pytest.approx(sum(probs.values()), abs=1e-10) == 1.0
    assert set(probs.keys()) == {b.market_id for b in ladder.buckets}


def test_distribution_forecast_raises_when_ladder_missing():
    f = DistributionForecast(event_id="UNKNOWN", cdf=normal_cdf(0, 1))
    with pytest.raises(ValueError, match="no .*BucketLadder"):
        f.market_probabilities(ladder_lookup=lambda eid: None)


# ---------------------------------------------------------------------------
# normal_cdf
# ---------------------------------------------------------------------------

def test_normal_cdf_is_monotone_and_bounded():
    cdf = normal_cdf(mu=0.0, sigma=1.0)
    xs = [-5.0, -1.0, 0.0, 1.0, 5.0]
    vals = [cdf(x) for x in xs]
    # bounded
    assert all(0.0 <= v <= 1.0 for v in vals)
    # monotone
    assert vals == sorted(vals)
    # symmetric around mu
    assert pytest.approx(cdf(0.0)) == 0.5
    assert pytest.approx(cdf(-1.0) + cdf(1.0), abs=1e-12) == 1.0


def test_normal_cdf_tails_approach_extremes():
    cdf = normal_cdf(0.0, 1.0)
    assert cdf(-100.0) < 1e-12
    assert cdf(100.0) > 1.0 - 1e-12


def test_normal_cdf_rejects_zero_or_negative_sigma():
    with pytest.raises(ValueError, match="sigma must be positive"):
        normal_cdf(0.0, 0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        normal_cdf(0.0, -1.0)


# ---------------------------------------------------------------------------
# empirical_cdf
# ---------------------------------------------------------------------------

def test_empirical_cdf_step_function():
    cdf = empirical_cdf([1.0, 2.0, 3.0, 4.0])
    assert cdf(0.0) == 0.0
    assert cdf(1.0) == 0.25
    assert cdf(2.5) == 0.5
    assert cdf(4.0) == 1.0
    assert cdf(100.0) == 1.0


def test_empirical_cdf_rejects_empty():
    with pytest.raises(ValueError):
        empirical_cdf([])


def test_empirical_cdf_handles_duplicates():
    cdf = empirical_cdf([1.0, 1.0, 1.0, 2.0])
    assert cdf(1.0) == 0.75  # three values <= 1.0
    assert cdf(0.5) == 0.0
    assert cdf(2.0) == 1.0


def test_empirical_cdf_in_distribution_forecast():
    ladder = make_cpi_like_ladder([2.0, 3.0, 4.0])
    f = DistributionForecast(event_id=ladder.event_id, cdf=empirical_cdf([3.5] * 100))
    probs = f.market_probabilities(lambda eid: ladder)
    # All mass at 3.5 → bucket [3.0, 4.0) gets ~1.0
    assert pytest.approx(probs[ladder.buckets[2].market_id]) == 1.0
