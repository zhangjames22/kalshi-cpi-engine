"""Tests for core.market — the load-bearing piece of the abstraction.

Most of the framework's correctness rides on BucketLadder being a proper
contiguous cover of R, and on .project producing probabilities consistent
with the supplied CDF. These tests focus on those invariants.
"""

from __future__ import annotations

import math

import pytest

from datetime import datetime, timezone

from core.market import BinaryResolution, BucketLadder, Outcome
from tests._fixtures import make_bucket, make_cpi_like_ladder


T = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# BucketMarket
# ---------------------------------------------------------------------------

def test_bucket_contains_is_half_open():
    b = make_bucket("M", 2.0, 3.0)
    assert b.contains(2.0)        # closed on the left
    assert b.contains(2.5)
    assert not b.contains(3.0)    # open on the right
    assert not b.contains(1.999)


def test_bucket_rejects_inverted_range():
    with pytest.raises(ValueError, match="strictly less than"):
        make_bucket("M", 3.0, 2.0)


def test_bucket_rejects_zero_width():
    with pytest.raises(ValueError, match="strictly less than"):
        make_bucket("M", 2.0, 2.0)


# ---------------------------------------------------------------------------
# BucketLadder construction invariants
# ---------------------------------------------------------------------------

def test_ladder_requires_at_least_one_bucket():
    with pytest.raises(ValueError, match="at least one"):
        BucketLadder(event_id="E", buckets=())


def test_ladder_rejects_non_contiguous_buckets():
    b1 = make_bucket("A", -math.inf, 2.0)
    b2 = make_bucket("B", 2.5, math.inf)  # gap from 2.0 to 2.5
    with pytest.raises(ValueError, match="not contiguous"):
        BucketLadder(event_id="EVT-26NOV", buckets=(b1, b2))


def test_ladder_rejects_overlapping_buckets():
    b1 = make_bucket("A", -math.inf, 2.5)
    b2 = make_bucket("B", 2.0, math.inf)  # overlap [2.0, 2.5)
    with pytest.raises(ValueError, match="not contiguous"):
        BucketLadder(event_id="EVT-26NOV", buckets=(b1, b2))


def test_ladder_requires_neg_inf_floor():
    b1 = make_bucket("A", 0.0, 2.0)  # bounded floor
    b2 = make_bucket("B", 2.0, math.inf)
    with pytest.raises(ValueError, match="floor must be -inf"):
        BucketLadder(event_id="EVT-26NOV", buckets=(b1, b2))


def test_ladder_requires_pos_inf_cap():
    b1 = make_bucket("A", -math.inf, 2.0)
    b2 = make_bucket("B", 2.0, 5.0)  # bounded cap
    with pytest.raises(ValueError, match="cap must be \\+inf"):
        BucketLadder(event_id="EVT-26NOV", buckets=(b1, b2))


def test_ladder_rejects_mixed_event_ids():
    b1 = make_bucket("A", -math.inf, 2.0, event_id="EVT1")
    b2 = make_bucket("B", 2.0, math.inf, event_id="EVT2")
    with pytest.raises(ValueError, match="belongs to event"):
        BucketLadder(event_id="EVT1", buckets=(b1, b2))


# ---------------------------------------------------------------------------
# BucketLadder.project — the CPI-shaped path
# ---------------------------------------------------------------------------

def test_project_uniform_cdf_partitions_unit_interval():
    """U(0,1) CDF projected onto a ladder with edges at 0.25, 0.5, 0.75
    should yield [0.25, 0.25, 0.25, 0.25] (plus zeros below 0 and above 1
    for the unbounded ends — the interior buckets are still 0.25 each)."""
    ladder = make_cpi_like_ladder([0.0, 0.25, 0.5, 0.75, 1.0])

    def uniform_cdf(x: float) -> float:
        return max(0.0, min(1.0, x))

    probs = ladder.project(uniform_cdf)
    # 6 buckets: (-inf,0), [0,.25), [.25,.5), [.5,.75), [.75,1), [1,inf)
    assert pytest.approx(sum(probs.values()), abs=1e-12) == 1.0
    values = list(probs.values())
    assert pytest.approx(values[0]) == 0.0   # below 0
    assert pytest.approx(values[1]) == 0.25
    assert pytest.approx(values[2]) == 0.25
    assert pytest.approx(values[3]) == 0.25
    assert pytest.approx(values[4]) == 0.25
    assert pytest.approx(values[5]) == 0.0   # above 1


def test_project_normal_cdf_sums_to_one():
    from core.forecast import normal_cdf

    ladder = make_cpi_like_ladder([2.0, 2.5, 3.0, 3.5, 4.0])
    probs = ladder.project(normal_cdf(mu=3.0, sigma=0.5))
    assert pytest.approx(sum(probs.values()), abs=1e-10) == 1.0
    # Symmetry: bucket centered at mu has highest mass.
    values = list(probs.values())
    assert values[3] == max(values)  # [3.0, 3.5) is the modal bucket


def test_project_returns_one_entry_per_market():
    ladder = make_cpi_like_ladder([1.0, 2.0])
    probs = ladder.project(lambda x: max(0.0, min(1.0, (x + 5) / 10)))
    assert set(probs.keys()) == {b.market_id for b in ladder.buckets}


def test_project_handles_degenerate_cdf_at_boundary():
    """A point-mass CDF at x=2.5 should put all mass in the [2.0, 3.0)
    bucket and zero everywhere else."""
    ladder = make_cpi_like_ladder([2.0, 3.0])

    def step_cdf(x: float) -> float:
        return 1.0 if x >= 2.5 else 0.0

    probs = ladder.project(step_cdf)
    values = list(probs.values())
    assert pytest.approx(values[0]) == 0.0       # (-inf, 2.0)
    assert pytest.approx(values[1]) == 1.0       # [2.0, 3.0)
    assert pytest.approx(values[2]) == 0.0       # [3.0, +inf)


# ---------------------------------------------------------------------------
# BucketLadder.winning_market
# ---------------------------------------------------------------------------

def test_winning_market_picks_correct_bucket():
    ladder = make_cpi_like_ladder([2.0, 3.0, 4.0])
    # buckets: (-inf,2), [2,3), [3,4), [4,inf)
    assert ladder.winning_market(1.5) == ladder.buckets[0].market_id
    assert ladder.winning_market(2.0) == ladder.buckets[1].market_id  # half-open
    assert ladder.winning_market(2.99) == ladder.buckets[1].market_id
    assert ladder.winning_market(3.0) == ladder.buckets[2].market_id
    assert ladder.winning_market(99.0) == ladder.buckets[3].market_id


def test_winning_market_handles_negative_values():
    ladder = make_cpi_like_ladder([0.0, 1.0])
    assert ladder.winning_market(-1000.0) == ladder.buckets[0].market_id


# ---------------------------------------------------------------------------
# Iteration / len
# ---------------------------------------------------------------------------

def test_ladder_is_iterable_and_sized():
    ladder = make_cpi_like_ladder([1.0, 2.0, 3.0])
    assert len(ladder) == 4
    assert [b.market_id for b in ladder] == [b.market_id for b in ladder.buckets]


# ---------------------------------------------------------------------------
# Outcome — binary vs ladder shapes.
# ---------------------------------------------------------------------------

def test_outcome_ladder_requires_winning_market_id():
    with pytest.raises(ValueError, match="winning_market_id is required"):
        Outcome(
            event_id="E",
            winning_market_id=None,
            expiration_value=3.5,
            resolved_at=T,
        )


def test_outcome_binary_yes_holds_winning_market_id():
    o = Outcome(
        event_id="GAME-1",
        winning_market_id="GAME-1",
        expiration_value=None,
        resolved_at=T,
        binary_resolution=BinaryResolution.YES,
    )
    assert o.winning_market_id == "GAME-1"
    assert o.binary_resolution is BinaryResolution.YES


def test_outcome_binary_no_uses_none_winner():
    o = Outcome(
        event_id="GAME-1",
        winning_market_id=None,
        expiration_value=None,
        resolved_at=T,
        binary_resolution=BinaryResolution.NO,
    )
    assert o.winning_market_id is None
    assert o.binary_resolution is BinaryResolution.NO


def test_outcome_binary_no_rejects_non_none_winner():
    """A NO resolution can't carry a winner — that would be ambiguous; the
    engine's payout logic uses None to mean "no YES winner exists"."""
    with pytest.raises(ValueError, match="binary_resolution=NO"):
        Outcome(
            event_id="GAME-1",
            winning_market_id="GAME-1#NO",  # the old sentinel
            expiration_value=None,
            resolved_at=T,
            binary_resolution=BinaryResolution.NO,
        )


def test_outcome_binary_yes_rejects_none_winner():
    """Symmetric: a YES outcome must name the YES market."""
    with pytest.raises(ValueError, match="winning_market_id is required"):
        Outcome(
            event_id="GAME-1",
            winning_market_id=None,
            expiration_value=None,
            resolved_at=T,
            binary_resolution=BinaryResolution.YES,
        )
