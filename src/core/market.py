"""Market data model.

Two flavors of Kalshi market in v1:

- BinaryMarket: standalone yes/no contract (politics, sports moneyline).
  One Market == one Event. Strategies output a single p_yes per market.

- BucketMarket: one rung of a numeric-strike ladder (e.g. CPI YoY in [3.0, 3.1)).
  Many Markets share an Event, organized as a BucketLadder. Strategies
  typically forecast a distribution over the underlying continuous variable;
  BucketLadder.project turns that into a per-bucket probability.

Settlement
----------
Outcome captures the resolved truth for an event. Provided by a SettlementLoader.
For CPI we reconstruct outcomes from FRED. For everything else we cache Kalshi's
expiration_value at data/kalshi/settlements/<series>.parquet with schema:

    series_ticker:         str
    event_ticker:          str
    expiration_value:      float | None   (None for pure binary events)
    winning_market_ticker: str            (the market that resolved YES)
    resolved_at_utc:       datetime (UTC)

The cache fetcher is a separate task; SettlementLoader is the read-side
interface that hides which source supplied the outcome.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Protocol


# ---------------------------------------------------------------------------
# Type aliases — Kalshi's ticker hierarchy.
# ---------------------------------------------------------------------------

MarketId = str   # e.g. "KXECONSTATCPIYOY-26NOV-T3.5"
EventId = str    # e.g. "KXECONSTATCPIYOY-26NOV"
SeriesId = str   # e.g. "KXECONSTATCPIYOY"


# ---------------------------------------------------------------------------
# Markets.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Market:
    """Base type for any tradeable Kalshi market.

    Subclasses specialize the resolution semantics. Always immutable; the
    catalog is the single source of truth, strategies should not mutate.
    """
    market_id: MarketId
    event_id: EventId
    series_id: SeriesId
    open_time: datetime
    close_time: datetime
    expiration_time: datetime
    title: str = ""


@dataclass(frozen=True, kw_only=True)
class BinaryMarket(Market):
    """A standalone yes/no market. Resolves YES (=$1) or NO (=$0).

    Used for politics, sports moneylines, and any single-question event.
    A BinaryMarket is its own event — there is no ladder to project onto.
    """


@dataclass(frozen=True, kw_only=True)
class BucketMarket(Market):
    """One rung of a numeric-strike ladder.

    Resolves YES iff the realized underlying value v satisfies
    floor <= v < cap. Use math.inf / -math.inf for the open ends of
    the bottom and top rungs respectively.

    `strike_value` is the raw quoted strike from Kalshi (the customStrike
    field). It will equal either floor or cap depending on the series'
    convention — it's preserved for joining back to source data, not for
    range math; range math always uses (floor, cap).
    """
    floor: float
    cap: float
    strike_value: float

    def __post_init__(self) -> None:
        if not self.floor < self.cap:
            raise ValueError(
                f"BucketMarket {self.market_id}: floor ({self.floor}) "
                f"must be strictly less than cap ({self.cap})"
            )

    def contains(self, value: float) -> bool:
        """True iff `value` falls in this bucket's half-open range."""
        return self.floor <= value < self.cap


# ---------------------------------------------------------------------------
# Events and ladders.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Event:
    """A Kalshi event — a question that resolves to one outcome.

    Has 1 BinaryMarket OR N BucketMarkets covering (-inf, +inf).
    """
    event_id: EventId
    series_id: SeriesId
    close_time: datetime
    expiration_time: datetime
    title: str = ""


@dataclass(frozen=True)
class BucketLadder:
    """An ordered, contiguous cover of (-inf, +inf) by BucketMarkets.

    Invariants (enforced at construction):
      - At least one bucket.
      - All buckets share the same event_id.
      - First floor == -inf, last cap == +inf.
      - Buckets are contiguous: bucket[i].cap == bucket[i+1].floor.
    """
    event_id: EventId
    buckets: tuple[BucketMarket, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("BucketLadder must contain at least one bucket")

        for b in self.buckets:
            if b.event_id != self.event_id:
                raise ValueError(
                    f"Bucket {b.market_id} belongs to event {b.event_id}, "
                    f"expected {self.event_id}"
                )

        for prev, nxt in zip(self.buckets, self.buckets[1:]):
            if prev.cap != nxt.floor:
                raise ValueError(
                    f"Ladder is not contiguous: {prev.market_id}.cap={prev.cap} "
                    f"!= {nxt.market_id}.floor={nxt.floor}"
                )

        if self.buckets[0].floor != -math.inf:
            raise ValueError(
                f"First bucket floor must be -inf, got {self.buckets[0].floor}"
            )
        if self.buckets[-1].cap != math.inf:
            raise ValueError(
                f"Last bucket cap must be +inf, got {self.buckets[-1].cap}"
            )

    def project(self, cdf: Callable[[float], float]) -> dict[MarketId, float]:
        """Project a continuous distribution onto this ladder.

        `cdf` must be a proper, monotone non-decreasing CDF on the reals.
        Returns {market_id: P(value in [floor, cap))}.

        The framework does NOT renormalize. A proper CDF integrates to 1
        across the ladder (since the ladder covers all of R); deviation
        from 1 indicates a malformed CDF and the caller should see it.
        """
        out: dict[MarketId, float] = {}
        for b in self.buckets:
            lo = 0.0 if b.floor == -math.inf else cdf(b.floor)
            hi = 1.0 if b.cap == math.inf else cdf(b.cap)
            out[b.market_id] = max(0.0, hi - lo)
        return out

    def winning_market(self, value: float) -> MarketId:
        """Return the market_id of the bucket that resolves YES at `value`."""
        for b in self.buckets:
            if b.contains(value):
                return b.market_id
        # Unreachable for a valid ladder: contains() handles all of R because
        # floor=-inf and cap=+inf at the ends. Defensive raise anyway.
        raise ValueError(f"value {value} fell outside ladder bounds")

    def __len__(self) -> int:
        return len(self.buckets)

    def __iter__(self):
        return iter(self.buckets)


# ---------------------------------------------------------------------------
# Outcomes.
# ---------------------------------------------------------------------------

class BinaryResolution(enum.Enum):
    """How a BinaryMarket event resolved. Only meaningful for binary events;
    ladder events leave `Outcome.binary_resolution` set to None."""
    YES = "yes"
    NO = "no"


@dataclass(frozen=True, kw_only=True)
class Outcome:
    """The resolved truth for an event.

    Three valid shapes:

    1. **Ladder event** — `binary_resolution=None`, `winning_market_id` is
       the bucket whose [floor, cap) contains the realized
       `expiration_value`.
    2. **Binary YES** — `binary_resolution=BinaryResolution.YES`,
       `winning_market_id` is the binary market itself. YES contracts pay $1.
    3. **Binary NO**  — `binary_resolution=BinaryResolution.NO`,
       `winning_market_id=None`. NO contracts pay $1; YES contracts pay $0.

    The engine's payout logic compares `market_id == winning_market_id` to
    decide whether YES contracts cash; the None case for a binary NO never
    matches any market_id, which gives NO contracts the payout by
    complementarity.
    """
    event_id: EventId
    winning_market_id: MarketId | None
    expiration_value: float | None
    resolved_at: datetime
    binary_resolution: BinaryResolution | None = None

    def __post_init__(self) -> None:
        if self.binary_resolution is BinaryResolution.NO:
            if self.winning_market_id is not None:
                raise ValueError(
                    f"Outcome for {self.event_id}: binary_resolution=NO "
                    f"requires winning_market_id=None, got "
                    f"{self.winning_market_id!r}"
                )
        else:
            # Ladder outcome OR binary YES — both require a non-None winner.
            if self.winning_market_id is None:
                raise ValueError(
                    f"Outcome for {self.event_id}: winning_market_id is "
                    f"required unless binary_resolution=NO"
                )


# ---------------------------------------------------------------------------
# Read-side protocols.
# ---------------------------------------------------------------------------

class MarketCatalog(Protocol):
    """Read-only view of all known markets/events at a point in time.

    Backtest catalogs scan historical metadata; live catalogs query Kalshi.
    Strategies use this in `universe(t)` to select markets they want to see
    orderbook snapshots for at tick t.
    """

    def markets_for_series(
        self, series_id: SeriesId, t: datetime
    ) -> Iterable[Market]:
        """All markets in `series_id` that exist at time `t` (open_time <= t,
        close_time > t)."""
        ...

    def market(self, market_id: MarketId) -> Market: ...

    def event(self, event_id: EventId) -> Event: ...

    def ladder(self, event_id: EventId) -> BucketLadder | None:
        """Return the BucketLadder for a numeric-ladder event, or None for
        binary events."""
        ...

    def events_resolving_between(
        self, t_start: datetime, t_end: datetime
    ) -> Iterable[Event]:
        """Events whose expiration_time falls in [t_start, t_end)."""
        ...


class SettlementLoader(Protocol):
    """Resolves events to outcomes.

    Implementations:
      - KalshiSettlementCache: reads data/kalshi/settlements/<series>.parquet
        (populated out-of-band by a separate fetcher task).
      - FredSettlementLoader: reconstructs CPI / macro outcomes from FRED
        revisions, used by the CPI strategy's backtest.

    Callers don't care which one; both honor this contract.
    """

    def outcome(self, event_id: EventId) -> Outcome | None:
        """Return the resolved outcome, or None if the event hasn't settled
        (or settlement data isn't available yet)."""
        ...
