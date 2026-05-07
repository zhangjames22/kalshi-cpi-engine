# Core concepts — study reference

Notes-to-self on the design choices in `src/core/`. Not user-facing docs.

---

## 1. Protocol vs ABC, and why FeatureView is a Protocol

Python has two ways to define an interface. An **ABC** uses *nominal* typing: you must `class MyThing(SomeABC)` and implement the abstract methods, otherwise instantiation raises `TypeError`. A **Protocol** uses *structural* typing: any object that happens to have the right methods satisfies the interface, no inheritance required.

`Strategy` is an ABC because we want hard enforcement — forgetting to implement `on_tick` should fail loudly at construction, and there's exactly one canonical way to write a strategy. `FeatureView`, by contrast, will have many heterogeneous backends: pandas-backed FRED loader, csv reader, in-memory dict for tests, future SQL adapter. Forcing all of those to inherit a common base couples otherwise-independent modules together and makes test fakes annoying to write. With `Protocol`, anyone who exposes `get(name, t)` and `get_series(name, t)` is automatically a `FeatureView` — no import, no inheritance. The `@runtime_checkable` decorator means `isinstance(x, FeatureView)` still works for sanity checks.

## 2. Why `kw_only=True` matters for financial dataclasses

Dataclass inheritance with defaults is a footgun: if a base class has any field with a default, a subclass can't add fields without defaults — Python raises `TypeError: non-default argument follows default argument`. `Market` has `title: str = ""`, so `BucketMarket` adding `floor: float` (no default) would explode without `kw_only=True`.

The bigger reason is positional-argument hazard. An `Order(strategy, market_id, side, action, qty, limit_price, tif)` call has seven parameters, several of them shaped similarly (two strings, two enums, two numbers). A positional swap — passing `qty` where `limit_price` belongs — silently constructs a 0.05-contract order at $10/contract instead of a 10-contract order at $0.05. With `kw_only=True` every call is `Order(strategy=..., market_id=..., qty=..., limit_price=...)`, so the wrong-slot bug becomes a `TypeError` at the call site. For anything money-shaped, optimize for "wrong code looks wrong" over "less typing."

## 3. How `BucketLadder.project` works mathematically

The ladder is a partition of the real line: buckets `B_1 = (-∞, e_1)`, `B_2 = [e_1, e_2)`, …, `B_n = [e_{n-1}, +∞)`. For a continuous random variable `X` with CDF `F`, the probability that `X` lands in bucket `[a, b)` is `F(b) - F(a)`. So `project` evaluates `F` at each interior edge, takes consecutive differences, and assigns the result to the corresponding market. Because the ladder covers all of R, the bucket probabilities sum to `F(+∞) - F(-∞) = 1` for any proper CDF — we don't renormalize, and a violation of that sum is a useful signal that the supplied CDF is malformed.

**Worked example.** Suppose CPI YoY is modeled as `Normal(μ=3.0, σ=0.5)`, and Kalshi has buckets at edges `[2.5, 3.0, 3.5]`, giving four markets with ranges `(-∞, 2.5)`, `[2.5, 3.0)`, `[3.0, 3.5)`, `[3.5, +∞)`. With `Φ` the standard normal CDF and `z = (x - 3) / 0.5`:

| Bucket | Range | Probability | Value |
|---|---|---|---|
| 1 | (-∞, 2.5) | `Φ(-1)` | 0.1587 |
| 2 | [2.5, 3.0) | `Φ(0) - Φ(-1)` | 0.3413 |
| 3 | [3.0, 3.5) | `Φ(1) - Φ(0)` | 0.3413 |
| 4 | [3.5, +∞) | `1 - Φ(1)` | 0.1587 |

Sum = 1.0; buckets adjacent to the mean carry the most mass; symmetry holds. This is exactly what notebook 03 is doing in 60 lines of inline scipy — `project` collapses it to a one-liner that any strategy can reuse.

## 4. Why `Portfolio` is mutable but everything else is frozen

Frozen dataclasses give us value semantics: two `Order(strategy="s", market_id="M1", ...)` instances with identical fields are interchangeable, hashable, safe to share across views, and trivially serializable. A frozen object is also impossible for a strategy to accidentally corrupt — when `on_tick` receives a `MarketState`, no amount of buggy strategy code can mutate the engine's view of the world.

`Portfolio` is the one exception because it's intrinsically stateful: as fills arrive, cash decreases and positions change. The pure-functional alternative is "return a new Portfolio per fill," but in a tight backtest loop with multiple strategies and dozens of fills per tick, that's a lot of allocation churn for no gain — the engine is the only writer, and the engine wants imperative semantics.

The compromise is the read/write split: the engine holds the mutable `Portfolio`, but strategies only ever see `PortfolioView`, which *is* frozen. From the strategy's perspective the contract is purely functional — `view.position(market_id)` returns the same `Position` value object you'd get from any frozen type — even though the underlying `Portfolio` is being mutated between ticks. This keeps the strategy-author API safe without paying the immutability tax in the engine's hot path.

## 5. The `SettlementLoader` contract

One method: `outcome(event_id: EventId) -> Outcome | None`. The framework treats this as the sole source of ground truth for resolved events; strategy scoring, P&L attribution, and `Strategy.on_settlement` calls all depend on it.

Return semantics:
- **`None`** means "the event has not settled yet, *or* settlement data is not available in this loader's data source." The engine treats both cases identically — the event is unresolved and no scoring/P&L happens for it.
- **`Outcome(...)`** means "this event has resolved." The fields:
  - `event_id` — echoes the query.
  - `winning_market_id` — the market that paid $1. For binary events this is the market itself iff it resolved YES; for ladder events it's the one bucket that contained the realized value.
  - `expiration_value` — the underlying numeric value for ladder events (e.g. realized CPI YoY = 3.27), or `None` for binary events.
  - `resolved_at` — when the outcome became publicly known. Used for ledger ordering, not for which-bucket-wins logic.

Implementations must be idempotent and cheap — the engine may call `outcome(event_id)` repeatedly during a backtest sweep and expects O(1)-ish lookups. Today we have two implementations in mind: `KalshiSettlementCache` reading `data/kalshi/settlements/<series>.parquet`, and `FredSettlementLoader` reconstructing CPI outcomes from FRED revisions.
