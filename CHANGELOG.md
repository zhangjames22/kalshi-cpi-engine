# Changelog

Phase-by-phase log of what's shipped. Dates are approximate; commits in
`git log` are authoritative.

## [Unreleased]

### Phase 3.5 — Architectural polish (2026-05)
- `core.market.BinaryResolution` enum and an optional
  `Outcome.binary_resolution` field replace ad-hoc string sentinels
  (`f"{event_id}#NO"`) for binary-NO outcomes.
- `data.features.NullFeatureView` ships as the canonical no-op
  `FeatureView` for strategies that don't read features.
- `PortfolioView` hardened with `__slots__` and read-only properties;
  strategies cannot mutate or rebind the underlying portfolio.
- 14 new tests across the three surfaces; suite at 145 passing.
- README rewritten as a single-page landing page; `EXAMPLES.md` adds
  copy-paste backtest wiring for both reference strategies.

### Phase 3 — NFL cross-vertical proof (2026-05)
- `data.sports.nfl` adapter (binary `BinaryMarket` catalog +
  `NflSettlementLoader` + `iter_game_snapshots`) and
  `strategies.nfl_home_favorite.NflHomeFavoriteStrategy` exercise the
  binary-market path end-to-end with zero changes to `core/`,
  `backtest/`, `data/features/`, or `data/kalshi/loader.py`.
- nflverse fixture committed: 570 resolved games across the 2024 + 2025
  NFL seasons. Reference Brier 0.2497 vs. closing-line market 0.2057.

### Phase 2.5 — Historical CPI backtest (2026-05)
- `scripts/fetch_historical_cpi.py` walks Kalshi's candlesticks endpoint
  for every settled `KXECONSTATCPIYOY` market and writes daily snapshots
  + a multi-year FRED panel into `tests/fixtures/historical_cpi/`.
- `CpiNormalStrategy` now produces a real `BacktestReport` with non-zero
  `n_scored`. Reference Brier 0.0618 vs. orderbook market 0.0592 across
  90,240 scored forecasts on n=2 events.

### Phase 2 — Real Kalshi data (2026-05)
- `data.kalshi.loader.KalshiClient` with `/markets` and `/candlesticks`
  REST endpoints; `KalshiCatalog` builders from API and parquet inputs.
- `data.features.fred.FredClient`, `FredFeatureView` (point-in-time,
  publication-lag aware), and `FredCpiSettlementLoader` reconstruct CPI
  YoY outcomes from raw FRED observations.

### Phase 1 — Engine (2026-04)
- `core.market` (BinaryMarket, BucketMarket, BucketLadder),
  `core.forecast` (DiscreteForecast, DistributionForecast),
  `core.strategy` (Strategy ABC, FeatureView Protocol), `core.state`
  (OrderbookSnapshot, MarketState, Portfolio, PortfolioView),
  `core.order` (Order, CancelOrder, Side, TimeInForce).
- `BacktestEngine` ticks per-snapshot, scores forecasts on Brier and
  log-loss, routes orders through pluggable `ExecutionModel`s, settles
  events from a `SettlementLoader`. `BacktestReport` writes tidy parquet
  ledgers + a calibration curve.
- 76 tests at the end of Phase 1; src-layout packaging via
  `pyproject.toml`.
