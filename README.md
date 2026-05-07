# kalshi-cpi-engine

A strategy management and observability platform for prediction-market quants.
Forecasts get scored on Brier and log-loss, calibration is binned at 10 levels,
and per-strategy P&L is attributed cleanly across multi-strategy runs. The
project is in early development and ships with two reference strategies that
exercise the platform's two market types — `CpiNormalStrategy` (macro,
distribution forecast over a bucketed CPI YoY ladder) and
`NflHomeFavoriteStrategy` (sports, discrete forecast on a binary moneyline).
Both reference strategies are deliberately simple and exist to validate the
abstractions; neither is a production trading system.

## Install

```bash
pip install -e ".[dev]"
pytest
```

You should see ~131 tests pass in under 10 seconds. Two of them
(`test_historical_cpi_benchmark.py`, `test_historical_nfl_benchmark.py`) re-run
the reference backtests against the committed fixtures, so any regression in
scoring or settlement plumbing fails the suite.

## Two reference strategies

Both strategies emit forecasts only — no orders. The framework still scores
them via Brier, log-loss, and a 10-bin calibration curve.

### CpiNormalStrategy — macro, distribution forecast

- Code: [src/strategies/cpi_normal.py](src/strategies/cpi_normal.py)
- Fixture: [tests/fixtures/historical_cpi/](tests/fixtures/historical_cpi/) (32 markets across 2 settled CPI YoY events, 2026-02 and 2026-03)
- Run: [scripts/run_cpi_historical_backtest.py](scripts/run_cpi_historical_backtest.py)

Fits Normal(μ, σ) to the historical YoY series, projects the CDF onto each
event's bucket ladder, emits one `DistributionForecast` per event per tick.

| | CpiNormal | Market (orderbook) |
| --- | --- | --- |
| Brier | **0.0618** | **0.0592** |
| Log-loss | 0.249 | – |
| n_scored | 90,240 forecasts across 2 events | 2,640 fully-quoted snapshots |

The strategy is **slightly worse than the market on average**, but beats the
market on the 26MAR event where the closing book priced the actual winner at
~0.5%. Two events is not statistically meaningful — the takeaway is that the
plumbing works end-to-end on real Kalshi candlestick data and real FRED
settlements.

### NflHomeFavoriteStrategy — sports, discrete forecast

- Code: [src/strategies/nfl_home_favorite.py](src/strategies/nfl_home_favorite.py)
- Fixture: [tests/fixtures/historical_nfl/](tests/fixtures/historical_nfl/) (570 resolved games, 2024 + 2025 NFL seasons)
- Run: [scripts/run_nfl_historical_backtest.py](scripts/run_nfl_historical_backtest.py)

Emits a fixed `DiscreteForecast` of P(home wins) = 0.55 for every active
game. Exists to exercise the binary-market path; no real predictive content.

| | NflHomeFavorite (p=0.55) | Closing-line market |
| --- | --- | --- |
| Brier | **0.2497** | **0.2057** |
| Log-loss | 0.6925 | – |
| Accuracy @ 0.5 threshold | 54.04% (= league HFA) | 68.25% |

The constant-prior strategy is **well-calibrated for the one claim it makes**
(empirical home-win rate 54.04%, forecast 0.55) but unsurprisingly lags the
sportsbook closing line. Calibration data lands entirely in the `[0.5, 0.6)`
bin, exactly where it should.

Neither reference strategy beats its market baseline — that's the point.
The contribution here is the platform: that you can swap from a 16-bucket
macro ladder to a 1-market binary moneyline by writing one strategy file
and one data adapter, with zero changes to `src/core/` or `src/backtest/`.

## Architecture

The platform separates four concerns: market structure (`core/market.py`),
forecast types (`core/forecast.py`), strategy lifecycle (`core/strategy.py`),
and tick-driven execution (`backtest/engine.py`). A `Strategy` returns
`StrategyOutput(forecasts, orders, cancels)` each tick; the engine scores
forecasts against an `Outcome` returned by a `SettlementLoader` and routes
orders through an `ExecutionModel`. Two market types — `BucketMarket` (numeric
ladder) and `BinaryMarket` (yes/no) — cover both reference strategies without
shared abstractions leaking between them.

Full design notes in [docs/concepts.md](docs/concepts.md).

## Repository layout

```
kalshi-cpi-engine/
├── src/
│   ├── core/                         # Strategy ABC, market types, forecasts, state
│   │   ├── market.py                 # BinaryMarket, BucketMarket, BucketLadder, Outcome
│   │   ├── forecast.py               # DiscreteForecast, DistributionForecast
│   │   ├── strategy.py               # Strategy ABC, FeatureView Protocol
│   │   ├── state.py                  # OrderbookSnapshot, MarketState, Portfolio
│   │   └── order.py                  # Order, CancelOrder, Side, TimeInForce
│   ├── backtest/                     # Tick-driven engine, execution models, report
│   │   ├── engine.py                 # BacktestEngine
│   │   ├── execution.py              # MidFill, CrossSpreadFill
│   │   └── report.py                 # BacktestReport, Brier, log-loss, calibration
│   ├── data/                         # Per-source loaders. No platform code lives here.
│   │   ├── kalshi/loader.py          # Kalshi REST: catalog, candlestick → snapshots
│   │   ├── features/fred.py          # FRED feature view + CPI settlement loader
│   │   └── sports/nfl.py             # nflverse: NFL catalog + settlement
│   └── strategies/
│       ├── cpi_normal.py             # Reference: macro distribution forecast
│       └── nfl_home_favorite.py      # Reference: binary fixed-prior forecast
├── scripts/                          # Fixture fetchers + reproducible backtest runners
├── tests/
│   ├── fixtures/historical_cpi/      # 32 markets × ~95 days of daily candlesticks
│   └── fixtures/historical_nfl/      # 570 resolved games, 2024–25 seasons
├── docs/concepts.md                  # Architecture and design rationale
└── pyproject.toml
```

## Status

Early development. Stable: market types, forecast scoring, the backtest
engine, and the two reference data adapters (Kalshi candlesticks + FRED;
nflverse). Not yet built: live trading, multi-strategy capital allocation,
deeper-than-top-of-book execution. Issues and PRs welcome.
