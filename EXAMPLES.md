# Examples

Two copy-paste snippets showing the shape of a backtest run. Each loads a
committed fixture, instantiates one of the reference strategies, and runs
it through `BacktestEngine`. The full versions — with reporting, market
baselines, and per-event attribution — live in
[scripts/run_cpi_historical_backtest.py](scripts/run_cpi_historical_backtest.py)
and [scripts/run_nfl_historical_backtest.py](scripts/run_nfl_historical_backtest.py).

## CpiNormalStrategy — bucketed macro forecast

Forecast each Kalshi CPI YoY event with a Normal(μ, σ) fit to the FRED
CPILFESL series, projected onto the event's bucket ladder.

```python
import pandas as pd
from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features.fred import FredCpiSettlementLoader, FredFeatureView
from data.kalshi.loader import iter_orderbook_snapshots, load_catalog_from_parquet
from strategies.cpi_normal import CpiNormalStrategy

FIXTURE = "tests/fixtures/historical_cpi"
catalog = load_catalog_from_parquet(f"{FIXTURE}/markets.parquet")
panel   = pd.read_parquet(f"{FIXTURE}/fred_core_cpi.parquet")
snaps   = list(iter_orderbook_snapshots(pd.read_parquet(f"{FIXTURE}/snapshots.parquet")))

engine = BacktestEngine(
    catalog=catalog,
    settlement=FredCpiSettlementLoader(catalog=catalog, panel=panel),
    snapshot_stream=snaps,
    feature_view=FredFeatureView(panel=panel),
    execution=CrossSpreadFill(),
    config=BacktestConfig(initial_cash=10_000.0),
)
engine.add_strategy(CpiNormalStrategy(series_id="KXECONSTATCPIYOY"))
report = engine.run()
print(f"Brier={report.metrics['brier']:.4f}  n_scored={report.metrics['n_scored_forecasts']}")
```

## NflHomeFavoriteStrategy — binary moneyline forecast

Emit `P(home wins) = 0.55` for every active NFL game. Same engine, same
report shape — different market type, different feature backend.

```python
import pandas as pd
from backtest import BacktestConfig, BacktestEngine, CrossSpreadFill
from data.features import NullFeatureView
from data.sports.nfl import (
    NflSettlementLoader, iter_game_snapshots, load_catalog_from_parquet,
)
from strategies.nfl_home_favorite import NflHomeFavoriteStrategy

FIXTURE = "tests/fixtures/historical_nfl"
games   = pd.read_parquet(f"{FIXTURE}/games.parquet")
catalog = load_catalog_from_parquet(f"{FIXTURE}/games.parquet")
snaps   = list(iter_game_snapshots(pd.read_parquet(f"{FIXTURE}/snapshots.parquet")))

engine = BacktestEngine(
    catalog=catalog,
    settlement=NflSettlementLoader(games_df=games),
    snapshot_stream=snaps,
    feature_view=NullFeatureView(),
    execution=CrossSpreadFill(),
    config=BacktestConfig(initial_cash=10_000.0),
)
engine.add_strategy(NflHomeFavoriteStrategy(series_id="KXNFLGAME"))
report = engine.run()
print(f"Brier={report.metrics['brier']:.4f}  n_scored={report.metrics['n_scored_forecasts']}")
```

## What changes between strategies

The diff is small and load-bearing:

| | CPI (bucket ladder) | NFL (binary) |
| --- | --- | --- |
| Catalog | `data.kalshi.loader.load_catalog_from_parquet` | `data.sports.nfl.load_catalog_from_parquet` |
| Settlement | `FredCpiSettlementLoader` | `NflSettlementLoader` |
| Features | `FredFeatureView` | `NullFeatureView` |
| Strategy | `CpiNormalStrategy` (DistributionForecast) | `NflHomeFavoriteStrategy` (DiscreteForecast) |
| Snapshot iterator | `iter_orderbook_snapshots` | `iter_game_snapshots` |

`BacktestEngine`, `CrossSpreadFill`, and `BacktestConfig` are identical in
both cases, and the report keys (`brier`, `log_loss`, `calibration`,
`per_strategy`) match. Writing a new strategy means writing a new
`Strategy` subclass plus a data adapter under `src/data/...`; nothing in
`src/core/` or `src/backtest/` needs to change.
