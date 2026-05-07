"""Sports-data adapters.

Mirrors the structure of `data/kalshi/` (Kalshi REST loaders) and
`data/features/` (FRED feature views): one module per source. Strategies
import the loader they need; the rest of the platform stays sport-agnostic.
"""

from .nfl import (
    NflCatalog,
    NflSettlementLoader,
    iter_game_snapshots,
    load_catalog_from_dataframe,
    load_catalog_from_parquet,
    moneyline_to_prob,
)

__all__ = [
    "NflCatalog",
    "NflSettlementLoader",
    "iter_game_snapshots",
    "load_catalog_from_dataframe",
    "load_catalog_from_parquet",
    "moneyline_to_prob",
]
