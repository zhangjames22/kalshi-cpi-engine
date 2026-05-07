"""Kalshi data layer — historical loader implementing core protocols."""

from .loader import (
    EVENT_CODE_REGEX,
    KalshiCatalog,
    KalshiClient,
    build_buckets,
    iter_orderbook_snapshots,
    load_catalog_from_dataframe,
    load_catalog_from_parquet,
    normalize_market_probs,
    parse_custom_strike,
)

__all__ = [
    "EVENT_CODE_REGEX",
    "KalshiCatalog",
    "KalshiClient",
    "build_buckets",
    "iter_orderbook_snapshots",
    "load_catalog_from_dataframe",
    "load_catalog_from_parquet",
    "normalize_market_probs",
    "parse_custom_strike",
]
