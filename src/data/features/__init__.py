"""Feature backends. FRED implementation lives in `fred.py`; the no-op
NullFeatureView is in `null.py` for strategies that don't read features."""

from .fred import (
    DEFAULT_PUBLICATION_LAG,
    FRED_SERIES_META,
    FredClient,
    FredCpiSettlementLoader,
    FredFeatureView,
    fetch_macro_panel,
    load_macro_panel_from_parquet,
)
from .null import NullFeatureView

__all__ = [
    "DEFAULT_PUBLICATION_LAG",
    "FRED_SERIES_META",
    "FredClient",
    "FredCpiSettlementLoader",
    "FredFeatureView",
    "NullFeatureView",
    "fetch_macro_panel",
    "load_macro_panel_from_parquet",
]
