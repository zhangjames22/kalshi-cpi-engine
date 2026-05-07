"""FRED-backed FeatureView and CPI settlement loader.

`FredClient`        — thin wrapper around the FRED `series/observations` API.
                      Pure-Python; no `fredapi` dependency.
`FredFeatureView`   — implements `core.FeatureView` Protocol over a panel
                      DataFrame. Honors per-series publication lag so a
                      strategy never sees a value before the real-world
                      release date.
`FredCpiSettlementLoader` — implements `core.SettlementLoader` for Kalshi
                      CPI YoY events. Parses the event's YYMMM code, looks
                      up CPI for that month, computes YoY against the same
                      month one year prior, and finds the winning bucket
                      via the catalog's ladder.

Notebook-00 logic is ported here verbatim where the shape matches
(SERIES_META map + paginated GET). Notebook 01's CPI-specific feature
engineering (YoY/MoM/lags) is intentionally NOT moved here — that work
lives in `strategies/cpi_normal.py` per the project conventions: feature
backends serve raw observations, strategies derive what they need.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

from core.market import EventId, MarketCatalog, Outcome


# ---------------------------------------------------------------------------
# Series metadata.
# ---------------------------------------------------------------------------

#: Default macro panel — series names match notebook 00 column names so
#: cached parquet files written by the existing notebook are interchangeable
#: with panels fetched fresh through this module.
FRED_SERIES_META: dict[str, str] = {
    "core_cpi": "CPILFESL",     # Core CPI, monthly, ~14-day publication lag
    "fed_funds": "FEDFUNDS",    # Effective fed funds rate, monthly
    "unemployment": "UNRATE",   # Unemployment rate, monthly
    "oil_price": "DCOILWTICO",  # WTI oil price, daily
    "m2": "M2SL",               # M2 money supply, monthly
}

#: Conservative default. CPI is published 10–14 days after month-end; we
#: round up to 14 days so backtests don't accidentally peek.
DEFAULT_PUBLICATION_LAG: timedelta = timedelta(days=14)

#: Per-series overrides. Daily series (oil prices, fed funds effective)
#: are essentially zero-lag; monthly indicators use the default.
PUBLICATION_LAG_BY_SERIES: dict[str, timedelta] = {
    "oil_price": timedelta(days=1),
    "fed_funds": timedelta(days=2),
}


_MONTH_ABBR_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# FredClient — HTTP wrapper, paginated fetch.
# ---------------------------------------------------------------------------

class FredClient:
    """Thin wrapper around https://api.stlouisfed.org/fred.

    Construct with an explicit api_key, or rely on the FRED_API_KEY
    environment variable (loaded via dotenv at the script level).
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not set. Pass api_key= or export FRED_API_KEY."
            )
        self.session = session or requests.Session()
        self.base_url = base_url or self.BASE_URL

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "1990-01-01",
        end_date: Optional[str] = None,
        timeout: int = 30,
    ) -> pd.Series:
        """Return a pandas Series of `series_id` observations.

        Index: DatetimeIndex (naive, UTC-aligned by FRED convention).
        Values: float (NaN where FRED returns ".").
        """
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }
        r = self.session.get(self.base_url, params=params, timeout=timeout)
        r.raise_for_status()
        observations = r.json().get("observations", [])

        df = pd.DataFrame(observations)
        if df.empty:
            return pd.Series(dtype=float, name=series_id)

        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"]
        s.name = series_id
        return s


def fetch_macro_panel(
    client: FredClient,
    series_meta: Optional[dict[str, str]] = None,
    start_date: str = "1990-01-01",
) -> pd.DataFrame:
    """Pull each series in `series_meta` and combine into a daily panel.

    Columns are the *internal* names (the keys of series_meta), not the FRED
    series IDs. This matches notebook 00's output and the existing
    `data/macro_raw.parquet`, so cached files round-trip.
    """
    meta = series_meta or FRED_SERIES_META
    out = {}
    for name, fred_id in meta.items():
        out[name] = client.fetch_series(fred_id, start_date=start_date)
    df = pd.concat(out.values(), axis=1)
    df.columns = list(meta.keys())
    return df


def load_macro_panel_from_parquet(path: str | Path) -> pd.DataFrame:
    """Convenience: read a cached macro_raw.parquet."""
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# FredFeatureView — implements core.FeatureView Protocol.
# ---------------------------------------------------------------------------

@dataclass
class FredFeatureView:
    """Point-in-time view over a FRED macro panel.

    Returns the latest float value (`get`) or full historical series
    (`get_series`) of any column, filtered to values whose `available_at`
    timestamp is <= the query time `t`. `available_at` is computed as
    `observation_date + publication_lag[series]`.

    Strategies that need derived features (YoY, MoM, lags) compute them
    from the raw series this view returns. Per project conventions, that
    feature work belongs to the strategy, not to the view.
    """

    panel: pd.DataFrame
    publication_lag: dict[str, timedelta] = field(
        default_factory=lambda: dict(PUBLICATION_LAG_BY_SERIES)
    )
    default_publication_lag: timedelta = DEFAULT_PUBLICATION_LAG

    def __post_init__(self) -> None:
        # Normalize the index to a tz-naive DatetimeIndex for arithmetic
        # consistency; the view treats observation dates as wall-clock dates,
        # not timezone-aware moments.
        idx = pd.to_datetime(self.panel.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        self.panel = self.panel.copy()
        self.panel.index = idx

    # ------------------------------------------------------------------

    def _lag_for(self, name: str) -> timedelta:
        return self.publication_lag.get(name, self.default_publication_lag)

    def _available_series(self, name: str, t: datetime) -> pd.Series:
        if name not in self.panel.columns:
            raise KeyError(f"feature {name!r} not in FRED panel")
        s = self.panel[name].dropna()

        # Strip tz from t for comparison against tz-naive index.
        if t.tzinfo is not None:
            t_naive = t.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            t_naive = t

        cutoff = pd.Timestamp(t_naive) - self._lag_for(name)
        return s[s.index <= cutoff]

    # ---- FeatureView Protocol ----------------------------------------

    def get(self, name: str, t: datetime) -> Any:
        s = self._available_series(name, t)
        return float(s.iloc[-1]) if not s.empty else None

    def get_series(self, name: str, t: datetime) -> list[float]:
        return self._available_series(name, t).tolist()

    # ---- Convenience (not part of the Protocol) ----------------------

    def get_indexed_series(self, name: str, t: datetime) -> pd.Series:
        """Return the available series with its DatetimeIndex preserved.
        Useful for strategies that need to know observation dates (e.g. to
        compute YoY by aligning on month)."""
        return self._available_series(name, t)


# ---------------------------------------------------------------------------
# FredCpiSettlementLoader — implements core.SettlementLoader for CPI events.
# ---------------------------------------------------------------------------

_EVENT_YYMMM_RE = re.compile(r"-(\d{2})([A-Z]{3})$")


@dataclass
class FredCpiSettlementLoader:
    """Resolve KXECONSTATCPIYOY-* events from underlying CPILFESL data.

    Expects an event_id of the form `<SERIES>-YYMMM` (e.g.
    `KXECONSTATCPIYOY-26NOV`). Computes:

        cpi_yoy_pct = (cpi[YYMMM] / cpi[YYMMM - 1y] - 1) * 100

    and looks up the winning bucket via `catalog.ladder(event_id)`. Returns
    None if either CPI observation is missing in the panel — most commonly
    because the event hasn't been observed yet.

    `panel` carries the raw FRED panel. `cpi_column` is the panel column
    to read (default "core_cpi" matches notebook 00's naming).
    """

    catalog: MarketCatalog
    panel: pd.DataFrame
    cpi_column: str = "core_cpi"
    publication_lag: timedelta = DEFAULT_PUBLICATION_LAG

    def __post_init__(self) -> None:
        if self.cpi_column not in self.panel.columns:
            raise KeyError(
                f"cpi_column {self.cpi_column!r} not in panel; "
                f"available: {list(self.panel.columns)}"
            )
        idx = pd.to_datetime(self.panel.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        self._panel = self.panel.copy()
        self._panel.index = idx

    # ------------------------------------------------------------------

    def _parse_event_yymmm(self, event_id: EventId) -> Optional[tuple[int, int]]:
        m = _EVENT_YYMMM_RE.search(event_id)
        if m is None:
            return None
        yy = int(m.group(1))
        mmm = m.group(2)
        month = _MONTH_ABBR_TO_NUM.get(mmm)
        if month is None:
            return None
        # 2-digit years: 00-79 -> 2000s, 80-99 -> 1900s. Backtest-relevant
        # markets are all 21st century but be defensive.
        year = 2000 + yy if yy < 80 else 1900 + yy
        return year, month

    def _cpi_for_month(self, year: int, month: int) -> Optional[float]:
        s = self._panel[self.cpi_column].dropna()
        target = pd.Period(year=year, month=month, freq="M")
        in_month = s[s.index.to_period("M") == target]
        if in_month.empty:
            return None
        return float(in_month.iloc[0])

    # ---- SettlementLoader Protocol -----------------------------------

    def outcome(self, event_id: EventId) -> Optional[Outcome]:
        parsed = self._parse_event_yymmm(event_id)
        if parsed is None:
            return None
        year, month = parsed

        v_target = self._cpi_for_month(year, month)
        v_prior = self._cpi_for_month(year - 1, month)
        if v_target is None or v_prior is None:
            return None

        yoy_pct = (v_target / v_prior - 1.0) * 100.0

        ladder = self.catalog.ladder(event_id)
        if ladder is None:
            return None
        winning_market_id = ladder.winning_market(yoy_pct)

        # Approximate resolution moment: first day of release month + lag.
        # The release month is one after the observation month.
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        resolved_at_naive = (
            datetime(next_year, next_month, 1) + self.publication_lag
        )
        resolved_at = resolved_at_naive.replace(tzinfo=timezone.utc)

        return Outcome(
            event_id=event_id,
            winning_market_id=winning_market_id,
            expiration_value=yoy_pct,
            resolved_at=resolved_at,
        )
