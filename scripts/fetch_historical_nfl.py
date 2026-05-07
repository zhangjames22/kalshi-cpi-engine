"""Fetch nflverse game data and build the NFL backtest fixture.

Pulls https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
(no auth required), filters to resolved 2024 + 2025-season games (regular
season + playoffs), and writes two parquets under
`tests/fixtures/historical_nfl/`:

  - games.parquet     — schema consumed by `data.sports.nfl.load_catalog_*`
                        and by NflSettlementLoader. Columns:
                            ticker, event_ticker, title,
                            open_time, close_time, expiration_time,
                            home_team, away_team, home_score, away_score,
                            home_moneyline, away_moneyline, spread_line,
                            season, week, game_type, kickoff_utc

  - snapshots.parquet — one orderbook snapshot per game, timestamped
                        24 hours before kickoff. yes_bid/yes_ask are derived
                        from the closing home moneyline (de-vigged against
                        the away moneyline) with a synthetic 1¢ spread.
                        These stand in for what we'd otherwise pull from
                        Kalshi's NFL series.

Re-runnable. Network-bound; takes ~5 seconds.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.sports.nfl import SERIES_ID, TICKER_PREFIX, devig, moneyline_to_prob


GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "historical_nfl"
SEASONS = (2024, 2025)


def fetch_raw_games() -> pd.DataFrame:
    print(f"[fetch] {GAMES_URL}")
    df = pd.read_csv(GAMES_URL)
    df = df[df["season"].isin(SEASONS)].copy()
    df = df.dropna(subset=["home_score", "away_score", "home_moneyline", "away_moneyline"])
    print(f"[fetch] {len(df)} resolved games across seasons {SEASONS}")
    return df


def kickoff_utc(row: pd.Series) -> pd.Timestamp:
    """Combine `gameday` (date, ET wall-clock) and `gametime` (HH:MM, ET)
    into a UTC timestamp. Falls back to 13:00 ET if `gametime` is missing
    (early-season Sunday-1pm slate is the league default)."""
    date = str(row["gameday"])
    raw_time = row.get("gametime")
    if pd.isna(raw_time) or not str(raw_time).strip():
        time_str = "13:00"
    else:
        time_str = str(raw_time).strip()
    ts = pd.Timestamp(f"{date} {time_str}", tz="US/Eastern")
    return ts.tz_convert("UTC")


def to_games_df(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in raw.iterrows():
        kt = kickoff_utc(r)
        # Markets in our model open ~24h before kickoff (mirrors when
        # closing lines start to stabilize), close at kickoff, and resolve
        # 4 hours later (covers OT).
        open_t = kt - timedelta(hours=24)
        close_t = kt
        exp_t = kt + timedelta(hours=4)
        ticker = TICKER_PREFIX + str(r["game_id"])
        title = f"{r['home_team']} home vs {r['away_team']} ({r['gameday']})"

        rows.append({
            "ticker": ticker,
            "event_ticker": ticker,    # binary: market == event
            "title": title,
            "open_time": open_t.isoformat(),
            "close_time": close_t.isoformat(),
            "expiration_time": exp_t.isoformat(),
            "kickoff_utc": kt.isoformat(),
            "season": int(r["season"]),
            "week": int(r["week"]),
            "game_type": str(r["game_type"]),
            "home_team": str(r["home_team"]),
            "away_team": str(r["away_team"]),
            "home_score": float(r["home_score"]),
            "away_score": float(r["away_score"]),
            "home_moneyline": float(r["home_moneyline"]),
            "away_moneyline": float(r["away_moneyline"]),
            "spread_line": float(r["spread_line"]) if not pd.isna(r["spread_line"]) else float("nan"),
        })
    df = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)
    return df


def to_snapshots_df(games: pd.DataFrame, *, spread_dollars: float = 0.01) -> pd.DataFrame:
    """One snapshot per game at kickoff - 1h.

    yes_mid = de-vigged implied probability that home team wins.
    yes_bid = mid - spread/2; yes_ask = mid + spread/2; clamped to [0.01, 0.99]
    so the synthetic bid/ask never quote a 0% or 100% wall.
    """
    rows = []
    for _, g in games.iterrows():
        p_home_raw = moneyline_to_prob(g["home_moneyline"])
        p_away_raw = moneyline_to_prob(g["away_moneyline"])
        p_home, _p_away = devig(p_home_raw, p_away_raw)

        yes_mid = max(0.01, min(0.99, p_home))
        half = spread_dollars / 2.0
        yes_bid = max(0.01, yes_mid - half)
        yes_ask = min(0.99, yes_mid + half)
        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid

        snap_ts = pd.Timestamp(g["kickoff_utc"]) - timedelta(hours=1)
        rows.append({
            "ticker": g["ticker"],
            "pulled_at_utc": snap_ts.isoformat(),
            "yes_bid_dollars": yes_bid,
            "yes_ask_dollars": yes_ask,
            "no_bid_dollars": no_bid,
            "no_ask_dollars": no_ask,
        })
    return pd.DataFrame(rows).sort_values("pulled_at_utc").reset_index(drop=True)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = fetch_raw_games()
    games = to_games_df(raw)
    snaps = to_snapshots_df(games)

    games_path = FIXTURE_DIR / "games.parquet"
    snaps_path = FIXTURE_DIR / "snapshots.parquet"
    games.to_parquet(games_path, index=False)
    snaps.to_parquet(snaps_path, index=False)

    print(f"[write] games -> {games_path} ({len(games)} rows)")
    print(f"[write] snapshots -> {snaps_path} ({len(snaps)} rows)")
    print(f"[summary] series={SERIES_ID}; "
          f"date range {games['kickoff_utc'].min()} -> {games['kickoff_utc'].max()}")
    print(f"[summary] home_win_rate = "
          f"{(games['home_score'] > games['away_score']).mean():.4f} "
          f"({(games['home_score'] > games['away_score']).sum()} of {len(games)})")


if __name__ == "__main__":
    main()
