import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn, read_universe, upsert
from m2_factor_engine.factors import compute_all_factors
from m2_factor_engine.ranker import rank_universe, WEIGHTS

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_WEEKLY_RANKINGS_DDL = """
    CREATE TABLE IF NOT EXISTS weekly_rankings (
        date        TEXT    NOT NULL,
        symbol      TEXT    NOT NULL,
        momentum_z  REAL,
        carry_z     REAL,
        size_z      REAL,
        liquidity_z REAL,
        oi_z        REAL,
        composite_z REAL,
        quintile    INT,
        rank        INT,
        PRIMARY KEY (date, symbol)
    )
"""


def init_m2_db() -> None:
    """Create M2 tables that live alongside M1 tables in the same DB."""
    with get_conn() as conn:
        conn.execute(_WEEKLY_RANKINGS_DDL)


def _last_monday() -> str:
    """Return most recent Monday that is <= yesterday (last completed rebalancing date)."""
    yesterday = date.today() - timedelta(days=1)
    monday = yesterday - timedelta(days=yesterday.weekday())
    return str(monday)


def _get_universe_for_date(date_str: str) -> pd.DataFrame:
    """
    Return the universe snapshot closest to (and not after) date_str.
    Falls back to the earliest available snapshot if none precedes the date.
    This is the survivorship-bias protection: each historical date uses the
    universe that was live at that time, not today's survivors.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM universe_history WHERE date <= ?", (date_str,)
        ).fetchone()
        snap_date = row[0] if row else None

        if snap_date is None:
            row = conn.execute("SELECT MIN(date) FROM universe_history").fetchone()
            snap_date = row[0] if row else None

    if snap_date is None:
        raise ValueError("No universe snapshots found. Run M1 pipeline first.")

    if snap_date != date_str:
        logger.debug(f"No snapshot for {date_str} -- using nearest available: {snap_date}")

    return read_universe(snap_date)


def run_weekly_ranking(date_str: str = None) -> pd.DataFrame:
    """
    Compute factor scores -> rank -> save to weekly_rankings for one date.
    Returns the ranked DataFrame (sorted by rank ascending).
    """
    init_m2_db()
    if date_str is None:
        date_str = _last_monday()

    logger.info(f"Factor ranking: {date_str}")

    universe = _get_universe_for_date(date_str)
    logger.info(f"Universe: {len(universe)} symbols")

    with get_conn() as conn:
        factors_df = compute_all_factors(date_str, universe, conn)

    ranked = rank_universe(factors_df)

    db_rows = ranked.assign(date=date_str).to_dict("records")
    upsert("weekly_rankings", db_rows)
    logger.info(f"Saved {len(ranked)} rankings -> weekly_rankings")

    return ranked


def run_historical_rankings(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Backfill rankings for every Monday in [start_date, end_date].
    Uses the closest historical universe snapshot for each date --
    not today's universe -- to avoid survivorship bias.
    """
    init_m2_db()
    mondays = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    logger.info(f"Backfilling {len(mondays)} weekly rankings  {start_date} -> {end_date}")

    all_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for mon in tqdm(mondays, desc="Weekly rankings", unit="week"):
        date_str = str(mon.date())
        try:
            ranked = run_weekly_ranking(date_str)
            if not ranked.empty:
                all_frames.append(ranked.assign(date=date_str))
        except Exception as exc:
            logger.warning(f"Skipped {date_str}: {exc}")
            errors.append(date_str)

    logger.info(
        f"Backfill complete -- {len(all_frames)} dates ranked, {len(errors)} skipped"
    )
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# CLI output helpers

def _print_summary(date_str: str, ranked: pd.DataFrame) -> None:
    n = len(ranked)
    sep = "=" * 68

    print(f"\n{sep}")
    print(f"  M2 Factor Rankings -- {date_str}   ({n} symbols ranked)")
    print(sep)

    _print_basket("TOP 10  (Q5 - long basket)", ranked[ranked["quintile"] == 5].head(10))
    _print_basket("BOTTOM 10  (Q1 - short basket)",
                  ranked[ranked["quintile"] == 1].sort_values("rank", ascending=False).head(10))
    _print_factor_breakdown(ranked.iloc[0])

    print(sep)


def _print_basket(title: str, subset: pd.DataFrame) -> None:
    hdr = f"  {'Rnk':<4} {'Symbol':<14} {'Composite':>10}  " \
          f"{'Mom':>7} {'Carry':>7} {'Size':>7} {'Liq':>7} {'OI':>6}"
    print(f"\n  {title}")
    print(hdr)
    print(f"  {'-'*66}")
    for _, row in subset.iterrows():
        print(
            f"  {int(row['rank']):<4} {row['symbol']:<14} {row['composite_z']:>+10.4f}  "
            f"{row['momentum_z']:>+7.3f} {row['carry_z']:>+7.3f} "
            f"{row['size_z']:>+7.3f} {row['liquidity_z']:>+7.3f} {row['oi_z']:>+6.3f}"
        )


def _print_factor_breakdown(row: pd.Series) -> None:
    print(f"\n  Factor breakdown - #{int(row['rank'])}: {row['symbol']}")
    print(f"  {'Factor':<14} {'Weight':>7}  {'Z-score':>9}  {'Contribution':>13}")
    print(f"  {'-'*50}")
    breakdown = [
        ("momentum",    "momentum_z",  0.35),
        ("carry",       "carry_z",     0.25),
        ("size",        "size_z",      0.20),
        ("liquidity",   "liquidity_z", 0.15),
        ("oi_momentum", "oi_z",        0.05),
    ]
    total = 0.0
    for label, col, w in breakdown:
        z = row[col]
        contrib = w * z
        total += contrib
        print(f"  {label:<14} {w:>6.0%}   {z:>+9.4f}  {contrib:>+13.4f}")
    print(f"  {'-'*50}")
    print(f"  {'composite_z':<14} {'100%':>7}   {row['composite_z']:>+9.4f}  {total:>+13.4f}")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 Factor Engine")
    sub = parser.add_mutually_exclusive_group()
    sub.add_argument("--date", type=str,
                     help="Single rebalancing date YYYY-MM-DD (default: last Monday)")
    sub.add_argument("--backfill", action="store_true",
                     help="Backfill historical rankings (requires --start and --end)")
    parser.add_argument("--start", type=str, help="Backfill start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, help="Backfill end date YYYY-MM-DD")
    args = parser.parse_args()

    if args.backfill:
        if not args.start or not args.end:
            parser.error("--backfill requires --start and --end")
        df = run_historical_rankings(args.start, args.end)
        print(f"\nBackfill complete -- {len(df)} total symbol-date rows saved.")
    else:
        date_str = args.date or _last_monday()
        ranked = run_weekly_ranking(date_str)
        _print_summary(date_str, ranked)
