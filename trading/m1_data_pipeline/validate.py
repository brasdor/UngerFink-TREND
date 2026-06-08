import os
import sys
import sqlite3
from datetime import date, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config import DB_PATH, MIN_VOLUME_USDT
from m1_data_pipeline.database import _resolve_db_path

_CHECKS: list[tuple[str, callable]] = []
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn
    return decorator


def _conn():
    return sqlite3.connect(_resolve_db_path())


@check("Universe >= 100 symbols with volume > $5M")
def check_universe_size():
    today = str(date.today())
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM universe_history WHERE date = ? AND quote_volume >= ?",
            (today, MIN_VOLUME_USDT),
        ).fetchone()
    n = row[0] if row else 0
    assert n >= 100, f"only {n} symbols (need >= 100)"


@check("OHLCV no gap > 3 days for top-20 symbols")
def check_ohlcv_gaps():
    today = str(date.today())
    with _conn() as c:
        top20 = [
            r[0]
            for r in c.execute(
                "SELECT symbol FROM universe_history WHERE date = ? "
                "ORDER BY quote_volume DESC LIMIT 20",
                (today,),
            ).fetchall()
        ]
        failures = []
        for sym in top20:
            rows = c.execute(
                "SELECT open_time FROM ohlcv WHERE symbol = ? ORDER BY open_time",
                (sym,),
            ).fetchall()
            if not rows:
                failures.append(f"{sym} has no data")
                continue
            ms_day = 86_400_000
            for i in range(1, len(rows)):
                gap = rows[i][0] - rows[i - 1][0]
                if gap > 3 * ms_day:
                    days = gap // ms_day
                    failures.append(f"{sym} gap {days}d")
    assert not failures, "; ".join(failures[:5])


@check("Funding rates for >= 80% of universe")
def check_funding_coverage():
    today = str(date.today())
    with _conn() as c:
        total = c.execute(
            "SELECT COUNT(*) FROM universe_history WHERE date = ?", (today,)
        ).fetchone()[0]
        covered = c.execute(
            "SELECT COUNT(DISTINCT f.symbol) FROM funding_rates f "
            "JOIN universe_history u ON f.symbol = u.symbol AND u.date = ?",
            (today,),
        ).fetchone()[0]
    pct = covered / max(total, 1)
    assert pct >= 0.8, f"{covered}/{total} symbols have funding data ({pct:.0%})"


@check("All 4 FRED macro series present with data in last 30 days")
def check_macro():
    cutoff = str(date.today() - timedelta(days=30))
    required = {"M2SL", "DTWEXBGS", "FEDFUNDS", "T10YIE"}
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT series_id FROM macro WHERE date >= ?", (cutoff,)
        ).fetchall()
    present = {r[0] for r in rows}
    missing = required - present
    assert not missing, f"Missing FRED series: {missing}"


@check("Universe history: at least 1 snapshot stored")
def check_universe_history():
    with _conn() as c:
        count = c.execute("SELECT COUNT(*) FROM universe_history").fetchone()[0]
    assert count > 0, "universe_history table is empty"


def run_all() -> bool:
    db_path = _resolve_db_path()
    if not os.path.exists(db_path):
        print(f"FAIL  DB not found at {db_path}")
        return False

    all_pass = True
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name} — {e}")
            all_pass = False
        except Exception as e:
            print(f"FAIL  {name} — unexpected error: {e}")
            all_pass = False

    return all_pass


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
