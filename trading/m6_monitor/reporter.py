"""
M6 Reporter
===========
Weekly performance reports and strategy health monitoring.

CLI:
  python -m m6_monitor.reporter --week 2026-06-07   # report for the week containing that date
  python -m m6_monitor.reporter --week current       # report for the current week
  python -m m6_monitor.reporter --health             # strategy health check only
"""
import argparse
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn

_BACKTEST_SHARPE = 1.132
_BACKTEST_MDD    = -45.4   # percent


# ── DB schema ─────────────────────────────────────────────────────────────────

def init_reporter_db(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weekly_reports (
            week_start       TEXT PRIMARY KEY,
            week_end         TEXT,
            weekly_pnl       REAL,
            weekly_return_pct REAL,
            win_rate         REAL,
            sharpe_live      REAL,
            sharpe_backtest  REAL DEFAULT 1.132,
            max_drawdown     REAL,
            best_trade       TEXT,
            worst_trade      TEXT,
            notes            TEXT
        )
    """)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _get_week_bounds(date_str: str) -> tuple:
    """Return (monday_iso, sunday_iso) for the week containing date_str."""
    d = date.fromisoformat(date_str)
    monday = d - timedelta(days=d.weekday())   # weekday() == 0 for Monday
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), sunday.isoformat()


def _week_ms_bounds(week_start: str, week_end: str) -> tuple:
    """Return (start_ms, end_exclusive_ms) covering the full week in UTC."""
    start_dt = datetime.fromisoformat(week_start).replace(tzinfo=timezone.utc)
    end_dt   = (datetime.fromisoformat(week_end) + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


# ── Sharpe helper ─────────────────────────────────────────────────────────────

def _annualised_sharpe(weekly_returns: list) -> float:
    """Annualised Sharpe from a list of weekly return percentages. NaN if < 4 obs."""
    n = len(weekly_returns)
    if n < 4:
        return float("nan")
    mean_r = sum(weekly_returns) / n
    var    = sum((r - mean_r) ** 2 for r in weekly_returns) / (n - 1)
    std_r  = math.sqrt(var) if var > 0 else 0.0
    if std_r <= 0:
        return float("nan")
    return (mean_r / std_r) * math.sqrt(52)


# ── Drawdown helper ───────────────────────────────────────────────────────────

def _max_drawdown_from_history(conn) -> float:
    """
    Compute running max drawdown from weekly_reports P&L history.
    Returns the worst drawdown seen (as a negative percentage of collateral).
    """
    rows = conn.execute(
        "SELECT weekly_pnl FROM weekly_reports ORDER BY week_start ASC"
    ).fetchall()
    pnls = [float(r[0]) for r in rows if r[0] is not None]
    if not pnls:
        return 0.0

    # Estimate starting collateral from paper_trades (sum of all collateral ever deployed)
    col_row = conn.execute(
        "SELECT COALESCE(SUM(collateral), 0.0) FROM paper_trades WHERE status='closed'"
    ).fetchone()
    open_col = conn.execute(
        "SELECT COALESCE(SUM(collateral), 0.0) FROM paper_trades WHERE status='open'"
    ).fetchone()
    baseline = float(col_row[0]) + float(open_col[0])
    if baseline <= 0:
        return 0.0

    cum    = 0.0
    peak   = 0.0
    worst  = 0.0
    for wp in pnls:
        cum  += wp
        peak  = max(peak, cum)
        dd    = (cum - peak) / (baseline + peak) * 100 if (baseline + peak) > 0 else 0.0
        worst = min(worst, dd)
    return round(worst, 4)


# ── Weekly report ─────────────────────────────────────────────────────────────

def generate_weekly_report(conn, week_start: str, week_end: str) -> dict:
    """
    Build a full performance report for the given week.
    Requires minimum 4 weeks of weekly_reports history to compute Sharpe.
    """
    init_reporter_db(conn)
    start_ms, end_ms = _week_ms_bounds(week_start, week_end)

    # Trades closed during the week
    trades = pd.read_sql_query(
        "SELECT symbol, direction, entry_price, exit_price, notional, collateral, pnl "
        "FROM paper_trades "
        "WHERE status='closed' AND exit_time >= ? AND exit_time < ?",
        conn,
        params=(start_ms, end_ms),
    )

    n_trades = len(trades)
    notes_parts = []

    if n_trades == 0:
        weekly_pnl       = 0.0
        weekly_return_pct = 0.0
        win_rate         = 0.0
        best_trade       = "n/a"
        worst_trade      = "n/a"
        notes_parts.append("No closed trades this week")
    else:
        weekly_pnl = float(trades["pnl"].sum())
        total_col  = float(trades["collateral"].sum())
        weekly_return_pct = weekly_pnl / total_col * 100 if total_col > 0 else 0.0
        winners    = (trades["pnl"] > 0).sum()
        win_rate   = winners / n_trades * 100

        best_row   = trades.loc[trades["pnl"].idxmax()]
        worst_row  = trades.loc[trades["pnl"].idxmin()]
        best_trade  = f"{best_row['symbol']} {best_row['pnl']:+.2f}"
        worst_trade = f"{worst_row['symbol']} {worst_row['pnl']:+.2f}"

    # Running Sharpe from all weekly_reports + this week
    hist = conn.execute(
        "SELECT weekly_return_pct FROM weekly_reports ORDER BY week_start ASC"
    ).fetchall()
    all_weekly_returns = [float(r[0]) for r in hist if r[0] is not None]
    all_weekly_returns.append(weekly_return_pct)   # include current week

    if len(all_weekly_returns) < 4:
        sharpe_live = float("nan")
        notes_parts.append(
            f"Insufficient data ({len(all_weekly_returns)} weeks) -- Sharpe requires 4+"
        )
    else:
        sharpe_live = _annualised_sharpe(all_weekly_returns)

    max_drawdown = _max_drawdown_from_history(conn)

    # Compare vs backtest
    if not math.isnan(sharpe_live):
        gap = _BACKTEST_SHARPE - sharpe_live
        if gap > 0.3:
            notes_parts.append(
                f"Live Sharpe {sharpe_live:.3f} vs backtest {_BACKTEST_SHARPE:.3f} "
                f"(gap: {gap:.3f} -- exceeds 0.3 threshold)"
            )

    if max_drawdown < _BACKTEST_MDD:
        notes_parts.append(
            f"Live MDD {max_drawdown:.1f}% exceeds backtest MDD {_BACKTEST_MDD:.1f}%"
        )

    return {
        "week_start":        week_start,
        "week_end":          week_end,
        "weekly_pnl":        round(weekly_pnl, 4),
        "weekly_return_pct": round(weekly_return_pct, 4),
        "win_rate":          round(win_rate, 2),
        "sharpe_live":       sharpe_live if not math.isnan(sharpe_live) else None,
        "sharpe_backtest":   _BACKTEST_SHARPE,
        "max_drawdown":      max_drawdown,
        "best_trade":        best_trade,
        "worst_trade":       worst_trade,
        "notes":             "; ".join(notes_parts) if notes_parts else "",
        "n_trades":          n_trades,
    }


def save_weekly_report(conn, report: dict) -> None:
    """Upsert report into weekly_reports table."""
    init_reporter_db(conn)
    conn.execute(
        "INSERT OR REPLACE INTO weekly_reports "
        "(week_start, week_end, weekly_pnl, weekly_return_pct, win_rate, "
        " sharpe_live, sharpe_backtest, max_drawdown, best_trade, worst_trade, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            report["week_start"],
            report["week_end"],
            report["weekly_pnl"],
            report["weekly_return_pct"],
            report["win_rate"],
            report.get("sharpe_live"),
            report["sharpe_backtest"],
            report["max_drawdown"],
            report["best_trade"],
            report["worst_trade"],
            report["notes"],
        ),
    )
    logger.info(f"Saved weekly report for {report['week_start']}")


def print_weekly_report(conn, week_start: str, week_end: str) -> dict:
    """Generate, print, and save the weekly report. Also sends via Telegram."""
    report = generate_weekly_report(conn, week_start, week_end)

    sharpe_str = (
        f"{report['sharpe_live']:.3f}"
        if report["sharpe_live"] is not None
        else "n/a (insufficient data)"
    )

    print()
    print("=" * 62)
    print(f"  Weekly Report  |  {week_start} to {week_end}")
    print("=" * 62)
    print(f"  Trades closed    : {report['n_trades']}")
    print(f"  Weekly P&L       : ${report['weekly_pnl']:>+10,.2f}  "
          f"({report['weekly_return_pct']:+.2f}%)")
    print(f"  Win rate         : {report['win_rate']:.1f}%")
    print(f"  Sharpe (live)    : {sharpe_str}")
    print(f"  Sharpe (backtest): {_BACKTEST_SHARPE:.3f}")
    print(f"  Max drawdown     : {report['max_drawdown']:.2f}%")
    print(f"  Best trade       : {report['best_trade']}")
    print(f"  Worst trade      : {report['worst_trade']}")
    if report["notes"]:
        print(f"  Notes            : {report['notes']}")
    print("=" * 62)
    print()

    save_weekly_report(conn, report)

    try:
        from m6_monitor.alerts import send_weekly_report
        send_weekly_report(report)
    except Exception as exc:
        logger.warning(f"Could not send weekly report via Telegram: {exc}")

    return report


# ── Strategy health ───────────────────────────────────────────────────────────

def check_strategy_health(conn) -> dict:
    """
    Compares live performance vs backtest benchmarks.
    Returns {ok: bool, warnings: list[str], critical: list[str]}.

    Checks:
    1. Live Sharpe < 0.5 after 8+ weeks              -> WARNING
    2. Live Sharpe < 50% of backtest Sharpe (1.132)  -> WARNING
    3. Live MDD > backtest MDD (-45.4%)              -> CRITICAL
    4. Q5 avg return < Q1 avg return for 3+ weeks    -> WARNING
    5. Estimated fee drag > 15% of gross returns     -> WARNING
    """
    init_reporter_db(conn)
    warnings  = []
    critical  = []

    # Collect weekly return history
    hist = conn.execute(
        "SELECT week_start, weekly_return_pct FROM weekly_reports ORDER BY week_start ASC"
    ).fetchall()
    weekly_returns = [float(r[1]) for r in hist if r[1] is not None]
    n_weeks = len(weekly_returns)

    sharpe = _annualised_sharpe(weekly_returns) if n_weeks >= 4 else float("nan")

    # Check 1: Sharpe below minimum after 8+ weeks
    if n_weeks >= 8 and not math.isnan(sharpe) and sharpe < 0.5:
        warnings.append(
            f"Sharpe below minimum threshold: {sharpe:.3f} < 0.5 ({n_weeks} weeks)"
        )

    # Check 2: Sharpe < 50% of backtest
    if not math.isnan(sharpe) and sharpe < _BACKTEST_SHARPE * 0.5:
        warnings.append(
            f"Strategy decay detected: live Sharpe {sharpe:.3f} < "
            f"50% of backtest {_BACKTEST_SHARPE:.3f}"
        )

    # Check 3: Live MDD exceeds backtest MDD
    mdd = _max_drawdown_from_history(conn)
    if mdd < _BACKTEST_MDD:
        critical.append(
            f"Drawdown exceeds backtest maximum: {mdd:.1f}% vs {_BACKTEST_MDD:.1f}%"
        )

    # Check 4: Q5 (longs) vs Q1 (shorts) factor edge
    # Look at last 5 weeks of closed trades grouped by week (exit_time // 604800000)
    trade_rows = pd.read_sql_query(
        "SELECT direction, pnl, collateral, "
        "CAST(exit_time / 604800000 AS INTEGER) AS week_bucket "
        "FROM paper_trades WHERE status='closed' AND collateral > 0 "
        "ORDER BY exit_time ASC",
        conn,
    )
    if not trade_rows.empty:
        trade_rows["return_pct"] = trade_rows["pnl"] / trade_rows["collateral"] * 100
        weekly_buckets = trade_rows.groupby("week_bucket")
        q5_better_count = 0
        q5_worse_count  = 0
        for _, grp in weekly_buckets:
            long_ret  = grp[grp["direction"] ==  1]["return_pct"].mean()
            short_ret = grp[grp["direction"] == -1]["return_pct"].mean()
            if pd.notna(long_ret) and pd.notna(short_ret):
                if long_ret >= short_ret:
                    q5_better_count += 1
                    q5_worse_count   = 0   # reset streak
                else:
                    q5_worse_count  += 1
                    q5_better_count  = 0
        if q5_worse_count >= 3:
            warnings.append(
                f"Factor edge reversing: Q5 (longs) underperforming Q1 (shorts) "
                f"for {q5_worse_count} consecutive weeks"
            )

    # Check 5: Fee drag
    # Estimated round-trip cost = 0.04% of notional (0.02% maker each leg)
    cost_row = conn.execute(
        "SELECT COALESCE(SUM(ABS(pnl)), 0.0), COALESCE(SUM(notional), 0.0) "
        "FROM paper_trades WHERE status='closed'"
    ).fetchone()
    if cost_row:
        gross_returns   = float(cost_row[0])
        total_notional  = float(cost_row[1])
        estimated_costs = total_notional * 0.0004   # 0.02% × 2 legs
        if gross_returns > 0:
            fee_drag = estimated_costs / gross_returns
            if fee_drag > 0.15:
                warnings.append(
                    f"Fee drag too high: {fee_drag:.1%} of gross returns "
                    f"-- consider reducing rebalancing frequency"
                )

    ok = len(critical) == 0
    return {"ok": ok, "warnings": warnings, "critical": critical, "n_weeks": n_weeks}


def print_strategy_health(conn) -> None:
    """Print strategy health check results to console."""
    result = check_strategy_health(conn)

    print()
    print("=" * 62)
    print("  Strategy Health Check")
    print("=" * 62)
    print(f"  Weeks of data    : {result['n_weeks']}")

    if not result["warnings"] and not result["critical"]:
        print("  All checks OK")
    for w in result["warnings"]:
        print(f"  [WARNING]  {w}")
    for c in result["critical"]:
        print(f"  [CRITICAL] {c}")
    print("=" * 62)
    print()

    if result["warnings"] or result["critical"]:
        try:
            from m6_monitor.alerts import send_alert
            lines = ["Strategy health check:"]
            for w in result["warnings"]:
                lines.append(f"WARNING: {w}")
            for c in result["critical"]:
                lines.append(f"CRITICAL: {c}")
            level = "critical" if result["critical"] else "warning"
            send_alert("\n".join(lines), level=level)
        except Exception as exc:
            logger.warning(f"Could not send health alert: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M6 Weekly Reporter")
    parser.add_argument(
        "--week", default="current",
        help="Date within the target week (YYYY-MM-DD) or 'current'"
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Run strategy health check only"
    )
    args = parser.parse_args()

    date_str = (
        datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if args.week == "current"
        else args.week
    )
    week_start, week_end = _get_week_bounds(date_str)

    with get_conn() as conn:
        if args.health:
            print_strategy_health(conn)
        else:
            print_weekly_report(conn, week_start, week_end)
            print_strategy_health(conn)
