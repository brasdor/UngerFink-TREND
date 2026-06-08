"""
M6 Dashboard
============
Real-time P&L and position monitor for the paper trading portfolio.
Read-only -- never writes to the database.

CLI: python -m m6_monitor.dashboard
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from m1_data_pipeline.database import get_conn


# ── Price lookup ──────────────────────────────────────────────────────────────

def _fetch_latest_prices(conn, symbols: list) -> dict:
    """Return {symbol: latest_close} from ohlcv for each symbol in the list."""
    if not symbols:
        return {}
    ph = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT o.symbol, o.close FROM ohlcv o "
        f"INNER JOIN ("
        f"  SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
        f"  WHERE symbol IN ({ph}) GROUP BY symbol"
        f") m ON o.symbol=m.symbol AND o.open_time=m.max_t",
        symbols,
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_portfolio_summary(conn) -> dict:
    """
    Compute portfolio summary with live unrealised P&L.
    Symbols with no ohlcv price (recently delisted) fall back to entry_price.
    """
    rows = conn.execute(
        "SELECT symbol, direction, entry_price, notional, collateral "
        "FROM paper_trades WHERE status='open'"
    ).fetchall()

    if not rows:
        return {
            "total_collateral":     0.0,
            "total_notional":       0.0,
            "unrealised_pnl":       0.0,
            "realised_pnl":         0.0,
            "total_pnl":            0.0,
            "pnl_pct":              0.0,
            "open_positions":       0,
            "long_count":           0,
            "short_count":          0,
            "peak_equity":          0.0,
            "current_drawdown_pct": 0.0,
        }

    syms = [r[0] for r in rows]
    prices = _fetch_latest_prices(conn, syms)

    missing: list = []
    unrealised = 0.0
    total_collateral = 0.0
    total_notional   = 0.0
    long_count = 0
    short_count = 0

    for sym, direction, entry_price, notional, collateral in rows:
        px = prices.get(sym)
        if not px or px <= 0:
            px = entry_price
            missing.append(sym)
        pnl = (px - entry_price) / entry_price * notional * direction
        unrealised       += pnl
        total_collateral += collateral
        total_notional   += notional
        if direction == 1:
            long_count += 1
        else:
            short_count += 1

    if missing:
        logger.warning(
            f"No current price for {len(missing)} symbol(s) -- using entry price: "
            + ", ".join(missing)
        )

    realised = float(conn.execute(
        "SELECT COALESCE(SUM(pnl), 0.0) FROM paper_trades WHERE status='closed'"
    ).fetchone()[0])

    total_pnl = unrealised + realised
    pnl_pct   = total_pnl / total_collateral * 100 if total_collateral > 0 else 0.0

    # Peak equity from weekly_reports cumulative history (table may not exist yet)
    try:
        hist_rows = conn.execute(
            "SELECT weekly_pnl FROM weekly_reports ORDER BY week_start ASC"
        ).fetchall()
    except Exception:
        hist_rows = []
    weekly_pnls = [float(r[0]) for r in hist_rows if r[0] is not None]

    if weekly_pnls:
        cum = 0.0
        peak_cum = 0.0
        for wp in weekly_pnls:
            cum += wp
            peak_cum = max(peak_cum, cum)
        # Approximate current equity vs historical peak
        current_cum = cum + unrealised
        drawdown_abs = current_cum - peak_cum
        peak_equity = total_collateral + peak_cum
        current_drawdown_pct = (
            drawdown_abs / peak_equity * 100 if peak_equity > 0 else 0.0
        )
        current_drawdown_pct = min(0.0, current_drawdown_pct)
    else:
        peak_equity = total_collateral + max(0.0, total_pnl)
        current_drawdown_pct = (
            min(0.0, total_pnl / total_collateral * 100) if total_collateral > 0 else 0.0
        )

    return {
        "total_collateral":     round(total_collateral, 2),
        "total_notional":       round(total_notional, 2),
        "unrealised_pnl":       round(unrealised, 2),
        "realised_pnl":         round(realised, 2),
        "total_pnl":            round(total_pnl, 2),
        "pnl_pct":              round(pnl_pct, 4),
        "open_positions":       len(rows),
        "long_count":           long_count,
        "short_count":          short_count,
        "peak_equity":          round(peak_equity, 2),
        "current_drawdown_pct": round(current_drawdown_pct, 4),
    }


# ── Position detail ───────────────────────────────────────────────────────────

def get_position_detail(conn) -> pd.DataFrame:
    """
    All open positions with live unrealised P&L and return %.
    Sorted by unrealised_pnl descending.
    Symbols with no ohlcv price are flagged in price_source column.
    """
    rows = conn.execute(
        "SELECT id, symbol, direction, entry_price, notional, collateral, entry_time "
        "FROM paper_trades WHERE status='open'"
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    syms = [r[1] for r in rows]
    prices = _fetch_latest_prices(conn, syms)

    records = []
    for trade_id, sym, direction, entry_price, notional, collateral, entry_time in rows:
        px = prices.get(sym)
        price_source = "live"
        if not px or px <= 0:
            px = entry_price
            price_source = "entry (no live price)"
        pnl     = (px - entry_price) / entry_price * notional * direction
        ret_pct = pnl / collateral * 100 if collateral > 0 else 0.0
        records.append({
            "symbol":         sym,
            "direction":      "LONG" if direction == 1 else "SHORT",
            "entry_price":    entry_price,
            "current_price":  px,
            "notional":       notional,
            "collateral":     collateral,
            "unrealised_pnl": round(pnl, 4),
            "return_pct":     round(ret_pct, 4),
            "price_source":   price_source,
            "entry_time":     datetime.fromtimestamp(
                entry_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(records)
    return df.sort_values("unrealised_pnl", ascending=False).reset_index(drop=True)


# ── Console output ────────────────────────────────────────────────────────────

def _print_pos_rows(df: pd.DataFrame) -> None:
    hdr = f"    {'Symbol':<18} {'Dir':<6} {'Entry':>10} {'Now':>10} {'P&L':>9} {'Ret%':>7}"
    sep = f"    {'-'*18} {'-'*6} {'-'*10} {'-'*10} {'-'*9} {'-'*7}"
    print(hdr)
    print(sep)
    for _, r in df.iterrows():
        flag = " *" if r["price_source"] != "live" else ""
        print(
            f"    {r['symbol']:<18} {r['direction']:<6} "
            f"{r['entry_price']:>10.5f} {r['current_price']:>10.5f} "
            f"{r['unrealised_pnl']:>+9.2f} {r['return_pct']:>+6.2f}%{flag}"
        )


def print_dashboard(conn) -> None:
    """Print formatted portfolio dashboard to console."""
    summary = get_portfolio_summary(conn)
    detail  = get_position_detail(conn)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print()
    print("=" * 64)
    print(f"  UngerFink-TREND  |  Paper Portfolio  |  {ts}")
    print("=" * 64)
    print(f"  Open positions   : {summary['open_positions']:>6}"
          f"  ({summary['long_count']}L / {summary['short_count']}S)")
    print(f"  Total notional   : ${summary['total_notional']:>12,.2f}")
    print(f"  Total collateral : ${summary['total_collateral']:>12,.2f}")
    print(f"  Unrealised P&L   : ${summary['unrealised_pnl']:>+12,.2f}")
    print(f"  Realised P&L     : ${summary['realised_pnl']:>+12,.2f}")
    print(f"  Total P&L        : ${summary['total_pnl']:>+12,.2f}"
          f"  ({summary['pnl_pct']:+.2f}%)")
    print(f"  Peak equity      : ${summary['peak_equity']:>12,.2f}")
    print(f"  Current drawdown : {summary['current_drawdown_pct']:>+11.2f}%")

    # Regime + CB from last risk_state row
    risk_row = conn.execute(
        "SELECT cb_state, regime_score, position_multiplier FROM risk_state "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if risk_row:
        cb, regime, pos_mult = risk_row
        print()
        print(f"  Circuit breaker  : {cb.upper()}")
        print(f"  Regime score     : {regime:.3f}")
        print(f"  Position mult    : {pos_mult:.3f}")

    if not detail.empty:
        print()
        print("  Top 5 winners:")
        _print_pos_rows(detail.head(5))

        print()
        print("  Top 5 losers:")
        _print_pos_rows(detail.tail(5).iloc[::-1].reset_index(drop=True))
        print()
        print("  * = no live price; using entry price")

    print("=" * 64)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with get_conn() as conn:
        print_dashboard(conn)
