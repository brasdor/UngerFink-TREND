#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9B DAILY SUMMARY
=================

Reads the current T9B paper engine output files and prints a clean,
phone-readable summary.

Designed to be run:
  - After phase_t9b_donchian_universev2_paper_engine.py completes
  - Directly from the command line for a quick status check
  - By the GitHub Actions workflow (output visible in the Actions log)

Usage:
  python t9b_daily_summary.py
  python t9b_daily_summary.py --json      # machine-readable JSON output

Reads (all in data/t9b_paper/):
  state.json           -- equity, positions, kill-switch
  open_positions.csv   -- current open position details
  signals_today.csv    -- signals detected on the last run
  equity_curve.csv     -- trade-by-trade equity history
  daily_log.csv        -- event log for all run dates

ASCII only in all print statements.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data" / "t9b_paper"
STATE_JSON   = DATA_DIR / "state.json"
OPEN_POS_CSV = DATA_DIR / "open_positions.csv"
SIGNALS_CSV  = DATA_DIR / "signals_today.csv"
EQUITY_CSV   = DATA_DIR / "equity_curve.csv"
DAILY_LOG    = DATA_DIR / "daily_log.csv"

OHLCV_CACHE   = DATA_DIR / "ohlcv_cache"
OHLCV_EXT_DIR = ROOT / "data" / "ohlcv_extended"

FREEZE_DATE    = date(2026, 5, 30)
REVIEW_DATE    = FREEZE_DATE + timedelta(days=90)    # 2026-08-28
INITIAL_CAP    = 10_000.0
EPS            = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def load_state() -> dict:
    if STATE_JSON.exists():
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def get_latest_close(symbol: str) -> float | None:
    """Load the most recent close price from any available cache."""
    safe = safe_sym(symbol)
    candidates = [
        OHLCV_EXT_DIR  / f"{safe}_1d_extended.csv",
        OHLCV_CACHE    / f"{safe}_1d.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["close"])
            if not df.empty:
                val = pd.to_numeric(df["close"], errors="coerce").dropna()
                if len(val) > 0:
                    return float(val.iloc[-1])
        except Exception:
            continue
    return None


# =============================================================================
# SUMMARY SECTIONS
# =============================================================================

def section_header(title: str, width: int = 56) -> None:
    print()
    print(title)
    print("-" * width)


def print_equity_section(state: dict, today: date) -> dict:
    eq       = float(state.get("paper_equity_usdt", INITIAL_CAP))
    peak     = float(state.get("peak_equity_usdt", INITIAL_CAP))
    dd       = float(state.get("drawdown_pct", 0.0))
    ret_pct  = (eq / INITIAL_CAP - 1.0) * 100.0
    kill_sw  = state.get("kill_switch_triggered", False)
    last_run = state.get("last_run_date", "n/a")

    days_running   = max((today - FREEZE_DATE).days, 0)
    days_to_review = (REVIEW_DATE - today).days

    section_header("EQUITY")
    print(f"  Paper equity   : ${eq:>10,.2f}  ({ret_pct:+.2f}% vs $10,000 start)")
    print(f"  Peak equity    : ${peak:>10,.2f}")
    print(f"  Max drawdown   : {dd:>+.2f}%")
    print(f"  Last run date  : {last_run}")
    print(f"  Days running   : {days_running}  (started {FREEZE_DATE})")
    if days_to_review > 0:
        print(f"  3-month review : {days_to_review} days away  ({REVIEW_DATE})")
    else:
        print(f"  3-month review : DUE  ({REVIEW_DATE})")
    if kill_sw:
        print()
        print("  *** KILL-SWITCH ACTIVE -- no new entries ***")

    return {
        "equity": eq, "return_pct": ret_pct, "max_dd": dd,
        "days_running": days_running, "days_to_review": days_to_review,
        "kill_switch": kill_sw,
    }


def print_positions_section(state: dict) -> list:
    positions = state.get("open_positions", [])
    section_header(f"OPEN POSITIONS ({len(positions)})")

    if not positions:
        print("  None")
        return []

    # Try to get current close for each position
    rows = []
    for p in positions:
        sym    = str(p.get("symbol", "?"))
        edate  = str(p.get("entry_date", "?"))
        eprice = float(p.get("entry_price", 0))
        stop   = float(p.get("current_stop", 0))
        init_r = float(p.get("initial_risk_per_unit", 0))
        maxr   = float(p.get("max_favorable_r", 0))
        ch     = bool(p.get("chandelier_active", False))
        ch_date = str(p.get("chandelier_activated_date", "")) or ""

        # Current R (if price available)
        cur_close = get_latest_close(sym)
        if cur_close and init_r > EPS:
            cur_r = (cur_close - eprice) / init_r
        else:
            cur_r = None

        rows.append({
            "symbol": sym, "entry_date": edate,
            "entry_price": eprice, "current_stop": stop,
            "max_r": maxr, "cur_r": cur_r,
            "chandelier": ch, "ch_date": ch_date,
        })

    print(f"  {'Symbol':<16} {'Entry':<12} {'Entry $':<12} {'Stop':<12} "
          f"{'Cur R':>7}  {'MaxR':>7}  Chandelier")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12} "
          f"{'-'*7}  {'-'*7}  {'-'*14}")

    for r in rows:
        cur_r_str = f"{r['cur_r']:+.2f}R" if r["cur_r"] is not None else "  n/a "
        ch_str    = f"YES (since {r['ch_date'][:10]})" if r["chandelier"] else "no"
        print(f"  {r['symbol']:<16} {r['entry_date']:<12} "
              f"{r['entry_price']:<12.6g} {r['current_stop']:<12.6g} "
              f"{cur_r_str:>7}  {r['max_r']:>+7.2f}R  {ch_str}")

    return rows


def print_signals_section() -> list:
    sig_df = load_csv(SIGNALS_CSV)
    section_header(f"SIGNALS TODAY ({len(sig_df)})")

    if sig_df.empty or len(sig_df) == 0:
        print("  None")
        return []

    for _, row in sig_df.iterrows():
        sym   = str(row.get("symbol", "?"))
        close = row.get("close", "?")
        stop  = row.get("initial_stop", "?")
        atr   = row.get("atr", "?")
        print(f"  {sym:<16}  close={close:<12.6g}  stop={stop:<12.6g}  atr={atr:<10.6g}")

    return sig_df.to_dict("records")


def print_trade_history_section() -> None:
    eq_df = load_csv(EQUITY_CSV)

    if eq_df.empty:
        section_header("TRADE HISTORY")
        print("  No closed trades yet.")
        return

    n         = len(eq_df)
    net_r_col = "gross_r" if "gross_r" in eq_df.columns else None
    pnl_col   = "pnl_usdt" if "pnl_usdt" in eq_df.columns else None

    section_header(f"TRADE HISTORY ({n} closed trades)")

    if net_r_col:
        r_vals  = pd.to_numeric(eq_df[net_r_col], errors="coerce").dropna()
        winners = r_vals[r_vals > 0]
        losers  = r_vals[r_vals < 0]
        win_rate = len(winners) / max(len(r_vals), 1) * 100.0
        avg_r    = r_vals.mean() if len(r_vals) > 0 else 0.0
        total_r  = r_vals.sum()

        print(f"  Total R      : {total_r:+.2f}R")
        print(f"  Avg R/trade  : {avg_r:+.3f}R")
        print(f"  Win rate     : {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")

    if pnl_col:
        pnl_vals = pd.to_numeric(eq_df[pnl_col], errors="coerce").dropna()
        total_pnl = pnl_vals.sum()
        print(f"  Total P&L    : ${total_pnl:+,.2f}")

    # Last 5 trades
    if n > 0:
        print()
        print(f"  Last {min(5, n)} trades:")
        last5 = eq_df.tail(5)
        for _, row in last5.iterrows():
            sym    = str(row.get("symbol", "?"))
            edate  = str(row.get("exit_date", "?"))
            r      = float(row.get("gross_r", 0)) if net_r_col else 0.0
            reason = str(row.get("exit_reason", "?"))
            eq_val = row.get("equity", "?")
            print(f"    {edate}  {sym:<16}  {r:+.3f}R  [{reason}]  eq=${eq_val:,.2f}" if eq_val != "?" else
                  f"    {edate}  {sym:<16}  {r:+.3f}R  [{reason}]")


def print_status_footer(eq_summary: dict) -> None:
    print()
    print("=" * 56)
    eq   = eq_summary.get("equity", INITIAL_CAP)
    ret  = eq_summary.get("return_pct", 0.0)
    dd   = eq_summary.get("max_dd", 0.0)
    kill = eq_summary.get("kill_switch", False)

    if kill:
        status = "KILL-SWITCH ACTIVE"
    elif dd < -20.0:
        status = "WARN: drawdown > 20%"
    elif ret < 0:
        status = "WARN: negative return"
    else:
        status = "OK"

    print(f"  STATUS: {status}")
    print(f"  equity=${eq:,.2f}  return={ret:+.2f}%  DD={dd:+.2f}%")
    days_r = eq_summary.get("days_to_review", 0)
    if days_r > 0:
        print(f"  {days_r} days until 3-month live deployment review")
    else:
        print(f"  3-month review period complete -- evaluate for live deployment")
    print("=" * 56)


# =============================================================================
# JSON OUTPUT
# =============================================================================

def print_json_output(state: dict, today: date) -> None:
    eq   = float(state.get("paper_equity_usdt", INITIAL_CAP))
    dd   = float(state.get("drawdown_pct", 0.0))
    ret  = (eq / INITIAL_CAP - 1.0) * 100.0
    kill = state.get("kill_switch_triggered", False)

    sig_df = load_csv(SIGNALS_CSV)
    n_sig  = len(sig_df) if not sig_df.empty else 0

    out = {
        "date":             str(today),
        "equity_usdt":      round(eq, 4),
        "return_pct":       round(ret, 4),
        "max_dd_pct":       round(dd, 4),
        "open_positions":   len(state.get("open_positions", [])),
        "closed_trades":    state.get("closed_trade_count", 0),
        "signals_today":    n_sig,
        "kill_switch":      kill,
        "days_running":     max((today - FREEZE_DATE).days, 0),
        "days_to_review":   (REVIEW_DATE - today).days,
        "review_date":      str(REVIEW_DATE),
    }
    print(json.dumps(out, indent=2))


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T9B daily summary")
    p.add_argument("--json", action="store_true",
                   help="Output machine-readable JSON instead of human-readable text")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    today = date.today()

    state = load_state()
    if not state:
        print("T9B state.json not found.")
        print("Run phase_t9b_donchian_universev2_paper_engine.py first.")
        return 1

    if args.json:
        print_json_output(state, today)
        return 0

    print("=" * 56)
    print("T9B PAPER ENGINE -- DAILY SUMMARY")
    print(f"As of: {today}  |  Started: {FREEZE_DATE}")
    print("=" * 56)

    eq_summary = print_equity_section(state, today)
    print_positions_section(state)
    print_signals_section()
    print_trade_history_section()
    print_status_footer(eq_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
