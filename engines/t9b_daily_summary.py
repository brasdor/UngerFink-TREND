#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9B PORTFOLIO SUMMARY
=====================

Combined daily summary for all three T9B paper engines.

Reads:
  data/t9b_paper/                   -- DonchianLong
  data/t9b_mr_paper/                -- RSI MeanReversion
  data/t9b_consecdowndays_paper/    -- ConsecDownDays

Usage:
  python t9b_daily_summary.py
  python t9b_daily_summary.py --json

ASCII only (Windows cp1252 compatible).
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parents[1]
OHLCV_CACHE   = ROOT / "data" / "universe" / "ohlcv_1d"
INITIAL_CAP   = 10_000.0
EPS           = 1e-12
W             = 64   # output width

SYSTEMS = [
    {
        "num":       1,
        "name":      "DonchianLong",
        "style":     "Trend Following",
        "data_dir":  ROOT / "data" / "t9b_paper",
        "equity_file": "engine_equity_curve.csv",
        "freeze":    date(2026, 5, 30),
        "pos_risk":  "initial_risk_per_unit",   # field in state open_positions
    },
    {
        "num":       2,
        "name":      "RSI MeanReversion",
        "style":     "All-Weather",
        "data_dir":  ROOT / "data" / "t9b_mr_paper",
        "equity_file": "engine_equity_curve.csv",
        "freeze":    date(2026, 6, 1),
        "pos_risk":  "initial_risk_per_unit",
    },
    {
        "num":       3,
        "name":      "ConsecDownDays",
        "style":     "Bull Specialist",
        "data_dir":  ROOT / "data" / "t9b_consecdowndays_paper",
        "equity_file": "engine_equity_curve.csv",
        "freeze":    date(2026, 6, 2),
        "pos_risk":  "initial_risk_per_unit",
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def latest_close(symbol: str) -> Optional[float]:
    """Read the most recent close from the committed OHLCV cache."""
    safe = symbol.replace("/", "_").replace(":", "_")
    path = OHLCV_CACHE / f"{safe}_1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["close"])
        vals = pd.to_numeric(df["close"], errors="coerce").dropna()
        return float(vals.iloc[-1]) if len(vals) > 0 else None
    except Exception:
        return None


def cur_r(entry_price: float, risk_unit: float, symbol: str) -> Optional[float]:
    close = latest_close(symbol)
    if close is None or risk_unit <= EPS:
        return None
    return (close - entry_price) / risk_unit


def fmt_r(r: Optional[float]) -> str:
    return f"{r:+.2f}R" if r is not None else "n/a"


def fmt_money(v: float) -> str:
    return f"${v:,.2f}"


# =============================================================================
# PER-SYSTEM SNAPSHOT
# =============================================================================

def system_snapshot(cfg: dict, today: date) -> dict:
    state   = load_json(cfg["data_dir"] / "state.json")
    sig_df  = load_csv(cfg["data_dir"] / "signals_today.csv")
    eq_df   = load_csv(cfg["data_dir"] / cfg.get("equity_file", "equity_curve.csv"))

    equity   = float(state.get("paper_equity_usdt", INITIAL_CAP))
    peak     = float(state.get("peak_equity_usdt", INITIAL_CAP))
    dd       = float(state.get("drawdown_pct", 0.0))
    ret_pct  = (equity / INITIAL_CAP - 1.0) * 100.0
    kill_sw  = state.get("kill_switch_triggered", False)
    n_closed = int(state.get("closed_trade_count", 0))

    freeze       = cfg["freeze"]
    days_running = max((today - freeze).days, 0)
    review_date  = freeze + timedelta(days=90)
    days_to_rev  = (review_date - today).days

    # Open positions from state.json (has initial_risk_per_unit, state.json is canonical)
    positions = state.get("open_positions", [])
    pos_list  = []
    for p in positions:
        ep   = float(p.get("entry_price", 0))
        ru   = float(p.get(cfg["pos_risk"], p.get("initial_risk_per_unit", 0)))
        sym  = str(p.get("symbol", "?"))
        held = int(p.get("bars_held", 0))
        rem  = int(p.get("bars_remaining", max(0, 20 - held)))
        pos_list.append({
            "symbol":     sym,
            "entry_date": str(p.get("entry_date", "?")),
            "entry_price":ep,
            "risk_unit":  ru,
            "cur_r":      cur_r(ep, ru, sym),
            "bars_held":  held,
            "bars_remaining": rem,
            "max_r":      float(p.get("max_favorable_r", 0)),
            "chandelier": bool(p.get("chandelier_active", False)),
        })

    # Signal list
    sig_list = []
    if not sig_df.empty:
        for _, row in sig_df.iterrows():
            sym  = str(row.get("symbol", "?"))
            info = {"symbol": sym}
            if "rsi" in row:
                info["detail"] = f"RSI={row['rsi']:.1f}"
            elif "consec" in row:
                info["detail"] = f"consec={int(row['consec'])}"
            elif "don_entry_high" in row:
                info["detail"] = f"breakout={row.get('don_entry_high','?'):.6g}"
            else:
                info["detail"] = ""
            sig_list.append(info)

    # Trade stats from equity curve
    avg_r = total_r = win_rate = 0.0
    if not eq_df.empty and "gross_r" in eq_df.columns:
        rs       = pd.to_numeric(eq_df["gross_r"], errors="coerce").dropna()
        n_trades = len(rs)
        avg_r    = float(rs.mean()) if n_trades else 0.0
        total_r  = float(rs.sum())
        win_rate = float((rs > 0).mean() * 100) if n_trades else 0.0

    return {
        "name":         cfg["name"],
        "style":        cfg["style"],
        "num":          cfg["num"],
        "equity":       equity,
        "ret_pct":      ret_pct,
        "dd_pct":       abs(dd),
        "kill_sw":      kill_sw,
        "n_open":       len(pos_list),
        "n_closed":     n_closed,
        "positions":    pos_list,
        "signals":      sig_list,
        "days_running": days_running,
        "days_to_rev":  days_to_rev,
        "review_date":  str(review_date),
        "freeze":       str(freeze),
        "loaded":       bool(state),
        "avg_r":        avg_r,
        "total_r":      total_r,
        "win_rate":     win_rate,
    }


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def hr(ch: str = "=") -> None:
    print(ch * W)


def print_system(s: dict) -> None:
    print()
    print(f"SYSTEM {s['num']} -- {s['name']} ({s['style']})")
    print("-" * W)

    if not s["loaded"]:
        print("  (no state.json -- engine has not run yet)")
        return

    ks = "  [KILL-SWITCH]" if s["kill_sw"] else ""
    print(f"Equity: {fmt_money(s['equity'])}  "
          f"Return: {s['ret_pct']:+.2f}%  "
          f"Max DD: {s['dd_pct']:.2f}%{ks}")

    # Open positions
    n = s["n_open"]
    if n == 0:
        print(f"Open positions (0): none")
    else:
        print(f"Open positions ({n}):")
        for p in s["positions"]:
            r_str = fmt_r(p["cur_r"])
            # System-specific extra info
            if p["chandelier"]:
                extra = "  [Chandelier active]"
            elif p.get("bars_remaining", 0) > 0:
                extra = f"  [{p['bars_remaining']} bars left]"
            else:
                extra = ""
            print(f"  {p['symbol']:<16} entry {p['entry_price']:.6g}  "
                  f"cur_R {r_str}{extra}")

    # Signals
    if not s["signals"]:
        print("Signals today: none")
    else:
        sigs = "  |  ".join(
            f"{sg['symbol']} ({sg['detail']})" if sg['detail'] else sg['symbol']
            for sg in s["signals"]
        )
        print(f"Signals today ({len(s['signals'])}): {sigs}")


def print_combined(snaps: list[dict], today: date) -> None:
    print()
    hr()
    print("COMBINED PORTFOLIO")
    print("-" * W)

    n_loaded     = sum(1 for s in snaps if s["loaded"])
    total_eq     = sum(s["equity"]  for s in snaps if s["loaded"])
    total_cap    = INITIAL_CAP * n_loaded
    combined_ret = (total_eq / total_cap - 1.0) * 100.0 if total_cap > 0 else 0.0
    total_open   = sum(s["n_open"]  for s in snaps if s["loaded"])
    total_closed = sum(s["n_closed"] for s in snaps if s["loaded"])

    # Days running from earliest freeze
    earliest_freeze = min(
        (date.fromisoformat(s["freeze"]) for s in snaps if s["loaded"]),
        default=today,
    )
    days_running = max((today - earliest_freeze).days, 0)

    print(f"Total deployed capital: {fmt_money(total_cap)}  "
          f"({n_loaded} systems x {fmt_money(INITIAL_CAP)})")
    print(f"Combined return: {combined_ret:+.2f}%  "
          f"(total equity: {fmt_money(total_eq)})")
    print(f"Active positions: {total_open} open  |  {total_closed} closed total")
    print(f"Days running: {days_running}  (since {earliest_freeze})")

    # Kill-switch alert
    kill_systems = [s["name"] for s in snaps if s["kill_sw"]]
    if kill_systems:
        print()
        print(f"  *** KILL-SWITCH ACTIVE: {', '.join(kill_systems)} ***")

    # Trade performance (only if any closed trades exist)
    all_eq = [load_csv(ROOT / "data" / d / f)
              for d, f in [("t9b_paper", "engine_equity_curve.csv"),
                           ("t9b_mr_paper", "engine_equity_curve.csv"),
                           ("t9b_consecdowndays_paper", "engine_equity_curve.csv")]]
    non_empty = [df for df in all_eq if not df.empty]
    if non_empty:
        combined_eq = pd.concat(non_empty, ignore_index=True)
        if "gross_r" in combined_eq.columns:
            rs = pd.to_numeric(combined_eq["gross_r"], errors="coerce").dropna()
            if len(rs) > 0:
                print(f"Trade stats (all systems): "
                      f"n={len(rs)}  avg_R={rs.mean():+.3f}  "
                      f"win={float((rs>0).mean()*100):.1f}%")


def print_header(snaps: list[dict], today: date) -> None:
    hr()
    # Earliest upcoming review
    loaded = [s for s in snaps if s["loaded"]]
    if loaded:
        min_snap = min(loaded, key=lambda s: s["days_to_rev"])
        rev_str  = (f"{min_snap['days_to_rev']} days until 3-month review "
                    f"({min_snap['review_date']})")
    else:
        rev_str = "no state loaded"

    print(f"T9B PORTFOLIO SUMMARY -- {today}")
    print(rev_str)
    hr()


# =============================================================================
# JSON OUTPUT
# =============================================================================

def print_json(snaps: list[dict], today: date) -> None:
    total_eq  = sum(s["equity"] for s in snaps if s["loaded"])
    total_cap = INITIAL_CAP * sum(1 for s in snaps if s["loaded"])
    out = {
        "date":           str(today),
        "systems":        [
            {
                "name":       s["name"],
                "equity":     round(s["equity"], 4),
                "return_pct": round(s["ret_pct"], 4),
                "dd_pct":     round(s["dd_pct"], 4),
                "n_open":     s["n_open"],
                "n_closed":   s["n_closed"],
                "kill_sw":    s["kill_sw"],
                "days_running": s["days_running"],
                "signals_today": len(s["signals"]),
            }
            for s in snaps
        ],
        "combined_equity":     round(total_eq, 4),
        "combined_return_pct": round((total_eq / total_cap - 1.0) * 100.0, 4) if total_cap else 0.0,
        "total_open":     sum(s["n_open"] for s in snaps if s["loaded"]),
    }
    print(json.dumps(out, indent=2))


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T9B combined portfolio summary")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    return p.parse_args()


def main() -> int:
    args  = parse_args()
    today = date.today()
    snaps = [system_snapshot(cfg, today) for cfg in SYSTEMS]

    if args.json:
        print_json(snaps, today)
        return 0

    print_header(snaps, today)
    for s in snaps:
        print_system(s)
    print_combined(snaps, today)
    print()
    hr()
    print("[PAPER ONLY] No real orders were placed.")
    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
