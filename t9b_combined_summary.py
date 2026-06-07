#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9B COMBINED SUMMARY
====================

Reads state.json + output CSVs for all four T9B paper engines and prints a
compact, phone-readable combined summary.

Designed to be run after all engines have completed in GitHub Actions.

Systems:
  1. DonchianLong_UniverseV2_ExitV2   (data/t9b_paper/)
  2. MeanReversionRSI 1D              (data/t9b_mr_paper/)
  3. ConsecDownDaysMR 1D              (data/t9b_consecdowndays_paper/)
  4. MomentumFactor Lb20 L20/S10      (data/t9b_momentum_paper/)

ASCII only in all print statements.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).resolve().parent
INITIAL_CAP   = 10_000.0
EPS           = 1e-12

SYSTEMS = [
    {
        "name":       "Donchian",
        "label":      "DonchianLong_UniverseV2_ExitV2",
        "data_dir":   ROOT / "data" / "t9b_paper",
        "freeze":     date(2026, 5, 30),
        "signal_col": "close",
        "signal_key": "symbol",
        "type":       "trend",
    },
    {
        "name":       "RSI_MR",
        "label":      "MeanReversionRSI 1D",
        "data_dir":   ROOT / "data" / "t9b_mr_paper",
        "freeze":     date(2026, 6, 1),
        "signal_col": "close",
        "signal_key": "symbol",
        "type":       "trend",
    },
    {
        "name":       "ConsecDown",
        "label":      "ConsecDownDaysMR 1D",
        "data_dir":   ROOT / "data" / "t9b_consecdowndays_paper",
        "freeze":     date(2026, 6, 2),
        "signal_col": "close",
        "signal_key": "symbol",
        "type":       "trend",
    },
    {
        "name":       "Momentum",
        "label":      "MomentumFactor Lb20 L20/S10 Biweekly",
        "data_dir":   ROOT / "data" / "t9b_momentum_paper",
        "freeze":     date(2026, 6, 7),
        "type":       "momentum",
    },
]


def load_state(data_dir: Path) -> dict:
    path = data_dir / "state.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def system_snapshot(sys_cfg: dict, today: date) -> dict:
    state    = load_state(sys_cfg["data_dir"])
    eq_df    = load_csv(sys_cfg["data_dir"] / "equity_curve.csv")
    sig_df   = load_csv(sys_cfg["data_dir"] / "signals_today.csv")
    open_df  = load_csv(sys_cfg["data_dir"] / "open_positions.csv")

    is_momentum = sys_cfg.get("type") == "momentum"

    eq       = float(state.get("paper_equity_usdt", INITIAL_CAP))
    peak     = float(state.get("peak_equity_usdt", INITIAL_CAP))
    dd       = float(state.get("drawdown_pct", 0.0))
    ret_pct  = (eq / INITIAL_CAP - 1.0) * 100.0
    kill_sw  = state.get("kill_switch_triggered", False)
    last_run = state.get("last_run_date", "n/a")

    if is_momentum:
        n_open   = (len(state.get("long_positions", [])) +
                    len(state.get("short_positions", [])))
        n_closed = 0
    else:
        n_open   = len(state.get("open_positions", []))
        n_closed = int(state.get("closed_trade_count", 0))

    freeze        = sys_cfg["freeze"]
    days_running  = max((today - freeze).days, 0)
    review_date   = freeze + timedelta(days=90)
    days_to_rev   = (review_date - today).days

    # Trade stats (trend systems only)
    avg_r    = total_r = win_rate = 0.0
    n_trades = 0
    if not is_momentum and not eq_df.empty and "gross_r" in eq_df.columns:
        r_vals   = pd.to_numeric(eq_df["gross_r"], errors="coerce").dropna()
        n_trades = len(r_vals)
        avg_r    = float(r_vals.mean()) if n_trades > 0 else 0.0
        total_r  = float(r_vals.sum())
        win_rate = float((r_vals > 0).mean() * 100) if n_trades > 0 else 0.0

    # Momentum-specific extras
    n_rebal       = int(state.get("total_rebal_count", 0)) if is_momentum else 0
    last_rebal    = state.get("last_rebal_date", "n/a") if is_momentum else None
    total_unreal  = 0.0
    if is_momentum and not open_df.empty and "unrealized_pnl" in open_df.columns:
        total_unreal = float(pd.to_numeric(open_df["unrealized_pnl"], errors="coerce").sum())

    return {
        "name":          sys_cfg["name"],
        "label":         sys_cfg["label"],
        "type":          sys_cfg.get("type", "trend"),
        "equity":        eq,
        "peak":          peak,
        "ret_pct":       ret_pct,
        "dd_pct":        dd,
        "kill_sw":       kill_sw,
        "n_open":        n_open,
        "n_closed":      n_closed,
        "last_run":      last_run,
        "days_running":  days_running,
        "days_to_rev":   days_to_rev,
        "review_date":   str(review_date),
        "n_signals":     len(sig_df) if not sig_df.empty else 0,
        "signals":       sig_df,
        "open_pos":      open_df,
        "avg_r":         avg_r,
        "total_r":       total_r,
        "win_rate":      win_rate,
        "n_trades":      n_trades,
        "state_loaded":  bool(state),
        "is_momentum":   is_momentum,
        "n_rebal":       n_rebal,
        "last_rebal":    last_rebal,
        "total_unreal":  total_unreal,
    }


def main() -> int:
    today = date.today()

    print("=" * 60)
    print("T9B COMBINED SUMMARY -- ALL SYSTEMS")
    print(f"Date: {today}")
    print("=" * 60)

    snapshots = []
    for sys_cfg in SYSTEMS:
        snap = system_snapshot(sys_cfg, today)
        snapshots.append(snap)

    # ----------------------------------------------------------------
    # Section 1: Equity comparison table
    # ----------------------------------------------------------------
    print()
    print("EQUITY OVERVIEW")
    print(f"  {'System':<14} {'Equity':>10}  {'Return':>8}  {'DD':>7}  {'Open':>5}  {'Closed':>7}  {'Days':>5}  {'KillSw'}")
    print(f"  {'-'*14} {'-'*10}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*6}")
    for s in snapshots:
        if not s["state_loaded"]:
            print(f"  {s['name']:<14}  (state not found -- engine may not have run yet)")
            continue
        ks = "YES" if s["kill_sw"] else "no"
        print(f"  {s['name']:<14} ${s['equity']:>9,.2f}  {s['ret_pct']:>+7.2f}%  "
              f"{s['dd_pct']:>+6.2f}%  {s['n_open']:>5}  {s['n_closed']:>7}  "
              f"{s['days_running']:>5}  {ks}")

    # Combined portfolio equity
    total_equity  = sum(s["equity"] for s in snapshots if s["state_loaded"])
    total_start   = INITIAL_CAP * sum(1 for s in snapshots if s["state_loaded"])
    combined_ret  = (total_equity / total_start - 1.0) * 100.0 if total_start > 0 else 0.0
    n_loaded      = sum(1 for s in snapshots if s["state_loaded"])
    if n_loaded > 1:
        print(f"  {'--- PORTFOLIO':<14} ${total_equity:>9,.2f}  {combined_ret:>+7.2f}%  "
              f"(${total_start:,.0f} deployed)")

    # ----------------------------------------------------------------
    # Section 2: Trade performance per system
    # ----------------------------------------------------------------
    print()
    print("TRADE PERFORMANCE (closed trades only)")
    print(f"  {'System':<14} {'Trades':>7}  {'WinRate':>8}  {'AvgR':>8}  {'TotalR':>8}")
    print(f"  {'-'*14} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in snapshots:
        if not s["state_loaded"]:
            continue
        if s["is_momentum"]:
            unreal_str = f"{s['total_unreal']:>+8.2f}" if s["total_unreal"] != 0.0 else "    $0.00"
            print(f"  {s['name']:<14}  rebalances={s['n_rebal']}  "
                  f"last={s['last_rebal']}  unrealized_pnl={unreal_str}")
        elif s["n_trades"] == 0:
            print(f"  {s['name']:<14}  (no closed trades yet)")
        else:
            print(f"  {s['name']:<14} {s['n_trades']:>7}  {s['win_rate']:>7.1f}%  "
                  f"{s['avg_r']:>+7.3f}R  {s['total_r']:>+7.2f}R")

    # ----------------------------------------------------------------
    # Section 3: Signals today (all systems)
    # ----------------------------------------------------------------
    total_signals = sum(s["n_signals"] for s in snapshots)
    print()
    print(f"SIGNALS TODAY ({total_signals} total across all systems)")
    any_signal = False
    for s in snapshots:
        sig_df = s["signals"]
        if sig_df.empty or s["n_signals"] == 0:
            continue
        any_signal = True
        print(f"\n  [{s['name']}] {s['n_signals']} signal(s):")
        if s["is_momentum"]:
            # signals_today.csv: symbol, side, action, rank, entry_price, quantity, notional, period_return_pct
            longs  = sig_df[sig_df.get("side", pd.Series()) == "LONG"]  if "side" in sig_df.columns else sig_df
            shorts = sig_df[sig_df.get("side", pd.Series()) == "SHORT"] if "side" in sig_df.columns else pd.DataFrame()
            if not longs.empty:
                print(f"    LONGS ({len(longs)}):")
                for _, row in longs.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    px   = row.get("entry_price", 0)
                    rk   = int(row.get("rank", 0))
                    ret  = row.get("period_return_pct", float("nan"))
                    ret_str = f"{ret:>+6.1f}%" if ret == ret else "   n/a"
                    print(f"      #{rk:<3} {sym:<16}  px={px:<12.6g}  20d_ret={ret_str}")
            if not shorts.empty:
                print(f"    SHORTS ({len(shorts)}):")
                for _, row in shorts.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    px   = row.get("entry_price", 0)
                    rk   = int(row.get("rank", 0))
                    ret  = row.get("period_return_pct", float("nan"))
                    ret_str = f"{ret:>+6.1f}%" if ret == ret else "   n/a"
                    print(f"      #{rk:<3} {sym:<16}  px={px:<12.6g}  20d_ret={ret_str}")
        else:
            for _, row in sig_df.iterrows():
                sym  = str(row.get("symbol", "?"))
                cls  = row.get("close", "?")
                stop = row.get("stop_loss", "?")
                if "rsi" in row:
                    extra = f"  RSI={row['rsi']:.1f}"
                elif "consec" in row:
                    extra = f"  consec={int(row['consec'])}"
                else:
                    extra = ""
                print(f"    {sym:<16}  close={cls:<12.6g}  stop={stop:<12.6g}{extra}")
    if not any_signal:
        print("  None today")

    # ----------------------------------------------------------------
    # Section 4: Open positions (all systems, compact)
    # ----------------------------------------------------------------
    total_open = sum(s["n_open"] for s in snapshots)
    print()
    print(f"OPEN POSITIONS ({total_open} total across all systems)")
    for s in snapshots:
        open_df = s["open_pos"]
        if open_df.empty or s["n_open"] == 0:
            continue
        print(f"\n  [{s['name']}] {s['n_open']} position(s):")
        if s["is_momentum"]:
            # open_positions.csv: symbol, side, entry_date, entry_price, current_price,
            #                     quantity, notional_usdt, unrealized_pnl, unrealized_pct, days_held
            longs  = open_df[open_df["side"] == "LONG"]  if "side" in open_df.columns else open_df
            shorts = open_df[open_df["side"] == "SHORT"] if "side" in open_df.columns else pd.DataFrame()
            if not longs.empty:
                print(f"    LONGS ({len(longs)}):")
                for _, row in longs.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    notl = float(row.get("notional_usdt", 0))
                    upnl = float(row.get("unrealized_pnl", 0))
                    upct = float(row.get("unrealized_pct", 0))
                    held = int(row.get("days_held", 0))
                    print(f"      {sym:<16}  notional=${notl:>8,.2f}  pnl={upnl:>+8.2f}  ({upct:>+5.2f}%)  d={held}")
            if not shorts.empty:
                print(f"    SHORTS ({len(shorts)}):")
                for _, row in shorts.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    notl = float(row.get("notional_usdt", 0))
                    upnl = float(row.get("unrealized_pnl", 0))
                    upct = float(row.get("unrealized_pct", 0))
                    held = int(row.get("days_held", 0))
                    print(f"      {sym:<16}  notional=${notl:>8,.2f}  pnl={upnl:>+8.2f}  ({upct:>+5.2f}%)  d={held}")
        else:
            for _, row in open_df.iterrows():
                sym   = str(row.get("symbol", "?"))
                edate = str(row.get("entry_date", "?"))
                held  = int(row.get("bars_held", 0))
                rem   = int(row.get("bars_remaining", 0))
                raw_stop = row.get("stop_loss", row.get("current_stop", None))
                try:
                    stop_str = f"{float(raw_stop):.6g}"
                except (TypeError, ValueError):
                    stop_str = str(raw_stop) if raw_stop is not None else "?"
                print(f"    {sym:<16}  entry={edate}  held={held}  rem={rem}  stop={stop_str}")

    # ----------------------------------------------------------------
    # Section 5: Review countdown
    # ----------------------------------------------------------------
    print()
    print("REVIEW COUNTDOWN (3-month live deployment eligibility)")
    for s in snapshots:
        if not s["state_loaded"]:
            continue
        if s["days_to_rev"] > 0:
            print(f"  {s['name']:<14}  {s['days_to_rev']:>3} days until review  ({s['review_date']})")
        else:
            print(f"  {s['name']:<14}  REVIEW DUE  ({s['review_date']})")

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    any_kill = any(s["kill_sw"] for s in snapshots if s["state_loaded"])
    any_warn = any(s["dd_pct"] < -20.0 for s in snapshots if s["state_loaded"])
    if any_kill:
        print("STATUS: KILL-SWITCH ACTIVE on one or more systems -- review required")
    elif any_warn:
        print("STATUS: WARN -- one or more systems have DD > 20%")
    else:
        print("STATUS: OK")
    print("[PAPER ONLY] No real orders were placed.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
