#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T6 -- DualMA BTC-only Capital Simulation

Replays Phase T5 BTC trades as a compounding capital simulation.

Capital model (fixed-fractional risk):
    risk_amount  = RISK_PER_TRADE_PCT * equity_before_entry
    pnl_usdt     = net_r * risk_amount
    equity_after = equity_before + pnl_usdt

Binance Spot, no leverage, LONG only.  No margin complications.

Input:  data/research_dualma_t5/phase_t5_portfolio_trades.csv
Output: data/research_dualma_t6/
    phase_t6_trade_log.csv
    phase_t6_equity_curve.csv
    phase_t6_variant_summary.csv
    phase_t6_master_report.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_TRADES    = PROJECT_ROOT / "data" / "research_dualma_t5" / "phase_t5_portfolio_trades.csv"
OUT_DIR         = PROJECT_ROOT / "data" / "research_dualma_t6"

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE  = 0.0025     # 0.25%
LEVERAGE        = 1.0
STOP_DD_PCT     = 35.0       # kill-switch: halt new entries if closed-equity DD > 35%


# =============================================================================
# HELPERS
# =============================================================================

def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak * 100).min())


def profit_factor(pnl: np.ndarray) -> float:
    gains  = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def cagr(start: float, end: float, days: float) -> float:
    if days <= 0 or start <= 0:
        return 0.0
    years = days / 365.25
    return (end / start) ** (1.0 / years) - 1.0


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE T6 -- DualMA BTC-only Capital Simulation")
    print("=" * 70)

    if not INPUT_TRADES.exists():
        print(f"[ERROR] T5 trades not found: {INPUT_TRADES}")
        return 1

    df = pd.read_csv(INPUT_TRADES)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True, errors="coerce")
    df["net_r"]      = pd.to_numeric(df["net_r"], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time", "net_r"]).sort_values("entry_time").reset_index(drop=True)

    print(f"Trades loaded: {len(df)}")
    print(f"Capital start: ${INITIAL_CAPITAL:,.2f}")
    print(f"Risk/trade:    {RISK_PER_TRADE*100:.2f}%")
    print()

    equity     = INITIAL_CAPITAL
    peak       = INITIAL_CAPITAL
    kill_active = False
    trade_log  = []
    eq_curve   = [{"date": str(df["entry_time"].iloc[0].date()), "equity": equity, "event": "start"}]

    for _, row in df.iterrows():
        net_r      = float(row["net_r"])
        entry_time = row["entry_time"]
        exit_time  = row["exit_time"]

        # Kill-switch check
        current_dd_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0
        if current_dd_pct <= -STOP_DD_PCT and not kill_active:
            kill_active = True
            print(f"  !! KILL-SWITCH at {exit_time.date()}: DD={current_dd_pct:.1f}%")

        risk_amount = RISK_PER_TRADE * equity
        pnl         = net_r * risk_amount
        equity_after = equity + pnl

        trade_log.append({
            "symbol":        row.get("symbol", "BTC/USDT"),
            "entry_time":    str(entry_time),
            "exit_time":     str(exit_time),
            "exit_reason":   row.get("exit_reason", ""),
            "chandelier_active": row.get("chandelier_active", ""),
            "net_r":         round(net_r, 6),
            "risk_amount":   round(risk_amount, 4),
            "pnl_usdt":      round(pnl, 4),
            "equity_before": round(equity, 4),
            "equity_after":  round(equity_after, 4),
            "peak_equity":   round(peak, 4),
            "dd_pct":        round(current_dd_pct, 4),
            "kill_switch":   kill_active,
        })

        equity = equity_after
        if equity > peak:
            peak = equity

        eq_curve.append({
            "date":   str(exit_time.date()),
            "equity": round(equity, 4),
            "event":  "trade_close",
        })

    df_log      = pd.DataFrame(trade_log)
    df_eq_curve = pd.DataFrame(eq_curve)

    # Summary stats
    pnl_series   = df_log["pnl_usdt"].to_numpy()
    net_r_series = df_log["net_r"].to_numpy()
    eq_series    = df_log["equity_after"].to_numpy()
    n            = len(pnl_series)

    final_equity = float(eq_series[-1]) if n > 0 else INITIAL_CAPITAL
    total_return_pct = (final_equity / INITIAL_CAPITAL - 1) * 100

    date_start = df["entry_time"].min()
    date_end   = df["exit_time"].max()
    total_days = (date_end - date_start).days

    cagr_pct    = cagr(INITIAL_CAPITAL, final_equity, total_days) * 100
    max_dd      = max_drawdown_pct(np.concatenate([[INITIAL_CAPITAL], eq_series]))
    pf_usdt     = profit_factor(pnl_series)
    win_pct     = float((net_r_series > 0).mean() * 100) if n > 0 else 0.0
    total_r     = float(net_r_series.sum())
    avg_r       = float(net_r_series.mean()) if n > 0 else 0.0
    avg_win_pnl = float(pnl_series[pnl_series > 0].mean()) if (pnl_series > 0).any() else 0.0
    avg_los_pnl = float(pnl_series[pnl_series < 0].mean()) if (pnl_series < 0).any() else 0.0
    expectancy  = avg_win_pnl * (win_pct / 100) + avg_los_pnl * (1 - win_pct / 100)

    summary = {
        "variant":            "DualMA_BTC_max1",
        "asset":              "BTC/USDT",
        "initial_capital":    INITIAL_CAPITAL,
        "final_equity":       round(final_equity, 2),
        "total_return_pct":   round(total_return_pct, 4),
        "cagr_pct":           round(cagr_pct, 4),
        "max_dd_pct":         round(max_dd, 4),
        "n_trades":           n,
        "total_r":            round(total_r, 4),
        "avg_r":              round(avg_r, 4),
        "win_rate_pct":       round(win_pct, 2),
        "profit_factor":      round(pf_usdt, 4),
        "avg_win_pnl":        round(avg_win_pnl, 4),
        "avg_los_pnl":        round(avg_los_pnl, 4),
        "expectancy_usdt":    round(expectancy, 4),
        "date_start":         str(date_start),
        "date_end":           str(date_end),
        "total_days":         total_days,
        "kill_switch_fired":  kill_active,
    }

    df_log.to_csv(OUT_DIR / "phase_t6_trade_log.csv", index=False)
    df_eq_curve.to_csv(OUT_DIR / "phase_t6_equity_curve.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "phase_t6_variant_summary.csv", index=False)

    # Master report
    report_lines = [
        "=" * 70,
        "PHASE T6 -- DualMA BTC-only Capital Simulation Report",
        "=" * 70,
        f"Asset:           BTC/USDT",
        f"Variant:         DualMA_BTC_max1",
        f"Initial capital: ${INITIAL_CAPITAL:,.2f}",
        f"Risk per trade:  {RISK_PER_TRADE*100:.2f}%",
        f"Leverage:        {LEVERAGE}x  (Binance Spot, no margin)",
        f"Date range:      {date_start.date()} -> {date_end.date()}  ({total_days} days)",
        "",
        "CAPITAL RESULTS",
        "-" * 70,
        f"  Final equity:       ${final_equity:,.2f}",
        f"  Total return:       {total_return_pct:.2f}%",
        f"  CAGR:               {cagr_pct:.2f}%",
        f"  Max DD (closed eq): {max_dd:.2f}%",
        "",
        "TRADE STATISTICS",
        "-" * 70,
        f"  Trades:             {n}",
        f"  Total R:            {total_r:.4f}R",
        f"  Avg R:              {avg_r:.4f}R",
        f"  Win rate:           {win_pct:.1f}%",
        f"  Profit factor:      {pf_usdt:.4f}",
        f"  Avg win (USDT):     ${avg_win_pnl:.2f}",
        f"  Avg loss (USDT):    ${avg_los_pnl:.2f}",
        f"  Expectancy/trade:   ${expectancy:.2f}",
        f"  Kill-switch fired:  {kill_active}",
        "",
        "COMPARISON vs DONCHIAN T6 BENCHMARK",
        "-" * 70,
        "  Donchian max3: +19.43% return, -1.78% DD",
        f"  DualMA BTC:    {total_return_pct:+.2f}% return, {max_dd:.2f}% DD",
        "",
        "GATE CHECKS",
        "-" * 70,
        f"  CAGR > 0:           {'PASS' if cagr_pct > 0 else 'FAIL'}",
        f"  Max DD < 25%:       {'PASS' if abs(max_dd) < 25 else 'WARN'} ({max_dd:.2f}%)",
        f"  PF > 1.5:           {'PASS' if pf_usdt > 1.5 else 'WARN'} ({pf_usdt:.4f})",
        "",
        "T7: SKIPPED -- single-asset system; remove-best-asset test is N/A.",
        "Proceed to T8 (config freeze) if capital results are satisfactory.",
        "=" * 70,
    ]
    report_text = "\n".join(report_lines)
    (OUT_DIR / "phase_t6_master_report.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
