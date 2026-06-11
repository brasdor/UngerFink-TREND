#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T6 + T7 -- MeanReversionRSI Capital Execution Engine + Asset Robustness
UngerFink Pipeline / Andrea Unger Methodology

T6: Simulates real capital allocation on max10 trades.
T7: Re-runs T6 with top-1 asset (HBAR) removed.

Input  : data/research_meanreversionrsi_t5_1d/phase_t5_trades_max10.csv
Output : data/research_meanreversionrsi_t6_1d/
         data/research_meanreversionrsi_t7_1d/

Capital config:
  Starting capital : $10,000
  Risk per trade   : 0.25% of current equity
  Leverage         : 1.0 (Binance Spot)
  Kill-switch      : halt if equity < peak * (1 - 0.35)

Benchmark (DonchianLong):
  CAGR +16.0%  max_dd -4.4%  avg_r +1.511R

Note: MR and Donchian are uncorrelated -- lower CAGR expected
      but lower DD and negative correlation improve portfolio Sharpe.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"


def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


# =============================================================================
# CONFIG
# =============================================================================

ROOT       = Path(__file__).resolve().parents[1]
T5_DIR     = ROOT / "data" / "research_meanreversionrsi_t5_1d"
OUT_T6     = ROOT / "data" / "research_meanreversionrsi_t6_1d"
OUT_T7     = ROOT / "data" / "research_meanreversionrsi_t7_1d"
OUT_T6.mkdir(parents=True, exist_ok=True)
OUT_T7.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL  = 10_000.0
RISK_PCT         = 0.0025          # 0.25% risk per trade
LEVERAGE         = 1.0
KILL_SWITCH_DD   = 0.35            # halt if equity drops 35% from peak

BENCHMARK = {
    "label":   "DonchianLong",
    "cagr":    0.160,
    "max_dd":  -0.044,
    "avg_r":   1.511,
}

MR_GATES = {"avg_r_min": 0.10, "pf_min": 1.0}


# =============================================================================
# CAPITAL SIMULATION
# =============================================================================

def simulate_capital(trades_df: pd.DataFrame,
                     label: str) -> dict:
    """
    Simulate equity curve using fixed fractional risk sizing.
    Returns a results dict with equity_curve, stats, kill_switch info.
    """
    if trades_df.empty:
        return {}

    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    n  = len(df)

    equity        = INITIAL_CAPITAL
    peak_equity   = INITIAL_CAPITAL
    equity_curve  = [INITIAL_CAPITAL]
    entry_times   = []
    kill_fired    = False
    kill_at_trade = None
    kill_at_eq    = None
    accepted_i    = 0

    pnl_list   = []
    trade_data = []

    for i, row in df.iterrows():
        if kill_fired:
            break

        risk_amount = equity * RISK_PCT
        pnl_usd     = row["net_r"] * risk_amount
        equity     += pnl_usd

        peak_equity = max(peak_equity, equity)

        dd_from_peak = (peak_equity - equity) / peak_equity
        if dd_from_peak >= KILL_SWITCH_DD:
            kill_fired    = True
            kill_at_trade = i
            kill_at_eq    = equity

        pnl_list.append(pnl_usd)
        equity_curve.append(equity)
        entry_times.append(row["entry_time"])
        accepted_i += 1

        trade_data.append({
            "trade_idx":    i,
            "symbol":       row["symbol"],
            "entry_time":   row["entry_time"],
            "exit_time":    row["exit_time"],
            "net_r":        row["net_r"],
            "pnl_usd":      round(pnl_usd, 2),
            "equity_after": round(equity, 2),
        })

    # Date range for CAGR
    first_entry = df["entry_time"].iloc[0]
    last_exit   = df["exit_time"].iloc[accepted_i - 1] if accepted_i > 0 else first_entry
    years       = max((last_exit - first_entry).days / 365.25, 0.01)

    final_equity = equity
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    cagr         = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1

    # Max drawdown in $ and %
    eq_arr       = np.array(equity_curve)
    peak_arr     = np.maximum.accumulate(eq_arr)
    dd_arr       = (peak_arr - eq_arr) / peak_arr
    max_dd_pct   = float(np.max(dd_arr))
    max_dd_usd   = float(np.max(peak_arr - eq_arr))

    # Per-trade metrics
    rs  = np.array([t["net_r"] for t in trade_data])
    wrs = rs[rs > 0]
    lss = np.abs(rs[rs < 0])
    pf  = wrs.sum() / lss.sum() if lss.sum() > 0 else 99.0
    win_rate = len(wrs) / len(rs) if len(rs) > 0 else 0.0
    avg_r    = float(np.mean(rs)) if len(rs) > 0 else 0.0

    # Year-by-year USD P&L
    trade_df = pd.DataFrame(trade_data)
    if not trade_df.empty:
        trade_df["year"] = pd.to_datetime(trade_df["entry_time"]).dt.year
        year_pnl = (trade_df.groupby("year")
                   .agg(n=("pnl_usd", "count"),
                        pnl_usd=("pnl_usd", "sum"),
                        avg_r=("net_r", "mean"),
                        win_rate=("net_r", lambda x: (x > 0).mean()))
                   .reset_index())
    else:
        year_pnl = pd.DataFrame()

    # Equity shape: compute monthly drawdown depth as proxy for smoothness
    eq_arr_sm  = np.array(equity_curve)
    slope_sign = np.sign(np.diff(eq_arr_sm))
    pct_up     = float((slope_sign > 0).mean())

    return {
        "label":          label,
        "n_trades":       accepted_i,
        "n_input":        n,
        "kill_fired":     kill_fired,
        "kill_at_trade":  kill_at_trade,
        "kill_at_equity": kill_at_eq,
        "initial_eq":     INITIAL_CAPITAL,
        "final_equity":   round(final_equity, 2),
        "total_return":   round(total_return * 100, 2),
        "cagr":           round(cagr * 100, 2),
        "years":          round(years, 2),
        "max_dd_pct":     round(max_dd_pct * 100, 2),
        "max_dd_usd":     round(max_dd_usd, 2),
        "avg_r":          round(avg_r, 4),
        "win_rate":       round(win_rate, 4),
        "pf":             round(pf, 2),
        "pct_up_trades":  round(pct_up * 100, 1),
        "year_pnl":       year_pnl,
        "trade_df":       trade_df,
        "equity_curve":   equity_curve,
    }


# =============================================================================
# PRINT SCORECARD
# =============================================================================

def print_scorecard(r: dict, out_dir: Path, benchmark: dict | None = None) -> None:
    p()
    p("=" * 65)
    p(f"  T6 SCORECARD -- {r['label']}")
    p("=" * 65)
    p(f"  Trades simulated : {r['n_trades']} / {r['n_input']}")
    if r["kill_fired"]:
        p(f"  Kill-switch      : FIRED at trade {r['kill_at_trade']} (equity=${r['kill_at_equity']:,.0f})")
    else:
        p("  Kill-switch      : NOT fired")
    p()
    p(f"  Initial capital  : ${r['initial_eq']:>10,.2f}")
    p(f"  Final equity     : ${r['final_equity']:>10,.2f}")
    p(f"  Total return     : {r['total_return']:>+8.1f}%")
    p(f"  CAGR             : {r['cagr']:>+8.2f}%  (over {r['years']:.1f} years)")
    p()
    p(f"  Max DD (%)       : {r['max_dd_pct']:>8.2f}%")
    p(f"  Max DD ($)       : ${r['max_dd_usd']:>9,.2f}")
    p()
    p(f"  Avg R per trade  : {r['avg_r']:>+8.4f}R")
    p(f"  Win rate         : {r['win_rate']*100:>8.1f}%")
    p(f"  Profit factor    : {r['pf']:>8.2f}")
    p(f"  Equity direction : {r['pct_up_trades']:>8.1f}% of trades move equity up")

    if benchmark:
        p()
        p(f"  --- vs {benchmark['label']} benchmark ---")
        p(f"  CAGR   : {r['cagr']:>+6.2f}%  vs  {benchmark['cagr']*100:>+6.2f}%  "
          f"({'BETTER' if r['cagr'] > benchmark['cagr']*100 else 'LOWER'})")
        p(f"  Max DD : {r['max_dd_pct']:>+6.2f}%  vs  {benchmark['max_dd']*100:>+6.2f}%  "
          f"({'BETTER' if r['max_dd_pct'] < abs(benchmark['max_dd']*100) else 'HIGHER'})")
        p(f"  Note   : Lower CAGR expected; key value is negative correlation to Donchian")

    p()
    p("  Year-by-Year P&L:")
    p(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>7}  {'P&L ($)':>10}  {'Cumul ($)':>10}  Note")
    cumul = r["initial_eq"]
    for _, row in r["year_pnl"].iterrows():
        yr    = int(row["year"])
        cumul += row["pnl_usd"]
        note  = "<<< BEAR" if yr == 2022 else ("(partial)" if yr == 2026 else "")
        neg   = "  <<< WEAK" if row["pnl_usd"] < 0 else ""
        p(f"  {yr:>5}  {int(row['n']):>5}  {row['win_rate']*100:>5.1f}%  "
          f"{row['avg_r']:>+6.3f}R  ${row['pnl_usd']:>+9,.2f}  "
          f"${cumul:>9,.2f}  {note}{neg}")

    # Gate checks
    p()
    p("  Gate checks:")
    avgr_ok = r["avg_r"] >= MR_GATES["avg_r_min"]
    pf_ok   = r["pf"]    >= MR_GATES["pf_min"]
    kill_ok = not r["kill_fired"]
    p(f"    avg_r > 0.10R  : {'PASS' if avgr_ok else 'FAIL'}  ({r['avg_r']:+.4f}R)")
    p(f"    PF > 1.0       : {'PASS' if pf_ok else 'FAIL'}  ({r['pf']:.2f})")
    p(f"    Kill-switch    : {'PASS (not fired)' if kill_ok else 'FAIL (fired!)'}")
    overall = "PASS" if (avgr_ok and pf_ok and kill_ok) else "FAIL"
    p(f"  OVERALL GATE : {overall}")

    # Save files
    r["trade_df"].to_csv(out_dir / "phase_t6_equity_trades.csv", index=False)
    r["year_pnl"].to_csv(out_dir / "phase_t6_year_pnl.csv", index=False)
    pd.DataFrame({"equity": r["equity_curve"]}).to_csv(
        out_dir / "phase_t6_equity_curve.csv", index=False)

    with open(out_dir / "phase_t6_scorecard.txt", "w", encoding="utf-8") as f:
        f.write(f"Phase T6 Scorecard -- {r['label']}\n")
        f.write(f"Config: rsi14/os25/time_exit=20/atr3/no_filter/1D  max10  $10k  0.25%risk\n\n")
        f.write(f"Initial capital : ${r['initial_eq']:,.2f}\n")
        f.write(f"Final equity    : ${r['final_equity']:,.2f}\n")
        f.write(f"Total return    : {r['total_return']:+.1f}%\n")
        f.write(f"CAGR            : {r['cagr']:+.2f}%\n")
        f.write(f"Max DD %        : {r['max_dd_pct']:.2f}%\n")
        f.write(f"Max DD $        : ${r['max_dd_usd']:,.2f}\n")
        f.write(f"Kill-switch     : {'FIRED' if r['kill_fired'] else 'not fired'}\n")
        f.write(f"Avg R           : {r['avg_r']:+.4f}R\n")
        f.write(f"Win rate        : {r['win_rate']*100:.1f}%\n")
        f.write(f"PF              : {r['pf']:.2f}\n\n")
        f.write("Year-by-Year P&L:\n")
        f.write(r["year_pnl"].to_string(index=False))
        f.write(f"\n\nOVERALL GATE: {overall}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 65)
    p("  Phase T6+T7 -- MeanReversionRSI Capital Engine + Asset Robustness")
    p(f"  Capital: ${INITIAL_CAPITAL:,.0f}  Risk: {RISK_PCT*100}%/trade  Leverage: {LEVERAGE}x")
    p(f"  Kill-switch: halt if equity drops {KILL_SWITCH_DD*100:.0f}% from peak")
    p("=" * 65)

    # Load max10 trades
    trades_path = T5_DIR / "phase_t5_trades_max10.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found. Run T5 first.")
        sys.exit(1)

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    p(f"  Loaded {len(df)} trades (max10 cap)")

    # =========================================================================
    # T6 -- full max10 universe
    # =========================================================================
    p("\n  Running T6 (full max10 universe)...")
    r_t6 = simulate_capital(df, "MR-1D max10 (full)")
    print_scorecard(r_t6, OUT_T6, BENCHMARK)

    # =========================================================================
    # T7 -- remove HBAR_USDT (top-1 asset by total R)
    # =========================================================================
    p()
    p("=" * 65)
    p("  Phase T7 -- Asset Robustness: Remove HBAR_USDT (top-1)")
    p("=" * 65)

    # Re-apply max10 cap on HBAR-removed trades
    # Load original T3MR Variant E trades, remove HBAR, re-cap
    t3mr_path = ROOT / "data" / "research_meanreversionrsi_t3mr_1d" / "phase_t3mr_trades_E.csv"
    df_full = pd.read_csv(t3mr_path)
    df_full["entry_time"] = pd.to_datetime(df_full["entry_time"], utc=True)
    df_full["exit_time"]  = pd.to_datetime(df_full["exit_time"],  utc=True)

    df_no_hbar = df_full[df_full["symbol"] != "HBAR_USDT"].copy()
    p(f"  Trades after removing HBAR: {len(df_no_hbar)} (from {len(df_full)})")

    # Re-apply max10 cap
    from phase_t5_meanreversionrsi_portfolio_filter import apply_cap
    df_t7 = apply_cap(df_no_hbar, 10)
    df_t7 = df_t7.sort_values("entry_time").reset_index(drop=True)
    p(f"  After max10 re-cap: {len(df_t7)} trades")

    r_t7 = simulate_capital(df_t7, "MR-1D max10 (no HBAR)")
    print_scorecard(r_t7, OUT_T7)

    # =========================================================================
    # SIDE-BY-SIDE SCORECARD
    # =========================================================================
    p()
    p("=" * 65)
    p("  T6 vs T7 SIDE-BY-SIDE")
    p("=" * 65)
    p(f"  {'Metric':<22} {'T6 (full)':>15} {'T7 (no HBAR)':>15}  {'Delta':>10}")
    p("  " + "-" * 65)

    def row(label, v6, v7, fmt=".2f", suffix=""):
        delta = v7 - v6
        p(f"  {label:<22} {f'{v6:{fmt}}{suffix}':>15} {f'{v7:{fmt}}{suffix}':>15}  "
          f"{f'{delta:+{fmt}}{suffix}':>10}")

    row("Trades",          r_t6["n_trades"],    r_t7["n_trades"],    fmt="d")
    row("Final equity ($)", r_t6["final_equity"], r_t7["final_equity"], fmt=",.2f")
    row("Total return (%)", r_t6["total_return"], r_t7["total_return"], fmt=".1f", suffix="%")
    row("CAGR (%)",         r_t6["cagr"],          r_t7["cagr"],          fmt=".2f", suffix="%")
    row("Max DD (%)",        r_t6["max_dd_pct"],    r_t7["max_dd_pct"],    fmt=".2f", suffix="%")
    row("Max DD ($)",        r_t6["max_dd_usd"],    r_t7["max_dd_usd"],    fmt=",.2f")
    row("Avg R",             r_t6["avg_r"],         r_t7["avg_r"],         fmt=".4f", suffix="R")
    row("Profit factor",     r_t6["pf"],            r_t7["pf"],            fmt=".2f")
    row("Win rate (%)",      r_t6["win_rate"]*100,  r_t7["win_rate"]*100,  fmt=".1f", suffix="%")
    p("  " + "-" * 65)

    t6_pass = (not r_t6["kill_fired"] and
               r_t6["avg_r"] >= MR_GATES["avg_r_min"] and
               r_t6["pf"]    >= MR_GATES["pf_min"])
    t7_pass = (not r_t7["kill_fired"] and
               r_t7["avg_r"] >= MR_GATES["avg_r_min"] and
               r_t7["pf"]    >= MR_GATES["pf_min"])

    p(f"  T6 gate : {'PASS' if t6_pass else 'FAIL'}")
    p(f"  T7 gate : {'PASS' if t7_pass else 'FAIL'}")
    p()
    if t6_pass and t7_pass:
        p("  Both T6 and T7 PASS.")
        p("  System is ready for T8 config freeze.")
        p("  Run: python phase_t8_meanreversionrsi_config_freeze.py")
    else:
        p("  One or more gates FAILED -- do not proceed to T8.")

    # Save combined scorecard
    with open(OUT_T6 / "phase_t6t7_combined_scorecard.txt", "w", encoding="utf-8") as f:
        f.write("T6+T7 Combined Scorecard\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"T6 (full max10):   CAGR={r_t6['cagr']:+.2f}%  "
                f"DD={r_t6['max_dd_pct']:.2f}%  gate={'PASS' if t6_pass else 'FAIL'}\n")
        f.write(f"T7 (no HBAR):      CAGR={r_t7['cagr']:+.2f}%  "
                f"DD={r_t7['max_dd_pct']:.2f}%  gate={'PASS' if t7_pass else 'FAIL'}\n\n")
        f.write(f"DonchianLong ref:  CAGR=+{BENCHMARK['cagr']*100:.1f}%  "
                f"DD={BENCHMARK['max_dd']*100:.1f}%\n\n")
        f.write("Note: MR and Donchian are structurally uncorrelated.\n"
                "Portfolio combination improves Sharpe ratio.\n")

    sys.exit(0 if (t6_pass and t7_pass) else 1)


if __name__ == "__main__":
    main()
