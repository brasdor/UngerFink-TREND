#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T5 -- MeanReversionRSI Portfolio Filter
UngerFink Pipeline / Andrea Unger Methodology

Simulates concurrent-position caps on the Variant E trade sequence.
Trades are processed in entry_time order; when cap is reached, new
signals are skipped until an existing position closes.

Caps tested: uncapped, max5, max10, max15, max20

Input : data/research_meanreversionrsi_t3mr_1d/phase_t3mr_trades_E.csv
Output: data/research_meanreversionrsi_t5_1d/

Selection criteria:
  - Minimum 100 accepted trades
  - avg_r > 0.10R
  - Prefer lower DD over higher total_r

Note: MR and Donchian Long are uncorrelated -- they can run on the SAME
capital account with separate position limits.
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

ROOT       = Path(__file__).parent
T3MR_DIR   = ROOT / "data" / "research_meanreversionrsi_t3mr_1d"
OUT_DIR    = ROOT / "data" / "research_meanreversionrsi_t5_1d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPS       = [None, 5, 10, 15, 20]   # None = uncapped
CAP_LABELS = ["uncapped", "max5", "max10", "max15", "max20"]

MR_GATES = {
    "min_trades":    100,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "pf_min":        1.0,
}


# =============================================================================
# METRICS
# =============================================================================

def metrics(rs: np.ndarray) -> dict:
    if len(rs) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0,
                "max_dd_r": 0.0, "total_r": 0.0}
    wins   = rs[rs > 0]
    losses = np.abs(rs[rs < 0])
    pf     = wins.sum() / losses.sum() if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    cum    = np.cumsum(rs)
    dd     = float(np.max(np.maximum.accumulate(cum) - cum))
    return {
        "n":        len(rs),
        "win_rate": float(len(wins) / len(rs)),
        "avg_r":    float(np.mean(rs)),
        "pf":       float(pf),
        "max_dd_r": dd,
        "total_r":  float(np.sum(rs)),
    }


def gate_str(m: dict) -> str:
    ok = (
        m["n"]        >= MR_GATES["min_trades"]  and
        MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"] and
        m["avg_r"]    >= MR_GATES["avg_r_min"]   and
        m["pf"]       >= MR_GATES["pf_min"]
    )
    return "PASS" if ok else "FAIL"


# =============================================================================
# PORTFOLIO CAP SIMULATION
# =============================================================================

def apply_cap(df: pd.DataFrame, max_concurrent: int | None) -> pd.DataFrame:
    """
    Process trades in chronological order.
    Accept a trade only if open position count < max_concurrent.
    An open position is one where entry_time <= now < exit_time.
    """
    if max_concurrent is None:
        return df.copy()

    df_sorted    = df.sort_values("entry_time").reset_index(drop=True)
    open_exits   = []   # list of exit_times for currently open positions
    accepted_idx = []

    for i, row in df_sorted.iterrows():
        entry_t = row["entry_time"]
        exit_t  = row["exit_time"]

        # Close out positions that have already exited
        open_exits = [et for et in open_exits if et > entry_t]

        # Accept if under cap
        if len(open_exits) < max_concurrent:
            open_exits.append(exit_t)
            accepted_idx.append(i)

    return df_sorted.loc[accepted_idx].reset_index(drop=True)


# =============================================================================
# YEAR-BY-YEAR HELPER
# =============================================================================

def year_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(df["year"].unique()):
        ydf = df[df["year"] == yr]
        m   = metrics(ydf["net_r"].values)
        rows.append({"year": int(yr), **m})
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Phase T5 -- MeanReversionRSI Portfolio Filter")
    p("  Config  : rsi14/os25/time_exit=20/atr3/no_filter/1D  (Variant E)")
    p(f"  Caps    : {CAP_LABELS}")
    p("=" * 70)

    # Load Variant E trades
    trades_path = T3MR_DIR / "phase_t3mr_trades_E.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found. Run T3MR first.")
        sys.exit(1)

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    if "year" not in df.columns:
        df["year"] = df["entry_time"].dt.year

    p(f"  Total input trades : {len(df)}")
    p(f"  Date range         : {df['entry_time'].iloc[0].date()} -- "
      f"{df['entry_time'].iloc[-1].date()}")
    total_input = len(df)

    # ------------------------------------------------------------------
    # Run each cap
    # ------------------------------------------------------------------
    summary_rows = []
    year_tables  = {}

    for cap, label in zip(CAPS, CAP_LABELS):
        acc_df  = apply_cap(df, cap)
        rej     = total_input - len(acc_df)
        acc_pct = len(acc_df) / total_input * 100
        m       = metrics(acc_df["net_r"].values)
        g       = gate_str(m)

        p(f"\n  --- {label} ---")
        p(f"  Accepted : {m['n']:3d} / {total_input}  ({acc_pct:.1f}%)  rejected={rej}")
        p(f"  Win rate : {m['win_rate']*100:.1f}%")
        p(f"  Avg R    : {m['avg_r']:+.4f}R")
        p(f"  Total R  : {m['total_r']:+.2f}R")
        p(f"  PF       : {m['pf']:.2f}")
        p(f"  Max DD   : {m['max_dd_r']:.2f}R")
        p(f"  Gate     : {g}")

        # Concurrent load: how often was the cap binding?
        if cap is not None and m["n"] > 0:
            rejected_df    = df[~df.index.isin(acc_df.index)] if False else None
            # Compute max concurrent at any entry point
            max_conc_seen  = 0
            open_exits_chk = []
            for _, row in df.sort_values("entry_time").iterrows():
                open_exits_chk = [et for et in open_exits_chk
                                  if et > row["entry_time"]]
                max_conc_seen  = max(max_conc_seen, len(open_exits_chk) + 1)
                open_exits_chk.append(row["exit_time"])
            p(f"  Max concurrent (uncapped) ever reached : {max_conc_seen}")

        summary_rows.append({
            "cap":           label,
            "max_conc":      cap if cap is not None else "none",
            "accepted":      m["n"],
            "accept_pct":    round(acc_pct, 1),
            "rejected":      rej,
            "win_rate":      round(m["win_rate"], 4),
            "avg_r":         round(m["avg_r"], 4),
            "total_r":       round(m["total_r"], 2),
            "pf":            round(m["pf"], 2),
            "max_dd_r":      round(m["max_dd_r"], 2),
            "gate":          g,
        })

        # Year breakdown
        if not acc_df.empty:
            yb = year_breakdown(acc_df)
            year_tables[label] = yb
            acc_df.to_csv(OUT_DIR / f"phase_t5_trades_{label}.csv", index=False)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "phase_t5_portfolio_summary.csv", index=False)

    p()
    p("=" * 70)
    p("  T5 PORTFOLIO FILTER COMPARISON")
    p("=" * 70)
    p(f"  {'Cap':<10} {'Acc':>5} {'Acc%':>6} {'WR%':>6} {'AvgR':>8} "
      f"{'TotR':>8} {'PF':>5} {'MaxDD':>7} {'Gate'}")
    p("  " + "-" * 70)
    for r in summary_rows:
        p(f"  {r['cap']:<10} {r['accepted']:>5} {r['accept_pct']:>5.1f}% "
          f"{r['win_rate']*100:>5.1f}% {r['avg_r']:>+7.4f}R "
          f"{r['total_r']:>+7.2f}R {r['pf']:>5.2f} {r['max_dd_r']:>6.2f}R {r['gate']}")
    p("  " + "-" * 70)

    # ------------------------------------------------------------------
    # Year-by-year for each cap
    # ------------------------------------------------------------------
    p()
    p("  Year-by-year detail:")
    for label in CAP_LABELS:
        if label not in year_tables:
            continue
        yb = year_tables[label]
        p(f"\n  {label}:")
        p(f"    {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotalR':>8}  {'DD':>6}  Note")
        for _, row in yb.iterrows():
            yr   = int(row["year"])
            note = "<<< BEAR" if yr == 2022 else ("(partial)" if yr == 2026 else "")
            p(f"    {yr:>5}  {int(row['n']):>5}  {row['win_rate']*100:>5.1f}%  "
              f"{row['avg_r']:>+7.3f}R  {row['total_r']:>+7.2f}R  "
              f"{row['max_dd_r']:>5.2f}R  {note}")

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    p()
    p("=" * 70)
    p("  RECOMMENDATION")
    p("=" * 70)
    passing = [r for r in summary_rows if r["gate"] == "PASS"]
    if passing:
        # Prefer: passes all gates + lowest DD + avg_r > threshold
        # Sort by DD ascending among passing configs
        passing_sorted = sorted(passing, key=lambda x: (x["max_dd_r"], -x["avg_r"]))
        best = passing_sorted[0]
        p(f"  Best cap by lowest DD: {best['cap']}")
        p(f"    Accepted={best['accepted']}  avg_r={best['avg_r']:+.4f}R  "
          f"PF={best['pf']:.2f}  DD={best['max_dd_r']:.2f}R")

        # Also flag uncapped DD for reference
        uncapped = next(r for r in summary_rows if r["cap"] == "uncapped")
        p(f"  Uncapped reference   : avg_r={uncapped['avg_r']:+.4f}R  "
          f"DD={uncapped['max_dd_r']:.2f}R")

        dd_reduction = (uncapped["max_dd_r"] - best["max_dd_r"]) / uncapped["max_dd_r"] * 100
        r_reduction  = (uncapped["total_r"]  - best["total_r"])  / uncapped["total_r"]  * 100
        p(f"  Capping to {best['cap']}: DD reduced {dd_reduction:.1f}%  "
          f"at cost of {r_reduction:.1f}% total R")

    # Write master report
    with open(OUT_DIR / "phase_t5_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Phase T5 -- MeanReversionRSI Portfolio Filter\n")
        f.write("Config: rsi14/os25/time_exit=20/atr3/no_filter/1D (Variant E)\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nYear-by-year per cap:\n")
        for label, yb in year_tables.items():
            f.write(f"\n{label}:\n")
            f.write(yb.to_string(index=False))
            f.write("\n")

    p()
    p(f"[OK] phase_t5_portfolio_summary.csv ({len(summary_df)} caps)")
    p(f"[OK] phase_t5_trades_{{cap}}.csv (one per cap)")
    p(f"[OK] phase_t5_report.txt")
    sys.exit(0)


if __name__ == "__main__":
    main()
