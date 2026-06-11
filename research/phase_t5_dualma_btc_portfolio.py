#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T5 -- DualMA BTC-only Portfolio Replay

Filters T3B best-variant trades to BTC/USDT only and applies a max1
concurrent-position cap (single-asset system).

For a single asset the max1 cap is trivially satisfied -- the backtester
guarantees sequential trades per symbol.  The replay still tracks and reports
any overlapping signals so the assumption can be verified.

Input:  data/research_dualma_t3b/phase_t3b_wide_exit_trades.csv
Config: dualma_btc_config.json (chan_mult=3.0, act=4.0R)

Outputs:
    data/research_dualma_t5/
        phase_t5_portfolio_trades.csv
        phase_t5_portfolio_equity.csv
        phase_t5_portfolio_skipped_signals.csv
        phase_t5_portfolio_summary.csv
        phase_t5_master_report.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

T3B_TRADES = PROJECT_ROOT / "data" / "research_dualma_t3b" / "phase_t3b_wide_exit_trades.csv"
OUT_DIR     = PROJECT_ROOT / "data" / "research_dualma_t5"

ASSET          = "BTC/USDT"
VARIANT_NAME   = "DualMA_BTC_max1"
CHAN_MULT      = 3.0
CHAN_ACT_R     = 4.0
MAX_CONCURRENT = 1


# =============================================================================
# HELPERS
# =============================================================================

def max_drawdown_r(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    eq   = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def profit_factor(r: np.ndarray) -> float:
    gains  = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def losing_streak(r: np.ndarray) -> int:
    best = cur = 0
    for v in r:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE T5 -- DualMA BTC-only Portfolio Replay")
    print("=" * 70)

    if not T3B_TRADES.exists():
        print(f"[ERROR] T3B trades not found: {T3B_TRADES}")
        return 1

    raw = pd.read_csv(T3B_TRADES)
    print(f"T3B total trades loaded: {len(raw)}")

    # Filter: BTC/USDT + best variant params
    btc = raw[
        (raw["symbol"] == ASSET) &
        (raw["chandelier_atr_mult"] == CHAN_MULT) &
        (raw["chandelier_activate_r"] == CHAN_ACT_R)
    ].copy()

    print(f"BTC/USDT best-variant trades: {len(btc)}")

    if btc.empty:
        print("[ERROR] No BTC trades found after filtering.")
        return 1

    # Normalise time columns
    btc["entry_time"] = pd.to_datetime(btc["entry_bar"], utc=True, errors="coerce")
    btc["exit_time"]  = pd.to_datetime(btc["exit_bar"],  utc=True, errors="coerce")
    btc = btc.dropna(subset=["entry_time", "exit_time"]).copy()
    btc = btc.sort_values("entry_time").reset_index(drop=True)
    btc["side"]      = "LONG"
    btc["timeframe"] = "1d"
    btc["variant"]   = VARIANT_NAME

    # Apply max1 concurrent cap
    accepted_rows  = []
    skipped_rows   = []
    open_exit_time = None   # exit time of the currently open BTC position

    for _, row in btc.iterrows():
        entry_t = row["entry_time"]
        exit_t  = row["exit_time"]

        # Close open position if it exits before this entry
        if open_exit_time is not None and open_exit_time <= entry_t:
            open_exit_time = None

        if open_exit_time is None:
            # Slot free -- accept
            accepted_rows.append(row)
            open_exit_time = exit_t
        else:
            # Overlap -- skip
            skip = row.to_dict()
            skip["skip_reason"] = "ASSET_ALREADY_OPEN"
            skip["open_exit_time"] = open_exit_time
            skipped_rows.append(skip)

    accepted = pd.DataFrame(accepted_rows).reset_index(drop=True)
    skipped  = pd.DataFrame(skipped_rows).reset_index(drop=True) if skipped_rows else pd.DataFrame()

    print(f"  Accepted: {len(accepted)}  |  Skipped: {len(skipped)}")

    # Equity curve in R
    r_vals     = accepted["net_r"].to_numpy(dtype=float)
    cumulative = np.cumsum(r_vals)
    eq_rows    = []
    for i, row in accepted.iterrows():
        eq_rows.append({
            "trade_num":    i + 1,
            "exit_time":    row["exit_time"],
            "symbol":       row["symbol"],
            "net_r":        row["net_r"],
            "cumulative_r": cumulative[i],
        })
    df_equity = pd.DataFrame(eq_rows)

    # Summary metrics
    n       = len(r_vals)
    total_r = float(r_vals.sum())
    avg_r   = float(r_vals.mean()) if n > 0 else 0.0
    wins    = int((r_vals > 0).sum())
    win_pct = wins / n * 100 if n > 0 else 0.0
    pf      = profit_factor(r_vals)
    max_dd  = max_drawdown_r(r_vals)
    streak  = losing_streak(r_vals)
    std_r   = float(r_vals.std(ddof=1)) if n > 1 else 0.0
    t_score = avg_r / (std_r / math.sqrt(n)) if std_r > 0 and n > 1 else 0.0

    date_range_start = accepted["entry_time"].min()
    date_range_end   = accepted["exit_time"].max()

    summary = {
        "variant":          VARIANT_NAME,
        "asset":            ASSET,
        "n_trades":         n,
        "total_r":          round(total_r, 4),
        "avg_r":            round(avg_r, 4),
        "win_rate_pct":     round(win_pct, 2),
        "profit_factor":    round(pf, 4),
        "max_dd_r":         round(max_dd, 4),
        "losing_streak":    streak,
        "std_r":            round(std_r, 4),
        "t_score":          round(t_score, 4),
        "date_start":       str(date_range_start),
        "date_end":         str(date_range_end),
        "skipped_signals":  len(skipped),
    }

    # Write outputs
    accepted.to_csv(OUT_DIR / "phase_t5_portfolio_trades.csv", index=False)
    df_equity.to_csv(OUT_DIR / "phase_t5_portfolio_equity.csv", index=False)
    if not skipped.empty:
        skipped.to_csv(OUT_DIR / "phase_t5_portfolio_skipped_signals.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "entry_time", "skip_reason"]).to_csv(
            OUT_DIR / "phase_t5_portfolio_skipped_signals.csv", index=False
        )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "phase_t5_portfolio_summary.csv", index=False)

    # Master report
    report_lines = [
        "=" * 70,
        "PHASE T5 -- DualMA BTC-only Portfolio Report",
        "=" * 70,
        f"Variant:       {VARIANT_NAME}",
        f"Asset:         {ASSET}",
        f"Cap:           max1 concurrent (single asset)",
        f"Exit config:   chandelier_atr_mult={CHAN_MULT}, chandelier_activate_r={CHAN_ACT_R}R",
        f"Trades input:  {len(btc)}",
        f"Accepted:      {n}",
        f"Skipped:       {len(skipped)}",
        f"Date range:    {date_range_start} -> {date_range_end}",
        "",
        "PORTFOLIO METRICS (R)",
        "-" * 70,
        f"  total_r:        {total_r:.4f}R",
        f"  avg_r:          {avg_r:.4f}R",
        f"  win_rate:       {win_pct:.1f}%",
        f"  profit_factor:  {pf:.4f}",
        f"  max_dd_r:       {max_dd:.4f}R",
        f"  losing_streak:  {streak}",
        f"  t_score:        {t_score:.4f}",
        "",
        "GATE CHECKS",
        "-" * 70,
        f"  Cost floor (avg_r > 0.15R): {'PASS' if avg_r > 0.15 else 'FAIL'} ({avg_r:.4f}R)",
        f"  PF > 1.0:                   {'PASS' if pf > 1.0 else 'FAIL'} ({pf:.4f})",
        f"  Win rate (30-45%):          {'PASS' if 30 <= win_pct <= 45 else 'WARN'} ({win_pct:.1f}%)",
        "",
        "NOTE: T7 skipped -- single-asset system; remove-best-asset test is N/A.",
        "Proceed to T6 (capital simulation) if gates pass.",
        "=" * 70,
    ]
    report_text = "\n".join(report_lines)
    (OUT_DIR / "phase_t5_master_report.txt").write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
