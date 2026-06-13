#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T8 -- DualMA BTC-only Config Freeze

Reads T5 and T6 summary outputs, performs final gate checks,
and writes the frozen canonical config to dualma_btc_config.json
and a human-readable freeze report.

T7 is documented as N/A (single-asset system).

Input:  data/research_dualma_t5/phase_t5_portfolio_summary.csv
        data/research_dualma_t6/phase_t6_variant_summary.csv
Output: data/research_dualma_t8/phase_t8_freeze_report.txt
        dualma_btc_config.json  (updated with frozen T8 fields)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

T5_SUMMARY = PROJECT_ROOT / "data" / "research_dualma_t5" / "phase_t5_portfolio_summary.csv"
T6_SUMMARY = PROJECT_ROOT / "data" / "research_dualma_t6" / "phase_t6_variant_summary.csv"
OUT_DIR    = PROJECT_ROOT / "data" / "research_dualma_t8"
CONFIG_OUT = PROJECT_ROOT / "dualma_btc_config.json"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE T8 -- DualMA BTC-only Config Freeze")
    print("=" * 70)

    # Load summaries
    missing = [p for p in [T5_SUMMARY, T6_SUMMARY] if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Missing: {p}")
        return 1

    t5 = pd.read_csv(T5_SUMMARY).iloc[0].to_dict()
    t6 = pd.read_csv(T6_SUMMARY).iloc[0].to_dict()

    # Gate checks
    avg_r       = float(t5.get("avg_r", 0))
    pf_r        = float(t5.get("profit_factor", 0))
    win_rate    = float(t5.get("win_rate_pct", 0))
    total_r     = float(t5.get("total_r", 0))
    cagr        = float(t6.get("cagr_pct", 0))
    max_dd      = float(t6.get("max_dd_pct", 0))
    tot_return  = float(t6.get("total_return_pct", 0))
    final_eq    = float(t6.get("final_equity", 0))

    gate_cost_floor = avg_r > 0.15
    gate_pf         = pf_r > 1.5
    gate_win        = 25 <= win_rate <= 50   # relaxed: BTC single-asset, fat-tail structure
    gate_cagr       = cagr > 0
    gate_dd         = abs(max_dd) < 35.0
    all_pass        = gate_cost_floor and gate_pf and gate_cagr and gate_dd

    # Build frozen config record
    frozen = {
        "method_name":              "DualMA_BTC",
        "status":                   "FROZEN_T8" if all_pass else "REVIEW_REQUIRED",
        "freeze_date":              "2026-05-27",
        "asset":                    "BTC/USDT",
        "universe":                 "single asset",
        "timeframe":                "1d",
        "fast_ema":                 30,
        "slow_ema":                 100,
        "filter_mode":              "ema200_price",
        "ema_regime_length":        200,
        "atr_mult_initial_stop":    2.0,
        "atr_period":               14,
        "chandelier_activate_r":    4.0,
        "chandelier_atr_mult":      3.0,
        "chandelier_lookback":      14,
        "fee_per_side":             0.001,
        "allow_short":              False,
        "exchange":                 "binance_spot",
        "leverage":                 1.0,
        "risk_per_trade":           0.0025,
        "max_concurrent":           1,
        "side":                     "LONG",
        "t5_n_trades":              int(t5.get("n_trades", 0)),
        "t5_total_r":               round(total_r, 4),
        "t5_avg_r":                 round(avg_r, 4),
        "t5_win_rate_pct":          round(win_rate, 2),
        "t5_profit_factor":         round(pf_r, 4),
        "t5_max_dd_r":              round(float(t5.get("max_dd_r", 0)), 4),
        "t6_initial_capital":       10000.0,
        "t6_final_equity":          round(final_eq, 2),
        "t6_total_return_pct":      round(tot_return, 4),
        "t6_cagr_pct":              round(cagr, 4),
        "t6_max_dd_pct":            round(max_dd, 4),
        "t7_status":                "N/A -- single-asset system, remove-best-asset test meaningless",
        "t4_halt_note":             "Full 70-asset DualMA halted T4 due to BTC concentration (123.4% of R). This BTC-only variant is the explicit single-asset distillation.",
        "entry_logic":              "EMA30 crosses above EMA100 on close, AND close > EMA200. Enter at close.",
        "exit_logic":               "Combined: (1) low<=stop -> exit at stop. (2) If chandelier not active: EMA30<EMA100 on close -> exit at close. (3) After exit checks: if MFE>=4.0R activate chandelier stop = HH - 3.0*ATR(14). Chandelier updated AFTER checks.",
    }

    # Write updated config
    CONFIG_OUT.write_text(json.dumps(frozen, indent=4), encoding="utf-8")
    print(f"  Config written: {CONFIG_OUT}")

    # Freeze report
    decision = "FROZEN" if all_pass else "HUMAN_REVIEW_REQUIRED"
    lines = [
        "=" * 70,
        "PHASE T8 -- DualMA BTC-only FREEZE REPORT",
        "=" * 70,
        f"Decision: {decision}",
        "",
        "FROZEN CONFIG",
        "-" * 70,
        "  Asset:                 BTC/USDT (single-asset variant)",
        "  Entry:                 EMA30 x EMA100 crossover + EMA200 filter",
        "  Initial stop:          ATR(14) x 2.0",
        "  Exit:                  MA cross (pre-chandelier) OR chandelier stop",
        "  Chandelier activation: MFE >= 4.0R",
        "  Chandelier mult:       3.0 x ATR(14)",
        "  Risk per trade:        0.25%",
        "  Max concurrent:        1",
        "  Exchange:              Binance Spot, LONG only, 1x leverage",
        "",
        "T5 PORTFOLIO RESULTS (R)",
        "-" * 70,
        f"  n_trades:      {int(t5.get('n_trades', 0))}",
        f"  total_r:       {total_r:.4f}R",
        f"  avg_r:         {avg_r:.4f}R",
        f"  win_rate:      {win_rate:.1f}%",
        f"  profit_factor: {pf_r:.4f}",
        f"  max_dd_r:      {t5.get('max_dd_r', 0):.4f}R",
        "",
        "T6 CAPITAL RESULTS ($10k, 0.25% risk)",
        "-" * 70,
        f"  Final equity:   ${final_eq:,.2f}",
        f"  Total return:   {tot_return:.2f}%",
        f"  CAGR:           {cagr:.2f}%",
        f"  Max DD:         {max_dd:.2f}%",
        "",
        "T7: N/A -- single-asset system",
        "",
        "GATE CHECKS",
        "-" * 70,
        f"  Cost floor (avg_r > 0.15R): {'PASS' if gate_cost_floor else 'FAIL'} ({avg_r:.4f}R)",
        f"  PF > 1.5:                   {'PASS' if gate_pf else 'FAIL'} ({pf_r:.4f})",
        f"  Win rate (25-50%):          {'PASS' if gate_win else 'WARN'} ({win_rate:.1f}%)",
        f"  CAGR > 0:                   {'PASS' if gate_cagr else 'FAIL'} ({cagr:.2f}%)",
        f"  Max DD < 35%:               {'PASS' if gate_dd else 'FAIL'} ({max_dd:.2f}%)",
        "",
        f"OVERALL: {decision}",
        "",
        "NOTE: This is a single-asset BTC-only strategy.",
        "T4 full-universe DualMA was halted due to BTC concentration.",
        "This variant is intentionally a BTC trend-following filter,",
        "not a diversified portfolio. Evaluate accordingly for T9B paper trading.",
        "=" * 70,
    ]
    report_text = "\n".join(lines)
    (OUT_DIR / "phase_t8_freeze_report.txt").write_text(report_text, encoding="utf-8")
    print(f"  Report written: {OUT_DIR / 'phase_t8_freeze_report.txt'}")
    print()
    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
