#!/usr/bin/env python3
"""
T4 robustness engine for candidate 19 (52-week-high / proximity-to-high momentum) --
entry AND exit both frozen (T3 closed with the original flat exit -- no variant beat
baseline avg_r=0.6396R). Reuses the exact same tools/t4_robustness_engine.py
methodology/thresholds already applied to 11u/12/13u: Monte Carlo block bootstrap,
cost stress (1.5x/2x the 0.25R futures floor), remove-best-asset, remove-best-month.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END, _combo_stats, BATCH_TRACKER  # noqa: E402
from tools.t4_robustness_engine import (  # noqa: E402
    monte_carlo_block_bootstrap, cost_stress, remove_best_asset, remove_best_month, BASELINE_COST_R,
)

sys.path.insert(0, str(ROOT / "tools"))
import run_t3_candidate19 as t3_19  # noqa: E402


def baseline_stats(trades: pd.DataFrame, is_years: float) -> dict:
    st = _combo_stats(trades, is_years)
    r = trades["net_r"]
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(st["trades_per_year"])
              if len(r) > 1 and r.std(ddof=1) > 0 else np.nan)
    st["sharpe"] = sharpe
    return st


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_19.load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print(f"\n{'#'*80}\nT4: Candidate 19\n{'#'*80}")
    trades, prepared = t3_19.build_baseline_trades(panel)
    is_years = max((DEFAULT_IS_END - DEFAULT_IS_START).days / 365.25, 1e-6)
    base = baseline_stats(trades, is_years)
    print(f"Baseline: n={base['n']}, avg_r={base['avg_r']:.4f}R, sharpe={base['sharpe']:.2f}, "
          f"t_stat={base['t_stat']:.2f}")

    print(f"\n--- Monte Carlo block bootstrap (block_size=10, 2000 runs) ---")
    mc = monte_carlo_block_bootstrap(trades, is_years)
    print(f"avg_r:  p05={mc['avg_r_p05']:.4f}R  p50={mc['avg_r_p50']:.4f}R  p95={mc['avg_r_p95']:.4f}R")
    print(f"sharpe: p05={mc['sharpe_p05']:.2f}  p50={mc['sharpe_p50']:.2f}  p95={mc['sharpe_p95']:.2f}")
    print(f"P(avg_r > 0): {mc['prob_avg_r_positive']:.1%}")

    print(f"\n--- Cost stress (baseline cost = {BASELINE_COST_R}R futures floor) ---")
    cost_rows = []
    for mult in (1.0, 1.5, 2.0):
        cs = cost_stress(trades, is_years, mult)
        cost_rows.append(cs)
        print(f"  {mult:.1f}x: extra_cost={cs['extra_cost_r']:.3f}R  avg_r={cs['avg_r']:.4f}R  "
              f"t_stat={cs['t_stat']:.2f}  sharpe={cs['sharpe']:.2f}  "
              f"clears {BASELINE_COST_R}R floor={cs['clears_floor']}")

    print(f"\n--- Remove-best-asset ---")
    remove_asset_result = remove_best_asset(trades, is_years)
    print(f"Best asset: {remove_asset_result['best_asset']} "
          f"(contributed {remove_asset_result['best_asset_contribution_r']:.1f}R, "
          f"{remove_asset_result['pct_of_total_r']:.1%} of total)")
    print(f"After removal: n={remove_asset_result['n_after']}, "
          f"avg_r={remove_asset_result['avg_r_after']:.4f}R, "
          f"t_stat={remove_asset_result['t_stat_after']:.2f}, "
          f"clears floor={remove_asset_result['clears_floor_after']}")

    print(f"\n--- Remove-best-month ---")
    remove_month_result = remove_best_month(trades, is_years)
    print(f"Best month: {remove_month_result['best_month']} "
          f"(contributed {remove_month_result['best_month_contribution_r']:.1f}R, "
          f"{remove_month_result['pct_of_total_r']:.1%} of total)")
    print(f"After removal: n={remove_month_result['n_after']}, "
          f"avg_r={remove_month_result['avg_r_after']:.4f}R, "
          f"t_stat={remove_month_result['t_stat_after']:.2f}, "
          f"clears floor={remove_month_result['clears_floor_after']}")

    fragile_reasons = []
    if mc["avg_r_p05"] <= 0:
        fragile_reasons.append(f"MC block bootstrap 5th pct avg_r <= 0 ({mc['avg_r_p05']:.4f}R)")
    if not cost_rows[2]["clears_floor"]:
        fragile_reasons.append(f"fails floor at 2x cost stress (avg_r={cost_rows[2]['avg_r']:.4f}R)")
    if not remove_asset_result["clears_floor_after"]:
        fragile_reasons.append("fails floor after removing best asset")
    if not remove_month_result["clears_floor_after"]:
        fragile_reasons.append("fails floor after removing best month")

    print(f"\nFRAGILITY: {'FLAGGED -- ' + '; '.join(fragile_reasons) if fragile_reasons else 'none found, robust across all tests'}")

    print(f"\n{'='*100}\nT4 SUMMARY -- CANDIDATE 19\n{'='*100}")
    print(f"19: baseline avg_r={base['avg_r']:.4f}R, sharpe={base['sharpe']:.2f}")
    print(f"  MC 5th/95th pct avg_r: {mc['avg_r_p05']:.4f}R / {mc['avg_r_p95']:.4f}R")
    print(f"  Cost stress 2x: avg_r={cost_rows[2]['avg_r']:.4f}R (clears={cost_rows[2]['clears_floor']})")
    print(f"  Remove-best-asset: avg_r_after={remove_asset_result['avg_r_after']:.4f}R "
          f"(clears={remove_asset_result['clears_floor_after']})")
    print(f"  Remove-best-month: avg_r_after={remove_month_result['avg_r_after']:.4f}R "
          f"(clears={remove_month_result['clears_floor_after']})")
    print(f"  FRAGILITY: {'FLAGGED' if fragile_reasons else 'none'}")

    # tracker update
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == "19"
    tracker.loc[mask, "t4_status"] = "CAUTION" if fragile_reasons else "PASS"
    tracker.loc[mask, "current_stage"] = "T4"
    existing = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing} T4: baseline n={base['n']}, avg_r={base['avg_r']:.4f}R, sharpe={base['sharpe']:.2f}. "
            f"MC bootstrap P(avg_r>0)={mc['prob_avg_r_positive']:.1%}, 5th/95th pct avg_r="
            f"{mc['avg_r_p05']:.4f}R/{mc['avg_r_p95']:.4f}R. Cost stress: 1.5x avg_r="
            f"{cost_rows[1]['avg_r']:.4f}R (clears={cost_rows[1]['clears_floor']}), 2x avg_r="
            f"{cost_rows[2]['avg_r']:.4f}R (clears={cost_rows[2]['clears_floor']}). "
            f"Remove-best-asset ({remove_asset_result['best_asset']}, "
            f"{remove_asset_result['pct_of_total_r']:.1%} of total R): avg_r_after="
            f"{remove_asset_result['avg_r_after']:.4f}R (clears={remove_asset_result['clears_floor_after']}). "
            f"Remove-best-month ({remove_month_result['best_month']}, "
            f"{remove_month_result['pct_of_total_r']:.1%} of total R): avg_r_after="
            f"{remove_month_result['avg_r_after']:.4f}R (clears={remove_month_result['clears_floor_after']}). "
            f"FRAGILITY: {'FLAGGED -- ' + '; '.join(fragile_reasons) if fragile_reasons else 'none'}.").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


if __name__ == "__main__":
    main()
