#!/usr/bin/env python3
"""
T5 portfolio replay for candidate 19 (52-week-high / proximity-to-high momentum) --
entry AND exit both frozen (T4 closed, no fragility found). Reuses the exact same
tools/t5_portfolio_replay.py engine and parameters already applied to 11u/12/13u:
$60k capital, 0.25% risk-per-trade, max_open=20, max_position_pct=10%,
max_cluster_pct=30% (modules/asset_clustering.py, corr_threshold=0.6).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import pandas as pd  # noqa: E402

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END, BATCH_TRACKER  # noqa: E402
from tools.t5_portfolio_replay import (  # noqa: E402
    build_symbol_cluster_map, replay_with_caps, CAPITAL_BASE,
    DEFAULT_MAX_OPEN, DEFAULT_MAX_POSITION_PCT, DEFAULT_MAX_CLUSTER_PCT,
)
import run_t3_candidate19 as t3_19  # noqa: E402

UNCAPPED = dict(max_open=10**9, max_position_pct=1.0, max_cluster_pct=1.0)
CAPPED = dict(max_open=DEFAULT_MAX_OPEN, max_position_pct=DEFAULT_MAX_POSITION_PCT,
              max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT)


def fmt(res: dict) -> str:
    return (f"total_return={res['total_return']:+.1%}  sharpe={res['sharpe']:.2f}  "
            f"max_dd={res['max_dd']:.1%}  final_equity=${res['final_equity']:,.0f}\n"
            f"    n_trades={res['n_trades_total']}  accepted={res['n_accepted']}  "
            f"rejected={res['n_rejected']} ({res['rejection_rate']:.1%})  "
            f"reject_breakdown={res['reject_counts']}  max_open_observed={res['max_open_observed']}")


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_19.load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print(f"\nBuilding correlation clusters (corr_threshold=0.6, static full-window "
          f"{DEFAULT_IS_START.date()} to {DEFAULT_IS_END.date()})...")
    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)

    print(f"\nCapital base: ${CAPITAL_BASE:,.0f}  |  risk-per-trade: 0.25% "
          f"(${CAPITAL_BASE * 0.0025:,.0f}/trade target)")

    print(f"\n{'#'*80}\nT5: Candidate 19\n{'#'*80}")
    trades, prepared = t3_19.build_baseline_trades(panel)
    print(f"Baseline trade set: n={len(trades)}, unique symbols={trades['symbol'].nunique()}")

    uncapped = replay_with_caps(trades, cluster_map, **UNCAPPED)
    capped = replay_with_caps(trades, cluster_map, **CAPPED)

    print(f"\n  UNCAPPED (no position/exposure limits):\n    {fmt(uncapped)}")
    print(f"\n  CAPPED (max_open={CAPPED['max_open']}, "
          f"max_position_pct={CAPPED['max_position_pct']:.0%}, "
          f"max_cluster_pct={CAPPED['max_cluster_pct']:.0%}):\n    {fmt(capped)}")

    delta_return = capped["total_return"] - uncapped["total_return"]
    delta_sharpe = capped["sharpe"] - uncapped["sharpe"]
    print(f"\n  DELTA (capped - uncapped): return={delta_return:+.1%}  sharpe={delta_sharpe:+.2f}")

    is_years = max((DEFAULT_IS_END - DEFAULT_IS_START).days / 365.25, 1e-6)
    uncapped_cagr = (1 + uncapped["total_return"]) ** (1 / is_years) - 1
    capped_cagr = (1 + capped["total_return"]) ** (1 / is_years) - 1
    print(f"\n  CAGR: uncapped={uncapped_cagr:+.1%}  capped={capped_cagr:+.1%}")

    print(f"\n{'='*100}\nT5 SUMMARY -- CANDIDATE 19 ($60k capital, max_open={CAPPED['max_open']}, "
          f"max_position_pct={CAPPED['max_position_pct']:.0%}, "
          f"max_cluster_pct={CAPPED['max_cluster_pct']:.0%})\n{'='*100}")
    print(f"Uncapped: return={uncapped['total_return']:+.1%}  CAGR={uncapped_cagr:+.1%}  "
          f"sharpe={uncapped['sharpe']:.2f}  max_dd={uncapped['max_dd']:.1%}  "
          f"max_open_observed={uncapped['max_open_observed']}")
    print(f"Capped:   return={capped['total_return']:+.1%}  CAGR={capped_cagr:+.1%}  "
          f"sharpe={capped['sharpe']:.2f}  max_dd={capped['max_dd']:.1%}  "
          f"(rejected {capped['rejection_rate']:.1%} of trades)")
    print(f"Reject breakdown (uncapped run, cluster cap only): {uncapped['reject_counts']}")
    print(f"Reject breakdown (capped run, all caps active): {capped['reject_counts']}")

    # tracker update
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == "19"
    tracker.loc[mask, "t5_status"] = "PASS"
    tracker.loc[mask, "current_stage"] = "T5"
    existing = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing} T5 ($60k, max_open=20, max_position_pct=10%, max_cluster_pct=30%): "
            f"uncapped return={uncapped['total_return']:+.1%} (CAGR={uncapped_cagr:+.1%}), "
            f"sharpe={uncapped['sharpe']:.2f}, max_dd={uncapped['max_dd']:.1%}, "
            f"max_open_observed={uncapped['max_open_observed']}. "
            f"capped return={capped['total_return']:+.1%} (CAGR={capped_cagr:+.1%}), "
            f"sharpe={capped['sharpe']:.2f}, max_dd={capped['max_dd']:.1%}, "
            f"rejected {capped['rejection_rate']:.1%} of trades "
            f"({capped['reject_counts']}). Delta sharpe={delta_sharpe:+.2f}.").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


if __name__ == "__main__":
    main()
