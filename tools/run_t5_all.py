#!/usr/bin/env python3
"""
T5 portfolio replay for candidates 12, 11u, 13u -- entry AND exit both frozen
(T3/T4 closed, original flat exit kept for all three). Tests whether the edge
survives realistic position sizing and exposure limits at $60k target
deployment capital: max position size per symbol, max concurrent positions,
max correlation-cluster concentration (reusing modules/asset_clustering.py,
built for candidate 16). Reports uncapped vs capped return/Sharpe for each.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END  # noqa: E402
from tools.t5_portfolio_replay import (  # noqa: E402
    build_symbol_cluster_map, replay_with_caps, CAPITAL_BASE,
    DEFAULT_MAX_OPEN, DEFAULT_MAX_POSITION_PCT, DEFAULT_MAX_CLUSTER_PCT,
)

t3_12 = importlib.import_module("run_t3_candidate12")
t3_11u = importlib.import_module("run_t3_candidate11u")
t3_13u = importlib.import_module("run_t3_candidate13u")

UNCAPPED = dict(max_open=10**9, max_position_pct=1.0, max_cluster_pct=1.0)
CAPPED = dict(max_open=DEFAULT_MAX_OPEN, max_position_pct=DEFAULT_MAX_POSITION_PCT,
              max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT)


def fmt(res: dict) -> str:
    return (f"total_return={res['total_return']:+.1%}  sharpe={res['sharpe']:.2f}  "
            f"max_dd={res['max_dd']:.1%}  final_equity=${res['final_equity']:,.0f}\n"
            f"    n_trades={res['n_trades_total']}  accepted={res['n_accepted']}  "
            f"rejected={res['n_rejected']} ({res['rejection_rate']:.1%})  "
            f"reject_breakdown={res['reject_counts']}  max_open_observed={res['max_open_observed']}")


def run_candidate(label: str, panel: dict, build_baseline_trades_fn, cluster_map: dict[str, str]):
    print(f"\n{'#'*80}\nT5: Candidate {label}\n{'#'*80}")
    trades, _prepared = build_baseline_trades_fn(panel)
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

    return dict(label=label, uncapped=uncapped, capped=capped)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_12.load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print(f"\nBuilding correlation clusters (corr_threshold=0.6, static full-window "
          f"{DEFAULT_IS_START.date()} to {DEFAULT_IS_END.date()}, reused from candidate 16's "
          f"modules/asset_clustering.py)...")
    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)
    n_clusters = len(set(v for v in cluster_map.values() if not v.startswith("solo_")))
    n_solo = sum(1 for v in cluster_map.values() if v.startswith("solo_"))
    print(f"Found {n_clusters} multi-symbol clusters (>=3 members) + {n_solo} solo/unclustered symbols.")

    print(f"\nCapital base: ${CAPITAL_BASE:,.0f}  |  risk-per-trade: 0.25% "
          f"(${CAPITAL_BASE * 0.0025:,.0f}/trade target)")

    results = {}
    results["12"] = run_candidate("12", panel, t3_12.build_baseline_trades, cluster_map)
    results["11u"] = run_candidate("11u", panel, t3_11u.build_baseline_trades, cluster_map)
    results["13u"] = run_candidate("13u", panel, t3_13u.build_baseline_trades, cluster_map)

    print(f"\n{'='*100}\nT5 SUMMARY -- ALL CANDIDATES ($60k capital, "
          f"max_open={CAPPED['max_open']}, max_position_pct={CAPPED['max_position_pct']:.0%}, "
          f"max_cluster_pct={CAPPED['max_cluster_pct']:.0%})\n{'='*100}")
    for label, res in results.items():
        u, c = res["uncapped"], res["capped"]
        print(f"\n{label}:")
        print(f"  Uncapped: return={u['total_return']:+.1%}  sharpe={u['sharpe']:.2f}")
        print(f"  Capped:   return={c['total_return']:+.1%}  sharpe={c['sharpe']:.2f}  "
              f"(rejected {c['rejection_rate']:.1%} of trades, "
              f"max concurrent observed={c['max_open_observed']})")
        print(f"  Delta:    return={c['total_return']-u['total_return']:+.1%}  "
              f"sharpe={c['sharpe']-u['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
