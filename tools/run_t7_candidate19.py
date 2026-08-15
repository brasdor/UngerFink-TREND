#!/usr/bin/env python3
"""
T7 strict robustness for candidate 19 (52-week-high / proximity-to-high momentum) --
runs the hostile-condition battery on the LOCKED T6 variant, exact same methodology
already applied to 11u/12/13u:

  entries: frozen since T1 (diversification-gated pooled survivor combos, unchanged
           through T3/T4/T5/T6)
  exit:    original flat/opposite-signal exit (T3-closed)
  capital: $60k, max_open=20, max_position_pct=10%, max_cluster_pct=30% (T5-confirmed
           caps -- ACCEPTED subset only, rejected signals dropped, not resized)
  cost:    liquidity-tiered slippage applied (T6-confirmed, net_r_after_slippage)
  leverage: locked <=2x operationally (T6 liquidation sweep -- 18.3% at 3x is the
           highest of any candidate except 12, so this cap matters here too)

Uses the now-deterministic modules/asset_clustering.py (fixed prior to this run --
see git history / T7 report) so the locked trade set is reproducible run to run,
unlike the T5/T6 runs for this candidate which showed minor drift from the old
hash-order-dependent seed selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END, BATCH_TRACKER  # noqa: E402
from tools.t5_portfolio_replay import (  # noqa: E402
    build_symbol_cluster_map, DEFAULT_MAX_OPEN, DEFAULT_MAX_POSITION_PCT, DEFAULT_MAX_CLUSTER_PCT,
)
from tools.t6_capital_execution import add_adv, apply_execution_realism  # noqa: E402
from tools.t7_strict_robustness import run_t7_suite, BASELINE_COST_R  # noqa: E402
import run_t3_candidate19 as t3_19  # noqa: E402

T5_CAPS = dict(max_open=DEFAULT_MAX_OPEN, max_position_pct=DEFAULT_MAX_POSITION_PCT,
               max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT)
LOCKED_LEVERAGE = 3  # reporting only -- reuses T6's slippage-tier machinery


def _accepted_subset(sorted_trades: pd.DataFrame, cluster_map: dict[str, str]) -> pd.DataFrame:
    """Reproduce T5's replay_with_caps acceptance logic to get the accepted trade rows
    themselves (replay_with_caps only returns aggregate stats) -- same helper as
    run_t7_all.py."""
    from tools.t5_portfolio_replay import ATR_MULT_R, RISK_PCT_PER_TRADE

    t = sorted_trades
    initial_risk = ATR_MULT_R * t["atr_at_entry"]
    stop_distance_pct = (initial_risk / t["entry_px"]).clip(lower=1e-9)
    target_notional_pct = (RISK_PCT_PER_TRADE / stop_distance_pct)

    open_positions: list[dict] = []
    accepted_flags = np.zeros(len(t), dtype=bool)
    max_open, max_position_pct, max_cluster_pct = (
        DEFAULT_MAX_OPEN, DEFAULT_MAX_POSITION_PCT, DEFAULT_MAX_CLUSTER_PCT)

    for i, row in t.iterrows():
        open_positions = [p for p in open_positions if p["exit_date"] > row["entry_date"]]
        cluster = cluster_map.get(row["symbol"], "unclustered")
        cluster_pct_open = sum(p["notional_pct"] for p in open_positions if p["cluster"] == cluster)

        reason = None
        if len(open_positions) >= max_open:
            reason = "MAX_OPEN"
        else:
            actual_notional_pct = min(target_notional_pct[i], max_position_pct)
            if cluster_pct_open + actual_notional_pct > max_cluster_pct:
                reason = "MAX_CLUSTER_PCT"

        if reason:
            continue

        actual_notional_pct = min(target_notional_pct[i], max_position_pct)
        accepted_flags[i] = True
        open_positions.append({"symbol": row["symbol"], "cluster": cluster,
                                "notional_pct": actual_notional_pct, "exit_date": row["exit_date"]})

    return t[accepted_flags]


def print_rows(rows: list[dict], floor_note: bool = True):
    for r in rows:
        extra = f"  clears_floor={r['clears_floor']}" if floor_note and "clears_floor" in r else ""
        print(f"    {r['label']:<22} n={r['n']:>5}  avg_r={r['avg_r']:.4f}R  pf={r['pf']:.2f}  "
              f"dd_r={r['dd_r']:.1f}R  win={r['win_rate']:.1%}{extra}")


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_19.load_panel()
    print(f"Loaded {len(panel)} symbols.")

    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)
    print(f"Locked config: $60k capital, max_open={DEFAULT_MAX_OPEN}, "
          f"max_position_pct={DEFAULT_MAX_POSITION_PCT:.0%}, max_cluster_pct={DEFAULT_MAX_CLUSTER_PCT:.0%}, "
          f"liquidity-tiered slippage, leverage<=2x")

    print(f"\n{'#'*80}\nT7: Candidate 19 (locked T6 variant)\n{'#'*80}")
    trades, prepared = t3_19.build_baseline_trades(panel)
    prepared_adv = add_adv(prepared)
    t_realism = apply_execution_realism(trades, prepared_adv, leverage=LOCKED_LEVERAGE)

    sorted_trades = trades.sort_values("entry_date").reset_index(drop=True)
    sorted_realism = t_realism.sort_values("entry_date").reset_index(drop=True)
    accepted_trades = _accepted_subset(sorted_trades, cluster_map)
    locked = sorted_realism.loc[accepted_trades.index].copy()
    locked["net_r"] = locked["net_r_after_slippage"]

    print(f"Locked T6 variant: n={len(locked)} trades (T5-accepted, slippage-adjusted), "
          f"baseline avg_r={locked['net_r'].mean():.4f}R")

    suite = run_t7_suite(locked)
    b = suite["baseline"]
    print(f"\nBASELINE: n={b['n']}  avg_r={b['avg_r']:.4f}R  pf={b['pf']:.2f}  "
          f"dd_r={b['dd_r']:.1f}R  win={b['win_rate']:.1%}  clears_floor={b['clears_floor']}")

    print("\n-- Cost stress --")
    print_rows(suite["cost"])
    print("\n-- Remove top assets --")
    print_rows(suite["rm_assets"])
    print("\n-- Remove top months --")
    print_rows(suite["rm_months"])
    print("\n-- Long/short split --")
    print_rows(suite["sides"])
    print("\n-- Recent-trade degradation --")
    print_rows(suite["recent"])

    rolling = suite["rolling"]
    print(f"\n-- Rolling 30-trade windows -- n_windows={len(rolling)}")
    print(f"    windows with avg_r <= 0:                  {suite['rolling_neg_frac']:.1%}")
    print(f"    windows with avg_r <= {BASELINE_COST_R}R floor:          {suite['rolling_below_floor_frac']:.1%}")
    if len(rolling):
        print(f"    worst window avg_r={rolling['window_avg_r'].min():.4f}R  "
              f"worst window dd_r={rolling['window_dd_r'].min():.1f}R")

    fail_reasons = []
    if not suite["cost"][-1]["clears_floor"]:
        fail_reasons.append(f"fails floor at max cost stress (+{suite['cost'][-1]['extra_cost_r']:.2f}R)")
    if any(not r["clears_floor"] for r in suite["rm_assets"]):
        fail_reasons.append("fails floor after removing top assets")
    if any(not r["clears_floor"] for r in suite["rm_months"]):
        fail_reasons.append("fails floor after removing top months")
    if any((not r["clears_floor"]) and r["n"] > 30 for r in suite["sides"]):
        fail_reasons.append("one side (long/short) fails floor -- edge not side-balanced")
    if suite["rolling_below_floor_frac"] is not np.nan and suite["rolling_below_floor_frac"] > 0.5:
        fail_reasons.append(f"majority of rolling windows ({suite['rolling_below_floor_frac']:.0%}) fail floor")

    verdict = "FAIL" if fail_reasons else "PASS"
    print(f"\nT7 VERDICT: {'FAILS STRICT ROBUSTNESS -- ' + '; '.join(fail_reasons) if fail_reasons else 'PASSES -- robust under all hostile-condition tests'}")

    print(f"\n{'='*100}\nT7 SUMMARY -- CANDIDATE 19\n{'='*100}")
    print(f"19: baseline avg_r={b['avg_r']:.4f}R, pf={b['pf']:.2f}  -> T7 {verdict}")
    if fail_reasons:
        for reason in fail_reasons:
            print(f"    - {reason}")

    # tracker update
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == "19"
    tracker.loc[mask, "t7_status"] = verdict
    tracker.loc[mask, "current_stage"] = "T8_FROZEN_PAPER_LIVE" if verdict == "PASS" else "T7_FAIL_SET_ASIDE"
    existing = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing} T7 (locked variant, n={b['n']}): baseline avg_r={b['avg_r']:.4f}R, pf={b['pf']:.2f}. "
            f"Cost stress +0.15R: avg_r={suite['cost'][-1]['avg_r']:.4f}R (clears={suite['cost'][-1]['clears_floor']}). "
            f"Remove-top-5-assets: avg_r={suite['rm_assets'][-1]['avg_r']:.4f}R "
            f"(clears={suite['rm_assets'][-1]['clears_floor']}). "
            f"Remove-top-2-months: avg_r={suite['rm_months'][-1]['avg_r']:.4f}R "
            f"(clears={suite['rm_months'][-1]['clears_floor']}). "
            f"Rolling 30-trade windows below floor: {suite['rolling_below_floor_frac']:.1%}. "
            f"T7 VERDICT: {verdict}"
            + (f" -- {'; '.join(fail_reasons)}" if fail_reasons else "") + ".").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


if __name__ == "__main__":
    main()
