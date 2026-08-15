#!/usr/bin/env python3
"""
T6 capital/execution realism for candidate 19 (52-week-high / proximity-to-high
momentum) -- entry AND exit both frozen (T5 closed, strongest capped result of any
candidate this session). Reuses the exact same tools/t6_capital_execution.py engine
and parameters already applied to 11u/12/13u: T5's confirmed $60k/max_open=20/
max_position_pct=10%/max_cluster_pct=30% caps as the base trade set, liquidity-
tiered slippage, market-vs-limit fill test, and the 3x/5x/10x liquidation sweep.
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
from tools.t6_capital_execution import (  # noqa: E402
    add_adv, apply_execution_realism, position_sizing_check, LEVERAGE_SWEEP,
    TIER_BPS, TIER1_ADV_USD, TIER2_ADV_USD, LIMIT_EDGE_BPS, MIN_NOTIONAL_USD,
)
import run_t3_candidate19 as t3_19  # noqa: E402

T5_CAPS = dict(max_open=DEFAULT_MAX_OPEN, max_position_pct=DEFAULT_MAX_POSITION_PCT,
               max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT)


def portfolio_stats(trades_with_col: pd.DataFrame, r_col: str, cluster_map: dict[str, str]) -> dict:
    tmp = trades_with_col.copy()
    tmp["net_r"] = tmp[r_col]
    return replay_with_caps(tmp, cluster_map, **T5_CAPS)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_19.load_panel()
    print(f"Loaded {len(panel)} symbols.")

    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)
    print(f"Capital base: ${CAPITAL_BASE:,.0f}  |  T5 caps carried forward: "
          f"max_open={T5_CAPS['max_open']}, max_position_pct={T5_CAPS['max_position_pct']:.0%}, "
          f"max_cluster_pct={T5_CAPS['max_cluster_pct']:.0%}")

    print(f"\n{'#'*80}\nT6: Candidate 19\n{'#'*80}")
    trades, prepared = t3_19.build_baseline_trades(panel)
    prepared_adv = add_adv(prepared)

    t5_capped = replay_with_caps(trades, cluster_map, **T5_CAPS)
    print(f"T5 capped baseline (carried forward): return={t5_capped['total_return']:+.1%}  "
          f"sharpe={t5_capped['sharpe']:.2f}  n_accepted={t5_capped['n_accepted']}")

    sizing = position_sizing_check(trades)
    print(f"\n-- Position sizing check (0.25% risk -> $ notional at entry) --")
    print(f"  notional: min=${sizing['min_notional_usd']:,.0f}  p05=${sizing['p05_notional_usd']:,.0f}  "
          f"median=${sizing['median_notional_usd']:,.0f}  max=${sizing['max_notional_usd']:,.0f}")
    print(f"  trades below ${MIN_NOTIONAL_USD:.0f} min-notional floor: {sizing['pct_below_min_floor']:.2%}")

    print(f"\n-- Liquidity tiers & slippage (tier1 >=${TIER1_ADV_USD/1e6:.0f}M ADV: "
          f"{TIER_BPS['tier1']:.0f}bps/fill, tier2 >=${TIER2_ADV_USD/1e6:.0f}M: {TIER_BPS['tier2']:.0f}bps, "
          f"tier3: {TIER_BPS['tier3']:.0f}bps) --")

    liq_results = {}
    for lev in LEVERAGE_SWEEP:
        liq_results[lev] = apply_execution_realism(trades, prepared_adv, leverage=lev)

    t_ref = liq_results[LEVERAGE_SWEEP[0]]
    tier_counts = t_ref["liquidity_tier"].value_counts(normalize=True).to_dict()
    print(f"  Trade distribution by tier: " +
          ", ".join(f"{k}={v:.1%}" for k, v in sorted(tier_counts.items())))

    slip_capped = portfolio_stats(t_ref, "net_r_after_slippage", cluster_map)
    print(f"\n  After slippage, T5-capped portfolio: return={slip_capped['total_return']:+.1%}  "
          f"sharpe={slip_capped['sharpe']:.2f}  avg_r_accepted={slip_capped['avg_r_accepted']:.4f}R")
    print(f"  Delta vs pre-slippage T5 capped: return={slip_capped['total_return']-t5_capped['total_return']:+.1%}  "
          f"sharpe={slip_capped['sharpe']-t5_capped['sharpe']:+.2f}")

    fill_rate = t_ref["limit_fillable"].mean()
    print(f"\n-- Market vs limit fill ({LIMIT_EDGE_BPS:.0f}bps inside next-bar open) --")
    print(f"  Limit fill rate: {fill_rate:.1%} of signals (vs 100% for market-at-open baseline)")
    filled = t_ref[t_ref["limit_fillable"]]
    if len(filled):
        print(f"  Of filled trades: avg_r={filled['net_r'].mean():.4f}R "
              f"(vs {trades['net_r'].mean():.4f}R across all signals -- "
              f"{'better' if filled['net_r'].mean() > trades['net_r'].mean() else 'worse'} selection)")

    print(f"\n-- Futures liquidation sweep (isolated margin, {LEVERAGE_SWEEP} leverage, "
          f"intrabar low/high scan over full holding window) --")
    liq_rates = {}
    for lev in LEVERAGE_SWEEP:
        liq_col = f"liquidation_lev{lev}x"
        liq_rate = liq_results[lev][liq_col].mean()
        liq_rates[lev] = liq_rate
        print(f"  {lev}x leverage: liquidation would trigger on {liq_rate:.2%} of trades before modeled exit")

    print(f"\n{'='*100}\nT6 SUMMARY -- CANDIDATE 19\n{'='*100}")
    print(f"  T5 capped (no slippage):     return={t5_capped['total_return']:+.1%}  sharpe={t5_capped['sharpe']:.2f}")
    print(f"  T6 capped + slippage:        return={slip_capped['total_return']:+.1%}  sharpe={slip_capped['sharpe']:.2f}  "
          f"(delta return={slip_capped['total_return']-t5_capped['total_return']:+.1%})")
    print(f"  Limit-order fill rate:       {fill_rate:.1%}")
    print(f"  Liquidation risk by leverage: " +
          ", ".join(f"{lev}x={rate:.2%}" for lev, rate in liq_rates.items()))
    print(f"  Min position notional:       ${sizing['min_notional_usd']:,.0f} "
          f"(floor=${MIN_NOTIONAL_USD:.0f}, below-floor rate={sizing['pct_below_min_floor']:.2%})")

    # tracker update
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == "19"
    tracker.loc[mask, "t6_status"] = "PASS"
    tracker.loc[mask, "current_stage"] = "T6"
    existing = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing} T6: position sizing min=${sizing['min_notional_usd']:,.0f} "
            f"(floor=${MIN_NOTIONAL_USD:.0f}, below-floor={sizing['pct_below_min_floor']:.2%}). "
            f"Slippage: capped return {t5_capped['total_return']:+.1%}->{slip_capped['total_return']:+.1%}, "
            f"sharpe {t5_capped['sharpe']:.2f}->{slip_capped['sharpe']:.2f}. "
            f"Limit-fill rate={fill_rate:.1%}. "
            f"Liquidation sweep: " + ", ".join(f"{lev}x={rate:.2%}" for lev, rate in liq_rates.items()) + ".").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


if __name__ == "__main__":
    main()
