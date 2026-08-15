#!/usr/bin/env python3
"""
Stop-loss variant test for candidates 12 and 19 -- both PASS T7 on their locked
variants, but both showed real leverage/liquidation sensitivity at T6 (12: 21.6%
liquidated at 3x; 19: 18.3% at 3x). Both share the same design flaw driving this
work: no protective stop order is ever executed on either candidate -- ATR x 2.0 is
only a position-sizing/R-normalization unit, exits are purely signal-driven, so
adverse excursions during a trade's hold are entirely uncapped by the strategy
itself. This tests whether adding an actual executed stop meaningfully reduces
liquidation exposure without destroying the edge (avg_r/Sharpe, T7 robustness).

Applied identically to both candidates' LOCKED T7 trade sets (T5-accepted, T6
slippage-adjusted, entry+exit otherwise frozen) so results are directly comparable.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END  # noqa: E402
from tools.t5_portfolio_replay import (  # noqa: E402
    build_symbol_cluster_map, DEFAULT_MAX_OPEN, DEFAULT_MAX_POSITION_PCT, DEFAULT_MAX_CLUSTER_PCT,
)
from tools.t6_capital_execution import add_adv, apply_execution_realism, LEVERAGE_SWEEP  # noqa: E402
from tools.t7_strict_robustness import run_t7_suite, BASELINE_COST_R  # noqa: E402
from tools.stop_loss_variants import (  # noqa: E402
    compute_mae_r_distribution, apply_fixed_r_stop, apply_atr_mult_stop, apply_time_stop, ATR_MULT_R,
)

t3_12 = importlib.import_module("run_t3_candidate12")
t3_19 = importlib.import_module("run_t3_candidate19")
t7_all = importlib.import_module("run_t7_all")
_accepted_subset = t7_all._accepted_subset

T5_CAPS = dict(max_open=DEFAULT_MAX_OPEN, max_position_pct=DEFAULT_MAX_POSITION_PCT,
               max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT)
LOCKED_LEVERAGE_FOR_SLIPPAGE = 3  # reuses apply_execution_realism's slippage-tier machinery only


def build_locked_trades(build_baseline_trades_fn, panel: dict, cluster_map: dict):
    trades, prepared = build_baseline_trades_fn(panel)
    prepared_adv = add_adv(prepared)
    t_realism = apply_execution_realism(trades, prepared_adv, leverage=LOCKED_LEVERAGE_FOR_SLIPPAGE)

    sorted_trades = trades.sort_values("entry_date").reset_index(drop=True)
    sorted_realism = t_realism.sort_values("entry_date").reset_index(drop=True)
    accepted = _accepted_subset(sorted_trades, cluster_map)
    locked = sorted_realism.loc[accepted.index].copy()
    locked["net_r"] = locked["net_r_after_slippage"]
    return locked.reset_index(drop=True), prepared, prepared_adv


def stats(trades: pd.DataFrame, is_years: float) -> dict:
    r = trades["net_r"]
    n = len(trades)
    avg_r = r.mean() if n else np.nan
    trades_per_year = n / is_years
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(trades_per_year)
              if n > 1 and r.std(ddof=1) > 0 else np.nan)
    return dict(n=n, avg_r=avg_r, sharpe=sharpe)


def liquidation_rates(trades: pd.DataFrame, prepared_adv: dict) -> dict[int, float]:
    rates = {}
    for lev in LEVERAGE_SWEEP:
        t = apply_execution_realism(trades, prepared_adv, leverage=lev)
        rates[lev] = t[f"liquidation_lev{lev}x"].mean()
    return rates


def evaluate_variant(label: str, new_trades: pd.DataFrame, is_years: float, prepared_adv: dict) -> dict:
    if "exit_reason" in new_trades.columns:
        fire_rate = (new_trades["exit_reason"] != "ORIGINAL").mean()
    else:
        fire_rate = 0.0  # baseline (no stop simulation applied, no exit_reason column)
    st = stats(new_trades, is_years)
    suite = run_t7_suite(new_trades)
    liq = liquidation_rates(new_trades, prepared_adv)
    return dict(
        label=label, fire_rate=fire_rate, n=st["n"], avg_r=st["avg_r"], sharpe=st["sharpe"],
        cost15_avg_r=suite["cost"][-1]["avg_r"], cost15_clears=suite["cost"][-1]["clears_floor"],
        rm5_avg_r=suite["rm_assets"][-1]["avg_r"], rm5_clears=suite["rm_assets"][-1]["clears_floor"],
        rm2m_avg_r=suite["rm_months"][-1]["avg_r"], rm2m_clears=suite["rm_months"][-1]["clears_floor"],
        rolling_below_floor=suite["rolling_below_floor_frac"],
        liq_3x=liq[3], liq_5x=liq[5], liq_10x=liq[10],
    )


def print_table(rows: list[dict]):
    header = (f"{'variant':<22} {'fire%':>7} {'n':>5} {'avg_r':>8} {'sharpe':>7} "
              f"{'cost+.15':>9} {'ok':>3} {'rm5':>8} {'ok':>3} {'rm2m':>8} {'ok':>3} "
              f"{'roll<flr':>9} {'liq3x':>7} {'liq5x':>7} {'liq10x':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<22} {r['fire_rate']:>6.1%} {r['n']:>5} {r['avg_r']:>8.4f} "
              f"{r['sharpe']:>7.2f} {r['cost15_avg_r']:>9.4f} {str(r['cost15_clears'])[0]:>3} "
              f"{r['rm5_avg_r']:>8.4f} {str(r['rm5_clears'])[0]:>3} "
              f"{r['rm2m_avg_r']:>8.4f} {str(r['rm2m_clears'])[0]:>3} "
              f"{r['rolling_below_floor']:>8.1%} {r['liq_3x']:>6.1%} {r['liq_5x']:>6.1%} {r['liq_10x']:>6.1%}")


def run_for_candidate(cid: str, build_baseline_trades_fn, panel: dict, cluster_map: dict):
    print(f"\n{'#'*100}\nSTOP-LOSS VARIANT TEST -- Candidate {cid}\n{'#'*100}")
    locked, prepared, prepared_adv = build_locked_trades(build_baseline_trades_fn, panel, cluster_map)
    is_years = max((DEFAULT_IS_END - DEFAULT_IS_START).days / 365.25, 1e-6)

    baseline_row = evaluate_variant("BASELINE (no stop)", locked, is_years, prepared_adv)
    print(f"Locked trade set: n={len(locked)}, avg_r={baseline_row['avg_r']:.4f}R, "
          f"sharpe={baseline_row['sharpe']:.2f}")

    mae_r = compute_mae_r_distribution(locked, prepared)
    p1, p2 = np.percentile(mae_r, 1), np.percentile(mae_r, 2)
    hold_days = (pd.to_datetime(locked["exit_date"]) - pd.to_datetime(locked["entry_date"])).dt.days
    p75_hold, p90_hold = int(hold_days.quantile(0.75)), int(hold_days.quantile(0.90))
    print(f"MAE_R distribution: 1st pct={p1:.2f}R, 2nd pct={p2:.2f}R  "
          f"(these become the percentile-calibrated stop levels)")
    print(f"Holding period: median={hold_days.median():.0f}d, P75={p75_hold}d, P90={p90_hold}d "
          f"(these become the time-stop N values)")

    variants = [
        ("fixed_r_-4R", lambda: apply_fixed_r_stop(locked, prepared, r_mult=4.0)),
        ("fixed_r_-6R", lambda: apply_fixed_r_stop(locked, prepared, r_mult=6.0)),
        ("fixed_r_-8R", lambda: apply_fixed_r_stop(locked, prepared, r_mult=8.0)),
        (f"pctile_1_({abs(p1):.1f}R)", lambda: apply_fixed_r_stop(locked, prepared, r_mult=abs(p1))),
        (f"pctile_2_({abs(p2):.1f}R)", lambda: apply_fixed_r_stop(locked, prepared, r_mult=abs(p2))),
        ("atr_mult_4x_(=2R)", lambda: apply_atr_mult_stop(locked, prepared, atr_mult=4.0)),
        ("atr_mult_6x_(=3R)", lambda: apply_atr_mult_stop(locked, prepared, atr_mult=6.0)),
        (f"time_stop_P75({p75_hold}d)", lambda: apply_time_stop(locked, prepared, n_days=p75_hold)),
        (f"time_stop_P90({p90_hold}d)", lambda: apply_time_stop(locked, prepared, n_days=p90_hold)),
        ("bar_close_-4R", lambda: apply_fixed_r_stop(locked, prepared, r_mult=4.0, confirm_close=True)),
    ]

    rows = [baseline_row]
    for label, fn in variants:
        new_trades = fn()
        rows.append(evaluate_variant(label, new_trades, is_years, prepared_adv))

    print()
    print_table(rows)
    return rows


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_12.load_panel()
    print(f"Loaded {len(panel)} symbols.")
    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)

    print(f"\nNOTE: this system's own R-unit IS ATR_MULT_R({ATR_MULT_R}) x atr -- so "
          f"'atr_mult=4.0/6.0' stops are mathematically identical to fixed_r stops of "
          f"2.0R/3.0R, TIGHTER than even the -4R catastrophic tier. Reported explicitly, "
          f"not hidden, since this cuts against reading 'wider ATR multiple' as wider "
          f"than the R-multiple variants.")

    rows_12 = run_for_candidate("12", t3_12.build_baseline_trades, panel, cluster_map)
    rows_19 = run_for_candidate("19", t3_19.build_baseline_trades, panel, cluster_map)

    print(f"\n{'='*100}\nSTOP-LOSS TEST SUMMARY\n{'='*100}")
    for cid, rows in (("12", rows_12), ("19", rows_19)):
        base = rows[0]
        print(f"\nCandidate {cid} baseline: avg_r={base['avg_r']:.4f}R, sharpe={base['sharpe']:.2f}, "
              f"liq@3x/5x/10x={base['liq_3x']:.1%}/{base['liq_5x']:.1%}/{base['liq_10x']:.1%}")
        for r in rows[1:]:
            avg_r_retained = r['avg_r'] / base['avg_r'] if base['avg_r'] else np.nan
            print(f"  {r['label']:<22} fire={r['fire_rate']:.1%}  avg_r_retained={avg_r_retained:.1%}  "
                  f"liq@3x={r['liq_3x']:.1%} (delta={r['liq_3x']-base['liq_3x']:+.1%})  "
                  f"T7_ok={r['cost15_clears'] and r['rm5_clears'] and r['rm2m_clears']}")


if __name__ == "__main__":
    main()
