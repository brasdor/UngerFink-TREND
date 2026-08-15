#!/usr/bin/env python3
"""
Candidate 12 vs candidate 19 return-stream correlation -- same methodology as the
earlier 11u-vs-13u check: full-period Pearson correlation AND rolling-window (63-bar,
252-bar) MAX correlation (tools/t0_triage.py's max_rolling_corr), since the full-period
number alone was insufficient for 11u/13u (low full-period, high rolling overlap during
trending regimes -- exactly the kind of thing a single aggregate number hides).

Return-stream proxy: each candidate's LOCKED T7 trade set (T5-accepted, T6 slippage-
adjusted), net_r summed by exit date and reindexed to a daily calendar. This is a
dollar-scale-invariant proxy -- correlation is unaffected by position sizing, so daily
R-contribution is exactly as valid for THIS purpose as a full dollar equity curve, and
reuses the exact locked trade sets already validated through T7 rather than building a
new proxy construction.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.t0_triage import max_rolling_corr, WINDOWS  # noqa: E402
from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END  # noqa: E402
from tools.t5_portfolio_replay import build_symbol_cluster_map  # noqa: E402
from tools.t6_capital_execution import add_adv, apply_execution_realism  # noqa: E402

t3_12 = importlib.import_module("run_t3_candidate12")
t3_19 = importlib.import_module("run_t3_candidate19")
t7_all = importlib.import_module("run_t7_all")
_accepted_subset = t7_all._accepted_subset

LOCKED_LEVERAGE_FOR_SLIPPAGE = 3


def build_locked_daily_r(build_baseline_trades_fn, panel: dict, cluster_map: dict) -> pd.Series:
    trades, prepared = build_baseline_trades_fn(panel)
    prepared_adv = add_adv(prepared)
    t_realism = apply_execution_realism(trades, prepared_adv, leverage=LOCKED_LEVERAGE_FOR_SLIPPAGE)

    sorted_trades = trades.sort_values("entry_date").reset_index(drop=True)
    sorted_realism = t_realism.sort_values("entry_date").reset_index(drop=True)
    accepted = _accepted_subset(sorted_trades, cluster_map)
    locked = sorted_realism.loc[accepted.index].copy()
    locked["net_r"] = locked["net_r_after_slippage"]

    daily_r = locked.groupby(pd.to_datetime(locked["exit_date"]))["net_r"].sum()
    daily_idx = pd.date_range(DEFAULT_IS_START, DEFAULT_IS_END, freq="D")
    return daily_r.reindex(daily_idx).fillna(0.0), len(locked)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = t3_12.load_panel()
    print(f"Loaded {len(panel)} symbols.")
    cluster_map = build_symbol_cluster_map(panel, DEFAULT_IS_START, DEFAULT_IS_END)

    print("\nBuilding candidate 12's locked T7 daily-R return stream...")
    r12, n12 = build_locked_daily_r(t3_12.build_baseline_trades, panel, cluster_map)
    print(f"  n_trades={n12}, nonzero days={int((r12 != 0).sum())}")

    print("Building candidate 19's locked T7 daily-R return stream...")
    r19, n19 = build_locked_daily_r(t3_19.build_baseline_trades, panel, cluster_map)
    print(f"  n_trades={n19}, nonzero days={int((r19 != 0).sum())}")

    joined = pd.concat([r12.rename("c12"), r19.rename("c19")], axis=1)
    full_corr = joined["c12"].corr(joined["c19"])

    roll63 = joined["c19"].rolling(63, min_periods=16).corr(joined["c12"])
    roll252 = joined["c19"].rolling(252, min_periods=63).corr(joined["c12"])
    max63 = max_rolling_corr(r19, r12, 63)
    max252 = max_rolling_corr(r19, r12, 252)

    print(f"\n{'='*100}\nCANDIDATE 12 vs CANDIDATE 19 -- RETURN STREAM CORRELATION\n{'='*100}")
    print(f"Full-period Pearson correlation: {full_corr:.4f}")
    print(f"Rolling 63-bar:  mean={roll63.mean():.4f}  max={max63:.4f}  min={roll63.min():.4f}")
    print(f"Rolling 252-bar: mean={roll252.mean():.4f}  max={max252:.4f}  min={roll252.min():.4f}")
    print(f"\nT0-style redundancy read (0.7 max-window threshold, informational -- both "
          f"already independently validated through T7): "
          f"{'WOULD FLAG' if max(max63, max252) > 0.7 else 'would NOT flag'} redundant")


if __name__ == "__main__":
    main()
