#!/usr/bin/env python3
"""
T3 exit engineering for candidate 19 (52-week-high / proximity-to-high momentum) --
entry logic completely frozen as validated at T1/T2 (diversification-gated: 8/9
gate-surviving combos also survive drop-top-2). Same methodology as
run_t3_candidate12.py: tests three exit variants against the current flat/opposite-
signal exit, reusing _gate_and_finish with cost_floor=baseline_avg_r so the entire
hardened check suite (zone stability, drop-top-2 robustness, edge_clustering,
thin_sliver, year_concentration) applies unchanged.

Candidate 19 uses the generate_universe_positions(panel, params) ranking interface
(not per-symbol generate_signal), so baseline trades are pooled via the same pattern
used in tools/run_candidates_18_19_20.py's pool_trades_for_ranked_combos, rather than
run_t3_candidate12.py's per-symbol trade-store reuse.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t1_harness import (  # noqa: E402
    DEFAULT_IS_START, DEFAULT_IS_END, _add_atr, _positions_to_trades, _combo_stats, _gate_and_finish,
)
from tools.t3_exit_engine import add_atr_at_entry, apply_exit_variant  # noqa: E402
from signal_generators import candidate_19_proximity_high_momentum as c19  # noqa: E402

UNIVERSE_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
WARMUP_DAYS = 260
MIN_SYMBOL_BARS = 200

BREAKEVEN_GRID = {"trigger_r": [0.5, 1.0, 1.5]}
CHANDELIER_GRID = {"activate_r": [1.5, 2.0, 2.5], "atr_mult": [2.5, 3.5]}
PARTIAL_GRID = {"trigger_r": [1.5, 2.0, 3.0], "fraction": [0.33, 0.5, 0.67]}


def load_panel() -> dict[str, pd.DataFrame]:
    panel = {}
    for f in sorted(UNIVERSE_DIR.glob("*_1d.csv")):
        symbol = f.stem.replace("_1d", "")
        df = pd.read_csv(f)
        if "date" not in df.columns or not {"open", "high", "low", "close"} <= set(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        panel[symbol] = df[["open", "high", "low", "close", "volume"]]
    return panel


def build_baseline_trades(panel: dict) -> tuple[pd.DataFrame, dict]:
    slice_start = DEFAULT_IS_START - pd.Timedelta(days=WARMUP_DAYS)
    prepared = {}
    for sym, df in panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= DEFAULT_IS_END)]
        if len(sub) >= MIN_SYMBOL_BARS:
            prepared[sym] = _add_atr(sub)

    rob = pd.read_csv(ROOT / "data/research_candidate19_t1/t1_robustness.csv")
    survivors = rob[rob["survives"]]
    all_trades = []
    for _, row in survivors.iterrows():
        params = dict(lookback=int(row["lookback"]), quantile=float(row["quantile"]),
                      rebalance_n=int(row["rebalance_n"]), side=row["side"])
        positions_by_symbol = c19.generate_universe_positions(panel, params)
        for sym, sub_atr in prepared.items():
            pos = positions_by_symbol.get(sym)
            if pos is None:
                continue
            pos = pos.reindex(sub_atr.index).fillna(0).astype(int)
            tdf = _positions_to_trades(sub_atr, pos, DEFAULT_IS_START, DEFAULT_IS_END)
            if len(tdf):
                tdf = tdf.copy()
                tdf["symbol"] = sym
                all_trades.append(tdf)
    trades = pd.concat(all_trades, ignore_index=True)
    trades = add_atr_at_entry(trades, prepared)
    return trades, prepared


def stats_for(trades: pd.DataFrame, is_years: float) -> dict:
    st = _combo_stats(trades, is_years)
    trades_per_year = st["trades_per_year"]
    r = trades["net_r"]
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(trades_per_year)
              if len(r) > 1 and r.std(ddof=1) > 0 else np.nan)
    st["sharpe"] = sharpe
    return st


def run_variant_grid(label: str, mode: str, param_grid: dict, baseline_trades: pd.DataFrame,
                      prepared: dict, is_years: float, baseline_avg_r: float, baseline_sharpe: float):
    print(f"\n{'='*70}\nCandidate 19 T3: {label}\n{'='*70}")
    param_names = list(param_grid.keys())
    rows, trade_store = [], {}
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_names, combo))
        new_trades = apply_exit_variant(baseline_trades, prepared, mode, params)
        st = stats_for(new_trades, is_years)
        st.update(params)
        rows.append(st)
        trade_store[tuple(combo)] = new_trades

    result = _gate_and_finish(
        candidate_id=f"19_T3_{mode}", candidate_name=f"Candidate 19 T3 {label}",
        rows=rows, trade_store=trade_store, param_names=param_names,
        param_grids=param_grid, cost_floor=baseline_avg_r,
        actual_start=DEFAULT_IS_START, actual_end=DEFAULT_IS_END, window_adjusted=False,
        out_dir=ROOT / "data" / f"research_candidate19_t3_{mode}",
        update_tracker=False,
    )

    grid_display = result.grid.sort_values("avg_r", ascending=False)
    print(f"Baseline avg_r={baseline_avg_r:.4f}R, baseline_sharpe={baseline_sharpe:.2f}")
    print(grid_display[param_names + ["n", "avg_r", "sharpe", "t_stat", "zone_frac", "all_gates"]]
          .to_string(index=False))
    print(f"\n{result.n_pass}/{len(result.grid)} combos beat baseline avg_r "
          f"({baseline_avg_r:.4f}R) with stability >=67%")

    survives = False
    if len(result.robustness):
        r = result.robustness
        survives = bool(r["survives"].any())
        print(f"Robustness (drop-top-2 vs baseline floor): {int(r['survives'].sum())}/{len(r)} survive")
    elif result.n_pass:
        survives = True

    print(f"edge_clustering: {result.edge_clustering}")
    print(f"thin_sliver: {result.thin_sliver}")
    print(f"year_concentration: {result.year_concentration}")
    print(f"VERDICT ({label}): {'IMPROVES on baseline, robust' if survives else 'does NOT robustly improve'}")

    best_row = grid_display.iloc[0] if len(grid_display) else None
    if best_row is not None:
        print(f"Best combo: {dict((k, best_row[k]) for k in param_names)} -> "
              f"avg_r={best_row['avg_r']:.4f}R (vs baseline {baseline_avg_r:.4f}R), "
              f"sharpe={best_row['sharpe']:.2f} (vs baseline {baseline_sharpe:.2f})")
    return result, survives


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print("\n>>> Building baseline trades (frozen entry, current flat/opposite-signal exit)...")
    baseline_trades, prepared = build_baseline_trades(panel)
    is_years = max((DEFAULT_IS_END - DEFAULT_IS_START).days / 365.25, 1e-6)
    baseline_stats = stats_for(baseline_trades, is_years)
    baseline_avg_r = baseline_stats["avg_r"]
    baseline_sharpe = baseline_stats["sharpe"]
    print(f"Baseline: n={baseline_stats['n']} trades, avg_r={baseline_avg_r:.4f}R, "
          f"t_stat={baseline_stats['t_stat']:.2f}, sharpe={baseline_sharpe:.2f}")

    results = {}
    results["breakeven"] = run_variant_grid("Breakeven after +1R", "breakeven", BREAKEVEN_GRID,
                                             baseline_trades, prepared, is_years,
                                             baseline_avg_r, baseline_sharpe)
    results["chandelier"] = run_variant_grid("ATR/chandelier trailing stop", "chandelier", CHANDELIER_GRID,
                                              baseline_trades, prepared, is_years,
                                              baseline_avg_r, baseline_sharpe)
    results["partial"] = run_variant_grid("Partial profit-taking", "partial", PARTIAL_GRID,
                                           baseline_trades, prepared, is_years,
                                           baseline_avg_r, baseline_sharpe)

    print(f"\n{'='*70}\nCANDIDATE 19 T3 SUMMARY\n{'='*70}")
    print(f"Baseline (current flat exit): avg_r={baseline_avg_r:.4f}R, sharpe={baseline_sharpe:.2f}")
    for label, (result, survives) in results.items():
        print(f"  {label}: {'IMPROVES, robust' if survives else 'no robust improvement'} "
              f"({result.n_pass}/{len(result.grid)} beat baseline avg_r)")


if __name__ == "__main__":
    main()
