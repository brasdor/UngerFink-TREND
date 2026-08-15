#!/usr/bin/env python3
"""
Diversification-gate rework for 11u and 13u -- both passed T1-T6 only to fail T7
strict robustness once their trade set was realistically capped to $60k/20-position
sizing (T4's remove-best-month diagnostic had already flagged, but not gated, that
24.0% and 39.7% of their total R came from the SAME single month, 2021-05). This
re-runs both candidates' T1 with tools.t1_harness.check_combo_diversification active
(max_month_share, max_asset_share) -- a per-combo gate, not the aggregate
check_year_concentration diagnostic -- so concentrated combos are excluded from T1
itself, before they can ever reach T7.

Thresholds: primary run at MONTH=0.25 / ASSET=0.20 (see t1_harness.py for rationale),
plus a sensitivity sweep across a small threshold grid so the choice is auditable, not
an arbitrary constant, matching check_year_concentration's own sensitivity-table
convention.

T0 is NOT re-run: it tests whether the candidate CONCEPT is redundant against existing
live systems using a single fixed proxy parameterization, entirely independent of which
T1 combos later pass a downstream gate -- re-running it would reproduce bit-identical
results to the original T0 pass already on record for 11/11u and 13/13u.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t1_harness import (  # noqa: E402
    run_t1_universe, DEFAULT_IS_START, DEFAULT_IS_END,
    MONTH_CONCENTRATION_SENSITIVITY_THRESHOLDS, ASSET_CONCENTRATION_SENSITIVITY_THRESHOLDS,
)
from tools.t2_regime_check import yearly_breakdown, regime_dependence_flag, cost_realism_check  # noqa: E402
from signal_generators import candidate_11_kaufman_ama as c11  # noqa: E402
from signal_generators import candidate_13_pullback_continuation as c13  # noqa: E402

UNIVERSE_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
PRIMARY_MAX_MONTH_SHARE = 0.25
PRIMARY_MAX_ASSET_SHARE = 0.20


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


def sharpe_of(trades: pd.DataFrame, trades_per_year: float) -> float:
    r = trades["net_r"]
    return (r.mean() / r.std(ddof=1) * np.sqrt(trades_per_year)
            if len(r) > 1 and r.std(ddof=1) > 0 else np.nan)


def sensitivity_sweep(grid: pd.DataFrame, label: str):
    """Derived post-hoc from a single ungated grid run (dominant_month_share/
    dominant_asset_share are computed regardless of whether the gate is active) --
    avoids re-running the full 290-symbol signal generation once per threshold pair."""
    print(f"\n-- {label}: threshold sensitivity sweep (n_pass at each month/asset cap) --")
    base = grid["gate_avgr"] & grid["gate_stability"]
    header = "asset\\month  " + "  ".join(f"{m:.2f}" for m in MONTH_CONCENTRATION_SENSITIVITY_THRESHOLDS)
    print(header)
    for a in ASSET_CONCENTRATION_SENSITIVITY_THRESHOLDS:
        row = []
        for m in MONTH_CONCENTRATION_SENSITIVITY_THRESHOLDS:
            month_ok = grid["dominant_month_share"].isna() | (grid["dominant_month_share"] <= m)
            asset_ok = grid["dominant_asset_share"].isna() | (grid["dominant_asset_share"] <= a)
            row.append(str(int((base & month_ok & asset_ok).sum())))
        print(f"{a:.2f}          " + "  ".join(f"{v:>4}" for v in row))


def run_candidate(candidate_id: str, name: str, generate_signal, panel, param_grids,
                   out_dir: Path, original_avg_r: float, original_sharpe: float, original_n: int):
    print(f"\n{'#'*80}\n{name} -- T1 WITH DIVERSIFICATION GATE "
          f"(max_month_share={PRIMARY_MAX_MONTH_SHARE}, max_asset_share={PRIMARY_MAX_ASSET_SHARE})\n{'#'*80}")

    result = run_t1_universe(
        candidate_id=candidate_id, candidate_name=name, generate_signal=generate_signal,
        price_panel=panel, param_grids=param_grids, asset_class="futures",
        out_dir=out_dir, update_tracker=False,
        max_month_share=PRIMARY_MAX_MONTH_SHARE, max_asset_share=PRIMARY_MAX_ASSET_SHARE,
    )

    grid = result.grid
    sensitivity_sweep(grid, candidate_id)
    n_total = len(grid)
    n_pass_avgr_stability = int((grid["gate_avgr"] & grid["gate_stability"]).sum())
    n_pass_all = result.n_pass
    n_gate_diversification_kills = n_pass_avgr_stability - n_pass_all
    print(f"Grid: {n_total} combos. Pass avg_r+stability gates (pre-diversification): "
          f"{n_pass_avgr_stability}. Pass ALL gates (incl. diversification): {n_pass_all}.")
    print(f"Diversification gate excluded {n_gate_diversification_kills} combo(s) that "
          f"would otherwise have passed T1.")

    if n_pass_all == 0:
        print(f"\n{name}: NO combos survive with the diversification gate active -- CANDIDATE FAILS T1.")
        return dict(candidate_id=candidate_id, survives=False, n_pass=0)

    passing = grid[grid["all_gates"]]
    print("\nSurviving combos (with per-combo concentration):")
    cols = list(param_grids.keys()) + ["avg_r", "n", "dominant_month_share", "dominant_asset_share"]
    print(passing[cols].to_string(index=False))

    robustness = result.robustness
    survives = bool(robustness["survives"].any()) if len(robustness) else True
    print(f"\nRobustness (drop-top-2): {int(robustness['survives'].sum())}/{len(robustness)} survive")
    print(f"edge_clustering: {result.edge_clustering}")
    print(f"thin_sliver: {result.thin_sliver}")
    print(f"year_concentration (pooled, aggregate diagnostic): {result.year_concentration}")
    print(f"T1 VERDICT: {'PASS' if survives else 'FAIL'}")

    if not survives:
        return dict(candidate_id=candidate_id, survives=False, n_pass=n_pass_all)

    # Pool trades from the gate-surviving combos for T2 and for the return/Sharpe comparison.
    # trade_store isn't returned on T1Result -- rebuild pooled trades directly via generate_signal.
    from tools.t2_regime_check import trades_for_combo_universe
    survivor_rows = robustness[robustness["survives"]]
    param_names = list(param_grids.keys())
    pooled_trades = []
    for _, row in survivor_rows.iterrows():
        params = {p: (int(row[p]) if isinstance(param_grids[p][0], (int, np.integer)) else row[p])
                  for p in param_names}
        tdf = trades_for_combo_universe(generate_signal, params, panel, DEFAULT_IS_START, DEFAULT_IS_END)
        pooled_trades.append(tdf)
    pooled = pd.concat(pooled_trades, ignore_index=True)

    is_years = max((DEFAULT_IS_END - DEFAULT_IS_START).days / 365.25, 1e-6)
    gated_avg_r = pooled["net_r"].mean()
    gated_n = len(pooled)
    gated_sharpe = sharpe_of(pooled, gated_n / is_years)

    print(f"\n-- T2: regime/cost-realism check on gated survivor pool --")
    yearly = yearly_breakdown(pooled)
    print(yearly.to_string())
    regime = regime_dependence_flag(yearly)
    cost = cost_realism_check(pooled, base_floor=0.25)
    print(f"Regime dependence: flag={regime['flag']}  reason={regime.get('reason')}")
    print(f"Cost realism (base+0.05R floor): avg_r={cost['avg_r']:.4f}R vs stricter floor "
          f"{cost['stricter_floor']:.2f}R -> clears={cost['clears_stricter_floor']}  t_stat={cost['t_stat']:.2f}")
    t2_verdict = "T2_PASS" if (not regime["flag"] and cost["clears_stricter_floor"]) else "T2_CAUTION"
    print(f"T2 VERDICT: {t2_verdict}")

    print(f"\n-- Return/Sharpe given up vs original ungated survivor pool --")
    print(f"  Original (ungated): n={original_n}, avg_r={original_avg_r:.4f}R, sharpe={original_sharpe:.2f}")
    print(f"  Gated:              n={gated_n}, avg_r={gated_avg_r:.4f}R, sharpe={gated_sharpe:.2f}")
    print(f"  Delta:              n={gated_n-original_n:+d} ({(gated_n/original_n-1):+.1%}), "
          f"avg_r={gated_avg_r-original_avg_r:+.4f}R, sharpe={gated_sharpe-original_sharpe:+.2f}")

    return dict(candidate_id=candidate_id, survives=True, n_pass=n_pass_all,
                gated_avg_r=gated_avg_r, gated_sharpe=gated_sharpe, gated_n=gated_n,
                t2_verdict=t2_verdict, regime=regime, cost=cost)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print("\nT0: NOT re-run -- redundancy-vs-existing-systems check uses a fixed proxy "
          "parameterization, independent of which T1 combos later pass. Original T0 "
          "verdicts (11: not redundant vs S1/S6/S7/S8; 13: not redundant, priority-checked "
          "vs S1/Donchian-family) carry forward unchanged.")

    res11u = run_candidate(
        "11u", "Kaufman AMA adaptive-lookback trend (cross-sectional, diversification-gated)",
        c11.generate_signal, panel, c11.PARAM_GRID,
        ROOT / "data" / "research_candidate11_kaufman_ama_universe_t1_diversification_gated",
        original_avg_r=0.4629, original_sharpe=6.19, original_n=52760,
    )
    res13u = run_candidate(
        "13u", "Pullback continuation (cross-sectional, diversification-gated)",
        c13.generate_signal, panel, c13.PARAM_GRID,
        ROOT / "data" / "research_candidate13_pullback_continuation_universe_t1_diversification_gated",
        original_avg_r=0.6937, original_sharpe=3.54, original_n=16670,
    )

    print(f"\n{'='*100}\nDIVERSIFICATION-GATE REWORK SUMMARY\n{'='*100}")
    for res in (res11u, res13u):
        print(f"\n{res['candidate_id']}: {'PASSES' if res['survives'] else 'FAILS'} T1 with gate "
              f"(n_pass={res['n_pass']})")
        if res["survives"]:
            print(f"  gated avg_r={res['gated_avg_r']:.4f}R, sharpe={res['gated_sharpe']:.2f}, "
                  f"n={res['gated_n']}, T2={res['t2_verdict']}")


if __name__ == "__main__":
    main()
