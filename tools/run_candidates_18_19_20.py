#!/usr/bin/env python3
"""
Phase 2: three alternate momentum constructions, independent from candidate 12's raw
rank-momentum design -- T0 (redundancy vs S1/S6/S7/S8 AND vs candidate 12 specifically,
given conceptual overlap risk) then T1/T2, cross-sectional on the 290-symbol futures
universe, with check_combo_diversification active FROM THE START (not opt-in after the
fact -- Phase 1 showed 11u/13u passed T1-T6 only to fail T7 on concentration that a
per-combo gate would have caught immediately).

  18 -- idiosyncratic/residual momentum (BTC-beta stripped before ranking)
  19 -- 52-week-high / proximity-to-high momentum (different formation mechanism entirely)
  20 -- momentum turning points (dual-speed TSM blend gates the ranking eligibility pool)

See signal_generators/candidate_{18,19,20}_*.py for full construction rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t0_triage import run_triage, append_triage_log, max_rolling_corr, WINDOWS  # noqa: E402
from tools.t1_harness import (  # noqa: E402
    run_t1_universe_ranked, DEFAULT_IS_START, DEFAULT_IS_END, BATCH_TRACKER,
)
from tools.t2_regime_check import yearly_breakdown, regime_dependence_flag, cost_realism_check  # noqa: E402
from signal_generators import candidate_12_cross_sectional_momentum as c12  # noqa: E402
from signal_generators import candidate_18_residual_momentum as c18  # noqa: E402
from signal_generators import candidate_19_proximity_high_momentum as c19  # noqa: E402
from signal_generators import candidate_20_momentum_turning_points as c20  # noqa: E402

UNIVERSE_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
MAX_MONTH_SHARE = 0.25
MAX_ASSET_SHARE = 0.20

CANDIDATES = {
    "18": dict(name="Idiosyncratic/residual momentum (BTC-beta stripped)", module=c18,
               t0_params=dict(formation=60, quantile=0.2, rebalance_n=7, side="long_short")),
    "19": dict(name="52-week-high / proximity-to-high momentum", module=c19,
               t0_params=dict(lookback=180, quantile=0.2, rebalance_n=7, side="long_short")),
    "20": dict(name="Momentum turning points (dual-speed TSM blend gate)", module=c20,
               t0_params=dict(slow_n=180, fast_n=30, quantile=0.2, rebalance_n=7, side="long_short")),
}
C12_T0_PARAMS = dict(formation=60, quantile=0.2, rebalance_n=7, side="long_short")


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


def aggregate_portfolio_return(panel: dict, positions_by_symbol: dict) -> pd.Series:
    """Mean of each symbol's shifted-position x own-return, across the universe --
    matches run_candidate12.py's T0 proxy exactly (no single-asset proxy exists for
    a cross-sectional ranking construction)."""
    per_symbol_ret = {}
    for sym, df in panel.items():
        pos = positions_by_symbol.get(sym)
        if pos is None:
            continue
        ret = df["close"].pct_change()
        per_symbol_ret[sym] = (pos.shift(1).fillna(0) * ret).reindex(df.index)
    return pd.DataFrame(per_symbol_ret).mean(axis=1, skipna=True).dropna()


def run_t0(candidate_id: str, panel: dict, c12_proxy_ret: pd.Series) -> tuple[str, pd.Series]:
    spec = CANDIDATES[candidate_id]
    print(f"\n{'='*70}\nCandidate {candidate_id} ({spec['name']}): T0 triage\n{'='*70}")
    positions = spec["module"].generate_universe_positions(panel, spec["t0_params"])
    proxy_ret = aggregate_portfolio_return(panel, positions)

    row = run_triage(candidate_id, proxy_ret,
                      notes=f"T0 proxy signal: {spec['t0_params']} (aggregate cross-sectional "
                            f"portfolio return). Priority check vs candidate 12 given conceptual "
                            f"overlap risk (both are cross-sectional momentum constructions).")
    append_triage_log(row)

    # priority check vs candidate 12 specifically -- NOT part of the standard S1/S6/S7/S8
    # redundancy gate (that's about live-system overlap; the 0.7 max-rolling-window rule
    # was calibrated for testing against genuinely different trading styles, not sibling
    # concepts in the same momentum family where some overlap is expected). Reported for
    # every candidate, but INFORMATIONAL ONLY, not gating -- diagnosed after the initial
    # run showed wildly different overlap profiles under this same threshold (18: 0.87
    # average correlation, genuinely redundant; 20: 0.37 average, briefly spiking to 0.88
    # in one window). Letting T1's own hardened gates (incl. the diversification gate) be
    # the real filter, per explicit instruction, rather than pre-filtering on correlation.
    joined_c12 = pd.concat([proxy_ret.rename("c"), c12_proxy_ret.rename("c12")], axis=1).dropna()
    full_corr_vs_c12 = joined_c12["c"].corr(joined_c12["c12"])
    roll63_vs_c12 = joined_c12["c"].rolling(63, min_periods=16).corr(joined_c12["c12"])
    corrs_vs_c12 = [max_rolling_corr(proxy_ret, c12_proxy_ret, w) for w in WINDOWS]
    corrs_vs_c12_valid = [c for c in corrs_vs_c12 if pd.notna(c)]
    worst_vs_c12 = max(corrs_vs_c12_valid) if corrs_vs_c12_valid else float("nan")

    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    if not (tracker["candidate_id"] == candidate_id).any():
        blank = {c: "" for c in tracker.columns}
        blank.update(candidate_id=candidate_id, name=spec["name"], batch=5,
                      side="long_short", current_stage="not_started")
        tracker = pd.concat([tracker, pd.DataFrame([blank])], ignore_index=True)
    mask = tracker["candidate_id"] == candidate_id
    tracker.loc[mask, "t0_status"] = row["verdict"]
    if row["verdict"] == "REDUNDANT":
        tracker.loc[mask, "current_stage"] = "T0_FAILED"
    tracker.loc[mask, "notes"] = (
        f"vs candidate 12 (informational, not gating -- see run_candidates_18_19_20.py): "
        f"full-series corr={full_corr_vs_c12:.3f}, rolling-63d mean={roll63_vs_c12.mean():.3f}, "
        f"max-window corr={worst_vs_c12:.3f}."
    )
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)

    print(f"T0 verdict (vs S1/S6/S7/S8): {row['verdict']}")
    for sid in ("S1", "S6", "S7", "S8"):
        corr = row[f"corr_{sid}"]
        corr_str = f"{corr:.3f}" if pd.notna(corr) else "n/a"
        print(f"  vs {sid}: corr={corr_str}  confidence={row[f'corr_{sid}_confidence']}")
    print(f"  vs candidate 12 (informational, not gating): full-series corr={full_corr_vs_c12:.3f}, "
          f"rolling-63d mean={roll63_vs_c12.mean():.3f}, max-window corr={worst_vs_c12:.3f}")
    print(f"  notes: {row['notes']}")

    # Only the standard S1/S6/S7/S8 check gates T0 -- vs-candidate-12 is reported above
    # for context but does not block T1/T2, per explicit instruction.
    return row["verdict"], proxy_ret


def pool_trades_for_ranked_combos(generate_universe_positions, panel: dict, param_grid: dict,
                                   param_names: list[str], survivor_rows: pd.DataFrame,
                                   warmup_days: int = 260, min_symbol_bars: int = 200) -> pd.DataFrame:
    """T2 trade-pooling for the generate_universe_positions(panel, params) interface --
    trades_for_combo_universe (tools/t2_regime_check.py) assumes a per-symbol
    generate_signal(df, params) call and cannot be reused for a ranking construction
    that needs the whole panel at once, so this mirrors run_t1_universe_ranked's own
    per-combo trade-building loop, restricted to the drop-top-2-surviving combos."""
    from tools.t1_harness import _add_atr, _positions_to_trades

    slice_start = DEFAULT_IS_START - pd.Timedelta(days=warmup_days)
    prepared = {}
    for sym, df in panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= DEFAULT_IS_END)]
        if len(sub) >= min_symbol_bars:
            prepared[sym] = _add_atr(sub)

    all_trades = []
    for _, row in survivor_rows.iterrows():
        params = {}
        for p in param_names:
            grid_val0 = param_grid[p][0]
            params[p] = int(row[p]) if isinstance(grid_val0, int) else row[p]
        positions_by_symbol = generate_universe_positions(panel, params)
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
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "side", "entry_px", "exit_px", "net_r", "symbol"])


def run_t1_t2(candidate_id: str, panel: dict):
    spec = CANDIDATES[candidate_id]
    print(f"\n{'#'*80}\nCandidate {candidate_id} ({spec['name']}): T1 with diversification gate "
          f"(max_month_share={MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE})\n{'#'*80}")

    result = run_t1_universe_ranked(
        candidate_id=candidate_id, candidate_name=spec["name"],
        generate_universe_positions=spec["module"].generate_universe_positions,
        price_panel=panel, param_grids=spec["module"].PARAM_GRID, asset_class="futures",
        out_dir=ROOT / f"data" / f"research_candidate{candidate_id}_t1",
        update_tracker=False, max_month_share=MAX_MONTH_SHARE, max_asset_share=MAX_ASSET_SHARE,
    )

    grid = result.grid
    param_names = list(spec["module"].PARAM_GRID.keys())
    n_pre_div = int((grid["gate_avgr"] & grid["gate_stability"]).sum())
    print(f"Grid: {len(grid)} combos. Pass avg_r+stability (pre-diversification): {n_pre_div}. "
          f"Pass ALL gates (incl. diversification): {result.n_pass}.")
    print(f"Diversification gate excluded {n_pre_div - result.n_pass} combo(s).")

    if result.n_pass == 0:
        print(f"\nCandidate {candidate_id}: NO combos survive -- T1 VERDICT: FAIL")
        return dict(candidate_id=candidate_id, t1_survives=False, n_pass=0)

    passing = grid[grid["all_gates"]]
    cols = param_names + ["avg_r", "n", "dominant_month_share", "dominant_asset_share"]
    print("\nSurviving combos:")
    print(passing[cols].to_string(index=False))

    robustness = result.robustness
    survives = bool(robustness["survives"].any()) if len(robustness) else True
    print(f"\nRobustness (drop-top-2): {int(robustness['survives'].sum())}/{len(robustness)} survive")
    print(f"edge_clustering: {result.edge_clustering}")
    print(f"thin_sliver: {result.thin_sliver}")
    print(f"year_concentration (pooled diagnostic): {result.year_concentration}")
    print(f"T1 VERDICT: {'PASS' if survives else 'FAIL'}")

    if not survives:
        return dict(candidate_id=candidate_id, t1_survives=False, n_pass=result.n_pass)

    survivor_rows = robustness[robustness["survives"]]
    pooled = pool_trades_for_ranked_combos(spec["module"].generate_universe_positions, panel,
                                            spec["module"].PARAM_GRID, param_names, survivor_rows)

    print(f"\n-- T2: regime/cost-realism check on {len(survivor_rows)} gate-surviving combo(s) --")
    yearly = yearly_breakdown(pooled)
    print(yearly.to_string())
    regime = regime_dependence_flag(yearly)
    cost = cost_realism_check(pooled, base_floor=0.25)
    print(f"Regime dependence: flag={regime['flag']}  reason={regime.get('reason')}")
    print(f"Cost realism (base+0.05R floor): avg_r={cost['avg_r']:.4f}R vs stricter floor "
          f"{cost['stricter_floor']:.2f}R -> clears={cost['clears_stricter_floor']}  t_stat={cost['t_stat']:.2f}")
    t2_verdict = "T2_PASS" if (not regime["flag"] and cost["clears_stricter_floor"]) else "T2_CAUTION"
    print(f"T2 VERDICT: {t2_verdict}")

    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == candidate_id
    caution = bool(result.edge_clustering) or bool(result.thin_sliver) or \
        (bool(result.year_concentration) and not result.year_concentration.get("passes", True))
    t1_stage = "T1_PASS_HOLD_NOT_ADVANCED" if caution else "T2"
    stage = "T3" if t2_verdict == "T2_PASS" else "T2_CAUTION_HOLD"
    tracker.loc[mask, "t1_status"] = "PASS"
    tracker.loc[mask, "t2_status"] = "PASS" if t2_verdict == "T2_PASS" else "CAUTION"
    tracker.loc[mask, "current_stage"] = stage
    existing_notes = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing_notes} T1 (diversification-gated from the start, max_month_share="
            f"{MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE}): {n_pre_div} combos pass "
            f"avg_r+stability, {result.n_pass} pass all gates incl. diversification, "
            f"{int(robustness['survives'].sum())} survive drop-top-2. thin_sliver={result.thin_sliver}. "
            f"T2: yearly R% = {dict(zip(yearly.index.astype(str), yearly['pct_of_total_r'].round(2)))}. "
            f"regime_flag={regime['flag']} ({regime['reason']}). cost_realism avg_r={cost['avg_r']:.3f}R "
            f"vs {cost['stricter_floor']:.2f}R stricter floor -> clears={cost['clears_stricter_floor']}. "
            f"T2 verdict: {t2_verdict}.").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)

    return dict(candidate_id=candidate_id, t1_survives=True, n_pass=result.n_pass,
                t2_verdict=t2_verdict, pooled_n=len(pooled), pooled_avg_r=pooled["net_r"].mean())


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = load_panel()
    print(f"Loaded {len(panel)} symbols.")

    print("\nBuilding candidate 12's T0 proxy return series for the overlap-risk priority check...")
    c12_positions = c12.generate_universe_positions(panel, C12_T0_PARAMS)
    c12_proxy_ret = aggregate_portfolio_return(panel, c12_positions)

    results = {}
    for cid in ("18", "19", "20"):
        verdict0, _ = run_t0(cid, panel, c12_proxy_ret)
        if verdict0 == "REDUNDANT":
            print(f"Candidate {cid}: REDUNDANT at T0 -- skipping T1/T2 per pipeline spec.")
            results[cid] = dict(candidate_id=cid, t0_verdict=verdict0, t1_survives=None)
            continue
        r = run_t1_t2(cid, panel)
        r["t0_verdict"] = verdict0
        results[cid] = r

    print(f"\n{'='*100}\nPHASE 2 SUMMARY\n{'='*100}")
    for cid, r in results.items():
        t1_label = "PASS" if r.get("t1_survives") else ("SKIPPED" if r.get("t1_survives") is None else "FAIL")
        n_pass_suffix = f", n_pass={r['n_pass']}" if "n_pass" in r else ""
        print(f"\n{cid} ({CANDIDATES[cid]['name']}): T0={r.get('t0_verdict')}, T1={t1_label}{n_pass_suffix}")


if __name__ == "__main__":
    main()
