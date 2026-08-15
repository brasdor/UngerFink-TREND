#!/usr/bin/env python3
"""
Three shorting workstreams:

  27: BTC-beta-conditional short -- BTC-confirmed-downtrend gate (modules/
      btc_regime_gate.py) + trailing-beta-to-BTC selection (highest-beta subset).
      Ranking interface (generate_universe_positions).
  28: Regime-gated breakdown short -- SAME BTC-confirmed-downtrend gate as 27
      (trend_n=100 fixed), applied to candidate 7's plain breakdown-short logic,
      NO beta selection. Comparison baseline to isolate whether the regime gate
      alone (without candidate 27's beta-ranking piece) fixes the bull-biased
      base-rate problem that broke candidate 7 and the whole 21-26 batch.
      Per-symbol interface (generate_signal), needs a `btc_downtrend` column
      merged into every symbol's panel data first.
  25 refined: tighter grid around the T1 near-miss (lookback=30, trigger_n=20,
      exit_n=10), finer steps, climax_volume_mult swept below the original fixed
      2.0. T1/T2 only -- T0 is unchanged from the original run (redundancy-vs-
      live-systems is invariant to which combos are in the grid, same reasoning
      already established for the 11u/13u diversification-gate rework).

All T1 runs use check_combo_diversification active from the start (max_month_share=
0.25, max_asset_share=0.20), same as every batch since Phase 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t0_triage import run_triage, append_triage_log  # noqa: E402
from tools.t1_harness import (  # noqa: E402
    run_t1_universe, run_t1_universe_ranked, BATCH_TRACKER, _add_atr, _positions_to_trades,
)
from tools.t2_regime_check import (  # noqa: E402
    trades_for_combo_universe, yearly_breakdown, regime_dependence_flag, cost_realism_check,
)
from modules.btc_regime_gate import btc_confirmed_downtrend  # noqa: E402
from signal_generators import candidate_27_beta_conditional_short as c27  # noqa: E402
from signal_generators import candidate_28_regime_gated_breakdown_short as c28  # noqa: E402
from signal_generators import candidate_25_volume_exhaustion_short as c25  # noqa: E402

UNIVERSE_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
DEFAULT_IS_START = pd.Timestamp("2020-01-01")
DEFAULT_IS_END = pd.Timestamp("2024-12-31")
MAX_MONTH_SHARE = 0.25
MAX_ASSET_SHARE = 0.20

C27_T0_PARAMS = dict(beta_lookback=60, quantile=0.2, rebalance_n=7, trend_n=100)
C28_T0_PARAMS = dict(entry_n=55, exit_n=20)


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


def load_btc() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_DIR / "BTCUSDT_1d.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def merge_btc_downtrend(panel: dict[str, pd.DataFrame], btc_downtrend: pd.Series) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in panel.items():
        merged = df.join(btc_downtrend.rename("btc_downtrend"), how="left")
        merged["btc_downtrend"] = merged["btc_downtrend"].ffill().fillna(False)
        out[sym] = merged
    return out


def aggregate_portfolio_return(panel: dict, positions_by_symbol: dict) -> pd.Series:
    per_symbol_ret = {}
    for sym, df in panel.items():
        pos = positions_by_symbol.get(sym)
        if pos is None:
            continue
        ret = df["close"].pct_change()
        per_symbol_ret[sym] = (pos.shift(1).fillna(0) * ret).reindex(df.index)
    return pd.DataFrame(per_symbol_ret).mean(axis=1, skipna=True).dropna()


def _update_tracker(candidate_id: str, name: str, **fields) -> None:
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    if not (tracker["candidate_id"] == candidate_id).any():
        blank = {c: "" for c in tracker.columns}
        blank.update(candidate_id=candidate_id, name=name, batch=7, side="short_only",
                      current_stage="not_started")
        tracker = pd.concat([tracker, pd.DataFrame([blank])], ignore_index=True)
    mask = tracker["candidate_id"] == candidate_id
    for k, v in fields.items():
        if k == "notes_append":
            existing = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
            tracker.loc[mask, "notes"] = f"{existing} {v}".strip()
        else:
            tracker.loc[mask, k] = v
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


# ---------------------------------------------------------------------------
# Candidate 27
# ---------------------------------------------------------------------------

def run_c27_t0(panel: dict) -> str:
    print(f"\n{'='*70}\nCandidate 27 (BTC-beta-conditional short): T0 triage\n{'='*70}")
    positions = c27.generate_universe_positions(panel, C27_T0_PARAMS)
    proxy_ret = aggregate_portfolio_return(panel, positions)
    row = run_triage("27", proxy_ret, notes=f"T0 proxy signal: {C27_T0_PARAMS} (aggregate cross-sectional "
                      f"portfolio return, BTC-beta-conditional short). Priority check vs S7/S8 (short-side/"
                      f"funding-adjacent).")
    append_triage_log(row)
    _update_tracker("27", "BTC-beta-conditional short", t0_status=row["verdict"],
                     current_stage="T0_FAILED" if row["verdict"] == "REDUNDANT" else None)
    print(f"T0 verdict: {row['verdict']}")
    for sid in ("S1", "S6", "S7", "S8"):
        corr = row[f"corr_{sid}"]
        corr_str = f"{corr:.3f}" if pd.notna(corr) else "n/a"
        print(f"  vs {sid}: corr={corr_str}  confidence={row[f'corr_{sid}_confidence']}")
    print(f"  notes: {row['notes']}")
    return row["verdict"]


def run_c27_t1_t2(panel: dict):
    print(f"\n{'#'*80}\nCandidate 27: T1 with diversification gate "
          f"(max_month_share={MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE})\n{'#'*80}")

    result = run_t1_universe_ranked(
        candidate_id="27", candidate_name="BTC-beta-conditional short",
        generate_universe_positions=c27.generate_universe_positions,
        price_panel=panel, param_grids=c27.PARAM_GRID, asset_class="futures",
        out_dir=ROOT / "data" / "research_candidate27_t1", update_tracker=False,
        max_month_share=MAX_MONTH_SHARE, max_asset_share=MAX_ASSET_SHARE,
    )
    return _finish_t1_t2_ranked("27", "BTC-beta-conditional short", result, c27.PARAM_GRID,
                                 c27.generate_universe_positions, panel)


# ---------------------------------------------------------------------------
# Candidate 28
# ---------------------------------------------------------------------------

def run_c28_t0(df_btc_with_downtrend: pd.DataFrame) -> str:
    print(f"\n{'='*70}\nCandidate 28 (Regime-gated breakdown short): T0 triage\n{'='*70}")
    positions = c28.generate_signal(df_btc_with_downtrend, C28_T0_PARAMS)
    ret = df_btc_with_downtrend["close"].pct_change()
    cret = (positions.shift(1).fillna(0) * ret).dropna()
    row = run_triage("28", cret, notes=f"T0 proxy signal (single-asset BTC): {C28_T0_PARAMS}, "
                      f"btc_downtrend gate trend_n={c28.TREND_N_FIXED}. Priority check vs S7/S8.")
    append_triage_log(row)
    _update_tracker("28", "Regime-gated breakdown short (comparison baseline for 27)",
                     t0_status=row["verdict"],
                     current_stage="T0_FAILED" if row["verdict"] == "REDUNDANT" else None)
    print(f"T0 verdict: {row['verdict']}")
    for sid in ("S1", "S6", "S7", "S8"):
        corr = row[f"corr_{sid}"]
        corr_str = f"{corr:.3f}" if pd.notna(corr) else "n/a"
        print(f"  vs {sid}: corr={corr_str}  confidence={row[f'corr_{sid}_confidence']}")
    print(f"  notes: {row['notes']}")
    return row["verdict"]


def run_c28_t1_t2(panel_with_downtrend: dict):
    print(f"\n{'#'*80}\nCandidate 28: T1 with diversification gate "
          f"(max_month_share={MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE})\n{'#'*80}")

    result = run_t1_universe(
        candidate_id="28", candidate_name="Regime-gated breakdown short",
        generate_signal=c28.generate_signal, price_panel=panel_with_downtrend,
        param_grids=c28.PARAM_GRID, asset_class="futures",
        out_dir=ROOT / "data" / "research_candidate28_t1", update_tracker=False,
        max_month_share=MAX_MONTH_SHARE, max_asset_share=MAX_ASSET_SHARE,
    )
    return _finish_t1_t2_standard("28", "Regime-gated breakdown short", result, c28.PARAM_GRID,
                                   c28.generate_signal, panel_with_downtrend)


# ---------------------------------------------------------------------------
# Candidate 25 refined
# ---------------------------------------------------------------------------

def run_c25_refined_t1_t2(panel: dict):
    print(f"\n{'#'*80}\nCandidate 25 REFINED: T1 with diversification gate "
          f"(max_month_share={MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE})\n{'#'*80}")
    print("T0: NOT re-run -- redundancy-vs-live-systems is invariant to which combos are in "
          "the grid (same reasoning as the 11u/13u diversification-gate rework). Original T0 "
          "verdict (candidate 25: PASS_LOW_CONFIDENCE) carries forward unchanged.")

    result = run_t1_universe(
        candidate_id="25_refined", candidate_name="Volume exhaustion short (refined grid)",
        generate_signal=c25.generate_signal, price_panel=panel,
        param_grids=c25.PARAM_GRID_REFINED, asset_class="futures",
        out_dir=ROOT / "data" / "research_candidate25_refined_t1", update_tracker=False,
        max_month_share=MAX_MONTH_SHARE, max_asset_share=MAX_ASSET_SHARE,
    )
    return _finish_t1_t2_standard("25_refined", "Volume exhaustion short (refined grid)", result,
                                   c25.PARAM_GRID_REFINED, c25.generate_signal, panel)


# ---------------------------------------------------------------------------
# Shared T1/T2 finish + tracker update (ranked interface vs per-symbol interface)
# ---------------------------------------------------------------------------

def _print_t1_common(candidate_id, result, param_grids):
    grid = result.grid
    param_names = list(param_grids.keys())
    n_pre_div = int((grid["gate_avgr"] & grid["gate_stability"]).sum())
    print(f"Grid: {len(grid)} combos. Pass avg_r+stability (pre-diversification): {n_pre_div}. "
          f"Pass ALL gates (incl. diversification): {result.n_pass}.")
    print(f"Diversification gate excluded {n_pre_div - result.n_pass} combo(s).")
    if result.n_pass:
        passing = grid[grid["all_gates"]]
        cols = param_names + ["avg_r", "n", "dominant_month_share", "dominant_asset_share"]
        print("\nSurviving combos:")
        print(passing[cols].to_string(index=False))
    return n_pre_div, param_names


def _finish_t2(candidate_id, name, pooled, result, n_pre_div, robustness):
    print(f"\n-- T2: regime/cost-realism check --")
    yearly = yearly_breakdown(pooled)
    print(yearly.to_string())
    regime = regime_dependence_flag(yearly)
    cost = cost_realism_check(pooled, base_floor=0.25)
    print(f"Regime dependence: flag={regime['flag']}  reason={regime.get('reason')}")
    print(f"Cost realism (base+0.05R floor): avg_r={cost['avg_r']:.4f}R vs stricter floor "
          f"{cost['stricter_floor']:.2f}R -> clears={cost['clears_stricter_floor']}  t_stat={cost['t_stat']:.2f}")
    t2_verdict = "T2_PASS" if (not regime["flag"] and cost["clears_stricter_floor"]) else "T2_CAUTION"
    print(f"T2 VERDICT: {t2_verdict}")

    caution = bool(result.edge_clustering) or bool(result.thin_sliver) or \
        (bool(result.year_concentration) and not result.year_concentration.get("passes", True))
    stage = "T3" if t2_verdict == "T2_PASS" else "T2_CAUTION_HOLD"
    _update_tracker(candidate_id, name, t1_status="PASS",
                     t2_status="PASS" if t2_verdict == "T2_PASS" else "CAUTION", current_stage=stage,
                     notes_append=(f"T1 (diversification-gated, max_month_share={MAX_MONTH_SHARE}, "
                                    f"max_asset_share={MAX_ASSET_SHARE}): {n_pre_div} pass avg_r+stability, "
                                    f"{result.n_pass} pass all gates, {int(robustness['survives'].sum())}/"
                                    f"{len(robustness)} survive drop-top-2. thin_sliver={result.thin_sliver}. "
                                    f"T2: regime_flag={regime['flag']} ({regime['reason']}), "
                                    f"cost_realism avg_r={cost['avg_r']:.3f}R vs {cost['stricter_floor']:.2f}R "
                                    f"-> clears={cost['clears_stricter_floor']}. T2 verdict: {t2_verdict}."))
    return dict(candidate_id=candidate_id, t1_survives=True, n_pass=result.n_pass,
                t2_verdict=t2_verdict, pooled_n=len(pooled), pooled_avg_r=pooled["net_r"].mean())


def _fail_t1(candidate_id, name, n_pre_div, result):
    print(f"\nCandidate {candidate_id}: NO combos survive -- T1 VERDICT: FAIL")
    _update_tracker(candidate_id, name, t1_status="FAIL", current_stage="T1_FAILED",
                     notes_append=(f"T1 (diversification-gated): {n_pre_div} pass avg_r+stability, "
                                    f"{result.n_pass} pass all gates -- CANDIDATE FAILS T1."))
    return dict(candidate_id=candidate_id, t1_survives=False, n_pass=result.n_pass)


def _finish_t1_t2_standard(candidate_id, name, result, param_grids, generate_signal, panel):
    n_pre_div, param_names = _print_t1_common(candidate_id, result, param_grids)
    if result.n_pass == 0:
        return _fail_t1(candidate_id, name, n_pre_div, result)

    robustness = result.robustness
    survives = bool(robustness["survives"].any()) if len(robustness) else True
    print(f"\nRobustness (drop-top-2): {int(robustness['survives'].sum())}/{len(robustness)} survive")
    print(f"edge_clustering: {result.edge_clustering}")
    print(f"thin_sliver: {result.thin_sliver}")
    print(f"year_concentration: {result.year_concentration}")
    print(f"T1 VERDICT: {'PASS' if survives else 'FAIL'}")
    if not survives:
        return _fail_t1(candidate_id, name, n_pre_div, result)

    survivor_rows = robustness[robustness["survives"]]
    pooled_trades = []
    for _, row in survivor_rows.iterrows():
        params = {p: (int(row[p]) if isinstance(param_grids[p][0], int) else row[p]) for p in param_names}
        tdf = trades_for_combo_universe(generate_signal, params, panel, DEFAULT_IS_START, DEFAULT_IS_END)
        pooled_trades.append(tdf)
    pooled = pd.concat(pooled_trades, ignore_index=True)
    return _finish_t2(candidate_id, name, pooled, result, n_pre_div, robustness)


def _finish_t1_t2_ranked(candidate_id, name, result, param_grids, generate_universe_positions, panel):
    n_pre_div, param_names = _print_t1_common(candidate_id, result, param_grids)
    if result.n_pass == 0:
        return _fail_t1(candidate_id, name, n_pre_div, result)

    robustness = result.robustness
    survives = bool(robustness["survives"].any()) if len(robustness) else True
    print(f"\nRobustness (drop-top-2): {int(robustness['survives'].sum())}/{len(robustness)} survive")
    print(f"edge_clustering: {result.edge_clustering}")
    print(f"thin_sliver: {result.thin_sliver}")
    print(f"year_concentration: {result.year_concentration}")
    print(f"T1 VERDICT: {'PASS' if survives else 'FAIL'}")
    if not survives:
        return _fail_t1(candidate_id, name, n_pre_div, result)

    survivor_rows = robustness[robustness["survives"]]
    slice_start = DEFAULT_IS_START - pd.Timedelta(days=260)
    prepared = {}
    for sym, df in panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= DEFAULT_IS_END)]
        if len(sub) >= 200:
            prepared[sym] = _add_atr(sub)

    pooled_trades = []
    for _, row in survivor_rows.iterrows():
        params = {p: (int(row[p]) if isinstance(param_grids[p][0], int) else row[p]) for p in param_names}
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
                pooled_trades.append(tdf)
    pooled = pd.concat(pooled_trades, ignore_index=True) if pooled_trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "side", "entry_px", "exit_px", "net_r", "symbol"])
    return _finish_t2(candidate_id, name, pooled, result, n_pre_div, robustness)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = load_panel()
    print(f"Loaded {len(panel)} symbols.")
    df_btc = load_btc()

    print(f"\nComputing shared BTC-confirmed-downtrend gate (trend_n={c28.TREND_N_FIXED}) "
          f"for candidate 28's T0/T1...")
    downtrend_series = btc_confirmed_downtrend(df_btc, c28.TREND_N_FIXED)
    print(f"BTC confirmed-downtrend: {int(downtrend_series.sum())}/{len(downtrend_series)} days "
          f"({downtrend_series.mean():.1%})")
    df_btc_with_downtrend = df_btc.join(downtrend_series.rename("btc_downtrend"))
    panel_with_downtrend = merge_btc_downtrend(panel, downtrend_series)

    results = {}

    # Candidate 27
    v27 = run_c27_t0(panel)
    if v27 == "REDUNDANT":
        print("Candidate 27: REDUNDANT at T0 -- skipping T1/T2.")
        results["27"] = dict(t0_verdict=v27, t1_survives=None)
    else:
        r = run_c27_t1_t2(panel)
        r["t0_verdict"] = v27
        results["27"] = r

    # Candidate 28
    v28 = run_c28_t0(df_btc_with_downtrend)
    if v28 == "REDUNDANT":
        print("Candidate 28: REDUNDANT at T0 -- skipping T1/T2.")
        results["28"] = dict(t0_verdict=v28, t1_survives=None)
    else:
        r = run_c28_t1_t2(panel_with_downtrend)
        r["t0_verdict"] = v28
        results["28"] = r

    # Candidate 25 refined
    r25 = run_c25_refined_t1_t2(panel)
    r25["t0_verdict"] = "PASS_LOW_CONFIDENCE (unchanged from original)"
    results["25_refined"] = r25

    print(f"\n{'='*100}\nWORKSTREAM SUMMARY\n{'='*100}")
    for cid, r in results.items():
        t1_label = "PASS" if r.get("t1_survives") else ("SKIPPED" if r.get("t1_survives") is None else "FAIL")
        n_pass_suffix = f", n_pass={r['n_pass']}" if "n_pass" in r else ""
        t2_suffix = f", T2={r['t2_verdict']}" if "t2_verdict" in r else ""
        print(f"\n{cid}: T0={r.get('t0_verdict')}, T1={t1_label}{n_pass_suffix}{t2_suffix}")

    if results.get("27", {}).get("t1_survives") is not None or results.get("28", {}).get("t1_survives") is not None:
        print(f"\n{'-'*100}\nCANDIDATE 27 (regime + beta) vs CANDIDATE 28 (regime only) COMPARISON\n{'-'*100}")
        r27, r28 = results.get("27", {}), results.get("28", {})
        print(f"  27 (beta selection): T1={'PASS' if r27.get('t1_survives') else 'FAIL'}, "
              f"avg_r={r27.get('pooled_avg_r', float('nan')):.4f}R" if r27.get("t1_survives") else
              f"  27 (beta selection): T1=FAIL")
        print(f"  28 (regime gate only): T1={'PASS' if r28.get('t1_survives') else 'FAIL'}, "
              f"avg_r={r28.get('pooled_avg_r', float('nan')):.4f}R" if r28.get("t1_survives") else
              f"  28 (regime gate only): T1=FAIL")


if __name__ == "__main__":
    main()
