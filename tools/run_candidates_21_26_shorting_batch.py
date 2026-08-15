#!/usr/bin/env python3
"""
Shorting top-confirmation batch (candidates 21-26) -- T0 then T1/T2, cross-sectional
on the 290-symbol futures universe, with check_combo_diversification active FROM THE
START (max_month_share=0.25, max_asset_share=0.20 -- Phase 1's lesson: don't let a
concentrated combo pass T1 only to discover the fragility at T7).

Motivated by the empirical BTC test showing the real cost of a naive breakdown-short
concentrates at unconfirmed tops (routine pullbacks in an ongoing uptrend), not during
genuine trend reversals -- each candidate adds a distinct top-confirmation mechanism on
top of the same Donchian breakdown trigger candidate 7 used (and failed with, uncon-
firmed):
  21: funding-rate extreme (market-structure confirmation)
  22: multi-timeframe trend context (slow-MA confirmation)
  23: slower-only parameterization of the unconfirmed breakdown itself (no new filter)
  24: momentum divergence at highs
  25: volume exhaustion at highs (climax or declining)
  26: structural lower-high (swing-pivot confirmation)

All use the per-symbol generate_signal(df, params) interface (run_t1_universe), same
as candidates 02u/03u/11u/13u -- NOT candidate 12/18/19/20's ranking interface, since
these are independent per-symbol short setups, not a cross-sectional ranking construction.

T0 uses a single BTC-only proxy signal (matching candidates 11/13's convention for
per-symbol-interface candidates -- distinct from 12/18/19/20's aggregate-portfolio-
return proxy, which is only needed when no single-asset representation exists).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.t0_triage import run_triage, append_triage_log  # noqa: E402
from tools.t1_harness import run_t1_universe, BATCH_TRACKER  # noqa: E402
from tools.t2_regime_check import (  # noqa: E402
    trades_for_combo_universe, yearly_breakdown, regime_dependence_flag, cost_realism_check,
)
from signal_generators import candidate_21_funding_extreme_short as c21  # noqa: E402
from signal_generators import candidate_22_mtf_breakdown_short as c22  # noqa: E402
from signal_generators import candidate_23_slow_breakdown_short as c23  # noqa: E402
from signal_generators import candidate_24_momentum_divergence_short as c24  # noqa: E402
from signal_generators import candidate_25_volume_exhaustion_short as c25  # noqa: E402
from signal_generators import candidate_26_lower_high_structure_short as c26  # noqa: E402

UNIVERSE_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
FUNDING_DIR = ROOT / "data" / "futures_universe" / "funding_rates"
DEFAULT_IS_START = pd.Timestamp("2020-01-01")
DEFAULT_IS_END = pd.Timestamp("2024-12-31")
MAX_MONTH_SHARE = 0.25
MAX_ASSET_SHARE = 0.20

CANDIDATES = {
    "21": dict(name="Funding-rate-extreme confirmation short", module=c21, needs_funding=True,
               t0_params=dict(entry_n=55, exit_n=20, funding_threshold=0.0003)),
    "22": dict(name="Multi-timeframe breakdown confirmation short", module=c22, needs_funding=False,
               t0_params=dict(fast_n=20, slow_n=100, exit_n=20)),
    "23": dict(name="Slower-only Donchian breakdown short", module=c23, needs_funding=False,
               t0_params=dict(entry_n=200, exit_n=30)),
    "24": dict(name="Momentum/price divergence short (at highs)", module=c24, needs_funding=False,
               t0_params=dict(lookback=60, mom_n=14, trigger_n=20, exit_n=20)),
    "25": dict(name="Volume exhaustion short (at highs)", module=c25, needs_funding=False,
               t0_params=dict(lookback=60, pattern="declining", trigger_n=20, exit_n=20)),
    "26": dict(name="Structural lower-high confirmation short", module=c26, needs_funding=False,
               t0_params=dict(pivot_n=10, trigger_n=20, exit_n=20)),
}


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


def _merge_funding(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    f = FUNDING_DIR / f"{symbol}_funding.csv"
    if not f.exists():
        return None
    funding = pd.read_csv(f)
    if "funding_time" not in funding.columns or "funding_rate" not in funding.columns:
        return None
    funding["date"] = pd.to_datetime(funding["funding_time"], unit="ms").dt.floor("D")
    funding_daily = funding.groupby("date")["funding_rate"].mean()
    out = df.join(funding_daily.rename("funding_rate"), how="left")
    out["funding_rate"] = out["funding_rate"].ffill()
    out = out.dropna(subset=["funding_rate"])
    return out if len(out) else None


def load_panel_with_funding(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in panel.items():
        merged = _merge_funding(df, sym)
        if merged is not None:
            out[sym] = merged
    return out


def run_t0(candidate_id: str, df_btc: pd.DataFrame) -> str:
    spec = CANDIDATES[candidate_id]
    print(f"\n{'='*70}\nCandidate {candidate_id} ({spec['name']}): T0 triage\n{'='*70}")
    positions = spec["module"].generate_signal(df_btc, spec["t0_params"])
    ret = df_btc["close"].pct_change()
    cret = (positions.shift(1).fillna(0) * ret).dropna()

    row = run_triage(candidate_id, cret, notes=f"T0 proxy signal (single-asset BTC): {spec['t0_params']}. "
                      f"Priority check vs S7 (macross short) and S8 (funding-gated) given the "
                      f"shared short-side/funding-adjacent territory.")
    append_triage_log(row)

    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    if not (tracker["candidate_id"] == candidate_id).any():
        blank = {c: "" for c in tracker.columns}
        blank.update(candidate_id=candidate_id, name=spec["name"], batch=6,
                      side="short_only", current_stage="not_started")
        tracker = pd.concat([tracker, pd.DataFrame([blank])], ignore_index=True)
    mask = tracker["candidate_id"] == candidate_id
    tracker.loc[mask, "t0_status"] = row["verdict"]
    if row["verdict"] == "REDUNDANT":
        tracker.loc[mask, "current_stage"] = "T0_FAILED"
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)

    print(f"T0 verdict: {row['verdict']}")
    for sid in ("S1", "S6", "S7", "S8"):
        corr = row[f"corr_{sid}"]
        corr_str = f"{corr:.3f}" if pd.notna(corr) else "n/a"
        flag = "  <-- priority check" if sid in ("S7", "S8") else ""
        print(f"  vs {sid}: corr={corr_str}  confidence={row[f'corr_{sid}_confidence']}{flag}")
    print(f"  notes: {row['notes']}")
    return row["verdict"]


def run_t1_t2(candidate_id: str, panel: dict):
    spec = CANDIDATES[candidate_id]
    use_panel = load_panel_with_funding(panel) if spec["needs_funding"] else panel
    print(f"\n{'#'*80}\nCandidate {candidate_id} ({spec['name']}): T1 with diversification gate "
          f"(max_month_share={MAX_MONTH_SHARE}, max_asset_share={MAX_ASSET_SHARE})\n{'#'*80}")
    if spec["needs_funding"]:
        print(f"Funding-merged panel: {len(use_panel)}/{len(panel)} symbols have usable funding history")

    result = run_t1_universe(
        candidate_id=candidate_id, candidate_name=spec["name"],
        generate_signal=spec["module"].generate_signal,
        price_panel=use_panel, param_grids=spec["module"].PARAM_GRID, asset_class="futures",
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
        _update_tracker_t1_fail(candidate_id, n_pre_div, result)
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
        _update_tracker_t1_fail(candidate_id, n_pre_div, result)
        return dict(candidate_id=candidate_id, t1_survives=False, n_pass=result.n_pass)

    survivor_rows = robustness[robustness["survives"]]
    pooled_trades = []
    for _, row in survivor_rows.iterrows():
        params = {p: (int(row[p]) if isinstance(spec["module"].PARAM_GRID[p][0], int) else row[p])
                  for p in param_names}
        tdf = trades_for_combo_universe(spec["module"].generate_signal, params, use_panel,
                                         DEFAULT_IS_START, DEFAULT_IS_END)
        pooled_trades.append(tdf)
    pooled = pd.concat(pooled_trades, ignore_index=True)

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


def _update_tracker_t1_fail(candidate_id: str, n_pre_div: int, result) -> None:
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for c in tracker.columns:
        tracker[c] = tracker[c].astype(object)
    mask = tracker["candidate_id"] == candidate_id
    tracker.loc[mask, "t1_status"] = "FAIL"
    tracker.loc[mask, "current_stage"] = "T1_FAILED"
    existing_notes = tracker.loc[mask, "notes"].iloc[0] if mask.any() else ""
    note = (f"{existing_notes} T1 (diversification-gated from the start): {n_pre_div} combos pass "
            f"avg_r+stability, {result.n_pass} pass all gates incl. diversification -- "
            f"CANDIDATE FAILS T1.").strip()
    tracker.loc[mask, "notes"] = note
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    tracker.to_csv(BATCH_TRACKER, index=False)


def main():
    print("Loading 290-symbol futures universe panel...")
    panel = load_panel()
    print(f"Loaded {len(panel)} symbols.")
    df_btc = load_btc()

    print("\nBuilding BTC funding-merged series for candidate 21's T0 proxy...")
    df_btc_funding = _merge_funding(df_btc, "BTCUSDT")

    results = {}
    for cid in ("21", "22", "23", "24", "25", "26"):
        proxy_df = df_btc_funding if CANDIDATES[cid]["needs_funding"] else df_btc
        verdict0 = run_t0(cid, proxy_df)
        if verdict0 == "REDUNDANT":
            print(f"Candidate {cid}: REDUNDANT at T0 -- skipping T1/T2 per pipeline spec.")
            results[cid] = dict(candidate_id=cid, t0_verdict=verdict0, t1_survives=None)
            continue
        r = run_t1_t2(cid, panel)
        r["t0_verdict"] = verdict0
        results[cid] = r

    print(f"\n{'='*100}\nSHORTING BATCH SUMMARY\n{'='*100}")
    for cid, r in results.items():
        t1_label = "PASS" if r.get("t1_survives") else ("SKIPPED" if r.get("t1_survives") is None else "FAIL")
        n_pass_suffix = f", n_pass={r['n_pass']}" if "n_pass" in r else ""
        t2_suffix = f", T2={r['t2_verdict']}" if "t2_verdict" in r else ""
        print(f"\n{cid} ({CANDIDATES[cid]['name']}): T0={r.get('t0_verdict')}, T1={t1_label}{n_pass_suffix}{t2_suffix}")


if __name__ == "__main__":
    main()
