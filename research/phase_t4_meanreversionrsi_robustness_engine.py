#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T4 -- MeanReversionRSI Robustness Engine
UngerFink Pipeline / Andrea Unger Methodology

Input  : phase_t3mr_trades_E.csv  (Variant E -- time exit 20 bars)
Output : data/research_meanreversionrsi_t4_1d/

Tests:
  1. Baseline summary + t-score
  2. Block bootstrap Monte Carlo (2000 runs, blocks [1,3,5,10,20])
  3. Cost stress (extra_cost per trade)
  4. Period splits (count-half, time-half, last-100)
  5. Remove best assets (top 1/3/5)
  6. Remove best months (top 1/2/3)
  7. Asset concentration (top-1/3/5 % of total R)
  8. Year-by-year (2021-2026)

Frozen config: rsi14/os25/time_exit=20/atr3/no_filter/1D
Benchmark: avg_r=+0.309R  PF=2.81  total_r=+76.5R
MR gates:  win_rate 50-70%  avg_r>0.10R  PF>1.0
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"


def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


# =============================================================================
# CONFIG
# =============================================================================

ROOT         = Path(__file__).resolve().parents[1]
T3MR_DIR     = ROOT / "data" / "research_meanreversionrsi_t3mr_1d"
OUT_DIR      = ROOT / "data" / "research_meanreversionrsi_t4_1d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK    = {"avg_r": 0.309, "pf": 2.81, "total_r": 76.5}
MR_GATES     = {"win_rate_min": 0.50, "win_rate_max": 0.70,
                "avg_r_min": 0.10, "pf_min": 1.0}

MC_RUNS      = 2000
MC_BLOCKS    = [1, 3, 5, 10, 20]
COST_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

np.random.seed(42)


# =============================================================================
# METRICS
# =============================================================================

def metrics(rs: np.ndarray) -> dict:
    if len(rs) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0,
                "max_dd_r": 0.0, "total_r": 0.0, "std_r": 0.0, "t_score": 0.0}
    wins   = rs[rs > 0]
    losses = np.abs(rs[rs < 0])
    pf     = wins.sum() / losses.sum() if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    cum    = np.cumsum(rs)
    dd     = float(np.max(np.maximum.accumulate(cum) - cum))
    std    = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
    t      = float(np.mean(rs) / (std / np.sqrt(len(rs)))) if std > 0 and len(rs) > 1 else 0.0
    return {
        "n":        len(rs),
        "win_rate": float(len(wins) / len(rs)),
        "avg_r":    float(np.mean(rs)),
        "pf":       float(pf),
        "max_dd_r": dd,
        "total_r":  float(np.sum(rs)),
        "std_r":    std,
        "t_score":  t,
    }


def gate_str(m: dict) -> str:
    wr  = MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"]
    ar  = m["avg_r"] >= MR_GATES["avg_r_min"]
    pf_ = m["pf"]   >= MR_GATES["pf_min"]
    return "PASS" if (wr and ar and pf_) else "FAIL"


def flag(cond: bool, ok_str: str = "OK", fail_str: str = "FLAG") -> str:
    return ok_str if cond else fail_str


# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================

def block_bootstrap(rs: np.ndarray, block_size: int, n_runs: int) -> dict:
    n      = len(rs)
    n_blk  = max(1, n // block_size)
    totals, pfs, avgs = [], [], []

    for _ in range(n_runs):
        # Draw enough block start indices, concatenate, trim to original length
        starts  = np.random.randint(0, n, size=n_blk + 5)
        sample  = np.concatenate([
            rs[s: min(s + block_size, n)] for s in starts
        ])[:n]
        if len(sample) == 0:
            continue
        m = metrics(sample)
        totals.append(m["total_r"])
        pfs.append(m["pf"])
        avgs.append(m["avg_r"])

    totals = np.array(totals)
    pfs    = np.array(pfs)
    return {
        "block_size":    block_size,
        "p05_total_r":   float(np.percentile(totals, 5)),
        "p50_total_r":   float(np.percentile(totals, 50)),
        "p95_total_r":   float(np.percentile(totals, 95)),
        "prob_positive": float(np.mean(totals > 0)),
        "pf_p05":        float(np.percentile(pfs, 5)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Phase T4 -- MeanReversionRSI Robustness Engine")
    p("  Config : rsi14/os25/time_exit=20/atr3/no_filter/1D")
    p(f"  Benchmark: avg_r={BENCHMARK['avg_r']}R  PF={BENCHMARK['pf']}  "
      f"total_r={BENCHMARK['total_r']}R")
    p("=" * 70)

    # Load trades
    trades_path = T3MR_DIR / "phase_t3mr_trades_E.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found. Run T3MR first.")
        sys.exit(1)

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)

    if "year" not in df.columns:
        df["year"] = df["entry_time"].dt.year
    if "symbol" not in df.columns:
        p("  ERROR: no 'symbol' column in trades file.")
        sys.exit(1)

    rs_all = df["net_r"].values
    p(f"  Trades loaded: {len(df)}")

    report_lines: list[str] = []
    def rp(line: str = ""):
        p(line)
        report_lines.append(line)

    # =========================================================================
    # 1. BASELINE
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  1. BASELINE SUMMARY")
    rp("=" * 70)
    bm = metrics(rs_all)
    rp(f"  Trades    : {bm['n']}")
    rp(f"  Win rate  : {bm['win_rate']*100:.1f}%   (gate 50-70%: {flag(MR_GATES['win_rate_min']<=bm['win_rate']<=MR_GATES['win_rate_max'])})")
    rp(f"  Avg R     : {bm['avg_r']:+.4f}R  (gate >0.10R: {flag(bm['avg_r']>=MR_GATES['avg_r_min'])})")
    rp(f"  Total R   : {bm['total_r']:+.2f}R")
    rp(f"  Prof Fac  : {bm['pf']:.2f}   (gate >1.0: {flag(bm['pf']>=MR_GATES['pf_min'])})")
    rp(f"  Max DD R  : {bm['max_dd_r']:.2f}R")
    rp(f"  Std R     : {bm['std_r']:.4f}R")
    rp(f"  t-score   : {bm['t_score']:.2f}   ({flag(bm['t_score']>=2.0, 'significant (>=2.0)', 'weak (<2.0)')})")
    rp(f"  Gate      : {gate_str(bm)}")

    # =========================================================================
    # 2. YEAR-BY-YEAR
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  2. YEAR-BY-YEAR (2021-2026)")
    rp("=" * 70)
    rp(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotalR':>8}  {'MaxDD':>7}  {'PF':>5}  Note")
    year_rows = []
    for yr in sorted(df["year"].unique()):
        ydf  = df[df["year"] == yr]
        ym   = metrics(ydf["net_r"].values)
        note = ""
        if yr == 2022: note = "<<< BEAR MARKET"
        if yr == 2026: note = "(partial year)"
        neg  = ym["avg_r"] < 0
        bear_fail = (yr == 2022 and ym["total_r"] < 0)
        flags = []
        if neg:       flags.append("NEG AVG_R")
        if bear_fail: flags.append("FAIL-BEAR")
        if yr == 2026 and ym["total_r"] < -5: flags.append("DEEP-NEG-2026")
        flag_str = "  <<< " + ",".join(flags) if flags else ""
        rp(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
           f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
           f"{ym['max_dd_r']:>6.2f}R  {ym['pf']:>5.2f}  {note}{flag_str}")
        year_rows.append({"year": yr, **ym, "note": note})
    pd.DataFrame(year_rows).to_csv(OUT_DIR / "phase_t4_yearly.csv", index=False)
    y2022 = next((r for r in year_rows if r["year"] == 2022), None)
    y2026 = next((r for r in year_rows if r["year"] == 2026), None)

    # =========================================================================
    # 3. PERIOD SPLITS
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  3. PERIOD SPLITS")
    rp("=" * 70)

    # By trade count
    mid_n  = len(df) // 2
    h1_n   = df.iloc[:mid_n]
    h2_n   = df.iloc[mid_n:]
    m1n    = metrics(h1_n["net_r"].values)
    m2n    = metrics(h2_n["net_r"].values)
    rp(f"  Count-half 1 ({h1_n['entry_time'].iloc[0].date()} - {h1_n['entry_time'].iloc[-1].date()}):")
    rp(f"    n={m1n['n']}  avg_r={m1n['avg_r']:+.4f}R  PF={m1n['pf']:.2f}  total={m1n['total_r']:+.2f}R  {gate_str(m1n)}")
    rp(f"  Count-half 2 ({h2_n['entry_time'].iloc[0].date()} - {h2_n['entry_time'].iloc[-1].date()}):")
    rp(f"    n={m2n['n']}  avg_r={m2n['avg_r']:+.4f}R  PF={m2n['pf']:.2f}  total={m2n['total_r']:+.2f}R  "
       f"{gate_str(m2n)}  {flag(m2n['avg_r']>=0, '', '<<< NEG SECOND HALF')}")

    # By time
    mid_t  = df["entry_time"].quantile(0.5)
    h1_t   = df[df["entry_time"] <= mid_t]
    h2_t   = df[df["entry_time"] >  mid_t]
    mt1    = metrics(h1_t["net_r"].values)
    mt2    = metrics(h2_t["net_r"].values)
    rp(f"  Time-half 1 ({h1_t['entry_time'].iloc[0].date()} - {h1_t['entry_time'].iloc[-1].date()}):")
    rp(f"    n={mt1['n']}  avg_r={mt1['avg_r']:+.4f}R  PF={mt1['pf']:.2f}  total={mt1['total_r']:+.2f}R  {gate_str(mt1)}")
    rp(f"  Time-half 2 ({h2_t['entry_time'].iloc[0].date()} - {h2_t['entry_time'].iloc[-1].date()}):")
    rp(f"    n={mt2['n']}  avg_r={mt2['avg_r']:+.4f}R  PF={mt2['pf']:.2f}  total={mt2['total_r']:+.2f}R  "
       f"{gate_str(mt2)}  {flag(mt2['avg_r']>=0, '', '<<< NEG SECOND HALF')}")

    # Last 100
    last100  = df.iloc[-100:]
    ml100    = metrics(last100["net_r"].values)
    rp(f"  Last 100 trades ({last100['entry_time'].iloc[0].date()} - {last100['entry_time'].iloc[-1].date()}):")
    rp(f"    n={ml100['n']}  avg_r={ml100['avg_r']:+.4f}R  PF={ml100['pf']:.2f}  total={ml100['total_r']:+.2f}R  "
       f"{gate_str(ml100)}  {flag(ml100['avg_r']>=0, '', '<<< NEG RECENT')}")

    split_rows = [
        {"split": "count_h1", **m1n},
        {"split": "count_h2", **m2n},
        {"split": "time_h1",  **mt1},
        {"split": "time_h2",  **mt2},
        {"split": "last_100", **ml100},
    ]
    pd.DataFrame(split_rows).to_csv(OUT_DIR / "phase_t4_splits.csv", index=False)

    # =========================================================================
    # 4. ASSET CONCENTRATION
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  4. ASSET CONCENTRATION (§4.7)")
    rp("=" * 70)
    asset_totals = (df.groupby("symbol")["net_r"]
                    .sum().sort_values(ascending=False))
    total_r_pos = df[df["net_r"] > 0]["net_r"].sum()
    total_r_all = bm["total_r"]

    conc_rows = []
    for k in [1, 3, 5, 10]:
        topk     = asset_totals.head(k)
        share    = topk.sum() / total_r_all * 100 if total_r_all > 0 else 0.0
        assets   = ", ".join(topk.index.tolist()[:3]) + ("..." if k > 3 else "")
        flag_c   = "FLAG" if share > 50 else "OK"
        rp(f"  Top-{k:2d}: {share:5.1f}% of total R  ({assets})  {flag_c}")
        conc_rows.append({"top_k": k, "share_pct": round(share, 2), "flag": flag_c})
    pd.DataFrame(conc_rows).to_csv(OUT_DIR / "phase_t4_concentration.csv", index=False)

    # Full asset breakdown
    asset_df = []
    for sym, grp in df.groupby("symbol"):
        m = metrics(grp["net_r"].values)
        asset_df.append({"symbol": sym, **m})
    asset_df = pd.DataFrame(asset_df).sort_values("total_r", ascending=False).reset_index(drop=True)
    asset_df.to_csv(OUT_DIR / "phase_t4_asset_breakdown.csv", index=False)

    # =========================================================================
    # 5. REMOVE BEST ASSETS
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  5. REMOVE BEST ASSETS")
    rp("=" * 70)
    remove_rows = []
    for k in [1, 3, 5]:
        top_syms  = asset_totals.head(k).index.tolist()
        sub       = df[~df["symbol"].isin(top_syms)]
        m_sub     = metrics(sub["net_r"].values)
        ok_flag   = flag(m_sub["avg_r"] >= MR_GATES["avg_r_min"] and m_sub["pf"] >= MR_GATES["pf_min"])
        rp(f"  Remove top-{k} ({', '.join(top_syms[:3])}):")
        rp(f"    n={m_sub['n']}  avg_r={m_sub['avg_r']:+.4f}R  PF={m_sub['pf']:.2f}  "
           f"total={m_sub['total_r']:+.2f}R  {ok_flag}")
        remove_rows.append({"removed_k": k, "removed": str(top_syms), **m_sub, "flag": ok_flag})
    pd.DataFrame(remove_rows).to_csv(OUT_DIR / "phase_t4_remove_best_assets.csv", index=False)

    # =========================================================================
    # 6. REMOVE BEST MONTHS
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  6. REMOVE BEST MONTHS")
    rp("=" * 70)
    df["ym"] = df["entry_time"].dt.to_period("M")
    month_totals = (df.groupby("ym")["net_r"]
                    .sum().sort_values(ascending=False))
    remove_month_rows = []
    for k in [1, 2, 3]:
        top_months = month_totals.head(k).index.tolist()
        sub        = df[~df["ym"].isin(top_months)]
        m_sub      = metrics(sub["net_r"].values)
        ok_flag    = flag(m_sub["avg_r"] >= MR_GATES["avg_r_min"] and m_sub["pf"] >= MR_GATES["pf_min"])
        months_str = ", ".join(str(m) for m in top_months[:3])
        rp(f"  Remove top-{k} months ({months_str}):")
        rp(f"    n={m_sub['n']}  avg_r={m_sub['avg_r']:+.4f}R  PF={m_sub['pf']:.2f}  "
           f"total={m_sub['total_r']:+.2f}R  {ok_flag}")
        remove_month_rows.append({"removed_k": k, "months": months_str, **m_sub, "flag": ok_flag})
    pd.DataFrame(remove_month_rows).to_csv(OUT_DIR / "phase_t4_remove_best_months.csv", index=False)

    # =========================================================================
    # 7. COST STRESS
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  7. COST STRESS TEST")
    rp("=" * 70)
    rp(f"  {'ExtraCost':>10}  {'AvgR':>8}  {'PF':>6}  {'TotalR':>8}  {'Gate'}")
    cost_rows = []
    for ec in COST_LEVELS:
        rs_adj = rs_all - ec
        mc     = metrics(rs_adj)
        g      = gate_str(mc)
        rp(f"  {ec:>+10.2f}R  {mc['avg_r']:>+7.4f}R  {mc['pf']:>6.2f}  "
           f"{mc['total_r']:>+7.2f}R  {g}")
        cost_rows.append({"extra_cost_r": ec, **mc, "gate": g})
    pd.DataFrame(cost_rows).to_csv(OUT_DIR / "phase_t4_cost_stress.csv", index=False)

    # =========================================================================
    # 8. BLOCK BOOTSTRAP MONTE CARLO
    # =========================================================================
    rp()
    rp("=" * 70)
    rp(f"  8. BLOCK BOOTSTRAP MONTE CARLO ({MC_RUNS} runs)")
    rp("=" * 70)
    rp(f"  {'BlkSz':>6}  {'p05_TotR':>10}  {'p50_TotR':>10}  {'p95_TotR':>10}  "
       f"{'Prob+':>7}  {'PF_p05':>7}  MC_pass")
    mc_rows = []
    for bs in MC_BLOCKS:
        r = block_bootstrap(rs_all, bs, MC_RUNS)
        mc_pass = flag(r["p05_total_r"] > 0 and r["pf_p05"] >= MR_GATES["pf_min"])
        rp(f"  {bs:>6}  {r['p05_total_r']:>+9.2f}R  {r['p50_total_r']:>+9.2f}R  "
           f"{r['p95_total_r']:>+9.2f}R  {r['prob_positive']:>6.1%}  "
           f"{r['pf_p05']:>7.2f}  {mc_pass}")
        mc_rows.append({**r, "mc_pass": mc_pass})
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUT_DIR / "phase_t4_montecarlo.csv", index=False)

    # =========================================================================
    # 9. CRITICAL FLAGS SUMMARY
    # =========================================================================
    rp()
    rp("=" * 70)
    rp("  9. CRITICAL FLAGS SUMMARY")
    rp("=" * 70)

    mc_p05_ok    = all(r["p05_total_r"] > 0 for r in mc_rows)
    rm_top1_ok   = pd.read_csv(OUT_DIR / "phase_t4_remove_best_assets.csv")
    rm_top1_pass = rm_top1_ok[rm_top1_ok["removed_k"] == 1].iloc[0]["avg_r"] >= MR_GATES["avg_r_min"]

    rm_mon1      = pd.read_csv(OUT_DIR / "phase_t4_remove_best_months.csv")
    rm_mon1_pass = rm_mon1[rm_mon1["removed_k"] == 1].iloc[0]["avg_r"] >= MR_GATES["avg_r_min"]

    h2_avg_ok    = m2n["avg_r"] >= 0
    bear_ok      = y2022 is not None and y2022["total_r"] > 0
    y2026_ok     = y2026 is None or y2026["total_r"] > -10.0

    def checkline(label, ok, detail=""):
        sym = "PASS" if ok else "FAIL"
        rp(f"  [{sym}] {label:45s} {detail}")

    checkline("MC p05 total R > 0 (all block sizes)",      mc_p05_ok,
              f"p05={mc_rows[0]['p05_total_r']:+.2f}R (bs=1)")
    checkline("Remove top-1 asset: avg_r remains > 0.10R", rm_top1_pass,
              f"avg_r={rm_top1_ok[rm_top1_ok['removed_k']==1].iloc[0]['avg_r']:+.4f}R")
    checkline("Remove top-1 month: avg_r remains > 0.10R", rm_mon1_pass,
              f"avg_r={rm_mon1[rm_mon1['removed_k']==1].iloc[0]['avg_r']:+.4f}R")
    checkline("Second half (count) avg_r >= 0",             h2_avg_ok,
              f"avg_r={m2n['avg_r']:+.4f}R")
    checkline("2022 bear year total R > 0",                 bear_ok,
              f"total_r={y2022['total_r']:+.2f}R" if y2022 else "N/A")
    checkline("2026 partial year not deeply negative",      y2026_ok,
              f"total_r={y2026['total_r']:+.2f}R" if y2026 else "N/A")

    all_critical = all([mc_p05_ok, rm_top1_pass, rm_mon1_pass,
                        h2_avg_ok, bear_ok, y2026_ok])
    rp()
    rp(f"  Critical checks overall : {'ALL PASS' if all_critical else 'SOME FAIL -- review required'}")

    # =========================================================================
    # WRITE MASTER REPORT
    # =========================================================================
    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "phase_t4_master_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Phase T4 -- MeanReversionRSI Robustness Engine\n")
        f.write("Config: rsi14/os25/time_exit=20/atr3/no_filter/1D\n")
        f.write(f"Benchmark: avg_r={BENCHMARK['avg_r']}R  PF={BENCHMARK['pf']}  "
                f"total_r={BENCHMARK['total_r']}R\n")
        f.write("=" * 70 + "\n\n")
        f.write(report_text)

    p()
    p("[OK] phase_t4_master_report.txt")
    p("[OK] phase_t4_yearly.csv")
    p("[OK] phase_t4_splits.csv")
    p("[OK] phase_t4_concentration.csv")
    p("[OK] phase_t4_asset_breakdown.csv")
    p("[OK] phase_t4_remove_best_assets.csv")
    p("[OK] phase_t4_remove_best_months.csv")
    p("[OK] phase_t4_cost_stress.csv")
    p("[OK] phase_t4_montecarlo.csv")
    p()
    p(f"T4 GATE: {'PASS' if all_critical else 'FAIL -- review required'}")
    sys.exit(0 if all_critical else 1)


if __name__ == "__main__":
    main()
