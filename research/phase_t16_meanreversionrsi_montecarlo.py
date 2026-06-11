#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T16 -- MeanReversionRSI Monte Carlo
UngerFink Pipeline / Andrea Unger Methodology

Block bootstrap Monte Carlo on the canonical Variant E trade sequence.
5000 runs, block sizes [1, 5, 10, 20, 50].

Input : data/research_meanreversionrsi_t3mr_1d/phase_t3mr_trades_E.csv
        (248 uncapped trades -- full statistical power)

Output: data/research_meanreversionrsi_t16_1d/
    phase_t16_montecarlo_summary.csv
    phase_t16_montecarlo_report.txt
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


ROOT     = Path(__file__).resolve().parents[1]
T3MR_DIR = ROOT / "data" / "research_meanreversionrsi_t3mr_1d"
OUT_DIR  = ROOT / "data" / "research_meanreversionrsi_t16_1d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MC_RUNS   = 5000
MC_BLOCKS = [1, 5, 10, 20, 50]
SEED      = 42

MR_GATES = {"pf_min": 1.0, "avg_r_min": 0.10}

np.random.seed(SEED)


def metrics(rs: np.ndarray) -> dict:
    if len(rs) == 0:
        return {"n":0,"avg_r":0.0,"pf":0.0,"total_r":0.0}
    w  = rs[rs > 0]; l = np.abs(rs[rs < 0])
    pf = w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum = np.cumsum(rs)
    dd  = float(np.max(np.maximum.accumulate(cum) - cum))
    return {"n":len(rs),"avg_r":float(np.mean(rs)),"pf":float(pf),
            "total_r":float(np.sum(rs)),"max_dd":dd}


def block_bootstrap(rs: np.ndarray, block_size: int, n_runs: int) -> dict:
    n = len(rs)
    totals, pfs, avgs, dds = [], [], [], []

    for _ in range(n_runs):
        n_blk   = max(1, (n // block_size) + 3)
        starts  = np.random.randint(0, n, size=n_blk)
        sample  = np.concatenate([rs[s:min(s+block_size, n)] for s in starts])[:n]
        if len(sample) == 0:
            continue
        m = metrics(sample)
        totals.append(m["total_r"])
        pfs.append(m["pf"])
        avgs.append(m["avg_r"])
        dds.append(m["max_dd"])

    totals = np.array(totals)
    pfs    = np.array(pfs)
    dds    = np.array(dds)
    avgs   = np.array(avgs)

    mc_pass = (np.percentile(totals, 5) > 0 and
               np.percentile(pfs, 5) >= MR_GATES["pf_min"])

    return {
        "block_size":     block_size,
        "n_runs":         len(totals),
        "p05_total_r":    float(np.percentile(totals, 5)),
        "p25_total_r":    float(np.percentile(totals, 25)),
        "p50_total_r":    float(np.percentile(totals, 50)),
        "p75_total_r":    float(np.percentile(totals, 75)),
        "p95_total_r":    float(np.percentile(totals, 95)),
        "prob_positive":  float(np.mean(totals > 0)),
        "pf_p05":         float(np.percentile(pfs, 5)),
        "pf_p50":         float(np.percentile(pfs, 50)),
        "avg_r_p05":      float(np.percentile(avgs, 5)),
        "avg_r_p50":      float(np.percentile(avgs, 50)),
        "max_dd_p95":     float(np.percentile(dds, 95)),
        "max_dd_p50":     float(np.percentile(dds, 50)),
        "mc_pass":        mc_pass,
    }


def main() -> None:
    p("=" * 70)
    p("  Phase T16 -- MeanReversionRSI Monte Carlo")
    p(f"  Runs: {MC_RUNS}  |  Block sizes: {MC_BLOCKS}")
    p("=" * 70)

    trades_path = T3MR_DIR / "phase_t3mr_trades_E.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found.")
        sys.exit(1)

    df = pd.read_csv(trades_path)
    df = df.sort_values("entry_time").reset_index(drop=True)
    rs = df["net_r"].values
    n  = len(rs)

    p(f"  Input trades : {n}")
    p(f"  Actual total_r : {rs.sum():.2f}R  avg_r={rs.mean():.4f}R")

    rows = []
    p()
    p(f"  {'BlkSz':>6}  {'p05':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  "
      f"{'Prob+':>7}  {'PF_p05':>7}  {'DD_p95':>7}  Gate")
    p("  " + "-" * 85)

    for bs in MC_BLOCKS:
        p(f"  Running block_size={bs} ({MC_RUNS} runs)...", end=" ")
        r = block_bootstrap(rs, bs, MC_RUNS)
        p(f"done")
        p(f"  {r['block_size']:>6}  {r['p05_total_r']:>+7.2f}R  {r['p25_total_r']:>+7.2f}R  "
          f"{r['p50_total_r']:>+7.2f}R  {r['p75_total_r']:>+7.2f}R  {r['p95_total_r']:>+7.2f}R  "
          f"{r['prob_positive']:>6.1%}  {r['pf_p05']:>7.2f}  {r['max_dd_p95']:>6.2f}R  "
          f"{'OK' if r['mc_pass'] else 'FAIL'}")
        rows.append(r)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "phase_t16_montecarlo_summary.csv", index=False)

    # Summary
    p()
    p("=" * 70)
    p("  T16 MONTE CARLO SUMMARY")
    p("=" * 70)
    all_pass     = all(r["mc_pass"] for r in rows)
    min_p05      = min(r["p05_total_r"] for r in rows)
    min_prob_pos = min(r["prob_positive"] for r in rows)
    min_pf_p05   = min(r["pf_p05"] for r in rows)
    max_dd_p95   = max(r["max_dd_p95"] for r in rows)

    p(f"  Worst-case p05 total_r  : {min_p05:+.2f}R  (block={MC_BLOCKS[rows.index(min(rows, key=lambda x: x['p05_total_r']))]})")
    p(f"  Min prob_positive       : {min_prob_pos:.1%}")
    p(f"  Min PF at p05           : {min_pf_p05:.2f}")
    p(f"  Max drawdown at p95     : {max_dd_p95:.2f}R")
    p(f"  All block sizes pass    : {'YES' if all_pass else 'NO'}")

    with open(OUT_DIR / "phase_t16_montecarlo_report.txt", "w", encoding="utf-8") as f:
        f.write("Phase T16 -- MeanReversionRSI Monte Carlo\n")
        f.write(f"Runs: {MC_RUNS}  Block sizes: {MC_BLOCKS}  Seed: {SEED}\n")
        f.write(f"Input: {n} trades (Variant E, uncapped)\n\n")
        f.write(df_out.to_string(index=False))
        f.write(f"\n\nWorst-case p05 total_r : {min_p05:+.2f}R\n")
        f.write(f"Min prob_positive      : {min_prob_pos:.1%}\n")
        f.write(f"Min PF at p05          : {min_pf_p05:.2f}\n")
        f.write(f"T16 GATE: {'PASS' if all_pass else 'FAIL'}\n")

    p(f"\n[OK] phase_t16_montecarlo_summary.csv")
    p(f"[OK] phase_t16_montecarlo_report.txt")
    p(f"\nT16 GATE: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
