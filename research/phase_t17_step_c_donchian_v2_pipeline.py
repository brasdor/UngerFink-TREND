#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Step C -- DonchianLong_UniverseV2  Full Pipeline (T4 -> T8)

Input:  Step B trades (step_b_all_trades.csv) filtered to 24-symbol
        tightened universe (filtered_symbols_v2_included_only.csv).

Frozen Donchian config (unchanged):
  Donchian N=20 / ema200_price / ATR(14)x2.0 stop /
  Chandelier ACT+4R trail 3xATR / LONG only / 1D

Baseline (original 64-sym frozen pipeline):
  avg_r=+0.179R  PF=1.27  CAGR=+19%  DD=-1.78%  (max3 portfolio)

Output: data/research_donchianlong_universev2/
"""

from __future__ import annotations

import math, sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT     = Path(__file__).resolve().parents[1]
IN_TRADES  = ROOT / "data" / "universe" / "step_b_all_trades.csv"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUT_DIR    = ROOT / "data" / "research_donchianlong_universev2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_COL    = "net_r"
ENTRY_T  = "entry_time"
EXIT_T   = "exit_time"

# T4
MC_RUNS        = 2000
MC_BLOCKS      = [1, 3, 5, 10, 20]
MC_SEED        = 42
EXTRA_COSTS    = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# T5 portfolio variants
PORT_VARIANTS = [
    {"name": "max3",  "max_open": 3},
    {"name": "max5",  "max_open": 5},
    {"name": "max8",  "max_open": 8},
    {"name": "uncap", "max_open": 999},
]

# T6 capital
START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35

# Baseline
BASELINE = dict(avg_r=0.179, pf=1.27, cagr_pct=19.0, dd_pct=-1.78,
                label="Original 64-sym frozen pipeline (max3)")

CONC_FLAG = 0.50   # flag top-month > 50% of R


# =============================================================================
# HELPERS
# =============================================================================

def load_trades() -> pd.DataFrame:
    raw  = pd.read_csv(IN_TRADES)
    syms = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    df   = raw[raw["symbol"].isin(syms)].copy()
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[EXIT_T, ENTRY_T, R_COL])
    df = df.sort_values(EXIT_T).reset_index(drop=True)
    df["month"] = df[EXIT_T].dt.to_period("M").astype(str)
    df["year"]  = df[EXIT_T].dt.year
    return df


def max_dd(r: np.ndarray) -> float:
    if r.size == 0: return 0.0
    eq = np.cumsum(r); pk = np.maximum.accumulate(eq)
    return float((eq - pk).min())


def pf(r: np.ndarray) -> float:
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)


def summarize(r: np.ndarray) -> dict:
    if r.size == 0:
        return dict(trades=0, total_r=0.0, avg_r=0.0, win_rate=0.0,
                    profit_factor=0.0, max_dd_r=0.0, std_r=0.0, t_score=0.0)
    avg = float(r.mean()); std = float(r.std(ddof=1)) if r.size > 1 else 0.0
    t   = avg / (std / math.sqrt(r.size)) if std > 0 else 0.0
    return dict(trades=int(r.size), total_r=float(r.sum()), avg_r=avg,
                win_rate=float((r > 0).mean()), profit_factor=pf(r),
                max_dd_r=max_dd(r), std_r=std, t_score=float(t))


def block_bootstrap(vals: np.ndarray, bs: int, rng) -> np.ndarray:
    n = len(vals); out = []
    while len(out) < n:
        s = int(rng.integers(0, n)); e = min(s + bs, n)
        blk = vals[s:e]
        if len(blk) < bs:
            blk = np.concatenate([blk, vals[:bs - len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n], dtype=float)


# =============================================================================
# T4 -- ROBUSTNESS
# =============================================================================

def t4_baseline(df: pd.DataFrame) -> dict:
    r = df[R_COL].to_numpy(dtype=float)
    s = summarize(r)
    s.update(assets=df["symbol"].nunique(), months=df["month"].nunique(),
             start=str(df[EXIT_T].min().date()), end=str(df[EXIT_T].max().date()))
    return s


def t4_montecarlo(df: pd.DataFrame) -> pd.DataFrame:
    vals = df[R_COL].to_numpy(dtype=float)
    rng  = np.random.default_rng(MC_SEED)
    rows = []
    for bs in MC_BLOCKS:
        tots, dds, pfs = [], [], []
        for _ in range(MC_RUNS):
            s = block_bootstrap(vals, bs, rng)
            tots.append(s.sum()); dds.append(max_dd(s)); pfs.append(pf(s))
        tots = np.array(tots); dds = np.array(dds); pfs = np.array(pfs)
        rows.append(dict(block_size=bs, mc_runs=MC_RUNS,
            total_r_p05=float(np.percentile(tots, 5)),
            total_r_p50=float(np.percentile(tots, 50)),
            total_r_p95=float(np.percentile(tots, 95)),
            dd_p95=float(np.percentile(dds, 95)),
            pf_p05=float(np.percentile(pfs, 5)),
            pf_p50=float(np.percentile(pfs, 50)),
            prob_positive=float((tots > 0).mean())))
    return pd.DataFrame(rows)


def t4_cost_stress(df: pd.DataFrame) -> pd.DataFrame:
    vals = df[R_COL].to_numpy(dtype=float)
    rows = []
    for ec in EXTRA_COSTS:
        s = summarize(vals - ec); s["extra_cost"] = ec; rows.append(s)
    return pd.DataFrame(rows)


def t4_period_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(EXIT_T).reset_index(drop=True)
    mid = len(df) // 2; med_t = df[EXIT_T].median()
    slices = {
        "first_half_by_trade":  df.iloc[:mid],
        "second_half_by_trade": df.iloc[mid:],
        "last_100_trades":      df.tail(100),
        "first_half_by_time":   df[df[EXIT_T] <= med_t],
        "second_half_by_time":  df[df[EXIT_T] > med_t],
    }
    rows = []
    for name, sub in slices.items():
        s = summarize(sub[R_COL].to_numpy(dtype=float)); s["split"] = name; rows.append(s)
    return pd.DataFrame(rows)


def t4_remove_best_assets(df: pd.DataFrame) -> pd.DataFrame:
    asset_r = df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    rows = []
    for n in [0, 1, 3, 5]:
        removed = asset_r.head(n).index.tolist()
        sub = df[~df["symbol"].isin(removed)]
        s   = summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n, removed=",".join(removed),
                 assets_remaining=sub["symbol"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)


def t4_remove_best_months(df: pd.DataFrame) -> pd.DataFrame:
    month_r = df.groupby("month")[R_COL].sum().sort_values(ascending=False)
    rows = []
    for n in [0, 1, 2, 3]:
        removed = month_r.head(n).index.tolist()
        sub = df[~df["month"].isin(removed)]
        s   = summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n, removed=",".join(removed),
                 months_remaining=sub["month"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)


def t4_concentration(df: pd.DataFrame) -> dict:
    asset_r = df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    total   = float(asset_r.sum())
    pos_sym = int((asset_r > 0).sum())
    return dict(
        top1_symbol=str(asset_r.index[0]),
        top1_r=float(asset_r.iloc[0]),
        top1_pct=float(asset_r.iloc[0] / total) if total else 0.0,
        top3_r=float(asset_r.head(3).sum()),
        top3_pct=float(asset_r.head(3).sum() / total) if total else 0.0,
        top5_r=float(asset_r.head(5).sum()),
        top5_pct=float(asset_r.head(5).sum() / total) if total else 0.0,
        total_r=total,
        positive_assets=pos_sym,
        negative_assets=int((asset_r <= 0).sum()),
        total_assets=len(asset_r),
    )


# =============================================================================
# T5 -- PORTFOLIO REPLAY
# =============================================================================

def t5_replay(df: pd.DataFrame, max_open: int) -> Tuple[pd.DataFrame, dict]:
    df = df.sort_values(ENTRY_T).reset_index(drop=True)
    open_pos: List[dict] = []
    closed:   List[dict] = []

    def flush(now):
        nonlocal open_pos
        still = []
        for p in open_pos:
            if p["exit_time"] <= now:
                closed.append(p)
            else:
                still.append(p)
        open_pos = still

    for _, row in df.iterrows():
        flush(row[ENTRY_T])
        if any(p["symbol"] == row["symbol"] for p in open_pos):
            continue
        if len(open_pos) >= max_open:
            continue
        open_pos.append({
            "symbol":    row["symbol"],
            "entry_time": row[ENTRY_T],
            "exit_time":  row[EXIT_T],
            R_COL:        row[R_COL],
        })

    # flush remaining
    if open_pos:
        last = max(p["exit_time"] for p in open_pos)
        flush(last + pd.Timedelta(seconds=1))

    acc = pd.DataFrame(closed)
    if acc.empty:
        stats = dict(accepted=0, total_r=0.0, avg_r=0.0,
                     profit_factor=0.0, max_dd_r=0.0, win_rate=0.0)
    else:
        acc = acc.sort_values("exit_time").reset_index(drop=True)
        r   = acc[R_COL].to_numpy(dtype=float)
        stats = dict(accepted=int(len(r)), total_r=float(r.sum()),
                     avg_r=float(r.mean()), profit_factor=pf(r),
                     max_dd_r=max_dd(r), win_rate=float((r > 0).mean()))
    return acc, stats


def t5_all_variants(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []; all_acc = []
    for cfg in PORT_VARIANTS:
        acc, stats = t5_replay(df, cfg["max_open"])
        stats["variant"] = cfg["name"]; stats["max_open"] = cfg["max_open"]
        rows.append(stats)
        if not acc.empty:
            acc["variant"] = cfg["name"]
            all_acc.append(acc)
    summary = pd.DataFrame(rows)
    trades  = pd.concat(all_acc, ignore_index=True) if all_acc else pd.DataFrame()
    return summary, trades


# =============================================================================
# T6 -- CAPITAL EXECUTION
# =============================================================================

def t6_equity(acc_trades: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if acc_trades.empty:
        return pd.DataFrame(), {}

    equity   = START_CAP
    peak_eq  = START_CAP
    kill     = False
    rows     = []

    for _, row in acc_trades.sort_values("exit_time").iterrows():
        risk_amt = equity * RISK_PCT
        pnl      = row[R_COL] * risk_amt
        equity  += pnl
        peak_eq  = max(peak_eq, equity)
        dd_pct   = (equity - peak_eq) / peak_eq
        if dd_pct <= KILL_SWITCH_DD and not kill:
            kill = True
        rows.append(dict(
            exit_time=row["exit_time"], symbol=row["symbol"],
            net_r=row[R_COL], equity=equity, peak=peak_eq,
            dd_pct=float(dd_pct), kill_fired=kill,
        ))

    eq_df = pd.DataFrame(rows)
    final = float(eq_df["equity"].iloc[-1])
    start = acc_trades["entry_time"].min()
    end   = acc_trades["exit_time"].max()
    years = float((end - start).days / 365.25) if pd.notnull(start) and pd.notnull(end) else 1.0
    cagr  = float((final / START_CAP) ** (1 / years) - 1) if years > 0 else 0.0
    max_dd_pct = float(eq_df["dd_pct"].min())

    stats = dict(
        start_capital=START_CAP, end_capital=final,
        total_return_pct=float((final - START_CAP) / START_CAP * 100),
        cagr_pct=float(cagr * 100),
        max_dd_pct=float(max_dd_pct * 100),
        years=round(years, 2),
        kill_switch_fired=kill,
    )
    return eq_df, stats


# =============================================================================
# T7 -- ASSET ROBUSTNESS
# =============================================================================

def t7_remove_top1(df: pd.DataFrame, t5_max_open: int = 3) -> dict:
    asset_r    = df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    top1       = asset_r.index[0]
    df_no_top1 = df[df["symbol"] != top1].copy()
    acc, stats = t5_replay(df_no_top1, t5_max_open)
    _, cap_stats = t6_equity(acc)
    stats.update(removed_symbol=top1,
                 removed_r=float(asset_r.iloc[0]),
                 cagr_pct=cap_stats.get("cagr_pct", 0.0),
                 max_dd_pct=cap_stats.get("max_dd_pct", 0.0))
    return stats


# =============================================================================
# T8 -- MASTER REPORT
# =============================================================================

def write_master_report(
    baseline: dict, mc: pd.DataFrame, cost: pd.DataFrame,
    splits: pd.DataFrame, rem_assets: pd.DataFrame,
    rem_months: pd.DataFrame, conc: dict,
    t5_summary: pd.DataFrame, t6_by_variant: Dict[str, dict],
    t7: dict,
) -> None:

    lines = [
        "PHASE T17 STEP C -- DonchianLong_UniverseV2 Master Report",
        "=" * 70,
        "",
        "Universe:  24 symbols (tightened Step B filter)",
        "Config:    Donchian N=20 / ema200_price / ATR(14)x2.0 / Chandelier ACT+4R trail 3xATR",
        "Data:      2020-05-20 -> 2026-05-29 (≈6 years per symbol)",
        "",
        f"BASELINE (original 64-sym, ~4yr): avg_r=+0.179R  PF=1.27  CAGR=+19%  DD=-1.78% (max3)",
        "",
        "=" * 70, "T4 -- BASELINE SUMMARY", "=" * 70,
        f"  Trades:     {baseline['trades']}",
        f"  Total R:    {baseline['total_r']:+.2f}R",
        f"  Avg R:      {baseline['avg_r']:+.4f}R   (baseline: +0.179R)",
        f"  PF:         {baseline['profit_factor']:.3f}   (baseline: 1.27)",
        f"  Win rate:   {baseline['win_rate']:.1%}",
        f"  Max DD:     {baseline['max_dd_r']:+.2f}R",
        f"  t-score:    {baseline['t_score']:.2f}",
        f"  Assets:     {baseline['assets']}",
        f"  Period:     {baseline['start']} -> {baseline['end']}",
    ]

    mc10 = mc[mc["block_size"] == 10].iloc[0]
    lines += [
        "", "=" * 70, "T4 -- MONTE CARLO (2000 runs, block_size=10)", "=" * 70,
        f"  totalR  p05/p50/p95 = {mc10['total_r_p05']:.1f} / {mc10['total_r_p50']:.1f} / {mc10['total_r_p95']:.1f}",
        f"  prob(totalR > 0)    = {mc10['prob_positive']:.1%}",
        f"  PF      p05/p50     = {mc10['pf_p05']:.2f} / {mc10['pf_p50']:.2f}",
        f"  DD      p95 (worst) = {mc10['dd_p95']:.2f}R",
    ]

    lines += ["", "=" * 70, "T4 -- COST STRESS", "=" * 70]
    for _, r in cost.iterrows():
        lines.append(
            f"  +{r['extra_cost']:.2f}R/trade -> totalR={r['total_r']:+.1f}  "
            f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")

    lines += ["", "=" * 70, "T4 -- PERIOD SPLITS", "=" * 70]
    for _, r in splits.iterrows():
        lines.append(
            f"  {r['split']:30s}: trades={int(r['trades']):3d}  "
            f"totalR={r['total_r']:+7.1f}  avgR={r['avg_r']:+.4f}  "
            f"PF={r['profit_factor']:.3f}  win={r['win_rate']:.1%}")

    lines += [
        "", "=" * 70, "T4 -- REMOVE BEST ASSETS", "=" * 70,
        f"  Top asset: {conc['top1_symbol']}  "
        f"({conc['top1_r']:.2f}R = {conc['top1_pct']:.1%} of total)",
    ]
    for _, r in rem_assets.iterrows():
        lines.append(
            f"  remove top {int(r['removed_n'])}: totalR={r['total_r']:+7.1f}  "
            f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  "
            f"remaining={int(r['assets_remaining'])}  [{r['removed']}]")

    lines += ["", "=" * 70, "T4 -- REMOVE BEST MONTHS", "=" * 70]
    for _, r in rem_months.iterrows():
        lines.append(
            f"  remove top {int(r['removed_n'])}: totalR={r['total_r']:+7.1f}  "
            f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  "
            f"months_remaining={int(r['months_remaining'])}  [{r['removed']}]")

    lines += [
        "", "=" * 70, "T4 -- ASSET CONCENTRATION", "=" * 70,
        f"  Top 1:  {conc['top1_r']:.1f}R  ({conc['top1_pct']:.1%} of total)",
        f"  Top 3:  {conc['top3_r']:.1f}R  ({conc['top3_pct']:.1%} of total)",
        f"  Top 5:  {conc['top5_r']:.1f}R  ({conc['top5_pct']:.1%} of total)",
        f"  Positive assets: {conc['positive_assets']}/{conc['total_assets']}",
    ]

    lines += ["", "=" * 70, "T5 -- PORTFOLIO VARIANTS", "=" * 70,
              f"  {'Variant':8s}  {'Accepted':8s}  {'TotalR':8s}  "
              f"{'AvgR':7s}  {'PF':5s}  {'DD_R':7s}  {'Win%':5s}"]
    for _, r in t5_summary.sort_values("max_open").iterrows():
        lines.append(
            f"  {r['variant']:8s}  {int(r['accepted']):8d}  {r['total_r']:+8.2f}  "
            f"{r['avg_r']:+7.4f}  {r['profit_factor']:.3f}  "
            f"{r['max_dd_r']:+7.2f}  {r['win_rate']:.1%}")

    lines += ["", "=" * 70, "T6 -- CAPITAL EXECUTION ($10,000 / 0.25% risk / 1.0x leverage)", "=" * 70]
    for variant, cs in t6_by_variant.items():
        ks = "YES ***FIRED***" if cs.get("kill_switch_fired") else "no"
        lines.append(
            f"  {variant:8s}: "
            f"${cs.get('end_capital', 0):>9,.0f}  "
            f"return={cs.get('total_return_pct', 0):+.1f}%  "
            f"CAGR={cs.get('cagr_pct', 0):+.1f}%  "
            f"DD={cs.get('max_dd_pct', 0):+.1f}%  "
            f"kill={ks}  "
            f"({cs.get('years', 0):.1f}yr)")
    lines.append(f"  baseline: CAGR=+19.0%  DD=-1.78%  kill=no  (max3, 4yr)")

    lines += [
        "", "=" * 70, "T7 -- ASSET ROBUSTNESS (remove top-1 symbol, max3 portfolio)", "=" * 70,
        f"  Removed:  {t7['removed_symbol']}  ({t7['removed_r']:.1f}R)",
        f"  Accepted: {t7['accepted']}  total_r={t7['total_r']:+.2f}  "
        f"avgR={t7['avg_r']:+.4f}  PF={t7['profit_factor']:.3f}",
        f"  CAGR={t7['cagr_pct']:+.1f}%  DD={t7['max_dd_pct']:+.1f}%",
    ]

    # T8 Scorecard
    max3_t5 = t5_summary[t5_summary["variant"] == "max3"].iloc[0]
    max3_t6 = t6_by_variant.get("max3", {})
    mc_p05  = float(mc10["total_r_p05"])
    cost10  = cost[cost["extra_cost"] == 0.10].iloc[0]
    sec_half = splits[splits["split"] == "second_half_by_trade"].iloc[0]
    rem_top1 = rem_assets[rem_assets["removed_n"] == 1].iloc[0]
    rem_top1_month = rem_months[rem_months["removed_n"] == 1].iloc[0]

    def gate(cond, label):
        return f"  {'PASS' if cond else 'FAIL'}  {label}"

    lines += [
        "", "=" * 70, "T8 -- FINAL SCORECARD  (DonchianLong_UniverseV2 vs Baseline)", "=" * 70, "",
        f"  METHOD:          DonchianLong_UniverseV2",
        f"  UNIVERSE:        24 symbols (tightened filter: avg_r>0, trades>=15, top_month<60%, total_r>0)",
        f"  DATA RANGE:      2020-05-20 -> 2026-05-29  (~6 years)",
        f"  TRADES (T5 max3):{int(max3_t5['accepted'])}",
        f"  WIN RATE:        {max3_t5['win_rate']:.1%}       [gate: 30-45%]",
        f"  AVG R:           {max3_t5['avg_r']:+.4f}R     [gate: >0.15R]",
        f"  PF:              {max3_t5['profit_factor']:.3f}          [gate: >1.0]",
        f"  MAX DD (R):      {max3_t5['max_dd_r']:+.2f}R",
        f"  MAX DD (%):      {max3_t6.get('max_dd_pct', 0):+.1f}%      (baseline: -1.78%)",
        f"  CAGR:            {max3_t6.get('cagr_pct', 0):+.1f}%          (baseline: +19.0%)",
        f"  MC P05 TOTAL R:  {mc_p05:+.1f}R       [gate: >0]",
        f"  COST +0.10R PF:  {float(cost10['profit_factor']):.3f}          [gate: >1.0]",
        f"  2ND HALF AVG R:  {float(sec_half['avg_r']):+.4f}R     [gate: >0]",
        f"  KILL SWITCH:     {'FIRED' if max3_t6.get('kill_switch_fired') else 'no'}",
        "",
        "  GATE RESULTS:",
        gate(30 <= max3_t5["win_rate"] * 100 <= 45, f"§4.1 win rate 30-45%  (got {max3_t5['win_rate']:.1%})"),
        gate(max3_t5["avg_r"] > 0.15, f"§4.2 avg_r > 0.15R    (got {max3_t5['avg_r']:+.4f}R)"),
        gate(mc_p05 > 0, f"MC p05 total_r > 0    (got {mc_p05:+.1f}R)"),
        gate(float(cost10["profit_factor"]) > 1.0, f"Cost +0.10R PF > 1.0  (got {float(cost10['profit_factor']):.3f})"),
        gate(float(rem_top1["total_r"]) > 0, f"Remove top-1 asset profitable  (totalR={float(rem_top1['total_r']):+.1f}R)"),
        gate(float(rem_top1_month["total_r"]) > 0, f"Remove top-1 month profitable  (totalR={float(rem_top1_month['total_r']):+.1f}R)"),
        gate(float(sec_half["avg_r"]) > 0, f"2nd half avg_r > 0    (got {float(sec_half['avg_r']):+.4f}R)"),
        gate(not max3_t6.get("kill_switch_fired", False), "Kill switch not fired"),
        "",
        "  vs BASELINE:",
        f"  avg_r:   {max3_t5['avg_r']:+.4f}R  vs  +0.179R  -> "
        f"{'BETTER' if max3_t5['avg_r'] > 0.179 else 'WORSE'}",
        f"  PF:      {max3_t5['profit_factor']:.3f}  vs  1.27  -> "
        f"{'BETTER' if max3_t5['profit_factor'] > 1.27 else 'WORSE'}",
        f"  CAGR:    {max3_t6.get('cagr_pct', 0):+.1f}%  vs  +19.0%  -> "
        f"{'BETTER' if max3_t6.get('cagr_pct', 0) > 19.0 else 'WORSE'}",
        f"  Max DD%: {max3_t6.get('max_dd_pct', 0):+.1f}%  vs  -1.78%  -> "
        f"{'BETTER' if max3_t6.get('max_dd_pct', 0) > -1.78 else 'WORSE (larger DD)'}",
    ]

    rpt = OUT_DIR / "phase_c_master_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Master report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("Phase T17 Step C -- DonchianLong_UniverseV2  T4->T8")
    print("=" * 70)

    trades = load_trades()
    print(f"Trades loaded:  {len(trades)}")
    print(f"Symbols:        {trades['symbol'].nunique()}")
    print(f"Period:         {trades[EXIT_T].min().date()} -> {trades[EXIT_T].max().date()}")
    print()

    # ── T4 ──────────────────────────────────────────────────────────────────
    print("[T4] Running robustness battery...")
    baseline   = t4_baseline(trades)
    mc         = t4_montecarlo(trades)
    cost       = t4_cost_stress(trades)
    splits     = t4_period_splits(trades)
    rem_assets = t4_remove_best_assets(trades)
    rem_months = t4_remove_best_months(trades)
    conc       = t4_concentration(trades)

    mc.to_csv(OUT_DIR / "phase_c_t4_montecarlo.csv", index=False)
    cost.to_csv(OUT_DIR / "phase_c_t4_cost_stress.csv", index=False)
    splits.to_csv(OUT_DIR / "phase_c_t4_period_splits.csv", index=False)
    rem_assets.to_csv(OUT_DIR / "phase_c_t4_remove_best_assets.csv", index=False)
    rem_months.to_csv(OUT_DIR / "phase_c_t4_remove_best_months.csv", index=False)
    print(f"  baseline: avg_r={baseline['avg_r']:+.4f}  PF={baseline['profit_factor']:.3f}  "
          f"trades={baseline['trades']}")
    mc10 = mc[mc["block_size"] == 10].iloc[0]
    print(f"  MC(bs=10): p05={mc10['total_r_p05']:+.1f}  p50={mc10['total_r_p50']:+.1f}  "
          f"prob_pos={mc10['prob_positive']:.1%}")

    # ── T5 ──────────────────────────────────────────────────────────────────
    print("\n[T5] Portfolio replay...")
    t5_sum, t5_trades = t5_all_variants(trades)
    t5_sum.to_csv(OUT_DIR / "phase_c_t5_portfolio_summary.csv", index=False)
    if not t5_trades.empty:
        t5_trades.to_csv(OUT_DIR / "phase_c_t5_portfolio_trades.csv", index=False)
    for _, r in t5_sum.sort_values("max_open").iterrows():
        print(f"  {r['variant']:8s}: accepted={int(r['accepted'])}  "
              f"totalR={r['total_r']:+.2f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")

    # ── T6 ──────────────────────────────────────────────────────────────────
    print("\n[T6] Capital execution...")
    t6_by_variant: Dict[str, dict] = {}
    for cfg in PORT_VARIANTS:
        name    = cfg["name"]
        acc_sub = t5_trades[t5_trades["variant"] == name] if not t5_trades.empty else pd.DataFrame()
        eq_df, cs = t6_equity(acc_sub)
        t6_by_variant[name] = cs
        if not eq_df.empty:
            eq_df.to_csv(OUT_DIR / f"phase_c_t6_equity_{name}.csv", index=False)
        ks = " ***KILL***" if cs.get("kill_switch_fired") else ""
        print(f"  {name:8s}: ${cs.get('end_capital',0):>9,.0f}  "
              f"CAGR={cs.get('cagr_pct',0):+.1f}%  "
              f"DD={cs.get('max_dd_pct',0):+.1f}%{ks}")
    print(f"  baseline:  CAGR=+19.0%  DD=-1.78%  (max3)")

    # ── T7 ──────────────────────────────────────────────────────────────────
    print("\n[T7] Asset robustness (remove top-1, max3)...")
    t7 = t7_remove_top1(trades, t5_max_open=3)
    print(f"  Removed {t7['removed_symbol']}: "
          f"remaining totalR={t7['total_r']:+.2f}  "
          f"avgR={t7['avg_r']:+.4f}  CAGR={t7['cagr_pct']:+.1f}%")

    # ── T8 / Master report ──────────────────────────────────────────────────
    print("\n[T8] Writing master report...")
    write_master_report(
        baseline=baseline, mc=mc, cost=cost, splits=splits,
        rem_assets=rem_assets, rem_months=rem_months, conc=conc,
        t5_summary=t5_sum, t6_by_variant=t6_by_variant, t7=t7,
    )

    print()
    print("=" * 70)
    print("Step C complete -> data/research_donchianlong_universev2/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
