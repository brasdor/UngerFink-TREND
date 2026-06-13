#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T4 -- LinearRegressionBreakout Robustness Engine

Runs on both Config A (N=40/mult=2.5) and Config B (N=55/mult=3.0).
T3B found chandelier DEGRADES performance for both; T2 midline exit is canonical.

Battery per config:
  - Baseline summary
  - Block bootstrap Monte Carlo (2000 runs x 5 block sizes)
  - Extra cost stress (+0.00 to +0.20R/trade)
  - Rolling calendar windows (180d, step 90d)
  - Period splits (first/second half, last 100/200 trades)
  - Remove best assets (top 1/3/5/10)
  - Remove best months (top 1/2/3)
  - Asset concentration

Input:  data/research_linregbreakout_t2/phase_t2_linregbreakout_cfg{A,B}_trades.csv
Output: data/research_linregbreakout_t4/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT    = Path(__file__).resolve().parents[1]
IN_DIR  = ROOT / "data" / "research_linregbreakout_t2"
OUT_DIR = ROOT / "data" / "research_linregbreakout_t4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    dict(
        label="A",
        linreg_n=40, linreg_mult=2.5, stop_mult=2.0,
        input_file=IN_DIR / "phase_t2_linregbreakout_cfgA_trades.csv",
        t2_trades=271, t2_total_r=64.24, t2_avg_r=0.2370, t2_pf=1.393,
    ),
    dict(
        label="B",
        linreg_n=55, linreg_mult=3.0, stop_mult=2.0,
        input_file=IN_DIR / "phase_t2_linregbreakout_cfgB_trades.csv",
        t2_trades=196, t2_total_r=71.04, t2_avg_r=0.3624, t2_pf=1.528,
    ),
]

R_COL    = "net_r"
TIME_COL = "exit_time"

MC_RUNS        = 2000
MC_BLOCK_SIZES = [1, 3, 5, 10, 20]
MC_SEED        = 42

EXTRA_COST_R_VALUES     = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
REMOVE_TOP_ASSET_COUNTS = [1, 3, 5, 10]
REMOVE_TOP_MONTH_COUNTS = [1, 2, 3]

ROLLING_WINDOW_DAYS = 180
ROLLING_STEP_DAYS   = 90

MIN_TRADES_FOR_MC = 30


# =============================================================================
# HELPERS
# =============================================================================

def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if "timeframe" not in df.columns:
        df["timeframe"] = "1d"
    if "side" not in df.columns:
        df["side"] = "LONG"

    required = {"symbol", "timeframe", "side", TIME_COL, R_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce", utc=True,
                                   format="mixed")
    out = out.dropna(subset=[TIME_COL, R_COL])

    # Fix 1970-epoch bug: stored Unix-ms integers were read as nanoseconds.
    if not out.empty and out[TIME_COL].dt.year.max() < 2000:
        ns_vals = out[TIME_COL].astype("int64")
        out[TIME_COL] = pd.to_datetime(ns_vals, unit="ms", utc=True)

    out = out.sort_values(TIME_COL).reset_index(drop=True)
    out["month"]     = out[TIME_COL].dt.to_period("M").astype(str)
    out["year"]      = out[TIME_COL].dt.year
    out["date"]      = out[TIME_COL].dt.date
    out["side"]      = out["side"].astype(str).str.upper()
    out["timeframe"] = out["timeframe"].astype(str).str.lower()
    return out


def max_drawdown_r(r_values: Iterable[float]) -> float:
    arr = np.asarray(list(r_values), dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.cumsum(arr)
    peak   = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def longest_losing_streak(r_values: Iterable[float]) -> int:
    max_s = cur = 0
    for r in r_values:
        if r < 0:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def profit_factor(r_values: Iterable[float]) -> float:
    arr = np.asarray(list(r_values), dtype=float)
    g = arr[arr > 0].sum()
    l = -arr[arr < 0].sum()
    if l <= 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def summarize_r(r_values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(r_values), dtype=float)
    if arr.size == 0:
        return dict(trades=0, total_r=0.0, avg_r=0.0, median_r=0.0,
                    win_rate=0.0, profit_factor=0.0, max_dd_r=0.0,
                    best_trade_r=0.0, worst_trade_r=0.0,
                    losing_streak=0, std_r=0.0, t_score=0.0)
    avg = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    t   = avg / (std / math.sqrt(arr.size)) if std > 0 and arr.size > 1 else 0.0
    return dict(
        trades=int(arr.size),
        total_r=float(arr.sum()),
        avg_r=avg,
        median_r=float(np.median(arr)),
        win_rate=float((arr > 0).mean()),
        profit_factor=profit_factor(arr),
        max_dd_r=max_drawdown_r(arr),
        best_trade_r=float(arr.max()),
        worst_trade_r=float(arr.min()),
        losing_streak=int(longest_losing_streak(arr)),
        std_r=std,
        t_score=float(t),
    )


def group_slices(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    groups: List[Tuple[str, str, pd.DataFrame]] = []
    for tf in sorted(df["timeframe"].unique()):
        tf_df = df[df["timeframe"] == tf].copy()
        groups.append((tf, "ALL", tf_df))
        for side in ["LONG", "SHORT"]:
            s = tf_df[tf_df["side"] == side].copy()
            if not s.empty:
                groups.append((tf, side, s))
    groups.append(("ALL_TF", "ALL", df.copy()))
    return groups


# =============================================================================
# ANALYSES
# =============================================================================

def baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        s = summarize_r(g[R_COL].to_numpy())
        s.update(dict(
            timeframe=tf, side=side,
            start=g[TIME_COL].min(), end=g[TIME_COL].max(),
            assets=g["symbol"].nunique(), months=g["month"].nunique(),
        ))
        rows.append(s)
    cols = ["timeframe", "side", "start", "end", "assets", "months",
            "trades", "total_r", "avg_r", "median_r", "win_rate",
            "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
            "losing_streak", "std_r", "t_score"]
    return pd.DataFrame(rows)[cols]


def block_bootstrap(values: np.ndarray, block_size: int,
                    rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    sampled: List[float] = []
    while len(sampled) < n:
        start = int(rng.integers(0, n))
        end   = min(start + block_size, n)
        block = values[start:end]
        if len(block) < block_size:
            block = np.concatenate([block, values[:block_size - len(block)]])
        sampled.extend(block.tolist())
    return np.asarray(sampled[:n], dtype=float)


def montecarlo_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(MC_SEED)
    for tf, side, g in group_slices(df):
        values = g[R_COL].to_numpy(dtype=float)
        if len(values) < MIN_TRADES_FOR_MC:
            continue
        for bs in MC_BLOCK_SIZES:
            totals = np.empty(MC_RUNS)
            dds    = np.empty(MC_RUNS)
            pfs    = np.empty(MC_RUNS)
            for i in range(MC_RUNS):
                s         = block_bootstrap(values, bs, rng)
                totals[i] = s.sum()
                dds[i]    = max_drawdown_r(s)
                pfs[i]    = profit_factor(s)
            rows.append(dict(
                timeframe=tf, side=side, block_size=bs,
                trades=len(values), mc_runs=MC_RUNS,
                total_r_p05=float(np.percentile(totals, 5)),
                total_r_p50=float(np.percentile(totals, 50)),
                total_r_p95=float(np.percentile(totals, 95)),
                dd_r_p05=float(np.percentile(dds, 5)),
                dd_r_p50=float(np.percentile(dds, 50)),
                dd_r_p95=float(np.percentile(dds, 95)),
                pf_p05=float(np.percentile(pfs, 5)),
                pf_p50=float(np.percentile(pfs, 50)),
                pf_p95=float(np.percentile(pfs, 95)),
                prob_total_r_positive=float((totals > 0).mean()),
                prob_pf_above_1=float((pfs > 1.0).mean()),
            ))
    return pd.DataFrame(rows)


def cost_stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        values = g[R_COL].to_numpy(dtype=float)
        for ec in EXTRA_COST_R_VALUES:
            s = summarize_r(values - ec)
            s.update(dict(timeframe=tf, side=side, extra_cost_r_per_trade=ec))
            rows.append(s)
    cols = ["timeframe", "side", "extra_cost_r_per_trade",
            "trades", "total_r", "avg_r", "median_r", "win_rate",
            "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
            "losing_streak", "std_r", "t_score"]
    return pd.DataFrame(rows)[cols]


def rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        if g.empty:
            continue
        start = g[TIME_COL].min().normalize()
        end   = g[TIME_COL].max().normalize()
        cur   = start
        while cur < end:
            win_end = cur + pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            w = g[(g[TIME_COL] >= cur) & (g[TIME_COL] < win_end)]
            s = summarize_r(w[R_COL].to_numpy(dtype=float))
            s.update(dict(timeframe=tf, side=side,
                          window_start=cur, window_end=win_end,
                          assets=w["symbol"].nunique() if not w.empty else 0))
            rows.append(s)
            cur += pd.Timedelta(days=ROLLING_STEP_DAYS)
    cols = ["timeframe", "side", "window_start", "window_end", "assets",
            "trades", "total_r", "avg_r", "median_r", "win_rate",
            "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
            "losing_streak", "std_r", "t_score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def period_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        if g.empty:
            continue
        mid_idx  = len(g) // 2
        median_t = g[TIME_COL].median()
        slices = {
            "first_half_by_trade_count":  g.iloc[:mid_idx],
            "second_half_by_trade_count": g.iloc[mid_idx:],
            "last_100_trades":  g.tail(100),
            "last_200_trades":  g.tail(200),
            "first_half_by_time":  g[g[TIME_COL] <= median_t],
            "second_half_by_time": g[g[TIME_COL] >  median_t],
        }
        for name, sub in slices.items():
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update(dict(
                timeframe=tf, side=side, split=name,
                start=sub[TIME_COL].min() if not sub.empty else pd.NaT,
                end=sub[TIME_COL].max()   if not sub.empty else pd.NaT,
                assets=sub["symbol"].nunique() if not sub.empty else 0,
            ))
            rows.append(s)
    cols = ["timeframe", "side", "split", "start", "end", "assets",
            "trades", "total_r", "avg_r", "median_r", "win_rate",
            "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
            "losing_streak", "std_r", "t_score"]
    return pd.DataFrame(rows)[cols]


def remove_best_assets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        if g.empty:
            continue
        asset_perf = g.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
        for n in [0] + REMOVE_TOP_ASSET_COUNTS:
            removed = asset_perf.head(n).index.tolist()
            sub = g[~g["symbol"].isin(removed)].copy()
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update(dict(
                timeframe=tf, side=side,
                removed_top_assets_n=n,
                removed_assets=",".join(removed),
                assets_remaining=sub["symbol"].nunique(),
            ))
            rows.append(s)
    cols = ["timeframe", "side", "removed_top_assets_n", "removed_assets",
            "assets_remaining", "trades", "total_r", "avg_r", "median_r",
            "win_rate", "profit_factor", "max_dd_r", "best_trade_r",
            "worst_trade_r", "losing_streak", "std_r", "t_score"]
    return pd.DataFrame(rows)[cols]


def remove_best_months(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        if g.empty:
            continue
        month_perf = g.groupby("month")[R_COL].sum().sort_values(ascending=False)
        for n in [0] + REMOVE_TOP_MONTH_COUNTS:
            removed = month_perf.head(n).index.tolist()
            sub = g[~g["month"].isin(removed)].copy()
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update(dict(
                timeframe=tf, side=side,
                removed_top_months_n=n,
                removed_months=",".join(removed),
                months_remaining=sub["month"].nunique(),
            ))
            rows.append(s)
    cols = ["timeframe", "side", "removed_top_months_n", "removed_months",
            "months_remaining", "trades", "total_r", "avg_r", "median_r",
            "win_rate", "profit_factor", "max_dd_r", "best_trade_r",
            "worst_trade_r", "losing_streak", "std_r", "t_score"]
    return pd.DataFrame(rows)[cols]


def asset_concentration(df: pd.DataFrame, cfg_label: str) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        if g.empty:
            continue
        asset = (
            g.groupby("symbol")
            .agg(trades=(R_COL, "size"), total_r=(R_COL, "sum"),
                 avg_r=(R_COL, "mean"), best_trade_r=(R_COL, "max"),
                 worst_trade_r=(R_COL, "min"))
            .reset_index()
            .sort_values("total_r", ascending=False)
        )
        total_pos = asset.loc[asset["total_r"] > 0, "total_r"].sum()
        total_all = asset["total_r"].sum()
        top1  = asset["total_r"].head(1).sum()
        top3  = asset["total_r"].head(3).sum()
        top5  = asset["total_r"].head(5).sum()
        top10 = asset["total_r"].head(10).sum()
        rows.append(dict(
            timeframe=tf, side=side,
            assets=int(asset["symbol"].nunique()),
            positive_assets=int((asset["total_r"] > 0).sum()),
            negative_assets=int((asset["total_r"] < 0).sum()),
            total_r=float(total_all),
            top1_asset=asset["symbol"].iloc[0] if len(asset) else "",
            top1_r=float(top1),
            top3_r=float(top3),
            top5_r=float(top5),
            top10_r=float(top10),
            top1_pct_of_total=float(top1 / total_all) if total_all != 0 else float("nan"),
            top3_pct_of_total=float(top3 / total_all) if total_all != 0 else float("nan"),
            top5_pct_of_total=float(top5 / total_all) if total_all != 0 else float("nan"),
            top10_pct_of_total=float(top10 / total_all) if total_all != 0 else float("nan"),
            top1_pct_of_positive_r=float(top1 / total_pos) if total_pos > 0 else float("nan"),
            top3_pct_of_positive_r=float(top3 / total_pos) if total_pos > 0 else float("nan"),
            top5_pct_of_positive_r=float(top5 / total_pos) if total_pos > 0 else float("nan"),
            top10_pct_of_positive_r=float(top10 / total_pos) if total_pos > 0 else float("nan"),
        ))
        safe = f"{str(tf).replace('/', '_')}_{str(side).replace('/', '_')}"
        asset.to_csv(
            OUT_DIR / f"phase_t4_linregbreakout_cfg{cfg_label}_asset_rank_{safe}.csv",
            index=False
        )
    return pd.DataFrame(rows)


# =============================================================================
# MASTER REPORT (per config)
# =============================================================================

def write_master_report(
    cfg: dict,
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    mc: pd.DataFrame,
    cost: pd.DataFrame,
    splits: pd.DataFrame,
    rem_assets: pd.DataFrame,
    rem_months: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    label      = cfg["label"]
    report_path = OUT_DIR / f"phase_t4_linregbreakout_cfg{label}_master_report.txt"

    top1_asset = ""
    top1_r     = 0.0
    top1_pct   = 0.0
    conc1 = concentration[(concentration["timeframe"] == "1d") & (concentration["side"] == "ALL")]
    if not conc1.empty:
        c = conc1.iloc[0]
        top1_asset = c["top1_asset"]
        top1_r     = c["top1_r"]
        top1_pct   = c["top1_pct_of_total"]

    def fmt(row: pd.Series) -> str:
        return (
            f"  {row['timeframe']} {row['side']}: "
            f"trades={int(row['trades'])}, "
            f"totalR={row['total_r']:.2f}, "
            f"avgR={row['avg_r']:.3f}, "
            f"PF={row['profit_factor']:.2f}, "
            f"DD={row['max_dd_r']:.2f}, "
            f"win={row['win_rate']:.1%}"
        )

    lines = [
        f"PHASE T4 -- LinearRegressionBreakout Config {label} ROBUSTNESS MASTER REPORT",
        "=" * 70,
        "",
        f"Method:  LinearRegressionBreakout (LinReg(N) + mult*StdErr(N) upper band)",
        f"Config:  N={cfg['linreg_n']} / mult={cfg['linreg_mult']} / sm={cfg['stop_mult']} / exit: close < LinReg(N) (midline)",
        "Note:    T3B found chandelier DEGRADES performance; T2 midline exit is canonical.",
        f"Input:   {cfg['input_file']}",
        f"Trades:  {len(df)}",
        f"Assets:  {df['symbol'].nunique()}",
        f"Range:   {df[TIME_COL].min()} -> {df[TIME_COL].max()}",
        "",
        "=" * 70,
        "BASELINE SUMMARY",
        "=" * 70,
    ]
    for _, row in baseline.sort_values(["timeframe", "side"]).iterrows():
        lines.append(fmt(row))

    lines += ["", "=" * 70, "MONTE CARLO (1d ALL, block_size=10)", "=" * 70]
    mc1 = mc[(mc["timeframe"] == "1d") & (mc["side"] == "ALL") & (mc["block_size"] == 10)]
    if not mc1.empty:
        r = mc1.iloc[0]
        lines.append(
            f"  totalR p05/p50/p95 = {r['total_r_p05']:.2f} / {r['total_r_p50']:.2f} / {r['total_r_p95']:.2f}"
        )
        lines.append(f"  prob(totalR>0)     = {r['prob_total_r_positive']:.1%}")
        lines.append(f"  PF p05/p50/p95     = {r['pf_p05']:.2f} / {r['pf_p50']:.2f} / {r['pf_p95']:.2f}")
        lines.append(f"  DD p95 (worst)     = {r['dd_r_p95']:.2f}R")

    lines += ["", "=" * 70, "COST STRESS (1d ALL)", "=" * 70]
    c1 = cost[(cost["timeframe"] == "1d") & (cost["side"] == "ALL")]
    for _, r in c1.iterrows():
        lines.append(
            f"  +{r['extra_cost_r_per_trade']:.2f}R/trade -> "
            f"totalR={r['total_r']:.2f}  avgR={r['avg_r']:.3f}  "
            f"PF={r['profit_factor']:.2f}  DD={r['max_dd_r']:.2f}"
        )

    lines += ["", "=" * 70, "PERIOD SPLITS (1d ALL)", "=" * 70]
    sp1 = splits[(splits["timeframe"] == "1d") & (splits["side"] == "ALL")]
    for _, r in sp1.iterrows():
        lines.append(
            f"  {r['split']:35s}: trades={int(r['trades']):3d}  "
            f"totalR={r['total_r']:+7.2f}  avgR={r['avg_r']:+.3f}  "
            f"PF={r['profit_factor']:.2f}  win={r['win_rate']:.1%}"
        )

    lines += [
        "", "=" * 70, "REMOVE BEST ASSETS (1d ALL)", "=" * 70,
        f"  Note: {top1_asset} = {top1_r:.2f}R / {cfg['t2_total_r']:.2f}R total = "
        f"{top1_pct:.1%} concentration (pre-flagged T2).",
        "  Key question: does removing top-1 invert the system?"
    ]
    ra1 = rem_assets[(rem_assets["timeframe"] == "1d") & (rem_assets["side"] == "ALL")]
    for _, r in ra1.iterrows():
        n = int(r["removed_top_assets_n"])
        lines.append(
            f"  remove top {n}: totalR={r['total_r']:+7.2f}  avgR={r['avg_r']:+.3f}  "
            f"PF={r['profit_factor']:.2f}  remaining={int(r['assets_remaining'])}  "
            f"removed=[{r['removed_assets']}]"
        )

    lines += ["", "=" * 70, "REMOVE BEST MONTHS (1d ALL)", "=" * 70]
    rm1 = rem_months[(rem_months["timeframe"] == "1d") & (rem_months["side"] == "ALL")]
    for _, r in rm1.iterrows():
        lines.append(
            f"  remove top {int(r['removed_top_months_n'])}: "
            f"totalR={r['total_r']:+7.2f}  avgR={r['avg_r']:+.3f}  "
            f"PF={r['profit_factor']:.2f}  months_remaining={int(r['months_remaining'])}  "
            f"removed=[{r['removed_months']}]"
        )

    if not conc1.empty:
        c = conc1.iloc[0]
        lines += [
            "", "=" * 70, "ASSET CONCENTRATION (1d ALL)", "=" * 70,
            f"  Top asset: {c['top1_asset']}  ({c['top1_r']:.2f}R, {c['top1_pct_of_total']:.1%} of total)",
            f"  Top 3:     {c['top3_r']:.2f}R  ({c['top3_pct_of_total']:.1%} of total)",
            f"  Top 5:     {c['top5_r']:.2f}R  ({c['top5_pct_of_total']:.1%} of total)",
            f"  Top 10:    {c['top10_r']:.2f}R  ({c['top10_pct_of_total']:.1%} of total)",
            f"  Positive assets: {int(c['positive_assets'])} / {int(c['assets'])}",
            f"  Negative assets: {int(c['negative_assets'])} / {int(c['assets'])}",
        ]

    lines += [
        "", "=" * 70, "INTERPRETATION GUIDE", "=" * 70, "",
        "  PROCEED TO T5 if:",
        "  - MC p05 totalR remains positive (or near zero)",
        "  - PF stays > 1.0 after moderate cost stress (+0.10R)",
        "  - Removing top 1-3 assets leaves system profitable",
        "  - First/second half both positive (no structural decay)",
        "",
        "  HALT / REVIEW if:",
        f"  - Removing {top1_asset} (top-1) inverts the system (total_r < 0)",
        "  - MC p05 deeply negative",
        "  - Recent last-100/200 trades sharply worse (decay)",
        "  - Cost stress +0.10R destroys edge",
        "",
        f"  {top1_asset} concentration note: {top1_pct:.1%} is significant.",
        "  If remaining symbols stay profitable after top-1 removal -> diversified edge exists.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path}")


# =============================================================================
# PER-CONFIG RUNNER
# =============================================================================

def run_config(cfg: dict) -> None:
    label = cfg["label"]
    input_file = cfg["input_file"]
    prefix = f"phase_t4_linregbreakout_cfg{label}"

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return

    raw    = pd.read_csv(input_file)
    trades = normalize_trades(raw)

    print(f"\n{'=' * 70}")
    print(f"Config {label}  (N={cfg['linreg_n']} / mult={cfg['linreg_mult']} / sm={cfg['stop_mult']})")
    print(f"{'=' * 70}")
    print(f"Input:  {input_file}")
    print(f"Trades: {len(trades)}")
    print(f"Assets: {trades['symbol'].nunique()}")
    print(f"Range:  {trades[TIME_COL].min()} -> {trades[TIME_COL].max()}")

    baseline = baseline_summary(trades)
    baseline.to_csv(OUT_DIR / f"{prefix}_baseline_summary.csv", index=False)
    print("[OK] baseline summary")

    mc = montecarlo_summary(trades)
    mc.to_csv(OUT_DIR / f"{prefix}_montecarlo_summary.csv", index=False)
    print(f"[OK] Monte Carlo ({MC_RUNS} runs x {len(MC_BLOCK_SIZES)} block sizes)")

    cost = cost_stress_summary(trades)
    cost.to_csv(OUT_DIR / f"{prefix}_cost_stress_summary.csv", index=False)
    print("[OK] cost stress")

    rolling = rolling_windows(trades)
    rolling.to_csv(OUT_DIR / f"{prefix}_rolling_windows.csv", index=False)
    print("[OK] rolling windows")

    splits = period_splits(trades)
    splits.to_csv(OUT_DIR / f"{prefix}_period_splits.csv", index=False)
    print("[OK] period splits")

    rem_assets = remove_best_assets(trades)
    rem_assets.to_csv(OUT_DIR / f"{prefix}_remove_best_assets.csv", index=False)
    print("[OK] remove best assets")

    rem_months = remove_best_months(trades)
    rem_months.to_csv(OUT_DIR / f"{prefix}_remove_best_months.csv", index=False)
    print("[OK] remove best months")

    concentration = asset_concentration(trades, label)
    concentration.to_csv(OUT_DIR / f"{prefix}_asset_concentration.csv", index=False)
    print("[OK] asset concentration")

    write_master_report(
        cfg=cfg,
        df=trades,
        baseline=baseline,
        mc=mc,
        cost=cost,
        splits=splits,
        rem_assets=rem_assets,
        rem_months=rem_months,
        concentration=concentration,
    )
    print("[OK] master report")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T4 -- LinearRegressionBreakout Robustness Engine")
    print("=" * 70)
    print(f"Configs: {len(CONFIGS)}  (A: N=40/mult=2.5  B: N=55/mult=3.0)")
    print(f"Output:  {OUT_DIR}")

    for cfg in CONFIGS:
        run_config(cfg)

    print()
    print("=" * 70)
    print("T4 COMPLETE -- review master reports before T5")
    print(f"  Config A: {OUT_DIR / 'phase_t4_linregbreakout_cfgA_master_report.txt'}")
    print(f"  Config B: {OUT_DIR / 'phase_t4_linregbreakout_cfgB_master_report.txt'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
