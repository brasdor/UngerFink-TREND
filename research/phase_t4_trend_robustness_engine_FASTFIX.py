#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T4 — Trend Robustness Engine

Purpose
-------
Take the Phase T3B wide-exit trend trades and try to "break" the system.

This script does NOT change entry/exit logic.
It only analyzes closed trades from Phase T3B.

Main tests:
- baseline stats by timeframe / side
- block bootstrap Monte Carlo
- extra execution cost stress
- rolling calendar windows
- first half / second half / last 100 trades
- remove best assets
- remove best months
- asset concentration diagnostics

Expected input:
    data/research_trend_t3b/phase_t3b_wide_exit_trades.csv

Fallback input:
    phase_t3b_wide_exit_trades.csv in the current working directory

Outputs:
    data/research_trend_t4/
        phase_t4_baseline_summary.csv
        phase_t4_montecarlo_summary.csv
        phase_t4_cost_stress_summary.csv
        phase_t4_rolling_windows.csv
        phase_t4_period_splits.csv
        phase_t4_remove_best_assets.csv
        phase_t4_remove_best_months.csv
        phase_t4_asset_concentration.csv
        phase_t4_master_report.txt

Author:
    Jean / ChatGPT workflow
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

PROJECT_ROOT = Path.cwd()

INPUT_CANDIDATES = [
    PROJECT_ROOT / "data" / "research_trend_t3b" / "phase_t3b_wide_exit_trades.csv",
    PROJECT_ROOT / "phase_t3b_wide_exit_trades.csv",
]

OUT_DIR = PROJECT_ROOT / "data" / "research_trend_t4"

R_COL = "net_r"
TIME_COL = "exit_time"

# Monte Carlo settings
MC_RUNS = 500
MC_BLOCK_SIZES = [1, 5, 10]
# Fast but still useful for first T4 diagnosis. Later you can raise MC_RUNS to 2000 and add blocks 3/20.
FAST_MODE = True
MC_SEED = 42

# Cost stress expressed in additional R lost per trade.
# Example: 0.05 means every trade loses an additional 0.05R.
EXTRA_COST_R_VALUES = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# Robustness removals
REMOVE_TOP_ASSET_COUNTS = [1, 3, 5, 10]
REMOVE_TOP_MONTH_COUNTS = [1, 2, 3]

# Rolling window settings
ROLLING_WINDOW_DAYS = 180
ROLLING_STEP_DAYS = 90

# Focus groups. "ALL" is included automatically.
# The script will analyze all timeframe + side groups it finds.
MIN_TRADES_FOR_MC = 30


# =============================================================================
# CORE HELPERS
# =============================================================================

def find_input_file() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find Phase T3B trades file. Expected one of:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "timeframe", "side", TIME_COL, R_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trades file: {sorted(missing)}")

    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce", utc=True)
    out = out.dropna(subset=[TIME_COL, R_COL])
    out = out.sort_values(TIME_COL).reset_index(drop=True)

    # Remove timezone before Period conversion to avoid the pandas warning.
    out["month"] = out[TIME_COL].dt.tz_convert(None).dt.to_period("M").astype(str)
    out["year"] = out[TIME_COL].dt.year
    out["date"] = out[TIME_COL].dt.date

    # Standardize side strings
    out["side"] = out["side"].astype(str).str.upper()
    out["timeframe"] = out["timeframe"].astype(str).str.lower()

    return out


def max_drawdown_r(r_values: Iterable[float]) -> float:
    arr = np.asarray(r_values, dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def longest_losing_streak(r_values: Iterable[float]) -> int:
    max_streak = 0
    cur = 0
    for r in r_values:
        if r < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def profit_factor(r_values: Iterable[float]) -> float:
    arr = np.asarray(r_values, dtype=float)
    gross_profit = arr[arr > 0].sum()
    gross_loss = -arr[arr < 0].sum()
    if gross_loss <= 0:
        return np.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def summarize_r(r_values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(r_values, dtype=float)
    if arr.size == 0:
        return {
            "trades": 0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "best_trade_r": 0.0,
            "worst_trade_r": 0.0,
            "losing_streak": 0,
            "std_r": 0.0,
            "t_score": 0.0,
        }

    avg = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    t_score = avg / (std / math.sqrt(arr.size)) if std > 0 and arr.size > 1 else 0.0

    return {
        "trades": int(arr.size),
        "total_r": float(arr.sum()),
        "avg_r": avg,
        "median_r": float(np.median(arr)),
        "win_rate": float((arr > 0).mean()),
        "profit_factor": profit_factor(arr),
        "max_dd_r": max_drawdown_r(arr),
        "best_trade_r": float(arr.max()),
        "worst_trade_r": float(arr.min()),
        "losing_streak": int(longest_losing_streak(arr)),
        "std_r": std,
        "t_score": float(t_score),
    }


def group_slices(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    groups: List[Tuple[str, str, pd.DataFrame]] = []

    for tf in sorted(df["timeframe"].unique()):
        tf_df = df[df["timeframe"] == tf].copy()
        groups.append((tf, "ALL", tf_df))
        for side in ["LONG", "SHORT"]:
            s = tf_df[tf_df["side"] == side].copy()
            if not s.empty:
                groups.append((tf, side, s))

    # Full-universe aggregate across all timeframes, mostly as diagnostic.
    groups.append(("ALL_TF", "ALL", df.copy()))
    return groups


# =============================================================================
# ANALYSES
# =============================================================================

def baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        s = summarize_r(g[R_COL].to_numpy())
        s.update({
            "timeframe": tf,
            "side": side,
            "start": g[TIME_COL].min(),
            "end": g[TIME_COL].max(),
            "assets": g["symbol"].nunique(),
            "months": g["month"].nunique(),
        })
        rows.append(s)

    cols = ["timeframe", "side", "start", "end", "assets", "months"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def block_bootstrap(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Fast circular block bootstrap. Avoids Python list extension in the MC loop."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    if block_size <= 1:
        return values[rng.integers(0, n, size=n)]

    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]) % n
    return values[idx.ravel()[:n]]


def max_drawdown_many(samples: np.ndarray) -> np.ndarray:
    """Vectorized max drawdown for a 2D array: rows=simulations, cols=trades."""
    if samples.size == 0:
        return np.array([], dtype=float)
    equity = np.cumsum(samples, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    dd = equity - peaks
    return dd.min(axis=1)


def profit_factor_many(samples: np.ndarray) -> np.ndarray:
    gp = np.where(samples > 0, samples, 0.0).sum(axis=1)
    gl = -np.where(samples < 0, samples, 0.0).sum(axis=1)
    return np.divide(gp, gl, out=np.zeros_like(gp, dtype=float), where=gl > 0)


def montecarlo_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(MC_SEED)

    for tf, side, g in group_slices(df):
        values = g[R_COL].to_numpy(dtype=float)
        n = len(values)
        if n < MIN_TRADES_FOR_MC:
            continue

        print(f"[MC] {tf} {side}: trades={n}, runs={MC_RUNS}, blocks={MC_BLOCK_SIZES}")

        for block_size in MC_BLOCK_SIZES:
            if block_size <= 1:
                idx = rng.integers(0, n, size=(MC_RUNS, n))
                samples = values[idx]
            else:
                n_blocks = int(math.ceil(n / block_size))
                starts = rng.integers(0, n, size=(MC_RUNS, n_blocks))
                offsets = np.arange(block_size)
                idx = (starts[:, :, None] + offsets[None, None, :]) % n
                idx = idx.reshape(MC_RUNS, -1)[:, :n]
                samples = values[idx]

            totals = samples.sum(axis=1)
            dds = max_drawdown_many(samples)
            pfs = profit_factor_many(samples)

            rows.append({
                "timeframe": tf,
                "side": side,
                "block_size": block_size,
                "trades": n,
                "mc_runs": MC_RUNS,
                "total_r_p05": float(np.percentile(totals, 5)),
                "total_r_p50": float(np.percentile(totals, 50)),
                "total_r_p95": float(np.percentile(totals, 95)),
                "dd_r_p05": float(np.percentile(dds, 5)),
                "dd_r_p50": float(np.percentile(dds, 50)),
                "dd_r_p95": float(np.percentile(dds, 95)),
                "pf_p05": float(np.percentile(pfs, 5)),
                "pf_p50": float(np.percentile(pfs, 50)),
                "pf_p95": float(np.percentile(pfs, 95)),
                "prob_total_r_positive": float((totals > 0).mean()),
                "prob_pf_above_1": float((pfs > 1.0).mean()),
            })

    return pd.DataFrame(rows)


def cost_stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf, side, g in group_slices(df):
        values = g[R_COL].to_numpy(dtype=float)
        for extra_cost in EXTRA_COST_R_VALUES:
            stressed = values - extra_cost
            s = summarize_r(stressed)
            s.update({
                "timeframe": tf,
                "side": side,
                "extra_cost_r_per_trade": extra_cost,
            })
            rows.append(s)

    cols = ["timeframe", "side", "extra_cost_r_per_trade"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tf, side, g in group_slices(df):
        if g.empty:
            continue

        start = g[TIME_COL].min().normalize()
        end = g[TIME_COL].max().normalize()

        cur = start
        while cur < end:
            win_end = cur + pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            w = g[(g[TIME_COL] >= cur) & (g[TIME_COL] < win_end)]
            s = summarize_r(w[R_COL].to_numpy(dtype=float))
            s.update({
                "timeframe": tf,
                "side": side,
                "window_start": cur,
                "window_end": win_end,
                "assets": w["symbol"].nunique() if not w.empty else 0,
            })
            rows.append(s)
            cur = cur + pd.Timedelta(days=ROLLING_STEP_DAYS)

    cols = ["timeframe", "side", "window_start", "window_end", "assets"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def period_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tf, side, g in group_slices(df):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        if g.empty:
            continue

        mid_idx = len(g) // 2
        slices = {
            "first_half_by_trade_count": g.iloc[:mid_idx],
            "second_half_by_trade_count": g.iloc[mid_idx:],
            "last_100_trades": g.tail(100),
            "last_200_trades": g.tail(200),
        }

        # Time split by median date
        median_time = g[TIME_COL].median()
        slices["first_half_by_time"] = g[g[TIME_COL] <= median_time]
        slices["second_half_by_time"] = g[g[TIME_COL] > median_time]

        for name, sub in slices.items():
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update({
                "timeframe": tf,
                "side": side,
                "split": name,
                "start": sub[TIME_COL].min() if not sub.empty else pd.NaT,
                "end": sub[TIME_COL].max() if not sub.empty else pd.NaT,
                "assets": sub["symbol"].nunique() if not sub.empty else 0,
            })
            rows.append(s)

    cols = ["timeframe", "side", "split", "start", "end", "assets"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def remove_best_assets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tf, side, g in group_slices(df):
        if g.empty:
            continue

        asset_perf = (
            g.groupby("symbol")[R_COL]
            .sum()
            .sort_values(ascending=False)
        )

        for n in [0] + REMOVE_TOP_ASSET_COUNTS:
            removed = asset_perf.head(n).index.tolist()
            sub = g[~g["symbol"].isin(removed)].copy()
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update({
                "timeframe": tf,
                "side": side,
                "removed_top_assets_n": n,
                "removed_assets": ",".join(removed),
                "assets_remaining": sub["symbol"].nunique(),
            })
            rows.append(s)

    cols = ["timeframe", "side", "removed_top_assets_n", "removed_assets", "assets_remaining"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def remove_best_months(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tf, side, g in group_slices(df):
        if g.empty:
            continue

        month_perf = (
            g.groupby("month")[R_COL]
            .sum()
            .sort_values(ascending=False)
        )

        for n in [0] + REMOVE_TOP_MONTH_COUNTS:
            removed = month_perf.head(n).index.tolist()
            sub = g[~g["month"].isin(removed)].copy()
            s = summarize_r(sub[R_COL].to_numpy(dtype=float))
            s.update({
                "timeframe": tf,
                "side": side,
                "removed_top_months_n": n,
                "removed_months": ",".join(removed),
                "months_remaining": sub["month"].nunique(),
            })
            rows.append(s)

    cols = ["timeframe", "side", "removed_top_months_n", "removed_months", "months_remaining"] + [
        "trades", "total_r", "avg_r", "median_r", "win_rate",
        "profit_factor", "max_dd_r", "best_trade_r", "worst_trade_r",
        "losing_streak", "std_r", "t_score",
    ]
    return pd.DataFrame(rows)[cols]


def asset_concentration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tf, side, g in group_slices(df):
        if g.empty:
            continue

        asset = (
            g.groupby("symbol")
            .agg(
                trades=(R_COL, "size"),
                total_r=(R_COL, "sum"),
                avg_r=(R_COL, "mean"),
                best_trade_r=(R_COL, "max"),
                worst_trade_r=(R_COL, "min"),
            )
            .reset_index()
            .sort_values("total_r", ascending=False)
        )

        total_pos = asset.loc[asset["total_r"] > 0, "total_r"].sum()
        total_all = asset["total_r"].sum()
        top1 = asset["total_r"].head(1).sum()
        top3 = asset["total_r"].head(3).sum()
        top5 = asset["total_r"].head(5).sum()
        top10 = asset["total_r"].head(10).sum()

        rows.append({
            "timeframe": tf,
            "side": side,
            "assets": int(asset["symbol"].nunique()),
            "positive_assets": int((asset["total_r"] > 0).sum()),
            "negative_assets": int((asset["total_r"] < 0).sum()),
            "total_r": float(total_all),
            "top1_asset": asset["symbol"].iloc[0] if len(asset) else "",
            "top1_r": float(top1),
            "top3_r": float(top3),
            "top5_r": float(top5),
            "top10_r": float(top10),
            "top1_pct_of_total": float(top1 / total_all) if total_all != 0 else np.nan,
            "top3_pct_of_total": float(top3 / total_all) if total_all != 0 else np.nan,
            "top5_pct_of_total": float(top5 / total_all) if total_all != 0 else np.nan,
            "top10_pct_of_total": float(top10 / total_all) if total_all != 0 else np.nan,
            "top1_pct_of_positive_r": float(top1 / total_pos) if total_pos > 0 else np.nan,
            "top3_pct_of_positive_r": float(top3 / total_pos) if total_pos > 0 else np.nan,
            "top5_pct_of_positive_r": float(top5 / total_pos) if total_pos > 0 else np.nan,
            "top10_pct_of_positive_r": float(top10 / total_pos) if total_pos > 0 else np.nan,
        })

        # Also export detailed asset ranking per group.
        safe_tf = str(tf).replace("/", "_")
        safe_side = str(side).replace("/", "_")
        asset.to_csv(OUT_DIR / f"phase_t4_asset_rank_{safe_tf}_{safe_side}.csv", index=False)

    return pd.DataFrame(rows)


def write_master_report(
    input_path: Path,
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    mc: pd.DataFrame,
    cost: pd.DataFrame,
    splits: pd.DataFrame,
    rem_assets: pd.DataFrame,
    rem_months: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "phase_t4_master_report.txt"

    def fmt_row(row: pd.Series) -> str:
        return (
            f"{row['timeframe']} {row['side']}: "
            f"trades={int(row['trades'])}, "
            f"totalR={row['total_r']:.2f}, "
            f"avgR={row['avg_r']:.3f}, "
            f"PF={row['profit_factor']:.2f}, "
            f"DD={row['max_dd_r']:.2f}, "
            f"win={row['win_rate']:.1%}"
        )

    lines = []
    lines.append("PHASE T4 — TREND ROBUSTNESS MASTER REPORT")
    lines.append("=" * 60)
    lines.append(f"Input: {input_path}")
    lines.append(f"Trades loaded: {len(df)}")
    lines.append(f"Date range: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")
    lines.append(f"Assets: {df['symbol'].nunique()}")
    lines.append("")

    lines.append("BASELINE SUMMARY")
    lines.append("-" * 60)
    for _, row in baseline.sort_values(["timeframe", "side"]).iterrows():
        lines.append(fmt_row(row))
    lines.append("")

    lines.append("KEY 6H DIAGNOSTICS")
    lines.append("-" * 60)
    b6 = baseline[(baseline["timeframe"] == "6h") & (baseline["side"] == "ALL")]
    if not b6.empty:
        lines.append(fmt_row(b6.iloc[0]))

    mc6 = mc[(mc["timeframe"] == "6h") & (mc["side"] == "ALL") & (mc["block_size"] == 10)]
    if not mc6.empty:
        r = mc6.iloc[0]
        lines.append(
            f"6H ALL MC block10: totalR p05/p50/p95 = "
            f"{r['total_r_p05']:.2f} / {r['total_r_p50']:.2f} / {r['total_r_p95']:.2f}; "
            f"prob positive={r['prob_total_r_positive']:.1%}; "
            f"PF p05={r['pf_p05']:.2f}"
        )

    c6 = cost[(cost["timeframe"] == "6h") & (cost["side"] == "ALL")]
    if not c6.empty:
        lines.append("6H ALL cost stress:")
        for _, r in c6.iterrows():
            lines.append(
                f"  extra_cost={r['extra_cost_r_per_trade']:.2f}R -> "
                f"totalR={r['total_r']:.2f}, PF={r['profit_factor']:.2f}, DD={r['max_dd_r']:.2f}"
            )

    ra6 = rem_assets[(rem_assets["timeframe"] == "6h") & (rem_assets["side"] == "ALL")]
    if not ra6.empty:
        lines.append("6H ALL remove-best-assets:")
        for _, r in ra6.iterrows():
            lines.append(
                f"  remove top {int(r['removed_top_assets_n'])}: "
                f"totalR={r['total_r']:.2f}, PF={r['profit_factor']:.2f}, "
                f"remaining_assets={int(r['assets_remaining'])}, removed={r['removed_assets']}"
            )

    rm6 = rem_months[(rem_months["timeframe"] == "6h") & (rem_months["side"] == "ALL")]
    if not rm6.empty:
        lines.append("6H ALL remove-best-months:")
        for _, r in rm6.iterrows():
            lines.append(
                f"  remove top {int(r['removed_top_months_n'])}: "
                f"totalR={r['total_r']:.2f}, PF={r['profit_factor']:.2f}, "
                f"remaining_months={int(r['months_remaining'])}, removed={r['removed_months']}"
            )

    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 60)
    lines.append("Good signs:")
    lines.append("- MC p05 totalR remains positive or near zero.")
    lines.append("- PF stays above 1.0 after moderate cost stress.")
    lines.append("- Removing top 1-3 assets does not destroy the system.")
    lines.append("- First/second half and last 100/200 trades are not completely contradictory.")
    lines.append("- Rolling windows show multiple profitable periods, not one isolated miracle.")
    lines.append("")
    lines.append("Bad signs:")
    lines.append("- Result depends only on one asset or one month.")
    lines.append("- Extra 0.05R/trade cost turns the edge strongly negative.")
    lines.append("- MC p05 is deeply negative and probability of positive result is low.")
    lines.append("- Recent last 100/200 trades are structurally worse than historical trades.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ensure_output_dir()

    try:
        input_path = find_input_file()
        raw = pd.read_csv(input_path)
        trades = normalize_trades(raw)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=" * 80)
    print("PHASE T4 — TREND ROBUSTNESS ENGINE")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Trades loaded: {len(trades)}")
    print(f"Assets: {trades['symbol'].nunique()}")
    print(f"Date range: {trades[TIME_COL].min()} -> {trades[TIME_COL].max()}")
    print(f"Output dir: {OUT_DIR}")
    print("")

    baseline = baseline_summary(trades)
    baseline.to_csv(OUT_DIR / "phase_t4_baseline_summary.csv", index=False)
    print("[OK] baseline summary")

    mc = montecarlo_summary(trades)
    mc.to_csv(OUT_DIR / "phase_t4_montecarlo_summary.csv", index=False)
    print("[OK] Monte Carlo summary")

    cost = cost_stress_summary(trades)
    cost.to_csv(OUT_DIR / "phase_t4_cost_stress_summary.csv", index=False)
    print("[OK] cost stress summary")

    rolling = rolling_windows(trades)
    rolling.to_csv(OUT_DIR / "phase_t4_rolling_windows.csv", index=False)
    print("[OK] rolling windows")

    splits = period_splits(trades)
    splits.to_csv(OUT_DIR / "phase_t4_period_splits.csv", index=False)
    print("[OK] period splits")

    rem_assets = remove_best_assets(trades)
    rem_assets.to_csv(OUT_DIR / "phase_t4_remove_best_assets.csv", index=False)
    print("[OK] remove best assets")

    rem_months = remove_best_months(trades)
    rem_months.to_csv(OUT_DIR / "phase_t4_remove_best_months.csv", index=False)
    print("[OK] remove best months")

    concentration = asset_concentration(trades)
    concentration.to_csv(OUT_DIR / "phase_t4_asset_concentration.csv", index=False)
    print("[OK] asset concentration")

    write_master_report(
        input_path=input_path,
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

    print("")
    print("DONE.")
    print("Created files:")
    for p in sorted(OUT_DIR.glob("phase_t4_*")):
        print(f" - {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
