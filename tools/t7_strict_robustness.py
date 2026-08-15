#!/usr/bin/env python3
"""
T7 strict robustness engine, generic across candidates -- entry AND exit both
frozen (T3/T4/T5/T6 closed). Methodology matched to the existing repo
convention (research/phase_t7_strict_robustness_engine.py), which stress-tests
"the best T6 capital variant[s]" with: incremental cost stress, remove-top-N-
assets, remove-top-N-months, long/short-only splits, recent-trade degradation,
and rolling-window drawdown/degradation analysis. Goal is not optimization --
it is to verify the edge survives hostile conditions, applied here to the
LOCKED T6 variant for each candidate (see run_t7_all.py for what "locked"
means for 11u/12/13u).

BASELINE_COST_R = 0.25R is reused as the pass/fail floor throughout, matching
T1-T4's convention -- it represents the futures round-trip cost assumption
that avg_r must clear to still be considered a real, tradeable edge net of
costs (T6 slippage is a real deduction already baked into the input net_r
here; this floor is the SEPARATE conceptual cost-floor threshold, unchanged
from T1-T4 for methodological continuity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_COST_R = 0.25
EXTRA_COST_LEVELS = (0.00, 0.05, 0.10, 0.15)
REMOVE_TOP_ASSETS = (1, 3, 5)
REMOVE_TOP_MONTHS = (1, 2)
RECENT_N = (20, 30, 50)
ROLLING_WINDOW = 30


def profit_factor(r: np.ndarray) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 1e-12:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def summarize(r: np.ndarray, label: str) -> dict:
    if len(r) == 0:
        return dict(label=label, n=0, avg_r=np.nan, pf=np.nan, dd_r=0.0,
                    win_rate=np.nan, clears_floor=False)
    eq = np.cumsum(r)
    dd_r = float((eq - np.maximum.accumulate(eq)).min())
    avg_r = float(r.mean())
    return dict(label=label, n=len(r), avg_r=avg_r, pf=profit_factor(r), dd_r=dd_r,
                win_rate=float((r > 0).mean()), clears_floor=bool(avg_r > BASELINE_COST_R))


def cost_stress(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for extra in EXTRA_COST_LEVELS:
        r = trades["net_r"].to_numpy(dtype=float) - extra
        row = summarize(r, f"COST_+{extra:.2f}R")
        row["extra_cost_r"] = extra
        rows.append(row)
    return rows


def remove_top_assets(trades: pd.DataFrame) -> list[dict]:
    asset_perf = trades.groupby("symbol")["net_r"].sum().sort_values(ascending=False)
    rows = []
    for n in REMOVE_TOP_ASSETS:
        removed = asset_perf.head(n).index.tolist()
        sub = trades[~trades["symbol"].isin(removed)]
        row = summarize(sub["net_r"].to_numpy(dtype=float), f"REMOVE_TOP_{n}_ASSETS")
        row["removed"] = ",".join(removed)
        rows.append(row)
    return rows


def remove_top_months(trades: pd.DataFrame) -> list[dict]:
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_date"]).dt.to_period("M")
    month_perf = t.groupby("month")["net_r"].sum().sort_values(ascending=False)
    rows = []
    for n in REMOVE_TOP_MONTHS:
        removed = month_perf.head(n).index.tolist()
        sub = t[~t["month"].isin(removed)]
        row = summarize(sub["net_r"].to_numpy(dtype=float), f"REMOVE_TOP_{n}_MONTHS")
        row["removed"] = ",".join(str(m) for m in removed)
        rows.append(row)
    return rows


def long_short_split(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for side in ("long", "short"):
        sub = trades[trades["side"] == side]
        rows.append(summarize(sub["net_r"].to_numpy(dtype=float), f"{side.upper()}_ONLY"))
    return rows


def recent_degradation(trades: pd.DataFrame) -> list[dict]:
    t = trades.sort_values("exit_date")
    rows = []
    for n in RECENT_N:
        sub = t.tail(n)
        rows.append(summarize(sub["net_r"].to_numpy(dtype=float), f"LAST_{n}"))
    return rows


def rolling_windows(trades: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    t = trades.sort_values("exit_date")
    r = t["net_r"].to_numpy(dtype=float)
    if len(r) < window:
        return pd.DataFrame()
    rows = []
    for i in range(len(r) - window + 1):
        chunk = r[i:i + window]
        rows.append(dict(start=i, end=i + window, window_avg_r=chunk.mean(),
                          window_pf=profit_factor(chunk), window_dd_r=float(
                              (np.cumsum(chunk) - np.maximum.accumulate(np.cumsum(chunk))).min())))
    return pd.DataFrame(rows)


def run_t7_suite(trades: pd.DataFrame) -> dict:
    baseline = summarize(trades["net_r"].to_numpy(dtype=float), "BASELINE")
    cost = cost_stress(trades)
    rm_assets = remove_top_assets(trades)
    rm_months = remove_top_months(trades)
    sides = long_short_split(trades)
    recent = recent_degradation(trades)
    rolling = rolling_windows(trades)
    rolling_neg_frac = float((rolling["window_avg_r"] <= 0).mean()) if len(rolling) else np.nan
    rolling_below_floor_frac = float((rolling["window_avg_r"] <= BASELINE_COST_R).mean()) if len(rolling) else np.nan
    return dict(baseline=baseline, cost=cost, rm_assets=rm_assets, rm_months=rm_months,
                sides=sides, recent=recent, rolling=rolling,
                rolling_neg_frac=rolling_neg_frac, rolling_below_floor_frac=rolling_below_floor_frac)
