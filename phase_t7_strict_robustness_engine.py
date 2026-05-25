#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T7 — STRICT ROBUSTNESS ENGINE
===================================

Purpose
-------
Stress-test the best T6 capital variants.

This phase intentionally tries to BREAK the system using:
- extra execution cost
- remove best assets
- remove best months
- long-only / short-only splits
- recent-trade degradation
- rolling windows

The goal is NOT optimisation.
The goal is to verify whether the edge survives hostile conditions.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/research_trend_t6/phase_t6_trade_log.csv"
)

OUTPUT_DIR = Path(
    "data/research_trend_t7"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_VARIANTS = [
    "T5_6H_ALL_max5",
    "T5_6H_ALL_max3",
    "T5_6H_SHORT_max5",
]

EXTRA_COST_LEVELS = [0.00, 0.05, 0.10, 0.15]
REMOVE_TOP_ASSETS = [1, 3, 5]
REMOVE_TOP_MONTHS = [1, 2]
ROLLING_WINDOW = 30

# ============================================================
# HELPERS
# ============================================================


def profit_factor(values):
    arr = np.asarray(values, dtype=float)

    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()

    if losses <= 1e-12:
        return np.inf if gains > 0 else 0.0

    return gains / losses



def max_drawdown(values):
    arr = np.asarray(values, dtype=float)

    if len(arr) == 0:
        return 0.0

    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak

    return dd.min()



def summarize(df, label):
    if len(df) == 0:
        return {
            "label": label,
            "trades": 0,
            "total_r": 0,
            "avg_r": 0,
            "pf": 0,
            "dd_r": 0,
            "win_rate": 0,
        }

    r = df["net_r_after_extra_cost"]

    return {
        "label": label,
        "trades": len(df),
        "total_r": r.sum(),
        "avg_r": r.mean(),
        "pf": profit_factor(r),
        "dd_r": max_drawdown(r),
        "win_rate": (r > 0).mean() * 100,
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("PHASE T7 — STRICT ROBUSTNESS ENGINE")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

trades = pd.read_csv(INPUT_FILE)

trades["exit_time"] = pd.to_datetime(
    trades["exit_time"],
    utc=True,
    errors="coerce",
)

trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")

print(f"Trades loaded: {len(trades)}")
print(f"Variants: {trades['variant'].nunique()}")

# ============================================================
# MAIN
# ============================================================

master_rows = []
rolling_rows = []

for variant in BEST_VARIANTS:

    vdf = trades[
        trades["variant"] == variant
    ].copy()

    if len(vdf) == 0:
        continue

    print(f"\n[RUN] {variant}")

    baseline = summarize(vdf, f"{variant}_BASELINE")
    master_rows.append(baseline)

    # --------------------------------------------------------
    # COST STRESS
    # --------------------------------------------------------

    for cost in EXTRA_COST_LEVELS:

        temp = vdf.copy()
        temp["net_r_after_extra_cost"] = (
            temp["net_r_after_extra_cost"] - cost
        )

        row = summarize(
            temp,
            f"{variant}_COST_{cost:.2f}R"
        )

        row["extra_cost_r"] = cost

        master_rows.append(row)

    # --------------------------------------------------------
    # REMOVE BEST ASSETS
    # --------------------------------------------------------

    asset_perf = (
        vdf.groupby("symbol")
        ["net_r_after_extra_cost"]
        .sum()
        .sort_values(ascending=False)
    )

    for n in REMOVE_TOP_ASSETS:

        remove_assets = asset_perf.head(n).index.tolist()

        temp = vdf[
            ~vdf["symbol"].isin(remove_assets)
        ].copy()

        row = summarize(
            temp,
            f"{variant}_REMOVE_TOP_{n}_ASSETS"
        )

        row["removed_assets"] = ",".join(remove_assets)

        master_rows.append(row)

    # --------------------------------------------------------
    # REMOVE BEST MONTHS
    # --------------------------------------------------------

    month_perf = (
        vdf.groupby("month")
        ["net_r_after_extra_cost"]
        .sum()
        .sort_values(ascending=False)
    )

    for n in REMOVE_TOP_MONTHS:

        remove_months = month_perf.head(n).index.tolist()

        temp = vdf[
            ~vdf["month"].isin(remove_months)
        ].copy()

        row = summarize(
            temp,
            f"{variant}_REMOVE_TOP_{n}_MONTHS"
        )

        row["removed_months"] = ",".join(remove_months)

        master_rows.append(row)

    # --------------------------------------------------------
    # LONG / SHORT SPLIT
    # --------------------------------------------------------

    for side in ["LONG", "SHORT"]:

        temp = vdf[
            vdf["side"].str.upper() == side
        ].copy()

        row = summarize(
            temp,
            f"{variant}_{side}_ONLY"
        )

        master_rows.append(row)

    # --------------------------------------------------------
    # RECENT TRADE DEGRADATION
    # --------------------------------------------------------

    for last_n in [20, 30, 50]:

        temp = vdf.tail(last_n).copy()

        row = summarize(
            temp,
            f"{variant}_LAST_{last_n}"
        )

        master_rows.append(row)

    # --------------------------------------------------------
    # ROLLING WINDOWS
    # --------------------------------------------------------

    r = vdf["net_r_after_extra_cost"].values

    if len(r) >= ROLLING_WINDOW:

        for i in range(len(r) - ROLLING_WINDOW + 1):

            chunk = r[i:i + ROLLING_WINDOW]

            rolling_rows.append({
                "variant": variant,
                "start_trade": i,
                "end_trade": i + ROLLING_WINDOW,
                "window_total_r": chunk.sum(),
                "window_avg_r": chunk.mean(),
                "window_pf": profit_factor(chunk),
                "window_dd_r": max_drawdown(chunk),
            })

# ============================================================
# SAVE
# ============================================================

master_df = pd.DataFrame(master_rows)
rolling_df = pd.DataFrame(rolling_rows)

master_path = OUTPUT_DIR / "phase_t7_master_summary.csv"
rolling_path = OUTPUT_DIR / "phase_t7_rolling_windows.csv"
report_path = OUTPUT_DIR / "phase_t7_master_report.txt"

master_df.to_csv(master_path, index=False)
rolling_df.to_csv(rolling_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("PHASE T7 — STRICT ROBUSTNESS REPORT\n")
    f.write("=" * 70 + "\n\n")

    for _, row in master_df.iterrows():

        f.write(
            f"{row['label']}: "
            f"trades={row['trades']}, "
            f"totalR={row['total_r']:.2f}, "
            f"avgR={row['avg_r']:.3f}, "
            f"PF={row['pf']:.2f}, "
            f"DD={row['dd_r']:.2f}, "
            f"win={row['win_rate']:.1f}%\n"
        )

print("\n[OK] phase_t7_master_summary.csv")
print("[OK] phase_t7_rolling_windows.csv")
print("[OK] phase_t7_master_report.txt")

print("\nDone.")
