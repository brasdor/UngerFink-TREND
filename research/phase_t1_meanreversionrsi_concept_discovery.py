#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- MeanReversionRSI Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology

Usage:
    python phase_t1_meanreversionrsi_concept_discovery.py --timeframe 1d
    python phase_t1_meanreversionrsi_concept_discovery.py --timeframe 4h
    python phase_t1_meanreversionrsi_concept_discovery.py --timeframe 6h
    python phase_t1_meanreversionrsi_concept_discovery.py  # all timeframes

Entry : RSI(N) < oversold_threshold  (positional, re-enter once prior closed)
Exit  : RSI > exit_rsi_level  OR  ATR stop  OR  time_exit_bars elapsed
Filter: none | ema200_price_above

MR Gate checks:
  - min_trades >= 100
  - win_rate   50-80%
  - avg_r      >= 0.10R
  - stability  >= 67% of neighbourhood profitable

Outputs -> data/research_meanreversionrsi_t1/
  phase_t1_stability_ranking.csv
  phase_t1_summary.txt
  phase_t1_grid_{tf}.csv  (one per timeframe)
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"


def p(*args, **kwargs):
    """Print with immediate flush."""
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


# =============================================================================
# CONFIG
# =============================================================================

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_meanreversionrsi_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "AAVE_USDT", "ADA_USDT",  "ALT_USDT",  "APT_USDT",  "ARB_USDT",
    "ARKM_USDT", "ASTER_USDT","ATOM_USDT", "AVAX_USDT", "BCH_USDT",
    "BNB_USDT",  "BTC_USDT",  "CHZ_USDT",  "DASH_USDT", "DOGE_USDT",
    "DOT_USDT",  "EIGEN_USDT","ENA_USDT",  "ETH_USDT",  "FET_USDT",
    "FIL_USDT",  "GRT_USDT",  "HBAR_USDT", "ICP_USDT",  "INJ_USDT",
    "JTO_USDT",  "LINK_USDT", "LPT_USDT",  "LTC_USDT",  "MORPHO_USDT",
    "NEAR_USDT", "NIL_USDT",  "ONDO_USDT", "ORDI_USDT", "PENDLE_USDT",
    "PENGU_USDT","PEPE_USDT", "RENDER_USDT","SAGA_USDT","SEI_USDT",
    "SOL_USDT",  "SPK_USDT",  "SUI_USDT",  "TAO_USDT",  "TIA_USDT",
    "TON_USDT",  "TRX_USDT",  "UNI_USDT",  "WLD_USDT",  "XRP_USDT",
    "ZEC_USDT",  "ZEN_USDT",
]

MAX_BARS = {"1d": 1500, "4h": 6000, "6h": 4000, "8h": 3000, "2h": 17520}
MIN_BARS = 200

# rsi_n=[2,3] dropped after 1D review — near-worthless on crypto MR
RSI_N     = [5, 7, 10, 14]
OVERSOLD  = [10, 15, 20, 25, 30]
EXIT_RSI  = [50, 55, 60, 65]
ATR_MULTS = [2.0, 3.0, 4.0]
TIME_EXITS= [5, 10, 15, 20]
FILTERS   = ["none", "ema200_price_above"]

MR_GATES = {
    "min_trades":    100,
    "win_rate_min":  0.50,
    "win_rate_max":  0.80,
    "avg_r_min":     0.10,
    "stability_min": 0.67,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ohlcv(symbol: str, tf: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return None
        limit = MAX_BARS.get(tf, 1500)
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return None


# =============================================================================
# VECTORIZED INDICATORS  (compute once per symbol, cache all combos)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI for all N values, ATR(14), EMA(200) — vectorized."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # EMA(200)
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI for each N — vectorized Wilder EMA
    for n in RSI_N:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta.clip(upper=0))
        avg_g = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        avg_l = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        df[f"rsi{n}"] = 100 - (100 / (1 + rs))

    return df


# =============================================================================
# VECTORIZED BACKTEST  (all param combos for one symbol, one timeframe)
# =============================================================================

def backtest_symbol_all_combos(
    df: pd.DataFrame,
    filter_mode: str,
    param_combos: list[tuple],
) -> dict[tuple, list[dict]]:
    """
    Run all (rsi_n, oversold, exit_rsi, atr_mult, time_exit) combos
    for a single symbol and filter_mode.
    Returns {combo_key: [trades]} dict.
    """
    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values

    if filter_mode == "ema200_price_above":
        filt = close > ema200
    else:
        filt = np.ones(len(df), dtype=bool)

    results: dict[tuple, list[dict]] = {}

    for (rsi_n, oversold, exit_rsi, atr_mult, time_exit) in param_combos:
        rsi = df[f"rsi{rsi_n}"].values
        trades: list[dict] = []
        in_pos  = False
        e_price = 0.0
        stop    = 0.0
        e_bar   = 0

        for i in range(len(df)):
            if np.isnan(rsi[i]) or np.isnan(atr[i]):
                continue

            if not in_pos:
                if rsi[i] < oversold and filt[i]:
                    in_pos  = True
                    e_price = close[i]
                    e_bar   = i
                    stop    = e_price - atr_mult * atr[i]
            else:
                bars_held  = i - e_bar
                exit_price = None
                reason     = None

                if low_v[i] <= stop:
                    exit_price = stop
                    reason     = "atr_stop"
                elif rsi[i] >= exit_rsi:
                    exit_price = close[i]
                    reason     = "rsi_exit"
                elif bars_held >= time_exit:
                    exit_price = close[i]
                    reason     = "time_exit"

                if reason is not None:
                    risk = e_price - stop
                    if risk > 1e-9:
                        trades.append({
                            "r_multiple": (exit_price - e_price) / risk,
                            "bars_held":  bars_held,
                            "reason":     reason,
                        })
                    in_pos = False

        key = (rsi_n, oversold, exit_rsi, atr_mult, time_exit)
        results[key] = trades

    return results


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd_r": 0.0, "avg_bars": 0.0, "total_r": 0.0}
    rs      = [t["r_multiple"] for t in trades]
    wins    = [r for r in rs if r > 0]
    losses  = [abs(r) for r in rs if r < 0]
    total_w = sum(wins)
    total_l = sum(losses)
    pf      = total_w / total_l if total_l > 0 else (99.0 if total_w > 0 else 0.0)
    cum     = np.cumsum(rs)
    peak    = np.maximum.accumulate(cum)
    dd      = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
    return {
        "n":        len(rs),
        "win_rate": len(wins) / len(rs),
        "avg_r":    float(np.mean(rs)),
        "pf":       pf,
        "max_dd_r": dd,
        "avg_bars": float(np.mean([t["bars_held"] for t in trades])),
        "total_r":  float(np.sum(rs)),
    }


# =============================================================================
# GRID SCAN  (one timeframe)
# =============================================================================

def run_grid_for_tf(tf: str) -> pd.DataFrame:
    param_combos = list(itertools.product(
        RSI_N, OVERSOLD, EXIT_RSI, ATR_MULTS, TIME_EXITS
    ))
    n_combos = len(param_combos)
    n_filters = len(FILTERS)
    p(f"  Param combos per filter: {n_combos}  x  {n_filters} filters  =  {n_combos*n_filters} total")

    # Load all symbol data for this TF
    loaded: list[tuple[str, pd.DataFrame]] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym, tf)
        if df is not None:
            df = add_indicators(df)
            loaded.append((sym, df))
    p(f"  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}", flush=True)

    if not loaded:
        p(f"  ERROR: no data for timeframe {tf}", flush=True)
        return pd.DataFrame()

    # Accumulate trades per (filter, combo_key)
    combo_trades: dict[tuple, list[dict]] = {}
    for filt in FILTERS:
        for key in param_combos:
            combo_trades[(filt, *key)] = []

    for sym_idx, (sym, df) in enumerate(loaded, 1):
        if sym_idx % 10 == 0 or sym_idx == 1 or sym_idx == len(loaded):
            p(f"  Processing symbol {sym_idx}/{len(loaded)} {sym}...", flush=True)

        for filt in FILTERS:
            sym_results = backtest_symbol_all_combos(df, filt, param_combos)
            for key, trades in sym_results.items():
                combo_trades[(filt, *key)].extend(trades)

    # Build results DataFrame
    rows = []
    for filt in FILTERS:
        for (rsi_n, oversold, exit_rsi, atr_mult, time_exit) in param_combos:
            full_key = (filt, rsi_n, oversold, exit_rsi, atr_mult, time_exit)
            trades   = combo_trades[full_key]
            m        = calc_metrics(trades)
            rows.append({
                "timeframe":      tf,
                "filter_mode":    filt,
                "rsi_n":          rsi_n,
                "oversold":       oversold,
                "exit_rsi":       exit_rsi,
                "atr_mult":       atr_mult,
                "time_exit_bars": time_exit,
                "num_trades":     m["n"],
                "win_rate":       round(m["win_rate"], 4),
                "avg_r":          round(m["avg_r"], 4),
                "profit_factor":  round(m["pf"], 4),
                "max_dd_r":       round(m["max_dd_r"], 2),
                "avg_bars_held":  round(m["avg_bars"], 1),
                "total_r":        round(m["total_r"], 2),
            })

    return pd.DataFrame(rows)


# =============================================================================
# STABILITY ZONE
# =============================================================================

def stability_score(grid: pd.DataFrame, tf: str, filt: str,
                    rsi_n: int, oversold: int,
                    exit_rsi: int, atr_mult: float, time_exit: int) -> float:
    rsi_ns    = sorted(RSI_N)
    oversolds = sorted(OVERSOLD)
    try:
        ri = rsi_ns.index(rsi_n)
        oi = oversolds.index(oversold)
    except ValueError:
        return 0.0

    n_nbrs = [rsi_ns[j]    for j in range(max(0, ri-1), min(len(rsi_ns), ri+2))]
    o_nbrs = [oversolds[j] for j in range(max(0, oi-1), min(len(oversolds), oi+2))]

    zone = grid[
        (grid["timeframe"]      == tf)      &
        (grid["filter_mode"]    == filt)    &
        (grid["rsi_n"].isin(n_nbrs))        &
        (grid["oversold"].isin(o_nbrs))     &
        (grid["exit_rsi"]       == exit_rsi)&
        (grid["atr_mult"]       == atr_mult)&
        (grid["time_exit_bars"] == time_exit)
    ]

    if zone.empty:
        return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


# =============================================================================
# REPORT
# =============================================================================

def print_and_save_report(grid: pd.DataFrame, tf_label: str) -> bool:
    """Print T1 results, save files. Returns True if PASS."""
    p(f"\n--- Results for timeframe {tf_label} ---", flush=True)

    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    p(f"  Combos with >= {MR_GATES['min_trades']} trades : {len(viable)}", flush=True)

    if viable.empty:
        top_by_trades = grid.nlargest(10, "num_trades")[
            ["timeframe","filter_mode","rsi_n","oversold","exit_rsi",
             "atr_mult","time_exit_bars","num_trades","win_rate","avg_r"]
        ]
        p(f"  FAIL: no combos meet min_trades={MR_GATES['min_trades']}", flush=True)
        p("\n  Top 10 by trade count (debug):", flush=True)
        p(top_by_trades.to_string(index=False), flush=True)
        return False

    viable = viable[
        (viable["win_rate"] >= MR_GATES["win_rate_min"]) &
        (viable["win_rate"] <= MR_GATES["win_rate_max"])
    ]
    p(f"  After win_rate gate : {len(viable)}", flush=True)

    viable = viable[viable["avg_r"] >= MR_GATES["avg_r_min"]]
    p(f"  After avg_r gate    : {len(viable)}", flush=True)

    if viable.empty:
        cands = grid[grid["num_trades"] >= MR_GATES["min_trades"]].nlargest(10, "avg_r")[
            ["timeframe","filter_mode","rsi_n","oversold","exit_rsi",
             "atr_mult","time_exit_bars","num_trades","win_rate","avg_r","total_r"]
        ]
        p("  FAIL: no combos pass win_rate + avg_r gates.", flush=True)
        p("\n  Best avg_r (debug):", flush=True)
        p(cands.to_string(index=False), flush=True)
        return False

    p("  Computing stability zones...", flush=True)
    viable["stability"] = viable.apply(
        lambda r: stability_score(
            grid, r["timeframe"], r["filter_mode"],
            int(r["rsi_n"]), int(r["oversold"]),
            int(r["exit_rsi"]), float(r["atr_mult"]), int(r["time_exit_bars"])
        ), axis=1
    )

    stable = viable[viable["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"  After stability >= {MR_GATES['stability_min']} : {len(stable)}", flush=True)

    if stable.empty:
        p("  NOTE: no combos pass stability gate -- showing best by stability:", flush=True)
        stable = viable.nlargest(30, "stability")
        gate_pass = False
    else:
        gate_pass = True

    stable = stable.sort_values(["stability","avg_r"], ascending=[False,False]).reset_index(drop=True)

    # EMA200 filter impact
    no_filt  = viable[viable["filter_mode"] == "none"]
    ema_filt = viable[viable["filter_mode"] == "ema200_price_above"]
    p(f"\n  EMA200 filter impact (viable combos, avg of avg_r):", flush=True)
    p(f"    no filter    : {no_filt['avg_r'].mean():.4f}R  ({len(no_filt)} combos)", flush=True)
    p(f"    ema200 above : {ema_filt['avg_r'].mean():.4f}R  ({len(ema_filt)} combos)", flush=True)

    top = stable.head(20)
    p(f"\n  Top 10 stable combos:", flush=True)
    p(top.head(10)[[
        "timeframe","filter_mode","rsi_n","oversold","exit_rsi",
        "atr_mult","time_exit_bars","num_trades","win_rate","avg_r","stability"
    ]].to_string(index=False), flush=True)

    # Heatmap pivot
    best_tf   = stable.iloc[0]["timeframe"]
    best_filt = stable.iloc[0]["filter_mode"]
    pivot = grid[
        (grid["timeframe"] == best_tf) & (grid["filter_mode"] == best_filt)
    ].groupby(["rsi_n","oversold"])["avg_r"].mean().unstack("oversold")
    p(f"\n  avg_r heatmap (rsi_n vs oversold) | TF={best_tf} filter={best_filt}:", flush=True)
    p(pivot.round(3).to_string(), flush=True)

    # Save files
    stable.to_csv(OUT_DIR / "phase_t1_stability_ranking.csv", index=False)

    with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("Phase T1 -- MeanReversionRSI Concept Discovery\n")
        f.write(f"Timeframe: {tf_label}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Grid size          : {len(grid)}\n")
        f.write(f"Viable (trades)    : {len(grid[grid['num_trades']>=MR_GATES['min_trades']])}\n")
        f.write(f"Viable (all gates) : {len(viable)}\n")
        f.write(f"Stable combos      : {len(stable)}\n\n")
        f.write("EMA200 filter impact:\n")
        f.write(f"  no filter     : {no_filt['avg_r'].mean():.4f}R  ({len(no_filt)} combos)\n")
        f.write(f"  ema200 above  : {ema_filt['avg_r'].mean():.4f}R  ({len(ema_filt)} combos)\n\n")
        f.write("Top 20 stable combos (stability desc, avg_r desc):\n")
        f.write(top[[
            "timeframe","filter_mode","rsi_n","oversold","exit_rsi",
            "atr_mult","time_exit_bars","num_trades","win_rate","avg_r",
            "profit_factor","max_dd_r","stability","total_r"
        ]].to_string(index=False))
        f.write("\n\navg_r heatmap (rsi_n vs oversold):\n")
        f.write(pivot.round(3).to_string())
        f.write("\n\nGate summary:\n")
        f.write(f"  min_trades >= {MR_GATES['min_trades']}   : PASS\n")
        f.write(f"  win_rate {MR_GATES['win_rate_min']}-{MR_GATES['win_rate_max']} : PASS\n")
        f.write(f"  avg_r >= {MR_GATES['avg_r_min']}         : PASS\n")
        f.write(f"  stability >= {MR_GATES['stability_min']} : {'PASS' if gate_pass else 'MARGINAL'}\n")
        f.write(f"\nT1 GATE: {'PASS' if gate_pass else 'MARGINAL -- review needed'}\n")

    p(f"\n[OK] phase_t1_stability_ranking.csv  ({len(stable)} rows)", flush=True)
    p(f"[OK] phase_t1_summary.txt", flush=True)
    return gate_pass


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase T1 MeanReversionRSI")
    parser.add_argument("--timeframe", default=None,
                        help="Run single timeframe: 1d, 4h, 6h, or 8h. Default: all. (No 2h data in cache — use 6h instead.)")
    args = parser.parse_args()

    if args.timeframe:
        tf_arg = args.timeframe.lower()
        if tf_arg not in MAX_BARS:
            p(f"ERROR: --timeframe must be one of {list(MAX_BARS.keys())}", flush=True)
            sys.exit(1)
        timeframes = [tf_arg]
    else:
        timeframes = list(MAX_BARS.keys())

    p("=" * 65, flush=True)
    p("  Phase T1 -- MeanReversionRSI Concept Discovery", flush=True)
    p(f"  Timeframes : {timeframes}", flush=True)
    p(f"  Symbols    : {len(SYMBOLS)}", flush=True)
    p(f"  RSI_N      : {RSI_N}", flush=True)
    p(f"  Oversold   : {OVERSOLD}", flush=True)
    p(f"  Exit RSI   : {EXIT_RSI}", flush=True)
    p(f"  ATR mults  : {ATR_MULTS}", flush=True)
    p(f"  Time exits : {TIME_EXITS}", flush=True)
    p(f"  Filters    : {FILTERS}", flush=True)
    p("=" * 65, flush=True)

    available = [(sym, tf) for sym in SYMBOLS for tf in timeframes
                 if (RAW_DIR / f"{sym}_{tf}.csv").exists()]
    p(f"Data files found: {len(available)}", flush=True)
    if not available:
        p(f"ERROR: no data files in {RAW_DIR}", flush=True)
        sys.exit(1)

    all_grids: list[pd.DataFrame] = []

    for tf in timeframes:
        p(f"\n{'='*50}", flush=True)
        p(f"  Running timeframe: {tf}", flush=True)
        p(f"{'='*50}", flush=True)
        grid_tf = run_grid_for_tf(tf)
        if grid_tf.empty:
            continue
        grid_tf.to_csv(OUT_DIR / f"phase_t1_grid_{tf}.csv", index=False)
        p(f"  Grid saved: phase_t1_grid_{tf}.csv  ({len(grid_tf)} rows)", flush=True)
        all_grids.append(grid_tf)

    if not all_grids:
        p("ERROR: no results produced.", flush=True)
        sys.exit(1)

    combined = pd.concat(all_grids, ignore_index=True)
    gate_pass = print_and_save_report(combined, "+".join(timeframes))

    if gate_pass:
        p("\nT1 GATE: PASS", flush=True)
        sys.exit(0)
    else:
        p("\nT1 GATE: FAIL", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
