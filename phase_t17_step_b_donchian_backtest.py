#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Step B — Donchian Universe Backtest + Symbol Filter

Frozen Donchian config (validated T1→T18):
  Entry:  close > Donchian upper N=20 (highest high of prior 20 bars)
  Filter: close > EMA(200)
  Stop:   entry - ATR(14) x 2.0
  Exit A: close < Donchian lower N=10 (lowest low of prior 10 bars) [midline]
  Exit B: Chandelier — activates at +4R, trails at (highest_high - ATR(14)x3)
          Exit B overrides Exit A once chandelier is active.
  Side:   LONG only

Additional exclusions beyond Step A:
  LUNA/USDT  — Terra collapse (price→0, catastrophic distortion)
  FTT/USDT   — FTX collapse
  EUR/USDT   — forex, not crypto
  PAXG/USDT  — gold-backed stablecoin

Symbol filter condition (remove if ALL THREE are true):
  avg_r < 0 AND total_r < 0 AND trades < 5

Benchmark (frozen pipeline, N=20 / ~64 symbols):
  avg_r=+0.179R  PF=1.27  trades per symbol ≈ 3

Output:
  data/universe/filtered_symbols.csv   — per-symbol stats + included flag
  data/universe/step_b_all_trades.csv  — every individual trade
  data/universe/step_b_summary.txt     — comparison report
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT      = Path(__file__).parent
OHLCV_DIR = ROOT / "data" / "universe" / "ohlcv_1d"
UNIV_CSV  = ROOT / "data" / "universe" / "expanded_symbols.csv"
OUT_DIR   = ROOT / "data" / "universe"

STEP_B_EXCLUDE = {"LUNA/USDT", "FTT/USDT", "EUR/USDT", "PAXG/USDT"}

# Frozen Donchian config
DONCHIAN_N  = 20
EXIT_N      = 10      # = DONCHIAN_N // 2
ATR_N       = 14
STOP_MULT   = 2.0
CHAN_ACT_R  = 4.0     # chandelier activates after +4R
CHAN_TRAIL  = 3.0     # chandelier trails at HH - ATR*3

# Baseline (frozen pipeline, max3 portfolio)
BASELINE = dict(avg_r=0.179, pf=1.27, total_r=None)

# Filter: remove symbol if ALL THREE true
FILTER_AVG_R   = 0.0
FILTER_TOTAL_R = 0.0
FILTER_TRADES  = 5


# =============================================================================
# INDICATORS
# =============================================================================

def compute_ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(close)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1 : n + 1]))
        for i in range(n + 1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i - 1]):
                atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


# =============================================================================
# DATA
# =============================================================================

def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    return df[["time", "open", "high", "low", "close"]].dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)


# =============================================================================
# BACKTEST
# =============================================================================

@dataclass
class Trade:
    symbol:       str
    entry_time:   object
    exit_time:    object
    entry_price:  float
    exit_price:   float
    initial_stop: float
    initial_risk: float
    exit_reason:  str
    bars_held:    int
    net_r:        float
    mae_r:        float
    mfe_r:        float


def run_donchian_backtest(df: pd.DataFrame, symbol: str) -> List[Trade]:
    df = df.sort_values("time").reset_index(drop=True)
    n  = len(df)
    if n < max(DONCHIAN_N, ATR_N, 200) + 20:
        return []

    cl  = df["close"].to_numpy(dtype=float)
    hi  = df["high"].to_numpy(dtype=float)
    lo  = df["low"].to_numpy(dtype=float)
    ts  = df["time"].to_numpy()

    ema200 = compute_ema(cl, 200)
    atr14  = compute_atr(hi, lo, cl, ATR_N)

    # Donchian bands (prior-bar, no lookahead)
    # Upper: highest high of prior DONCHIAN_N bars
    # Lower: lowest low of prior EXIT_N bars
    don_upper = pd.Series(hi).shift(1).rolling(DONCHIAN_N).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(EXIT_N).min().to_numpy()

    trades: List[Trade] = []
    pos: Optional[dict] = None

    for i in range(1, n):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        if not (np.isfinite(ema200[i - 1]) and np.isfinite(atr14[i - 1])):
            continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])):
            continue

        # ---------- MANAGE OPEN POSITION ----------
        if pos is not None:
            # Track MFE/MAE
            raw_gain = hi[i] - pos["entry"]
            raw_loss = lo[i] - pos["entry"]
            pos["mfe_r"] = max(pos["mfe_r"], raw_gain / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], raw_loss / pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            exit_px     = None
            exit_reason = None

            # Exit 1: initial stop
            if lo[i] <= pos["stop"]:
                exit_px     = pos["stop"]
                exit_reason = "initial_stop"
            # Exit 2: chandelier (overrides midline when active)
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px     = pos["chan_stop"]
                exit_reason = "chandelier_stop"
            # Exit 3: midline — only if chandelier not active
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px     = cl[i]
                exit_reason = "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(Trade(
                    symbol       = symbol,
                    entry_time   = pos["entry_time"],
                    exit_time    = ts[i],
                    entry_price  = pos["entry"],
                    exit_price   = exit_px,
                    initial_stop = pos["stop"],
                    initial_risk = pos["risk"],
                    exit_reason  = exit_reason,
                    bars_held    = pos["bars"],
                    net_r        = net_r,
                    mae_r        = min(pos["mae_r"], net_r),
                    mfe_r        = pos["mfe_r"],
                ))
                pos = None
            else:
                # Update chandelier after exit checks
                if pos["mfe_r"] >= CHAN_ACT_R:
                    pos["chan_active"] = True
                if pos["chan_active"]:
                    new_cs = pos["hh"] - atr14[i] * CHAN_TRAIL
                    pos["chan_stop"] = max(pos["chan_stop"], new_cs)

        # ---------- ENTRY ----------
        if pos is None:
            if cl[i] > ema200[i - 1] and cl[i] > don_upper[i]:
                risk = atr14[i - 1] * STOP_MULT
                if risk <= 0:
                    continue
                stop_px = cl[i] - risk
                pos = {
                    "entry":      cl[i],
                    "stop":       stop_px,
                    "risk":       risk,
                    "entry_time": ts[i],
                    "hh":         hi[i],
                    "chan_active": False,
                    "chan_stop":  stop_px,  # initialized below trigger
                    "mfe_r":      0.0,
                    "mae_r":      0.0,
                    "bars":       1,
                }

    # Force-close any open position at end
    if pos is not None:
        exit_px = cl[-1]
        net_r   = (exit_px - pos["entry"]) / pos["risk"]
        trades.append(Trade(
            symbol       = symbol,
            entry_time   = pos["entry_time"],
            exit_time    = ts[-1],
            entry_price  = pos["entry"],
            exit_price   = exit_px,
            initial_stop = pos["stop"],
            initial_risk = pos["risk"],
            exit_reason  = "end_of_data",
            bars_held    = pos["bars"],
            net_r        = net_r,
            mae_r        = min(pos["mae_r"], net_r),
            mfe_r        = pos["mfe_r"],
        ))

    return trades


# =============================================================================
# AGGREGATION
# =============================================================================

def symbol_stats(trades: List[Trade], symbol: str) -> dict:
    if not trades:
        return dict(
            symbol=symbol, trades=0, total_r=0.0, avg_r=0.0,
            win_rate=0.0, profit_factor=0.0, max_dd_r=0.0,
            best_r=0.0, worst_r=0.0, avg_mfe_r=0.0, avg_mae_r=0.0,
            top_month="", top_month_pct=0.0,
        )

    r   = np.array([t.net_r for t in trades])
    gw  = r[r > 0].sum();  gl = -r[r < 0].sum()
    pf  = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    eq  = np.cumsum(r);  peak = np.maximum.accumulate(eq)
    dd  = float((eq - peak).min())

    # Month concentration
    months = pd.Series([t.exit_time for t in trades]).dt.to_period("M").astype(str)
    month_r = {}
    for m, rv in zip(months, r):
        month_r[m] = month_r.get(m, 0.0) + rv
    top_m   = max(month_r, key=month_r.get) if month_r else ""
    total_r = float(r.sum())
    top_pct = float(month_r.get(top_m, 0.0) / total_r) if total_r > 0 else 0.0

    return dict(
        symbol        = symbol,
        trades        = int(len(r)),
        total_r       = total_r,
        avg_r         = float(r.mean()),
        win_rate      = float((r > 0).mean()),
        profit_factor = pf,
        max_dd_r      = dd,
        best_r        = float(r.max()),
        worst_r       = float(r.min()),
        avg_mfe_r     = float(np.mean([t.mfe_r for t in trades])),
        avg_mae_r     = float(np.mean([t.mae_r for t in trades])),
        top_month     = top_m,
        top_month_pct = top_pct,
    )


def should_exclude(row: dict) -> bool:
    """Remove if avg_r < 0 AND total_r < 0 AND trades < FILTER_TRADES."""
    return (row["avg_r"]   < FILTER_AVG_R and
            row["total_r"] < FILTER_TOTAL_R and
            row["trades"]  < FILTER_TRADES)


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    all_stats: List[dict],
    filtered_stats: List[dict],
    total_trades: int,
    filtered_trades: int,
) -> None:
    all_df  = pd.DataFrame(all_stats)
    filt_df = pd.DataFrame(filtered_stats)

    lines = [
        "PHASE T17 STEP B — Donchian Universe Expansion: Backtest + Symbol Filter",
        "=" * 70,
        "",
        "Frozen Donchian config:",
        "  Entry:  close > Donchian N=20 upper (prior bar highs)",
        "  Filter: close > EMA(200)",
        "  Stop:   ATR(14) x 2.0",
        "  Exit:   Donchian N=10 lower OR Chandelier (ACT+4R, trail ATR(14)x3)",
        "",
        f"Additional exclusions: LUNA, FTT, EUR, PAXG",
        f"Filter condition: remove if avg_r < 0 AND total_r < 0 AND trades < 5",
        "",
        "=" * 70,
        "FULL UNIVERSE RESULTS",
        "=" * 70,
        f"  Symbols backtested : {len(all_stats)}",
        f"  Total trades       : {total_trades}",
        f"  Positive symbols   : {(all_df['avg_r'] > 0).sum()}",
        f"  Negative symbols   : {(all_df['avg_r'] <= 0).sum()}",
        f"  Overall avg_r      : {all_df['avg_r'].mean():+.4f}R  (unweighted across symbols)",
        f"  Overall PF         : {(all_df['profit_factor'].replace([float('inf')], np.nan).mean()):.3f}",
        "",
        "=" * 70,
        "FILTERED UNIVERSE RESULTS",
        "=" * 70,
        f"  Symbols after filter: {len(filtered_stats)}",
        f"  Symbols removed     : {len(all_stats) - len(filtered_stats)}",
        f"  Total trades        : {filtered_trades}",
        f"  Positive symbols    : {(filt_df['avg_r'] > 0).sum()}",
        f"  Overall avg_r       : {filt_df['avg_r'].mean():+.4f}R",
        f"  Overall PF          : {(filt_df['profit_factor'].replace([float('inf')], np.nan).mean()):.3f}",
        "",
        "=" * 70,
        "BENCHMARK COMPARISON",
        "=" * 70,
        f"  Baseline (64 syms):  avg_r=+0.179R  PF=1.27",
        f"  Expanded (185 syms): avg_r={all_df['avg_r'].mean():+.4f}R",
        f"  Filtered ({len(filtered_stats):3d} syms): avg_r={filt_df['avg_r'].mean():+.4f}R",
        "",
        "=" * 70,
        "PER-SYMBOL DETAIL (sorted by avg_r desc)",
        "=" * 70,
        f"  {'Symbol':22s} {'Trades':6s} {'avg_r':7s} {'total_r':8s} {'PF':5s} {'win%':5s} {'DD':7s} {'TopMonth':8s} {'Pct':5s} INCL",
        "  " + "-" * 90,
    ]

    for row in sorted(all_stats, key=lambda x: x["avg_r"], reverse=True):
        excl = should_exclude(row)
        pf_s = f"{row['profit_factor']:.3f}" if row["profit_factor"] != float("inf") else " inf"
        lines.append(
            f"  {row['symbol']:22s} {int(row['trades']):6d} {row['avg_r']:+7.4f}"
            f" {row['total_r']:+8.2f} {pf_s:5s} {row['win_rate']:5.1%}"
            f" {row['max_dd_r']:+7.2f} {row['top_month']:>8s} {row['top_month_pct']:>5.1%}"
            f" {'NO ' if excl else 'YES'}"
        )

    lines += [
        "",
        "=" * 70,
        "REMOVED SYMBOLS (excluded by filter)",
        "=" * 70,
    ]
    for row in sorted(all_stats, key=lambda x: x["avg_r"]):
        if should_exclude(row):
            lines.append(
                f"  {row['symbol']:22s}  trades={int(row['trades'])}  avg_r={row['avg_r']:+.4f}  total_r={row['total_r']:+.2f}"
            )

    rpt = OUT_DIR / "step_b_summary.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("Phase T17 Step B — Donchian Universe Backtest + Symbol Filter")
    print("=" * 70)

    # Load symbol list
    univ = pd.read_csv(UNIV_CSV)
    symbols = (
        univ[univ["included"] == True]["symbol"]
        .tolist()
    )
    symbols = [s for s in symbols if s not in STEP_B_EXCLUDE]
    print(f"Symbols to backtest: {len(symbols)}")
    print(f"Frozen config: Donchian N={DONCHIAN_N} / ATR(14)x{STOP_MULT} stop / "
          f"Chandelier ACT+{CHAN_ACT_R:.0f}R trail {CHAN_TRAIL:.0f}xATR")
    print()

    all_trades: List[Trade] = []
    all_stats:  List[dict]  = []

    for i, sym in enumerate(sorted(symbols), 1):
        df = load_ohlcv(sym)
        if df is None or len(df) < 250:
            print(f"  [{i:3d}/{len(symbols)}] {sym:22s} -- NO DATA, skip")
            continue

        trades = run_donchian_backtest(df, sym)
        stats  = symbol_stats(trades, sym)
        all_trades.extend(trades)
        all_stats.append(stats)

        status = f"trades={stats['trades']:3d}  avg_r={stats['avg_r']:+.4f}  PF={stats['profit_factor']:.3f}"
        flag   = "  [EXCL]" if should_exclude(stats) else ""
        print(f"  [{i:3d}/{len(symbols)}] {sym:22s}  {status}{flag}")

    print()
    print(f"Total: {len(all_stats)} symbols, {len(all_trades)} trades")

    # Apply filter
    filtered_stats  = [s for s in all_stats  if not should_exclude(s)]
    filtered_set    = {s["symbol"] for s in filtered_stats}
    filtered_trades = [t for t in all_trades if t.symbol in filtered_set]

    print(f"After filter: {len(filtered_stats)} symbols, {len(filtered_trades)} trades")
    print(f"Removed:      {len(all_stats) - len(filtered_stats)} symbols")

    # Save trades CSV
    if all_trades:
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
            OUT_DIR / "step_b_all_trades.csv", index=False
        )

    # Save filtered_symbols.csv
    stats_df = pd.DataFrame(all_stats)
    stats_df["step_b_included"] = stats_df["symbol"].isin(filtered_set)
    stats_df.to_csv(OUT_DIR / "filtered_symbols.csv", index=False)

    write_report(all_stats, filtered_stats, len(all_trades), len(filtered_trades))

    # Console summary
    all_df  = pd.DataFrame(all_stats)
    filt_df = pd.DataFrame(filtered_stats)
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline (64 syms):   avg_r=+0.179R  PF=1.27")
    print(f"  All 185 syms:         avg_r={all_df['avg_r'].mean():+.4f}R")
    print(f"  Filtered ({len(filtered_stats):3d} syms):   avg_r={filt_df['avg_r'].mean():+.4f}R")
    print(f"  Positive symbols (filtered): {(filt_df['avg_r'] > 0).sum()}/{len(filtered_stats)}")
    print()
    print("Top 15 symbols by avg_r (filtered):")
    top15 = filt_df.sort_values("avg_r", ascending=False).head(15)
    for _, r in top15.iterrows():
        print(f"  {r['symbol']:22s}  avg_r={r['avg_r']:+.4f}  trades={int(r['trades'])}  PF={r['profit_factor']:.3f}")
    print()
    print("=" * 70)
    print("Step B complete. Outputs: data/universe/filtered_symbols.csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
