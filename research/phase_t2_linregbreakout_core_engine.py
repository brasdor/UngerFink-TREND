#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T2 -- LinearRegressionBreakout Core Engine

Runs two T1 canonical configs side-by-side with full per-symbol resolution
and MAE/MFE tracking per trade (required for T3B chandelier development).

Config A (wide zone):   N=40 / mult=2.5 / sm=2.0 / ema200_price
  T1: avg_r=+0.284R  zone=[30,40,55] 100% PASS  trades=269

Config B (high avg_r):  N=55 / mult=3.0 / sm=2.0 / ema200_price
  T1: avg_r=+0.402R  zone=[40,55]   100% PASS  trades=195

Entry:  Close > LinReg(N)[i-1] + mult * StdErr(N)[i-1]  (shifted, no lookahead)
Filter: Close > EMA(200)
Stop:   Entry - ATR(N)[i-1] * stop_mult
Exit 1: Low <= stop
Exit 2: Close < LinReg(N)  (midline cross-under, current bar)

Data:   data/research_trend_t15/ohlcv_cache/  (1d cache)
Output: data/research_linregbreakout_t2/
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT      = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_linregbreakout_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME     = "1d"
BACKTEST_BARS = 1500
EMA200_N      = 200
FILTER_MODE   = "ema200_price"
MIN_BARS      = 300

COST_FLOOR_R = 0.15
WIN_RATE_LO  = 30.0
WIN_RATE_HI  = 50.0

CONFIGS = [
    dict(label="A", linreg_n=40, linreg_mult=2.5, stop_mult=2.0),
    dict(label="B", linreg_n=55, linreg_mult=3.0, stop_mult=2.0),
]

EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI", "WBTC", "BTTC",
}
EXCLUDE_KW = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")


# =============================================================================
# DATA
# =============================================================================

def discover_symbols() -> List[str]:
    syms = []
    for f in sorted(CACHE_DIR.glob("*_1d.csv")):
        parts = f.stem.split("_")
        if parts and parts[-1] == "1d":
            parts = parts[:-1]
        if len(parts) < 2:
            continue
        quote = parts[-1]
        base  = "_".join(parts[:-1])
        if quote != "USDT":
            continue
        if base in EXCLUDE_BASES:
            continue
        if any(kw in base for kw in EXCLUDE_KW):
            continue
        syms.append(f"{base}/{quote}")
    return syms


def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    fname = symbol.replace("/", "_") + "_1d.csv"
    path  = CACHE_DIR / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
        if len(df) < MIN_BARS:
            return None
        return df.tail(BACKTEST_BARS)
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def linreg_rolling(values: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS over last N closes. Returns (lr_end, lr_se, lr_slope)."""
    nbar    = len(values)
    xs      = np.arange(n, dtype=float)
    xs_mean = (n - 1) / 2.0
    ss_xx   = ((xs - xs_mean) ** 2).sum()

    lr_end   = np.full(nbar, np.nan)
    lr_se    = np.full(nbar, np.nan)
    lr_slope = np.full(nbar, np.nan)

    for i in range(n - 1, nbar):
        ys = values[i - n + 1 : i + 1]
        if not np.all(np.isfinite(ys)):
            continue
        ys_mean   = ys.mean()
        ss_xy     = ((xs - xs_mean) * (ys - ys_mean)).sum()
        slope     = ss_xy / ss_xx
        intercept = ys_mean - slope * xs_mean
        lr_end[i]   = intercept + slope * (n - 1)
        fitted      = intercept + slope * xs
        resid_ss    = ((ys - fitted) ** 2).sum()
        lr_se[i]    = np.sqrt(resid_ss / (n - 2)) if n > 2 else 0.0
        lr_slope[i] = slope

    return lr_end, lr_se, lr_slope


def add_indicators(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d      = df.copy()
    closes = d["close"].to_numpy(dtype=float)

    lr_end_arr, lr_se_arr, _ = linreg_rolling(closes, n)

    # ATR(N) -- Wilder simple rolling
    pc = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean().to_numpy(dtype=float)

    d["lr_end"]    = lr_end_arr
    d["lr_se"]     = lr_se_arr
    d["atr_n"]     = atr_n

    # Shifted 1 bar -- no lookahead at entry
    d["lr_end_s1"] = d["lr_end"].shift(1)
    d["lr_se_s1"]  = d["lr_se"].shift(1)
    d["atr_n_s1"]  = d["atr_n"].shift(1)

    # Regime filter
    d["ema200"]    = d["close"].ewm(span=EMA200_N, adjust=False).mean()

    return d


# =============================================================================
# TRADE DATACLASS
# =============================================================================

@dataclass
class Trade:
    trade_id:     int
    symbol:       str
    timeframe:    str
    side:         str
    filter_mode:  str
    linreg_n:     int
    linreg_mult:  float
    stop_mult:    float
    entry_time:   str
    exit_time:    str
    entry_price:  float
    exit_price:   float
    initial_stop: float
    initial_risk: float
    exit_reason:  str
    bars_held:    int
    net_r:        float
    pnl_pct:      float
    mae_r:        float
    mfe_r:        float


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    linreg_n: int,
    linreg_mult: float,
    stop_mult: float,
    first_id: int,
) -> List[Trade]:
    d      = add_indicators(df, linreg_n)
    warmup = EMA200_N + linreg_n + 5
    pos    = None
    trades: List[Trade] = []
    tid    = first_id

    for i in range(warmup, len(d)):
        row = d.iloc[i]

        close      = float(row["close"])
        high       = float(row["high"])
        low        = float(row["low"])
        t          = d.index[i]
        lr_end     = float(row["lr_end"])
        atr_n      = float(row["atr_n"])
        ema200     = float(row["ema200"])
        lr_end_s1  = float(row["lr_end_s1"])
        lr_se_s1   = float(row["lr_se_s1"])
        atr_n_s1   = float(row["atr_n_s1"])

        if not all(math.isfinite(x) for x in
                   [lr_end, atr_n, ema200, lr_end_s1, lr_se_s1, atr_n_s1]):
            continue
        if lr_se_s1 <= 0:
            continue

        upper_band = lr_end_s1 + lr_se_s1 * linreg_mult

        if pos is not None:
            bars_held = i - pos["entry_i"]

            # MAE / MFE update
            pos["mfe_r"] = max(pos["mfe_r"], (high - pos["entry"]) / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (low  - pos["entry"]) / pos["risk"])

            exit_px = None
            reason  = None

            # Exit 1: initial stop
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                reason  = "initial_stop"
            # Exit 2: midline cross-under
            elif close < lr_end:
                exit_px = close
                reason  = "midline_exit"

            if exit_px is not None:
                net_r   = (float(exit_px) - pos["entry"]) / pos["risk"]
                pnl_pct = (float(exit_px) / pos["entry"] - 1.0) * 100.0
                trades.append(Trade(
                    trade_id=tid,
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                    side="LONG",
                    filter_mode=FILTER_MODE,
                    linreg_n=linreg_n,
                    linreg_mult=linreg_mult,
                    stop_mult=stop_mult,
                    entry_time=str(pos["entry_time"]),
                    exit_time=str(t),
                    entry_price=float(pos["entry"]),
                    exit_price=float(exit_px),
                    initial_stop=float(pos["stop"]),
                    initial_risk=float(pos["risk"]),
                    exit_reason=reason,
                    bars_held=int(bars_held),
                    net_r=float(net_r),
                    pnl_pct=float(pnl_pct),
                    mae_r=float(pos["mae_r"]),
                    mfe_r=float(pos["mfe_r"]),
                ))
                tid += 1
                pos = None
            continue

        # Entry
        if close <= ema200:
            continue
        if close > upper_band:
            risk = atr_n_s1 * stop_mult
            if risk <= 0:
                continue
            pos = dict(
                entry_time=t,
                entry_i=i,
                entry=close,
                stop=close - risk,
                risk=risk,
                mfe_r=0.0,
                mae_r=0.0,
            )

    return trades


# =============================================================================
# STATS HELPERS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = float(r[r > 0].sum())
    l = float(-r[r < 0].sum())
    return g / l if l > 1e-12 else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


# =============================================================================
# OUTPUT BUILDERS
# =============================================================================

def build_summary(trades_df: pd.DataFrame, cfg: dict, n_syms: int) -> pd.DataFrame:
    r = trades_df["net_r"].to_numpy(dtype=float) if not trades_df.empty else np.array([])
    r = r[np.isfinite(r)]
    b = trades_df["bars_held"].to_numpy(dtype=float) if not trades_df.empty else np.array([])
    n = len(r)
    row = dict(
        config=cfg["label"],
        timeframe=TIMEFRAME,
        filter_mode=FILTER_MODE,
        linreg_n=cfg["linreg_n"],
        linreg_mult=cfg["linreg_mult"],
        stop_mult=cfg["stop_mult"],
        symbols=n_syms,
        trades=n,
        total_r=float(r.sum())        if n else 0.0,
        avg_r=float(r.mean())         if n else 0.0,
        median_r=float(np.median(r))  if n else 0.0,
        win_rate_pct=float(100.0 * (r > 0).mean()) if n else 0.0,
        profit_factor=_pf(r),
        max_dd_r=_dd(r),
        best_trade_r=float(r.max())   if n else 0.0,
        worst_trade_r=float(r.min())  if n else 0.0,
        avg_bars_held=float(b.mean()) if len(b) else 0.0,
        avg_mfe_r=float(trades_df["mfe_r"].mean()) if not trades_df.empty else 0.0,
        avg_mae_r=float(trades_df["mae_r"].mean()) if not trades_df.empty else 0.0,
        cost_floor_pass=("PASS" if (float(r.mean()) if n else 0.0) > COST_FLOOR_R else "FAIL"),
        win_rate_gate=("PASS" if WIN_RATE_LO <= (float(100.0 * (r > 0).mean()) if n else 0.0) <= WIN_RATE_HI
                       else "WARN"),
    )
    return pd.DataFrame([row])


def build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    agg = (
        trades_df.groupby("symbol")
        .agg(
            trades=("net_r", "size"),
            total_r=("net_r", "sum"),
            avg_r=("net_r", "mean"),
            profit_factor=("net_r", lambda x: _pf(x.to_numpy())),
            win_rate_pct=("net_r", lambda x: 100.0 * (x > 0).mean()),
            avg_mfe_r=("mfe_r", "mean"),
            avg_mae_r=("mae_r", "mean"),
        )
        .reset_index()
        .sort_values("total_r", ascending=False)
    )
    return agg


def build_equity(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df[["exit_time", "net_r"]].copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, format="mixed",
                                     errors="coerce")
    # Fix 1970-epoch timestamps
    if not df.empty and df["exit_time"].dt.year.max() < 2000:
        ns_vals = df["exit_time"].astype("int64")
        df["exit_time"] = pd.to_datetime(ns_vals, unit="ms", utc=True)
    df = df.sort_values("exit_time").reset_index(drop=True)
    df["cumulative_r"] = df["net_r"].cumsum()
    return df


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T2 -- LinearRegressionBreakout Core Engine")
    print("=" * 70)
    print(f"Timeframe:  {TIMEFRAME}")
    print(f"Filter:     {FILTER_MODE}")
    for cfg in CONFIGS:
        print(f"  Config {cfg['label']}: N={cfg['linreg_n']}  mult={cfg['linreg_mult']}  sm={cfg['stop_mult']}")
    print(f"Cache:      {CACHE_DIR}")
    print()

    symbols = discover_symbols()
    print(f"Symbols found: {len(symbols)}")
    print("Loading OHLCV data...")

    sym_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None:
            print(f"  SKIP {sym}")
            continue
        sym_data[sym] = df
    print(f"Loaded: {len(sym_data)} / {len(symbols)} symbols")
    print()

    all_summaries: List[pd.DataFrame] = []

    for cfg in CONFIGS:
        n      = cfg["linreg_n"]
        mult   = cfg["linreg_mult"]
        sm     = cfg["stop_mult"]
        label  = cfg["label"]
        prefix = f"phase_t2_linregbreakout_cfg{label}"

        print(f"Running Config {label}: N={n}  mult={mult}  sm={sm}")
        print("-" * 50)

        all_trades: List[Trade] = []
        tid = 0

        for sym, df in sym_data.items():
            trades = backtest_symbol(sym, df, n, mult, sm, tid)
            if trades:
                tid += len(trades)
            all_trades.extend(trades)

            r_sym = np.asarray([t.net_r for t in trades])
            print(f"  {sym:<20}  trades={len(trades):3d}  R={r_sym.sum():+7.2f}")

        trades_df = pd.DataFrame([asdict(t) for t in all_trades])
        n_syms    = len([s for s in sym_data if any(t.symbol == s for t in all_trades)])

        summary = build_summary(trades_df, cfg, n_syms)
        asset   = build_asset_summary(trades_df)
        equity  = build_equity(trades_df)

        trades_df.to_csv(OUT_DIR / f"{prefix}_trades.csv", index=False)
        summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
        if not asset.empty:
            asset.to_csv(OUT_DIR / f"{prefix}_asset_summary.csv", index=False)
        if not equity.empty:
            equity.to_csv(OUT_DIR / f"{prefix}_equity.csv", index=False)

        all_summaries.append(summary)

        # Print summary block
        r_arr = trades_df["net_r"].to_numpy(dtype=float) if not trades_df.empty else np.array([])
        r_arr = r_arr[np.isfinite(r_arr)]
        tot   = float(r_arr.sum()) if len(r_arr) else 0.0
        avg   = float(r_arr.mean()) if len(r_arr) else 0.0
        pf    = _pf(r_arr)
        wr    = float(100.0 * (r_arr > 0).mean()) if len(r_arr) else 0.0
        dd    = _dd(r_arr)
        avg_mfe = float(trades_df["mfe_r"].mean()) if not trades_df.empty else 0.0
        avg_mae = float(trades_df["mae_r"].mean()) if not trades_df.empty else 0.0

        cf_gate = "PASS" if avg > COST_FLOOR_R else "FAIL"
        wr_gate = "PASS" if WIN_RATE_LO <= wr <= WIN_RATE_HI else "WARN"

        print()
        print(f"  Config {label} SUMMARY  (N={n} / mult={mult} / sm={sm})")
        print(f"  Trades:        {len(r_arr)}")
        print(f"  Total R:       {tot:+.2f}R")
        print(f"  Avg R/trade:   {avg:+.4f}R  [cost floor {COST_FLOOR_R}R: {cf_gate}]")
        print(f"  Profit factor: {pf:.3f}")
        print(f"  Win rate:      {wr:.1f}%   [Sec 4.1 {WIN_RATE_LO}-{WIN_RATE_HI}%: {wr_gate}]")
        print(f"  Max DD (R):    {dd:.2f}R")
        print(f"  Avg MFE:       {avg_mfe:+.3f}R")
        print(f"  Avg MAE:       {avg_mae:+.3f}R")
        print(f"  Chan act (50% avg MFE): ~{avg_mfe * 0.5:.1f}R  (T3B suggestion)")
        print()

        if not asset.empty:
            total_pos_r = float(asset.loc[asset["total_r"] > 0, "total_r"].sum())
            print(f"  Top 10 symbols by total R:")
            for _, row in asset.head(10).iterrows():
                pct = 100.0 * row["total_r"] / tot if tot != 0 else 0.0
                print(f"    {row['symbol']:<20}  trades={int(row['trades']):3d}"
                      f"  total_r={row['total_r']:+7.2f}R"
                      f"  avg_r={row['avg_r']:+.3f}R"
                      f"  ({pct:+.1f}%)")
        print()

    # Save combined summary
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(OUT_DIR / "phase_t2_linregbreakout_combined_summary.csv", index=False)

    # Side-by-side comparison
    print("=" * 70)
    print("CONFIG COMPARISON")
    print("=" * 70)
    if len(all_summaries) == 2:
        sa = all_summaries[0].iloc[0]
        sb = all_summaries[1].iloc[0]
        rows = [
            ("Config",       f"A (N={CONFIGS[0]['linreg_n']} m={CONFIGS[0]['linreg_mult']})",
                             f"B (N={CONFIGS[1]['linreg_n']} m={CONFIGS[1]['linreg_mult']})"),
            ("Trades",       str(int(sa["trades"])),           str(int(sb["trades"]))),
            ("Total R",      f"{sa['total_r']:+.2f}R",         f"{sb['total_r']:+.2f}R"),
            ("Avg R/trade",  f"{sa['avg_r']:+.4f}R",           f"{sb['avg_r']:+.4f}R"),
            ("Win rate",     f"{sa['win_rate_pct']:.1f}%",     f"{sb['win_rate_pct']:.1f}%"),
            ("PF",           f"{sa['profit_factor']:.3f}",     f"{sb['profit_factor']:.3f}"),
            ("Max DD",       f"{sa['max_dd_r']:.2f}R",         f"{sb['max_dd_r']:.2f}R"),
            ("Avg MFE",      f"{sa['avg_mfe_r']:+.3f}R",       f"{sb['avg_mfe_r']:+.3f}R"),
            ("Cost floor",   sa["cost_floor_pass"],             sb["cost_floor_pass"]),
            ("Win rate gate",sa["win_rate_gate"],               sb["win_rate_gate"]),
        ]
        w = max(len(r[0]) for r in rows) + 2
        wv = 22
        print(f"  {'Metric':<{w}}  {'Config A':<{wv}}  {'Config B':<{wv}}")
        print(f"  {'-'*(w+2*wv+4)}")
        for metric, va, vb in rows:
            print(f"  {metric:<{w}}  {va:<{wv}}  {vb:<{wv}}")

    print()
    print("=" * 70)
    print("GATE CHECKS")
    print("=" * 70)
    for cfg, summ in zip(CONFIGS, all_summaries):
        s = summ.iloc[0]
        print(f"  Config {cfg['label']} (N={cfg['linreg_n']} mult={cfg['linreg_mult']}):")
        print(f"    Sec 4.2 cost floor (avg_r > {COST_FLOOR_R}R): {s['cost_floor_pass']}  ({s['avg_r']:.4f}R)")
        print(f"    Sec 4.1 win rate ({WIN_RATE_LO}-{WIN_RATE_HI}%):      {s['win_rate_gate']}  ({s['win_rate_pct']:.1f}%)")
        print(f"    PF > 1.0:                           {'PASS' if s['profit_factor'] > 1.0 else 'FAIL'}  ({s['profit_factor']:.3f})")
    print()

    print("=" * 70)
    print("NEXT STEP: T3B  (Wide-Exit / Chandelier Engineering)")
    print("=" * 70)
    for cfg, summ in zip(CONFIGS, all_summaries):
        s = summ.iloc[0]
        print(f"  Config {cfg['label']}: avg_MFE={s['avg_mfe_r']:.3f}R  "
              f"-> chandelier activation suggestion: ~{s['avg_mfe_r']*0.5:.1f}R")
    print()
    print("  Outputs:")
    for cfg in CONFIGS:
        lbl = cfg["label"]
        print(f"    Config {lbl}: {OUT_DIR / f'phase_t2_linregbreakout_cfg{lbl}_trades.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
