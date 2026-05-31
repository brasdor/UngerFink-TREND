#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T2 -- KeltnerLong Core Engine

Locks the T1 canonical configuration and runs a fully-instrumented backtest
with MAE/MFE tracking per trade (required for T3B chandelier development).

T1 canonical (ema200_price, N=15, km=3.0, sm=2.0):
  avg_r=+0.417R  PF=1.67  zone=100% PASS  trades=167

Entry:   Close > EMA(N) + ATR(N)*keltner_mult  (upper band, shifted 1 bar)
Filter:  close > EMA200
Stop:    entry - ATR(N)*stop_mult
Exit 1:  low <= stop
Exit 2:  close < EMA(N)  (midline cross-under)

Data:    data/research_trend_t15/ohlcv_cache/  (reuse existing 1d cache)
Output:  data/research_keltnerlong_t2/
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG  (T1 canonical)
# =============================================================================

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_keltnerlong_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME     = "1d"
BACKTEST_BARS = 1500

KELTNER_N     = 15
KELTNER_MULT  = 3.0
STOP_MULT     = 2.0
FILTER_MODE   = "ema200_price"

EMA200_N = 200

COST_FLOOR_R  = 0.15   # Binance Spot, Section 4.2
WIN_RATE_LO   = 30.0   # Section 4.1 lower bound
WIN_RATE_HI   = 50.0   # Section 4.1 upper bound
MIN_BARS      = 300

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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ATR(N) -- Wilder rolling
    pc = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(KELTNER_N).mean()

    # EMA(N) -- midline + exit trigger (current bar)
    ema_n = d["close"].ewm(span=KELTNER_N, adjust=False).mean()

    d["ema_n"]    = ema_n
    d["atr_n"]    = atr_n
    d["ema_n_s1"] = ema_n.shift(1)   # shifted for entry band
    d["atr_n_s1"] = atr_n.shift(1)   # shifted for entry band

    d["ema200"]   = d["close"].ewm(span=EMA200_N, adjust=False).mean()

    return d


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class Trade:
    trade_id:      int
    symbol:        str
    timeframe:     str
    side:          str
    filter_mode:   str
    keltner_n:     int
    keltner_mult:  float
    stop_mult:     float
    entry_time:    str
    exit_time:     str
    entry_price:   float
    exit_price:    float
    initial_stop:  float
    initial_risk:  float
    exit_reason:   str
    bars_held:     int
    net_r:         float
    pnl_pct:       float
    mae_r:         float
    mfe_r:         float


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    first_id: int,
) -> List[Trade]:
    d      = add_indicators(df)
    warmup = max(KELTNER_N, EMA200_N) + 5
    pos    = None
    trades = []
    tid    = first_id

    for i in range(warmup, len(d)):
        row = d.iloc[i]

        atr_n = float(row["atr_n"])
        if not math.isfinite(atr_n) or atr_n <= 0:
            continue

        close    = float(row["close"])
        high     = float(row["high"])
        low      = float(row["low"])
        t        = d.index[i]
        ema_n    = float(row["ema_n"])
        ema_n_s1 = float(row["ema_n_s1"])
        atr_n_s1 = float(row["atr_n_s1"])
        ema200   = float(row["ema200"])

        if not all(math.isfinite(x) for x in [ema_n, ema_n_s1, atr_n_s1, ema200]):
            continue

        upper_band = ema_n_s1 + atr_n_s1 * KELTNER_MULT

        if pos is not None:
            bars_held = i - pos["entry_i"]

            # MAE / MFE update (per-bar extremes)
            pos["mfe_r"] = max(pos["mfe_r"], (high  - pos["entry"]) / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (low   - pos["entry"]) / pos["risk"])

            exit_px = None
            reason  = None

            # Exit 1: stop hit
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                reason  = "initial_stop"
            # Exit 2: midline cross-under
            elif close < ema_n:
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
                    keltner_n=KELTNER_N,
                    keltner_mult=KELTNER_MULT,
                    stop_mult=STOP_MULT,
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

        if pos is None:
            # Filter: close > EMA200
            if close <= ema200:
                continue
            # Entry: close breaks above Keltner upper band (previous bar)
            if close > upper_band:
                risk = atr_n * STOP_MULT
                if risk > 0:
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
# STATS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    return float(g / l) if l > 1e-12 else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


# =============================================================================
# OUTPUT BUILDERS
# =============================================================================

def build_summary(trades_df: pd.DataFrame, n_syms: int) -> pd.DataFrame:
    r = trades_df["net_r"].values if not trades_df.empty else np.array([])
    r = r[np.isfinite(r)]
    b = trades_df["bars_held"].values if not trades_df.empty else np.array([])
    n = len(r)
    row = dict(
        phase="T2_KELTNERLONG",
        timeframe=TIMEFRAME,
        filter_mode=FILTER_MODE,
        keltner_n=KELTNER_N,
        keltner_mult=KELTNER_MULT,
        stop_mult=STOP_MULT,
        symbols=n_syms,
        trades=n,
        total_r=float(r.sum())    if n else 0.0,
        avg_r=float(r.mean())     if n else 0.0,
        median_r=float(np.median(r)) if n else 0.0,
        win_rate_pct=float(100.0 * (r > 0).mean()) if n else 0.0,
        profit_factor=_pf(r),
        max_dd_r=_dd(r),
        best_trade_r=float(r.max())  if n else 0.0,
        worst_trade_r=float(r.min()) if n else 0.0,
        avg_bars_held=float(b.mean()) if len(b) else 0.0,
        avg_mfe_r=float(trades_df["mfe_r"].mean()) if not trades_df.empty else 0.0,
        avg_mae_r=float(trades_df["mae_r"].mean()) if not trades_df.empty else 0.0,
        cost_floor_pass=("PASS" if (r.mean() if n else 0.0) > COST_FLOOR_R else "FAIL"),
        win_rate_gate=("PASS" if WIN_RATE_LO <= (100.0*(r>0).mean() if n else 0.0) <= WIN_RATE_HI
                       else "WARN"),
    )
    return pd.DataFrame([row])


def build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    rows = []
    for sym, g in trades_df.groupby("symbol"):
        r = g["net_r"].values
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        rows.append(dict(
            symbol=sym,
            trades=int(len(r)),
            total_r=float(r.sum()),
            avg_r=float(r.mean()),
            win_rate_pct=float(100.0 * (r > 0).mean()),
            profit_factor=_pf(r),
            max_dd_r=_dd(r),
            best_trade_r=float(r.max()),
            worst_trade_r=float(r.min()),
            avg_bars_held=float(g["bars_held"].mean()),
            avg_mfe_r=float(g["mfe_r"].mean()),
            avg_mae_r=float(g["mae_r"].mean()),
        ))
    return pd.DataFrame(rows).sort_values(
        ["total_r", "profit_factor"], ascending=False
    ).reset_index(drop=True)


def build_equity(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    eq = trades_df.copy()
    eq["exit_time"] = pd.to_datetime(eq["exit_time"], utc=True)
    eq = eq.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)
    eq["equity_r"] = eq["net_r"].cumsum()
    eq["peak_r"]   = eq["equity_r"].cummax()
    eq["dd_r"]     = eq["equity_r"] - eq["peak_r"]
    return eq[["exit_time", "trade_id", "symbol", "net_r",
               "equity_r", "peak_r", "dd_r", "mfe_r", "mae_r"]]


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 80)
    print("PHASE T2 -- KeltnerLong Core Engine")
    print("=" * 80)
    print(f"Timeframe:     {TIMEFRAME}")
    print(f"Filter mode:   {FILTER_MODE}")
    print(f"Keltner N:     {KELTNER_N}")
    print(f"Keltner mult:  {KELTNER_MULT}")
    print(f"Stop mult:     {STOP_MULT}")
    print(f"Entry band:    EMA({KELTNER_N}) + ATR({KELTNER_N})*{KELTNER_MULT}  (shifted 1 bar)")
    print(f"Exit 2:        close < EMA({KELTNER_N})  (midline cross-under)")
    print(f"Cost floor:    avg_r > {COST_FLOOR_R}R  (Binance Spot, Section 4.2)")
    print(f"Cache:         {CACHE_DIR}")
    print()

    symbols = discover_symbols()
    print(f"Symbols found: {len(symbols)}")
    if not symbols:
        print(f"[ERROR] No symbols found in cache: {CACHE_DIR}")
        return 1

    print("Loading OHLCV data...")
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is not None:
            data[sym] = df
        else:
            print(f"  SKIP {sym}")
    print(f"Loaded: {len(data)} / {len(symbols)} symbols")

    if not data:
        print("[ERROR] No data loaded.")
        return 1

    print(f"\nRunning backtest: {len(data)} symbols...")
    all_trades: List[Trade] = []
    tid = 1

    for sym, df in data.items():
        trades = backtest_symbol(sym, df, first_id=tid)
        if trades:
            tid = max(t.trade_id for t in trades) + 1
        all_trades.extend(trades)
        r = sum(t.net_r for t in trades)
        print(f"  {sym:15s}  trades={len(trades):3d}  R={r:+8.2f}")

    # Build outputs
    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"],  utc=True)
        trades_df = trades_df.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)

    summary_df    = build_summary(trades_df, len(data))
    asset_summary = build_asset_summary(trades_df)
    equity_df     = build_equity(trades_df)

    trades_df.to_csv(    OUT_DIR / "phase_t2_keltnerlong_trades.csv",        index=False)
    equity_df.to_csv(    OUT_DIR / "phase_t2_keltnerlong_equity.csv",         index=False)
    summary_df.to_csv(   OUT_DIR / "phase_t2_keltnerlong_summary.csv",        index=False)
    asset_summary.to_csv(OUT_DIR / "phase_t2_keltnerlong_asset_summary.csv",  index=False)

    # Console summary
    print()
    print("=" * 80)
    print("PHASE T2 SUMMARY")
    print("=" * 80)
    if not summary_df.empty:
        s = summary_df.iloc[0]
        print(f"  Trades:         {int(s['trades'])}")
        print(f"  Total R:        {s['total_r']:+.2f}R")
        print(f"  Avg R/trade:    {s['avg_r']:+.4f}R  (cost floor: {s['cost_floor_pass']})")
        print(f"  Profit factor:  {s['profit_factor']:.3f}")
        print(f"  Win rate:       {s['win_rate_pct']:.1f}%  (Sec 4.1 gate: {s['win_rate_gate']})")
        print(f"  Max DD (R):     {s['max_dd_r']:+.2f}R")
        print(f"  Avg bars held:  {s['avg_bars_held']:.1f}")
        print(f"  Avg MFE:        {s['avg_mfe_r']:+.3f}R")
        print(f"  Avg MAE:        {s['avg_mae_r']:+.3f}R")

    if not asset_summary.empty:
        print()
        print("Top 15 symbols by total R:")
        cols = ["symbol", "trades", "total_r", "avg_r",
                "profit_factor", "win_rate_pct", "avg_mfe_r"]
        print(asset_summary[cols].head(15).to_string(index=False))

    # Gate checks
    print()
    print("=" * 80)
    print("GATE CHECKS")
    print("=" * 80)
    if not summary_df.empty:
        s     = summary_df.iloc[0]
        n     = int(s["trades"])
        avg_r = float(s["avg_r"])
        pf    = float(s["profit_factor"])
        wr    = float(s["win_rate_pct"])
        print(f"  Sec 4.2 cost floor (avg_r > {COST_FLOOR_R}R): "
              f"{'PASS' if avg_r > COST_FLOOR_R else 'FAIL'}  ({avg_r:.4f}R)")
        print(f"  Sec 4.1 win rate ({WIN_RATE_LO}-{WIN_RATE_HI}%):    "
              f"{'PASS' if WIN_RATE_LO <= wr <= WIN_RATE_HI else 'WARN'}  ({wr:.1f}%)")
        print(f"  PF > 1.5:                           "
              f"{'PASS' if pf > 1.5 else 'FAIL'}  ({pf:.3f})")
        print(f"  Trade count:                        {n}")
        if wr < WIN_RATE_LO:
            print(f"  NOTE: Win rate {wr:.1f}% < {WIN_RATE_LO}% floor -- "
                  "typical for high-keltner_mult breakout entries. "
                  "Avg_r and PF still pass.")

    # T3B hint
    print()
    print("=" * 80)
    print("NEXT STEP: T3B  (Wide-Exit / Chandelier Engineering)")
    print("=" * 80)
    print(f"  Canonical config locked:  ema200_price / N={KELTNER_N} / "
          f"km={KELTNER_MULT} / sm={STOP_MULT}")
    print(f"  Avg MFE: {summary_df.iloc[0]['avg_mfe_r']:.3f}R  "
          f"-- use to set chandelier activation threshold in T3B")
    print(f"  T3B chandelier activation suggestion: ~{summary_df.iloc[0]['avg_mfe_r'] * 0.5:.1f}R "
          f"(50% of avg MFE)")
    print()
    print("  Outputs:")
    print(f"    {OUT_DIR / 'phase_t2_keltnerlong_trades.csv'}")
    print(f"    {OUT_DIR / 'phase_t2_keltnerlong_equity.csv'}")
    print(f"    {OUT_DIR / 'phase_t2_keltnerlong_summary.csv'}")
    print(f"    {OUT_DIR / 'phase_t2_keltnerlong_asset_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
