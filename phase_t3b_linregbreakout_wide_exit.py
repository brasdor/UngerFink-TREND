#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T3B -- LinearRegressionBreakout Wide-Exit / Chandelier Engineering

Tests whether a chandelier trailing stop improves on T2's midline cross-under exit.

T2 baselines:
  Config A (N=40 / mult=2.5 / sm=2.0): trades=271  avg_r=+0.2370R  PF=1.393  total_r=+64.24R
  Config B (N=55 / mult=3.0 / sm=2.0): trades=196  avg_r=+0.3624R  PF=1.528  total_r=+71.04R

Exit priority (per-bar, applied in order):
  1. low <= initial_stop          -> exit at stop
  2. chan_active AND low <= chan_stop -> exit at chan_stop
  3. NOT chan_active AND close < LinReg(N) -> exit at close (midline)
  Chandelier updated AFTER exit checks (never before -- critical sequencing).

Chandelier stop = HH_since_entry - ATR(N) * chan_atr_mult  (ratchets up)
Chandelier activates when trade reaches MFE >= chan_activate_r

Grid:
  CHAN_ACTIVATE_R_GRID : [1.0, 1.5, 2.0, 2.5, 3.0]
  CHAN_ATR_MULT_GRID   : [2.0, 2.5, 3.0]
  Total: 15 combos per config, 30 total

Data:   data/research_trend_t15/ohlcv_cache/  (1d)
Output: data/research_linregbreakout_t3b/
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

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_linregbreakout_t3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME    = "1d"
BACKTEST_BARS = 1500
EMA200_N     = 200
FILTER_MODE  = "ema200_price"
MIN_BARS     = 300

CHAN_ACTIVATE_R_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
CHAN_ATR_MULT_GRID   = [2.0, 2.5, 3.0]

# T2 baseline stats (from T2 run -- used for BETTER/WORSE annotation)
T2_BASELINES = {
    "A": dict(linreg_n=40, linreg_mult=2.5, stop_mult=2.0,
              trades=271, avg_r=0.2370, pf=1.393, total_r=64.24),
    "B": dict(linreg_n=55, linreg_mult=3.0, stop_mult=2.0,
              trades=196, avg_r=0.3624, pf=1.528, total_r=71.04),
}

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

def linreg_rolling(values: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling OLS. Returns (lr_end, lr_se) arrays."""
    nbar    = len(values)
    xs      = np.arange(n, dtype=float)
    xs_mean = (n - 1) / 2.0
    ss_xx   = ((xs - xs_mean) ** 2).sum()

    lr_end = np.full(nbar, np.nan)
    lr_se  = np.full(nbar, np.nan)

    for i in range(n - 1, nbar):
        ys = values[i - n + 1 : i + 1]
        if not np.all(np.isfinite(ys)):
            continue
        ys_mean   = ys.mean()
        slope     = ((xs - xs_mean) * (ys - ys_mean)).sum() / ss_xx
        intercept = ys_mean - slope * xs_mean
        lr_end[i] = intercept + slope * (n - 1)
        fitted    = intercept + slope * xs
        resid_ss  = ((ys - fitted) ** 2).sum()
        lr_se[i]  = np.sqrt(resid_ss / (n - 2)) if n > 2 else 0.0

    return lr_end, lr_se


def add_indicators(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d      = df.copy()
    closes = d["close"].to_numpy(dtype=float)

    lr_end_arr, lr_se_arr = linreg_rolling(closes, n)

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
    d["lr_end_s1"] = d["lr_end"].shift(1)
    d["lr_se_s1"]  = d["lr_se"].shift(1)
    d["atr_n_s1"]  = d["atr_n"].shift(1)
    d["ema200"]    = d["close"].ewm(span=EMA200_N, adjust=False).mean()

    return d


# =============================================================================
# TRADE DATACLASS
# =============================================================================

@dataclass
class Trade:
    trade_id:      int
    symbol:        str
    config:        str
    linreg_n:      int
    linreg_mult:   float
    stop_mult:     float
    chan_activate_r: float
    chan_atr_mult:   float
    entry_time:    str
    exit_time:     str
    entry_price:   float
    exit_price:    float
    initial_stop:  float
    initial_risk:  float
    exit_reason:   str
    bars_held:     int
    net_r:         float
    mfe_r:         float
    mae_r:         float
    chan_activated: bool


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_symbol(
    symbol: str,
    ind: pd.DataFrame,
    linreg_n: int,
    linreg_mult: float,
    stop_mult: float,
    chan_activate_r: float,
    chan_atr_mult: float,
    config_label: str,
    first_id: int,
) -> List[Trade]:
    warmup = EMA200_N + linreg_n + 5
    pos    = None
    trades: List[Trade] = []
    tid    = first_id

    for i in range(warmup, len(ind)):
        row = ind.iloc[i]

        close     = float(row["close"])
        high      = float(row["high"])
        low       = float(row["low"])
        t         = ind.index[i]
        lr_end    = float(row["lr_end"])
        atr_n     = float(row["atr_n"])
        ema200    = float(row["ema200"])
        lr_end_s1 = float(row["lr_end_s1"])
        lr_se_s1  = float(row["lr_se_s1"])
        atr_n_s1  = float(row["atr_n_s1"])

        if not all(math.isfinite(x) for x in
                   [lr_end, atr_n, ema200, lr_end_s1, lr_se_s1, atr_n_s1]):
            continue
        if lr_se_s1 <= 0 or atr_n <= 0:
            continue

        upper_band = lr_end_s1 + lr_se_s1 * linreg_mult

        if pos is not None:
            bars_held = i - pos["entry_i"]

            # Update running stats and HH
            pos["mfe_r"] = max(pos["mfe_r"], (high - pos["entry"]) / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (low  - pos["entry"]) / pos["risk"])
            pos["hh"]    = max(pos["hh"], high)

            exit_px = None
            reason  = None

            # Exit 1: initial stop (always first)
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                reason  = "initial_stop"
            # Exit 2: chandelier stop (only when active)
            elif pos["chan_active"] and low <= pos["chan_stop"]:
                exit_px = pos["chan_stop"]
                reason  = "chandelier_stop"
            # Exit 3: LinReg midline cross-under (only when chandelier inactive)
            elif not pos["chan_active"] and close < lr_end:
                exit_px = close
                reason  = "midline_exit"

            if exit_px is not None:
                net_r = (float(exit_px) - pos["entry"]) / pos["risk"]
                trades.append(Trade(
                    trade_id=tid,
                    symbol=symbol,
                    config=config_label,
                    linreg_n=linreg_n,
                    linreg_mult=linreg_mult,
                    stop_mult=stop_mult,
                    chan_activate_r=chan_activate_r,
                    chan_atr_mult=chan_atr_mult,
                    entry_time=str(pos["entry_time"]),
                    exit_time=str(t),
                    entry_price=float(pos["entry"]),
                    exit_price=float(exit_px),
                    initial_stop=float(pos["stop"]),
                    initial_risk=float(pos["risk"]),
                    exit_reason=reason,
                    bars_held=int(bars_held),
                    net_r=float(net_r),
                    mfe_r=float(pos["mfe_r"]),
                    mae_r=float(pos["mae_r"]),
                    chan_activated=bool(pos["chan_active"]),
                ))
                tid += 1
                pos = None
                continue

            # Update chandelier AFTER all exit checks (critical sequencing)
            if pos["mfe_r"] >= chan_activate_r:
                pos["chan_active"] = True
            if pos["chan_active"]:
                new_cs = pos["hh"] - atr_n * chan_atr_mult
                pos["chan_stop"] = max(pos["chan_stop"], new_cs)

        else:
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
                    hh=close,
                    mfe_r=0.0,
                    mae_r=0.0,
                    chan_active=False,
                    chan_stop=close - risk,
                )

    return trades


# =============================================================================
# STATS
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


def _stats(trades: List[Trade]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, avg_mfe_r=0.0,
                    chan_activated_pct=0.0)
    r   = np.array([t.net_r for t in trades], dtype=float)
    r   = r[np.isfinite(r)]
    mfe = np.array([t.mfe_r for t in trades], dtype=float)
    return dict(
        trades=int(len(r)),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        pf=_pf(r),
        max_dd_r=_dd(r),
        win_rate_pct=float(100.0 * (r > 0).mean()) if len(r) else 0.0,
        avg_mfe_r=float(mfe.mean()),
        chan_activated_pct=float(100.0 * sum(t.chan_activated for t in trades) / len(trades)),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 72)
    print("PHASE T3B -- LinearRegressionBreakout Wide-Exit / Chandelier")
    print("=" * 72)
    print(f"Chan activate grid: {CHAN_ACTIVATE_R_GRID}")
    print(f"Chan ATR-mult grid: {CHAN_ATR_MULT_GRID}")
    total_combos = len(CHAN_ACTIVATE_R_GRID) * len(CHAN_ATR_MULT_GRID)
    print(f"Total combos:       {total_combos} per config  ({total_combos * 2} total)")
    print(f"Cache:              {CACHE_DIR}")
    print()

    symbols = discover_symbols()
    print(f"Loading OHLCV data ({len(symbols)} symbols found)...")
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None:
            print(f"  SKIP {sym}")
            continue
        sym_data[sym] = df
    print(f"Loaded: {len(sym_data)} / {len(symbols)} symbols")
    print()

    all_variant_rows: List[dict] = []   # for combined variant summary across configs

    for cfg_label, t2 in T2_BASELINES.items():
        n      = t2["linreg_n"]
        lm     = t2["linreg_mult"]
        sm     = t2["stop_mult"]
        t2_avg = t2["avg_r"]
        t2_pf  = t2["pf"]
        t2_tot = t2["total_r"]

        print(f"{'='*72}")
        print(f"Config {cfg_label}  (N={n} / mult={lm} / sm={sm})")
        print(f"T2 baseline: trades={t2['trades']}  avg_r={t2_avg:+.4f}R  PF={t2_pf:.3f}  total_r={t2_tot:+.2f}R")
        print(f"{'='*72}")

        # Precompute indicators once per symbol for this N
        print(f"Precomputing indicators (N={n})...")
        ind_cache: Dict[str, pd.DataFrame] = {}
        for sym, df in sym_data.items():
            ind_cache[sym] = add_indicators(df, n)
        print(f"  {len(ind_cache)} symbols ready")
        print()

        variant_rows: List[dict] = []
        combo_idx = 0

        for act_r in CHAN_ACTIVATE_R_GRID:
            for cm in CHAN_ATR_MULT_GRID:
                combo_idx += 1
                combo_trades: List[Trade] = []
                tid = 0
                for sym in sym_data:
                    combo_trades.extend(
                        backtest_symbol(
                            sym, ind_cache[sym],
                            n, lm, sm, act_r, cm,
                            cfg_label, tid,
                        )
                    )
                    tid += len(combo_trades)

                s = _stats(combo_trades)
                delta_avg = s["avg_r"] - t2_avg
                delta_pf  = s["pf"]  - t2_pf
                tag = " BETTER" if (s["avg_r"] > t2_avg and s["pf"] > t2_pf) else "      "

                print(
                    f"  [{combo_idx:2d}/{total_combos}]"
                    f" act={act_r:.1f}R cm={cm:.1f}"
                    f"  trades={s['trades']:4d}"
                    f"  total_r={s['total_r']:+7.2f}R"
                    f"  avg_r={s['avg_r']:+.4f}R ({delta_avg:+.4f})"
                    f"  pf={s['pf']:.3f}"
                    f"  win%={s['win_rate_pct']:.1f}%"
                    f"  chan%={s['chan_activated_pct']:.0f}%"
                    f"  {tag}"
                )

                row = dict(
                    config=cfg_label,
                    linreg_n=n,
                    linreg_mult=lm,
                    stop_mult=sm,
                    chan_activate_r=act_r,
                    chan_atr_mult=cm,
                    **s,
                    vs_t2_avg_r=round(delta_avg, 4),
                    vs_t2_pf=round(delta_pf, 4),
                    beats_t2=(s["avg_r"] > t2_avg and s["pf"] > t2_pf),
                )
                variant_rows.append(row)
                all_variant_rows.append(row)

        # Best combo for this config
        vdf = pd.DataFrame(variant_rows).sort_values("total_r", ascending=False)
        best = vdf.iloc[0]

        print()
        print(f"  T2 baseline:  trades={t2['trades']}  avg_r={t2_avg:+.4f}R  PF={t2_pf:.3f}  total_r={t2_tot:+.2f}R")
        print(f"  Best T3B:     trades={int(best['trades'])}  avg_r={best['avg_r']:+.4f}R  PF={best['pf']:.3f}  total_r={best['total_r']:+.2f}R")
        beat = "BETTER" if (best["avg_r"] > t2_avg and best["pf"] > t2_pf) else "WORSE"
        print(f"  Best combo:   act={best['chan_activate_r']:.1f}R  cm={best['chan_atr_mult']:.1f}  -> {beat} than T2")
        print()

        # Save best combo trades
        best_trades: List[Trade] = []
        tid2 = 0
        for sym in sym_data:
            best_trades.extend(
                backtest_symbol(
                    sym, ind_cache[sym],
                    n, lm, sm,
                    float(best["chan_activate_r"]),
                    float(best["chan_atr_mult"]),
                    cfg_label, tid2,
                )
            )
            tid2 += len(best_trades)

        best_df = pd.DataFrame([asdict(t) for t in best_trades])

        # Fix 1970-epoch timestamps before saving
        for col in ["entry_time", "exit_time"]:
            if col in best_df.columns:
                best_df[col] = pd.to_datetime(best_df[col], utc=True,
                                              format="mixed", errors="coerce")
                if not best_df.empty and best_df[col].dt.year.max() < 2000:
                    ns_vals = best_df[col].astype("int64")
                    best_df[col] = pd.to_datetime(ns_vals, unit="ms", utc=True)

        prefix = f"phase_t3b_linregbreakout_cfg{cfg_label}"
        best_df.to_csv(OUT_DIR / f"{prefix}_best_trades.csv", index=False)
        vdf.to_csv(OUT_DIR / f"{prefix}_variant_summary.csv", index=False)

        # Equity for best combo
        if not best_df.empty:
            eq = best_df[["exit_time", "net_r", "exit_reason", "chan_activated"]].copy()
            eq = eq.sort_values("exit_time").reset_index(drop=True)
            eq["equity_r"] = eq["net_r"].cumsum()
            eq.to_csv(OUT_DIR / f"{prefix}_best_equity.csv", index=False)

        # Asset summary for best combo
        if not best_df.empty:
            asset_rows = []
            for sym, g in best_df.groupby("symbol"):
                r = g["net_r"].to_numpy(dtype=float)
                r = r[np.isfinite(r)]
                if not len(r):
                    continue
                asset_rows.append(dict(
                    symbol=sym, trades=len(r), total_r=float(r.sum()),
                    avg_r=float(r.mean()),
                    win_rate_pct=float(100.0 * (r > 0).mean()),
                    profit_factor=_pf(r),
                    avg_mfe_r=float(g["mfe_r"].mean()),
                ))
            if asset_rows:
                pd.DataFrame(asset_rows).sort_values("total_r", ascending=False).to_csv(
                    OUT_DIR / f"{prefix}_best_asset_summary.csv", index=False
                )

    # Combined variant summary
    combined_vdf = pd.DataFrame(all_variant_rows)
    combined_vdf.to_csv(OUT_DIR / "phase_t3b_linregbreakout_all_variants.csv", index=False)

    # Final summary
    print("=" * 72)
    print("T3B SUMMARY")
    print("=" * 72)
    for cfg_label, t2 in T2_BASELINES.items():
        t2_avg = t2["avg_r"]
        t2_pf  = t2["pf"]
        cfg_rows = combined_vdf[combined_vdf["config"] == cfg_label]
        best5 = cfg_rows.sort_values("total_r", ascending=False).head(5)
        print(f"  Config {cfg_label} T2 baseline:  trades={t2['trades']}  avg_r={t2_avg:+.4f}R  PF={t2_pf:.3f}  total_r={t2['total_r']:+.2f}R")
        print(f"  Config {cfg_label} top 5 T3B combos by total_r:")
        for _, row in best5.iterrows():
            beat = "BETTER" if (row["avg_r"] > t2_avg and row["pf"] > t2_pf) else "WORSE "
            print(
                f"    act={row['chan_activate_r']:.1f}R  cm={row['chan_atr_mult']:.1f}"
                f"  total_r={row['total_r']:+7.2f}R"
                f"  avg_r={row['avg_r']:+.4f}R"
                f"  pf={row['pf']:.3f}"
                f"  [{beat}]"
            )
        any_better = cfg_rows["beats_t2"].any()
        if any_better:
            print(f"  Config {cfg_label}: chandelier IMPROVES on T2 for some combos")
        else:
            print(f"  Config {cfg_label}: chandelier DOES NOT improve on T2 (midline exit is canonical)")
        print()

    print("=" * 72)
    print("T3B COMPLETE -- awaiting human review before T4")
    print("=" * 72)
    print(f"  Outputs: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
