#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- LinearRegressionBreakout Concept Discovery

Entry:   Close > LinReg(N)[i-1] + mult * StdErr(N)[i-1]  (shifted 1 bar, no lookahead)
         LinReg = least-squares regression line fitted to last N closes, end-point value
         StdErr = standard error of regression residuals

Filter:  ema200_price:          close > EMA(200)
         linreg_slope_positive: slope of LinReg(N)[i-1] > 0

Stop:    entry - ATR(N) * stop_mult  (below entry, long)
Exit 1:  low <= stop  -> exit at stop
Exit 2:  close < LinReg(N)  -> exit at close  (midline cross-under, no shift for exit)

Timeframes: 1d, 4h, 8h
  1d: reuses data/research_trend_t15/ohlcv_cache/
  4h: downloads to data/research_linregbreakout_t1/ohlcv_cache/
  8h: downloads to data/research_linregbreakout_t1/ohlcv_cache/

Stability zone: canonical N +/-1 grid step, PASS >= 67%
Cost floor:     avg_r > 0.15R  (Section 4.2, Binance Spot)
Output:         data/research_linregbreakout_t1/
"""

from __future__ import annotations

import math
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("Install ccxt:  pip install ccxt")

# =============================================================================
# CONFIG
# =============================================================================

ROOT         = Path(__file__).resolve().parent
CACHE_1D     = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
CACHE_INTRA  = ROOT / "data" / "research_linregbreakout_t1" / "ohlcv_cache"
OUT_DIR      = ROOT / "data" / "research_linregbreakout_t1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INTRA.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["1d", "4h", "8h"]

# Target bars per timeframe (approx 4 years of history)
TF_TARGET_BARS = {"1d": 1500, "4h": 6000, "8h": 4500}
TF_BAR_MS      = {"1d": 86_400_000, "4h": 14_400_000, "8h": 28_800_000}

LINREG_NS    = [10, 15, 20, 25, 30, 40, 55]
LINREG_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_MULTS   = [2.0, 3.0]
FILTER_MODES = ["ema200_price", "linreg_slope_positive"]

EMA200_N = 200

COST_FLOOR_R       = 0.15
STABILITY_PASS_PCT = 67.0

# Minimum bars to accept a loaded symbol (warmup buffer)
TF_MIN_BARS = {"1d": 300, "4h": 1200, "8h": 900}

EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI", "WBTC", "BTTC",
}
EXCLUDE_KW = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")


# =============================================================================
# DATA LOADING / DOWNLOAD
# =============================================================================

def _init_exchange() -> "ccxt.binance":
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })


def _stem_to_symbol(stem: str, tf: str) -> Optional[str]:
    parts = stem.split("_")
    if parts and parts[-1] == tf:
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    quote = parts[-1]
    base  = "_".join(parts[:-1])
    if quote != "USDT":
        return None
    if base in EXCLUDE_BASES:
        return None
    if any(kw in base for kw in EXCLUDE_KW):
        return None
    return f"{base}/{quote}"


def discover_symbols_1d() -> List[str]:
    syms = []
    for f in sorted(CACHE_1D.glob("*_1d.csv")):
        sym = _stem_to_symbol(f.stem, "1d")
        if sym:
            syms.append(sym)
    return syms


def _fetch_paginated(exchange: "ccxt.binance", symbol: str, tf: str,
                     target_bars: int) -> Optional[pd.DataFrame]:
    bar_ms   = TF_BAR_MS[tf]
    since_ms = int(time_mod.time() * 1000) - target_bars * bar_ms
    all_rows: List[list] = []
    max_pages = (target_bars // 1000) + 3

    for _ in range(max_pages):
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=1000)
            time_mod.sleep(exchange.rateLimit / 1000.0)
        except Exception:
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts  = rows[-1][0]
        since_ms = last_ts + bar_ms
        if since_ms > int(time_mod.time() * 1000):
            break
        if len(rows) < 1000:
            break

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows,
                      columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def _load_csv(path: Path, tf: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                             errors="coerce", format="mixed")
            df = df.set_index("timestamp").sort_index()
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce", format="mixed")
            df = df.sort_index()
        # Fix 1970-epoch bug if timestamps were stored as ms integers then read as ns
        if not df.empty and df.index.year.max() < 2000:
            ns_vals = df.index.astype("int64")
            df.index = pd.to_datetime(ns_vals, unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
        min_cached = TF_TARGET_BARS[tf] // 2
        if len(df) < min_cached:
            return None
        return df.tail(TF_TARGET_BARS[tf])
    except Exception:
        return None


def load_or_download(symbol: str, tf: str,
                     exchange: Optional["ccxt.binance"] = None) -> Optional[pd.DataFrame]:
    fname = symbol.replace("/", "_") + f"_{tf}.csv"
    path  = (CACHE_1D if tf == "1d" else CACHE_INTRA) / fname

    if path.exists():
        df = _load_csv(path, tf)
        if df is not None and len(df) >= TF_MIN_BARS[tf]:
            return df

    if tf == "1d" or exchange is None:
        return None

    df = _fetch_paginated(exchange, symbol, tf, TF_TARGET_BARS[tf])
    if df is None or len(df) < TF_MIN_BARS[tf]:
        return None

    try:
        df.to_csv(path)
    except Exception:
        pass

    return df.tail(TF_TARGET_BARS[tf])


# =============================================================================
# LINEAR REGRESSION (rolling, vectorized inner ops)
# =============================================================================

def linreg_rolling(values: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling OLS line (last N closes) and return arrays of:
      lr_end   -- endpoint value of the regression line (at the last bar of window)
      lr_se    -- standard error of regression residuals
      lr_slope -- slope coefficient (positive = uptrend)

    x is always [0, 1, ..., N-1] so x-statistics are precomputed constants.
    """
    nbar    = len(values)
    xs      = np.arange(n, dtype=float)
    xs_mean = (n - 1) / 2.0
    ss_xx   = ((xs - xs_mean) ** 2).sum()   # = n*(n^2-1)/12

    lr_end   = np.full(nbar, np.nan)
    lr_se    = np.full(nbar, np.nan)
    lr_slope = np.full(nbar, np.nan)

    for i in range(n - 1, nbar):
        ys = values[i - n + 1 : i + 1]
        if not np.all(np.isfinite(ys)):
            continue
        ys_mean  = ys.mean()
        ss_xy    = ((xs - xs_mean) * (ys - ys_mean)).sum()
        slope    = ss_xy / ss_xx
        intercept = ys_mean - slope * xs_mean
        # Endpoint: regression value at x=N-1
        lr_end[i]   = intercept + slope * (n - 1)
        # Standard error of residuals
        fitted      = intercept + slope * xs
        resid_ss    = ((ys - fitted) ** 2).sum()
        lr_se[i]    = np.sqrt(resid_ss / (n - 2)) if n > 2 else 0.0
        lr_slope[i] = slope

    return lr_end, lr_se, lr_slope


def compute_indicators(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute LinReg channel indicators for period N plus ATR(N) and EMA(200).
    Shifted-1-bar (_s1) columns are used for entry signals (no lookahead).
    Current-bar lr_end is used for exit midline check (bar already closed).
    """
    d      = df.copy()
    closes = d["close"].to_numpy(dtype=float)

    lr_end_arr, lr_se_arr, lr_slope_arr = linreg_rolling(closes, n)

    # ATR(N) -- Wilder simple rolling
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean().to_numpy(dtype=float)

    # Store current-bar values
    d["lr_end"]   = lr_end_arr
    d["lr_se"]    = lr_se_arr
    d["lr_slope"] = lr_slope_arr
    d["atr_n"]    = atr_n

    # Shift 1 bar for entry logic (no lookahead)
    d["lr_end_s1"]   = d["lr_end"].shift(1)
    d["lr_se_s1"]    = d["lr_se"].shift(1)
    d["lr_slope_s1"] = d["lr_slope"].shift(1)
    d["atr_n_s1"]    = d["atr_n"].shift(1)

    # Regime filter: EMA(200)
    d["ema200"] = d["close"].ewm(span=EMA200_N, adjust=False).mean()

    return d


# =============================================================================
# BACKTESTER
# =============================================================================

def _make_trade(
    symbol: str, pos: dict, t, exit_price: float, net_r: float,
    reason: str, n: int, linreg_mult: float, stop_mult: float,
    filter_mode: str, tf: str,
) -> dict:
    return {
        "timeframe":    tf,
        "symbol":       symbol,
        "filter_mode":  filter_mode,
        "linreg_n":     n,
        "linreg_mult":  linreg_mult,
        "stop_mult":    stop_mult,
        "entry_time":   pos["entry_time"],
        "exit_time":    t,
        "entry_price":  pos["entry"],
        "exit_price":   exit_price,
        "initial_stop": pos["stop"],
        "exit_reason":  reason,
        "net_r":        net_r,
    }


def backtest_symbol(
    symbol: str,
    ind: pd.DataFrame,
    linreg_mult: float,
    stop_mult: float,
    filter_mode: str,
    n: int,
    tf: str,
) -> List[dict]:
    warmup = EMA200_N + n + 5
    pos    = None
    trades = []

    for i in range(warmup, len(ind)):
        row = ind.iloc[i]

        # Current-bar values
        close      = float(row["close"])
        high       = float(row["high"])
        low        = float(row["low"])
        t          = ind.index[i]
        lr_end     = float(row["lr_end"])    # for midline exit
        atr_n      = float(row["atr_n"])
        ema200     = float(row["ema200"])

        # Previous-bar values (entry signal, no lookahead)
        lr_end_s1   = float(row["lr_end_s1"])
        lr_se_s1    = float(row["lr_se_s1"])
        lr_slope_s1 = float(row["lr_slope_s1"])
        atr_n_s1    = float(row["atr_n_s1"])

        # Skip if any required indicator is non-finite
        if not all(math.isfinite(x) for x in
                   [lr_end, lr_se_s1, lr_end_s1, atr_n, atr_n_s1, ema200]):
            continue
        if lr_se_s1 <= 0:
            continue

        upper_band = lr_end_s1 + lr_se_s1 * linreg_mult

        if pos is not None:
            # Exit 1: stop hit
            if low <= pos["stop"]:
                net_r = (pos["stop"] - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, pos["stop"], net_r, "stop",
                    n, linreg_mult, stop_mult, filter_mode, tf,
                ))
                pos = None
                continue

            # Exit 2: close drops below LinReg midline
            if close < lr_end:
                net_r = (close - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "midline",
                    n, linreg_mult, stop_mult, filter_mode, tf,
                ))
                pos = None
                continue

        else:
            # Regime filter
            if filter_mode == "ema200_price":
                if close <= ema200:
                    continue
            elif filter_mode == "linreg_slope_positive":
                if lr_slope_s1 <= 0:
                    continue

            # Entry: close breaks above upper channel band (from prior bar)
            if close > upper_band:
                risk = atr_n_s1 * stop_mult
                if risk <= 0:
                    continue
                stop = close - risk
                pos  = {
                    "entry":      close,
                    "stop":       stop,
                    "risk":       risk,
                    "entry_time": t,
                }

    # Close open position at last bar
    if pos is not None:
        i_last = len(ind) - 1
        close_last = float(ind["close"].iloc[i_last])
        t_last     = ind.index[i_last]
        net_r = (close_last - pos["entry"]) / pos["risk"]
        trades.append(_make_trade(
            symbol, pos, t_last, close_last, net_r, "end_of_data",
            n, linreg_mult, stop_mult, filter_mode, tf,
        ))

    return trades


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def _zone_for_n(n: int) -> List[int]:
    """Return [N_prev, N, N_next] from LINREG_NS grid for N +/-1 step."""
    idx = LINREG_NS.index(n)
    zone = [LINREG_NS[idx]]
    if idx > 0:
        zone.insert(0, LINREG_NS[idx - 1])
    if idx < len(LINREG_NS) - 1:
        zone.append(LINREG_NS[idx + 1])
    return zone


def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["timeframe", "linreg_mult", "stop_mult", "filter_mode"]

    for keys, g in grid.groupby(groups):
        tf, lm, sm, fm = keys
        # Per-N summary
        n_summary = (
            g.groupby("linreg_n")
            .agg(trades=("net_r", "size"), total_r=("net_r", "sum"),
                 avg_r=("net_r", "mean"))
            .reset_index()
        )
        if n_summary.empty:
            continue

        # Canonical N = highest avg_r
        best_row    = n_summary.loc[n_summary["avg_r"].idxmax()]
        canonical_n = int(best_row["linreg_n"])
        zone_ns     = _zone_for_n(canonical_n)
        zone_rows   = n_summary[n_summary["linreg_n"].isin(zone_ns)]

        profitable   = int((zone_rows["avg_r"] > 0).sum())
        zone_size    = len(zone_rows)
        zone_pass_pct = (profitable / zone_size * 100) if zone_size > 0 else 0.0
        stability    = "PASS" if zone_pass_pct >= STABILITY_PASS_PCT else "FAIL"

        best_avg_r   = float(best_row["avg_r"])
        cost_gate    = "PASS" if best_avg_r > COST_FLOOR_R else "FAIL"

        # N=20 (central zone anchor) win pct
        n20_rows     = n_summary[n_summary["linreg_n"] == 20]
        n20_avg_r    = float(n20_rows["avg_r"].values[0]) if len(n20_rows) > 0 else float("nan")

        rows.append(dict(
            timeframe=tf,
            linreg_mult=lm,
            stop_mult=sm,
            filter_mode=fm,
            canonical_n=canonical_n,
            zone_ns=str(zone_ns),
            zone_profitable=profitable,
            zone_total=zone_size,
            zone_pass_pct=round(zone_pass_pct, 1),
            stability=stability,
            best_avg_r=round(best_avg_r, 4),
            best_total_r=round(float(best_row["total_r"]), 2),
            best_trades=int(best_row["trades"]),
            cost_gate=cost_gate,
            n20_avg_r=round(n20_avg_r, 4),
            overall_gate="PASS" if (stability == "PASS" and cost_gate == "PASS") else "FAIL",
        ))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["overall_gate", "best_avg_r"],
                        ascending=[True, False]).reset_index(drop=True)
    return df


# =============================================================================
# N SENSITIVITY REPORT
# =============================================================================

def n_sensitivity(grid: pd.DataFrame) -> pd.DataFrame:
    groups = ["timeframe", "linreg_mult", "stop_mult", "filter_mode", "linreg_n"]
    rows = []
    for keys, g in grid.groupby(groups):
        tf, lm, sm, fm, n = keys
        arr = g["net_r"].to_numpy(dtype=float)
        wins = int((arr > 0).sum())
        losses = int((arr <= 0).sum())
        gross_win  = float(arr[arr > 0].sum()) if wins > 0 else 0.0
        gross_loss = float(-arr[arr <= 0].sum()) if losses > 0 else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        rows.append(dict(
            timeframe=tf, linreg_n=n, linreg_mult=lm, stop_mult=sm, filter_mode=fm,
            trades=len(arr), total_r=round(float(arr.sum()), 3),
            avg_r=round(float(arr.mean()), 4),
            profit_factor=round(pf, 3),
            win_rate_pct=round(wins / len(arr) * 100, 1) if len(arr) > 0 else 0.0,
        ))
    return pd.DataFrame(rows).sort_values(
        ["timeframe", "filter_mode", "linreg_mult", "stop_mult", "linreg_n"]
    ).reset_index(drop=True)


# =============================================================================
# REPORT WRITER
# =============================================================================

def write_report(grid: pd.DataFrame, sa: pd.DataFrame,
                 sym_count: Dict[str, int]) -> None:
    report_path = OUT_DIR / "phase_t1_linregbreakout_summary.txt"
    lines = []

    lines += [
        "=" * 72,
        "PHASE T1 -- LinearRegressionBreakout Concept Discovery",
        "=" * 72,
        f"Timeframes tested: {TIMEFRAMES}",
        f"N grid:            {LINREG_NS}",
        f"mult grid:         {LINREG_MULTS}",
        f"stop_mult grid:    {STOP_MULTS}",
        f"Filter modes:      {FILTER_MODES}",
        f"Cost floor:        avg_r > {COST_FLOOR_R}R  (Binance Spot)",
        f"Stability gate:    >= {STABILITY_PASS_PCT}% of zone combos profitable",
        "",
    ]

    for tf in TIMEFRAMES:
        tf_sa = sa[sa["timeframe"] == tf] if not sa.empty else pd.DataFrame()
        n_syms = sym_count.get(tf, 0)
        tf_grid = grid[grid["timeframe"] == tf]
        total_combos = (tf_grid[["linreg_n", "linreg_mult", "stop_mult", "filter_mode"]]
                        .drop_duplicates().shape[0])

        lines += [
            "-" * 72,
            f"TIMEFRAME: {tf}  ({n_syms} symbols, {total_combos} param combos)",
            "-" * 72,
        ]

        if tf_sa.empty:
            lines.append("  No stability data (no symbols loaded or all below MIN_BARS)")
            lines.append("")
            continue

        pass_rows = tf_sa[tf_sa["overall_gate"] == "PASS"] if not tf_sa.empty else pd.DataFrame()

        if pass_rows.empty:
            lines.append("  No combos passed stability + cost floor gate")
        else:
            lines.append(f"  PASS combos: {len(pass_rows)} / {len(tf_sa)}")
            lines.append("")
            lines.append("  Top PASS combos by best_avg_r:")
            for _, row in pass_rows.head(10).iterrows():
                lines.append(
                    f"    fm={row['filter_mode']:<24}  mult={row['linreg_mult']:.1f}"
                    f"  sm={row['stop_mult']:.1f}  N={row['canonical_n']}"
                    f"  avg_r={row['best_avg_r']:+.4f}R"
                    f"  zone={row['zone_pass_pct']:.0f}%"
                    f"  trades={row['best_trades']}"
                    f"  [{row['stability']}/{row['cost_gate']}]"
                )

        lines.append("")
        lines.append("  FAIL summary (stability or cost gate):")
        fail_rows = tf_sa[tf_sa["overall_gate"] == "FAIL"]
        if fail_rows.empty:
            lines.append("    None")
        else:
            for _, row in fail_rows.head(5).iterrows():
                lines.append(
                    f"    fm={row['filter_mode']:<24}  mult={row['linreg_mult']:.1f}"
                    f"  sm={row['stop_mult']:.1f}  N={row['canonical_n']}"
                    f"  avg_r={row['best_avg_r']:+.4f}R"
                    f"  zone={row['zone_pass_pct']:.0f}%"
                    f"  [{row['stability']}/{row['cost_gate']}]"
                )
        lines.append("")

    # Overall best candidate
    if not sa.empty:
        all_pass = sa[sa["overall_gate"] == "PASS"]
        lines += ["=" * 72, "RECOMMENDED CANONICAL CONFIG (best overall PASS)", "=" * 72]
        if all_pass.empty:
            lines.append("  NONE -- no combo passed both stability and cost floor")
            lines.append("  HALT: LinearRegressionBreakout has no edge on this universe/period.")
        else:
            best = all_pass.sort_values("best_avg_r", ascending=False).iloc[0]
            lines += [
                f"  Timeframe:    {best['timeframe']}",
                f"  Filter:       {best['filter_mode']}",
                f"  N (canonical):{best['canonical_n']}",
                f"  LinReg mult:  {best['linreg_mult']}",
                f"  Stop mult:    {best['stop_mult']}",
                f"  Avg R/trade:  {best['best_avg_r']:+.4f}R",
                f"  Total R:      {best['best_total_r']:+.2f}R",
                f"  Trades:       {best['best_trades']}",
                f"  Zone PASS:    {best['zone_pass_pct']:.0f}%",
                "",
                "  NOTE: Review stability ranking CSV before locking T2 config.",
                "  Do NOT proceed to T2 without human review.",
            ]

    lines.append("=" * 72)
    report_path.write_text("\n".join(lines), encoding="ascii", errors="replace")
    print(f"  Report: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 72)
    print("PHASE T1 -- LinearRegressionBreakout Concept Discovery")
    print("=" * 72)
    print(f"Timeframes:  {TIMEFRAMES}")
    print(f"N grid:      {LINREG_NS}")
    print(f"mult grid:   {LINREG_MULTS}")
    print(f"stop grid:   {STOP_MULTS}")
    print(f"Filters:     {FILTER_MODES}")
    print(f"Cache (1d):  {CACHE_1D}")
    print(f"Cache (sub): {CACHE_INTRA}")
    print()

    all_trades: List[dict] = []
    sym_count:  Dict[str, int] = {}

    symbols_1d  = discover_symbols_1d()
    exchange    = _init_exchange()   # used for 4H/8H downloads

    for tf in TIMEFRAMES:
        target_bars = TF_TARGET_BARS[tf]
        min_bars    = TF_MIN_BARS[tf]
        total_combos = (len(LINREG_NS) * len(LINREG_MULTS) *
                        len(STOP_MULTS) * len(FILTER_MODES))

        print(f"--- Timeframe: {tf}  (target={target_bars} bars, "
              f"{total_combos} combos) ---")

        # Load OHLCV for this TF
        sym_data: Dict[str, pd.DataFrame] = {}
        skipped = 0
        for sym in symbols_1d:
            df = load_or_download(sym, tf, exchange if tf != "1d" else None)
            if df is None or len(df) < min_bars:
                skipped += 1
                if tf != "1d":
                    print(f"  SKIP {sym}")
                continue
            sym_data[sym] = df

        n_loaded = len(sym_data)
        sym_count[tf] = n_loaded
        print(f"  Loaded: {n_loaded} symbols  (skipped {skipped})")

        if n_loaded == 0:
            print(f"  WARNING: no symbols for {tf} -- skipping timeframe")
            continue

        # Precompute indicators per (symbol, N)
        print(f"  Precomputing LinReg indicators...")
        ind_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
        for sym, df in sym_data.items():
            for n in LINREG_NS:
                ind_cache[(sym, n)] = compute_indicators(df, n)
        print(f"    {n_loaded * len(LINREG_NS)} indicator sets ready")

        # Run all combos
        combo_idx = 0
        total_c   = len(LINREG_NS) * len(LINREG_MULTS) * len(STOP_MULTS) * len(FILTER_MODES)
        for n in LINREG_NS:
            for lm in LINREG_MULTS:
                for sm in STOP_MULTS:
                    for fm in FILTER_MODES:
                        combo_idx += 1
                        combo_trades: List[dict] = []
                        for sym in sym_data:
                            ind = ind_cache[(sym, n)]
                            combo_trades.extend(
                                backtest_symbol(sym, ind, lm, sm, fm, n, tf)
                            )

                        all_trades.extend(combo_trades)

                        # Quick summary line
                        if combo_trades:
                            arr = np.asarray([t["net_r"] for t in combo_trades])
                            total_r = arr.sum()
                            avg_r   = arr.mean()
                            pf_g    = arr[arr > 0].sum()
                            pf_l    = -arr[arr < 0].sum()
                            pf      = pf_g / pf_l if pf_l > 0 else float("inf")
                            flag    = ("*PASS" if avg_r > COST_FLOOR_R
                                       else "     ")
                        else:
                            total_r = avg_r = 0.0
                            pf      = 0.0
                            flag    = ""

                        print(
                            f"  [{combo_idx:3d}/{total_c}] "
                            f"N={n:<2}  mult={lm:.1f}  sm={sm:.1f}  "
                            f"fm={fm:<24}  "
                            f"trades={len(combo_trades):4d}  "
                            f"total_r={total_r:+7.2f}R  "
                            f"avg_r={avg_r:+.4f}R  "
                            f"pf={pf:.3f}  {flag}"
                        )

        print()

    if not all_trades:
        print("ERROR: no trades generated. Check data and config.")
        return 1

    grid = pd.DataFrame(all_trades)

    # Stability analysis
    sa = stability_analysis(grid)

    # N sensitivity table
    n_sens = n_sensitivity(grid)

    # Write outputs
    grid.to_csv(OUT_DIR / "phase_t1_linregbreakout_all_trades.csv", index=False)
    n_sens.to_csv(OUT_DIR / "phase_t1_linregbreakout_n_sensitivity.csv", index=False)

    if not sa.empty:
        sa.to_csv(OUT_DIR / "phase_t1_linregbreakout_stability_ranking.csv", index=False)

    write_report(grid, sa, sym_count)

    # Print stability ranking to console
    print("=" * 72)
    print("STABILITY RANKING  (sorted: PASS first, then by avg_r)")
    print("=" * 72)
    if sa.empty:
        print("  No stability data.")
    else:
        print(sa.to_string(index=False))

    print()
    print("=" * 72)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 72)
    print(f"  Trades file: {OUT_DIR / 'phase_t1_linregbreakout_all_trades.csv'}")
    print(f"  Stability:   {OUT_DIR / 'phase_t1_linregbreakout_stability_ranking.csv'}")
    print(f"  N-sens:      {OUT_DIR / 'phase_t1_linregbreakout_n_sensitivity.csv'}")
    print(f"  Report:      {OUT_DIR / 'phase_t1_linregbreakout_summary.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
