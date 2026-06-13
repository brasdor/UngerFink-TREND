#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track A -- RSI MR Universe Filtering
UngerFink Pipeline / Andrea Unger Methodology (Section 17.1 pattern)

Steps 1-3 only -- STOPS for human review before T3-T4.

Step 1: Run T2 per-symbol using frozen RSI MR 1D config
Step 2: Filter symbols:
          avg_r > 0
          trades >= 8
          top_month_pct < 0.60  (no single month > 60% of symbol total_r)
Step 3: Show filtered list + pooled stats vs full universe

Frozen config:
  Timeframe : 1D
  RSI window: 14
  Oversold  : 25
  Exit bars : 20  (fixed time exit -- primary)
  ATR mult  : 3.0 (safety stop)
  Filter    : none (validated canonical)

Output: data/research_meanreversionrsi_universe_filtered/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"


def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_meanreversionrsi_universe_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen RSI MR 1D config
RSI_N      = 14
OVERSOLD   = 25
TIME_EXIT  = 20
ATR_MULT   = 3.0
TIMEFRAME  = "1d"
MAX_BARS   = 2000
MIN_BARS   = 200

FILTER_MIN_TRADES   = 8
FILTER_MIN_AVG_R    = 0.0
FILTER_MAX_TOP_MONTH= 0.60


# =============================================================================
# DATA
# =============================================================================

def scan_symbols() -> list[str]:
    files = sorted(RAW_DIR.glob(f"*_{TIMEFRAME}.csv"))
    return [f.stem.replace(f"_{TIMEFRAME}", "") for f in files]


def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{symbol}_{TIMEFRAME}.csv"
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
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:].reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def calc_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    avg_g = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_l = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_symbol(df: pd.DataFrame, symbol: str) -> list[dict]:
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"], RSI_N)
    df["atr"] = calc_atr(df["high"], df["low"], df["close"], 14)
    df = df.dropna(subset=["rsi", "atr"]).reset_index(drop=True)
    if len(df) < 50:
        return []

    close = df["close"].values
    low_v = df["low"].values
    rsi   = df["rsi"].values
    atr   = df["atr"].values
    ts    = df["timestamp"].values

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]):
            continue
        if not in_pos:
            if rsi[i] < OVERSOLD:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price - ATR_MULT * atr[i]
        else:
            bars_held = i - e_bar
            exit_price = None
            reason     = None
            if low_v[i] <= stop:
                exit_price = stop
                reason     = "atr_stop"
            elif bars_held >= TIME_EXIT:
                exit_price = close[i]
                reason     = "time_exit"
            if reason is not None:
                risk = e_price - stop
                if risk > 1e-9:
                    r_mult   = (exit_price - e_price) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    ym       = f"{entry_dt.year}-{entry_dt.month:02d}"
                    trades.append({
                        "symbol":     symbol,
                        "net_r":      round(float(r_mult), 4),
                        "win":        int(r_mult > 0),
                        "year":       entry_dt.year,
                        "year_month": ym,
                    })
                in_pos = False
    return trades


# =============================================================================
# PER-SYMBOL STATS
# =============================================================================

def symbol_stats(symbol: str, trades: list[dict]) -> dict:
    if not trades:
        return {
            "symbol": symbol, "n_trades": 0,
            "avg_r": 0.0, "total_r": 0.0, "win_rate": 0.0,
            "max_dd_r": 0.0, "top_month_r": 0.0, "top_month_pct": 1.0,
            "top_month": "", "pass_filter": False, "skip_reason": "no_trades",
        }
    rs      = [t["net_r"] for t in trades]
    rs_arr  = np.array(rs)
    wins    = rs_arr[rs_arr > 0]
    total_r = float(rs_arr.sum())
    avg_r   = float(rs_arr.mean())
    wr      = float(len(wins) / len(rs_arr))

    cum  = np.cumsum(rs_arr)
    peak = np.maximum.accumulate(cum)
    dd   = float(np.max(peak - cum)) if len(cum) > 0 else 0.0

    # Top month concentration
    monthly: dict[str, float] = {}
    for t in trades:
        ym = t["year_month"]
        monthly[ym] = monthly.get(ym, 0.0) + t["net_r"]

    if monthly and total_r > 0:
        top_m_key = max(monthly, key=lambda k: monthly[k])
        top_m_r   = monthly[top_m_key]
        top_m_pct = top_m_r / total_r
    else:
        top_m_key = ""
        top_m_r   = 0.0
        top_m_pct = 1.0

    # Filter logic
    if len(trades) < FILTER_MIN_TRADES:
        skip = "min_trades"
        pass_f = False
    elif avg_r <= FILTER_MIN_AVG_R:
        skip = "avg_r<=0"
        pass_f = False
    elif top_m_pct >= FILTER_MAX_TOP_MONTH:
        skip = f"top_month_pct={top_m_pct:.2f}"
        pass_f = False
    else:
        skip = ""
        pass_f = True

    return {
        "symbol":        symbol,
        "n_trades":      len(trades),
        "avg_r":         round(avg_r, 4),
        "total_r":       round(total_r, 2),
        "win_rate":      round(wr, 4),
        "max_dd_r":      round(dd, 2),
        "top_month_r":   round(top_m_r, 2),
        "top_month_pct": round(top_m_pct, 3),
        "top_month":     top_m_key,
        "pass_filter":   pass_f,
        "skip_reason":   skip,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Track A -- RSI MR Universe Filtering (Steps 1-3)")
    p(f"  Config    : rsi{RSI_N}/os{OVERSOLD}/t{TIME_EXIT}/atr{ATR_MULT}/none")
    p(f"  Filter    : avg_r>0  trades>={FILTER_MIN_TRADES}  top_month<{FILTER_MAX_TOP_MONTH}")
    p("  STOPS after Step 3 -- awaiting human review")
    p("=" * 70)

    symbols = scan_symbols()
    p(f"\n  Symbols in raw_trend_t1/: {len(symbols)}")

    rows: list[dict] = []
    skipped_data = 0

    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None:
            skipped_data += 1
            continue
        trades = backtest_symbol(df, sym)
        rows.append(symbol_stats(sym, trades))

    p(f"  Symbols loaded       : {len(rows)}")
    p(f"  Skipped (no data)    : {skipped_data}")

    stats_df = pd.DataFrame(rows).sort_values("total_r", ascending=False).reset_index(drop=True)
    stats_df.to_csv(OUT_DIR / "phase_universe_per_symbol_stats.csv", index=False)

    # ---------- Full universe aggregate ----------
    valid = stats_df[stats_df["n_trades"] > 0]
    all_trades_flat: list[dict] = []
    for sym in valid["symbol"].tolist():
        df = load_ohlcv(sym)
        if df is not None:
            all_trades_flat.extend(backtest_symbol(df, sym))

    full_rs  = np.array([t["net_r"] for t in all_trades_flat]) if all_trades_flat else np.array([])
    full_wr  = float((full_rs > 0).mean()) if len(full_rs) > 0 else 0.0
    full_avg = float(full_rs.mean()) if len(full_rs) > 0 else 0.0
    full_tot = float(full_rs.sum()) if len(full_rs) > 0 else 0.0

    p(f"\n  --- Step 2: Filter Results ---")
    passed = stats_df[stats_df["pass_filter"]]
    failed = stats_df[~stats_df["pass_filter"]]

    p(f"  Symbols PASS filter  : {len(passed)}")
    p(f"  Symbols FAIL filter  : {len(failed)}")

    if failed.shape[0] > 0:
        fail_counts = failed["skip_reason"].value_counts()
        for reason, cnt in fail_counts.items():
            p(f"    {reason:30s}: {cnt}")

    # ---------- Filtered universe aggregate ----------
    filtered_trades: list[dict] = []
    for sym in passed["symbol"].tolist():
        df = load_ohlcv(sym)
        if df is not None:
            filtered_trades.extend(backtest_symbol(df, sym))

    filt_rs  = np.array([t["net_r"] for t in filtered_trades]) if filtered_trades else np.array([])
    filt_wr  = float((filt_rs > 0).mean()) if len(filt_rs) > 0 else 0.0
    filt_avg = float(filt_rs.mean()) if len(filt_rs) > 0 else 0.0
    filt_tot = float(filt_rs.sum()) if len(filt_rs) > 0 else 0.0

    if len(filt_rs) > 0:
        filt_cum  = np.cumsum(filt_rs)
        filt_peak = np.maximum.accumulate(filt_cum)
        filt_dd   = float(np.max(filt_peak - filt_cum))
    else:
        filt_dd = 0.0

    p(f"\n  --- Step 3: Pooled Stats Comparison ---")
    p(f"  {'Metric':20s}  {'Full universe':>15}  {'Filtered':>15}")
    p(f"  {'-'*55}")
    p(f"  {'Symbols':20s}  {len(valid):>15}  {len(passed):>15}")
    p(f"  {'Trades':20s}  {len(full_rs):>15}  {len(filt_rs):>15}")
    p(f"  {'Win rate':20s}  {full_wr*100:>14.1f}%  {filt_wr*100:>14.1f}%")
    p(f"  {'Avg R':20s}  {full_avg:>+14.4f}R  {filt_avg:>+14.4f}R")
    p(f"  {'Total R':20s}  {full_tot:>+14.2f}R  {filt_tot:>+14.2f}R")
    p(f"  {'Max DD R':20s}  {'--':>15}  {filt_dd:>+14.2f}R")

    # ---------- Filtered symbol list ----------
    p(f"\n  --- Filtered Symbol List ({len(passed)} symbols) ---")
    p(f"  {'Symbol':20s}  {'Trades':>6}  {'AvgR':>8}  {'TotR':>8}  "
      f"{'WR%':>6}  {'TopMoPct':>9}  {'TopMo':>8}")
    p("  " + "-" * 75)
    for _, row in passed.iterrows():
        p(f"  {row['symbol']:20s}  {int(row['n_trades']):>6}  "
          f"{row['avg_r']:>+7.4f}R  {row['total_r']:>+7.2f}R  "
          f"{row['win_rate']*100:>5.1f}%  {row['top_month_pct']:>8.2f}  "
          f"{row['top_month']:>8}")

    # ---------- Rejected symbol summary ----------
    p(f"\n  --- Rejected Symbols (top R among rejects) ---")
    failed_sorted = failed.sort_values("total_r", ascending=False)
    for _, row in failed_sorted.head(20).iterrows():
        p(f"  {row['symbol']:20s}  n={int(row['n_trades']):>4}  "
          f"avg_r={row['avg_r']:>+.4f}R  total_r={row['total_r']:>+7.2f}R  "
          f"top_mo={row['top_month_pct']:.2f}  [{row['skip_reason']}]")

    # ---------- Save outputs ----------
    passed.to_csv(OUT_DIR / "phase_universe_filtered_symbols.csv", index=False)

    with open(OUT_DIR / "phase_universe_filter_report.txt", "w", encoding="utf-8") as f:
        f.write("RSI MR Universe Filter -- Steps 1-3\n")
        f.write(f"Config: rsi{RSI_N}/os{OVERSOLD}/t{TIME_EXIT}/atr{ATR_MULT}/none\n")
        f.write(f"Filter: avg_r>0  trades>={FILTER_MIN_TRADES}  top_month<{FILTER_MAX_TOP_MONTH}\n\n")
        f.write(f"Full universe  : {len(valid)} symbols, {len(full_rs)} trades, "
                f"avg_r={full_avg:+.4f}R, total_r={full_tot:+.2f}R\n")
        f.write(f"Filtered       : {len(passed)} symbols, {len(filt_rs)} trades, "
                f"avg_r={filt_avg:+.4f}R, total_r={filt_tot:+.2f}R\n\n")
        f.write("Filtered symbols:\n")
        f.write(passed[["symbol","n_trades","avg_r","total_r","win_rate",
                         "top_month_pct","top_month"]].to_string(index=False))
        f.write("\n\nAll per-symbol stats:\n")
        f.write(stats_df[["symbol","n_trades","avg_r","total_r","win_rate",
                           "top_month_pct","pass_filter","skip_reason"]].to_string(index=False))

    p(f"\n[OK] phase_universe_per_symbol_stats.csv  ({len(stats_df)} symbols)")
    p(f"[OK] phase_universe_filtered_symbols.csv  ({len(passed)} pass)")
    p(f"[OK] phase_universe_filter_report.txt")
    p(f"\nSTOPPING -- awaiting human review of filtered universe before T3-T4.")


if __name__ == "__main__":
    main()
