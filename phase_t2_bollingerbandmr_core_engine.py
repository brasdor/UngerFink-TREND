#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T2 -- BollingerBandMR Core Engine
UngerFink Pipeline / Andrea Unger Methodology

Canonical config from T1:
  Timeframe  : 1D
  Filter     : ema200_price_above (long only when close > EMA200)
  BB window  : 10
  Std mult   : 2.0   (lower band = SMA10 - 2.0 * StdDev10)
  Hold bars  : 15    (fixed time exit -- primary exit)
  ATR mult   : 3.0   (safety stop = entry - 3.0 * ATR14 -- backstop only)

Entry  : close < lower_band AND close > EMA200 --> enter long at bar close
Exit   : after 15 bars (fixed time), OR ATR safety stop triggered first

MR Gate checks:
  §4.1  win rate : 50-70%
  §4.2  avg_r    : > 0.10R
  §4.7  top-1 asset <= 50% of total R
  Bear  2022 total R not catastrophic (>= -20R)
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


# =============================================================================
# CONFIG
# =============================================================================

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_bollingerbandmr_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME  = "1d"
BB_N       = 10
STD_MULT   = 2.0
HOLD_BARS  = 15
ATR_MULT   = 3.0
FILTER     = "ema200_price_above"

MAX_BARS = 2000
MIN_BARS = 200

MR_GATES = {
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "conc_max":      0.50,
    "bear_r_floor": -20.0,
}

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


# =============================================================================
# DATA
# =============================================================================

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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr14"]  = tr.rolling(14).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    out["sma_bb"] = close.rolling(BB_N).mean()
    out["std_bb"] = close.rolling(BB_N).std(ddof=0)
    out["lower_band"] = out["sma_bb"] - STD_MULT * out["std_bb"]
    return out


# =============================================================================
# BACKTEST (single symbol)
# =============================================================================

def backtest_symbol(df: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Entry : close < lower_band AND close > EMA200
            Enter at close of signal bar
    Exit  : after HOLD_BARS bars (fixed time, primary exit)
            OR ATR safety stop below entry (backstop only)
    """
    df = df.copy()
    df = add_indicators(df)
    df = df.dropna(subset=["atr14", "ema200", "lower_band"]).reset_index(drop=True)

    if len(df) < 50:
        return []

    close    = df["close"].values
    low_v    = df["low"].values
    ema200   = df["ema200"].values
    lb       = df["lower_band"].values
    atr      = df["atr14"].values
    ts       = df["timestamp"].values

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(ema200[i]) or np.isnan(lb[i]):
            continue

        if not in_pos:
            if close[i] < lb[i] and close[i] > ema200[i]:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price - ATR_MULT * atr[i]
        else:
            bars_held  = i - e_bar
            exit_price = None
            reason     = None

            if low_v[i] <= stop:
                exit_price = stop
                reason     = "atr_stop"
            elif bars_held >= HOLD_BARS:
                exit_price = close[i]
                reason     = "time_exit"

            if reason is not None:
                risk = e_price - stop
                if risk > 1e-9:
                    r_mult   = (exit_price - e_price) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    exit_dt  = pd.Timestamp(ts[i])
                    trades.append({
                        "symbol":      symbol,
                        "entry_time":  entry_dt,
                        "exit_time":   exit_dt,
                        "entry_price": round(float(e_price), 6),
                        "exit_price":  round(float(exit_price), 6),
                        "stop_loss":   round(float(stop), 6),
                        "net_r":       round(float(r_mult), 4),
                        "win":         int(r_mult > 0),
                        "bars_held":   bars_held,
                        "exit_reason": reason,
                        "year":        entry_dt.year,
                    })
                in_pos = False

    return trades


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd_r": 0.0, "total_r": 0.0, "avg_bars": 0.0}
    rs     = trades_df["net_r"].values
    wins   = rs[rs > 0]
    losses = np.abs(rs[rs < 0])
    pf     = wins.sum() / losses.sum() if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    cum    = np.cumsum(rs)
    peak   = np.maximum.accumulate(cum)
    dd     = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
    return {
        "n":        len(rs),
        "win_rate": float(len(wins) / len(rs)),
        "avg_r":    float(np.mean(rs)),
        "pf":       float(pf),
        "max_dd_r": dd,
        "total_r":  float(np.sum(rs)),
        "avg_bars": float(trades_df["bars_held"].mean()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    config_str = (f"bb{BB_N}/std{STD_MULT}/hold{HOLD_BARS}/"
                  f"atr{ATR_MULT}/{FILTER}")

    p("=" * 65)
    p("  Phase T2 -- BollingerBandMR Core Engine")
    p(f"  Timeframe : {TIMEFRAME}")
    p(f"  Config    : {config_str}")
    p(f"  Universe  : {len(SYMBOLS)} symbols")
    p(f"  Output    : {OUT_DIR.name}/")
    p("=" * 65)

    all_trades: list[dict] = []
    loaded = 0

    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is None:
            continue
        loaded += 1
        trades = backtest_symbol(df, sym)
        all_trades.extend(trades)

    p(f"  Symbols loaded : {loaded}/{len(SYMBOLS)}")
    p(f"  Total trades   : {len(all_trades)}")

    if not all_trades:
        p("  ERROR: no trades generated.")
        sys.exit(1)

    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(OUT_DIR / "phase_t2_trades.csv", index=False)

    # ---------- Overall metrics ----------
    m = calc_metrics(trades_df)
    p(f"\n  --- Overall Results ---")
    p(f"  Trades    : {m['n']}")
    p(f"  Win rate  : {m['win_rate']*100:.1f}%")
    p(f"  Avg R     : {m['avg_r']:.4f}R")
    p(f"  Total R   : {m['total_r']:.2f}R")
    p(f"  Max DD R  : {m['max_dd_r']:.2f}R")
    p(f"  Prof Fac  : {m['pf']:.2f}")
    p(f"  Avg bars  : {m['avg_bars']:.1f}")

    # ---------- Exit reason breakdown ----------
    exit_counts = trades_df["exit_reason"].value_counts()
    p(f"\n  Exit reasons:")
    for reason, cnt in exit_counts.items():
        p(f"    {reason:20s}: {cnt:4d}  ({cnt/len(trades_df)*100:.1f}%)")

    # ---------- Year-by-year ----------
    p(f"\n  --- Year-by-Year Breakdown ---")
    years     = sorted(trades_df["year"].unique())
    year_rows = []
    for yr in years:
        yt = trades_df[trades_df["year"] == yr]
        ym = calc_metrics(yt)
        bear_flag = " <<< BEAR MARKET" if yr == 2022 else ""
        p(f"  {yr}  n={ym['n']:4d}  wr={ym['win_rate']*100:5.1f}%  "
          f"avg_r={ym['avg_r']:+.3f}R  total={ym['total_r']:+7.2f}R  "
          f"dd={ym['max_dd_r']:.2f}R{bear_flag}")
        year_rows.append({"year": yr, **ym})

    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(OUT_DIR / "phase_t2_yearly.csv", index=False)

    # ---------- Asset concentration ----------
    p(f"\n  --- Asset Concentration (S4.7) ---")
    asset_stats = []
    for sym, grp in trades_df.groupby("symbol"):
        am = calc_metrics(grp)
        asset_stats.append({
            "symbol":   sym,
            "n":        am["n"],
            "win_rate": round(am["win_rate"], 4),
            "avg_r":    round(am["avg_r"], 4),
            "total_r":  round(am["total_r"], 2),
            "max_dd_r": round(am["max_dd_r"], 2),
        })
    asset_df = (pd.DataFrame(asset_stats)
                .sort_values("total_r", ascending=False)
                .reset_index(drop=True))
    asset_df.to_csv(OUT_DIR / "phase_t2_asset_summary.csv", index=False)

    top1         = asset_df.iloc[0]
    top1_contrib = top1["total_r"] / m["total_r"] * 100 if m["total_r"] > 0 else 0.0
    conc_flag    = top1_contrib > MR_GATES["conc_max"] * 100

    p(f"  Top asset  : {top1['symbol']:20s}  total_r={top1['total_r']:+.2f}R  "
      f"share={top1_contrib:.1f}%  {'<<< FLAG' if conc_flag else 'OK'}")
    p(f"\n  Top 10 assets by total R:")
    p(asset_df.head(10)[["symbol", "n", "win_rate", "avg_r", "total_r"]].to_string(index=False))

    # ---------- Gate checks ----------
    p(f"\n  --- Gate Checks ---")

    wr_pass   = MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"]
    avgr_pass = m["avg_r"] >= MR_GATES["avg_r_min"]
    conc_pass = not conc_flag

    yr2022 = year_df[year_df["year"] == 2022]
    if not yr2022.empty:
        r2022     = float(yr2022.iloc[0]["total_r"])
        bear_pass = r2022 >= MR_GATES["bear_r_floor"]
        bear_str  = f"{r2022:+.2f}R  {'PASS' if bear_pass else 'FAIL -- CATASTROPHIC'}"
    else:
        bear_pass = None
        bear_str  = "N/A -- 2022 not in data range"

    p(f"  S4.1 win rate 50-70%   : {m['win_rate']*100:.1f}%  {'PASS' if wr_pass else 'FAIL'}")
    p(f"  S4.2 avg_r > 0.10R     : {m['avg_r']:.4f}R  {'PASS' if avgr_pass else 'FAIL'}")
    p(f"  S4.7 concentration     : top={top1['symbol']} {top1_contrib:.1f}%  {'PASS' if conc_pass else 'FLAG'}")
    p(f"  Bear 2022 total R      : {bear_str}")

    all_pass = wr_pass and avgr_pass and conc_pass and (bear_pass is True or bear_pass is None)
    gate_str = "PASS" if all_pass else "FAIL"
    p(f"\n  T2 GATE: {gate_str}")

    # ---------- Write scorecard ----------
    with open(OUT_DIR / "phase_t2_scorecard.txt", "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("Phase T2 -- BollingerBandMR Core Engine\n")
        f.write(f"Timeframe : {TIMEFRAME}\n")
        f.write(f"Config    : {config_str}\n")
        f.write(f"Symbols   : {loaded} loaded / {len(SYMBOLS)} universe\n")
        f.write("=" * 65 + "\n\n")
        f.write("Overall Results:\n")
        f.write(f"  Trades    : {m['n']}\n")
        f.write(f"  Win rate  : {m['win_rate']*100:.1f}%\n")
        f.write(f"  Avg R     : {m['avg_r']:.4f}R\n")
        f.write(f"  Total R   : {m['total_r']:.2f}R\n")
        f.write(f"  Max DD R  : {m['max_dd_r']:.2f}R\n")
        f.write(f"  Prof Fac  : {m['pf']:.2f}\n")
        f.write(f"  Avg bars  : {m['avg_bars']:.1f}\n\n")
        f.write("Year-by-Year:\n")
        for _, row in year_df.iterrows():
            yr   = int(row["year"])
            bear = " <<< BEAR" if yr == 2022 else ""
            f.write(f"  {yr}  n={int(row['n']):4d}  wr={row['win_rate']*100:5.1f}%  "
                    f"avg_r={row['avg_r']:+.3f}R  total={row['total_r']:+7.2f}R  "
                    f"dd={row['max_dd_r']:.2f}R{bear}\n")
        f.write("\nAsset Concentration (top 10):\n")
        f.write(asset_df.head(10)[["symbol", "n", "win_rate", "avg_r", "total_r"]].to_string(index=False))
        f.write(f"\n  Top-1 share: {top1_contrib:.1f}%\n\n")
        f.write("Gate Checks:\n")
        f.write(f"  S4.1 win rate : {m['win_rate']*100:.1f}%  {'PASS' if wr_pass else 'FAIL'}\n")
        f.write(f"  S4.2 avg_r    : {m['avg_r']:.4f}R  {'PASS' if avgr_pass else 'FAIL'}\n")
        f.write(f"  S4.7 conc     : {top1['symbol']} {top1_contrib:.1f}%  {'PASS' if conc_pass else 'FLAG'}\n")
        f.write(f"  Bear 2022     : {bear_str}\n")
        f.write(f"\nT2 GATE: {gate_str}\n")

    p(f"\n[OK] phase_t2_trades.csv         ({len(trades_df)} trades)")
    p(f"[OK] phase_t2_asset_summary.csv  ({len(asset_df)} symbols)")
    p(f"[OK] phase_t2_yearly.csv         ({len(year_df)} years)")
    p(f"[OK] phase_t2_scorecard.txt")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
