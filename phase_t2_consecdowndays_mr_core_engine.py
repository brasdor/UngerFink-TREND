#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T2 -- ConsecDownDaysMR Core Engine
UngerFink Pipeline / Andrea Unger Methodology

Entry : N consecutive down closes (close < prev_close for exactly consec_n bars)
        Enter long at the close of the Nth consecutive down bar.
Exit  : Fixed time exit after hold_bars (Variant E validated as best on RSI MR)
Stop  : ATR x atr_mult below entry (safety net only)
Filter: none | ema200_price_above

Gate checks (MR §4.x):
  §4.1 win rate : 50-70%
  §4.2 avg_r    : > 0.10R
  §4.7 concentration : top-1 asset <= 50% of total R
  Bear 2022     : must be positive

Usage:
    # Primary config (canonical)
    python phase_t2_consecdowndays_mr_core_engine.py \
        --consec-n 5 --hold-bars 20 --atr-mult 2.0 --filter ema200_price_above

    # Secondary config
    python phase_t2_consecdowndays_mr_core_engine.py \
        --consec-n 6 --hold-bars 20 --atr-mult 2.0 --filter none

Output: data/research_consecdowndays_mr_t2_{label}/
"""

from __future__ import annotations

import argparse
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


ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw_trend_t1"

SYMBOLS = [
    "AAVE_USDT","ADA_USDT","ALT_USDT","APT_USDT","ARB_USDT","ARKM_USDT",
    "ASTER_USDT","ATOM_USDT","AVAX_USDT","BCH_USDT","BNB_USDT","BTC_USDT",
    "CHZ_USDT","DASH_USDT","DOGE_USDT","DOT_USDT","EIGEN_USDT","ENA_USDT",
    "ETH_USDT","FET_USDT","FIL_USDT","GRT_USDT","HBAR_USDT","ICP_USDT",
    "INJ_USDT","JTO_USDT","LINK_USDT","LPT_USDT","LTC_USDT","MORPHO_USDT",
    "NEAR_USDT","NIL_USDT","ONDO_USDT","ORDI_USDT","PENDLE_USDT","PENGU_USDT",
    "PEPE_USDT","RENDER_USDT","SAGA_USDT","SEI_USDT","SOL_USDT","SPK_USDT",
    "SUI_USDT","TAO_USDT","TIA_USDT","TON_USDT","TRX_USDT","UNI_USDT",
    "WLD_USDT","XRP_USDT","ZEC_USDT","ZEN_USDT",
]

MAX_BARS = 2000
MIN_BARS = 200

MR_GATES = {
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "conc_max":      0.50,
    "bear_r_floor":  -20.0,
}


# =============================================================================
# DATA
# =============================================================================

def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{symbol}_1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open","high","low","close","volume"):
            if col not in df.columns:
                return None
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:].reset_index(drop=True)
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calc_ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def calc_streak(close: pd.Series) -> pd.Series:
    """Consecutive down-day streak ending at each bar."""
    down   = (close < close.shift(1)).astype(int)
    streak = [0] * len(down)
    for i in range(1, len(down)):
        streak[i] = streak[i-1] + 1 if down.iloc[i] else 0
    return pd.Series(streak, index=close.index)


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_symbol(df: pd.DataFrame, symbol: str,
                    consec_n: int, hold_bars: int,
                    atr_mult: float, filter_mode: str) -> list[dict]:
    df = df.copy()
    df["atr"]    = calc_atr(df["high"], df["low"], df["close"], 14)
    df["streak"] = calc_streak(df["close"])
    if filter_mode == "ema200_price_above":
        df["filter"] = df["close"] > calc_ema(df["close"], 200)
    else:
        df["filter"] = True

    df = df.dropna(subset=["atr"]).reset_index(drop=True)
    if len(df) < 50:
        return []

    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr"].values
    streak = df["streak"].values
    filt   = df["filter"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(atr[i]):
            continue

        if not in_pos:
            # Enter when streak exactly reaches consec_n AND filter passes
            if int(streak[i]) == consec_n and bool(filt[i]):
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price - atr_mult * atr[i]
        else:
            bars_held  = i - e_bar
            ep = reason = None

            if low_v[i] <= stop:
                ep, reason = stop, "atr_stop"
            elif bars_held >= hold_bars:
                ep, reason = close[i], "time_exit"

            if reason:
                risk = e_price - stop
                if risk > 1e-9:
                    r_mult = (ep - e_price) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    exit_dt  = pd.Timestamp(ts[i])
                    trades.append({
                        "symbol":      symbol,
                        "entry_time":  entry_dt,
                        "exit_time":   exit_dt,
                        "entry_price": round(float(e_price), 6),
                        "exit_price":  round(float(ep), 6),
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

def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,
                "max_dd_r":0.0,"total_r":0.0,"avg_bars":0.0}
    rs   = df["net_r"].values
    wins = rs[rs>0]; losses = np.abs(rs[rs<0])
    pf   = wins.sum()/losses.sum() if losses.sum()>0 else (99.0 if wins.sum()>0 else 0.0)
    cum  = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd   = float(np.max(peak - cum))
    return {"n":len(rs),"win_rate":float(len(wins)/len(rs)),
            "avg_r":float(np.mean(rs)),"pf":float(pf),
            "max_dd_r":dd,"total_r":float(np.sum(rs)),
            "avg_bars":float(df["bars_held"].mean())}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="T2 ConsecDownDaysMR Core Engine")
    parser.add_argument("--consec-n",   type=int,   required=True)
    parser.add_argument("--hold-bars",  type=int,   required=True)
    parser.add_argument("--atr-mult",   type=float, required=True)
    parser.add_argument("--filter",     type=str,   default="none",
                        choices=["none","ema200_price_above"])
    args = parser.parse_args()

    consec_n  = args.consec_n
    hold_bars = args.hold_bars
    atr_mult  = args.atr_mult
    filt      = args.filter

    label    = f"consec{consec_n}_{'ema200' if filt != 'none' else 'nofilt'}_hold{hold_bars}_atr{atr_mult}"
    out_dir  = ROOT / f"data/research_consecdowndays_mr_t2"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_str = f"consec_n={consec_n} / hold={hold_bars} / atr={atr_mult} / filter={filt}"

    p("=" * 65)
    p(f"  Phase T2 -- ConsecDownDaysMR Core Engine")
    p(f"  Config    : {config_str}")
    p(f"  Universe  : {len(SYMBOLS)} symbols")
    p(f"  Output    : {out_dir.name}/")
    p("=" * 65)

    # Run backtest
    all_trades: list[dict] = []
    loaded = 0
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is None:
            continue
        loaded += 1
        all_trades.extend(backtest_symbol(df, sym, consec_n, hold_bars, atr_mult, filt))

    p(f"  Symbols loaded : {loaded}/{len(SYMBOLS)}")
    p(f"  Total trades   : {len(all_trades)}")

    if not all_trades:
        p("  ERROR: no trades generated.")
        sys.exit(1)

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(out_dir / f"phase_t2_trades_{label}.csv", index=False)

    # Overall metrics
    m = metrics(trades_df)
    p(f"\n  --- Overall Results ---")
    p(f"  Trades    : {m['n']}")
    p(f"  Win rate  : {m['win_rate']*100:.1f}%")
    p(f"  Avg R     : {m['avg_r']:.4f}R")
    p(f"  Total R   : {m['total_r']:.2f}R")
    p(f"  Max DD R  : {m['max_dd_r']:.2f}R")
    p(f"  Prof Fac  : {m['pf']:.2f}")
    p(f"  Avg bars  : {m['avg_bars']:.1f}")

    # Year-by-year
    p(f"\n  --- Year-by-Year ---")
    year_rows = []
    for yr in sorted(trades_df["year"].unique()):
        yt = trades_df[trades_df["year"]==yr]
        ym = metrics(yt)
        bear = " <<< BEAR" if yr == 2022 else ""
        neg  = "  <<< WEAK" if ym["avg_r"] < 0 else ""
        p(f"  {yr}  n={ym['n']:4d}  wr={ym['win_rate']*100:5.1f}%  "
          f"avg_r={ym['avg_r']:+.3f}R  total={ym['total_r']:+7.2f}R  "
          f"dd={ym['max_dd_r']:.2f}R{bear}{neg}")
        year_rows.append({"year": yr, **ym})
    pd.DataFrame(year_rows).to_csv(out_dir / f"phase_t2_yearly_{label}.csv", index=False)

    # Asset concentration
    p(f"\n  --- Asset Concentration (§4.7) ---")
    asset_rows = []
    for sym, grp in trades_df.groupby("symbol"):
        am = metrics(grp)
        asset_rows.append({"symbol": sym, **am})
    asset_df = pd.DataFrame(asset_rows).sort_values("total_r", ascending=False).reset_index(drop=True)
    asset_df.to_csv(out_dir / f"phase_t2_assets_{label}.csv", index=False)

    top1        = asset_df.iloc[0]
    top1_share  = top1["total_r"] / m["total_r"] * 100 if m["total_r"] > 0 else 0.0
    conc_flag   = top1_share > MR_GATES["conc_max"] * 100

    p(f"  Top asset : {top1['symbol']:20s}  total_r={top1['total_r']:+.2f}R  "
      f"share={top1_share:.1f}%  {'<<< FLAG' if conc_flag else 'OK'}")
    p(f"\n  Top 10 assets:")
    p(asset_df.head(10)[["symbol","n","win_rate","avg_r","total_r"]].to_string(index=False))

    # Gate checks
    p(f"\n  --- Gate Checks ---")
    y2022 = next((r for r in year_rows if r["year"]==2022), None)
    if y2022:
        bear_ok = y2022["total_r"] >= MR_GATES["bear_r_floor"]
        bear_str = f"{y2022['total_r']:+.2f}R  {'PASS' if bear_ok else 'FAIL <<< CATASTROPHIC'}"
    else:
        bear_ok, bear_str = None, "N/A -- 2022 not in data range"

    wr_pass   = MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"]
    avgr_pass = m["avg_r"] >= MR_GATES["avg_r_min"]
    conc_pass = not conc_flag

    p(f"  §4.1 win rate 50-70% : {m['win_rate']*100:.1f}%  {'PASS' if wr_pass else 'FAIL'}")
    p(f"  §4.2 avg_r > 0.10R   : {m['avg_r']:.4f}R  {'PASS' if avgr_pass else 'FAIL'}")
    p(f"  §4.7 concentration   : {top1['symbol']} {top1_share:.1f}%  {'PASS' if conc_pass else 'FLAG'}")
    p(f"  Bear 2022 total R    : {bear_str}")

    all_pass = wr_pass and avgr_pass and conc_pass and (bear_ok is True or bear_ok is None)
    gate = "PASS" if all_pass else "FAIL"
    p(f"\n  T2 GATE: {gate}")

    # Write scorecard
    with open(out_dir / f"phase_t2_scorecard_{label}.txt", "w", encoding="utf-8") as f:
        f.write(f"Phase T2 -- ConsecDownDaysMR\nConfig: {config_str}\n\n")
        f.write(f"Trades    : {m['n']}\n")
        f.write(f"Win rate  : {m['win_rate']*100:.1f}%\n")
        f.write(f"Avg R     : {m['avg_r']:.4f}R\n")
        f.write(f"Total R   : {m['total_r']:.2f}R\n")
        f.write(f"Max DD R  : {m['max_dd_r']:.2f}R\n")
        f.write(f"PF        : {m['pf']:.2f}\n\n")
        f.write("Year-by-Year:\n")
        for r in year_rows:
            yr = int(r["year"])
            f.write(f"  {yr}  n={int(r['n']):4d}  wr={r['win_rate']*100:5.1f}%  "
                    f"avg_r={r['avg_r']:+.3f}R  total={r['total_r']:+7.2f}R  "
                    f"dd={r['max_dd_r']:.2f}R{'  <<< BEAR' if yr==2022 else ''}\n")
        f.write(f"\n§4.1 win rate : {'PASS' if wr_pass else 'FAIL'}\n")
        f.write(f"§4.2 avg_r    : {'PASS' if avgr_pass else 'FAIL'}\n")
        f.write(f"§4.7 conc     : {top1['symbol']} {top1_share:.1f}% {'PASS' if conc_pass else 'FLAG'}\n")
        f.write(f"Bear 2022     : {bear_str}\n")
        f.write(f"\nT2 GATE: {gate}\n")

    p(f"\n[OK] phase_t2_trades_{label}.csv     ({len(trades_df)} trades)")
    p(f"[OK] phase_t2_assets_{label}.csv")
    p(f"[OK] phase_t2_yearly_{label}.csv")
    p(f"[OK] phase_t2_scorecard_{label}.txt")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
