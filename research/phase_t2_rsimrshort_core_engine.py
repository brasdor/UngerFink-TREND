#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T2 -- RSIMeanReversionShort Core Engine
UngerFink Pipeline / Andrea Unger Methodology  (Section 19A)

Config (frozen from T1):
  Timeframe  : 4H
  Entry      : RSI(10) > 75 AND close < EMA(200) --> SHORT at close
  Exit       : time exit after 15 bars (15 x 4h = 60 hours)
  Safety stop: ATR(14) x 2.0 ABOVE entry  (short stop-out if price rises)
  Filter     : ema200_price_below (MANDATORY -- 0 combos pass without it)
  Side       : SHORT (Binance Futures USD-M)

R calculation for shorts:
  risk     = stop_loss - entry_price  (positive)
  pnl_r    = (entry_price - exit_price) / risk  (positive when price falls)

Gate checks:
  §4.1 win rate : 50-70%
  §4.2 avg_r    : > 0.15R (Futures cost floor)
  §4.7 concentration : top-1 <= 50% of total R
  Bear 2022     : must be BEST year (filter active in downtrends)

Output: data/research_rsimrshort_t2/
"""

from __future__ import annotations
import os, sys, warnings
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
OUT_DIR = ROOT / "data" / "research_rsimrshort_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

TF          = "4h"
RSI_N       = 10
OVERBOUGHT  = 75
HOLD_BARS   = 15
ATR_MULT    = 2.0
EMA_N       = 200
MAX_BARS    = 6000
MIN_BARS    = 200

MR_SHORT_GATES = {
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.15,
    "conc_max":      0.50,
    "bear_r_floor":  -20.0,
}


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{sym}_{TF}.csv"
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


def backtest_symbol(df: pd.DataFrame, sym: str) -> list[dict]:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # RSI(10)
    d  = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    rsi = (100 - (100 / (1 + ag/al.replace(0,np.nan)))).values

    # ATR(14)
    pc = close.shift(1)
    tr = pd.concat([high-low,(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values

    # EMA(200)
    ema200 = close.ewm(span=EMA_N, adjust=False).mean().values

    close_v = close.values
    high_v  = high.values
    ts      = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos  = False
    e_price = stop = 0.0
    e_bar   = 0; e_ts = None

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            # SHORT entry: RSI overbought AND price below EMA200 (confirmed downtrend)
            if rsi[i] > OVERBOUGHT and close_v[i] < ema200[i]:
                in_pos  = True
                e_price = close_v[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price + ATR_MULT * atr[i]  # stop ABOVE entry for short
        else:
            bars_held = i - e_bar
            ep = reason = None

            # Short stop-out: high touches stop (price rising against us)
            if high_v[i] >= stop:
                ep, reason = stop, "atr_stop"
            elif bars_held >= HOLD_BARS:
                ep, reason = close_v[i], "time_exit"

            if reason:
                risk = stop - e_price  # positive (stop above entry)
                if risk > 1e-9:
                    # Short R: positive when price fell (ep < e_price)
                    r_mult = (e_price - ep) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    trades.append({
                        "symbol":      sym,
                        "entry_time":  entry_dt,
                        "exit_time":   pd.Timestamp(ts[i]),
                        "entry_price": round(float(e_price), 8),
                        "exit_price":  round(float(ep), 8),
                        "stop_loss":   round(float(stop), 8),
                        "net_r":       round(float(r_mult), 4),
                        "win":         int(r_mult > 0),
                        "bars_held":   bars_held,
                        "exit_reason": reason,
                        "year":        entry_dt.year,
                        "rsi_at_entry":round(float(rsi[i-bars_held] if bars_held < i else rsi[e_bar]),2),
                        "ema200_at_entry": round(float(ema200[e_bar]),6),
                    })
                in_pos = False

    return trades


def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0,"avg_bars":0.0}
    rs=df["net_r"].values
    w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum=np.cumsum(rs); peak=np.maximum.accumulate(cum)
    dd=float(np.max(peak-cum))
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":dd,"total_r":float(np.sum(rs)),
            "avg_bars":float(df["bars_held"].mean())}


def main() -> None:
    config_str = f"RSI({RSI_N})>{OVERBOUGHT} / ema200_price_below / hold{HOLD_BARS}bars / atr{ATR_MULT} / {TF}"
    p("="*65)
    p("  Phase T2 -- RSIMeanReversionShort Core Engine")
    p(f"  Config : {config_str}")
    p(f"  Side   : SHORT  |  Exchange: Binance Futures  |  Cost floor: 0.15R")
    p(f"  Universe: {len(SYMBOLS)} symbols")
    p("="*65)

    all_trades: list[dict] = []
    loaded = 0
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is None:
            continue
        loaded += 1
        all_trades.extend(backtest_symbol(df, sym))

    p(f"  Symbols loaded : {loaded}/{len(SYMBOLS)}")
    p(f"  Total trades   : {len(all_trades)}")

    if not all_trades:
        p("  ERROR: no trades generated."); sys.exit(1)

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(OUT_DIR/"phase_t2_trades.csv", index=False)

    m = calc_metrics(trades_df)
    p(f"\n  --- Overall Results (SHORT side, 4H) ---")
    p(f"  Trades    : {m['n']}")
    p(f"  Win rate  : {m['win_rate']*100:.1f}%")
    p(f"  Avg R     : {m['avg_r']:.4f}R")
    p(f"  Total R   : {m['total_r']:.2f}R")
    p(f"  Max DD R  : {m['max_dd_r']:.2f}R")
    p(f"  Prof Fac  : {m['pf']:.2f}")
    p(f"  Avg bars  : {m['avg_bars']:.1f}  ({m['avg_bars']*4:.0f} hours)")

    # Year-by-year — critical: 2022 should be BEST year
    p(f"\n  --- Year-by-Year ---")
    p(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotalR':>8}  {'DD':>7}  Expect")
    year_rows = []
    for yr in sorted(trades_df["year"].unique()):
        yt = trades_df[trades_df["year"]==yr]
        ym = calc_metrics(yt)
        expect = "BEST (bear yr -- EMA200 filter active)" if yr==2022 else \
                 "FEW  (price mostly above EMA200)" if yr in (2021,2024) else ""
        neg = "  <<< WEAK" if ym["avg_r"]<0 else ""
        p(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
          f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
          f"{ym['max_dd_r']:>6.2f}R  {expect}{neg}")
        year_rows.append({"year":yr,**ym})
    pd.DataFrame(year_rows).to_csv(OUT_DIR/"phase_t2_yearly.csv", index=False)

    # Asset concentration
    p(f"\n  --- Asset Concentration (§4.7) ---")
    asset_rows = []
    for sym, grp in trades_df.groupby("symbol"):
        am = calc_metrics(grp)
        asset_rows.append({"symbol":sym,**am})
    asset_df = pd.DataFrame(asset_rows).sort_values("total_r",ascending=False).reset_index(drop=True)
    asset_df.to_csv(OUT_DIR/"phase_t2_asset_summary.csv", index=False)

    top1 = asset_df.iloc[0]
    top1_share = top1["total_r"]/m["total_r"]*100 if m["total_r"]>0 else 0.0
    conc_flag  = top1_share > MR_SHORT_GATES["conc_max"]*100
    p(f"  Top asset : {top1['symbol']:20s}  total_r={top1['total_r']:+.2f}R  "
      f"share={top1_share:.1f}%  {'<<< FLAG' if conc_flag else 'OK'}")
    p(f"\n  Top 10 assets:")
    p(asset_df.head(10)[["symbol","n","win_rate","avg_r","total_r"]].to_string(index=False))

    # Trade count by year (filter activity diagnostic)
    p(f"\n  --- Filter Activity by Year ---")
    p(f"  (ema200_price_below = active only when price < EMA200)")
    yr_counts = trades_df.groupby("year")["symbol"].count()
    total_count = yr_counts.sum()
    for yr in sorted(yr_counts.index):
        pct = yr_counts[yr]/total_count*100
        bar = "#" * int(pct/5)
        p(f"  {yr}: {yr_counts[yr]:4d} trades ({pct:4.1f}%)  {bar}")

    # Gate checks
    p(f"\n  --- Gate Checks ---")
    y2022 = next((r for r in year_rows if r["year"]==2022), None)
    wr_pass   = MR_SHORT_GATES["win_rate_min"] <= m["win_rate"] <= MR_SHORT_GATES["win_rate_max"]
    avgr_pass = m["avg_r"] >= MR_SHORT_GATES["avg_r_min"]
    conc_pass = not conc_flag

    if y2022:
        bear_ok  = y2022["total_r"] >= MR_SHORT_GATES["bear_r_floor"]
        # Is 2022 the best year?
        yr_totals = {r["year"]: r["total_r"] for r in year_rows}
        best_yr   = max(yr_totals, key=yr_totals.get)
        bear_best = (best_yr == 2022)
        bear_str  = (f"{y2022['total_r']:+.2f}R  {'PASS' if bear_ok else 'FAIL'}  "
                     f"{'(best year -- as expected)' if bear_best else f'(NOT best year -- best was {best_yr})'}")
    else:
        bear_ok = None; bear_str = "N/A"; bear_best = False

    p(f"  §4.1 win rate 50-70% : {m['win_rate']*100:.1f}%  {'PASS' if wr_pass else 'FAIL'}")
    p(f"  §4.2 avg_r > 0.15R   : {m['avg_r']:.4f}R  {'PASS' if avgr_pass else 'FAIL'}")
    p(f"  §4.7 concentration   : {top1['symbol']} {top1_share:.1f}%  {'PASS' if conc_pass else 'FLAG'}")
    p(f"  Bear 2022 total R    : {bear_str}")

    all_pass = wr_pass and avgr_pass and conc_pass and (bear_ok is True or bear_ok is None)
    gate = "PASS" if all_pass else "FAIL"
    p(f"\n  T2 GATE: {gate}")

    with open(OUT_DIR/"phase_t2_scorecard.txt","w",encoding="utf-8") as f:
        f.write(f"Phase T2 -- RSIMeanReversionShort\n")
        f.write(f"Config: {config_str}\n\n")
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
                    f"avg_r={r['avg_r']:+.3f}R  total={r['total_r']:+7.2f}R\n")
        f.write(f"\n§4.1 win rate : {'PASS' if wr_pass else 'FAIL'}\n")
        f.write(f"§4.2 avg_r    : {'PASS' if avgr_pass else 'FAIL'}\n")
        f.write(f"§4.7 conc     : {top1['symbol']} {top1_share:.1f}% {'PASS' if conc_pass else 'FLAG'}\n")
        f.write(f"Bear 2022     : {bear_str}\n")
        f.write(f"\nT2 GATE: {gate}\n")

    p(f"\n[OK] phase_t2_trades.csv       ({len(trades_df)} trades)")
    p(f"[OK] phase_t2_asset_summary.csv ({len(asset_df)} symbols)")
    p(f"[OK] phase_t2_yearly.csv        ({len(year_rows)} years)")
    p(f"[OK] phase_t2_scorecard.txt")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
