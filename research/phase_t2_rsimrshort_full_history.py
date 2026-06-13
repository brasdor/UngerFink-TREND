#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch full 4H OHLCV from 2021-01-01, then rerun RSI MR Short T2
with complete 2021-2026 history for proper 2022 bear-market gate check.

Step 1: Fetch 4H data -> data/research_rsimrshort_t1/ohlcv_cache/{sym}_4h.csv
Step 2: Run T2 reading from that cache
        Config: RSI(10)>75 / ema200_price_below / hold15 / atr2.0 / 4H / SHORT
Output: data/research_rsimrshort_t2_full/
"""

from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)

ROOT      = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "research_rsimrshort_t1" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_rsimrshort_t2_full"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS_US = [
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

TF         = "4h"
SINCE_MS   = 1609459200000  # 2021-01-01 UTC
BATCH_SIZE = 1000
TARGET     = 13000   # ~5.5 years at 4H
MIN_BARS   = 200
SLEEP_SEC  = 0.15

# T2 config
RSI_N      = 10
OVERBOUGHT = 75
HOLD_BARS  = 15
ATR_MULT   = 2.0
EMA_N      = 200

MR_SHORT_GATES = {
    "win_rate_min":  0.50, "win_rate_max":  0.70,
    "avg_r_min":     0.15, "conc_max":      0.50,
    "bear_r_floor": -20.0,
}


# =============================================================================
# STEP 1: FETCH 4H DATA
# =============================================================================

def underscore_to_ccxt(sym: str) -> str:
    parts = sym.split("_")
    return f"{parts[0]}/{parts[1]}"


def fetch_symbol(exchange, ccxt_sym: str, out_path: Path) -> int:
    rows = []; since = SINCE_MS
    for _ in range((TARGET//BATCH_SIZE)+5):
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(ccxt_sym, TF, limit=BATCH_SIZE, since=since)
                time.sleep(exchange.rateLimit/1000.0)
                break
            except Exception as e:
                if attempt==2:
                    p(f"    [ERROR] {ccxt_sym}: {e}"); return -1
                time.sleep(1.0*(attempt+1))
        if not batch: break
        rows.extend(batch); since = batch[-1][0]+1
        if len(batch) < BATCH_SIZE: break
        if len(rows) >= TARGET: break
    if len(rows) < MIN_BARS: return 0
    rows = rows[-TARGET:][:-1]  # trim and drop forming candle
    df = pd.DataFrame(rows, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df[["timestamp","open","high","low","close","volume"]].to_csv(out_path, index=False)
    return len(rows)


def fetch_all_4h() -> None:
    try:
        import ccxt
    except ImportError:
        p("ERROR: ccxt not installed"); sys.exit(1)

    exchange = ccxt.binanceus({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    p(f"  Fetching {TF} data from 2021-01-01 for {len(SYMBOLS_US)} symbols...")

    ok=skip=fail=0
    for i, sym_us in enumerate(SYMBOLS_US, 1):
        ccxt_sym = underscore_to_ccxt(sym_us)
        out_path = CACHE_DIR / f"{sym_us}_4h.csv"
        if out_path.exists():
            try:
                ex = pd.read_csv(out_path)
                if len(ex) >= 2000:
                    p(f"  [{i:2d}/52] {sym_us:20s} SKIP ({len(ex)} bars cached)")
                    skip += 1; continue
            except Exception: pass
        p(f"  [{i:2d}/52] {sym_us:20s} fetching...", end=" ")
        n = fetch_symbol(exchange, ccxt_sym, out_path)
        if n > 0:
            p(f"{n} bars"); ok += 1
        elif n == 0:
            p("too few bars"); fail += 1
        else:
            p("FETCH ERROR"); fail += 1

    p(f"\n  Done: {ok} fetched, {skip} skipped, {fail} failed")


# =============================================================================
# STEP 2: T2 BACKTEST
# =============================================================================

def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{sym}_4h.csv"
    if not path.exists(): return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open","high","low","close","volume"):
            if col not in df.columns: return None
        return df if len(df) >= MIN_BARS else None
    except Exception: return None


def backtest_symbol(df: pd.DataFrame, sym: str) -> list[dict]:
    close = df["close"]; high = df["high"]; low = df["low"]
    d  = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    rsi    = (100-(100/(1+ag/al.replace(0,np.nan)))).values
    pc     = close.shift(1)
    tr     = pd.concat([high-low,(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    atr    = tr.rolling(14).mean().values
    ema200 = close.ewm(span=EMA_N, adjust=False).mean().values
    c_v    = close.values; h_v = high.values
    ts     = df["timestamp"].values

    trades=[]; in_pos=False; e_price=stop=0.0; e_bar=0; e_ts=None

    for i in range(len(df)):
        if any(np.isnan(x) for x in [rsi[i],atr[i],ema200[i]]): continue
        if not in_pos:
            if rsi[i]>OVERBOUGHT and c_v[i]<ema200[i]:
                in_pos=True; e_price=c_v[i]; e_bar=i; e_ts=ts[i]
                stop=e_price+ATR_MULT*atr[i]
        else:
            bars_held=i-e_bar; ep=reason=None
            if h_v[i]>=stop: ep,reason=stop,"atr_stop"
            elif bars_held>=HOLD_BARS: ep,reason=c_v[i],"time_exit"
            if reason:
                risk=stop-e_price
                if risk>1e-9:
                    rm=(e_price-ep)/risk
                    entry_dt=pd.Timestamp(e_ts)
                    trades.append({"symbol":sym,"entry_time":entry_dt,
                        "exit_time":pd.Timestamp(ts[i]),"net_r":round(float(rm),4),
                        "win":int(rm>0),"bars_held":bars_held,"exit_reason":reason,
                        "year":entry_dt.year})
                in_pos=False
    return trades


def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0,"avg_bars":0.0}
    rs=df["net_r"].values; w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum=np.cumsum(rs); peak=np.maximum.accumulate(cum)
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":float(np.max(peak-cum)),"total_r":float(np.sum(rs)),
            "avg_bars":float(df["bars_held"].mean())}


def run_t2() -> None:
    p("="*65)
    p("  Phase T2 -- RSIMeanReversionShort (full 4H history)")
    p(f"  RSI({RSI_N})>{OVERBOUGHT} / ema200_price_below / hold{HOLD_BARS} / atr{ATR_MULT}")
    p("="*65)

    all_trades=[]; loaded=0
    for sym in SYMBOLS_US:
        df = load_ohlcv(sym)
        if df is None: continue
        loaded += 1
        all_trades.extend(backtest_symbol(df, sym))

    p(f"  Symbols loaded : {loaded}/{len(SYMBOLS_US)}")
    p(f"  Total trades   : {len(all_trades)}")
    if not all_trades:
        p("  ERROR: no trades"); sys.exit(1)

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(OUT_DIR/"phase_t2_trades.csv", index=False)
    m = calc_metrics(trades_df)

    p(f"\n  Overall Results (SHORT / 4H / full history):")
    p(f"  Trades    : {m['n']}")
    p(f"  Win rate  : {m['win_rate']*100:.1f}%")
    p(f"  Avg R     : {m['avg_r']:.4f}R")
    p(f"  Total R   : {m['total_r']:.2f}R")
    p(f"  Max DD R  : {m['max_dd_r']:.2f}R")
    p(f"  PF        : {m['pf']:.2f}")
    p(f"  Avg bars  : {m['avg_bars']:.1f}  ({m['avg_bars']*4:.0f} hours)")

    # Year-by-year -- critical 2022 check
    p(f"\n  Year-by-Year (CRITICAL: 2022 must be BEST year):")
    p(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotalR':>8}  {'DD':>7}  Note")
    year_rows = []
    for yr in sorted(trades_df["year"].unique()):
        yt  = trades_df[trades_df["year"]==yr]
        ym  = calc_metrics(yt)
        note = "<<< BEAR -- EMA200 filter most active" if yr==2022 else \
               "FEW expected (bull yr, price > EMA200)" if yr in (2021,2024) else ""
        neg  = "  <<< WEAK" if ym["avg_r"]<0 else ""
        p(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
          f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
          f"{ym['max_dd_r']:>6.2f}R  {note}{neg}")
        year_rows.append({"year":yr,**ym})
    pd.DataFrame(year_rows).to_csv(OUT_DIR/"phase_t2_yearly.csv", index=False)

    # Filter activity
    yr_counts = trades_df.groupby("year")["symbol"].count()
    total_cnt = yr_counts.sum()
    p(f"\n  Filter activity by year (ema200_price_below):")
    for yr in sorted(yr_counts.index):
        pct = yr_counts[yr]/total_cnt*100
        bar = "#"*int(pct/3)
        p(f"  {yr}: {yr_counts[yr]:4d} trades ({pct:4.1f}%)  {bar}")

    # Asset concentration
    asset_rows = []
    for sym, grp in trades_df.groupby("symbol"):
        am = calc_metrics(grp)
        asset_rows.append({"symbol":sym,**am})
    asset_df = pd.DataFrame(asset_rows).sort_values("total_r",ascending=False).reset_index(drop=True)
    asset_df.to_csv(OUT_DIR/"phase_t2_asset_summary.csv", index=False)
    top1 = asset_df.iloc[0]
    top1_share = top1["total_r"]/m["total_r"]*100 if m["total_r"]>0 else 0.0

    # Gate checks
    p(f"\n  Gate Checks:")
    wr_pass   = MR_SHORT_GATES["win_rate_min"]<=m["win_rate"]<=MR_SHORT_GATES["win_rate_max"]
    avgr_pass = m["avg_r"]>=MR_SHORT_GATES["avg_r_min"]
    conc_pass = not (top1_share>MR_SHORT_GATES["conc_max"]*100)
    y2022     = next((r for r in year_rows if r["year"]==2022), None)
    yr_totals = {r["year"]: r["total_r"] for r in year_rows}
    best_yr   = max(yr_totals, key=yr_totals.get) if yr_totals else None

    if y2022:
        bear_ok   = y2022["total_r"]>=MR_SHORT_GATES["bear_r_floor"]
        bear_best = (best_yr==2022)
        few_2022  = y2022["n"] < 20
        bear_str  = (f"{y2022['total_r']:+.2f}R  n={y2022['n']}  "
                     f"{'PASS' if bear_ok else 'FAIL'}  "
                     f"{'(BEST YEAR - as expected)' if bear_best else f'(NOT best: best={best_yr})'}  "
                     f"{'<<< FEW TRADES (<20) -- insufficient bear evidence' if few_2022 else ''}")
    else:
        bear_ok=None; bear_str="N/A -- 2022 not in data"; few_2022=None

    p(f"  §4.1 win rate 50-70% : {m['win_rate']*100:.1f}%  {'PASS' if wr_pass else 'FAIL'}")
    p(f"  §4.2 avg_r > 0.15R   : {m['avg_r']:.4f}R  {'PASS' if avgr_pass else 'FAIL'}")
    p(f"  §4.7 concentration   : {top1['symbol']} {top1_share:.1f}%  {'PASS' if conc_pass else 'FLAG'}")
    p(f"  Bear 2022            : {bear_str}")

    all_pass = wr_pass and avgr_pass and conc_pass and (bear_ok is True or bear_ok is None)
    gate = "PASS" if all_pass else "FAIL"
    p(f"\n  T2 GATE: {gate}")

    with open(OUT_DIR/"phase_t2_scorecard.txt","w",encoding="utf-8") as f:
        f.write(f"Phase T2 -- RSIMeanReversionShort (full 4H history)\n")
        f.write(f"Config: RSI({RSI_N})>{OVERBOUGHT}/ema200_below/hold{HOLD_BARS}/atr{ATR_MULT}/4H/SHORT\n\n")
        f.write(f"Trades    : {m['n']}\nWin rate  : {m['win_rate']*100:.1f}%\n")
        f.write(f"Avg R     : {m['avg_r']:.4f}R\nTotal R   : {m['total_r']:.2f}R\n")
        f.write(f"Max DD R  : {m['max_dd_r']:.2f}R\nPF        : {m['pf']:.2f}\n\n")
        f.write("Year-by-Year:\n")
        for r in year_rows:
            f.write(f"  {int(r['year'])}  n={int(r['n']):4d}  wr={r['win_rate']*100:5.1f}%  "
                    f"avg_r={r['avg_r']:+.3f}R  total={r['total_r']:+7.2f}R\n")
        f.write(f"\nT2 GATE: {gate}\n")

    p(f"\n[OK] phase_t2_trades.csv       ({len(trades_df)} trades)")
    p(f"[OK] phase_t2_asset_summary.csv")
    p(f"[OK] phase_t2_yearly.csv")
    p(f"[OK] phase_t2_scorecard.txt")
    sys.exit(0 if all_pass else 1)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("="*70)
    p("  RSI MR Short: Fetch full 4H history + T2 with 2022 bear test")
    p("="*70)

    p("\n--- Step 1: Fetch 4H OHLCV from 2021-01-01 ---")
    fetch_all_4h()

    p("\n--- Step 2: Run T2 Short with full history ---")
    run_t2()


if __name__ == "__main__":
    main()
