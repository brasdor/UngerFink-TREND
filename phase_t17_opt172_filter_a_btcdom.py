#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.2 -- Regime Filter A: BTC Dominance Proxy Slope

Filter: 7-day linear regression slope of BTC/USDT log-price < 0.
Proxy for BTC dominance declining (alts gaining market share).

Rationale:
  When BTC price is weakening relative to recent history (7-day slope < 0),
  capital tends to rotate from BTC into altcoins, reducing BTC dominance.
  Altcoin Donchian breakouts in this regime are more likely to represent
  genuine alt-season moves rather than BTC-correlated rallies.

Implementation (no lookahead):
  1. Load BTC/USDT daily closes (shared reference file).
  2. For each daily bar i, compute 7-day linear regression slope of
     ln(BTC_close) over bars [i-6 .. i].
  3. Pre-build a {timestamp: slope} map aligned to all symbols.
  4. At entry bar i: allow entry only when btc_slope[i] < 0.
     If BTC data unavailable for bar i: skip entry (conservative).

Note on proxy quality:
  7-day BTC log-price slope is a coarse proxy for BTC.D slope.
  BTC.D depends on relative market caps. In periods where BTC falls
  faster than alts, BTC.D can still rise. This proxy captures direction,
  not magnitude. Results should be interpreted with this limitation.

Universe   : 24 symbols (filtered_symbols_v2_included_only.csv)
Base config: Donchian N=20 / ema200_price / ATR(14)x2.0 stop /
             Chandelier ACT+4R trail 3xATR / LONG only / 1D

Tests
  Stability zone : N=[15, 20, 25] -- all must remain profitable
  T4 robustness  : MC, cost stress, period splits, remove-best, remove-best-month
  Trade guard    : >= 80 trades
  Portfolio/CAGR : T5 max8 + T6 capital execution

Baseline: avg_r=+1.101R  PF=3.072  CAGR=+14.3% (max8)

Output: data/research_donchian_regimeV2_filter_a/
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

ROOT       = Path(__file__).parent
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUT_DIR    = ROOT / "data" / "research_donchian_regimeV2_filter_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BTC_SYMBOL = "BTC/USDT"
R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# Frozen Donchian config
ATR_N      = 14
STOP_MULT  = 2.0
CHAN_ACT_R = 4.0
CHAN_TRAIL = 3.0

# Filter A: BTC slope
BTC_SLOPE_WINDOW  = 7    # bars for linear regression
BTC_SLOPE_THRESH  = 0.0  # slope < 0 → BTC declining → allow entry

STABILITY_N = [15, 20, 25]
CANONICAL_N = 20

MC_RUNS     = 2000
MC_BLOCKS   = [1, 3, 5, 10, 20]
MC_SEED     = 42
EXTRA_COSTS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35
PORT_MAX_OPEN  = 8

BASELINE  = dict(avg_r=1.1011, pf=3.072, cagr_pct=14.3, trades=461)
MIN_TRADES = 80


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i-1]):
            out[i] = close[i]*k + out[i-1]*(1.0-k)
    return out

def _atr(high, low, close, n):
    nb = len(close); tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1:n+1]))
        for i in range(n+1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i-1]):
                atr[i] = (atr[i-1]*(n-1) + tr[i]) / n
    return atr


# =============================================================================
# BTC SLOPE MAP
# =============================================================================

def build_btc_slope_map(window: int = BTC_SLOPE_WINDOW) -> Dict[object, float]:
    """
    Compute 7-day log-price linear regression slope for BTC/USDT.
    Returns {timestamp: slope} covering all BTC daily bars.
    Slope is in log-return/day units. Negative = BTC declining.
    """
    clean = BTC_SYMBOL.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found. Filter A will block all entries.")
        return {}

    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df["close"] = pd.to_numeric(df.get("close", pd.Series(dtype=float)), errors="coerce")
    df = df.dropna(subset=["time","close"]).sort_values("time").reset_index(drop=True)

    ts_arr  = df["time"].to_numpy()
    cl_arr  = df["close"].to_numpy(dtype=float)
    log_cl  = np.log(cl_arr)
    x       = np.arange(window, dtype=float)
    slope_map: Dict[object, float] = {}

    for i in range(window - 1, len(cl_arr)):
        window_vals = log_cl[i - window + 1 : i + 1]
        if not np.all(np.isfinite(window_vals)):
            continue
        slope = float(np.polyfit(x, window_vals, 1)[0])
        slope_map[ts_arr[i]] = slope

    return slope_map


# =============================================================================
# DATA
# =============================================================================

def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists(): return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    return (df[["time","open","high","low","close"]]
            .dropna(subset=["time","close"])
            .sort_values("time")
            .reset_index(drop=True))


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
    btc_slope:    float   # slope at entry (for diagnostics)


def run_backtest(df: pd.DataFrame, symbol: str,
                 btc_slope_map: Dict,
                 entry_n: int = CANONICAL_N,
                 apply_btc: bool = True) -> List[Trade]:
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(entry_n, ATR_N, 200) + 20:
        return []

    exit_n    = entry_n // 2
    cl        = df["close"].to_numpy(dtype=float)
    hi        = df["high"].to_numpy(dtype=float)
    lo        = df["low"].to_numpy(dtype=float)
    ts        = df["time"].to_numpy()
    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades: List[Trade] = []
    pos: Optional[dict] = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])):
            continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])):
            continue

        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry"])/pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["entry"])/pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            exit_px = exit_reason = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_reason = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_reason = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_reason = cl[i], "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(Trade(
                    symbol=symbol, entry_time=pos["entry_time"], exit_time=ts[i],
                    entry_price=pos["entry"], exit_price=exit_px,
                    initial_stop=pos["stop"], initial_risk=pos["risk"],
                    exit_reason=exit_reason, bars_held=pos["bars"],
                    net_r=net_r, mae_r=min(pos["mae_r"], net_r),
                    mfe_r=pos["mfe_r"], btc_slope=pos["btc_slope"],
                ))
                pos = None
            else:
                if pos["mfe_r"] >= CHAN_ACT_R: pos["chan_active"] = True
                if pos["chan_active"]:
                    pos["chan_stop"] = max(pos["chan_stop"],
                                          pos["hh"] - atr14[i]*CHAN_TRAIL)

        if pos is None:
            if not (cl[i] > ema200[i-1] and cl[i] > don_upper[i]):
                continue

            slope_val = np.nan
            if apply_btc:
                slope_val = btc_slope_map.get(ts[i], np.nan)
                if not np.isfinite(slope_val):
                    continue
                if slope_val >= BTC_SLOPE_THRESH:
                    continue   # BTC not declining → skip

            risk = atr14[i-1] * STOP_MULT
            if risk <= 0: continue
            stop_px = cl[i] - risk
            pos = dict(entry=cl[i], stop=stop_px, risk=risk, entry_time=ts[i],
                       hh=hi[i], chan_active=False, chan_stop=stop_px,
                       mfe_r=0.0, mae_r=0.0, bars=1, btc_slope=float(slope_val))

    if pos is not None:
        exit_px = cl[-1]; net_r = (exit_px-pos["entry"])/pos["risk"]
        trades.append(Trade(
            symbol=symbol, entry_time=pos["entry_time"], exit_time=ts[-1],
            entry_price=pos["entry"], exit_price=exit_px,
            initial_stop=pos["stop"], initial_risk=pos["risk"],
            exit_reason="end_of_data", bars_held=pos["bars"],
            net_r=net_r, mae_r=min(pos["mae_r"], net_r),
            mfe_r=pos["mfe_r"], btc_slope=pos["btc_slope"],
        ))
    return trades


def run_universe(entry_n: int, apply_filter: bool,
                 symbols: List[str], btc_slope_map: Dict) -> pd.DataFrame:
    all_trades: List[Trade] = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(run_backtest(df, sym, btc_slope_map,
                                       entry_n=entry_n, apply_btc=apply_filter))
    if not all_trades: return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in all_trades])
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[EXIT_T, ENTRY_T, R_COL]).sort_values(EXIT_T).reset_index(drop=True)
    df["month"] = df[EXIT_T].dt.tz_convert(None).dt.to_period("M").astype(str)
    df["year"]  = df[EXIT_T].dt.year
    return df


# =============================================================================
# STATS / T4 / T5 / T6
# =============================================================================

def _pf(r):
    g=r[r>0].sum(); l=-r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.0)

def _max_dd(r):
    if r.size==0: return 0.0
    eq=np.cumsum(r); pk=np.maximum.accumulate(eq); return float((eq-pk).min())

def summarize(r):
    if r.size==0:
        return dict(trades=0,total_r=0.0,avg_r=0.0,win_rate=0.0,
                    profit_factor=0.0,max_dd_r=0.0,std_r=0.0,t_score=0.0)
    avg=float(r.mean()); std=float(r.std(ddof=1)) if r.size>1 else 0.0
    t=avg/(std/math.sqrt(r.size)) if std>0 else 0.0
    return dict(trades=int(r.size),total_r=float(r.sum()),avg_r=avg,
                win_rate=float((r>0).mean()),profit_factor=_pf(r),
                max_dd_r=_max_dd(r),std_r=std,t_score=float(t))

def _bstrap(vals, bs, rng):
    n=len(vals); out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); e=min(s+bs,n); blk=vals[s:e]
        if len(blk)<bs: blk=np.concatenate([blk,vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n],dtype=float)

def t4_montecarlo(df):
    vals=df[R_COL].to_numpy(dtype=float); rng=np.random.default_rng(MC_SEED); rows=[]
    for bs in MC_BLOCKS:
        tots,dds,pfs=[],[],[]
        for _ in range(MC_RUNS):
            s=_bstrap(vals,bs,rng); tots.append(s.sum()); dds.append(_max_dd(s)); pfs.append(_pf(s))
        tots=np.array(tots); dds=np.array(dds); pfs=np.array(pfs)
        rows.append(dict(block_size=bs,mc_runs=MC_RUNS,
            total_r_p05=float(np.percentile(tots,5)),total_r_p50=float(np.percentile(tots,50)),
            total_r_p95=float(np.percentile(tots,95)),dd_p95=float(np.percentile(dds,95)),
            pf_p05=float(np.percentile(pfs,5)),pf_p50=float(np.percentile(pfs,50)),
            prob_positive=float((tots>0).mean())))
    return pd.DataFrame(rows)

def t4_cost_stress(df):
    vals=df[R_COL].to_numpy(dtype=float); rows=[]
    for ec in EXTRA_COSTS:
        s=summarize(vals-ec); s["extra_cost"]=ec; rows.append(s)
    return pd.DataFrame(rows)

def t4_period_splits(df):
    df=df.sort_values(EXIT_T).reset_index(drop=True)
    mid=len(df)//2; med_t=df[EXIT_T].median()
    slices={"first_half_by_trade":df.iloc[:mid],"second_half_by_trade":df.iloc[mid:],
            "last_100_trades":df.tail(100),
            "first_half_by_time":df[df[EXIT_T]<=med_t],"second_half_by_time":df[df[EXIT_T]>med_t]}
    rows=[]
    for name,sub in slices.items():
        s=summarize(sub[R_COL].to_numpy(dtype=float)); s["split"]=name; rows.append(s)
    return pd.DataFrame(rows)

def t4_remove_best_assets(df):
    asset_r=df.groupby("symbol")[R_COL].sum().sort_values(ascending=False); rows=[]
    for n in [0,1,3,5]:
        removed=asset_r.head(n).index.tolist(); sub=df[~df["symbol"].isin(removed)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(removed),assets_remaining=sub["symbol"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)

def t4_remove_best_months(df):
    month_r=df.groupby("month")[R_COL].sum().sort_values(ascending=False); rows=[]
    for n in [0,1,2,3]:
        removed=month_r.head(n).index.tolist(); sub=df[~df["month"].isin(removed)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(removed),months_remaining=sub["month"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)

def t4_concentration(df):
    asset_r=df.groupby("symbol")[R_COL].sum().sort_values(ascending=False); total=float(asset_r.sum())
    return dict(top1_symbol=str(asset_r.index[0]),top1_r=float(asset_r.iloc[0]),
                top1_pct=float(asset_r.iloc[0]/total) if total else 0.0,
                top3_r=float(asset_r.head(3).sum()),
                top3_pct=float(asset_r.head(3).sum()/total) if total else 0.0,
                top5_r=float(asset_r.head(5).sum()),
                top5_pct=float(asset_r.head(5).sum()/total) if total else 0.0,
                total_r=total,positive_assets=int((asset_r>0).sum()),
                negative_assets=int((asset_r<=0).sum()),total_assets=len(asset_r))

def t5_replay(df, max_open):
    df=df.sort_values(ENTRY_T).reset_index(drop=True); open_pos=[]; closed=[]
    def _flush(now):
        nonlocal open_pos
        still=[p for p in open_pos if p["exit_time"]>now]
        closed.extend(p for p in open_pos if p["exit_time"]<=now); open_pos[:]=still
    for _,row in df.iterrows():
        _flush(row[ENTRY_T])
        if any(p["symbol"]==row["symbol"] for p in open_pos): continue
        if len(open_pos)>=max_open: continue
        open_pos.append({"symbol":row["symbol"],"entry_time":row[ENTRY_T],
                          "exit_time":row[EXIT_T],R_COL:row[R_COL]})
    if open_pos:
        last=max(p["exit_time"] for p in open_pos); _flush(last+pd.Timedelta(seconds=1))
    acc=pd.DataFrame(closed)
    if acc.empty:
        return acc,dict(accepted=0,total_r=0.0,avg_r=0.0,profit_factor=0.0,max_dd_r=0.0,win_rate=0.0)
    acc=acc.sort_values("exit_time").reset_index(drop=True); r=acc[R_COL].to_numpy(dtype=float)
    return acc,dict(accepted=int(len(r)),total_r=float(r.sum()),avg_r=float(r.mean()),
                    profit_factor=_pf(r),max_dd_r=_max_dd(r),win_rate=float((r>0).mean()))

def t6_equity(acc):
    if acc.empty: return pd.DataFrame(),{}
    equity=START_CAP; peak=START_CAP; kill=False; rows=[]
    for _,row in acc.sort_values("exit_time").iterrows():
        pnl=row[R_COL]*equity*RISK_PCT; equity+=pnl; peak=max(peak,equity)
        dd_pct=(equity-peak)/peak
        if dd_pct<=KILL_SWITCH_DD and not kill: kill=True
        rows.append(dict(exit_time=row["exit_time"],symbol=row["symbol"],net_r=row[R_COL],
                         equity=equity,peak=peak,dd_pct=float(dd_pct),kill_fired=kill))
    eq_df=pd.DataFrame(rows); final=float(eq_df["equity"].iloc[-1])
    start=acc[ENTRY_T].min(); end=acc["exit_time"].max()
    years=float((end-start).days/365.25) if pd.notnull(start) and pd.notnull(end) else 1.0
    cagr=float((final/START_CAP)**(1/years)-1) if years>0 else 0.0
    return eq_df,dict(start_capital=START_CAP,end_capital=final,
                      total_return_pct=float((final-START_CAP)/START_CAP*100),
                      cagr_pct=float(cagr*100),max_dd_pct=float(eq_df["dd_pct"].min()*100),
                      years=round(years,2),kill_switch_fired=kill)


# =============================================================================
# MASTER REPORT
# =============================================================================

def write_master_report(stab, baseline, mc, cost, splits, rem_assets,
                        rem_months, conc, t5_stats, t6_stats,
                        btc_diag, trade_guard_pass):

    def gate(cond, label): return f"  {'PASS' if cond else 'FAIL'}  {label}"

    mc10=mc[mc["block_size"]==10].iloc[0]
    cost10=cost[cost["extra_cost"]==0.10].iloc[0]
    sec_half=splits[splits["split"]=="second_half_by_trade"].iloc[0]
    rem_top1=rem_assets[rem_assets["removed_n"]==1].iloc[0]
    rem_top1m=rem_months[rem_months["removed_n"]==1].iloc[0]

    all_stab_pass = all(r["avg_r"]>0 and r["total_r"]>0 for _,r in stab.iterrows())
    avg_r_ok=baseline["avg_r"]>0.15; pf_ok=baseline["profit_factor"]>1.0
    mc_ok=float(mc10["total_r_p05"])>0; cost_ok=float(cost10["profit_factor"])>1.0
    sec_ok=float(sec_half["avg_r"])>0; rem_a_ok=float(rem_top1["total_r"])>0
    rem_m_ok=float(rem_top1m["total_r"])>0
    kill_ok=not t6_stats.get("kill_switch_fired",False)
    win_ok=0.30<=baseline["win_rate"]<=0.45
    all_gates=all([avg_r_ok,pf_ok,mc_ok,cost_ok,sec_ok,rem_a_ok,rem_m_ok,
                   kill_ok,win_ok,trade_guard_pass,all_stab_pass])

    lines=[
        "PHASE T17 OPT-17.2 -- Regime Filter A: BTC Dominance Proxy Slope",
        "="*70,"",
        f"Filter   : 7-day log-price slope of BTC/USDT < 0",
        f"           (proxy for BTC dominance declining → alts gaining)",
        "Universe : 24 symbols / 1D / Donchian N=20 / ATR×2.0 / Chandelier ACT4","",
        f"Baseline : avg_r=+1.101R  PF=3.072  CAGR=+14.3%  trades=461","",
        "="*70,"BTC SLOPE DIAGNOSTICS","="*70,
        f"  BTC bars with slope computed : {btc_diag['total_btc_bars']}",
        f"  Bars with slope < 0 (enter)  : {btc_diag['neg_slope_bars']}  ({btc_diag['neg_slope_pct']:.1%})",
        f"  Bars with slope >= 0 (skip)  : {btc_diag['pos_slope_bars']}  ({1-btc_diag['neg_slope_pct']:.1%})",
        f"  Avg slope when < 0  : {btc_diag['avg_neg_slope']:.6f} log-return/day",
        f"  Avg slope when >= 0 : {btc_diag['avg_pos_slope']:.6f} log-return/day","",
        "="*70,"STABILITY ZONE","="*70,
        f"  {'N':>4s}  {'exit_n':>6s}  {'trades':>7s}  {'total_r':>8s}  {'avg_r':>8s}  {'PF':>5s}  PASS?",
        "  "+"-"*58,
    ]
    for _,r in stab.iterrows():
        ok=r["avg_r"]>0 and r["total_r"]>0
        lines.append(f"  {int(r['entry_n']):>4d}  {int(r['exit_n']):>6d}  "
                     f"{int(r['trades']):>7d}  {r['total_r']:>+8.2f}  "
                     f"{r['avg_r']:>+8.4f}  {r['profit_factor']:>5.3f}  "
                     f"{'YES' if ok else 'NO '}")
    lines+=[""  ,f"  Stability zone: {'ALL PASS' if all_stab_pass else 'FAIL'}",
            "","="*70,"T4 -- BASELINE (N=20)","="*70,
        f"  Trades   : {baseline['trades']}  [guard >={MIN_TRADES}: {'PASS' if trade_guard_pass else 'FAIL'}]",
        f"  Total R  : {baseline['total_r']:+.2f}R",
        f"  Avg R    : {baseline['avg_r']:+.4f}R  (baseline: +1.101R)",
        f"  PF       : {baseline['profit_factor']:.3f}  (baseline: 3.072)",
        f"  Win rate : {baseline['win_rate']:.1%}",
        f"  Max DD   : {baseline['max_dd_r']:+.2f}R",
        f"  t-score  : {baseline['t_score']:.2f}",
        f"  Period   : {baseline['start']} -> {baseline['end']}",
        "","="*70,"T4 -- MONTE CARLO (2000 runs, block=10)","="*70,
        f"  totalR p05/p50/p95 = {mc10['total_r_p05']:.1f} / {mc10['total_r_p50']:.1f} / {mc10['total_r_p95']:.1f}",
        f"  prob(totalR > 0)   = {mc10['prob_positive']:.1%}",
        f"  PF p05/p50         = {mc10['pf_p05']:.2f} / {mc10['pf_p50']:.2f}",
        f"  DD p95 (worst)     = {mc10['dd_p95']:.2f}R",
        "","="*70,"T4 -- COST STRESS","="*70,
    ]
    for _,r in cost.iterrows():
        lines.append(f"  +{r['extra_cost']:.2f}R -> totalR={r['total_r']:+.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- PERIOD SPLITS","="*70]
    for _,r in splits.iterrows():
        lines.append(f"  {r['split']:30s}: t={int(r['trades']):3d}  totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  win={r['win_rate']:.1%}")
    lines+=["","="*70,"T4 -- REMOVE BEST ASSETS","="*70,
            f"  Top: {conc['top1_symbol']} ({conc['top1_r']:.2f}R = {conc['top1_pct']:.1%})"]
    for _,r in rem_assets.iterrows():
        lines.append(f"  remove top {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  [{r['removed']}]")
    lines+=["","="*70,"T4 -- REMOVE BEST MONTHS","="*70]
    for _,r in rem_months.iterrows():
        lines.append(f"  remove top {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  [{r['removed']}]")
    lines+=["","="*70,"T5/T6 -- max8 PORTFOLIO","="*70,
            f"  Accepted : {t5_stats.get('accepted',0)}",
            f"  avg_r    : {t5_stats.get('avg_r',0):+.4f}R",
            f"  PF       : {t5_stats.get('profit_factor',0):.3f}"]
    if t6_stats:
        lines+=[f"  CAGR     : {t6_stats.get('cagr_pct',0):+.1f}%  (baseline: +14.3%)",
                f"  Max DD   : {t6_stats.get('max_dd_pct',0):+.1f}%  (baseline: -3.4%)",
                f"  Kill sw  : {'YES' if t6_stats.get('kill_switch_fired') else 'no'}",
                f"  Duration : {t6_stats.get('years',0):.1f} yr"]
    lines+=["","="*70,"FINAL SCORECARD","="*70,"",
        gate(trade_guard_pass,f"Trade count >={MIN_TRADES} (got {baseline['trades']})"),
        gate(all_stab_pass,"Stability zone all profitable"),
        gate(win_ok,f"§4.1 win rate 30-45% (got {baseline['win_rate']:.1%})"),
        gate(avg_r_ok,f"§4.2 avg_r > 0.15R (got {baseline['avg_r']:+.4f}R)"),
        gate(pf_ok,f"PF > 1.0 (got {baseline['profit_factor']:.3f})"),
        gate(mc_ok,f"MC p05 > 0 (got {float(mc10['total_r_p05']):+.1f}R)"),
        gate(cost_ok,f"Cost +0.10R PF > 1.0 (got {float(cost10['profit_factor']):.3f})"),
        gate(sec_ok,f"2nd-half avg_r > 0 (got {float(sec_half['avg_r']):+.4f}R)"),
        gate(rem_a_ok,f"Remove top-1 asset > 0 (got {float(rem_top1['total_r']):+.1f}R)"),
        gate(rem_m_ok,f"Remove top-1 month > 0 (got {float(rem_top1m['total_r']):+.1f}R)"),
        gate(kill_ok,"Kill switch not fired"),"",
        "  vs BASELINE:",
        f"  avg_r  : {baseline['avg_r']:+.4f}R vs +1.101R -> {'BETTER' if baseline['avg_r']>1.1011 else 'WORSE'}",
        f"  PF     : {baseline['profit_factor']:.3f} vs 3.072 -> {'BETTER' if baseline['profit_factor']>3.072 else 'WORSE'}",
    ]
    if t6_stats:
        cagr=t6_stats.get('cagr_pct',0)
        lines.append(f"  CAGR   : {cagr:+.1f}% vs +14.3% -> {'BETTER' if cagr>14.3 else 'WORSE'}")
    lines+=[f"  trades : {baseline['trades']} vs 461 (removed {461-baseline['trades']} = {(461-baseline['trades'])/461:.1%})",
            "","  NOTE: Proxy quality — 7-day BTC price slope is a directional approximation",
            "  of BTC dominance. BTC.D also depends on alt market caps (not captured here).",
            "",f"  OVERALL: {'PASS' if all_gates else 'FAIL'} -- "
               f"{'all gates cleared' if all_gates else 'one or more gates failed'}"]
    rpt=OUT_DIR/"filter_a_master_report.txt"
    rpt.write_text("\n".join(lines),encoding="utf-8")
    print(f"\n[OK] {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("="*70)
    print("Phase T17 Opt-17.2 -- Regime Filter A: BTC Dominance Proxy Slope")
    print("="*70)
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    print(f"Universe: {len(symbols)} symbols | Proxy: 7-day log-slope BTC/USDT < 0")

    print("\n[0] Building BTC slope map ...")
    slope_map = build_btc_slope_map(BTC_SLOPE_WINDOW)
    slopes = np.array(list(slope_map.values()))
    neg_slopes = slopes[slopes < BTC_SLOPE_THRESH]
    pos_slopes = slopes[slopes >= BTC_SLOPE_THRESH]
    btc_diag = dict(
        total_btc_bars=len(slopes),
        neg_slope_bars=int(len(neg_slopes)),
        pos_slope_bars=int(len(pos_slopes)),
        neg_slope_pct=float(len(neg_slopes)/len(slopes)) if len(slopes) else 0.0,
        avg_neg_slope=float(neg_slopes.mean()) if len(neg_slopes) else 0.0,
        avg_pos_slope=float(pos_slopes.mean()) if len(pos_slopes) else 0.0,
    )
    print(f"  BTC bars: {btc_diag['total_btc_bars']}  "
          f"slope<0: {btc_diag['neg_slope_bars']} ({btc_diag['neg_slope_pct']:.1%})  "
          f"slope>=0: {btc_diag['pos_slope_bars']}")

    print("\n[1] Stability zone ...")
    stab_rows=[]
    for n in STABILITY_N:
        df_n=run_universe(n,True,symbols,slope_map)
        r=df_n[R_COL].to_numpy(dtype=float) if not df_n.empty else np.array([])
        s=summarize(r)
        stab_rows.append(dict(entry_n=n,exit_n=n//2,**{k:s[k] for k in
                              ["trades","total_r","avg_r","profit_factor"]}))
        ok=s["avg_r"]>0 and s["total_r"]>0
        print(f"  N={n:2d}: trades={s['trades']:4d}  avg_r={s['avg_r']:+.4f}  "
              f"PF={s['profit_factor']:.3f}  {'PASS' if ok else 'FAIL'}")
    stab_df=pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT_DIR/"filter_a_stability_zone.csv",index=False)

    print(f"\n[2] Canonical N={CANONICAL_N} ...")
    trades=run_universe(CANONICAL_N,True,symbols,slope_map)
    if trades.empty:
        print("  ERROR: No trades."); return 1
    trades.to_csv(OUT_DIR/"filter_a_all_trades_N20.csv",index=False)
    tc=len(trades); tgp=tc>=MIN_TRADES
    print(f"  Trades: {tc}  Symbols: {trades['symbol'].nunique()}  "
          f"[guard: {'PASS' if tgp else 'FAIL'}]")
    print(f"  Period: {trades[EXIT_T].min().date()} -> {trades[EXIT_T].max().date()}")

    print("\n[3] T4 robustness ...")
    r=trades[R_COL].to_numpy(dtype=float)
    baseline=summarize(r)
    baseline.update(assets=trades["symbol"].nunique(),months=trades["month"].nunique(),
                    start=str(trades[EXIT_T].min().date()),end=str(trades[EXIT_T].max().date()))
    mc=t4_montecarlo(trades); cost=t4_cost_stress(trades)
    splits=t4_period_splits(trades); rem_assets=t4_remove_best_assets(trades)
    rem_months=t4_remove_best_months(trades); conc=t4_concentration(trades)
    for obj,name in [(mc,"mc"),(cost,"cost"),(splits,"splits"),
                     (rem_assets,"remove_best_assets"),(rem_months,"remove_best_months")]:
        obj.to_csv(OUT_DIR/f"filter_a_t4_{name}.csv",index=False)
    mc10=mc[mc["block_size"]==10].iloc[0]
    print(f"  avg_r={baseline['avg_r']:+.4f}  PF={baseline['profit_factor']:.3f}  t={baseline['t_score']:.2f}")
    print(f"  MC p05={mc10['total_r_p05']:+.1f}  prob_pos={mc10['prob_positive']:.1%}")

    print(f"\n[4] T5 max{PORT_MAX_OPEN} + T6 ...")
    acc,t5_stats=t5_replay(trades,PORT_MAX_OPEN)
    t6_stats={}
    if not acc.empty:
        eq_df,t6_stats=t6_equity(acc)
        eq_df.to_csv(OUT_DIR/"filter_a_t6_equity.csv",index=False)
    print(f"  Accepted: {t5_stats.get('accepted',0)}  "
          f"avgR={t5_stats.get('avg_r',0):+.4f}  PF={t5_stats.get('profit_factor',0):.3f}")
    if t6_stats:
        print(f"  CAGR={t6_stats.get('cagr_pct',0):+.1f}%  DD={t6_stats.get('max_dd_pct',0):+.1f}%")

    print("\n[5] Writing master report ...")
    write_master_report(stab_df,baseline,mc,cost,splits,rem_assets,
                        rem_months,conc,t5_stats,t6_stats,btc_diag,tgp)

    print("\n"+"="*70)
    print(f"Filter A complete -> {OUT_DIR}")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
