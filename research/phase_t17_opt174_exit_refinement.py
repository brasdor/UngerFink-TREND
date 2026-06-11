#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.4 -- Exit Refinement (4 Variants)

Baseline: DonchianLong_UniverseV2 / N=20 / ema200_price / ATR×2.0 /
          Chandelier ACT=4R, trail=3×ATR / max8
          avg_r=+1.101R  PF=3.072  CAGR=+14.3%  trades=461

Variant A -- Chandelier activation threshold grid
  Test ACT = [2R, 3R, 4R, 6R, 8R], trail fixed at 3×ATR.
  Stability: zone [prev, ACT, next] in grid must have >=67% beating baseline avg_r.
  Full T4 on best stable candidate.

Variant B -- Chandelier trail multiplier grid
  Test trail = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0], ACT fixed at 4R.
  Stability: zone [prev, trail, next] in grid must have >=67% beating baseline avg_r.
  Full T4 on best stable candidate.

Variant C -- Partial profit taking
  100% position at entry. Take 50% off at +4R fixed target; trail remaining
  50% with Chandelier ACT4_ATR3. net_r = 0.5*4R + 0.5*r_remaining if partial
  triggered, else full r (stop/midline exit before +4R).

Variant D -- Time-based exit backstop
  Exit any trade still open after 60 calendar days at close of day 60.
  Normal exits (stop, chandelier, midline) take precedence if triggered earlier.

All variants: Universe 24 symbols / output data/research_donchian_exitV2_{A,B,C,D}/
No pauses between variants. Results printed as single comparison table at end.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT       = Path(__file__).resolve().parents[1]
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# Frozen base config (entry unchanged)
ENTRY_N   = 20
EXIT_N    = ENTRY_N // 2
ATR_N     = 14
STOP_MULT = 2.0

# Baseline chandelier params
BASE_ACT   = 4.0   # R
BASE_TRAIL = 3.0   # ATR multiplier

# Variant A grid  (ACT values, trail fixed at 3.0)
GRID_ACT   = [2.0, 3.0, 4.0, 6.0, 8.0]

# Variant B grid  (trail values, ACT fixed at 4.0)
GRID_TRAIL = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# Variant C
PARTIAL_TARGET_R = 4.0  # take 50% off at +4R

# Variant D
TIME_BACKSTOP_DAYS = 60

# Stability
STABILITY_THRESH = 0.67

# T4
MC_RUNS     = 2000
MC_BLOCKS   = [1, 3, 5, 10, 20]
MC_SEED     = 42
EXTRA_COSTS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# T6
START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35
PORT_MAX_OPEN  = 8

# Baseline reference
BL = dict(avg_r=1.1011, pf=3.072, cagr=14.3, trades=461)
MIN_TRADES = 80


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1.0)
    out[n-1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i-1]):
            out[i] = close[i]*k + out[i-1]*(1.0-k)
    return out

def _atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, n: int) -> np.ndarray:
    nb = len(cl); tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1:n+1]))
        for i in range(n+1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i-1]):
                atr[i] = (atr[i-1]*(n-1)+tr[i])/n
    return atr


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
            .dropna(subset=["time","close"]).sort_values("time").reset_index(drop=True))


# =============================================================================
# PARAMETRIC BACKTEST
# All exit variants share this single function; behaviour is controlled by args.
# =============================================================================

def run_backtest(df: pd.DataFrame, symbol: str,
                 act_r:      float = BASE_ACT,
                 trail:      float = BASE_TRAIL,
                 partial_r:  float = 0.0,    # 0 = disabled; >0 = partial target in R
                 time_days:  int   = 0,       # 0 = disabled
                 ) -> List[dict]:

    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(ENTRY_N, ATR_N, 200) + 20: return []

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()

    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(ENTRY_N).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(EXIT_N).min().to_numpy()

    trades: List[dict] = []
    pos: Optional[dict] = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])): continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])): continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])): continue

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["e"])/pos["r"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["e"])/pos["r"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            # ── Partial profit check (Variant C) ─────────────────────────
            if partial_r > 0 and not pos.get("partial_done", False):
                if hi[i] >= pos["e"] + partial_r * pos["r"]:
                    pos["partial_done"] = True
                    pos["chan_active"]  = True   # force chandelier on
                    pos["chan_stop"]    = max(pos["chan_stop"],
                                             pos["hh"] - atr14[i]*trail)

            # ── Time backstop (Variant D) ─────────────────────────────────
            days_held = 0
            if time_days > 0:
                days_held = int((ts[i] - pos["entry_time"]) / np.timedelta64(1, "D"))

            # ── Exit checks ───────────────────────────────────────────────
            exit_px = exit_rsn = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_rsn = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_rsn = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_rsn = cl[i], "midline_exit"
            elif time_days > 0 and days_held >= time_days:
                exit_px, exit_rsn = cl[i], "time_backstop"

            if exit_px is not None:
                r_full = (exit_px - pos["e"]) / pos["r"]
                if partial_r > 0 and pos.get("partial_done", False):
                    net_r = 0.5 * partial_r + 0.5 * r_full
                else:
                    net_r = r_full
                trades.append(dict(symbol=symbol,
                    entry_time=pos["entry_time"], exit_time=ts[i],
                    net_r=net_r,
                    mae_r=min(pos["mae_r"], r_full), mfe_r=pos["mfe_r"],
                    exit_reason=exit_rsn,
                    partial_done=pos.get("partial_done", False)))
                pos = None
                continue

            # ── Update chandelier ─────────────────────────────────────────
            if pos["mfe_r"] >= act_r: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*trail)

        # ── ENTRY ────────────────────────────────────────────────────────────
        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            risk = atr14[i-1] * STOP_MULT
            if risk <= 0: continue
            stop = cl[i] - risk
            pos = dict(e=cl[i], stop=stop, r=risk, entry_time=ts[i],
                       hh=hi[i], chan_active=False, chan_stop=stop,
                       mfe_r=0., mae_r=0., bars=1,
                       partial_done=False)

    if pos is not None:
        exit_px = cl[-1]; r_full = (exit_px-pos["e"])/pos["r"]
        net_r = (0.5*partial_r + 0.5*r_full
                 if partial_r>0 and pos.get("partial_done",False)
                 else r_full)
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1], net_r=net_r,
            mae_r=min(pos["mae_r"],r_full), mfe_r=pos["mfe_r"],
            exit_reason="end_of_data", partial_done=pos.get("partial_done",False)))
    return trades


def run_universe(symbols: List[str], **kwargs) -> pd.DataFrame:
    all_trades = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(run_backtest(df, sym, **kwargs))
    if not all_trades: return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[EXIT_T, ENTRY_T, R_COL]).sort_values(EXIT_T).reset_index(drop=True)
    df["month"] = df[EXIT_T].dt.tz_convert(None).dt.to_period("M").astype(str)
    return df


# =============================================================================
# STATS / T4 / T5 / T6
# =============================================================================

def _pf(r):
    g=r[r>0].sum(); l=-r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.)

def _mdd(r):
    if r.size==0: return 0.
    eq=np.cumsum(r); pk=np.maximum.accumulate(eq); return float((eq-pk).min())

def summarize(r: np.ndarray) -> dict:
    if r.size==0:
        return dict(trades=0,total_r=0.,avg_r=0.,win_rate=0.,profit_factor=0.,
                    max_dd_r=0.,std_r=0.,t_score=0.)
    avg=float(r.mean()); std=float(r.std(ddof=1)) if r.size>1 else 0.
    t=avg/(std/math.sqrt(r.size)) if std>0 else 0.
    return dict(trades=int(r.size),total_r=float(r.sum()),avg_r=avg,
                win_rate=float((r>0).mean()),profit_factor=_pf(r),
                max_dd_r=_mdd(r),std_r=std,t_score=float(t))

def _bstrap(vals, bs, rng):
    n=len(vals); out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); e=min(s+bs,n); blk=vals[s:e]
        if len(blk)<bs: blk=np.concatenate([blk,vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n],dtype=float)

def t4_full(df: pd.DataFrame) -> dict:
    vals=df[R_COL].to_numpy(dtype=float)
    rng=np.random.default_rng(MC_SEED)
    mc_rows=[]
    for bs in MC_BLOCKS:
        tots,dds,pfs=[],[],[]
        for _ in range(MC_RUNS):
            s=_bstrap(vals,bs,rng); tots.append(s.sum()); dds.append(_mdd(s)); pfs.append(_pf(s))
        tots=np.array(tots); dds=np.array(dds); pfs=np.array(pfs)
        mc_rows.append(dict(block_size=bs,
            total_r_p05=float(np.percentile(tots,5)),total_r_p50=float(np.percentile(tots,50)),
            total_r_p95=float(np.percentile(tots,95)),dd_p95=float(np.percentile(dds,95)),
            pf_p05=float(np.percentile(pfs,5)),pf_p50=float(np.percentile(pfs,50)),
            prob_positive=float((tots>0).mean())))
    mc=pd.DataFrame(mc_rows)
    cost_rows=[]
    for ec in EXTRA_COSTS:
        s=summarize(vals-ec); s["extra_cost"]=ec; cost_rows.append(s)
    cost=pd.DataFrame(cost_rows)
    df2=df.sort_values(EXIT_T).reset_index(drop=True)
    mid=len(df2)//2; med_t=df2[EXIT_T].median()
    split_rows=[]
    for name,sub in [("first_half_by_trade",df2.iloc[:mid]),
                     ("second_half_by_trade",df2.iloc[mid:]),
                     ("last_100_trades",df2.tail(100)),
                     ("first_half_by_time",df2[df2[EXIT_T]<=med_t]),
                     ("second_half_by_time",df2[df2[EXIT_T]>med_t])]:
        s=summarize(sub[R_COL].to_numpy(dtype=float)); s["split"]=name; split_rows.append(s)
    splits=pd.DataFrame(split_rows)
    asset_r=df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    ra_rows=[]
    for n in [0,1,3,5]:
        rem=asset_r.head(n).index.tolist(); sub=df[~df["symbol"].isin(rem)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(rem),assets_remaining=sub["symbol"].nunique())
        ra_rows.append(s)
    rem_assets=pd.DataFrame(ra_rows)
    month_r=df.groupby("month")[R_COL].sum().sort_values(ascending=False)
    rm_rows=[]
    for n in [0,1,2,3]:
        rem=month_r.head(n).index.tolist(); sub=df[~df["month"].isin(rem)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(rem),months_remaining=sub["month"].nunique())
        rm_rows.append(s)
    rem_months=pd.DataFrame(rm_rows)
    return dict(mc=mc,cost=cost,splits=splits,rem_assets=rem_assets,rem_months=rem_months)

def t5t6(df: pd.DataFrame) -> Tuple[dict, dict]:
    df2=df.sort_values(ENTRY_T).reset_index(drop=True)
    open_pos=[]; closed=[]
    def _flush(now):
        nonlocal open_pos
        still=[p for p in open_pos if p["exit_time"]>now]
        closed.extend(p for p in open_pos if p["exit_time"]<=now); open_pos[:]=still
    for _,row in df2.iterrows():
        _flush(row[ENTRY_T])
        if any(p["symbol"]==row["symbol"] for p in open_pos): continue
        if len(open_pos)>=PORT_MAX_OPEN: continue
        open_pos.append({"symbol":row["symbol"],"entry_time":row[ENTRY_T],
                          "exit_time":row[EXIT_T],R_COL:row[R_COL]})
    if open_pos:
        last=max(p["exit_time"] for p in open_pos); _flush(last+pd.Timedelta(seconds=1))
    acc=pd.DataFrame(closed)
    if acc.empty: return {}, {}
    acc=acc.sort_values("exit_time").reset_index(drop=True)
    r=acc[R_COL].to_numpy(dtype=float)
    t5=dict(accepted=int(len(r)),total_r=float(r.sum()),avg_r=float(r.mean()),
            profit_factor=_pf(r),win_rate=float((r>0).mean()))
    equity=START_CAP; peak=START_CAP; rows=[]
    for _,row in acc.sort_values("exit_time").iterrows():
        pnl=row[R_COL]*equity*RISK_PCT; equity+=pnl; peak=max(peak,equity)
        rows.append(dict(exit_time=row["exit_time"],symbol=row["symbol"],net_r=row[R_COL],
                         equity=equity,peak=peak,dd_pct=(equity-peak)/peak))
    eq=pd.DataFrame(rows); final=float(eq["equity"].iloc[-1])
    start=acc[ENTRY_T].min(); end=acc["exit_time"].max()
    years=float((end-start).days/365.25) if pd.notnull(start) and pd.notnull(end) else 1.
    cagr=float((final/START_CAP)**(1/years)-1) if years>0 else 0.
    t6=dict(cagr_pct=float(cagr*100),max_dd_pct=float(eq["dd_pct"].min()*100),
            end_capital=final,kill_switch_fired=False,years=round(years,2))
    return t5, t6


# =============================================================================
# GRID STABILITY CHECKER
# =============================================================================

def zone_stability(grid: List[float], grid_stats: Dict[float,dict]) -> Dict[float,dict]:
    """
    For each value in grid, compute the zone (prev, val, next) and
    check what fraction beats BL["avg_r"].
    Returns {val: {"zone_vals": [...], "zone_pass": bool, "zone_pct": float}}
    """
    results = {}
    for idx, val in enumerate(grid):
        zone = []
        if idx > 0:          zone.append(grid[idx-1])
        zone.append(val)
        if idx < len(grid)-1: zone.append(grid[idx+1])
        n_beat = sum(1 for v in zone if grid_stats[v]["avg_r"] > BL["avg_r"])
        pct    = n_beat / len(zone)
        results[val] = dict(zone_vals=zone, n_beat=n_beat,
                            zone_pct=pct, zone_pass=pct >= STABILITY_THRESH)
    return results


# =============================================================================
# VARIANT RUNNERS
# =============================================================================

def run_variant_grid(variant: str, param_name: str, grid: List[float],
                     fixed_kwargs: dict, symbols: List[str],
                     out_dir: Path) -> dict:
    """
    Run grid scan for Variant A or B.
    Returns the metrics dict for the best stable candidate
    (or canonical if nothing beats baseline stably).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid scan
    grid_stats: Dict[float, dict] = {}
    for val in grid:
        kw = {**fixed_kwargs, param_name: val}
        df_t = run_universe(symbols, **kw)
        if df_t.empty:
            grid_stats[val] = dict(avg_r=0., pf=0., trades=0, total_r=0.)
        else:
            r = df_t[R_COL].to_numpy(dtype=float)
            grid_stats[val] = dict(avg_r=float(r.mean()), pf=_pf(r),
                                   trades=int(len(r)), total_r=float(r.sum()))

    # Stability check per zone
    stab = zone_stability(grid, grid_stats)

    # Save grid CSV
    rows = []
    for val in grid:
        s = grid_stats[val]; z = stab[val]
        rows.append(dict(**{param_name: val}, **s,
                         beats_baseline=s["avg_r"]>BL["avg_r"],
                         zone_pct=z["zone_pct"], zone_pass=z["zone_pass"]))
    pd.DataFrame(rows).to_csv(out_dir / f"variant_{variant.lower()}_grid.csv", index=False)

    # Identify best stable candidate (highest avg_r among stable values)
    stable_improving = [v for v in grid
                        if stab[v]["zone_pass"] and grid_stats[v]["avg_r"] > BL["avg_r"]]
    if stable_improving:
        best_val = max(stable_improving, key=lambda v: grid_stats[v]["avg_r"])
    else:
        best_val = None

    # Full T4 on best (or canonical if none)
    run_val = best_val if best_val is not None else (BASE_ACT if param_name=="act_r" else BASE_TRAIL)
    kw = {**fixed_kwargs, param_name: run_val}
    df_best = run_universe(symbols, **kw)
    df_best.to_csv(out_dir / f"variant_{variant.lower()}_best_trades.csv", index=False)

    r = df_best[R_COL].to_numpy(dtype=float)
    base_stats = summarize(r)
    base_stats.update(start=str(df_best[EXIT_T].min().date()),
                      end=str(df_best[EXIT_T].max().date()),
                      assets=df_best["symbol"].nunique())
    t4 = t4_full(df_best)
    t5, t6 = t5t6(df_best)

    for key, obj in t4.items():
        obj.to_csv(out_dir / f"variant_{variant.lower()}_t4_{key}.csv", index=False)
    if not df_best.empty and t6:
        # save equity curve placeholder (not critical)
        pass

    mc10  = t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10 = t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec   = t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1   = t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1   = t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]
    all_gates = all([
        base_stats["avg_r"]>0.15, base_stats["profit_factor"]>1.,
        float(mc10["total_r_p05"])>0, float(cost10["profit_factor"])>1.,
        float(sec["avg_r"])>0, float(ra1["total_r"])>0, float(rm1["total_r"])>0,
        not t6.get("kill_switch_fired",False),
        0.30<=base_stats["win_rate"]<=0.45,
        base_stats["trades"]>=MIN_TRADES,
    ])

    # Write grid report
    write_grid_report(out_dir, variant, param_name, grid, grid_stats, stab,
                      best_val, run_val, base_stats, t4, t5, t6)

    return dict(
        label=variant, best_param_val=run_val, found_improvement=best_val is not None,
        trades=base_stats["trades"], avg_r=base_stats["avg_r"],
        pf=base_stats["profit_factor"], win_rate=base_stats["win_rate"],
        t_score=base_stats["t_score"],
        mc_p05=float(mc10["total_r_p05"]),
        cagr=t6.get("cagr_pct",0.), max_dd=t6.get("max_dd_pct",0.),
        all_gates=all_gates,
    )


def run_variant_single(variant: str, desc: str, symbols: List[str],
                       out_dir: Path, **kwargs) -> dict:
    """Full run for Variant C or D (single parameter set)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_universe(symbols, **kwargs)
    if df.empty:
        return dict(label=variant, desc=desc, trades=0, avg_r=0., pf=0.,
                    cagr=0., max_dd=0., mc_p05=0., t_score=0., win_rate=0.,
                    all_gates=False)

    df.to_csv(out_dir / f"variant_{variant.lower()}_trades.csv", index=False)
    r = df[R_COL].to_numpy(dtype=float)
    base = summarize(r)
    base.update(start=str(df[EXIT_T].min().date()), end=str(df[EXIT_T].max().date()),
                assets=df["symbol"].nunique())
    t4 = t4_full(df)
    t5, t6 = t5t6(df)

    for key, obj in t4.items():
        obj.to_csv(out_dir / f"variant_{variant.lower()}_t4_{key}.csv", index=False)

    mc10   = t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10 = t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec    = t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1    = t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1    = t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]
    all_gates = all([
        base["avg_r"]>0.15, base["profit_factor"]>1.,
        float(mc10["total_r_p05"])>0, float(cost10["profit_factor"])>1.,
        float(sec["avg_r"])>0, float(ra1["total_r"])>0, float(rm1["total_r"])>0,
        not t6.get("kill_switch_fired",False),
        0.30<=base["win_rate"]<=0.45,
        base["trades"]>=MIN_TRADES,
    ])

    write_single_report(out_dir, variant, desc, base, t4, t5, t6, kwargs)

    return dict(
        label=variant, desc=desc,
        trades=base["trades"], avg_r=base["avg_r"], pf=base["profit_factor"],
        win_rate=base["win_rate"], t_score=base["t_score"],
        mc_p05=float(mc10["total_r_p05"]),
        cagr=t6.get("cagr_pct",0.), max_dd=t6.get("max_dd_pct",0.),
        all_gates=all_gates,
    )


# =============================================================================
# REPORT WRITERS
# =============================================================================

def _gate(cond, label):
    return f"  {'PASS' if cond else 'FAIL'}  {label}"

def write_grid_report(out_dir, variant, param_name, grid, grid_stats, stab,
                      best_val, run_val, base, t4, t5, t6):
    mc10=t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10=t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec=t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1=t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1=t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]

    lines=[f"PHASE T17 OPT-17.4 -- Exit Refinement Variant {variant}","="*70,"",
           f"Parameter: {param_name}  |  Grid: {grid}","",
           f"Baseline:  avg_r=+{BL['avg_r']:.4f}R  PF={BL['pf']:.3f}  CAGR=+{BL['cagr']:.1f}%","",
           "="*70,"GRID SCAN RESULTS","="*70,"",
           f"  {'Value':>8s}  {'Trades':>7s}  {'avg_r':>8s}  {'PF':>5s}  "
           f"{'Beats?':>6s}  {'Zone%':>6s}  {'Stable?':>7s}",
           "  "+"-"*56,]
    for val in grid:
        s=grid_stats[val]; z=stab[val]
        bl_flag="YES" if s["avg_r"]>BL["avg_r"] else "no "
        st_flag="YES" if z["zone_pass"] else "no "
        canon=" <-" if val==run_val else ""
        lines.append(f"  {val:>8.1f}  {s['trades']:>7d}  {s['avg_r']:>+8.4f}  "
                     f"{s['pf']:>5.3f}  {bl_flag:>6s}  {z['zone_pct']:>5.1%}  "
                     f"{st_flag:>7s}{canon}")
    lines+=[""  ,
        f"  Best stable candidate: {best_val if best_val else 'NONE (using canonical '+str(run_val)+')'}",
        f"  Running full T4 on   : {run_val}",
        "","="*70,f"FULL T4 RESULTS (run_val={run_val})","="*70,
        f"  Trades   : {base['trades']}",f"  Avg R    : {base['avg_r']:+.4f}R  (baseline: +1.101R)",
        f"  PF       : {base['profit_factor']:.3f}  (baseline: 3.072)",
        f"  Win rate : {base['win_rate']:.1%}",f"  t-score  : {base['t_score']:.2f}",
        f"  Period   : {base['start']} -> {base['end']}",
        "","="*70,"T4 -- MONTE CARLO (bs=10)","="*70,
        f"  p05/p50/p95 = {mc10['total_r_p05']:.1f} / {mc10['total_r_p50']:.1f} / {mc10['total_r_p95']:.1f}",
        f"  prob>0 = {mc10['prob_positive']:.1%}",
        "","="*70,"T4 -- COST STRESS","="*70,]
    for _,r in t4["cost"].iterrows():
        lines.append(f"  +{r['extra_cost']:.2f}R -> avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- PERIOD SPLITS","="*70]
    for _,r in t4["splits"].iterrows():
        lines.append(f"  {r['split']:30s}: t={int(r['trades']):3d}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- REMOVE BEST ASSETS","="*70]
    for _,r in t4["rem_assets"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- REMOVE BEST MONTHS","="*70]
    for _,r in t4["rem_months"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T5/T6","="*70,
            f"  Accepted: {t5.get('accepted',0)}  avgR={t5.get('avg_r',0):+.4f}",
            f"  CAGR={t6.get('cagr_pct',0):+.1f}%  DD={t6.get('max_dd_pct',0):+.1f}%"]
    lines+=["","="*70,"GATE RESULTS","="*70,"",
        _gate(base["trades"]>=MIN_TRADES,f"Trade count >={MIN_TRADES} (got {base['trades']})"),
        _gate(0.30<=base["win_rate"]<=0.45,f"win rate 30-45% (got {base['win_rate']:.1%})"),
        _gate(base["avg_r"]>0.15,f"avg_r > 0.15R (got {base['avg_r']:+.4f}R)"),
        _gate(base["profit_factor"]>1.,f"PF > 1.0 (got {base['profit_factor']:.3f})"),
        _gate(float(mc10["total_r_p05"])>0,f"MC p05 > 0 (got {float(mc10['total_r_p05']):+.1f}R)"),
        _gate(float(cost10["profit_factor"])>1.,f"cost +0.10R PF > 1.0 (got {float(cost10['profit_factor']):.3f})"),
        _gate(float(sec["avg_r"])>0,f"2nd half avgR > 0 (got {float(sec['avg_r']):+.4f}R)"),
        _gate(float(ra1["total_r"])>0,f"remove top-1 asset > 0 (got {float(ra1['total_r']):+.1f}R)"),
        _gate(float(rm1["total_r"])>0,f"remove top-1 month > 0 (got {float(rm1['total_r']):+.1f}R)"),]
    rpt=out_dir/f"variant_{variant.lower()}_report.txt"
    rpt.write_text("\n".join(lines),encoding="utf-8")


def write_single_report(out_dir, variant, desc, base, t4, t5, t6, kwargs):
    mc10=t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10=t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec=t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1=t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1=t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]

    lines=[f"PHASE T17 OPT-17.4 -- Exit Refinement Variant {variant}","="*70,"",
           f"Variant: {desc}","Config: {kwargs}","",
           f"Baseline: avg_r=+1.101R  PF=3.072  CAGR=+14.3%","",
           "="*70,"T4 RESULTS","="*70,
           f"  Trades   : {base['trades']}",
           f"  Avg R    : {base['avg_r']:+.4f}R  (baseline: +1.101R)",
           f"  PF       : {base['profit_factor']:.3f}  (baseline: 3.072)",
           f"  Win rate : {base['win_rate']:.1%}",f"  t-score  : {base['t_score']:.2f}",
           f"  Period   : {base['start']} -> {base['end']}",
           "","T4 -- MONTE CARLO (bs=10):",
           f"  p05/p50/p95 = {mc10['total_r_p05']:.1f}/{mc10['total_r_p50']:.1f}/{mc10['total_r_p95']:.1f}",
           f"  prob>0={mc10['prob_positive']:.1%}",
           "","T4 -- COST STRESS:"]
    for _,r in t4["cost"].iterrows():
        lines.append(f"  +{r['extra_cost']:.2f}R -> avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","T4 -- PERIOD SPLITS:"]
    for _,r in t4["splits"].iterrows():
        lines.append(f"  {r['split']:30s}: t={int(r['trades']):3d}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","T4 -- REMOVE BEST ASSETS:"]
    for _,r in t4["rem_assets"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}")
    lines+=["","T4 -- REMOVE BEST MONTHS:"]
    for _,r in t4["rem_months"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}")
    lines+=["",f"T5/T6: accepted={t5.get('accepted',0)}  CAGR={t6.get('cagr_pct',0):+.1f}%  DD={t6.get('max_dd_pct',0):+.1f}%",
           "","GATE RESULTS:",
        _gate(base["trades"]>=MIN_TRADES,f"trades >={MIN_TRADES} (got {base['trades']})"),
        _gate(0.30<=base["win_rate"]<=0.45,f"win% 30-45% (got {base['win_rate']:.1%})"),
        _gate(base["avg_r"]>0.15,f"avg_r > 0.15R (got {base['avg_r']:+.4f}R)"),
        _gate(base["profit_factor"]>1.,f"PF > 1.0 (got {base['profit_factor']:.3f})"),
        _gate(float(mc10["total_r_p05"])>0,f"MC p05 > 0 (got {float(mc10['total_r_p05']):+.1f}R)"),
        _gate(float(cost10["profit_factor"])>1.,f"cost +0.10R PF > 1.0"),
        _gate(float(sec["avg_r"])>0,f"2nd half avgR > 0 (got {float(sec['avg_r']):+.4f}R)"),
        _gate(float(ra1["total_r"])>0,f"remove top-1 asset > 0 (got {float(ra1['total_r']):+.1f}R)"),
        _gate(float(rm1["total_r"])>0,f"remove top-1 month > 0 (got {float(rm1['total_r']):+.1f}R)"),
        "","vs BASELINE:",
        f"  avg_r: {base['avg_r']:+.4f}R vs +1.101R -> {'BETTER' if base['avg_r']>1.1011 else 'WORSE'}",
        f"  PF   : {base['profit_factor']:.3f} vs 3.072 -> {'BETTER' if base['profit_factor']>3.072 else 'WORSE'}",
        f"  CAGR : {t6.get('cagr_pct',0):+.1f}% vs +14.3% -> {'BETTER' if t6.get('cagr_pct',0)>14.3 else 'WORSE'}",]
    rpt=out_dir/f"variant_{variant.lower()}_report.txt"
    rpt.write_text("\n".join(lines),encoding="utf-8")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(results: List[dict]) -> None:
    SEP = "=" * 80
    print()
    print(SEP)
    print("OPT-17.4 EXIT REFINEMENT - COMBINED COMPARISON TABLE")
    print(SEP)
    print()

    hdr = (f"  {'Variant':<18s}  {'Trades':>6s}  {'dTrades':>8s}  "
           f"{'avg_r':>8s}  {'PF':>5s}  {'CAGR':>6s}  {'DD':>6s}  "
           f"{'MC_p05':>8s}  {'t':>5s}  {'Win%':>5s}  {'Gates':>5s}")
    print(hdr)
    print("  " + "-" * 78)
    print(f"  {'Baseline':<18s}  {461:>6d}  {'---':>8s}  "
          f"{'+1.1011R':>8s}  {'3.072':>5s}  {'+14.3%':>6s}  {'-3.4%':>6s}  "
          f"{'+347.7R':>8s}  {'7.33':>5s}  {'41.2%':>5s}  {'ref':>5s}")
    print("  " + "-" * 78)

    for r in results:
        delta = r["trades"] - 461
        gates = "PASS" if r["all_gates"] else "FAIL"
        label = r["label"]
        # Extra info for grid variants
        if "best_param_val" in r:
            impr = " ^" if r["found_improvement"] else "  "
            label = f"{r['label']}({r['best_param_val']:.1f}R){impr}" if r["label"]=="A" else \
                    f"{r['label']}({r['best_param_val']:.1f}x){impr}"
        print(f"  {label:<18s}  {r['trades']:>6d}  {delta:>+8d}  "
              f"{r['avg_r']:>+8.4f}  {r['pf']:>5.3f}  {r['cagr']:>+6.1f}%  "
              f"{r['max_dd']:>+6.1f}%  {r['mc_p05']:>+8.1f}R  "
              f"{r['t_score']:>5.2f}  {r['win_rate']:>5.1%}  {gates:>5s}")

    print()
    print("  Gate failures per variant:")
    for r in results:
        fails = []
        if r["mc_p05"] <= 0:    fails.append("MC_p05<=0")
        if r["avg_r"] <= 0.15:  fails.append("avg_r<=0.15")
        if r["pf"] <= 1.0:      fails.append("PF<=1.0")
        if not (0.30 <= r["win_rate"] <= 0.45): fails.append("win%")
        if r["trades"] < MIN_TRADES: fails.append(f"trades<{MIN_TRADES}")
        print(f"  Variant {r['label']}: {', '.join(fails) if fails else 'none'}")

    print()
    print("  Benchmark (vs +1.101R / 3.072 / +14.3%):")
    for r in results:
        rr = "^BETTER" if r["avg_r"]>1.1011 else "vWORSE "
        pr = "^BETTER" if r["pf"]>3.072     else "vWORSE "
        cr = "^BETTER" if r["cagr"]>14.3    else "vWORSE "
        print(f"  Variant {r['label']}: avg_r {rr} (+{r['avg_r']:.4f}R)  "
              f"PF {pr} ({r['pf']:.3f})  CAGR {cr} ({r['cagr']:+.1f}%)")
    print()
    print(SEP)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    results: List[dict] = []

    # ── VARIANT A: ACT grid (trail fixed 3.0) ───────────────────────────────
    print("="*60)
    print("[VARIANT A] Chandelier ACT threshold grid")
    print(f"  Grid: {GRID_ACT}  |  trail fixed at {BASE_TRAIL}")
    print("="*60)
    out_a = ROOT / "data" / "research_donchian_exitV2_A"
    res_a = run_variant_grid("A", "act_r", GRID_ACT,
                             fixed_kwargs=dict(trail=BASE_TRAIL),
                             symbols=symbols, out_dir=out_a)
    results.append(res_a)
    impr_str = f"best={res_a['best_param_val']:.1f}R" if res_a["found_improvement"] else "no improvement"
    print(f"  -> {impr_str}  avg_r={res_a['avg_r']:+.4f}  PF={res_a['pf']:.3f}  "
          f"CAGR={res_a['cagr']:+.1f}%  gates={'PASS' if res_a['all_gates'] else 'FAIL'}")

    # ── VARIANT B: trail grid (ACT fixed 4R) ────────────────────────────────
    print()
    print("="*60)
    print("[VARIANT B] Chandelier trail multiplier grid")
    print(f"  Grid: {GRID_TRAIL}  |  ACT fixed at {BASE_ACT}R")
    print("="*60)
    out_b = ROOT / "data" / "research_donchian_exitV2_B"
    res_b = run_variant_grid("B", "trail", GRID_TRAIL,
                             fixed_kwargs=dict(act_r=BASE_ACT),
                             symbols=symbols, out_dir=out_b)
    results.append(res_b)
    impr_str = f"best={res_b['best_param_val']:.1f}x" if res_b["found_improvement"] else "no improvement"
    print(f"  -> {impr_str}  avg_r={res_b['avg_r']:+.4f}  PF={res_b['pf']:.3f}  "
          f"CAGR={res_b['cagr']:+.1f}%  gates={'PASS' if res_b['all_gates'] else 'FAIL'}")

    # ── VARIANT C: partial profit at +4R ────────────────────────────────────
    print()
    print("="*60)
    print("[VARIANT C] Partial profit at +4R, trail remaining with Chandelier ACT4_ATR3")
    print("="*60)
    out_c = ROOT / "data" / "research_donchian_exitV2_C"
    res_c = run_variant_single("C",
        "Partial 50% at +4R, trail 50% with Chandelier ACT4_ATR3",
        symbols, out_c,
        act_r=BASE_ACT, trail=BASE_TRAIL, partial_r=PARTIAL_TARGET_R)
    results.append(res_c)
    print(f"  -> trades={res_c['trades']}  avg_r={res_c['avg_r']:+.4f}  "
          f"PF={res_c['pf']:.3f}  CAGR={res_c['cagr']:+.1f}%  "
          f"gates={'PASS' if res_c['all_gates'] else 'FAIL'}")

    # ── VARIANT D: 60-day time backstop ─────────────────────────────────────
    print()
    print("="*60)
    print(f"[VARIANT D] Time backstop: exit after {TIME_BACKSTOP_DAYS} calendar days")
    print("="*60)
    out_d = ROOT / "data" / "research_donchian_exitV2_D"
    res_d = run_variant_single("D",
        f"Time backstop at {TIME_BACKSTOP_DAYS} calendar days",
        symbols, out_d,
        act_r=BASE_ACT, trail=BASE_TRAIL, time_days=TIME_BACKSTOP_DAYS)
    results.append(res_d)
    print(f"  -> trades={res_d['trades']}  avg_r={res_d['avg_r']:+.4f}  "
          f"PF={res_d['pf']:.3f}  CAGR={res_d['cagr']:+.1f}%  "
          f"gates={'PASS' if res_d['all_gates'] else 'FAIL'}")

    print_comparison_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
