#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.4 -- Combined Exit Config Test

Tests three combined configurations:
  Config 1: ACT=8R  + trail=5.0x ATR
  Config 2: ACT=6R  + trail=5.0x ATR
  Config 3: Best(C1/C2) + 60-day time backstop (Variant D layered on top)

For each:
  Stability check: 3x3 grid of (ACT+-1R) x (trail+-0.5x) around canonical
                   Requirement: >=67% of 9 combos beat baseline avg_r (+1.101R)
  Full T4 robustness battery (MC 2000 runs, cost stress, period splits,
                               remove-best-asset, remove-best-month)
  Trade count >= 80
  Benchmark: avg_r=+1.101R  PF=3.072  CAGR=+14.3%

Output: data/research_donchian_exitV2_combined/
Single comparison table at end. No frozen config until user review.
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

ROOT       = Path(__file__).parent
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUT_DIR    = ROOT / "data" / "research_donchian_exitV2_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# Frozen entry config
ENTRY_N   = 20
EXIT_N    = ENTRY_N // 2
ATR_N     = 14
STOP_MULT = 2.0

# Canonical configs to test
C1_ACT, C1_TRAIL = 8.0, 5.0
C2_ACT, C2_TRAIL = 6.0, 5.0
TIME_BACKSTOP    = 60  # days (layered on best of C1/C2)

# Stability zone step sizes
ACT_STEP   = 1.0   # R
TRAIL_STEP = 0.5   # ATR multiplier

STABILITY_THRESH = 0.67  # 67% of 9-combo zone must beat baseline avg_r

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

# Baselines
BL_AVG_R = 1.1011
BL_PF    = 3.072
BL_CAGR  = 14.3
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
# =============================================================================

def run_backtest(df: pd.DataFrame, symbol: str,
                 act_r: float, trail: float,
                 time_days: int = 0) -> List[dict]:
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

        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["e"])/pos["r"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["e"])/pos["r"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            days_held = 0
            if time_days > 0:
                days_held = int((ts[i] - pos["entry_time"]) / np.timedelta64(1, "D"))

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
                net_r = (exit_px - pos["e"]) / pos["r"]
                trades.append(dict(symbol=symbol,
                    entry_time=pos["entry_time"], exit_time=ts[i],
                    net_r=net_r, mae_r=min(pos["mae_r"], net_r),
                    mfe_r=pos["mfe_r"], exit_reason=exit_rsn))
                pos = None; continue

            if pos["mfe_r"] >= act_r: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*trail)

        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            risk = atr14[i-1] * STOP_MULT
            if risk <= 0: continue
            stop = cl[i] - risk
            pos = dict(e=cl[i], stop=stop, r=risk, entry_time=ts[i],
                       hh=hi[i], chan_active=False, chan_stop=stop,
                       mfe_r=0., mae_r=0., bars=1)

    if pos is not None:
        exit_px = cl[-1]; net_r = (exit_px-pos["e"])/pos["r"]
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1], net_r=net_r,
            mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"],
            exit_reason="end_of_data"))
    return trades


def run_universe(symbols: List[str], act_r: float, trail: float,
                 time_days: int = 0) -> pd.DataFrame:
    all_trades = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(run_backtest(df, sym, act_r, trail, time_days))
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
    return np.asarray(out[:n], dtype=float)

def t4_full(df: pd.DataFrame) -> dict:
    vals=df[R_COL].to_numpy(dtype=float); rng=np.random.default_rng(MC_SEED)
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
            end_capital=final,years=round(years,2),kill_switch_fired=False)
    return t5, t6


# =============================================================================
# 2D STABILITY GRID
# =============================================================================

def run_stability_grid(act_c: float, trail_c: float,
                       symbols: List[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Run 3x3 grid of (act_c+-ACT_STEP) x (trail_c+-TRAIL_STEP).
    Returns (grid_df, stability_dict).
    Stability: >=67% of 9 combos must beat BL_AVG_R.
    """
    act_vals   = [round(act_c - ACT_STEP, 4),
                  round(act_c,            4),
                  round(act_c + ACT_STEP, 4)]
    trail_vals = [round(trail_c - TRAIL_STEP, 4),
                  round(trail_c,              4),
                  round(trail_c + TRAIL_STEP, 4)]

    rows = []
    for act in act_vals:
        if act <= 0: continue   # skip degenerate ACT
        for trail in trail_vals:
            if trail <= 0: continue
            df_t = run_universe(symbols, act_r=act, trail=trail)
            if df_t.empty:
                avg_r_v, pf_v, n = 0., 0., 0
            else:
                r = df_t[R_COL].to_numpy(dtype=float)
                avg_r_v, pf_v, n = float(r.mean()), _pf(r), int(len(r))
            rows.append(dict(act_r=act, trail=trail, trades=n,
                             avg_r=avg_r_v, pf=pf_v,
                             beats_bl=avg_r_v > BL_AVG_R,
                             is_canonical=(act==act_c and trail==trail_c)))
    grid_df = pd.DataFrame(rows)
    n_total  = len(grid_df)
    n_beat   = int(grid_df["beats_bl"].sum())
    pct      = n_beat / n_total if n_total > 0 else 0.
    stab     = dict(n_total=n_total, n_beat=n_beat, pct=pct,
                    passes=pct >= STABILITY_THRESH,
                    act_c=act_c, trail_c=trail_c,
                    act_vals=act_vals, trail_vals=trail_vals)
    return grid_df, stab


# =============================================================================
# FULL CONFIG RUN
# =============================================================================

def run_config(label: str, act_r: float, trail: float, time_days: int,
               symbols: List[str], run_stab: bool = True) -> dict:
    """Run full T4 + T5/T6 + optional 2D stability for one config."""
    print(f"  [{label}] ACT={act_r:.0f}R  trail={trail:.1f}x  "
          f"{'+ 60-day backstop' if time_days>0 else ''}")

    # Stability grid (only for configs without time backstop)
    stab_info = {"passes": None, "pct": None, "n_beat": None, "n_total": None}
    if run_stab:
        print(f"    Stability: running 3x3 grid ...")
        grid_df, stab_info = run_stability_grid(act_r, trail, symbols)
        grid_df.to_csv(OUT_DIR / f"{label}_stability_grid.csv", index=False)
        print(f"    Stability: {stab_info['n_beat']}/{stab_info['n_total']} "
              f"= {stab_info['pct']:.1%}  "
              f"{'PASS' if stab_info['passes'] else 'FAIL'}")

    # Full backtest
    df = run_universe(symbols, act_r=act_r, trail=trail, time_days=time_days)
    df.to_csv(OUT_DIR / f"{label}_trades.csv", index=False)

    r = df[R_COL].to_numpy(dtype=float) if not df.empty else np.array([])
    base = summarize(r)
    if not df.empty:
        base.update(start=str(df[EXIT_T].min().date()),
                    end=str(df[EXIT_T].max().date()),
                    assets=df["symbol"].nunique())

    # T4
    t4 = t4_full(df) if not df.empty else {}
    if t4:
        for key, obj in t4.items():
            obj.to_csv(OUT_DIR / f"{label}_t4_{key}.csv", index=False)

    # T5/T6
    t5, t6 = ({}, {}) if df.empty else t5t6(df)

    mc10 = t4["mc"][t4["mc"]["block_size"]==10].iloc[0] if t4 else None
    cost10 = t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0] if t4 else None
    sec    = t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0] if t4 else None
    ra1    = t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0] if t4 else None
    rm1    = t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0] if t4 else None

    all_gates = (t4 and all([
        base["avg_r"]>0.15, base["profit_factor"]>1.,
        float(mc10["total_r_p05"])>0, float(cost10["profit_factor"])>1.,
        float(sec["avg_r"])>0, float(ra1["total_r"])>0, float(rm1["total_r"])>0,
        not t6.get("kill_switch_fired",False),
        0.30<=base["win_rate"]<=0.45, base["trades"]>=MIN_TRADES,
    ]))

    print(f"    T4: avg_r={base['avg_r']:+.4f}  PF={base['profit_factor']:.3f}  "
          f"trades={base['trades']}  CAGR={t6.get('cagr_pct',0):+.1f}%  "
          f"gates={'PASS' if all_gates else 'FAIL'}")
    if mc10 is not None:
        print(f"    MC(bs=10): p05={mc10['total_r_p05']:+.1f}  "
              f"prob>0={mc10['prob_positive']:.1%}")

    return dict(
        label=label, act_r=act_r, trail=trail, time_days=time_days,
        trades=base["trades"], avg_r=base["avg_r"], pf=base["profit_factor"],
        win_rate=base["win_rate"], t_score=base["t_score"],
        mc_p05=float(mc10["total_r_p05"]) if mc10 is not None else 0.,
        cagr=t6.get("cagr_pct", 0.), max_dd=t6.get("max_dd_pct", 0.),
        all_gates=all_gates,
        stab_pct=stab_info["pct"], stab_pass=stab_info["passes"],
        stab_beat=stab_info["n_beat"], stab_total=stab_info["n_total"],
        t4=t4, t5=t5, t6=t6, base=base,
    )


# =============================================================================
# MASTER REPORT
# =============================================================================

def write_master_report(configs: List[dict]) -> None:
    lines = [
        "PHASE T17 OPT-17.4 -- Combined Exit Config Test",
        "="*70, "",
        f"Tested configs:",
        f"  C1: ACT={C1_ACT:.0f}R  + trail={C1_TRAIL:.1f}x ATR",
        f"  C2: ACT={C2_ACT:.0f}R  + trail={C2_TRAIL:.1f}x ATR",
        f"  C3: Best(C1/C2) + {TIME_BACKSTOP}-day time backstop",
        "",
        f"Stability: 3x3 grid (ACT+-{ACT_STEP:.0f}R x trail+-{TRAIL_STEP:.1f}x)",
        f"           >=67% of 9 combos must beat baseline avg_r ({BL_AVG_R:.4f}R)",
        "",
        f"Baseline: avg_r=+{BL_AVG_R:.4f}R  PF={BL_PF:.3f}  CAGR=+{BL_CAGR:.1f}%",
    ]

    for cfg in configs:
        lines += [
            "", "="*70,
            f"CONFIG {cfg['label']}: ACT={cfg['act_r']:.0f}R  trail={cfg['trail']:.1f}x"
            + (f"  + {cfg['time_days']}d backstop" if cfg['time_days']>0 else ""),
            "="*70,
        ]

        # Stability grid
        if cfg["stab_pct"] is not None:
            gp = pd.read_csv(OUT_DIR / f"{cfg['label']}_stability_grid.csv")
            act_vals = sorted(gp["act_r"].unique())
            trail_vals = sorted(gp["trail"].unique())
            lines += [
                "", "Stability grid (avg_r; * = beats baseline):",
                f"  {'trail->':>8s}" + "".join(f"  {t:.1f}x  " for t in trail_vals),
                "  " + "-"*50,
            ]
            for act in act_vals:
                row_str = f"  ACT={act:.0f}R:  "
                for trail in trail_vals:
                    cell = gp[(gp["act_r"]==act) & (gp["trail"]==trail)]
                    if cell.empty:
                        row_str += "    ---  "
                    else:
                        v = float(cell["avg_r"].iloc[0])
                        canon = cell["is_canonical"].iloc[0]
                        marker = "*" if v > BL_AVG_R else " "
                        bracket = "[]" if canon else "  "
                        row_str += f"  {bracket[0]}{v:+.4f}{marker}{bracket[1]}  "
                lines.append(row_str)
            lines += [
                "",
                f"  Zone: {cfg['stab_beat']}/{cfg['stab_total']} combos beat baseline  "
                f"({cfg['stab_pct']:.1%})  "
                f"-> {'PASS' if cfg['stab_pass'] else 'FAIL'}",
            ]

        # T4 summary
        base = cfg["base"]
        t4   = cfg["t4"]
        t6   = cfg["t6"]
        if t4:
            mc10   = t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
            cost10 = t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
            sec    = t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
            ra1    = t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
            rm1    = t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]

            lines += [
                "", "Full T4 results:",
                f"  Trades   : {base['trades']}",
                f"  Avg R    : {base['avg_r']:+.4f}R  (baseline: +1.101R)",
                f"  PF       : {base['profit_factor']:.3f}  (baseline: 3.072)",
                f"  Win rate : {base['win_rate']:.1%}",
                f"  t-score  : {base['t_score']:.2f}",
                f"  Period   : {base.get('start','?')} -> {base.get('end','?')}",
                "",
                f"  MC(bs=10): p05={mc10['total_r_p05']:+.1f}  "
                f"p50={mc10['total_r_p50']:+.1f}  p95={mc10['total_r_p95']:+.1f}  "
                f"prob>0={mc10['prob_positive']:.1%}",
                "",
                "  Cost stress:",
            ]
            for _,r in t4["cost"].iterrows():
                lines.append(f"    +{r['extra_cost']:.2f}R -> avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
            lines += ["", "  Period splits:"]
            for _,r in t4["splits"].iterrows():
                lines.append(f"    {r['split']:30s}: t={int(r['trades']):3d}  "
                             f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
            lines += ["", "  Remove best assets:"]
            for _,r in t4["rem_assets"].iterrows():
                lines.append(f"    remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  "
                             f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  [{r['removed']}]")
            lines += ["", "  Remove best months:"]
            for _,r in t4["rem_months"].iterrows():
                lines.append(f"    remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  "
                             f"avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}  [{r['removed']}]")
            lines += [
                "",
                f"  T5/T6: accepted={cfg['t5'].get('accepted',0)}  "
                f"CAGR={t6.get('cagr_pct',0):+.1f}%  "
                f"DD={t6.get('max_dd_pct',0):+.1f}%",
                "",
                "  Gate results:",
                f"    {'PASS' if base['trades']>=MIN_TRADES else 'FAIL'}  trades>={MIN_TRADES} (got {base['trades']})",
                f"    {'PASS' if 0.30<=base['win_rate']<=0.45 else 'FAIL'}  win% 30-45% (got {base['win_rate']:.1%})",
                f"    {'PASS' if base['avg_r']>0.15 else 'FAIL'}  avg_r > 0.15R (got {base['avg_r']:+.4f}R)",
                f"    {'PASS' if base['profit_factor']>1. else 'FAIL'}  PF > 1.0 (got {base['profit_factor']:.3f})",
                f"    {'PASS' if float(mc10['total_r_p05'])>0 else 'FAIL'}  MC p05 > 0 (got {float(mc10['total_r_p05']):+.1f}R)",
                f"    {'PASS' if float(cost10['profit_factor'])>1. else 'FAIL'}  cost+0.10R PF>1.0 (got {float(cost10['profit_factor']):.3f})",
                f"    {'PASS' if float(sec['avg_r'])>0 else 'FAIL'}  2nd-half avgR>0 (got {float(sec['avg_r']):+.4f}R)",
                f"    {'PASS' if float(ra1['total_r'])>0 else 'FAIL'}  remove top-1 asset>0 (got {float(ra1['total_r']):+.1f}R)",
                f"    {'PASS' if float(rm1['total_r'])>0 else 'FAIL'}  remove top-1 month>0 (got {float(rm1['total_r']):+.1f}R)",
                "",
                f"  vs BASELINE:",
                f"    avg_r : {base['avg_r']:+.4f}R vs +1.101R -> {'BETTER' if base['avg_r']>BL_AVG_R else 'WORSE'}",
                f"    PF    : {base['profit_factor']:.3f} vs 3.072 -> {'BETTER' if base['profit_factor']>BL_PF else 'WORSE'}",
                f"    CAGR  : {t6.get('cagr_pct',0):+.1f}% vs +14.3% -> {'BETTER' if t6.get('cagr_pct',0)>BL_CAGR else 'WORSE'}",
                "",
                f"  OVERALL: {'PASS' if cfg['all_gates'] else 'FAIL'}",
            ]

    rpt = OUT_DIR / "combined_master_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Master report: {rpt}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(configs: List[dict]) -> None:
    SEP = "=" * 82
    print()
    print(SEP)
    print("OPT-17.4 COMBINED EXIT TEST - COMPARISON TABLE")
    print(SEP)
    print()
    hdr = (f"  {'Config':<24s}  {'Trades':>6s}  {'dTrades':>8s}  "
           f"{'avg_r':>8s}  {'PF':>5s}  {'CAGR':>6s}  {'DD':>6s}  "
           f"{'MC_p05':>8s}  {'Stab':>7s}  {'Gates':>5s}")
    print(hdr)
    print("  " + "-"*80)

    # Baseline row
    print(f"  {'Baseline':24s}  {461:>6d}  {'---':>8s}  "
          f"{'+1.101R':>8s}  {'3.072':>5s}  {'+14.3%':>6s}  {'-3.4%':>6s}  "
          f"{'+347.7R':>8s}  {'---':>7s}  {'ref':>5s}")
    # Var B best individual (context row)
    print(f"  {'B: trail=5.0x (alone)':24s}  {400:>6d}  {-61:>+8d}  "
          f"{'+1.363R':>8s}  {'3.596':>5s}  {'+15.1%':>6s}  {'-5.2%':>6s}  "
          f"{'+299.1R':>8s}  {'3/3 P':>7s}  {'PASS':>5s}")
    print("  " + "-"*80)

    for cfg in configs:
        delta = cfg["trades"] - 461
        stab_str = (f"{cfg['stab_beat']}/{cfg['stab_total']} "
                    f"{'P' if cfg['stab_pass'] else 'F'}"
                    if cfg["stab_pct"] is not None else "n/a  ")
        gates = "PASS" if cfg["all_gates"] else "FAIL"
        better_r = "^" if cfg["avg_r"] > BL_AVG_R else "v"
        label_str = (f"{cfg['label']}: ACT={cfg['act_r']:.0f}R,t={cfg['trail']:.1f}x"
                     + (" +60d" if cfg["time_days"]>0 else ""))
        print(f"  {label_str:<24s}  {cfg['trades']:>6d}  {delta:>+8d}  "
              f"{cfg['avg_r']:>+7.4f}{better_r}  {cfg['pf']:>5.3f}  "
              f"{cfg['cagr']:>+6.1f}%  {cfg['max_dd']:>+6.1f}%  "
              f"{cfg['mc_p05']:>+8.1f}R  {stab_str:>7s}  {gates:>5s}")

    print()
    print("  Gate failures:")
    for cfg in configs:
        fails = []
        if cfg["mc_p05"] <= 0:    fails.append("MC_p05<=0")
        if cfg["avg_r"] <= 0.15:  fails.append("avg_r<=0.15")
        if cfg["pf"] <= 1.0:      fails.append("PF<=1.0")
        if not (0.30 <= cfg["win_rate"] <= 0.45): fails.append("win%")
        if cfg["trades"] < MIN_TRADES: fails.append(f"trades<{MIN_TRADES}")
        print(f"  {cfg['label']}: {', '.join(fails) if fails else 'none'}")

    print()
    print("  Benchmark vs baseline (+1.101R / 3.072 / +14.3%):")
    for cfg in configs:
        rr = "BETTER" if cfg["avg_r"] > BL_AVG_R else "WORSE "
        pr = "BETTER" if cfg["pf"] > BL_PF       else "WORSE "
        cr = "BETTER" if cfg["cagr"] > BL_CAGR   else "WORSE "
        print(f"  {cfg['label']}: avg_r {rr} ({cfg['avg_r']:+.4f}R)  "
              f"PF {pr} ({cfg['pf']:.3f})  "
              f"CAGR {cr} ({cfg['cagr']:+.1f}%)")
    print()
    print(SEP)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    configs: List[dict] = []

    print("="*70)
    print("Phase T17 Opt-17.4 -- Combined Exit Config Test")
    print("="*70)
    print(f"Universe: {len(symbols)} symbols")
    print()

    # ── Config 1: ACT=8R + trail=5.0x ──────────────────────────────────────
    print("Config 1: ACT=8R + trail=5.0x ...")
    c1 = run_config("C1", C1_ACT, C1_TRAIL, 0, symbols, run_stab=True)
    configs.append(c1)

    # ── Config 2: ACT=6R + trail=5.0x ──────────────────────────────────────
    print("\nConfig 2: ACT=6R + trail=5.0x ...")
    c2 = run_config("C2", C2_ACT, C2_TRAIL, 0, symbols, run_stab=True)
    configs.append(c2)

    # ── Config 3: Best(C1/C2) + 60-day backstop ─────────────────────────────
    best = c1 if c1["avg_r"] >= c2["avg_r"] else c2
    print(f"\nConfig 3: {best['label']} (ACT={best['act_r']:.0f}R, "
          f"trail={best['trail']:.1f}x) + 60-day backstop ...")
    c3 = run_config("C3",
                    act_r=best["act_r"], trail=best["trail"],
                    time_days=TIME_BACKSTOP,
                    symbols=symbols, run_stab=False)
    c3["label"] = f"C3({best['label']}+D)"
    configs.append(c3)

    write_master_report(configs)
    print_comparison_table(configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
