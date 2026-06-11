#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T15 + T16 -- ConsecDownDaysMR Param Stability + Monte Carlo
UngerFink Pipeline / Andrea Unger Methodology

T15: Test consec_n neighbourhood [4, 5, 6] -- all must remain profitable
T16: 5000-run block bootstrap MC on Variant E trades
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

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw_trend_t1"
T3MR_DIR = ROOT / "data" / "research_consecdowndays_mr_t3mr"
OUT_T15  = ROOT / "data" / "research_consecdowndays_mr_t15"
OUT_T16  = ROOT / "data" / "research_consecdowndays_mr_t16"
OUT_T15.mkdir(parents=True, exist_ok=True)
OUT_T16.mkdir(parents=True, exist_ok=True)

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

# Canonical config
HOLD_BARS = 20
ATR_MULT  = 2.0
EMA_N     = 200

MC_RUNS   = 5000
MC_BLOCKS = [1, 5, 10, 20]
np.random.seed(42)


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{sym}_1d.csv"
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
        if len(df) > 2000:
            df = df.iloc[-2000:].reset_index(drop=True)
        return df if len(df) >= 200 else None
    except Exception:
        return None


def backtest(df: pd.DataFrame, consec_n: int) -> list[float]:
    close = df["close"]
    high  = df["high"]; low = df["low"]
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr    = tr.rolling(14).mean().values
    ema200 = close.ewm(span=EMA_N, adjust=False).mean().values
    close_v = close.values; low_v = low.values

    down = (close < close.shift(1)).astype(int)
    streak = [0]*len(down)
    for i in range(1, len(down)):
        streak[i] = streak[i-1]+1 if down.iloc[i] else 0
    streak = np.array(streak)

    rs: list[float] = []
    in_pos = False; e_price = stop = 0.0; e_bar = 0

    for i in range(len(df)):
        if np.isnan(atr[i]): continue
        if not in_pos:
            if int(streak[i])==consec_n and close_v[i]>ema200[i]:
                in_pos=True; e_price=close_v[i]; e_bar=i
                stop=e_price-ATR_MULT*atr[i]
        else:
            bars_held=i-e_bar; ep=None
            if low_v[i]<=stop: ep=stop
            elif bars_held>=HOLD_BARS: ep=close_v[i]
            if ep is not None:
                risk=e_price-stop
                if risk>1e-9: rs.append((ep-e_price)/risk)
                in_pos=False
    return rs


def metrics(rs: np.ndarray) -> dict:
    if len(rs)==0:
        return {"n":0,"avg_r":0.0,"pf":0.0,"win_rate":0.0,"total_r":0.0}
    w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    return {"n":len(rs),"avg_r":float(np.mean(rs)),"pf":float(pf),
            "win_rate":float(len(w)/len(rs)),"total_r":float(np.sum(rs))}


def block_bootstrap(rs: np.ndarray, block_size: int, n_runs: int) -> dict:
    n=len(rs); totals=[]; pfs=[]
    for _ in range(n_runs):
        starts=np.random.randint(0,n,size=max(1,n//block_size)+5)
        sample=np.concatenate([rs[s:min(s+block_size,n)] for s in starts])[:n]
        if not len(sample): continue
        m=metrics(sample); totals.append(m["total_r"]); pfs.append(m["pf"])
    totals=np.array(totals); pfs=np.array(pfs)
    return {"block_size":block_size,
            "p05_total_r":float(np.percentile(totals,5)),
            "p50_total_r":float(np.percentile(totals,50)),
            "p95_total_r":float(np.percentile(totals,95)),
            "prob_positive":float(np.mean(totals>0)),
            "pf_p05":float(np.percentile(pfs,5)),
            "mc_pass":bool(np.percentile(totals,5)>0 and np.percentile(pfs,5)>=1.0)}


def main() -> None:
    # =========================================================================
    # T15 -- PARAMETER STABILITY
    # =========================================================================
    p("="*65)
    p("  Phase T15 -- ConsecDownDaysMR Parameter Stability")
    p(f"  Testing consec_n=[4, 5, 6]  hold={HOLD_BARS}  atr={ATR_MULT}  filter=ema200")
    p(f"  Canonical: consec_n=5")
    p("="*65)

    loaded = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append(df)
    p(f"  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")

    t15_rows = []
    p(f"\n  {'Label':<15} {'consec_n':>8} {'N':>5} {'AvgR':>8} {'PF':>5} {'WR%':>6} {'TotR':>8}  Pass")
    for cn in [4, 5, 6]:
        all_rs: list[float] = []
        for df in loaded:
            all_rs.extend(backtest(df, cn))
        m = metrics(np.array(all_rs))
        passing = m["avg_r"] > 0 and m["pf"] >= 1.0
        canon = " <-- CANONICAL" if cn == 5 else ""
        p(f"  consec_n={cn:<8} {cn:>8} {m['n']:>5} {m['avg_r']:>+7.4f}R "
          f"{m['pf']:>5.2f} {m['win_rate']*100:>5.1f}% {m['total_r']:>+7.2f}R  "
          f"{'PASS' if passing else 'FAIL'}{canon}")
        t15_rows.append({"consec_n":cn,**m,"pass":passing})

    t15_df = pd.DataFrame(t15_rows)
    t15_df.to_csv(OUT_T15/"phase_t15_param_stability.csv", index=False)

    all_t15_pass = all(r["pass"] for r in t15_rows)
    p(f"\n  T15 GATE: {'PASS -- all consec_n variants profitable' if all_t15_pass else 'FAIL'}")
    for r in t15_rows:
        p(f"  [{'PASS' if r['pass'] else 'FAIL'}] consec_n={r['consec_n']}  avg_r={r['avg_r']:+.4f}R  PF={r['pf']:.2f}")

    p(f"[OK] phase_t15_param_stability.csv")

    # =========================================================================
    # T16 -- MONTE CARLO
    # =========================================================================
    p()
    p("="*65)
    p(f"  Phase T16 -- ConsecDownDaysMR Monte Carlo ({MC_RUNS} runs)")
    p("="*65)

    trades_path = T3MR_DIR / "phase_t3mr_trades_E.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found"); sys.exit(1)

    df_t = pd.read_csv(trades_path)
    rs_all = df_t["net_r"].values
    p(f"  Input trades : {len(rs_all)}  total_r={rs_all.sum():.2f}R  avg_r={rs_all.mean():.4f}R")

    p(f"\n  {'BlkSz':>6}  {'p05':>9}  {'p50':>9}  {'p95':>9}  {'Prob+':>7}  {'PF_p05':>7}  Gate")
    mc_rows = []
    for bs in MC_BLOCKS:
        p(f"  Running block_size={bs}...", end=" ")
        r = block_bootstrap(rs_all, bs, MC_RUNS)
        p("done")
        p(f"  {r['block_size']:>6}  {r['p05_total_r']:>+8.2f}R  {r['p50_total_r']:>+8.2f}R  "
          f"{r['p95_total_r']:>+8.2f}R  {r['prob_positive']:>6.1%}  "
          f"{r['pf_p05']:>7.2f}  {'OK' if r['mc_pass'] else 'FAIL'}")
        mc_rows.append(r)

    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUT_T16/"phase_t16_montecarlo_summary.csv", index=False)

    all_mc_pass  = all(r["mc_pass"] for r in mc_rows)
    min_p05      = min(r["p05_total_r"] for r in mc_rows)
    min_prob_pos = min(r["prob_positive"] for r in mc_rows)

    p(f"\n  Worst-case p05 total_r  : {min_p05:+.2f}R")
    p(f"  Min prob_positive       : {min_prob_pos:.1%}")
    p(f"  All block sizes pass    : {'YES' if all_mc_pass else 'NO'}")
    p(f"\n  T16 GATE: {'PASS' if all_mc_pass else 'FAIL'}")
    p(f"[OK] phase_t16_montecarlo_summary.csv")

    # Final summary
    p()
    p("="*65)
    p("  T15 + T16 SUMMARY")
    p("="*65)
    p(f"  T15 GATE : {'PASS' if all_t15_pass else 'FAIL'}")
    p(f"  T16 GATE : {'PASS' if all_mc_pass else 'FAIL'}")
    if all_t15_pass and all_mc_pass:
        p("  ConsecDownDaysMR 1D: FULLY VALIDATED -- ready for T9B paper trading")

    sys.exit(0 if (all_t15_pass and all_mc_pass) else 1)


if __name__ == "__main__":
    main()
