#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T15 -- MeanReversionRSI Parameter Stability
UngerFink Pipeline / Andrea Unger Methodology

Tests parameter neighbourhood around the frozen config to verify
the edge is not a point-optimisation artefact.

Frozen config: rsi_n=14, oversold=25, time_exit=20, atr_mult=3.0

Tests:
  A) rsi_n sweep    : [10, 14, 20]  with oversold=25 fixed
  B) oversold sweep : [20, 25, 30]  with rsi_n=14 fixed

All variants use Variant E exit (time_only, 20 bars).
All must remain profitable (avg_r > 0, PF > 1.0).

Output: data/research_meanreversionrsi_t15_1d/phase_t15_param_stability.csv
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
OUT_DIR = ROOT / "data" / "research_meanreversionrsi_t15_1d"
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

CANONICAL = {"rsi_n": 14, "oversold": 25, "atr_mult": 3.0, "time_exit": 20}

RSI_N_SWEEP    = [10, 14, 20]
OVERSOLD_SWEEP = [20, 25, 30]


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


def backtest(df: pd.DataFrame, rsi_n: int, oversold: int,
             atr_mult: float, time_exit: int) -> list[float]:
    close  = df["close"].values
    low_v  = df["low"].values
    high_v = df["high"].values

    # RSI (Wilder EMA)
    delta = np.diff(close, prepend=close[0])
    avg_g = pd.Series(np.where(delta > 0, delta, 0)).ewm(
        alpha=1/rsi_n, min_periods=rsi_n, adjust=False).mean().values
    avg_l = pd.Series(np.where(delta < 0, -delta, 0)).ewm(
        alpha=1/rsi_n, min_periods=rsi_n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs  = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - (100 / (1 + rs))

    # ATR(14)
    tr  = np.maximum.reduce([
        high_v[1:] - low_v[1:],
        np.abs(high_v[1:] - close[:-1]),
        np.abs(low_v[1:]  - close[:-1]),
    ])
    tr  = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(14).mean().values

    rs_list: list[float] = []
    in_pos   = False
    e_price  = 0.0
    stop     = 0.0
    e_bar    = 0

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]):
            continue
        if not in_pos:
            if rsi[i] < oversold:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                stop    = e_price - atr_mult * atr[i]
        else:
            bars_held = i - e_bar
            ep = None
            if low_v[i] <= stop:
                ep = stop
            elif bars_held >= time_exit:
                ep = close[i]
            if ep is not None:
                risk = e_price - stop
                if risk > 1e-9:
                    rs_list.append((ep - e_price) / risk)
                in_pos = False
    return rs_list


def metrics(rs: list[float]) -> dict:
    if not rs:
        return {"n":0,"avg_r":0.0,"pf":0.0,"win_rate":0.0,"total_r":0.0}
    a  = np.array(rs)
    w  = a[a > 0]; l = np.abs(a[a < 0])
    pf = w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    return {"n":len(a),"avg_r":float(np.mean(a)),"pf":float(pf),
            "win_rate":float(len(w)/len(a)),"total_r":float(np.sum(a))}


def run_combo(rsi_n: int, oversold: int, label: str) -> dict:
    all_rs: list[float] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            all_rs.extend(backtest(df, rsi_n, oversold,
                                   CANONICAL["atr_mult"], CANONICAL["time_exit"]))
    m = metrics(all_rs)
    canonical = (rsi_n == CANONICAL["rsi_n"] and oversold == CANONICAL["oversold"])
    passing   = m["avg_r"] > 0 and m["pf"] >= 1.0
    return {
        "label":     label,
        "rsi_n":     rsi_n,
        "oversold":  oversold,
        "n":         m["n"],
        "avg_r":     round(m["avg_r"], 4),
        "pf":        round(m["pf"], 2),
        "win_rate":  round(m["win_rate"], 4),
        "total_r":   round(m["total_r"], 2),
        "canonical": canonical,
        "pass":      passing,
    }


def main() -> None:
    p("=" * 65)
    p("  Phase T15 -- MeanReversionRSI Parameter Stability")
    p(f"  Canonical: rsi_n={CANONICAL['rsi_n']}  oversold={CANONICAL['oversold']}")
    p(f"  Exit: time_only {CANONICAL['time_exit']} bars  ATR: {CANONICAL['atr_mult']}")
    p("=" * 65)

    rows = []

    # Sweep A: rsi_n
    p("\n  Sweep A: rsi_n  (oversold=25 fixed)")
    p(f"  {'Label':<20} {'rsi_n':>6} {'os':>4} {'N':>5} {'AvgR':>8} {'PF':>5} {'WR%':>6} {'TotR':>8}  Pass")
    for rsi_n in RSI_N_SWEEP:
        label = f"rsi{rsi_n}/os25"
        r = run_combo(rsi_n, 25, label)
        canon = " <-- CANONICAL" if r["canonical"] else ""
        p(f"  {label:<20} {r['rsi_n']:>6} {r['oversold']:>4} {r['n']:>5} "
          f"{r['avg_r']:>+7.4f}R {r['pf']:>5.2f} {r['win_rate']*100:>5.1f}% "
          f"{r['total_r']:>+7.2f}R  {'PASS' if r['pass'] else 'FAIL'}{canon}")
        rows.append({"sweep": "rsi_n", **r})

    # Sweep B: oversold
    p("\n  Sweep B: oversold  (rsi_n=14 fixed)")
    p(f"  {'Label':<20} {'rsi_n':>6} {'os':>4} {'N':>5} {'AvgR':>8} {'PF':>5} {'WR%':>6} {'TotR':>8}  Pass")
    for os in OVERSOLD_SWEEP:
        label = f"rsi14/os{os}"
        r = run_combo(14, os, label)
        canon = " <-- CANONICAL" if r["canonical"] else ""
        p(f"  {label:<20} {r['rsi_n']:>6} {r['oversold']:>4} {r['n']:>5} "
          f"{r['avg_r']:>+7.4f}R {r['pf']:>5.2f} {r['win_rate']*100:>5.1f}% "
          f"{r['total_r']:>+7.2f}R  {'PASS' if r['pass'] else 'FAIL'}{canon}")
        rows.append({"sweep": "oversold", **r})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "phase_t15_param_stability.csv", index=False)

    # Summary
    p("\n  --- T15 Gate Summary ---")
    all_pass = all(r["pass"] for r in rows)
    n_pass   = sum(r["pass"] for r in rows)
    p(f"  {n_pass}/{len(rows)} parameter variants profitable (avg_r > 0 and PF >= 1.0)")

    for r in rows:
        p(f"  [{' PASS' if r['pass'] else ' FAIL'}] {r['label']:<20}  "
          f"avg_r={r['avg_r']:+.4f}R  PF={r['pf']:.2f}")

    p(f"\n  T15 GATE: {'PASS -- all variants profitable' if all_pass else 'FAIL -- some variants unprofitable'}")
    p(f"[OK] phase_t15_param_stability.csv")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
