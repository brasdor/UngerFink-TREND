#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 ExitV2 -- T15 + T16 Validation

Validates the DonchianLong_UniverseV2_ExitV2 frozen config:
  ACT=6R  |  trail=5.0x ATR  |  N=20  |  ema200_price  |  ATR×2.0 stop

T15 -- Entry parameter stability
  Test N=[15, 20, 25] with ExitV2 params fixed.
  Gate: all three N values must be profitable (avg_r > 0  AND  total_r > 0).
  Also report avg_r vs baseline (+1.101R) for each N.
  Output: data/research_donchian_exitv2_t15/

T16 -- Monte Carlo stress (extended)
  5000 runs per block size.
  Block sizes: [1, 5, 10, 20, 50].
  Gate: p05 totalR > 0 at ALL block sizes.
  Also report prob_positive, PF p05/p50, DD p95 per block size.
  Output: data/research_donchian_exitv2_t16/

Results printed at end. No action taken after — awaiting review.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT       = Path(__file__).resolve().parents[1]
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUT_T15    = ROOT / "data" / "research_donchian_exitv2_t15"
OUT_T16    = ROOT / "data" / "research_donchian_exitv2_t16"
OUT_T15.mkdir(parents=True, exist_ok=True)
OUT_T16.mkdir(parents=True, exist_ok=True)

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# ExitV2 frozen params
ACT_R   = 6.0
TRAIL   = 5.0
ATR_N   = 14
STOP_MULT = 2.0

# T15
T15_N_VALUES = [15, 20, 25]

# T16
T16_MC_RUNS   = 5000
T16_MC_BLOCKS = [1, 5, 10, 20, 50]
T16_SEED      = 42

# References
BL_AVG_R = 1.1011    # UniverseV2 baseline (ACT4_ATR3)
BL_PF    = 3.072
BL_CAGR  = 14.3


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
# BACKTEST  (ExitV2 — parametric N)
# =============================================================================

def run_backtest(df: pd.DataFrame, symbol: str, entry_n: int) -> List[dict]:
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    exit_n = entry_n // 2
    if nb < max(entry_n, ATR_N, 200) + 20: return []

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()

    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

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

            exit_px = exit_rsn = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_rsn = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_rsn = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_rsn = cl[i], "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["e"]) / pos["r"]
                trades.append(dict(symbol=symbol,
                    entry_time=pos["entry_time"], exit_time=ts[i],
                    net_r=net_r, mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
                pos = None; continue

            if pos["mfe_r"] >= ACT_R: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*TRAIL)

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
            mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
    return trades


def run_universe(symbols: List[str], entry_n: int) -> pd.DataFrame:
    all_trades = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(run_backtest(df, sym, entry_n))
    if not all_trades: return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    return df.dropna(subset=[EXIT_T, ENTRY_T, R_COL]).sort_values(EXIT_T).reset_index(drop=True)


# =============================================================================
# STATS
# =============================================================================

def _pf(r):
    g=r[r>0].sum(); l=-r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.)

def _mdd(r):
    if r.size==0: return 0.
    eq=np.cumsum(r); pk=np.maximum.accumulate(eq); return float((eq-pk).min())

def summarize(r: np.ndarray) -> dict:
    if r.size==0:
        return dict(trades=0,total_r=0.,avg_r=0.,win_rate=0.,
                    profit_factor=0.,max_dd_r=0.,t_score=0.)
    avg=float(r.mean()); std=float(r.std(ddof=1)) if r.size>1 else 0.
    t=avg/(std/math.sqrt(r.size)) if std>0 else 0.
    return dict(trades=int(r.size),total_r=float(r.sum()),avg_r=avg,
                win_rate=float((r>0).mean()),profit_factor=_pf(r),
                max_dd_r=_mdd(r),t_score=float(t))

def _bstrap(vals, bs, rng):
    n=len(vals); out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); e=min(s+bs,n); blk=vals[s:e]
        if len(blk)<bs: blk=np.concatenate([blk,vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n], dtype=float)


# =============================================================================
# T15 — N STABILITY
# =============================================================================

def run_t15(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for n in T15_N_VALUES:
        df = run_universe(symbols, entry_n=n)
        r  = df[R_COL].to_numpy(dtype=float) if not df.empty else np.array([])
        s  = summarize(r)
        canon = (n == 20)
        profitable = s["avg_r"] > 0 and s["total_r"] > 0
        rows.append(dict(
            entry_n=n, exit_n=n//2,
            trades=s["trades"], total_r=s["total_r"], avg_r=s["avg_r"],
            win_rate=s["win_rate"], profit_factor=s["profit_factor"],
            max_dd_r=s["max_dd_r"], t_score=s["t_score"],
            beats_baseline=s["avg_r"] > BL_AVG_R,
            profitable=profitable, is_canonical=canon,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# T16 — MC STRESS (extended)
# =============================================================================

def run_t16(r: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(T16_SEED)
    rows = []
    for bs in T16_MC_BLOCKS:
        tots, dds, pfs = [], [], []
        for _ in range(T16_MC_RUNS):
            s = _bstrap(r, bs, rng)
            tots.append(s.sum()); dds.append(_mdd(s)); pfs.append(_pf(s))
        tots = np.array(tots); dds = np.array(dds); pfs = np.array(pfs)
        rows.append(dict(
            block_size=bs, mc_runs=T16_MC_RUNS,
            total_r_p05=float(np.percentile(tots, 5)),
            total_r_p50=float(np.percentile(tots, 50)),
            total_r_p95=float(np.percentile(tots, 95)),
            dd_p95=float(np.percentile(dds, 95)),
            pf_p05=float(np.percentile(pfs, 5)),
            pf_p50=float(np.percentile(pfs, 50)),
            prob_positive=float((tots > 0).mean()),
            gate_pass=float(np.percentile(tots, 5)) > 0,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================

def write_t15_report(df: pd.DataFrame, all_pass: bool) -> None:
    lines = [
        "T15 -- Entry Parameter Stability (ExitV2: ACT=6R, trail=5.0x)",
        "="*65, "",
        f"Fixed exit params : ACT={ACT_R:.0f}R  trail={TRAIL:.1f}x ATR",
        f"Baseline ref      : avg_r=+{BL_AVG_R:.4f}R  PF={BL_PF:.3f}",
        "",
        f"  {'N':>4s}  {'exit_n':>6s}  {'trades':>7s}  {'total_r':>8s}  "
        f"{'avg_r':>8s}  {'PF':>5s}  {'win%':>5s}  {'t':>5s}  {'profitable':>10s}  {'bl?':>4s}",
        "  " + "-"*75,
    ]
    for _, r in df.iterrows():
        canon = " <-- canonical" if r["is_canonical"] else ""
        p_flag = "YES" if r["profitable"] else "NO "
        b_flag = "YES" if r["beats_baseline"] else "no "
        lines.append(
            f"  {int(r['entry_n']):>4d}  {int(r['exit_n']):>6d}  "
            f"{int(r['trades']):>7d}  {r['total_r']:>+8.2f}  "
            f"{r['avg_r']:>+8.4f}  {r['profit_factor']:>5.3f}  "
            f"{r['win_rate']:>5.1%}  {r['t_score']:>5.2f}  "
            f"{p_flag:>10s}  {b_flag:>4s}{canon}"
        )
    lines += [
        "",
        f"  ALL N PROFITABLE: {'YES -- T15 PASS' if all_pass else 'NO -- T15 FAIL'}",
        "",
        "  Gate: all of N=[15, 20, 25] must have avg_r > 0 AND total_r > 0",
        f"  Result: {'PASS' if all_pass else 'FAIL'}",
    ]
    (OUT_T15 / "t15_stability_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_t16_report(df: pd.DataFrame, all_pass: bool, base_stats: dict) -> None:
    lines = [
        "T16 -- Monte Carlo Stress (Extended, ExitV2: ACT=6R, trail=5.0x)",
        "="*65, "",
        f"Canonical N=20  |  trades={base_stats['trades']}  "
        f"avg_r={base_stats['avg_r']:+.4f}R  PF={base_stats['profit_factor']:.3f}",
        f"MC runs per block: {T16_MC_RUNS}  |  RNG seed: {T16_SEED}",
        "",
        f"  {'block':>7s}  {'p05_totalR':>10s}  {'p50_totalR':>10s}  "
        f"{'p95_totalR':>10s}  {'prob>0':>7s}  {'PF_p05':>7s}  {'DD_p95':>8s}  {'gate':>5s}",
        "  " + "-"*72,
    ]
    for _, r in df.iterrows():
        gate = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"  {int(r['block_size']):>7d}  {r['total_r_p05']:>+10.1f}  "
            f"{r['total_r_p50']:>+10.1f}  {r['total_r_p95']:>+10.1f}  "
            f"{r['prob_positive']:>7.1%}  {r['pf_p05']:>7.3f}  "
            f"{r['dd_p95']:>+8.2f}R  {gate:>5s}"
        )
    lines += [
        "",
        f"  ALL BLOCK SIZES p05 > 0: {'YES -- T16 PASS' if all_pass else 'NO -- T16 FAIL'}",
        "",
        "  Gate: p05 totalR > 0 at ALL block sizes [1, 5, 10, 20, 50]",
        f"  Result: {'PASS' if all_pass else 'FAIL'}",
    ]
    (OUT_T16 / "t16_mc_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    print("="*65)
    print("Phase T17 ExitV2 -- T15 + T16 Validation")
    print(f"Config: ACT={ACT_R:.0f}R  trail={TRAIL:.1f}x ATR  |  Universe: {len(symbols)} symbols")
    print("="*65)

    # ── T15 ─────────────────────────────────────────────────────────────────
    print("\n[T15] Entry parameter stability (N=[15, 20, 25]) ...")
    t15_df = run_t15(symbols)
    t15_df.to_csv(OUT_T15 / "t15_stability.csv", index=False)
    t15_all_pass = bool(t15_df["profitable"].all())
    write_t15_report(t15_df, t15_all_pass)

    print(f"\n  {'N':>4s}  {'trades':>7s}  {'avg_r':>8s}  {'PF':>5s}  {'profitable':>10s}")
    for _, r in t15_df.iterrows():
        canon = " <-" if r["is_canonical"] else ""
        print(f"  {int(r['entry_n']):>4d}  {int(r['trades']):>7d}  "
              f"{r['avg_r']:>+8.4f}  {r['profit_factor']:>5.3f}  "
              f"{'YES' if r['profitable'] else 'NO ':>10s}{canon}")
    print(f"\n  T15 gate: {'PASS -- all N profitable' if t15_all_pass else 'FAIL -- not all N profitable'}")

    # ── T16 ─────────────────────────────────────────────────────────────────
    print(f"\n[T16] Monte Carlo stress ({T16_MC_RUNS} runs, blocks={T16_MC_BLOCKS}) ...")
    # Use canonical N=20 trades
    canon_df = t15_df[t15_df["is_canonical"]]
    if canon_df.empty or not t15_all_pass:
        print("  Skipping T16 — T15 not passed.")
        return 1

    # Re-run N=20 to get full trades DataFrame
    print("  Loading canonical N=20 trades ...")
    df_canon = run_universe(symbols, entry_n=20)
    df_canon.to_csv(OUT_T16 / "t16_canonical_trades.csv", index=False)
    r_canon = df_canon[R_COL].to_numpy(dtype=float)
    base_stats = summarize(r_canon)

    print(f"  Trades: {base_stats['trades']}  avg_r={base_stats['avg_r']:+.4f}R  "
          f"PF={base_stats['profit_factor']:.3f}")
    print("  Running MC ...")

    t16_df = run_t16(r_canon)
    t16_df.to_csv(OUT_T16 / "t16_mc_results.csv", index=False)
    t16_all_pass = bool(t16_df["gate_pass"].all())
    write_t16_report(t16_df, t16_all_pass, base_stats)

    print(f"\n  {'block':>7s}  {'p05':>10s}  {'p50':>10s}  {'prob>0':>7s}  {'gate':>5s}")
    for _, r in t16_df.iterrows():
        print(f"  {int(r['block_size']):>7d}  {r['total_r_p05']:>+10.1f}  "
              f"{r['total_r_p50']:>+10.1f}  {r['prob_positive']:>7.1%}  "
              f"{'PASS' if r['gate_pass'] else 'FAIL':>5s}")
    print(f"\n  T16 gate: {'PASS -- p05 > 0 at all block sizes' if t16_all_pass else 'FAIL -- p05 <= 0 at one or more block sizes'}")

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print()
    print("="*65)
    print("VALIDATION SUMMARY")
    print("="*65)
    print(f"  T15 (N stability) : {'PASS' if t15_all_pass else 'FAIL'}")
    print(f"  T16 (MC stress)   : {'PASS' if t16_all_pass else 'FAIL'}")
    both = t15_all_pass and t16_all_pass
    print()
    print(f"  DonchianLong_UniverseV2_ExitV2  -->  "
          f"{'VALIDATED -- both T15 and T16 pass' if both else 'NOT YET VALIDATED -- one or more failed'}")
    if both:
        print()
        print("  Frozen config:  data/research_donchian_exitV2_combined/phase_exitv2_frozen_config.py")
        print("  Set T15_PASS=True, T16_P05_ALL_POSITIVE=True, T15_T16_VALIDATED=True")
        print("  then proceed to T17 (walk-forward) when ready.")
    print("="*65)

    # Update frozen config flags in-place if both pass
    if both:
        cfg_path = (ROOT / "data" / "research_donchian_exitV2_combined"
                    / "phase_exitv2_frozen_config.py")
        txt = cfg_path.read_text(encoding="utf-8")
        txt = txt.replace("T15_PASS             = None", "T15_PASS             = True")
        txt = txt.replace("T16_P05_ALL_POSITIVE = None", "T16_P05_ALL_POSITIVE = True")
        txt = txt.replace("T15_T16_VALIDATED    = False", "T15_T16_VALIDATED    = True")
        cfg_path.write_text(txt, encoding="utf-8")
        print("\n[OK] Frozen config updated: T15_PASS=True, T16_P05_ALL_POSITIVE=True, T15_T16_VALIDATED=True")

    return 0


if __name__ == "__main__":
    sys.exit(main())
