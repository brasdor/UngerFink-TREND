#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.3 -- Variant A Sensitivity Analysis

Tests a 5x4 grid of (pullback_depth, cancel_window) values to verify that
Variant A's improvement over baseline is robust rather than a parameter peak.

Grid:
  pullback_depth : [0.10, 0.25, 0.50, 0.75, 1.00] x ATR(14)
  cancel_window  : [1, 2, 3, 5] bars
  Total          : 20 combinations

Canonical (adopted from Opt-17.3 Variant A): depth=0.25, window=2

Stability rule (Unger §2.1 applied to entry params):
  Improvement = avg_r > 1.1011R (baseline) AND PF > 3.072 (baseline)
  Adjacent = 8 cells in the 3x3 neighbourhood around (0.25, 2):
    depths {0.10, 0.50}, windows {1, 2, 3} cross-product, minus canonical
  Requirement: >= 67% of 8 adjacent cells must show improvement.
  If not met: Variant A is a parameter peak -- REJECT.

Metrics per combo: trades, avg_r, PF, CAGR (max8), max_DD, MC_p05 (1000 runs).

Baseline: avg_r=+1.101R  PF=3.072  CAGR=+14.3%

Output: data/research_donchian_entryV2_A_sensitivity/
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
OUT_DIR    = ROOT / "data" / "research_donchian_entryV2_A_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

ATR_N      = 14
STOP_MULT  = 2.0
CHAN_ACT_R = 4.0
CHAN_TRAIL = 3.0
ENTRY_N    = 20
EXIT_N     = ENTRY_N // 2

# Grid
PULLBACK_DEPTHS  = [0.10, 0.25, 0.50, 0.75, 1.00]
CANCEL_WINDOWS   = [1, 2, 3, 5]

CANONICAL_DEPTH  = 0.25
CANONICAL_WINDOW = 2

# Neighbours of canonical in each dimension (immediate adjacent values)
DEPTH_NEIGHBOURS  = {0.10, 0.50}    # immediate neighbours of 0.25 in the grid
WINDOW_NEIGHBOURS = {1, 3}          # immediate neighbours of 2 in the grid

# MC
MC_RUNS  = 1000
MC_SEED  = 42
MC_BLOCKS = [10]   # block=10 only — sufficient for sensitivity scan

# T6
START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35
PORT_MAX_OPEN  = 8

# Baseline
BL_AVG_R = 1.1011
BL_PF    = 3.072
BL_CAGR  = 14.3

STABILITY_THRESHOLD = 0.67   # 67% of adjacent cells must show improvement


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(close); tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
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
# BACKTEST -- parametric Variant A
# =============================================================================

def backtest_a(df: pd.DataFrame, symbol: str,
               pullback_frac: float, cancel_bars: int) -> List[dict]:
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
    pos:     Optional[dict] = None
    pending: Optional[dict] = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])): continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])): continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])): continue

        # MANAGE OPEN POSITION
        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry"])/pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["entry"])/pos["risk"])
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
                net_r = (exit_px-pos["entry"])/pos["risk"]
                trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
                    exit_time=ts[i], net_r=net_r,
                    mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
                pos = None; pending = None; continue
            if pos["mfe_r"] >= CHAN_ACT_R: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"]-atr14[i]*CHAN_TRAIL)

        # TRY LIMIT FILL
        if pos is None and pending is not None:
            if lo[i] <= pending["limit_px"]:
                lx = pending["limit_px"]; risk = pending["risk"]
                pos = dict(entry=lx, stop=lx-risk, risk=risk, entry_time=ts[i],
                           hh=hi[i], chan_active=False, chan_stop=lx-risk,
                           mfe_r=0., mae_r=0., bars=1)
                pending = None
            else:
                pending["bars_left"] -= 1
                if pending["bars_left"] <= 0:
                    pending = None

        # NEW SIGNAL
        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            lx   = don_upper[i] - pullback_frac * atr14[i-1]
            risk = atr14[i-1] * STOP_MULT
            pending = dict(limit_px=lx, risk=risk, bars_left=cancel_bars)

    if pos is not None:
        exit_px = cl[-1]; net_r = (exit_px-pos["entry"])/pos["risk"]
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1], net_r=net_r,
            mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
    return trades


def run_universe(pullback_frac: float, cancel_bars: int,
                 symbols: List[str]) -> pd.DataFrame:
    all_trades = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(backtest_a(df, sym, pullback_frac, cancel_bars))
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

def _bstrap(vals, bs, rng):
    n=len(vals); out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); e=min(s+bs,n); blk=vals[s:e]
        if len(blk)<bs: blk=np.concatenate([blk,vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n], dtype=float)

def mc_p05(r: np.ndarray) -> float:
    rng = np.random.default_rng(MC_SEED)
    tots = [_bstrap(r, 10, rng).sum() for _ in range(MC_RUNS)]
    return float(np.percentile(tots, 5))


def t5_replay(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = df.sort_values(ENTRY_T).reset_index(drop=True)
    open_pos: list = []; closed: list = []
    def _flush(now):
        nonlocal open_pos
        still=[p for p in open_pos if p["exit_time"]>now]
        closed.extend(p for p in open_pos if p["exit_time"]<=now); open_pos[:]=still
    for _,row in df.iterrows():
        _flush(row[ENTRY_T])
        if any(p["symbol"]==row["symbol"] for p in open_pos): continue
        if len(open_pos)>=PORT_MAX_OPEN: continue
        open_pos.append({"symbol":row["symbol"],"entry_time":row[ENTRY_T],
                          "exit_time":row[EXIT_T],R_COL:row[R_COL]})
    if open_pos:
        last=max(p["exit_time"] for p in open_pos); _flush(last+pd.Timedelta(seconds=1))
    acc=pd.DataFrame(closed)
    if acc.empty: return acc, {}
    return acc.sort_values("exit_time").reset_index(drop=True), {}


def t6_equity(acc: pd.DataFrame) -> dict:
    if acc.empty: return {}
    equity=START_CAP; peak=START_CAP; rows=[]
    for _,row in acc.sort_values("exit_time").iterrows():
        pnl=row[R_COL]*equity*RISK_PCT; equity+=pnl; peak=max(peak,equity)
        rows.append(dict(equity=equity, peak=peak, dd_pct=(equity-peak)/peak))
    eq=pd.DataFrame(rows); final=float(eq["equity"].iloc[-1])
    start=acc[ENTRY_T].min(); end=acc["exit_time"].max()
    years=float((end-start).days/365.25) if pd.notnull(start) and pd.notnull(end) else 1.
    cagr=float((final/START_CAP)**(1/years)-1) if years>0 else 0.
    return dict(cagr_pct=float(cagr*100), max_dd_pct=float(eq["dd_pct"].min()*100))


def evaluate_combo(depth: float, window: int, symbols: List[str]) -> dict:
    trades = run_universe(depth, window, symbols)
    if trades.empty:
        return dict(depth=depth, window=window, trades=0, avg_r=0., pf=0.,
                    cagr=0., max_dd=0., mc_p05=0.,
                    beats_avg_r=False, beats_pf=False, improves=False)

    r = trades[R_COL].to_numpy(dtype=float)
    avg_r_val = float(r.mean())
    pf_val    = _pf(r)
    mc5       = mc_p05(r)

    acc, _ = t5_replay(trades)
    t6 = t6_equity(acc)
    cagr   = t6.get("cagr_pct", 0.)
    max_dd = t6.get("max_dd_pct", 0.)

    beats_r  = avg_r_val > BL_AVG_R
    beats_pf = pf_val    > BL_PF
    improves = beats_r and beats_pf   # both must improve for "improvement"

    return dict(depth=depth, window=window, trades=int(len(r)),
                avg_r=avg_r_val, pf=pf_val, cagr=cagr,
                max_dd=max_dd, mc_p05=mc5,
                beats_avg_r=beats_r, beats_pf=beats_pf, improves=improves)


# =============================================================================
# STABILITY CHECK
# =============================================================================

def stability_check(results_df: pd.DataFrame) -> dict:
    """
    Adjacent cells in the 3x3 neighbourhood of canonical (0.25, 2):
      depths  : {0.10, 0.50}
      windows : {1, 2, 3}        (canonical window 2 + its immediate neighbours)
      All combos: 2 depths x 3 windows = 6, plus 1 depth (0.25) x 2 non-canonical windows = 2
      Total = 8 adjacent cells
    Improvement criterion: avg_r > baseline AND PF > baseline
    """
    adjacent_mask = (
        (results_df["depth"].isin(DEPTH_NEIGHBOURS) &
         results_df["window"].isin(WINDOW_NEIGHBOURS | {CANONICAL_WINDOW})) |
        (results_df["depth"] == CANONICAL_DEPTH) &
         results_df["window"].isin(WINDOW_NEIGHBOURS)
    )
    adjacent = results_df[adjacent_mask & ~(
        (results_df["depth"] == CANONICAL_DEPTH) &
        (results_df["window"] == CANONICAL_WINDOW)
    )].copy()

    n_adj      = len(adjacent)
    n_improve  = int(adjacent["improves"].sum())
    n_beats_r  = int(adjacent["beats_avg_r"].sum())
    pct        = n_improve / n_adj if n_adj > 0 else 0.
    passes     = pct >= STABILITY_THRESHOLD

    return dict(
        n_adjacent=n_adj,
        n_improving=n_improve,
        n_beats_avg_r=n_beats_r,
        pct_improving=pct,
        passes=passes,
        adjacent_rows=adjacent,
    )


# =============================================================================
# REPORT
# =============================================================================

def write_report(results_df: pd.DataFrame, stab: dict) -> None:
    lines = [
        "PHASE T17 OPT-17.3 -- Variant A Limit Pullback Sensitivity Analysis",
        "="*72, "",
        f"Grid     : {len(PULLBACK_DEPTHS)} depths x {len(CANCEL_WINDOWS)} windows = {len(PULLBACK_DEPTHS)*len(CANCEL_WINDOWS)} combos",
        f"Canonical: depth={CANONICAL_DEPTH}xATR  window={CANONICAL_WINDOW} bars",
        f"Baseline : avg_r=+{BL_AVG_R:.4f}R  PF={BL_PF:.3f}  CAGR=+{BL_CAGR:.1f}%",
        f"Improve? : avg_r > baseline AND PF > baseline",
        "",
        "="*72,
        "FULL GRID (rows=cancel_window, cols=pullback_depth)",
        "="*72,
        "",
    ]

    # --- avg_r sub-table ---
    lines.append("  avg_r (baseline=+1.101R):")
    lines.append(f"  {'win\\dep':>8s}" +
                 "".join(f"  {d:.2f}xATR" for d in PULLBACK_DEPTHS))
    lines.append("  " + "-"*62)
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            val = results_df[(results_df["depth"]==d) & (results_df["window"]==w)]["avg_r"].iloc[0]
            marker = "*" if val > BL_AVG_R else " "
            row_vals.append(f"  {val:>+8.4f}{marker}")
        is_canon = f"  <- canonical" if w == CANONICAL_WINDOW else ""
        lines.append(f"  {w:>8d}bars" + "".join(row_vals) + is_canon)

    lines.append("")
    lines.append("  (* = beats baseline avg_r of +1.101R)")
    lines.append("")

    # --- PF sub-table ---
    lines.append("  PF (baseline=3.072):")
    lines.append(f"  {'win\\dep':>8s}" +
                 "".join(f"  {d:.2f}xATR " for d in PULLBACK_DEPTHS))
    lines.append("  " + "-"*62)
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            val = results_df[(results_df["depth"]==d) & (results_df["window"]==w)]["pf"].iloc[0]
            marker = "*" if val > BL_PF else " "
            row_vals.append(f"  {val:>8.3f}{marker}")
        is_canon = f"  <- canonical" if w == CANONICAL_WINDOW else ""
        lines.append(f"  {w:>8d}bars" + "".join(row_vals) + is_canon)

    lines.append("")
    lines.append("  (* = beats baseline PF of 3.072)")
    lines.append("")

    # --- CAGR sub-table ---
    lines.append("  CAGR % (baseline=+14.3%):")
    lines.append(f"  {'win\\dep':>8s}" +
                 "".join(f"  {d:.2f}xATR" for d in PULLBACK_DEPTHS))
    lines.append("  " + "-"*62)
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            val = results_df[(results_df["depth"]==d) & (results_df["window"]==w)]["cagr"].iloc[0]
            marker = "*" if val > BL_CAGR else " "
            row_vals.append(f"  {val:>+8.1f}%{marker}")
        is_canon = f"  <- canonical" if w == CANONICAL_WINDOW else ""
        lines.append(f"  {w:>8d}bars" + "".join(row_vals) + is_canon)

    lines.append("")
    lines.append("  (* = beats baseline CAGR of +14.3%)")
    lines.append("")

    # --- trades sub-table ---
    lines.append("  Trade count (baseline=461):")
    lines.append(f"  {'win\\dep':>8s}" +
                 "".join(f"  {d:.2f}xATR" for d in PULLBACK_DEPTHS))
    lines.append("  " + "-"*62)
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            val = results_df[(results_df["depth"]==d) & (results_df["window"]==w)]["trades"].iloc[0]
            row_vals.append(f"  {val:>9d} ")
        is_canon = f"  <- canonical" if w == CANONICAL_WINDOW else ""
        lines.append(f"  {w:>8d}bars" + "".join(row_vals) + is_canon)

    lines.append("")

    # --- MC p05 sub-table ---
    lines.append("  MC p05 totalR (baseline=+347.7R):")
    lines.append(f"  {'win\\dep':>8s}" +
                 "".join(f"  {d:.2f}xATR" for d in PULLBACK_DEPTHS))
    lines.append("  " + "-"*62)
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            val = results_df[(results_df["depth"]==d) & (results_df["window"]==w)]["mc_p05"].iloc[0]
            marker = "*" if val > 0 else " "
            row_vals.append(f"  {val:>+8.1f}R{marker}")
        is_canon = f"  <- canonical" if w == CANONICAL_WINDOW else ""
        lines.append(f"  {w:>8d}bars" + "".join(row_vals) + is_canon)

    lines.append("")
    lines.append("  (* = MC p05 > 0)")
    lines.append("")

    # --- improvement map ---
    lines += [
        "="*72,
        "IMPROVEMENT MAP (avg_r > baseline AND PF > baseline)",
        "="*72, "",
        f"  {'win\\dep':>8s}" +
        "".join(f"  {d:.2f}xATR" for d in PULLBACK_DEPTHS),
        "  " + "-"*62,
    ]
    for w in CANCEL_WINDOWS:
        row_vals = []
        for d in PULLBACK_DEPTHS:
            row = results_df[(results_df["depth"]==d) & (results_df["window"]==w)].iloc[0]
            if d == CANONICAL_DEPTH and w == CANONICAL_WINDOW:
                cell = "  [CANON]  "
            elif row["improves"]:
                cell = "  [ YES ]  "
            else:
                cell = "  [  NO ]  "
            row_vals.append(cell)
        lines.append(f"  {w:>8d}bars" + "".join(row_vals))

    lines.append("")

    # --- stability verdict ---
    adj = stab["adjacent_rows"]
    lines += [
        "="*72,
        "STABILITY CHECK (Unger §2.1 applied to entry params)",
        "="*72, "",
        f"  Canonical    : depth={CANONICAL_DEPTH}xATR, window={CANONICAL_WINDOW} bars",
        f"  Canonical improves? : {'YES' if results_df[(results_df['depth']==CANONICAL_DEPTH)&(results_df['window']==CANONICAL_WINDOW)]['improves'].iloc[0] else 'NO'}",
        "",
        f"  Adjacent cells ({stab['n_adjacent']} total — 3x3 neighbourhood minus canonical):",
    ]
    for _, r in adj.iterrows():
        flag = "IMPROVE" if r["improves"] else "no     "
        lines.append(f"    depth={r['depth']:.2f}xATR  window={int(r['window'])}bars  "
                     f"avg_r={r['avg_r']:+.4f}  PF={r['pf']:.3f}  [{flag}]")

    lines += [
        "",
        f"  Adjacent improving   : {stab['n_improving']} / {stab['n_adjacent']}  "
        f"({stab['pct_improving']:.1%})",
        f"  Threshold            : >= {STABILITY_THRESHOLD:.0%}",
        f"  Beats avg_r only     : {stab['n_beats_avg_r']} / {stab['n_adjacent']}",
        "",
        f"  STABILITY VERDICT: {'PASS -- improvement is robust' if stab['passes'] else 'FAIL -- parameter peak, reject Variant A'}",
        "",
        "="*72,
        "RECOMMENDATION",
        "="*72, "",
    ]

    if stab["passes"]:
        lines += [
            f"  Stability passed: {stab['pct_improving']:.1%} of adjacent cells improve on both",
            f"  avg_r and PF. The (0.25xATR, 2-bar) canonical is NOT a parameter peak.",
            "",
            "  Variant A may be adopted IF the CAGR deficit vs baseline (-1.1pp) and",
            "  increased DD (-5.9% vs -3.4%) are acceptable trade-offs for the improved",
            "  per-trade quality. Decision requires human review.",
        ]
    else:
        lines += [
            f"  Stability FAILED: only {stab['pct_improving']:.1%} of adjacent cells improve.",
            "  The (0.25xATR, 2-bar) result is a parameter peak -- do NOT adopt Variant A.",
        ]

    rpt = OUT_DIR / "sensitivity_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    total   = len(PULLBACK_DEPTHS) * len(CANCEL_WINDOWS)

    print("="*72)
    print("Phase T17 Opt-17.3 -- Variant A Sensitivity Analysis")
    print("="*72)
    print(f"Grid: {len(PULLBACK_DEPTHS)} depths x {len(CANCEL_WINDOWS)} windows = {total} combos")
    print(f"MC:   {MC_RUNS} runs per combo  |  Universe: {len(symbols)} symbols")
    print()

    rows = []
    done = 0
    for depth in PULLBACK_DEPTHS:
        for window in CANCEL_WINDOWS:
            done += 1
            result = evaluate_combo(depth, window, symbols)
            rows.append(result)
            flag = "IMPROVE" if result["improves"] else "      "
            canon = " <- canonical" if (depth==CANONICAL_DEPTH and window==CANONICAL_WINDOW) else ""
            print(f"  [{done:>2d}/{total}] depth={depth:.2f}xATR  win={window}  "
                  f"trades={result['trades']:4d}  avg_r={result['avg_r']:+.4f}  "
                  f"PF={result['pf']:.3f}  CAGR={result['cagr']:+.1f}%  "
                  f"p05={result['mc_p05']:+.1f}R  [{flag}]{canon}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False)

    stab = stability_check(results_df)

    print()
    print("="*72)
    print("STABILITY CHECK")
    print("="*72)
    print(f"  Adjacent cells improving: {stab['n_improving']}/{stab['n_adjacent']}  "
          f"({stab['pct_improving']:.1%})")
    print(f"  Threshold: >= {STABILITY_THRESHOLD:.0%}")
    print(f"  VERDICT: {'PASS -- robust improvement' if stab['passes'] else 'FAIL -- parameter peak'}")

    write_report(results_df, stab)

    print()
    print("="*72)
    print(f"Sensitivity analysis complete -> {OUT_DIR}")
    print("="*72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
