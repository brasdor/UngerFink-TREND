#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T15 + T16 Validation -- DonchianLong_UniverseV2

T15: Parameter stability -- test N=[15, 20, 25] on 24-symbol universe.
     All three N values must show positive avg_r and PF > 1.0.
     N=20 baseline already done in Step C; N=15 and N=25 re-run here.

T16: Full Monte Carlo -- 5000 runs x block sizes [1, 5, 10, 20, 50]
     on T5 max8 canonical trades (318 trades).
     Gate: p05 total_r > 0.

T15 output: data/research_donchianlong_universev2_t15/
T16 output: data/research_donchianlong_universev2_t16/
"""

from __future__ import annotations

import math, sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

T15_DIR = ROOT / "data" / "research_donchianlong_universev2_t15"
T16_DIR = ROOT / "data" / "research_donchianlong_universev2_t16"
T15_DIR.mkdir(parents=True, exist_ok=True)
T16_DIR.mkdir(parents=True, exist_ok=True)

OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
SYMBOLS_CSV = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
T5_TRADES  = ROOT / "data" / "research_donchianlong_universev2" / "phase_c_t5_portfolio_trades.csv"
STEP_B_TRADES = ROOT / "data" / "universe" / "step_b_all_trades.csv"

# T15 grid
T15_N_VALUES = [15, 20, 25]
ATR_N        = 14
STOP_MULT    = 2.0
CHAN_ACT_R   = 4.0
CHAN_TRAIL   = 3.0
EMA_N        = 200

# T16
MC_RUNS   = 5000
MC_BLOCKS = [1, 5, 10, 20, 50]
MC_SEED   = 42

# Step C N=20 baseline (from master report)
N20_BASELINE = dict(n=20, trades=461, avg_r=1.1011, profit_factor=3.072,
                    win_rate=0.412, total_r=507.60, max_dd_r=-22.48)


# =============================================================================
# INDICATORS
# =============================================================================

def compute_ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_atr(high, low, close, n):
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


def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time","open","high","low","close"]].dropna(subset=["time","close"]).sort_values("time").reset_index(drop=True)


# =============================================================================
# BACKTEST (parameterised N)
# =============================================================================

def run_backtest(df: pd.DataFrame, symbol: str, n_val: int) -> List[dict]:
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(n_val, ATR_N, EMA_N) + 20:
        return []

    cl  = df["close"].to_numpy(dtype=float)
    hi  = df["high"].to_numpy(dtype=float)
    lo  = df["low"].to_numpy(dtype=float)
    ts  = df["time"].to_numpy()

    ema200    = compute_ema(cl, EMA_N)
    atr14     = compute_atr(hi, lo, cl, ATR_N)
    exit_n    = max(n_val // 2, 1)
    don_upper = pd.Series(hi).shift(1).rolling(n_val).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades = []; pos = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])):
            continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])):
            continue

        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry"])/pos["risk"])
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
                trades.append(dict(symbol=symbol, n=n_val,
                    entry_time=pos["entry_time"], exit_time=ts[i],
                    net_r=(exit_px-pos["entry"])/pos["risk"],
                    exit_reason=exit_rsn))
                pos = None
            else:
                if pos["mfe_r"] >= CHAN_ACT_R:
                    pos["chan_active"] = True
                if pos["chan_active"]:
                    pos["chan_stop"] = max(pos["chan_stop"],
                                          pos["hh"] - atr14[i]*CHAN_TRAIL)

        if pos is None:
            if cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
                risk = atr14[i-1] * STOP_MULT
                if risk <= 0: continue
                pos = dict(entry=cl[i], stop=cl[i]-risk, risk=risk,
                           entry_time=ts[i], hh=hi[i],
                           chan_active=False, chan_stop=cl[i]-risk,
                           mfe_r=0.0, bars=1)

    if pos is not None:
        trades.append(dict(symbol=symbol, n=n_val,
            entry_time=pos["entry_time"], exit_time=ts[-1],
            net_r=(cl[-1]-pos["entry"])/pos["risk"],
            exit_reason="end_of_data"))

    return trades


def summarize(r: np.ndarray) -> dict:
    if r.size == 0:
        return dict(trades=0, total_r=0.0, avg_r=0.0, win_rate=0.0,
                    profit_factor=0.0, max_dd_r=0.0)
    g = r[r>0].sum(); l = -r[r<0].sum()
    pf = g/l if l>0 else (float("inf") if g>0 else 0.0)
    eq = np.cumsum(r); pk = np.maximum.accumulate(eq)
    return dict(trades=int(r.size), total_r=float(r.sum()), avg_r=float(r.mean()),
                win_rate=float((r>0).mean()), profit_factor=pf,
                max_dd_r=float((eq-pk).min()))


# =============================================================================
# T15
# =============================================================================

def run_t15(symbols: List[str]) -> pd.DataFrame:
    all_trades = []

    for n_val in T15_N_VALUES:
        if n_val == 20:
            print(f"  N={n_val}: using Step C results (trades={N20_BASELINE['trades']}, "
                  f"avg_r={N20_BASELINE['avg_r']:+.4f})")
            continue

        print(f"  N={n_val}: backtesting {len(symbols)} symbols...", end=" ", flush=True)
        count = 0
        for sym in symbols:
            df = load_ohlcv(sym)
            if df is None or len(df) < EMA_N + 50:
                continue
            trades = run_backtest(df, sym, n_val)
            all_trades.extend(trades)
            count += 1
        print(f"loaded={count}, trades so far={len(all_trades)}")

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def build_t15_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_val in T15_N_VALUES:
        if n_val == 20:
            s = N20_BASELINE.copy()
            s["n"] = 20
        else:
            sub = trades_df[trades_df["n"] == n_val]
            r   = sub["net_r"].to_numpy(dtype=float)
            s   = summarize(r); s["n"] = n_val
        s["pass_avg_r"] = bool(s.get("avg_r", 0) > 0)
        s["pass_pf"]    = bool(s.get("profit_factor", 0) > 1.0)
        s["stable"]     = s["pass_avg_r"] and s["pass_pf"]
        rows.append(s)
    return pd.DataFrame(rows)


def write_t15_report(table: pd.DataFrame) -> None:
    all_pass = bool(table["stable"].all())
    lines = [
        "T15 -- DonchianLong_UniverseV2 Parameter Stability",
        "=" * 60,
        "",
        "Test: N=[15, 20, 25] on 24-symbol universe.",
        "Gate: ALL three N values must show avg_r > 0 AND PF > 1.0.",
        f"Canonical: N=20 (frozen config).",
        "",
        f"  {'N':>4}  {'Trades':>6}  {'avg_r':>7}  {'PF':>5}  {'win%':>5}  {'total_r':>8}  PASS",
        "  " + "-" * 52,
    ]
    for _, r in table.sort_values("n").iterrows():
        lines.append(
            f"  {int(r['n']):>4}  {int(r['trades']):>6}  {r['avg_r']:>+7.4f}"
            f"  {r['profit_factor']:>5.3f}  {r['win_rate']:>5.1%}"
            f"  {r['total_r']:>+8.2f}  {'PASS' if r['stable'] else 'FAIL'}"
        )
    lines += [
        "",
        f"OVERALL T15: {'PASS -- all N values stable' if all_pass else 'FAIL -- not all N values stable'}",
        "",
        "Interpretation:",
        "  PASS: canonical N=20 is not a local optimum;",
        "        adjacent N values also produce positive edge.",
        "  FAIL: would indicate curve-fitting -- N=20 cherry-picked.",
    ]
    rpt = T15_DIR / "phase_t15_stability_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] T15 report: {rpt}")
    return all_pass


# =============================================================================
# T16
# =============================================================================

def block_bootstrap(vals: np.ndarray, bs: int, rng) -> np.ndarray:
    n = len(vals); out = []
    while len(out) < n:
        s = int(rng.integers(0, n)); e = min(s+bs, n)
        blk = vals[s:e]
        if len(blk) < bs:
            blk = np.concatenate([blk, vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n], dtype=float)


def run_t16(r: np.ndarray) -> pd.DataFrame:
    rng  = np.random.default_rng(MC_SEED)
    rows = []
    for bs in MC_BLOCKS:
        tots, dds, pfs = [], [], []
        for _ in range(MC_RUNS):
            s    = block_bootstrap(r, bs, rng)
            tots.append(float(s.sum()))
            eq   = np.cumsum(s); pk = np.maximum.accumulate(eq)
            dds.append(float((eq-pk).min()))
            g = s[s>0].sum(); l = -s[s<0].sum()
            pfs.append(float(g/l) if l>0 else float("inf"))
        tots = np.array(tots); dds = np.array(dds); pfs_a = np.array(pfs)
        rows.append(dict(
            block_size=bs, mc_runs=MC_RUNS,
            total_r_p05=float(np.percentile(tots, 5)),
            total_r_p50=float(np.percentile(tots, 50)),
            total_r_p95=float(np.percentile(tots, 95)),
            dd_p05=float(np.percentile(dds, 5)),
            dd_p50=float(np.percentile(dds, 50)),
            dd_p95=float(np.percentile(dds, 95)),
            pf_p05=float(np.percentile(pfs_a[np.isfinite(pfs_a)], 5) if np.any(np.isfinite(pfs_a)) else 0.0),
            pf_p50=float(np.percentile(pfs_a[np.isfinite(pfs_a)], 50) if np.any(np.isfinite(pfs_a)) else 0.0),
            prob_positive=float((tots > 0).mean()),
        ))
        print(f"    bs={bs:2d}: p05={rows[-1]['total_r_p05']:+.1f}  "
              f"p50={rows[-1]['total_r_p50']:+.1f}  p95={rows[-1]['total_r_p95']:+.1f}  "
              f"prob_pos={rows[-1]['prob_positive']:.1%}")
    return pd.DataFrame(rows)


def write_t16_report(mc: pd.DataFrame, n_trades: int) -> None:
    gate_pass = bool((mc["total_r_p05"] > 0).all())
    lines = [
        "T16 -- DonchianLong_UniverseV2 Full Monte Carlo",
        "=" * 60,
        "",
        f"Input:    T5 max8 canonical trades  (n={n_trades})",
        f"Runs:     {MC_RUNS} per block size",
        f"Blocks:   {MC_BLOCKS}",
        "Gate:     p05 total_r > 0 for ALL block sizes",
        "",
        f"  {'bs':>4}  {'p05':>8}  {'p50':>8}  {'p95':>8}  "
        f"{'dd_p95':>8}  {'pf_p05':>6}  {'prob_pos':>8}  GATE",
        "  " + "-" * 68,
    ]
    for _, r in mc.iterrows():
        gate = "PASS" if r["total_r_p05"] > 0 else "FAIL"
        lines.append(
            f"  {int(r['block_size']):>4}  {r['total_r_p05']:>+8.1f}  "
            f"{r['total_r_p50']:>+8.1f}  {r['total_r_p95']:>+8.1f}  "
            f"{r['dd_p95']:>+8.2f}  {r['pf_p05']:>6.3f}  "
            f"{r['prob_positive']:>8.1%}  {gate}"
        )
    lines += [
        "",
        f"OVERALL T16: {'PASS -- p05 > 0 for all block sizes' if gate_pass else 'FAIL -- some p05 <= 0'}",
        "",
        "Interpretation:",
        "  p05 total_r > 0 at all block sizes confirms the edge is not",
        "  an artefact of trade ordering. The worst-case 5th percentile",
        "  of reshuffled trades remains profitable.",
        "  If p05 is positive at large block sizes (bs=20, 50), the edge",
        "  is robust to clustered market regimes -- not just isolated wins.",
    ]
    rpt = T16_DIR / "phase_t16_montecarlo_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] T16 report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(SYMBOLS_CSV)["symbol"].tolist()
    print(f"Universe: {len(symbols)} symbols")

    # ── T15 ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("T15 -- Parameter Stability  N=[15, 20, 25]")
    print("="*60)

    t15_raw = run_t15(symbols)
    if not t15_raw.empty:
        t15_raw.to_csv(T15_DIR / "phase_t15_all_trades.csv", index=False)

    t15_table = build_t15_table(t15_raw)
    t15_table.to_csv(T15_DIR / "phase_t15_stability_results.csv", index=False)

    print()
    print(f"  {'N':>4}  {'Trades':>6}  {'avg_r':>7}  {'PF':>5}  {'win%':>5}  STABLE")
    for _, r in t15_table.sort_values("n").iterrows():
        print(f"  {int(r['n']):>4}  {int(r['trades']):>6}  {r['avg_r']:>+7.4f}"
              f"  {r['profit_factor']:>5.3f}  {r['win_rate']:>5.1%}"
              f"  {'YES' if r['stable'] else 'NO'}")

    t15_pass = write_t15_report(t15_table)
    print(f"\n  T15 result: {'PASS' if t15_pass else 'FAIL'}")

    # ── T16 ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("T16 -- Full Monte Carlo  (5000 runs x block sizes [1,5,10,20,50])")
    print("="*60)

    # Load T5 max8 canonical trades
    if T5_TRADES.exists():
        t5_df = pd.read_csv(T5_TRADES)
        max8  = t5_df[t5_df["variant"] == "max8"].copy() if "variant" in t5_df.columns else t5_df
    else:
        # Fallback: use all Step B trades for the 24 symbols
        print("  [WARN] T5 trades not found, using uncapped Step B trades")
        raw = pd.read_csv(STEP_B_TRADES)
        max8 = raw[raw["symbol"].isin(symbols)]

    r_vals = max8["net_r"].to_numpy(dtype=float)
    print(f"  Input trades: {len(r_vals)}")
    print(f"  Actual avg_r: {r_vals.mean():+.4f}  total_r: {r_vals.sum():+.1f}")
    print()

    mc = run_t16(r_vals)
    mc.to_csv(T16_DIR / "phase_t16_montecarlo_results.csv", index=False)
    write_t16_report(mc, len(r_vals))

    t16_pass = bool((mc["total_r_p05"] > 0).all())
    print(f"\n  T16 result: {'PASS' if t16_pass else 'FAIL'}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  T15 stability: {'PASS' if t15_pass else 'FAIL'}")
    print(f"  T16 MC p05>0:  {'PASS' if t16_pass else 'FAIL'}")
    if t15_pass and t16_pass:
        print()
        print("  DonchianLong_UniverseV2 -- FULLY VALIDATED")
        print("  Ready for T9B paper trading with max8 portfolio cap.")
    else:
        print()
        print("  One or more validation phases failed -- review before deployment.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
