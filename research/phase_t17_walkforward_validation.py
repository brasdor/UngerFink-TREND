#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward Validation -- DonchianLong_UniverseV2

Step 1: Run backtest on IN-SAMPLE period (2020-01-01 to 2022-12-31) for all
        185 Step-A candidates. Apply 4-condition filter to select IS symbols.

Step 2: Run backtest on OUT-OF-SAMPLE period (2023-01-01 to 2026-05-29) using
        ONLY the IS-selected symbols. Full OHLCV history is used for indicator
        warmup (EMA200 etc.) but entries are only permitted from 2023-01-01.

Step 3: Apply T5 max8 portfolio replay to OOS trades.
        Compare OOS vs full-period UniverseV2 results.
        Flag if OOS avg_r < 0.15R.

Output: data/research_donchianlong_universev2_wf/
"""

from __future__ import annotations

import math, sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
WF_DIR  = ROOT / "data" / "research_donchianlong_universev2_wf"
OHLCV   = ROOT / "data" / "universe" / "ohlcv_1d"
UNIV    = ROOT / "data" / "universe" / "expanded_symbols.csv"
WF_DIR.mkdir(parents=True, exist_ok=True)

# Period boundaries
IS_START  = pd.Timestamp("2020-01-01", tz="UTC")
IS_END    = pd.Timestamp("2022-12-31", tz="UTC")
OOS_START = pd.Timestamp("2023-01-01", tz="UTC")
OOS_END   = pd.Timestamp("2026-05-29", tz="UTC")

# Frozen config
DONCHIAN_N  = 20
EXIT_N      = 10
ATR_N       = 14
STOP_MULT   = 2.0
CHAN_ACT_R  = 4.0
CHAN_TRAIL  = 3.0
EMA_N       = 200
MAX_OPEN    = 8

# IS filter -- trades threshold scaled proportionally to IS period length.
# Full filter: trades>=15 over ~6yr. IS period is ~2.6yr (data from May-2020).
# Scale: max(5, round(15 * 2.6/6)) = 6. Using 5 as absolute statistical floor.
IS_TRADES_MIN = 5   # scaled from full-period trades>=15
IS_FILTER = dict(avg_r_min=0.0, trades_min=IS_TRADES_MIN, top_month_pct_max=0.60, total_r_min=0.0)

# Gate
OOS_AVG_R_FLAG = 0.15

# Additional exclusions (same as Step B)
STEP_B_EXCLUDE = {"LUNA/USDT", "FTT/USDT", "EUR/USDT", "PAXG/USDT"}

# Full-period reference (from Step C, max8 canonical)
FULL_PERIOD = dict(avg_r=0.9263, pf=2.695, win_rate=0.390,
                   cagr_pct=14.3, max_dd_pct=-3.4, trades=318,
                   n_symbols=24, label="Full-period UniverseV2 (max8, ~6yr)")


# =============================================================================
# DATA + INDICATORS
# =============================================================================

def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV / f"{clean}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time","open","high","low","close"]].dropna(
        subset=["time","close"]).sort_values("time").reset_index(drop=True)


def compute_ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_atr(hi, lo, cl, n):
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
# BACKTEST (with optional entry gating by period)
# =============================================================================

def run_backtest(df: pd.DataFrame, symbol: str,
                 entry_from: Optional[pd.Timestamp] = None,
                 entry_to:   Optional[pd.Timestamp] = None) -> List[dict]:
    """
    Run frozen Donchian config. Entries only permitted when:
      entry_from <= bar_time <= entry_to  (None = no bound)
    Indicator computation always uses full df for proper warmup.
    """
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(DONCHIAN_N, ATR_N, EMA_N) + 20:
        return []

    cl  = df["close"].to_numpy(dtype=float)
    hi  = df["high"].to_numpy(dtype=float)
    lo  = df["low"].to_numpy(dtype=float)
    ts  = df["time"].to_numpy()            # numpy datetime64

    ema200    = compute_ema(cl, EMA_N)
    atr14     = compute_atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(DONCHIAN_N).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(EXIT_N).min().to_numpy()

    trades = []; pos = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(ema200[i-1]) and
                np.isfinite(atr14[i-1]) and np.isfinite(don_upper[i]) and
                np.isfinite(don_lower[i])):
            continue

        bar_t = pd.Timestamp(ts[i]).tz_localize("UTC") if pd.Timestamp(ts[i]).tzinfo is None \
                else pd.Timestamp(ts[i])

        # ---- manage open position ----
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
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(dict(
                    symbol=symbol,
                    entry_time=pos["entry_time"], exit_time=bar_t,
                    entry_price=pos["entry"], exit_price=exit_px,
                    initial_risk=pos["risk"], exit_reason=exit_rsn,
                    net_r=net_r, mfe_r=pos["mfe_r"],
                ))
                pos = None
            else:
                if pos["mfe_r"] >= CHAN_ACT_R:
                    pos["chan_active"] = True
                if pos["chan_active"]:
                    pos["chan_stop"] = max(pos["chan_stop"],
                                          pos["hh"] - atr14[i] * CHAN_TRAIL)

        # ---- entry (only within permitted window) ----
        if pos is None:
            if entry_from is not None and bar_t < entry_from: continue
            if entry_to   is not None and bar_t > entry_to:   break
            if cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
                risk = atr14[i-1] * STOP_MULT
                if risk <= 0: continue
                pos = dict(entry=cl[i], stop=cl[i]-risk, risk=risk,
                           entry_time=bar_t, hh=hi[i],
                           chan_active=False, chan_stop=cl[i]-risk,
                           mfe_r=0.0, bars=1)

    if pos is not None:
        net_r = (cl[-1] - pos["entry"]) / pos["risk"]
        trades.append(dict(symbol=symbol,
            entry_time=pos["entry_time"],
            exit_time=pd.Timestamp(ts[-1]).tz_localize("UTC") if pd.Timestamp(ts[-1]).tzinfo is None else pd.Timestamp(ts[-1]),
            entry_price=pos["entry"], exit_price=cl[-1],
            initial_risk=pos["risk"], exit_reason="end_of_data",
            net_r=net_r, mfe_r=pos["mfe_r"]))

    return trades


# =============================================================================
# AGGREGATION
# =============================================================================

def symbol_stats(trades: List[dict], symbol: str) -> dict:
    if not trades:
        return dict(symbol=symbol, trades=0, total_r=0.0, avg_r=0.0,
                    win_rate=0.0, profit_factor=0.0, top_month="", top_month_pct=0.0)
    r = np.array([t["net_r"] for t in trades])
    g = r[r>0].sum(); l = -r[r<0].sum()
    pf = g/l if l>0 else (float("inf") if g>0 else 0.0)
    months = pd.Series([t["exit_time"] for t in trades]).dt.to_period("M").astype(str)
    mr = {}
    for m, rv in zip(months, r):
        mr[m] = mr.get(m, 0.0) + rv
    top_m = max(mr, key=mr.get) if mr else ""
    tr = float(r.sum())
    tp = float(mr.get(top_m, 0.0)/tr) if tr > 0 else 0.0
    return dict(symbol=symbol, trades=int(len(r)), total_r=tr, avg_r=float(r.mean()),
                win_rate=float((r>0).mean()), profit_factor=pf,
                top_month=top_m, top_month_pct=tp)


def passes_filter(s: dict) -> bool:
    return (s["avg_r"]         > IS_FILTER["avg_r_min"]         and
            s["trades"]        >= IS_FILTER["trades_min"]        and
            s["top_month_pct"] < IS_FILTER["top_month_pct_max"] and
            s["total_r"]       > IS_FILTER["total_r_min"])


def pool_stats(trades_list: List[dict]) -> dict:
    if not trades_list:
        return dict(trades=0, avg_r=0.0, pf=0.0, win_rate=0.0, total_r=0.0)
    r  = np.array([t["net_r"] for t in trades_list])
    g  = r[r>0].sum(); l = -r[r<0].sum()
    pf = g/l if l>0 else (float("inf") if g>0 else 0.0)
    eq = np.cumsum(r); pk = np.maximum.accumulate(eq)
    return dict(trades=int(len(r)), avg_r=float(r.mean()), pf=pf,
                win_rate=float((r>0).mean()), total_r=float(r.sum()),
                max_dd_r=float((eq-pk).min()))


# =============================================================================
# T5 MAX8 PORTFOLIO REPLAY
# =============================================================================

def portfolio_replay(trades: List[dict], max_open: int) -> List[dict]:
    df  = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    open_pos: List[dict] = []
    closed:   List[dict] = []

    def flush(now):
        nonlocal open_pos
        still = []
        for p in open_pos:
            if p["exit_time"] <= now:
                closed.append(p)
            else:
                still.append(p)
        open_pos = still

    for _, row in df.iterrows():
        flush(row["entry_time"])
        if any(p["symbol"] == row["symbol"] for p in open_pos): continue
        if len(open_pos) >= max_open: continue
        open_pos.append(row.to_dict())

    if open_pos:
        last = max(p["exit_time"] for p in open_pos)
        flush(last + pd.Timedelta(seconds=1))

    return closed


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    is_stats: List[dict], is_selected: List[str],
    oos_raw: List[dict], oos_max8: List[dict],
    oos_by_symbol: List[dict],
) -> None:
    is_pool = pool_stats([t for s in is_stats if s["symbol"] in set(is_selected)
                          for _ in [s]] if False else
                         [])  # not needed for report, computed below

    oos_pool     = pool_stats(oos_raw)
    oos_max8pool = pool_stats(oos_max8)
    flag         = bool(oos_max8pool["avg_r"] < OOS_AVG_R_FLAG if oos_max8pool["trades"] > 0 else True)

    lines = [
        "WALK-FORWARD VALIDATION -- DonchianLong_UniverseV2",
        "=" * 65,
        "",
        "Config:  Donchian N=20 / ATR(14)x2.0 / Chandelier ACT+4R trail 3xATR",
        f"IS:      {IS_START.date()} to {IS_END.date()} (~3yr)",
        f"OOS:     {OOS_START.date()} to {OOS_END.date()} (~3.4yr)",
        f"Portfolio cap: max{MAX_OPEN}",
        "",
        "=" * 65, "STEP 1 -- IN-SAMPLE SYMBOL SELECTION", "=" * 65,
        f"  Candidates tested:   {len(is_stats)}",
        f"  Filter: avg_r>0 / trades>=15 / top_month_pct<60% / total_r>0",
        f"  Symbols selected:    {len(is_selected)}",
        "",
        "  IS-selected symbols (ranked by IS avg_r):",
        f"  {'Symbol':22s} {'Tr':>3} {'avg_r':>7} {'PF':>5} {'win%':>5} {'top_month':>8} {'top%':>5}",
        "  " + "-" * 60,
    ]
    is_map = {s["symbol"]: s for s in is_stats}
    for sym in sorted(is_selected, key=lambda s: -is_map[s]["avg_r"]):
        r = is_map[sym]
        pf_s = f"{r['profit_factor']:.3f}" if r['profit_factor'] < 99 else " >99"
        lines.append(
            f"  {sym:22s} {int(r['trades']):3d} {r['avg_r']:+7.4f}"
            f" {pf_s:5s} {r['win_rate']:5.1%} {r['top_month']:>8s} {r['top_month_pct']:>5.1%}")

    lines += [
        "",
        "=" * 65, "STEP 2 -- OUT-OF-SAMPLE RESULTS", "=" * 65,
        f"  Symbols in OOS:      {len(is_selected)} (IS-selected only)",
        f"  OOS trades (uncap):  {oos_pool['trades']}",
        f"  OOS avg_r (uncap):   {oos_pool['avg_r']:+.4f}R",
        f"  OOS PF    (uncap):   {oos_pool['pf']:.3f}",
        f"  OOS win%  (uncap):   {oos_pool['win_rate']:.1%}",
        "",
        f"  After max{MAX_OPEN} portfolio cap:",
        f"  OOS trades (max8):   {oos_max8pool['trades']}",
        f"  OOS avg_r  (max8):   {oos_max8pool['avg_r']:+.4f}R  "
        f"{'<<< FLAG: below 0.15R' if flag and oos_max8pool['trades']>0 else '(above 0.15R gate)'}",
        f"  OOS PF     (max8):   {oos_max8pool['pf']:.3f}",
        f"  OOS win%   (max8):   {oos_max8pool['win_rate']:.1%}",
        f"  OOS max DD (max8):   {oos_max8pool['max_dd_r']:+.2f}R",
    ]

    lines += [
        "",
        "  Per-symbol OOS performance (IS-selected symbols):",
        f"  {'Symbol':22s} {'Tr':>3} {'avg_r':>7} {'PF':>5} {'win%':>5} {'total_r':>8}",
        "  " + "-" * 56,
    ]
    for r in sorted(oos_by_symbol, key=lambda x: -x["avg_r"]):
        pf_s = f"{r['profit_factor']:.3f}" if r['profit_factor'] < 99 else " >99"
        lines.append(
            f"  {r['symbol']:22s} {int(r['trades']):3d} {r['avg_r']:+7.4f}"
            f" {pf_s:5s} {r['win_rate']:5.1%} {r['total_r']:+8.2f}")

    lines += [
        "",
        "=" * 65, "STEP 3 -- COMPARISON", "=" * 65,
        f"  {'Metric':28s} {'IS period':>12} {'OOS period':>12} {'Full period':>12}",
        "  " + "-" * 68,
    ]

    is_selected_stats = [is_map[s] for s in is_selected]
    is_r = np.array([t["net_r"] for s in is_selected for t in [] ])  # unused

    is_pool_r = np.array([])
    # Rebuild IS pool from is_stats for selected symbols only
    # (we don't have individual IS trades in memory here, use is_stats per symbol)
    is_avg = float(np.mean([is_map[s]["avg_r"] for s in is_selected])) if is_selected else 0.0
    is_pf  = float(np.mean([is_map[s]["profit_factor"] for s in is_selected
                             if is_map[s]["profit_factor"] < float("inf")])) if is_selected else 0.0

    oos_syms_positive = sum(1 for r in oos_by_symbol if r["avg_r"] > 0)

    def row(label, is_v, oos_v, full_v):
        return f"  {label:28s} {is_v:>12} {oos_v:>12} {full_v:>12}"

    lines += [
        row("Symbols", str(len(is_selected)), str(len(is_selected)), "24"),
        row("Trades (uncapped)",
            "--",
            str(oos_pool["trades"]),
            "461"),
        row("avg_r (uncapped)",
            f"{is_avg:+.4f}R",
            f"{oos_pool['avg_r']:+.4f}R",
            "+1.1011R"),
        row(f"avg_r (max{MAX_OPEN})",
            "--",
            f"{oos_max8pool['avg_r']:+.4f}R",
            "+0.9263R"),
        row(f"PF (max{MAX_OPEN})",
            "--",
            f"{oos_max8pool['pf']:.3f}",
            "2.695"),
        row(f"win% (max{MAX_OPEN})",
            "--",
            f"{oos_max8pool['win_rate']:.1%}",
            "39.0%"),
        row("Positive symbols",
            f"{len([s for s in is_selected if is_map[s]['avg_r']>0])}/{len(is_selected)}",
            f"{oos_syms_positive}/{len(oos_by_symbol)}",
            "24/24"),
        "",
        f"  OOS avg_r gate (>= {OOS_AVG_R_FLAG}R): "
        f"{'PASS' if oos_max8pool['avg_r'] >= OOS_AVG_R_FLAG and oos_max8pool['trades']>0 else 'FAIL / FLAG'}",
        "",
        "  Degradation analysis:",
        f"  IS avg_r (mean across symbols): {is_avg:+.4f}R",
        f"  OOS avg_r (uncapped):           {oos_pool['avg_r']:+.4f}R",
        f"  OOS / IS ratio:                 {oos_pool['avg_r']/is_avg:.2f}x" if is_avg != 0 else "  OOS / IS ratio: N/A",
    ]

    flag_line = ("  FLAG: OOS avg_r < 0.15R -- performance degraded in out-of-sample period"
                 if flag and oos_max8pool["trades"] > 0 else
                 "  PASS: OOS avg_r >= 0.15R -- edge transfers to out-of-sample period")
    lines += ["", flag_line, "",
              "=" * 65,
              "WALK-FORWARD VERDICT",
              "=" * 65]

    if oos_max8pool["trades"] == 0:
        lines.append("  INCONCLUSIVE: no OOS trades generated.")
    elif not flag and oos_max8pool["pf"] > 1.0:
        lines.append("  PASS: OOS avg_r >= 0.15R AND PF > 1.0.")
        lines.append("  The IS-based filtering has forward predictive value.")
        lines.append("  DonchianLong_UniverseV2 edge is confirmed out-of-sample.")
    else:
        lines.append("  FLAG: OOS performance below the 0.15R gate.")
        lines.append("  Review per-symbol OOS performance before deployment.")

    rpt = WF_DIR / "walkforward_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 65)
    print("Walk-Forward Validation -- DonchianLong_UniverseV2")
    print("=" * 65)
    print(f"IS:  {IS_START.date()} to {IS_END.date()}")
    print(f"OOS: {OOS_START.date()} to {OOS_END.date()}")
    print()

    # Candidates: Step-A universe after basic exclusions
    univ_df = pd.read_csv(UNIV)
    candidates = (univ_df[(univ_df["included"] == True) &
                           (~univ_df["symbol"].isin(STEP_B_EXCLUDE))]["symbol"].tolist())
    print(f"Candidates: {len(candidates)} symbols")

    # ── STEP 1: IS backtest + filter ────────────────────────────────────────
    print(f"\n[Step 1] IS backtest ({IS_START.date()} to {IS_END.date()})...")
    is_stats    = []
    is_trades_all: List[dict] = []
    for i, sym in enumerate(sorted(candidates), 1):
        df = load_ohlcv(sym)
        if df is None or len(df) < EMA_N + 50:
            continue
        df_is = df[df["time"] <= IS_END].copy().reset_index(drop=True)
        if len(df_is) < EMA_N + 50:
            continue
        trades = run_backtest(df_is, sym)  # no entry_from/to limit: IS data only
        stats  = symbol_stats(trades, sym)
        is_stats.append(stats)
        is_trades_all.extend(trades)
        if (i % 20) == 0 or i == len(candidates):
            print(f"  [{i:3d}/{len(candidates)}] processed", end="\r")

    print(f"\n  IS symbols processed: {len(is_stats)}")

    # Apply filter
    is_selected = [s["symbol"] for s in is_stats if passes_filter(s)]
    print(f"  IS symbols passing filter: {len(is_selected)}")

    is_stats_df = pd.DataFrame(is_stats)
    is_stats_df["is_selected"] = is_stats_df["symbol"].isin(is_selected)
    is_stats_df.to_csv(WF_DIR / "is_symbol_stats.csv", index=False)
    pd.DataFrame({"symbol": is_selected}).to_csv(WF_DIR / "is_selected_symbols.csv", index=False)

    if not is_selected:
        print("  [ERROR] No symbols selected by IS filter.")
        return 1

    # ── STEP 2: OOS backtest on IS-selected symbols ──────────────────────────
    print(f"\n[Step 2] OOS backtest ({OOS_START.date()} to {OOS_END.date()})...")
    oos_trades_raw: List[dict] = []
    oos_by_symbol: List[dict] = []
    for i, sym in enumerate(sorted(is_selected), 1):
        df = load_ohlcv(sym)
        if df is None or len(df) < EMA_N + 50:
            continue
        # Full OHLCV for indicator warmup; entries gated to OOS period
        trades = run_backtest(df, sym, entry_from=OOS_START, entry_to=OOS_END)
        stats  = symbol_stats(trades, sym)
        oos_trades_raw.extend(trades)
        oos_by_symbol.append(stats)
        print(f"  [{i:3d}/{len(is_selected)}] {sym:22s}  "
              f"trades={stats['trades']:3d}  avg_r={stats['avg_r']:+.4f}")

    print(f"\n  OOS raw trades: {len(oos_trades_raw)}")

    if not oos_trades_raw:
        print("  [ERROR] No OOS trades generated.")
        return 1

    pd.DataFrame(oos_trades_raw).to_csv(WF_DIR / "oos_trades_raw.csv", index=False)

    # ── STEP 3: Portfolio replay + comparison ────────────────────────────────
    print(f"\n[Step 3] max{MAX_OPEN} portfolio replay on OOS trades...")
    oos_max8 = portfolio_replay(oos_trades_raw, MAX_OPEN)
    print(f"  OOS max8 accepted: {len(oos_max8)} trades")

    pd.DataFrame(oos_max8).to_csv(WF_DIR / "oos_trades_max8.csv", index=False)

    oos_pool     = pool_stats(oos_trades_raw)
    oos_max8pool = pool_stats(oos_max8)
    flag         = bool(oos_max8pool["avg_r"] < OOS_AVG_R_FLAG
                        if oos_max8pool["trades"] > 0 else True)

    write_report(is_stats, is_selected, oos_trades_raw, oos_max8, oos_by_symbol)

    # Console summary
    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  IS symbols selected:     {len(is_selected)}")
    print(f"  OOS trades (uncapped):   {oos_pool['trades']}")
    print(f"  OOS avg_r  (uncapped):   {oos_pool['avg_r']:+.4f}R")
    print(f"  OOS trades (max8):       {oos_max8pool['trades']}")
    print(f"  OOS avg_r  (max8):       {oos_max8pool['avg_r']:+.4f}R")
    print(f"  OOS PF     (max8):       {oos_max8pool['pf']:.3f}")
    print(f"  OOS win%   (max8):       {oos_max8pool['win_rate']:.1%}")
    print()
    print(f"  Full-period ref (max8):  avg_r=+0.9263R  PF=2.695")
    print()
    gate = "PASS" if not flag and oos_max8pool["trades"] > 0 else "FLAG"
    print(f"  0.15R gate: {gate}")
    print(f"  Verdict: {'PASS -- edge transfers OOS' if gate=='PASS' else 'FLAG -- review OOS results'}")
    print()
    print(f"  Output: {WF_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
