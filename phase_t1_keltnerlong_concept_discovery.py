#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- KeltnerLong Concept Discovery

Entry:   Close > EMA(N) + ATR(N)*keltner_mult  (Keltner upper band, shifted 1 bar)
Filter:  ema200_price: close > EMA200
         ema50_slope:  EMA50[i] > EMA50[i-1]  (slope positive)
Stop:    entry - ATR(N)*stop_mult  (below entry, long)
Exit 1:  low <= stop  -> exit at stop
Exit 2:  close < EMA(N)  -> exit at close  (midline cross-under)

Binance Spot, LONG only, 1d.
Cost floor:     avg_r > 0.15R  (Section 4.2)
Stability zone: canonical N +/-1 grid step, PASS >= 67%
Data:           data/research_trend_t15/ohlcv_cache/  (reuse long-side cache)
Output:         data/research_keltnerlong_t1/
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

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_keltnerlong_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME     = "1d"
BACKTEST_BARS = 1500

KELTNER_NS    = [10, 15, 20, 25, 30, 40, 55]
KELTNER_MULTS = [1.5, 2.0, 2.5, 3.0]
STOP_MULTS    = [2.0, 3.0]
FILTER_MODES  = ["ema200_price", "ema50_slope"]

EMA200_N = 200
EMA50_N  = 50

COST_FLOOR_R       = 0.15
STABILITY_PASS_PCT = 67.0
MIN_BARS           = 300

EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI", "WBTC", "BTTC",
}
EXCLUDE_KW = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")


# =============================================================================
# DATA
# =============================================================================

def discover_symbols() -> List[str]:
    syms = []
    for f in sorted(CACHE_DIR.glob("*_1d.csv")):
        parts = f.stem.split("_")
        if parts and parts[-1] == "1d":
            parts = parts[:-1]
        if len(parts) < 2:
            continue
        quote = parts[-1]
        base  = "_".join(parts[:-1])
        if quote != "USDT":
            continue
        if base in EXCLUDE_BASES:
            continue
        if any(kw in base for kw in EXCLUDE_KW):
            continue
        syms.append(f"{base}/{quote}")
    return syms


def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    fname = symbol.replace("/", "_") + "_1d.csv"
    path  = CACHE_DIR / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
        if len(df) < MIN_BARS:
            return None
        return df.tail(BACKTEST_BARS)
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def compute_indicators(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute Keltner Channel indicators for period N.
    Band components are stored shifted 1 bar (ema_n_s1, atr_n_s1)
    to avoid lookahead at entry.
    """
    d = df.copy()

    # ATR(N) -- Wilder simple rolling average of TR
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean()

    # EMA(N) -- midline of channel and exit trigger
    ema_n = d["close"].ewm(span=n, adjust=False).mean()

    # Current-bar values (used in exit logic and stop after entry)
    d["ema_n"] = ema_n
    d["atr_n"] = atr_n

    # Shifted-1-bar components (used for entry band -- no lookahead)
    d["ema_n_s1"] = ema_n.shift(1)
    d["atr_n_s1"] = atr_n.shift(1)

    # Regime filters (current bar, no shift needed -- bar already closed)
    d["ema200"]      = d["close"].ewm(span=EMA200_N, adjust=False).mean()
    d["ema50"]       = d["close"].ewm(span=EMA50_N,  adjust=False).mean()
    d["ema50_prev"]  = d["ema50"].shift(1)

    return d


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_symbol(
    symbol: str,
    ind: pd.DataFrame,
    keltner_mult: float,
    stop_mult: float,
    filter_mode: str,
    n: int,
) -> List[dict]:
    warmup = max(n, EMA200_N, EMA50_N) + 5
    pos    = None
    trades = []

    for i in range(warmup, len(ind)):
        row = ind.iloc[i]

        atr_n = float(row["atr_n"])
        if not math.isfinite(atr_n) or atr_n <= 0:
            continue

        close    = float(row["close"])
        high     = float(row["high"])
        low      = float(row["low"])
        t        = ind.index[i]
        ema_n    = float(row["ema_n"])
        ema_n_s1 = float(row["ema_n_s1"])
        atr_n_s1 = float(row["atr_n_s1"])
        ema200   = float(row["ema200"])
        ema50    = float(row["ema50"])
        ema50_p  = float(row["ema50_prev"])

        if not all(math.isfinite(x) for x in
                   [ema_n, ema_n_s1, atr_n_s1, ema200, ema50, ema50_p]):
            continue

        upper_band = ema_n_s1 + atr_n_s1 * keltner_mult

        if pos is not None:
            # Exit 1: stop hit (low touched or crossed stop)
            if low <= pos["stop"]:
                net_r = (pos["stop"] - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, pos["stop"], net_r, "stop",
                    n, keltner_mult, stop_mult, filter_mode,
                ))
                pos = None
                continue

            # Exit 2: midline cross-under (close drops below EMA(N))
            if close < ema_n:
                net_r = (close - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "midline_exit",
                    n, keltner_mult, stop_mult, filter_mode,
                ))
                pos = None
                continue

        if pos is None:
            # Filter
            if filter_mode == "ema200_price":
                regime_ok = close > ema200
            else:  # ema50_slope
                regime_ok = ema50 > ema50_p

            # Entry: close breaks above Keltner upper band (previous bar's band)
            if regime_ok and close > upper_band:
                risk = atr_n * stop_mult
                if risk > 0:
                    pos = {
                        "entry_time": t,
                        "entry":      close,
                        "stop":       close - risk,
                        "risk":       risk,
                    }

    return trades


def _make_trade(
    symbol: str,
    pos: dict,
    exit_time,
    exit_price: float,
    net_r: float,
    reason: str,
    n: int,
    keltner_mult: float,
    stop_mult: float,
    filter_mode: str,
) -> dict:
    return {
        "symbol":       symbol,
        "side":         "LONG",
        "timeframe":    TIMEFRAME,
        "filter_mode":  filter_mode,
        "keltner_n":    n,
        "keltner_mult": keltner_mult,
        "stop_mult":    stop_mult,
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry"],
        "exit_price":   float(exit_price),
        "initial_stop": pos["stop"],
        "initial_risk": pos["risk"],
        "net_r":        float(net_r),
        "exit_reason":  reason,
    }


# =============================================================================
# STATS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    return float(g / l) if l > 1e-12 else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(trades: List[dict]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, median_r=0.0)
    r = np.array([t["net_r"] for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, median_r=0.0)
    return dict(
        trades=int(len(r)),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        median_r=float(np.median(r)),
        pf=_pf(r),
        max_dd_r=_dd(r),
        win_rate_pct=float((r > 0).mean() * 100),
    )


# =============================================================================
# STABILITY
# =============================================================================

def _zone_for_n(n: int) -> List[int]:
    """Return N plus its adjacent grid neighbours (+/-1 step)."""
    idx  = KELTNER_NS.index(n)
    zone = []
    if idx > 0:
        zone.append(KELTNER_NS[idx - 1])
    zone.append(n)
    if idx < len(KELTNER_NS) - 1:
        zone.append(KELTNER_NS[idx + 1])
    return zone


def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, km, sm), g in grid.groupby(["filter_mode", "keltner_mult", "stop_mult"]):
        best: Optional[dict] = None

        for n in KELTNER_NS:
            zone_ns = _zone_for_n(n)
            zone_g  = g[g["keltner_n"].isin(zone_ns)]
            total   = len(zone_g)
            pf_gt1  = int((zone_g["pf"] > 1.0).sum())
            avg_r_p = int((zone_g["avg_r"] > 0).sum())
            cf_pass = int((zone_g["avg_r"] > COST_FLOOR_R).sum())
            pct     = 100.0 * pf_gt1 / max(total, 1)

            n_row    = g[g["keltner_n"] == n]
            c_trades = int(n_row["trades"].sum())          if not n_row.empty else 0
            c_avg_r  = float(n_row["avg_r"].mean())        if not n_row.empty else 0.0
            c_pf     = float(n_row["pf"].mean())           if not n_row.empty else 0.0
            c_dd     = float(n_row["max_dd_r"].mean())     if not n_row.empty else 0.0
            c_win    = float(n_row["win_rate_pct"].mean()) if not n_row.empty else 0.0

            candidate = dict(
                filter_mode=fm,
                keltner_mult=km,
                stop_mult=sm,
                canonical_n=n,
                zone_ns=str(zone_ns),
                zone_total=total,
                zone_pf_gt1=pf_gt1,
                zone_avg_r_positive=avg_r_p,
                zone_cost_floor_pass=cf_pass,
                zone_stability_pct=round(pct, 1),
                stability_verdict=(
                    "PASS" if pct >= STABILITY_PASS_PCT else
                    ("WARN" if pct >= 40 else "FAIL")
                ),
                can_trades=c_trades,
                can_avg_r=round(c_avg_r, 4),
                can_pf=round(c_pf, 3),
                can_max_dd_r=round(c_dd, 2),
                can_win_pct=round(c_win, 1),
            )

            if best is None:
                best = candidate
            elif pct > best["zone_stability_pct"]:
                best = candidate
            elif pct == best["zone_stability_pct"] and c_avg_r > best["can_avg_r"]:
                best = candidate

        if best is not None:
            rows.append(best)

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct", "can_avg_r"], ascending=False
    ).reset_index(drop=True)


# =============================================================================
# REPORT
# =============================================================================

def write_report(grid: pd.DataFrame, sa: pd.DataFrame, n_symbols: int) -> None:
    total_combos = (len(KELTNER_NS) * len(KELTNER_MULTS)
                    * len(STOP_MULTS) * len(FILTER_MODES))
    lines = [
        "PHASE T1 -- KeltnerLong Concept Discovery",
        "=" * 80,
        "",
        "Entry:   Close > EMA(N) + ATR(N)*keltner_mult  (upper band, shifted 1 bar)",
        "Filter:  ema200_price: close > EMA200",
        "         ema50_slope:  EMA50[i] > EMA50[i-1]",
        "Stop:    entry - ATR(N)*stop_mult",
        "Exit:    low<=stop  OR  close<EMA(N)  (midline cross-under)",
        "",
        "Exchange:        Binance Spot (LONG only)",
        f"Timeframe:       {TIMEFRAME}",
        f"Keltner N grid:  {KELTNER_NS}",
        f"Keltner mults:   {KELTNER_MULTS}",
        f"Stop mults:      {STOP_MULTS}",
        f"Filter modes:    {FILTER_MODES}",
        f"Total combos:    {total_combos}",
        f"Symbols:         {n_symbols}",
        f"Backtest bars:   {BACKTEST_BARS}  (~4yr on 1D)",
        "",
        f"Section 4.2 cost floor:  avg_r > {COST_FLOOR_R}R  (Binance Spot)",
        f"Stability zone:  canonical N +/-1 grid step, PASS if >={STABILITY_PASS_PCT:.0f}%",
        "",
        "=" * 80,
        "STABILITY RANKING  (best canonical N per config, sorted by stability)",
        "=" * 80,
    ]

    for _, row in sa.iterrows():
        v    = row["stability_verdict"]
        gate = "CF-PASS" if row["can_avg_r"] > COST_FLOOR_R else "CF-FAIL"
        lines.append(
            f"  [{v:4s}] {row['filter_mode']:15s} km={row['keltner_mult']:.1f}"
            f" sm={row['stop_mult']:.1f}  N={row['canonical_n']:2d}"
            f" zone={row['zone_ns']:12s}"
            f" stab={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}"
            f" ({row['zone_stability_pct']:.0f}%)"
            f"  avg_r={row['can_avg_r']:+.4f}  pf={row['can_pf']:.2f}"
            f"  win%={row['can_win_pct']:.1f}%  {gate}"
        )

    # Best candidate across all configs
    passes = sa[sa["stability_verdict"] == "PASS"]
    warns  = sa[sa["stability_verdict"] == "WARN"]
    best   = (passes.iloc[0] if not passes.empty else
              (warns.iloc[0]  if not warns.empty  else
               (sa.iloc[0]    if not sa.empty     else None)))

    if best is not None:
        gate42  = "PASS" if best["can_avg_r"] > COST_FLOOR_R else "FAIL"
        proceed = (
            "PROCEED TO T2"
            if best["stability_verdict"] in ("PASS", "WARN") and best["can_avg_r"] > COST_FLOOR_R
            else "DO NOT PROCEED"
        )
        lines += [
            "",
            "-" * 80,
            "RECOMMENDED CANONICAL CONFIGURATION (pending your review)",
            "-" * 80,
            f"  Timeframe:    {TIMEFRAME}",
            f"  Filter mode:  {best['filter_mode']}",
            f"  Keltner N:    {best['canonical_n']}",
            f"  Keltner mult: {best['keltner_mult']:.1f}",
            f"  Stop mult:    {best['stop_mult']:.1f}",
            f"  Stability:    zone={best['zone_ns']}  "
            f"{int(best['zone_pf_gt1'])}/{int(best['zone_total'])} profitable"
            f" ({best['zone_stability_pct']:.0f}%  {best['stability_verdict']})",
            f"  Section 4.2 gate (avg_r > {COST_FLOOR_R}R): "
            f"{gate42} ({best['can_avg_r']:.4f}R)",
            f"  Trades (canonical N): {best['can_trades']}",
            f"  Win rate:     {best['can_win_pct']:.1f}%",
            "",
            f"  !! {proceed} -- awaiting human review.",
        ]

    # Per-N sensitivity tables
    lines += ["", "=" * 80, "KELTNER N SENSITIVITY (all configs)", "=" * 80]

    for fm in FILTER_MODES:
        lines += ["", f"--- Filter: {fm} ---"]
        fm_grid = grid[grid["filter_mode"] == fm]
        for km in KELTNER_MULTS:
            for sm in STOP_MULTS:
                g = (fm_grid[(fm_grid["keltner_mult"] == km) &
                             (fm_grid["stop_mult"] == sm)]
                     .sort_values("keltner_n"))
                lines.append(f"\n  km={km:.1f} sm={sm:.1f}")
                for _, row in g.iterrows():
                    n    = int(row["keltner_n"])
                    sign = "[+]" if row["pf"] > 1.0 and row["avg_r"] > 0 else "[-]"
                    cf   = " [CF PASS]" if row["avg_r"] > COST_FLOOR_R else ""
                    lines.append(
                        f"    {sign} N={n:2d}  trades={int(row['trades']):4d}"
                        f"  avg_r={row['avg_r']:+.3f}  pf={row['pf']:.2f}"
                        f"  dd={row['max_dd_r']:+.2f}  win%={row['win_rate_pct']:.1f}%"
                        + cf
                    )

    lines += [
        "",
        "=" * 80,
        "INTERPRETATION",
        "=" * 80,
        "",
        "  PASS (>=67%): Keltner breakout edge is robust -- proceed to T2.",
        "  WARN (40-67%): Edge exists but parameter-sensitive -- review before T2.",
        "  FAIL (<40%):  No consistent breakout edge.  Do NOT proceed.",
        "",
        f"  Cost floor: avg_r > {COST_FLOOR_R}R (Binance Spot).",
        "  Win rate 30-50% typical for breakout trend-following.",
        "  Note: ema50_slope is more permissive than ema200_price.",
        "  Note: higher keltner_mult = fewer signals, higher breakout quality.",
        "",
        "  !! STOP HERE -- do not proceed to T2 until human review.",
    ]

    path = OUT_DIR / "phase_t1_keltnerlong_stability_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 80)
    print("PHASE T1 -- KeltnerLong Concept Discovery")
    print("=" * 80)
    total_combos = (len(KELTNER_NS) * len(KELTNER_MULTS)
                    * len(STOP_MULTS) * len(FILTER_MODES))
    print(f"Timeframe:      {TIMEFRAME}")
    print(f"Keltner N grid: {KELTNER_NS}")
    print(f"Keltner mults:  {KELTNER_MULTS}")
    print(f"Stop mults:     {STOP_MULTS}")
    print(f"Filter modes:   {FILTER_MODES}")
    print(f"Total combos:   {total_combos}")
    print(f"Cost floor:     avg_r > {COST_FLOOR_R}R  (Binance Spot, Section 4.2)")
    print(f"Cache:          {CACHE_DIR}")
    print()

    symbols = discover_symbols()
    print(f"Symbols found: {len(symbols)}")
    if not symbols:
        print(f"[ERROR] No symbols found in cache: {CACHE_DIR}")
        return 1

    print("Loading OHLCV data...")
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is not None:
            data[sym] = df
        else:
            print(f"  SKIP {sym}")
    print(f"Loaded: {len(data)} / {len(symbols)} symbols")

    if not data:
        print("[ERROR] No data loaded.")
        return 1

    # Precompute indicators once per (symbol, N) to avoid redundant computation
    print("\nPrecomputing indicators...")
    ind_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
    for sym, df in data.items():
        for n in KELTNER_NS:
            ind_cache[(sym, n)] = compute_indicators(df, n)
    print(f"  {len(ind_cache)} (symbol, N) pairs ready")

    # Grid sweep
    print(f"\nRunning grid: {total_combos} combos x {len(data)} symbols")
    done          = 0
    all_trades:   List[dict] = []
    summary_rows: List[dict] = []

    for n in KELTNER_NS:
        for km in KELTNER_MULTS:
            for sm in STOP_MULTS:
                for fm in FILTER_MODES:
                    combo_trades: List[dict] = []
                    for sym in data:
                        combo_trades.extend(
                            backtest_symbol(sym, ind_cache[(sym, n)], km, sm, fm, n)
                        )

                    s   = _stats(combo_trades)
                    row = dict(
                        timeframe=TIMEFRAME,
                        filter_mode=fm,
                        keltner_n=n,
                        keltner_mult=km,
                        stop_mult=sm,
                        symbols=len(data),
                    )
                    row.update(s)
                    summary_rows.append(row)
                    all_trades.extend(combo_trades)
                    done += 1

                    cf_tag = " [CF PASS]" if s["avg_r"] > COST_FLOOR_R else ""
                    print(
                        f"  [{done:3d}/{total_combos}] {fm:15s}"
                        f" N={n:2d} km={km:.1f} sm={sm:.1f}"
                        f"  trades={s['trades']:4d}"
                        f"  avg_r={s['avg_r']:+.3f}"
                        f"  pf={s['pf']:.2f}"
                        f"  win%={s['win_rate_pct']:.1f}%"
                        + cf_tag
                    )

    if not summary_rows:
        print("[ERROR] No trades produced.")
        return 1

    grid = pd.DataFrame(summary_rows)
    sa   = stability_analysis(grid)

    pd.DataFrame(all_trades).to_csv(
        OUT_DIR / "phase_t1_keltnerlong_trades.csv", index=False)
    grid.to_csv(OUT_DIR / "phase_t1_keltnerlong_summary.csv", index=False)
    sa.to_csv(OUT_DIR / "phase_t1_keltnerlong_stability.csv", index=False)

    write_report(grid, sa, len(data))

    print()
    print("=" * 80)
    print("STABILITY RANKING SUMMARY  (top 10)")
    print("=" * 80)
    for _, row in sa.head(10).iterrows():
        v    = row["stability_verdict"]
        gate = "CF-PASS" if row["can_avg_r"] > COST_FLOOR_R else "CF-FAIL"
        print(
            f"  [{v:4s}] {row['filter_mode']:15s}"
            f" km={row['keltner_mult']:.1f} sm={row['stop_mult']:.1f}"
            f"  N={row['canonical_n']:2d}"
            f" zone={row['zone_stability_pct']:.0f}%"
            f"  avg_r={row['can_avg_r']:+.4f}R"
            f"  pf={row['can_pf']:.2f}  {gate}"
        )

    print()
    print("=" * 80)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 80)
    print(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
