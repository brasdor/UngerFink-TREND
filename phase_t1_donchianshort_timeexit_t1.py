#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- DonchianShort Time-Exit Variant

Hypothesis: Chandelier trailing stop gets hit by violent bounces in crypto bear
markets, killing otherwise valid short trades. A fixed time exit avoids
stop-hunting — hold for N bars, close regardless.

This is the mirror of the RSI MR long finding: time exit outperformed ATR exit
decisively across all mean-reversion systems. Testing that same logic on the
short-side trend-following entry.

Entry:  Close < Donchian N-period LOW (shifted 1 bar, no lookahead)
Filter: Close < EMA200  AND  ATR(14) >= 50th percentile of trailing 252 bars
Exit:   Fixed time exit after hold_bars bars — NO chandelier, NO trailing stop
Stop:   ATR×mult ABOVE entry (hard safety stop only — never primary exit)

Parameter grid:
  timeframes:  [4h, 1d]
  donchian_n:  [10, 15, 20, 25, 30]
  hold_bars:   [5, 10, 15, 20, 25]
  atr_mult:    [2.0, 3.0]  (safety stop only)

Stability zone: N ±1 step AND hold_bars ±1 step, PASS if >=67% profitable
§4.2 cost floor: 0.25R (Futures)

Primary behavioural gate: 2022 must be POSITIVE
Concentration flag: 2025 > 40% of total R

Universe: symbols_pre2021.csv (bear cycle coverage, same for 4H and 1D)
Data:     data/futures_universe/ohlcv_4h/  and  ohlcv_1d/
Output:   data/research_donchianshort_timeexit_t1/
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
OHLCV_1D  = ROOT / "data" / "futures_universe" / "ohlcv_1d"
CACHE_4H  = ROOT / "data" / "futures_universe" / "ohlcv_4h"
SYM_PRE21 = ROOT / "data" / "futures_universe" / "symbols_pre2021.csv"
OUT_DIR   = ROOT / "data" / "research_donchianshort_timeexit_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES   = ["1d"]   # 4H data not cached locally; 1D only
DONCHIAN_NS  = [10, 15, 20, 25, 30]
HOLD_BARS    = [5, 10, 15, 20, 25]
ATR_MULTS    = [2.0, 3.0]          # safety stop only
ATR_PCT_MIN  = 50                  # fixed from V2 finding
FILTER_MODE  = "ema200_below_atr50"

ATR_N            = 14
EMA_N            = 200
ATR_PCT_WINDOW   = 252
ATR_PCT_MIN_BARS = 100

STABILITY_PASS_PCT = 67.0
COST_FLOOR_R       = 0.25
MIN_BARS           = 300
REPORT_YEARS       = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

EPS = 1e-12


def p(*a, **kw) -> None:
    kw.setdefault("flush", True)
    text = " ".join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), **kw)


# =============================================================================
# UNIVERSE / DATA
# =============================================================================

def load_universe(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]


def _parse_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
        try:
            df.index = pd.to_datetime(ts.astype(float), unit="ms", utc=True)
        except (ValueError, TypeError):
            df.index = pd.to_datetime(ts, utc=True, errors="coerce")
    else:
        return None
    df = df.sort_index()
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].astype(float).dropna()
    return df if len(df) >= MIN_BARS else None


def load_1d(symbol: str) -> Optional[pd.DataFrame]:
    path = OHLCV_1D / f"{symbol}_1d.csv"
    if not path.exists():
        return None
    try:
        return _parse_df(pd.read_csv(path))
    except Exception:
        return None


def load_4h(symbol: str) -> Optional[pd.DataFrame]:
    path = CACHE_4H / f"{symbol}_4h.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return _parse_df(df.reset_index(names=["date"]))
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame, don_n: int) -> Tuple[pd.DataFrame, int]:
    d = df.copy()

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    d["atr_pct_rank"] = (
        d["atr"]
        .rolling(ATR_PCT_WINDOW, min_periods=ATR_PCT_MIN_BARS)
        .apply(lambda x: (x[:-1] <= x[-1]).mean() * 100 if len(x) > 1 else 50.0,
               raw=True)
    )

    # Donchian entry band (shifted 1 bar — no lookahead)
    d["don_low"] = d["low"].shift(1).rolling(don_n).min()
    d["ema200"]  = d["close"].ewm(span=EMA_N, adjust=False).mean()

    warmup = max(don_n, EMA_N, ATR_N, ATR_PCT_WINDOW) + 5
    return d, warmup


# =============================================================================
# BACKTESTER — single symbol, time exit
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    don_n: int,
    hold_bars: int,
    atr_mult: float,
    tf: str,
) -> List[dict]:
    d, warmup = add_indicators(df, don_n)

    pos    = None   # open position state
    trades = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0:
            continue

        close        = float(row["close"])
        high         = float(row["high"])
        t            = d.index[i]
        don_low      = row.get("don_low")
        ema200       = row.get("ema200")
        atr_pct_rank = row.get("atr_pct_rank")

        for v in [don_low, ema200, atr_pct_rank]:
            try:
                if not math.isfinite(float(v)):
                    don_low = None
                    break
            except (TypeError, ValueError):
                don_low = None
                break
        if don_low is None:
            continue

        # --- Manage open position ---
        if pos is not None:
            bars_held = i - pos["entry_bar"]

            # Safety stop (hard stop — checked on HIGH)
            if high >= pos["safety_stop"]:
                net_r = (pos["entry"] - pos["safety_stop"]) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, pos["safety_stop"], net_r,
                    "safety_stop", don_n, hold_bars, atr_mult, tf,
                ))
                pos = None
                continue

            # Time exit — close after hold_bars
            if bars_held >= hold_bars:
                net_r = (pos["entry"] - close) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r,
                    "time_exit", don_n, hold_bars, atr_mult, tf,
                ))
                pos = None
                continue

        # --- New entry ---
        if pos is None:
            bear_regime  = close < float(ema200)
            atr_active   = float(atr_pct_rank) >= ATR_PCT_MIN
            short_signal = close < float(don_low)

            if bear_regime and atr_active and short_signal:
                risk = atr * atr_mult
                if risk > EPS:
                    pos = {
                        "entry_bar":   i,
                        "entry_time":  t,
                        "entry":       close,
                        "safety_stop": close + risk,
                        "risk":        risk,
                    }

    return trades


def _make_trade(
    symbol: str, pos: dict, exit_time, exit_price: float,
    net_r: float, reason: str,
    don_n: int, hold_bars: int, atr_mult: float, tf: str,
) -> dict:
    return {
        "symbol":       symbol,
        "side":         "SHORT",
        "timeframe":    tf,
        "filter_mode":  FILTER_MODE,
        "don_n":        don_n,
        "hold_bars":    hold_bars,
        "atr_mult":     atr_mult,
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry"],
        "exit_price":   float(exit_price),
        "safety_stop":  pos["safety_stop"],
        "initial_risk": pos["risk"],
        "net_r":        float(net_r),
        "exit_reason":  reason,
        "entry_year":   pos["entry_time"].year,
    }


# =============================================================================
# STATISTICS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    return float(g / l) if l > EPS else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(trades: List[dict]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0)
    r = np.array([t["net_r"] for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0)
    return dict(
        trades=int(len(r)),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        pf=_pf(r),
        max_dd_r=_dd(r),
        win_rate_pct=float((r > 0).mean() * 100),
    )


# =============================================================================
# YEAR-BY-YEAR
# =============================================================================

def year_by_year(trades: List[dict]) -> Dict[int, dict]:
    by_year: Dict[int, List[dict]] = {}
    for t in trades:
        yr = int(t.get("entry_year", 0))
        by_year.setdefault(yr, []).append(t)
    return {yr: _stats(by_year.get(yr, [])) for yr in REPORT_YEARS}


def _year_flags(yby: Dict[int, dict], total_r: float) -> List[str]:
    flags = []
    for yr in REPORT_YEARS:
        yr_r = yby[yr]["total_r"]
        # Flag 2025 concentration at 40% (lower threshold than generic 50%)
        threshold = 0.40 if yr == 2025 else 0.50
        if abs(total_r) > EPS and abs(yr_r) > threshold * abs(total_r):
            flags.append(
                f"[CONCENTRATED]  {yr} = {yr_r:+.1f}R = "
                f"{yr_r / total_r * 100:.0f}% of total R"
                + (" <-- 2025 RECENCY" if yr == 2025 else "")
            )
    # 2022 check
    y2022 = yby.get(2022, {})
    if y2022.get("trades", 0) > 0:
        if y2022["total_r"] < 0:
            flags.append(f"[!2022 NEGATIVE]  2022 = {y2022['total_r']:+.1f}R "
                         f"(short system must profit in 2022)")
    else:
        flags.append("[!2022 NO TRADES]")
    return flags


# =============================================================================
# STABILITY — 2D zone: N ±1 step AND hold_bars ±1 step
# =============================================================================

def _adjacent(val: int, grid: List[int]) -> List[int]:
    """Return val and its immediate neighbours in the grid."""
    idx = grid.index(val) if val in grid else -1
    if idx < 0:
        return [val]
    result = [grid[idx]]
    if idx > 0:
        result.append(grid[idx - 1])
    if idx < len(grid) - 1:
        result.append(grid[idx + 1])
    return result


def stability_analysis(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (tf, atr_mult, don_n, hold_bars) candidate:
      zone = adjacent_N × adjacent_hold_bars  (2D, ±1 step each dimension)
      PASS if >=67% of zone combos profitable (avg_r > 0)
    Returns the best stable candidate per (tf, atr_mult).
    """
    rows = []
    for (tf, am), g in grid_df.groupby(["timeframe", "atr_mult"]):
        # Build lookup: (n, hb) -> avg_r
        lookup: Dict[Tuple[int, int], float] = {}
        for _, row in g.iterrows():
            lookup[(int(row["don_n"]), int(row["hold_bars"]))] = float(row["avg_r"])

        for _, row in g.iterrows():
            n  = int(row["don_n"])
            hb = int(row["hold_bars"])
            s  = _stats([])  # dummy

            zone_ns  = _adjacent(n,  DONCHIAN_NS)
            zone_hbs = _adjacent(hb, HOLD_BARS)

            zone_combos = [(zn, zh) for zn in zone_ns for zh in zone_hbs]
            zone_pass   = [1 for zn, zh in zone_combos
                           if lookup.get((zn, zh), -999) > 0]
            pct = 100.0 * len(zone_pass) / max(len(zone_combos), 1)
            verdict = ("PASS" if pct >= STABILITY_PASS_PCT
                       else ("WARN" if pct >= 40 else "FAIL"))
            cf = "PASS" if float(row["avg_r"]) > COST_FLOOR_R else "FAIL"

            rows.append(dict(
                timeframe=tf,
                atr_mult=am,
                don_n=n,
                hold_bars=hb,
                trades=int(row["trades"]),
                total_r=float(row["total_r"]),
                avg_r=float(row["avg_r"]),
                pf=float(row["pf"]),
                max_dd_r=float(row["max_dd_r"]),
                win_rate_pct=float(row["win_rate_pct"]),
                zone_size=len(zone_combos),
                zone_pass=len(zone_pass),
                zone_pct=round(pct, 1),
                stability_verdict=verdict,
                sec42=cf,
            ))

    return pd.DataFrame(rows).sort_values(
        ["timeframe", "atr_mult", "zone_pct", "avg_r"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    grid_df: pd.DataFrame,
    sa: pd.DataFrame,
    all_trades_df: pd.DataFrame,
    sym_count: Dict[str, int],
) -> None:
    lines = [
        "PHASE T1 -- DonchianShort Time-Exit Variant",
        "=" * 80,
        "",
        "Hypothesis: Chandelier trailing stop gets hit by violent bounces in",
        "            crypto bear markets. Time exit avoids stop-hunting.",
        "            Mirror of RSI MR finding: time exit dominates MR systems.",
        "",
        "Entry:  Close < Donchian N-period LOW (shifted 1 bar, no lookahead)",
        "Filter: Close < EMA200  AND  ATR(14) >= 50th percentile (252-bar window)",
        "Exit:   Fixed time exit after hold_bars  [NO chandelier, NO trailing stop]",
        "Stop:   ATR×mult ABOVE entry  (hard safety stop — NOT primary exit)",
        "",
        f"Timeframes:     {TIMEFRAMES}",
        f"Donchian N:     {DONCHIAN_NS}",
        f"Hold bars:      {HOLD_BARS}",
        f"ATR mults:      {ATR_MULTS}  (safety stop only)",
        f"ATR pct gate:   >= {ATR_PCT_MIN}th percentile  (fixed from V2)",
        f"Cost floor:     avg_r > {COST_FLOOR_R}R  (Futures §4.2)",
        "Stability zone: N ±1 step AND hold_bars ±1 step, PASS if >=67%",
        "",
        "PRIMARY BEHAVIOURAL GATE: 2022 must be POSITIVE",
        "CONCENTRATION FLAG:       2025 > 40% of total R = recency bias",
        "",
    ]

    for tf in TIMEFRAMES:
        tf_sa   = sa[sa["timeframe"] == tf]
        n_syms  = sym_count.get(tf, 0)

        # Top combos by avg_r among PASS
        pass_sa  = tf_sa[tf_sa["stability_verdict"] == "PASS"]
        warn_sa  = tf_sa[tf_sa["stability_verdict"] == "WARN"]
        top_pass = pass_sa.sort_values("avg_r", ascending=False).head(10)
        top_warn = warn_sa.sort_values("avg_r", ascending=False).head(5)

        lines += [
            "=" * 80,
            f"TIMEFRAME: {tf}  (symbols: {n_syms})",
            "=" * 80,
            f"PASS combos (stability >=67%): {len(pass_sa)}  |  "
            f"WARN: {len(warn_sa)}  |  "
            f"FAIL: {len(tf_sa) - len(pass_sa) - len(warn_sa)}",
            "",
        ]

        if not top_pass.empty:
            lines += ["TOP PASS COMBOS (sorted by avg_r)", "-" * 60]
            for _, row in top_pass.iterrows():
                lines.append(
                    f"  [PASS] N={int(row['don_n']):2d}  hb={int(row['hold_bars']):2d}  "
                    f"ATR*{row['atr_mult']:.1f}  "
                    f"zone={int(row['zone_pass'])}/{int(row['zone_size'])}  "
                    f"({row['zone_pct']:.0f}%)  "
                    f"trades={int(row['trades'])}  "
                    f"avg_r={row['avg_r']:+.4f}  "
                    f"pf={row['pf']:.2f}  "
                    f"win%={row['win_rate_pct']:.1f}%  "
                    f"§4.2={row['sec42']}"
                )
            lines.append("")

        if not top_warn.empty:
            lines += ["TOP WARN COMBOS", "-" * 60]
            for _, row in top_warn.iterrows():
                lines.append(
                    f"  [WARN] N={int(row['don_n']):2d}  hb={int(row['hold_bars']):2d}  "
                    f"ATR*{row['atr_mult']:.1f}  "
                    f"zone={int(row['zone_pass'])}/{int(row['zone_size'])}  "
                    f"({row['zone_pct']:.0f}%)  "
                    f"trades={int(row['trades'])}  "
                    f"avg_r={row['avg_r']:+.4f}  "
                    f"pf={row['pf']:.2f}  "
                    f"§4.2={row['sec42']}"
                )
            lines.append("")

        # N × hold_bars heatmap of avg_r for best ATR mult
        for am in ATR_MULTS:
            g = grid_df[(grid_df["timeframe"] == tf) &
                        (grid_df["atr_mult"]  == am)]
            if g.empty:
                continue
            lines += [
                f"avg_r HEATMAP  ({tf}  ATR*{am:.1f})",
                "-" * 55,
                f"  {'N\\hb':>5}  " + "  ".join(f"hb={hb:2d}" for hb in HOLD_BARS),
            ]
            for n in DONCHIAN_NS:
                row_vals = []
                for hb in HOLD_BARS:
                    sl = g[(g["don_n"] == n) & (g["hold_bars"] == hb)]
                    if sl.empty:
                        row_vals.append("  ----")
                    else:
                        v = float(sl.iloc[0]["avg_r"])
                        sign = "+" if v >= 0 else ""
                        cf = "*" if v > COST_FLOOR_R else " "
                        row_vals.append(f"{cf}{sign}{v:.3f}")
                lines.append(f"  N={n:2d}:  " + "  ".join(row_vals))
            lines += ["  * = passes §4.2 cost floor", ""]

    # --- 2022 check on all PASS combos with §4.2 PASS ---
    cf_pass = sa[(sa["stability_verdict"] == "PASS") & (sa["sec42"] == "PASS")]
    lines += [
        "=" * 80,
        "2022 BEAR MARKET CHECK  (PASS + §4.2 combos only)",
        "=" * 80,
    ]
    if cf_pass.empty:
        lines.append("  No combos pass both stability AND §4.2 cost floor.")
        lines.append(f"  Best §4.2 gap: need 0.25R, best found = "
                     f"{sa['avg_r'].max():+.4f}R")
    else:
        t2022 = all_trades_df[all_trades_df["entry_year"] == 2022]
        for _, row in cf_pass.sort_values("avg_r", ascending=False).iterrows():
            tf  = row["timeframe"]
            am  = row["atr_mult"]
            n   = int(row["don_n"])
            hb  = int(row["hold_bars"])
            sl  = t2022[
                (t2022["timeframe"] == tf) &
                (t2022["atr_mult"]  == am) &
                (t2022["don_n"]     == n) &
                (t2022["hold_bars"] == hb)
            ]
            s = _stats(sl.to_dict("records"))
            status = "OK" if s["total_r"] > 0 else "FAIL -- 2022 NEGATIVE"
            lines.append(
                f"  {tf}  N={n:2d}  hb={hb:2d}  ATR*{am:.1f}  "
                f"2022: trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
                f"total_r={s['total_r']:+.1f}R  [{status}]"
            )

    # --- Overall best ---
    all_pass = sa[sa["stability_verdict"] == "PASS"].sort_values("avg_r", ascending=False)
    all_cf   = sa[sa["sec42"] == "PASS"]
    best     = all_pass.iloc[0] if not all_pass.empty else None

    lines += ["", "=" * 80, "OVERALL BEST CANDIDATE", "=" * 80]
    if best is not None:
        proceed = (
            "PROCEED TO T2"
            if best["stability_verdict"] in ("PASS", "WARN")
            and best["sec42"] == "PASS"
            else "DO NOT PROCEED"
        )
        lines += [
            f"  Timeframe:    {best['timeframe']}",
            f"  N:            {int(best['don_n'])}",
            f"  hold_bars:    {int(best['hold_bars'])}",
            f"  ATR mult:     {best['atr_mult']:.1f}  (safety stop)",
            f"  Stability:    {best['zone_pct']:.0f}%  ({best['stability_verdict']})",
            f"  §4.2:         {best['sec42']}  ({best['avg_r']:+.4f}R vs >{COST_FLOOR_R}R)",
            f"  Trades:       {int(best['trades'])}",
            f"  PF:           {best['pf']:.3f}",
            f"  Win rate:     {best['win_rate_pct']:.1f}%",
            "",
            f"  !! {proceed} — awaiting human review.",
        ]
        if all_cf.empty:
            lines += [
                "",
                "  NOTE: No combo passes §4.2 cost floor (0.25R).",
                f"  Closest: avg_r = {sa['avg_r'].max():+.4f}R",
                f"  §4.2 gap = {0.25 - sa['avg_r'].max():.4f}R",
            ]
    else:
        lines.append("  No candidates found.")

    lines += [
        "",
        "=" * 80,
        "DIAGNOSTIC",
        "=" * 80,
        "",
        "  If no combo passes both §4.2 AND 2022 positive:",
        "  -> Time exit also fails to produce sufficient edge",
        "  -> Root cause: insufficient bear data in 2019-2026 dataset",
        "  -> Donchian Short permanently closed",
        "  -> Short exposure covered by Momentum Factor short basket",
        "",
        "  If §4.2 passes but 2022 is negative:",
        "  -> Edge exists but concentrated in 2024-2025, not structural",
        "  -> HALT — recency bias, not a repeatable short-side system",
        "",
        "  !! STOP HERE — do not proceed to T2 until human review.",
    ]

    out = OUT_DIR / "phase_t1_donchianshort_timeexit_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"\nMain report: {out}")


def write_year_by_year(
    sa: pd.DataFrame, all_trades_df: pd.DataFrame
) -> None:
    # Show year-by-year for all PASS combos (regardless of §4.2)
    pass_sa = sa[sa["stability_verdict"] == "PASS"].sort_values(
        "avg_r", ascending=False
    )
    if pass_sa.empty:
        p("No PASS combos — skipping year-by-year.")
        return

    lines = [
        "YEAR-BY-YEAR ANALYSIS — DonchianShort Time-Exit",
        "=" * 80,
        "Showing all PASS combos (sorted by avg_r).",
        "PRIMARY GATE: 2022 must be POSITIVE.",
        "CONCENTRATION FLAG: 2025 > 40% of total R.",
        "",
    ]

    for _, row in pass_sa.head(20).iterrows():
        tf  = row["timeframe"]
        am  = row["atr_mult"]
        n   = int(row["don_n"])
        hb  = int(row["hold_bars"])

        sl = all_trades_df[
            (all_trades_df["timeframe"] == tf) &
            (all_trades_df["atr_mult"]  == am) &
            (all_trades_df["don_n"]     == n) &
            (all_trades_df["hold_bars"] == hb)
        ]
        if sl.empty:
            continue

        trade_list = sl.to_dict("records")
        s   = _stats(trade_list)
        yby = year_by_year(trade_list)
        flags = _year_flags(yby, s["total_r"])

        y2022_ok = yby.get(2022, {}).get("total_r", -1) > 0
        gate_tag = " [2022 OK]" if y2022_ok else " [2022 FAIL]"
        cf_tag   = " [cf PASS]" if s["avg_r"] > COST_FLOOR_R else ""

        lines += [
            f"{'=' * 65}",
            f"  [{row['stability_verdict']}]  {tf}  N={n}  hb={hb}  ATR*{am:.1f}"
            + gate_tag + cf_tag,
            f"  zone={int(row['zone_pass'])}/{int(row['zone_size'])}  "
            f"({row['zone_pct']:.0f}%)  "
            f"trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
            f"pf={s['pf']:.2f}  total_r={s['total_r']:+.1f}R",
            f"{'=' * 65}",
            "",
            f"  {'Year':>4}  {'Trades':>6}  {'Win%':>6}  "
            f"{'TotalR':>8}  {'AvgR':>7}  {'PF':>5}  {'StopHit%':>8}",
            f"  {'-' * 60}",
        ]

        for yr in REPORT_YEARS:
            ys = yby[yr]
            if ys["trades"] == 0:
                lines.append(f"  {yr:>4}  {'--':>6}  {'--':>6}  "
                              f"{'--':>8}  {'--':>7}  {'--':>5}")
                continue
            pct_of_total = ys["total_r"] / max(abs(s["total_r"]), EPS) * 100
            conc = ""
            if yr == 2025 and abs(pct_of_total) > 40:
                conc = " <-- 2025 RECENCY"
            elif abs(pct_of_total) > 50:
                conc = " <-- CONCENTRATED"
            neg_flag = " <-- 2022 NEGATIVE" if yr == 2022 and ys["total_r"] < 0 else ""

            # Stop hit rate for this year
            yr_trades = [t for t in trade_list if t.get("entry_year") == yr]
            stop_pct = (
                100.0 * sum(1 for t in yr_trades if t["exit_reason"] == "safety_stop")
                / max(len(yr_trades), 1)
            )
            lines.append(
                f"  {yr:>4}  {ys['trades']:>6}  {ys['win_rate_pct']:>5.1f}%  "
                f"  {ys['total_r']:>+7.1f}R  {ys['avg_r']:>+6.3f}  "
                f"{ys['pf']:>5.2f}  {stop_pct:>7.1f}%"
                + conc + neg_flag
            )

        if flags:
            lines += ["", "  ** FLAGS:"]
            for fl in flags:
                lines.append(f"     {fl}")
        lines.append("")

    out = OUT_DIR / "phase_t1_donchianshort_timeexit_year_by_year.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"Year-by-year report: {out}")


# =============================================================================
# UNIVERSE BUILDER
# =============================================================================

def _build_pre2021_universe() -> List[str]:
    """Scan ohlcv_1d/ and return symbols whose earliest bar is before 2021-01-01."""
    cutoff = pd.Timestamp("2021-01-01", tz="UTC")
    syms = []
    for path in sorted(OHLCV_1D.glob("*_1d.csv")):
        try:
            df = pd.read_csv(path, usecols=[0], nrows=5, header=0)
            col = df.columns[0]
            raw = pd.read_csv(path, usecols=[col], nrows=1).iloc[0, 0]
            dt = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(dt):
                # try reading first date column generically
                full = pd.read_csv(path, nrows=2)
                full.columns = [c.lower() for c in full.columns]
                date_col = next((c for c in ["date", "timestamp"] if c in full.columns), None)
                if date_col is None:
                    continue
                dt = pd.to_datetime(full[date_col].iloc[0], utc=True, errors="coerce")
            if pd.notna(dt) and dt < cutoff:
                sym = path.stem.replace("_1d", "").upper()
                syms.append(sym)
        except Exception:
            continue
    return syms


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    p("=" * 80)
    p("PHASE T1 -- DonchianShort Time-Exit Variant")
    p("=" * 80)
    p(f"Timeframes:   {TIMEFRAMES}")
    p(f"Donchian N:   {DONCHIAN_NS}")
    p(f"Hold bars:    {HOLD_BARS}")
    p(f"ATR mults:    {ATR_MULTS}  (safety stop only)")
    p(f"ATR pct gate: >= {ATR_PCT_MIN}th percentile  (fixed from V2)")
    p(f"Cost floor:   avg_r > {COST_FLOOR_R}R  (Futures §4.2)")
    p(f"Output:       {OUT_DIR}")
    p()

    pre21_syms = load_universe(SYM_PRE21)
    if not pre21_syms:
        p("[INFO] symbols_pre2021.csv not found — building from ohlcv_1d/ ...")
        pre21_syms = _build_pre2021_universe()
        if not pre21_syms:
            p("[ERROR] No 1D files found in ohlcv_1d/ — cannot build universe.")
            return 1
        pd.DataFrame({"symbol": pre21_syms}).to_csv(SYM_PRE21, index=False)
        p(f"[INFO] Saved {len(pre21_syms)} symbols to {SYM_PRE21}")

    p(f"Universe: {len(pre21_syms)} pre-2021 symbols")
    p()

    all_trades:   List[dict] = []
    summary_rows: List[dict] = []
    sym_count:    Dict[str, int] = {}

    total_combos = len(DONCHIAN_NS) * len(HOLD_BARS) * len(ATR_MULTS)

    for tf in TIMEFRAMES:
        p(f"{'=' * 60}")
        p(f"TIMEFRAME: {tf}")
        p(f"{'=' * 60}")

        p(f"Loading {len(pre21_syms)} symbols...")
        data: Dict[str, pd.DataFrame] = {}
        for sym in pre21_syms:
            df = load_1d(sym) if tf == "1d" else load_4h(sym)
            if df is not None and len(df) >= MIN_BARS:
                data[sym] = df

        n_loaded = len(data)
        sym_count[tf] = n_loaded
        p(f"Loaded: {n_loaded} / {len(pre21_syms)}")

        if not data:
            p(f"  [WARN] No data for {tf} — skipping.")
            continue

        p(f"Running {total_combos} combos x {n_loaded} symbols...")
        done = 0

        for don_n in DONCHIAN_NS:
            for hb in HOLD_BARS:
                for am in ATR_MULTS:
                    combo_trades: List[dict] = []
                    for sym, df in data.items():
                        combo_trades.extend(
                            backtest_symbol(sym, df, don_n, hb, am, tf)
                        )

                    s = _stats(combo_trades)
                    row = dict(
                        timeframe=tf,
                        don_n=don_n,
                        hold_bars=hb,
                        atr_mult=am,
                        symbols=n_loaded,
                    )
                    row.update(s)
                    summary_rows.append(row)
                    all_trades.extend(combo_trades)
                    done += 1

                    cf_tag = " [cf PASS]" if s["avg_r"] > COST_FLOOR_R else ""
                    p(
                        f"  [{done:3d}/{total_combos}] N={don_n:2d}  hb={hb:2d}  "
                        f"ATR*{am:.1f}  "
                        f"trades={s['trades']:4d}  avg_r={s['avg_r']:+.4f}  "
                        f"pf={s['pf']:.2f}  win%={s['win_rate_pct']:.1f}%"
                        + cf_tag
                    )

    if not summary_rows:
        p("[ERROR] No trades produced.")
        return 1

    grid_df = pd.DataFrame(summary_rows)
    sa      = stability_analysis(grid_df)

    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv(
        OUT_DIR / "phase_t1_donchianshort_timeexit_trades.csv", index=False)
    grid_df.to_csv(
        OUT_DIR / "phase_t1_donchianshort_timeexit_summary.csv", index=False)
    sa.to_csv(
        OUT_DIR / "phase_t1_donchianshort_timeexit_stability.csv", index=False)

    write_report(grid_df, sa, all_trades_df, sym_count)
    write_year_by_year(sa, all_trades_df)

    # --- Console summary ---
    p()
    p("=" * 80)
    p("STABILITY RANKING SUMMARY  (top 5 PASS per timeframe by avg_r)")
    p("=" * 80)
    for tf in TIMEFRAMES:
        tf_sa    = sa[(sa["timeframe"] == tf) &
                      (sa["stability_verdict"] == "PASS")].sort_values(
                          "avg_r", ascending=False)
        p(f"  [{tf}]  PASS: {len(tf_sa)}")
        for _, row in tf_sa.head(5).iterrows():
            p(
                f"    N={int(row['don_n']):2d}  hb={int(row['hold_bars']):2d}  "
                f"ATR*{row['atr_mult']:.1f}  "
                f"zone={int(row['zone_pass'])}/{int(row['zone_size'])}  "
                f"({row['zone_pct']:.0f}%)  "
                f"avg_r={row['avg_r']:+.4f}  pf={row['pf']:.2f}  "
                f"§4.2={row['sec42']}"
            )

    # §4.2 passing combos
    cf_pass = sa[(sa["stability_verdict"] == "PASS") & (sa["sec42"] == "PASS")]
    p()
    p(f"Combos passing BOTH stability AND §4.2: {len(cf_pass)}")

    # 2022 check on best combos per TF
    p()
    p("2022 check (top PASS combo per TF, N=20 or closest):")
    for tf in TIMEFRAMES:
        tf_sa = sa[(sa["timeframe"] == tf) &
                   (sa["stability_verdict"] == "PASS")].sort_values(
                       "avg_r", ascending=False)
        if tf_sa.empty:
            p(f"  {tf}: no PASS combos")
            continue
        best = tf_sa.iloc[0]
        n   = int(best["don_n"])
        hb  = int(best["hold_bars"])
        am  = best["atr_mult"]
        sl  = all_trades_df[
            (all_trades_df["timeframe"] == tf) &
            (all_trades_df["atr_mult"]  == am) &
            (all_trades_df["don_n"]     == n) &
            (all_trades_df["hold_bars"] == hb) &
            (all_trades_df["entry_year"] == 2022)
        ]
        s = _stats(sl.to_dict("records"))
        ok = "OK" if s["total_r"] > 0 else "FAIL"
        p(f"  [{ok}] {tf}  N={n}  hb={hb}  ATR*{am:.1f}  "
          f"2022: trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
          f"total_r={s['total_r']:+.1f}R")

        # Full year-by-year for the best combo
        sl_all = all_trades_df[
            (all_trades_df["timeframe"] == tf) &
            (all_trades_df["atr_mult"]  == am) &
            (all_trades_df["don_n"]     == n) &
            (all_trades_df["hold_bars"] == hb)
        ]
        yby = year_by_year(sl_all.to_dict("records"))
        s_all = _stats(sl_all.to_dict("records"))
        p(f"    Total: {s_all['total_r']:+.1f}R  avg_r={s_all['avg_r']:+.4f}")
        for yr in REPORT_YEARS:
            ys = yby[yr]
            if ys["trades"] > 0:
                p(f"    {yr}: {ys['trades']:3d} trades  "
                  f"avg_r={ys['avg_r']:+.4f}  total_r={ys['total_r']:+.1f}R")

    p()
    p("=" * 80)
    p("T1 COMPLETE -- awaiting human review before T2")
    p("=" * 80)
    p(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
