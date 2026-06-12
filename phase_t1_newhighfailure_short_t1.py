#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- New High Failure Short

Hypothesis: Short FAILED breakouts, not breakdowns.
When price makes a new N-day high BUT the bar closes below its open
(bearish candle despite new high), selling pressure at the top confirms
a bull trap. Enter short at close of that bar.

Structurally different from all previous short attempts:
  Donchian Short:   enter when price BREAKS DOWN (stops hit by violent bounces)
  New High Failure: enter when breakout FAILS (candle confirms selling pressure at highs)

Entry:
  1. high > max(high, previous lookback_n bars)  -- new N-day high
  2. close < open                                 -- bearish candle despite new high
  3. close < EMA200                              -- bear regime (mandatory)
  4. ATR percentile >= 40 (optional)             -- active market condition
  Enter SHORT at close of that bar.

Exit:   Fixed time exit after hold_bars bars
Stop:   entry_bar_high + ATR * atr_mult  (hard safety stop only)
Risk:   safety_stop - entry_price  (per-trade R denominator)

NOTE: 4H data is not available locally. Script runs 1D only.

Parameter grid:
  lookback_n:  [5, 10, 15, 20, 30]   new high lookback period
  hold_bars:   [5, 10, 15, 20]       time exit
  atr_mult:    [2.0, 3.0]            safety stop
  atr_filter:  ["none", "pct40"]     optional ATR percentile >= 40

Stability zone: lookback_n +/-1 step AND hold_bars +/-1 step, PASS if >=67%
§4.2 cost floor: 0.25R (Futures)

PRIMARY GATE:       2022 must be POSITIVE
CONCENTRATION FLAG: 2025 > 40% of total R

Universe: data/futures_universe/ohlcv_1d/ (all 290 symbols)
Output:   data/research_newhighfailure_short_t1/
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

ROOT     = Path(__file__).resolve().parent
OHLCV_1D = ROOT / "data" / "futures_universe" / "ohlcv_1d"
OUT_DIR  = ROOT / "data" / "research_newhighfailure_short_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_NS  = [5, 10, 15, 20, 30]
HOLD_BARS    = [5, 10, 15, 20]
ATR_MULTS    = [2.0, 3.0]
ATR_FILTERS  = ["none", "pct40"]

ATR_N            = 14
EMA_N            = 200
ATR_PCT_WINDOW   = 252
ATR_PCT_MIN_BARS = 100
ATR_FILTER_VAL   = 40.0

STABILITY_PASS_PCT = 67.0
COST_FLOOR_R       = 0.25
MIN_BARS           = 300

REPORT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
EPS          = 1e-12


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

def load_all_symbols() -> List[str]:
    return sorted(
        f.stem.replace("_1d", "").upper()
        for f in OHLCV_1D.glob("*_1d.csv")
    )


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


# =============================================================================
# PER-SYMBOL PRECOMPUTATION  (computed ONCE per symbol, not once per combo)
# =============================================================================

def precompute_symbol(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute all indicators that are independent of the combo parameters.
    ATR, ATR percentile, and EMA200 do NOT depend on lookback_n or hold_bars.
    Precomputing them once per symbol avoids 80x redundant rolling windows.
    """
    d = df.copy()

    # ATR(14)
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # ATR percentile rank -- vectorized via rank (avoids slow rolling lambda)
    # For each row i: fraction of previous ATR_PCT_WINDOW values <= atr[i]
    atr_arr   = d["atr"].values.astype(float)
    n         = len(atr_arr)
    pct_rank  = np.full(n, np.nan)
    for i in range(ATR_PCT_MIN_BARS, n):
        start     = max(0, i - ATR_PCT_WINDOW)
        window    = atr_arr[start:i]          # exclude current bar
        if len(window) < 2:
            continue
        pct_rank[i] = float((window <= atr_arr[i]).mean() * 100)
    d["atr_pct_rank"] = pct_rank

    # EMA200
    d["ema200"] = d["close"].ewm(span=EMA_N, adjust=False).mean()

    # Pre-compute bearish candle flag and bear regime flag (combo-independent)
    d["is_bearish"]     = (d["close"] < d["open"]).astype(bool)
    d["is_bear_regime"] = (d["close"] < d["ema200"]).astype(bool)

    # Warmup: rows before which indicators are not valid
    warmup = max(EMA_N, ATR_N, ATR_PCT_MIN_BARS) + 5
    return d, warmup


# =============================================================================
# BACKTESTER  (lookback_n and combo parameters only)
# =============================================================================

def backtest_symbol(
    symbol: str,
    d: pd.DataFrame,           # precomputed dataframe
    warmup: int,
    lookback_n: int,
    hold_bars: int,
    atr_mult: float,
    atr_filter: str,
) -> List[dict]:
    """
    Run one (lookback_n, hold_bars, atr_mult, atr_filter) combo on a precomputed df.
    Only the rolling max of previous highs varies per lookback_n -- computed here.
    """
    # Rolling max of previous lookback_n highs (no lookahead: shift(1))
    prev_high_max = d["high"].shift(1).rolling(lookback_n).max().values

    # Extract numpy arrays for speed
    high_arr         = d["high"].values.astype(float)
    low_arr          = d["low"].values.astype(float)
    close_arr        = d["close"].values.astype(float)
    atr_arr          = d["atr"].values.astype(float)
    pct_arr          = d["atr_pct_rank"].values.astype(float)
    is_bearish_arr   = d["is_bearish"].values
    is_bear_arr      = d["is_bear_regime"].values
    timestamps       = d.index

    pos    = None
    trades = []

    for i in range(warmup, len(d)):
        atr = atr_arr[i]
        if not math.isfinite(atr) or atr <= 0:
            continue

        close    = close_arr[i]
        high     = high_arr[i]
        phmax    = prev_high_max[i]
        pct_rank = pct_arr[i]
        t        = timestamps[i]

        if not math.isfinite(phmax):
            continue

        # --- Manage open position ---
        if pos is not None:
            bars_held = i - pos["entry_bar"]

            # Safety stop checked on this bar's high
            if high >= pos["safety_stop"]:
                net_r = (pos["entry_price"] - pos["safety_stop"]) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, pos["safety_stop"], net_r, "safety_stop",
                    lookback_n, hold_bars, atr_mult, atr_filter,
                ))
                pos = None
                continue

            # Time exit
            if bars_held >= hold_bars:
                net_r = (pos["entry_price"] - close) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "time_exit",
                    lookback_n, hold_bars, atr_mult, atr_filter,
                ))
                pos = None
                continue

        # --- New entry ---
        if pos is None:
            is_new_high  = high > phmax
            is_bearish   = bool(is_bearish_arr[i])
            is_bear      = bool(is_bear_arr[i])
            atr_ok       = (atr_filter == "none") or (math.isfinite(pct_rank) and pct_rank >= ATR_FILTER_VAL)

            if is_new_high and is_bearish and is_bear and atr_ok:
                safety_stop = high + atr * atr_mult
                risk        = safety_stop - close
                if risk > EPS:
                    pos = {
                        "entry_bar":   i,
                        "entry_time":  t,
                        "entry_price": close,
                        "entry_high":  high,
                        "safety_stop": safety_stop,
                        "risk":        risk,
                    }

    return trades


def _make_trade(
    symbol: str,
    pos: dict,
    exit_time,
    exit_price: float,
    net_r: float,
    reason: str,
    lookback_n: int,
    hold_bars: int,
    atr_mult: float,
    atr_filter: str,
) -> dict:
    return {
        "symbol":       symbol,
        "side":         "SHORT",
        "timeframe":    "1d",
        "lookback_n":   lookback_n,
        "hold_bars":    hold_bars,
        "atr_mult":     atr_mult,
        "atr_filter":   atr_filter,
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry_price"],
        "entry_high":   pos["entry_high"],
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
    g =  r[r > 0].sum()
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
        trades       = int(len(r)),
        total_r      = float(r.sum()),
        avg_r        = float(r.mean()),
        pf           = _pf(r),
        max_dd_r     = _dd(r),
        win_rate_pct = float((r > 0).mean() * 100),
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
        threshold = 0.40 if yr == 2025 else 0.50
        if abs(total_r) > EPS and abs(yr_r) > threshold * abs(total_r):
            flags.append(
                f"[CONCENTRATED]  {yr} = {yr_r:+.1f}R = "
                f"{yr_r / total_r * 100:.0f}% of total R"
                + (" <-- 2025 RECENCY" if yr == 2025 else "")
            )
    y2022 = yby.get(2022, {})
    if y2022.get("trades", 0) > 0:
        if y2022["total_r"] < 0:
            flags.append(
                f"[!2022 NEGATIVE]  2022 = {y2022['total_r']:+.1f}R "
                f"-- short system MUST profit in bear year"
            )
    else:
        flags.append("[!2022 NO TRADES]  (only 58 of 290 symbols have pre-2021 data)")
    return flags


# =============================================================================
# STABILITY
# =============================================================================

def _adjacent(val: int, grid: List[int]) -> List[int]:
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
    rows = []
    for (am, af), g in grid_df.groupby(["atr_mult", "atr_filter"]):
        lookup: Dict[Tuple[int, int], float] = {}
        for _, row in g.iterrows():
            lookup[(int(row["lookback_n"]), int(row["hold_bars"]))] = float(row["avg_r"])

        for _, row in g.iterrows():
            n  = int(row["lookback_n"])
            hb = int(row["hold_bars"])

            zone_ns     = _adjacent(n,  LOOKBACK_NS)
            zone_hbs    = _adjacent(hb, HOLD_BARS)
            zone_combos = [(zn, zh) for zn in zone_ns for zh in zone_hbs]
            zone_pass   = [1 for zn, zh in zone_combos
                           if lookup.get((zn, zh), -999) > 0]
            pct     = 100.0 * len(zone_pass) / max(len(zone_combos), 1)
            verdict = ("PASS" if pct >= STABILITY_PASS_PCT
                       else ("WARN" if pct >= 40 else "FAIL"))
            cf      = "PASS" if float(row["avg_r"]) > COST_FLOOR_R else "FAIL"

            rows.append(dict(
                atr_mult          = am,
                atr_filter        = af,
                lookback_n        = n,
                hold_bars         = hb,
                trades            = int(row["trades"]),
                total_r           = float(row["total_r"]),
                avg_r             = float(row["avg_r"]),
                pf                = float(row["pf"]),
                max_dd_r          = float(row["max_dd_r"]),
                win_rate_pct      = float(row["win_rate_pct"]),
                zone_size         = len(zone_combos),
                zone_pass         = len(zone_pass),
                zone_pct          = round(pct, 1),
                stability_verdict = verdict,
                sec42             = cf,
            ))

    return pd.DataFrame(rows).sort_values(
        ["atr_filter", "atr_mult", "zone_pct", "avg_r"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)


# =============================================================================
# MAIN GRID RUN  (precompute once per symbol)
# =============================================================================

def run_grid(symbols: List[str]) -> Tuple[pd.DataFrame, List[dict], int]:
    combos = [
        (n, hb, am, af)
        for n  in LOOKBACK_NS
        for hb in HOLD_BARS
        for am in ATR_MULTS
        for af in ATR_FILTERS
    ]

    combo_trades: Dict[Tuple, List[dict]] = {c: [] for c in combos}
    loaded       = 0
    pre2021_count = 0

    for sym_i, symbol in enumerate(symbols, 1):
        raw = load_1d(symbol)
        if raw is None:
            continue

        # Precompute indicators ONCE for this symbol
        result = precompute_symbol(raw)
        if result is None:
            continue
        d, warmup = result
        loaded += 1

        if d.index.min() < pd.Timestamp("2021-01-01", tz="UTC"):
            pre2021_count += 1

        # Run all combos on the precomputed dataframe
        for (n, hb, am, af) in combos:
            trades = backtest_symbol(symbol, d, warmup, n, hb, am, af)
            combo_trades[(n, hb, am, af)].extend(trades)

        if sym_i % 50 == 0 or sym_i == len(symbols):
            p(f"  [{sym_i}/{len(symbols)}]  loaded={loaded}  pre2021={pre2021_count}")

    # Aggregate per combo
    rows = []
    all_trades: List[dict] = []
    for (n, hb, am, af), trades in combo_trades.items():
        all_trades.extend(trades)
        s = _stats(trades)
        rows.append(dict(lookback_n=n, hold_bars=hb, atr_mult=am, atr_filter=af, **s))

    return pd.DataFrame(rows), all_trades, pre2021_count


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    grid_df: pd.DataFrame,
    sa: pd.DataFrame,
    all_trades_df: pd.DataFrame,
    n_loaded: int,
    pre2021_count: int,
) -> None:
    lines = [
        "PHASE T1 -- New High Failure Short",
        "=" * 80,
        "",
        "Hypothesis: Short FAILED breakouts, not breakdowns.",
        "  high > max(prev N highs)  AND  close < open  AND  close < EMA200",
        "  --> bearish candle at new high confirms bull trap",
        "  --> enter SHORT at close, safety stop above the new high",
        "",
        f"Timeframe:    1D  (4H not cached locally)",
        f"Universe:     {n_loaded} symbols  |  pre-2021 (2022 gate): {pre2021_count}",
        f"Total combos: {len(LOOKBACK_NS)*len(HOLD_BARS)*len(ATR_MULTS)*len(ATR_FILTERS)}",
        f"Total trades: {len(all_trades_df)} (across all combos)",
        "",
        f"lookback_n:   {LOOKBACK_NS}",
        f"hold_bars:    {HOLD_BARS}",
        f"atr_mult:     {ATR_MULTS}",
        f"atr_filter:   {ATR_FILTERS}",
        f"Stability:    N +/-1 AND hb +/-1, PASS if >={STABILITY_PASS_PCT:.0f}%",
        f"§4.2 floor:   avg_r > {COST_FLOOR_R}R",
        "",
        "PRIMARY GATE:       2022 must be POSITIVE",
        "CONCENTRATION FLAG: 2025 > 40% of total R",
        "",
    ]

    # Per filter and ATR mult
    for af in ATR_FILTERS:
        for am in ATR_MULTS:
            sub     = sa[(sa["atr_filter"] == af) & (sa["atr_mult"] == am)]
            if sub.empty:
                continue
            pass_sa = sub[sub["stability_verdict"] == "PASS"]
            warn_sa = sub[sub["stability_verdict"] == "WARN"]

            lines += [
                "=" * 80,
                f"atr_filter={af}  stop_mult={am:.1f}  |  "
                f"PASS: {len(pass_sa)}  WARN: {len(warn_sa)}  "
                f"FAIL: {len(sub) - len(pass_sa) - len(warn_sa)}",
                "=" * 80,
            ]

            for label, df_sub in [("PASS", pass_sa.head(10)), ("WARN", warn_sa.head(5))]:
                if df_sub.empty:
                    continue
                lines.append(f"TOP {label} COMBOS (sorted by avg_r)")
                lines.append("-" * 60)
                for _, row in df_sub.sort_values("avg_r", ascending=False).iterrows():
                    lines.append(
                        f"  [{label}] N={int(row['lookback_n']):2d}  hb={int(row['hold_bars']):2d}  "
                        f"zone={int(row['zone_pass'])}/{int(row['zone_size'])} ({row['zone_pct']:.0f}%)  "
                        f"t={int(row['trades'])}  avg_r={row['avg_r']:+.4f}  "
                        f"pf={row['pf']:.2f}  win%={row['win_rate_pct']:.1f}%  §4.2={row['sec42']}"
                    )
                lines.append("")

            # avg_r heatmap
            g = grid_df[(grid_df["atr_filter"] == af) & (grid_df["atr_mult"] == am)]
            if not g.empty:
                lines += [
                    f"avg_r HEATMAP  (filter={af}  ATR*{am:.1f})  * = passes §4.2",
                    "-" * 55,
                    "  N\\hb  " + "  ".join(f"hb={hb:2d}" for hb in HOLD_BARS),
                ]
                for n in LOOKBACK_NS:
                    vals = []
                    for hb in HOLD_BARS:
                        sl = g[(g["lookback_n"] == n) & (g["hold_bars"] == hb)]
                        if sl.empty:
                            vals.append("  ----")
                        else:
                            v  = float(sl.iloc[0]["avg_r"])
                            cf = "*" if v > COST_FLOOR_R else " "
                            vals.append(f"{cf}{'+' if v >= 0 else ''}{v:.3f}")
                    lines.append(f"  N={n:2d}:  " + "  ".join(vals))
                lines.append("")

    # 2022 Bear Market Check
    cf_pass = sa[(sa["stability_verdict"] == "PASS") & (sa["sec42"] == "PASS")]
    lines += [
        "=" * 80,
        "2022 BEAR MARKET CHECK  (PASS + §4.2 combos only)",
        "=" * 80,
        f"  Only {pre2021_count} of {n_loaded} symbols have pre-2021 data (2022 gate applies to these).",
    ]
    if cf_pass.empty:
        best_r = sa["avg_r"].max() if not sa.empty else 0.0
        lines.append(f"  No combos pass stability + §4.2.  Best avg_r = {best_r:+.4f}R")
    else:
        t2022 = all_trades_df[all_trades_df["entry_year"] == 2022]
        for _, row in cf_pass.sort_values("avg_r", ascending=False).head(15).iterrows():
            sl   = t2022[
                (t2022["lookback_n"]  == row["lookback_n"]) &
                (t2022["hold_bars"]   == row["hold_bars"]) &
                (t2022["atr_mult"]    == row["atr_mult"]) &
                (t2022["atr_filter"]  == row["atr_filter"])
            ]
            s22  = _stats(sl.to_dict("records"))
            ok   = "OK" if s22["total_r"] >= 0 else "FAIL -- 2022 NEGATIVE"
            lines.append(
                f"  N={int(row['lookback_n']):2d}  hb={int(row['hold_bars']):2d}  "
                f"ATR*{row['atr_mult']:.1f}  filter={row['atr_filter']}  "
                f"2022: t={s22['trades']}  avg_r={s22['avg_r']:+.4f}  "
                f"total={s22['total_r']:+.1f}R  [{ok}]"
            )
    lines.append("")

    # Year-by-year best combo
    best_rows = sa.sort_values(
        ["sec42", "stability_verdict", "avg_r"],
        ascending=[False, False, False]
    )
    lines += ["=" * 80, "YEAR-BY-YEAR  (best overall combo)", "=" * 80]
    if best_rows.empty:
        lines.append("  No data.")
    else:
        best = best_rows.iloc[0]
        bt   = all_trades_df[
            (all_trades_df["lookback_n"] == best["lookback_n"]) &
            (all_trades_df["hold_bars"]  == best["hold_bars"]) &
            (all_trades_df["atr_mult"]   == best["atr_mult"]) &
            (all_trades_df["atr_filter"] == best["atr_filter"])
        ].to_dict("records")
        yby   = year_by_year(bt)
        total = _stats(bt)
        flags = _year_flags(yby, total["total_r"])
        lines.append(
            f"  N={int(best['lookback_n'])}  hb={int(best['hold_bars'])}  "
            f"ATR*{best['atr_mult']:.1f}  filter={best['atr_filter']}  "
            f"verdict={best['stability_verdict']}  §4.2={best['sec42']}"
        )
        lines.append(f"  {'Year':>6}  {'Trades':>6}  {'TotalR':>8}  {'AvgR':>8}  {'Win%':>6}  {'PF':>5}")
        lines.append(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}")
        for yr in REPORT_YEARS:
            s = yby[yr]
            if s["trades"] == 0:
                continue
            lines.append(
                f"  {yr}  {s['trades']:>6}  {s['total_r']:>+8.2f}  "
                f"{s['avg_r']:>+8.4f}  {s['win_rate_pct']:>6.1f}%  {s['pf']:>5.2f}"
            )
        lines.append("")
        for fl in flags:
            lines.append(f"  {fl}")
        lines.append("")

    # Verdict
    lines += ["=" * 80, "OVERALL VERDICT", "=" * 80]
    if cf_pass.empty:
        best_r = sa["avg_r"].max() if not sa.empty else 0.0
        lines += [
            f"  NO combos pass stability + §4.2.  Best avg_r = {best_r:+.4f}R",
            "  VERDICT: HALT T1 -- insufficient edge.",
        ]
    else:
        best = cf_pass.sort_values("avg_r", ascending=False).iloc[0]
        lines += [
            f"  PASS combos (stability + §4.2): {len(cf_pass)}",
            f"  Best: N={int(best['lookback_n'])}  hb={int(best['hold_bars'])}  "
            f"ATR*{best['atr_mult']:.1f}  filter={best['atr_filter']}",
            f"  avg_r={best['avg_r']:+.4f}R  pf={best['pf']:.2f}  "
            f"zone={int(best['zone_pass'])}/{int(best['zone_size'])} ({best['zone_pct']:.0f}%)",
            "  VERDICT: PROCEED TO T2 -- review 2022 gate above",
        ]

    report_path = OUT_DIR / "phase_t1_newhighfailure_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        p(line)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    p("=" * 80)
    p("Phase T1 -- New High Failure Short")
    p("=" * 80)
    p()

    symbols = load_all_symbols()
    p(f"Universe: {len(symbols)} symbols in ohlcv_1d/")
    p(f"Combos: {len(LOOKBACK_NS)*len(HOLD_BARS)*len(ATR_MULTS)*len(ATR_FILTERS)} total")
    p()
    p("Performance note: indicators precomputed once per symbol (not per combo).")
    p()

    p("Running grid search ...")
    grid_df, all_trades, pre2021_count = run_grid(symbols)

    if not all_trades:
        p("ERROR: No trades generated.")
        sys.exit(1)

    all_trades_df = pd.DataFrame(all_trades)
    n_loaded      = int(grid_df["trades"].count())   # rough proxy
    p(f"\nTotal trades: {len(all_trades_df)}")

    grid_df.to_csv(OUT_DIR / "phase_t1_newhighfailure_grid.csv", index=False)
    all_trades_df.to_csv(OUT_DIR / "phase_t1_newhighfailure_trades.csv", index=False)

    p("\nRunning stability analysis ...")
    sa = stability_analysis(grid_df)
    sa.to_csv(OUT_DIR / "phase_t1_newhighfailure_stability.csv", index=False)

    # Count loaded symbols from trade data
    n_loaded = all_trades_df["symbol"].nunique() if len(all_trades_df) > 0 else 0

    p("\nWriting report ...")
    write_report(grid_df, sa, all_trades_df, n_loaded, pre2021_count)

    p(f"\nAll output in {OUT_DIR}")
    p("Done. Review report before proceeding to T2.")


if __name__ == "__main__":
    main()
