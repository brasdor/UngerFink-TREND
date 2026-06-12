#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 V2 -- DonchianShort Futures Concept Discovery (ATR Percentile Filter)

Previous attempt (V1) failed: stability PASS on 4H but §4.2 FAIL (avg_r=0.0019R).
Root cause: EMA200 filter alone does not suppress false short signals in
ranging/recovering markets (2021, 2023, 2024). 2022 1D was actually positive
(ATR*3.0 N=20: +71.4R) but non-bear years overwhelmed the edge.

V2 adds an ATR percentile gate:
  Only enter SHORT when ATR(14) > Nth percentile of trailing 252 bars.
  Rationale: genuine downtrends have elevated ATR (active selling).
  Consolidation / bull recovery: ATR compresses — blocks false entries.
  This directly addresses the 2021/2023/2024 false-signal problem.

Entry:  Close < Donchian N-period LOW (shifted 1 bar, no lookahead)
Filter: Close < EMA200  (mandatory bear regime gate)
        AND ATR(14) > atr_pct_threshold (N-th percentile of trailing 252 bars)
Exit:   Donchian N//2 upper band OR Chandelier trailing stop (whichever fires first)

All data is locally cached — no downloads required.
  1D:   data/futures_universe/ohlcv_1d/    (290 symbols, 2019-2026)
  4H:   data/futures_universe/ohlcv_4h/    (59 pre-2021 symbols)
  6H:   data/futures_universe/ohlcv_6h/    (59 pre-2021 symbols)

§4.2 cost floor: 0.25R (Futures, funding rate adjusted)
Stability zone:  N in [15, 20, 25], PASS if >=67% profitable

Output: data/research_donchianshort_futures_v2_t1/
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
CACHE_6H  = ROOT / "data" / "futures_universe" / "ohlcv_6h"
SYM_PRE21 = ROOT / "data" / "futures_universe" / "symbols_pre2021.csv"
SYM_ALL   = ROOT / "data" / "futures_universe" / "all_symbols.csv"
OUT_DIR   = ROOT / "data" / "research_donchianshort_futures_v2_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES      = ["1d", "4h", "6h"]
DONCHIAN_NS     = [10, 15, 20, 25, 30, 40, 55]
ATR_MULTS       = [2.0, 3.0, 4.0]           # extended: 4.0 added vs V1
ATR_PCT_FILTERS = [30, 40, 50]               # minimum ATR percentile to enter
FILTER_MODE     = "ema200_price_below"

ATR_N             = 14
EMA_N             = 200
ATR_PCT_WINDOW    = 252     # lookback for ATR percentile (1 trading year equiv)
ATR_PCT_MIN_BARS  = 100     # minimum bars for percentile to be valid

STABILITY_ZONE_NS  = [15, 20, 25]
STABILITY_PASS_PCT = 67.0
COST_FLOOR_R       = 0.25      # Futures §4.2 (funding rate adjusted)
MIN_BARS           = 300       # minimum bars to process a symbol

REPORT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

EPS = 1e-12


def p(*a, **kw) -> None:
    kw.setdefault("flush", True)
    text = " ".join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), **kw)


# =============================================================================
# UNIVERSE LOADING
# =============================================================================

def load_universe(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]


# =============================================================================
# DATA LOADING (all local — no downloads)
# =============================================================================

def _parse_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise column names and set DatetimeIndex (UTC). Returns None on error."""
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
        if ts.dtype == object or ts.astype(str).str.contains(r"\D").any():
            df.index = pd.to_datetime(ts, utc=True, errors="coerce")
        else:
            df.index = pd.to_datetime(ts.astype(float), unit="ms", utc=True)
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


def load_intra(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    cache_dir = CACHE_4H if tf == "4h" else CACHE_6H
    path = cache_dir / f"{symbol}_{tf}.csv"
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
    don_exit = max(don_n // 2, 5)

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # ATR percentile rank within trailing 252-bar window
    # Value = fraction of past 252 ATR values <= current ATR, scaled to [0,100]
    d["atr_pct_rank"] = (
        d["atr"]
        .rolling(ATR_PCT_WINDOW, min_periods=ATR_PCT_MIN_BARS)
        .apply(lambda x: (x[:-1] <= x[-1]).mean() * 100 if len(x) > 1 else 50.0,
               raw=True)
    )

    # Donchian bands (shifted 1 bar — no lookahead)
    d["don_low"]       = d["low"].shift(1).rolling(don_n).min()
    d["don_high_exit"] = d["high"].shift(1).rolling(don_exit).max()

    d["ema200"] = d["close"].ewm(span=EMA_N, adjust=False).mean()

    return d, don_exit


# =============================================================================
# BACKTESTER — single symbol, SHORT side
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    don_n: int,
    atr_mult: float,
    atr_pct_min: int,
    tf: str,
) -> List[dict]:
    d, don_exit = add_indicators(df, don_n)
    warmup = max(don_n, don_exit, EMA_N, ATR_N, ATR_PCT_WINDOW) + 5

    pos    = None
    trades = []

    for i in range(warmup, len(d)):
        row  = d.iloc[i]
        atr  = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0:
            continue

        close         = float(row["close"])
        high          = float(row["high"])
        low           = float(row["low"])
        t             = d.index[i]
        don_low       = row.get("don_low")
        don_high_ex   = row.get("don_high_exit")
        ema200        = row.get("ema200")
        atr_pct_rank  = row.get("atr_pct_rank")

        # Validate all required indicators
        for v in [don_low, don_high_ex, ema200, atr_pct_rank]:
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
            pos["lowest_low"] = min(pos["lowest_low"], low)
            chandelier_stop   = pos["lowest_low"] + pos["atr_at_entry"] * atr_mult
            active_stop       = min(pos["initial_stop"], chandelier_stop)
            reason = "chandelier" if chandelier_stop < pos["initial_stop"] else "stop"

            # Exit 1: stop / chandelier hit (checked on HIGH)
            if high >= active_stop:
                net_r = (pos["entry"] - active_stop) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, active_stop, net_r, reason,
                    don_n, don_exit, atr_mult, atr_pct_min, tf,
                ))
                pos = None
                continue

            # Exit 2: Donchian N//2 upper band (checked on CLOSE)
            if close > float(don_high_ex):
                net_r = (pos["entry"] - close) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "donchian_exit",
                    don_n, don_exit, atr_mult, atr_pct_min, tf,
                ))
                pos = None
                continue

        # --- Check for new entry ---
        if pos is None:
            bear_regime  = close < float(ema200)
            atr_active   = float(atr_pct_rank) >= atr_pct_min   # V2 gate
            short_signal = close < float(don_low)

            if bear_regime and atr_active and short_signal:
                risk = atr * atr_mult
                if risk > EPS:
                    pos = {
                        "entry_time":   t,
                        "entry":        close,
                        "initial_stop": close + risk,
                        "risk":         risk,
                        "atr_at_entry": atr,
                        "lowest_low":   low,
                    }

    return trades


def _make_trade(
    symbol: str, pos: dict, exit_time, exit_price: float,
    net_r: float, reason: str,
    don_n: int, don_exit: int, atr_mult: float, atr_pct_min: int, tf: str,
) -> dict:
    return {
        "symbol":       symbol,
        "side":         "SHORT",
        "timeframe":    tf,
        "filter_mode":  FILTER_MODE,
        "don_n":        don_n,
        "don_exit_n":   don_exit,
        "atr_mult":     atr_mult,
        "atr_pct_min":  atr_pct_min,
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry"],
        "exit_price":   float(exit_price),
        "initial_stop": pos["initial_stop"],
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
        if abs(total_r) > EPS and abs(yr_r) > 0.5 * abs(total_r):
            flags.append(
                f"[CONCENTRATED]  {yr} = {yr_r:+.1f}R = "
                f"{yr_r / total_r * 100:.0f}% of total R"
            )
    year_rs = {yr: yby[yr]["total_r"] for yr in REPORT_YEARS
               if yby[yr]["trades"] > 0}
    if year_rs:
        ranked = sorted(year_rs.items(), key=lambda x: x[1], reverse=True)
        top2   = {yr for yr, _ in ranked[:2]}
        if 2022 in year_rs and 2022 not in top2:
            rank_2022 = next(i + 1 for i, (yr, _) in enumerate(ranked) if yr == 2022)
            flags.append(
                f"[!2022 NOT TOP2] 2022 ranks #{rank_2022}  "
                f"(best={ranked[0][0]} at {ranked[0][1]:+.1f}R)"
            )
        elif 2022 not in year_rs:
            flags.append("[!2022 NO TRADES]  No trades in 2022")
    return flags


# =============================================================================
# STABILITY — grouped by (tf, atr_mult, atr_pct_min)
# =============================================================================

def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, am, pct_min), g in grid.groupby(
        ["timeframe", "atr_mult", "atr_pct_min"]
    ):
        zone      = g[g["don_n"].isin(STABILITY_ZONE_NS)]
        n_zone    = len(zone)
        pf_gt1    = int((zone["pf"] > 1.0).sum())
        cf_pass   = int((zone["avg_r"] > COST_FLOOR_R).sum())
        pct       = 100.0 * pf_gt1 / max(n_zone, 1)
        verdict   = ("PASS" if pct >= STABILITY_PASS_PCT
                     else ("WARN" if pct >= 40 else "FAIL"))

        can20 = g[g["don_n"] == 20]
        rows.append(dict(
            timeframe=tf,
            atr_mult=am,
            atr_pct_min=int(pct_min),
            zone_total=n_zone,
            zone_pf_gt1=pf_gt1,
            zone_cost_floor_pass=cf_pass,
            zone_stability_pct=round(pct, 1),
            stability_verdict=verdict,
            can_n20_trades=int(can20["trades"].sum())         if not can20.empty else 0,
            can_n20_avg_r=round(float(can20["avg_r"].mean()), 4) if not can20.empty else 0.0,
            can_n20_pf=round(float(can20["pf"].mean()), 3)   if not can20.empty else 0.0,
            can_n20_dd_r=round(float(can20["max_dd_r"].mean()), 2) if not can20.empty else 0.0,
            can_n20_win_pct=round(float(can20["win_rate_pct"].mean()), 1) if not can20.empty else 0.0,
        ))

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct", "can_n20_avg_r"], ascending=False
    ).reset_index(drop=True)


# =============================================================================
# REPORTS
# =============================================================================

def write_year_by_year_report(
    grid: pd.DataFrame, sa: pd.DataFrame, all_trades_df: pd.DataFrame
) -> None:
    pass_rows = sa[sa["stability_verdict"].isin(["PASS", "WARN"])]
    if pass_rows.empty:
        p("No PASS/WARN combos — skipping year-by-year report.")
        return

    lines = [
        "YEAR-BY-YEAR ANALYSIS  (PASS + WARN combinations) — DonchianShort V2",
        "=" * 80,
        "",
        "V2 filter: EMA200 below + ATR percentile gate",
        "Flags:",
        "  [CONCENTRATED]     any single year > 50% of total R",
        "  [!2022 NOT TOP2]   2022 is not the best or second-best year",
        "  [!2022 NO TRADES]  no trades at all in 2022",
        "",
    ]

    for _, row in sa.iterrows():
        tf      = row["timeframe"]
        am      = row["atr_mult"]
        pct_min = int(row["atr_pct_min"])
        verdict = row["stability_verdict"]
        if verdict not in ("PASS", "WARN"):
            continue

        lines += [
            f"{'=' * 70}",
            f"  [{verdict}]  TF={tf}  ATR*{am:.1f}  ATR_pct>={pct_min}  "
            f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}  "
            f"({row['zone_stability_pct']:.0f}%)  "
            f"cf_pass={int(row['zone_cost_floor_pass'])}/{int(row['zone_total'])}",
            f"{'=' * 70}",
        ]

        for don_n in sorted(set(STABILITY_ZONE_NS + [20])):
            sl = all_trades_df[
                (all_trades_df["timeframe"]  == tf) &
                (all_trades_df["atr_mult"]   == am) &
                (all_trades_df["atr_pct_min"] == pct_min) &
                (all_trades_df["don_n"]       == don_n)
            ]
            if sl.empty:
                continue

            trade_list = sl.to_dict("records")
            s   = _stats(trade_list)
            yby = year_by_year(trade_list)
            total_r = s["total_r"]

            zone_tag = " [zone]" if don_n in STABILITY_ZONE_NS else ""
            can_tag  = " [canonical]" if don_n == 20 else ""
            cf_tag   = " [cf PASS]" if s["avg_r"] > COST_FLOOR_R else ""
            lines += [
                "",
                f"  N={don_n}  exit={don_n // 2}  trades={s['trades']}  "
                f"avg_r={s['avg_r']:+.4f}  pf={s['pf']:.2f}  "
                f"total_r={total_r:+.1f}R" + zone_tag + can_tag + cf_tag,
                "",
                f"  {'Year':>4}  {'Trades':>6}  {'Win%':>6}  "
                f"{'TotalR':>8}  {'AvgR':>7}  {'PF':>5}  {'MaxDD':>7}",
                f"  {'-' * 58}",
            ]
            for yr in REPORT_YEARS:
                ys = yby[yr]
                if ys["trades"] == 0:
                    lines.append(f"  {yr:>4}  {'--':>6}  {'--':>6}  "
                                 f"{'--':>8}  {'--':>7}  {'--':>5}  {'--':>7}")
                else:
                    pct_of_total = ys["total_r"] / max(abs(total_r), EPS) * 100
                    conc = " <-- CONCENTRATED" if abs(pct_of_total) > 50 else ""
                    neg_flag = " <-- 2022 NEGATIVE" if yr == 2022 and ys["total_r"] < 0 else ""
                    lines.append(
                        f"  {yr:>4}  {ys['trades']:>6}  {ys['win_rate_pct']:>5.1f}%  "
                        f"  {ys['total_r']:>+7.1f}R  {ys['avg_r']:>+6.3f}  "
                        f"{ys['pf']:>5.2f}  {ys['max_dd_r']:>+7.2f}"
                        + conc + neg_flag
                    )

            flags = _year_flags(yby, total_r)
            if flags:
                lines += ["", "  ** FLAGS:"]
                for fl in flags:
                    lines.append(f"     {fl}")

    out = OUT_DIR / "phase_t1_donchianshort_v2_year_by_year.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"Year-by-year report: {out}")


def write_main_report(
    grid: pd.DataFrame,
    sa: pd.DataFrame,
    sym_count: Dict[str, int],
    all_trades_df: pd.DataFrame,
) -> None:
    lines = [
        "PHASE T1 V2 -- DonchianShort Futures (ATR Percentile Filter)",
        "=" * 80,
        "",
        "V1 diagnosis: EMA200 alone insufficient — false short signals in",
        "              bull/recovering markets (2021, 2023, 2024) killed the edge.",
        "              4H stability PASS but §4.2 FAIL (avg_r=0.0019R).",
        "              1D 2022 was positive (+71.4R at ATR*3 N=20) — edge exists.",
        "",
        "V2 solution:  Add ATR percentile gate — only short when ATR(14) is",
        "              above the Nth percentile of trailing 252 bars.",
        "              Hypothesis: genuine downtrends have elevated ATR;",
        "              consolidation/recovery has compressed ATR — gate blocks",
        "              false entries in 2021/2023/2024 bull phases.",
        "",
        "Entry:   Close < Donchian N-period LOW (shifted 1 bar, no lookahead)",
        "Filter:  Close < EMA200  AND  ATR(14) >= atr_pct_min-th percentile",
        f"         ATR percentile window: {ATR_PCT_WINDOW} bars",
        "Stop:    Entry + ATR(14) * atr_mult  (above entry for short)",
        "Trail:   Chandelier SHORT = lowest_low_since_entry + ATR_at_entry * atr_mult",
        "         active_stop = min(initial_stop, chandelier_stop)",
        "Exit:    high >= active_stop  OR  close > Donchian(N//2) upper band",
        "",
        f"Timeframes:         {TIMEFRAMES}",
        f"Donchian N:         {DONCHIAN_NS}",
        f"ATR mults:          {ATR_MULTS}  (4.0 added vs V1)",
        f"ATR pct filters:    {ATR_PCT_FILTERS}  (NEW in V2)",
        f"Cost floor (§4.2):  avg_r > {COST_FLOOR_R}R  (Futures)",
        f"Stability zone:     N in {STABILITY_ZONE_NS}  (PASS if >={STABILITY_PASS_PCT:.0f}%)",
        "",
    ]

    for tf in TIMEFRAMES:
        tf_sa   = sa[sa["timeframe"] == tf]
        tf_grid = grid[grid["timeframe"] == tf]
        n_syms  = sym_count.get(tf, 0)

        lines += [
            "=" * 80,
            f"TIMEFRAME: {tf}  (symbols loaded: {n_syms})",
            "=" * 80,
            "STABILITY RANKING  (sorted by stability%, then avg_r at N=20)",
            "-" * 70,
        ]

        for _, row in tf_sa.iterrows():
            v   = row["stability_verdict"]
            cf  = f"cf={int(row['zone_cost_floor_pass'])}/{int(row['zone_total'])}"
            lines.append(
                f"  [{v:4s}] ATR*{row['atr_mult']:.1f}  pct>={int(row['atr_pct_min']):2d}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])} profitable "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: trades={row['can_n20_trades']}  "
                f"avg_r={row['can_n20_avg_r']:+.4f}  "
                f"pf={row['can_n20_pf']:.3f}  "
                f"dd={row['can_n20_dd_r']:+.2f}  "
                f"win%={row['can_n20_win_pct']:.1f}%  "
                + cf
            )

        pass_rows = tf_sa[tf_sa["stability_verdict"] == "PASS"]
        warn_rows = tf_sa[tf_sa["stability_verdict"] == "WARN"]
        best = (pass_rows.iloc[0] if not pass_rows.empty
                else (warn_rows.iloc[0] if not warn_rows.empty else None))
        if best is not None:
            gate_42 = "PASS" if best["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
            lines += [
                "",
                f"  Best ({tf}):  ATR*{best['atr_mult']:.1f}  pct>={int(best['atr_pct_min'])}  "
                f"stability={best['zone_stability_pct']:.0f}% ({best['stability_verdict']})  "
                f"§4.2={gate_42} ({best['can_n20_avg_r']:.4f}R)",
            ]

        # N sensitivity for best ATR mult + best pct filter (PASS if exists)
        if best is not None:
            am_b   = best["atr_mult"]
            pct_b  = int(best["atr_pct_min"])
            g = tf_grid[(tf_grid["atr_mult"] == am_b) &
                        (tf_grid["atr_pct_min"] == pct_b)].sort_values("don_n")
            lines += [
                "",
                f"N SENSITIVITY  ({tf}  ATR*{am_b:.1f}  pct>={pct_b})",
                "-" * 50,
            ]
            for _, row in g.iterrows():
                n    = int(row["don_n"])
                sign = "[+]" if row["pf"] > 1.0 else "[-]"
                zn   = " <- zone"      if n in STABILITY_ZONE_NS else ""
                cn   = " <- canonical" if n == 20 else ""
                cf_t = " (cf PASS)"    if row["avg_r"] > COST_FLOOR_R else ""
                lines.append(
                    f"  {sign} N={n:2d}  exit={n // 2:2d}  "
                    f"trades={int(row['trades']):4d}  "
                    f"avg_r={row['avg_r']:+.4f}  "
                    f"pf={row['pf']:.2f}  "
                    f"dd={row['max_dd_r']:+.2f}  "
                    f"win%={row['win_rate_pct']:.1f}%"
                    + cf_t + zn + cn
                )

        lines.append("")

    # --- 2022 check across all PASS/WARN combos ---
    lines += [
        "=" * 80,
        "2022 BEAR MARKET CHECK  (must be positive for short system)",
        "=" * 80,
    ]
    pass_warn = sa[sa["stability_verdict"].isin(["PASS", "WARN"])]
    if pass_warn.empty:
        lines.append("  No PASS/WARN combos — 2022 check skipped.")
    else:
        t2022 = all_trades_df[all_trades_df["entry_year"] == 2022]
        for _, row in pass_warn.iterrows():
            tf  = row["timeframe"]
            am  = row["atr_mult"]
            pct = int(row["atr_pct_min"])
            sl  = t2022[
                (t2022["timeframe"]  == tf) &
                (t2022["atr_mult"]   == am) &
                (t2022["atr_pct_min"] == pct) &
                (t2022["don_n"] == 20)
            ]
            s = _stats(sl.to_dict("records"))
            status = "OK" if s["total_r"] > 0 else "NEGATIVE -- SHORT SYSTEM SHOULD PROFIT IN 2022"
            lines.append(
                f"  [{row['stability_verdict']:4s}] {tf}  ATR*{am:.1f}  pct>={pct}  N=20  "
                f"2022: trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
                f"total_r={s['total_r']:+.1f}R  [{status}]"
            )

    # --- V1 vs V2 comparison for equivalent combos ---
    lines += [
        "",
        "=" * 80,
        "V1 vs V2 COMPARISON  (pct_filter=0 equivalent to V1)",
        "=" * 80,
        "",
        "V1 best results (from previous run, for reference):",
        "  1D ATR*3.0 N=20: avg_r=-0.0120R  pf=0.96  [FAIL]",
        "  4H ATR*3.0 N=20: avg_r=+0.0019R  pf=1.01  [PASS stability, FAIL §4.2]",
        "  6H ATR*3.0 N=20: avg_r=-0.0080R  pf=0.97  [FAIL]",
        "",
        "V2 equivalent (pct>=0 = no filter, should match V1):",
    ]
    for tf in TIMEFRAMES:
        for am in ATR_MULTS:
            sl = all_trades_df[
                (all_trades_df["timeframe"] == tf) &
                (all_trades_df["atr_mult"]  == am) &
                (all_trades_df["atr_pct_min"] == 0) &
                (all_trades_df["don_n"] == 20)
            ] if 0 in ATR_PCT_FILTERS else pd.DataFrame()
            # Note: pct=0 not in grid if 0 not in ATR_PCT_FILTERS — show lowest
            min_pct = min(ATR_PCT_FILTERS)
            sl = all_trades_df[
                (all_trades_df["timeframe"] == tf) &
                (all_trades_df["atr_mult"]  == am) &
                (all_trades_df["atr_pct_min"] == min_pct) &
                (all_trades_df["don_n"] == 20)
            ]
            if sl.empty:
                continue
            s = _stats(sl.to_dict("records"))
            lines.append(
                f"  V2 {tf} ATR*{am:.1f} pct>={min_pct} N=20: "
                f"trades={s['trades']}  avg_r={s['avg_r']:+.4f}  pf={s['pf']:.2f}"
            )

    # --- Overall best ---
    all_pass = sa[sa["stability_verdict"] == "PASS"]
    all_warn = sa[sa["stability_verdict"] == "WARN"]
    best_overall = (
        all_pass.iloc[0] if not all_pass.empty
        else (all_warn.iloc[0] if not all_warn.empty
              else (sa.iloc[0] if not sa.empty else None))
    )

    lines += [
        "",
        "=" * 80,
        "OVERALL BEST CANDIDATE",
        "=" * 80,
    ]
    if best_overall is not None:
        gate_42  = "PASS" if best_overall["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
        proceed  = (
            "PROCEED TO T2"
            if best_overall["stability_verdict"] in ("PASS", "WARN")
            and best_overall["can_n20_avg_r"] > COST_FLOOR_R
            else "DO NOT PROCEED"
        )
        lines += [
            f"  Timeframe:     {best_overall['timeframe']}",
            f"  ATR mult:      {best_overall['atr_mult']:.1f}",
            f"  ATR pct min:   >= {int(best_overall['atr_pct_min'])}th percentile",
            f"  Stability:     {best_overall['zone_stability_pct']:.0f}%  "
            f"({best_overall['stability_verdict']})",
            f"  §4.2 gate:     {gate_42}  ({best_overall['can_n20_avg_r']:.4f}R vs >{COST_FLOOR_R}R)",
            f"  N=20 trades:   {int(best_overall['can_n20_trades'])}",
            f"  N=20 PF:       {best_overall['can_n20_pf']:.3f}",
            "",
            f"  !! {proceed} — awaiting human review.",
            "",
            "  RECOMMENDED CANONICAL CONFIG (if proceeding):",
            f"    TIMEFRAME    = '{best_overall['timeframe']}'",
            f"    ATR_MULT     = {best_overall['atr_mult']:.1f}",
            f"    ATR_PCT_MIN  = {int(best_overall['atr_pct_min'])}",
            f"    DONCHIAN_N   = 20",
            f"    EXIT_N       = 10",
        ]
    else:
        lines.append("  No candidates found.")

    lines += [
        "",
        "=" * 80,
        "DIAGNOSTIC — IF ALL COMBOS FAIL",
        "=" * 80,
        "",
        "  Root causes if V2 also fails:",
        "  1. FEES: avg_r consistently < 0.10R across all combos",
        "     -> Edge exists but Futures cost structure consumes it",
        "     -> Action: lower cost floor, consider equity futures instead",
        "",
        "  2. FALSE SIGNALS: 2022 still negative despite ATR filter",
        "     -> Even elevated-ATR periods in 2022 had whipsaw bounces",
        "     -> Action: try higher ATR pct threshold (60, 70) or add",
        "                price velocity filter (close < 5-day low)",
        "",
        "  3. INSUFFICIENT BEAR DATA: 2022 positive but overwhelmed by",
        "     2021/2023/2024/2025 bull losses despite ATR filter",
        "     -> ATR percentile filter insufficient — need structural regime change",
        "     -> Action: Donchian Short permanently closed; short coverage",
        "                already provided by Momentum Factor short basket",
        "",
        "  4. STRUCTURAL: crypto short-side Donchian has no consistent edge",
        "     on this universe at any parameter combination",
        "     -> Document and halt permanently",
        "",
        "  !! STOP HERE — do not proceed to T2 until human review.",
    ]

    out = OUT_DIR / "phase_t1_donchianshort_v2_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"\nMain report: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    p("=" * 80)
    p("PHASE T1 V2 -- DonchianShort Futures (ATR Percentile Filter)")
    p("=" * 80)
    p(f"Timeframes:      {TIMEFRAMES}")
    p(f"Donchian N:      {DONCHIAN_NS}")
    p(f"ATR mults:       {ATR_MULTS}")
    p(f"ATR pct filters: {ATR_PCT_FILTERS}")
    p(f"Cost floor:      avg_r > {COST_FLOOR_R}R  (Futures §4.2)")
    p(f"Output:          {OUT_DIR}")
    p()

    pre21_syms = load_universe(SYM_PRE21)
    all_syms   = load_universe(SYM_ALL)

    if not pre21_syms and not all_syms:
        p("[ERROR] No universe files found.")
        return 1

    p(f"Universe: {len(pre21_syms)} pre-2021 symbols  |  {len(all_syms)} all symbols")
    p(f"All data is locally cached — no downloads required.")
    p()

    intra_syms = pre21_syms if pre21_syms else all_syms
    d1_syms    = all_syms   if all_syms   else pre21_syms

    all_trades:   List[dict] = []
    summary_rows: List[dict] = []
    sym_count:    Dict[str, int] = {}

    total_pct_combos = len(DONCHIAN_NS) * len(ATR_MULTS) * len(ATR_PCT_FILTERS)

    for tf in TIMEFRAMES:
        p(f"{'=' * 60}")
        p(f"TIMEFRAME: {tf}")
        p(f"{'=' * 60}")

        syms_for_tf = d1_syms if tf == "1d" else intra_syms

        p(f"Loading {len(syms_for_tf)} symbols...")
        data: Dict[str, pd.DataFrame] = {}
        for sym in syms_for_tf:
            df = load_1d(sym) if tf == "1d" else load_intra(sym, tf)
            if df is not None and len(df) >= MIN_BARS:
                data[sym] = df

        n_loaded = len(data)
        sym_count[tf] = n_loaded
        p(f"Loaded: {n_loaded} / {len(syms_for_tf)} ({n_loaded * 100 // max(len(syms_for_tf), 1)}%)")

        if not data:
            p(f"  [WARN] No data for {tf} — skipping.")
            continue

        p(f"Running {total_pct_combos} combos x {n_loaded} symbols...")
        done = 0

        for don_n in DONCHIAN_NS:
            for am in ATR_MULTS:
                for pct_min in ATR_PCT_FILTERS:
                    combo_trades: List[dict] = []
                    for sym, df in data.items():
                        combo_trades.extend(
                            backtest_symbol(sym, df, don_n, am, pct_min, tf)
                        )

                    s = _stats(combo_trades)
                    row = dict(
                        timeframe=tf,
                        don_n=don_n,
                        don_exit_n=don_n // 2,
                        atr_mult=am,
                        atr_pct_min=pct_min,
                        symbols=n_loaded,
                    )
                    row.update(s)
                    summary_rows.append(row)
                    all_trades.extend(combo_trades)
                    done += 1

                    zone_tag = " [zone]" if don_n in STABILITY_ZONE_NS else ""
                    cf_tag   = " [cf PASS]" if s["avg_r"] > COST_FLOOR_R else ""
                    p(
                        f"  [{done:3d}/{total_pct_combos}] N={don_n:2d}  "
                        f"ATR*{am:.1f}  pct>={pct_min:2d}  "
                        f"trades={s['trades']:4d}  avg_r={s['avg_r']:+.4f}  "
                        f"pf={s['pf']:.2f}  win%={s['win_rate_pct']:.1f}%"
                        + cf_tag + zone_tag
                    )

    if not summary_rows:
        p("[ERROR] No trades produced.")
        return 1

    grid = pd.DataFrame(summary_rows)
    sa   = stability_analysis(grid)

    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv(
        OUT_DIR / "phase_t1_donchianshort_v2_trades.csv", index=False)
    grid.to_csv(
        OUT_DIR / "phase_t1_donchianshort_v2_summary.csv", index=False)
    sa.to_csv(
        OUT_DIR / "phase_t1_donchianshort_v2_stability.csv", index=False)

    write_main_report(grid, sa, sym_count, all_trades_df)
    write_year_by_year_report(grid, sa, all_trades_df)

    # --- Console stability summary ---
    p()
    p("=" * 80)
    p("STABILITY RANKING SUMMARY")
    p("=" * 80)
    for tf in TIMEFRAMES:
        tf_sa = sa[sa["timeframe"] == tf]
        if tf_sa.empty:
            continue
        p(f"  [{tf}]  (top 10 shown)")
        for _, row in tf_sa.head(10).iterrows():
            v = row["stability_verdict"]
            p(
                f"    [{v:4s}] ATR*{row['atr_mult']:.1f}  pct>={int(row['atr_pct_min']):2d}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}  "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: avg_r={row['can_n20_avg_r']:+.4f}R  "
                f"pf={row['can_n20_pf']:.3f}  "
                f"cf={int(row['zone_cost_floor_pass'])}/{int(row['zone_total'])}"
            )

    all_pass = sa[sa["stability_verdict"] == "PASS"]
    all_warn = sa[sa["stability_verdict"] == "WARN"]
    p()
    p(f"PASS combos: {len(all_pass)}  |  WARN: {len(all_warn)}  |  "
      f"FAIL: {len(sa) - len(all_pass) - len(all_warn)}")

    # --- 2022 console check for all PASS combos, N=20 ---
    if not all_pass.empty:
        p()
        p("2022 check on all PASS combos (N=20):")
        for _, row in all_pass.iterrows():
            tf  = row["timeframe"]
            am  = row["atr_mult"]
            pct = int(row["atr_pct_min"])
            sl  = all_trades_df[
                (all_trades_df["timeframe"]  == tf) &
                (all_trades_df["atr_mult"]   == am) &
                (all_trades_df["atr_pct_min"] == pct) &
                (all_trades_df["don_n"] == 20) &
                (all_trades_df["entry_year"] == 2022)
            ]
            s = _stats(sl.to_dict("records"))
            sign = "OK" if s["total_r"] > 0 else "NEG"
            p(f"  [{sign}] {tf}  ATR*{am:.1f}  pct>={pct}  "
              f"2022: trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
              f"total_r={s['total_r']:+.1f}R")

        # Year-by-year for the single best PASS combo
        best = all_pass.iloc[0]
        tf_b  = best["timeframe"]
        am_b  = best["atr_mult"]
        pct_b = int(best["atr_pct_min"])
        p()
        p(f"Year-by-year for best PASS: {tf_b}  ATR*{am_b:.1f}  pct>={pct_b}  N=20")
        sl = all_trades_df[
            (all_trades_df["timeframe"]  == tf_b) &
            (all_trades_df["atr_mult"]   == am_b) &
            (all_trades_df["atr_pct_min"] == pct_b) &
            (all_trades_df["don_n"] == 20)
        ]
        yby = year_by_year(sl.to_dict("records"))
        s   = _stats(sl.to_dict("records"))
        p(f"  Total: {s['total_r']:+.1f}R  avg_r={s['avg_r']:+.4f}  "
          f"pf={s['pf']:.3f}")
        for yr in REPORT_YEARS:
            ys = yby[yr]
            if ys["trades"] > 0:
                p(f"  {yr}: {ys['trades']:3d} trades  "
                  f"avg_r={ys['avg_r']:+.4f}  total_r={ys['total_r']:+.1f}R  "
                  f"pf={ys['pf']:.2f}")

    p()
    p("=" * 80)
    p("T1 V2 COMPLETE -- awaiting human review before T2")
    p("=" * 80)
    p(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
