#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- DonchianShort Futures Concept Discovery  (expanded universe)

Re-run of T1 short system against the expanded Binance Futures universe.
Substantially more data than the previous attempt:
  - 1D: 290 symbols from data/futures_universe/ohlcv_1d/  (all listed pairs)
  - 4H/6H: 58 symbols from symbols_pre2021.csv  (downloadable, bear-cycle coverage)
  - Pre-2021 symbols cover 2019/2020 bear cycle + 2022 crash + 2024 recovery
  - All symbols cover 2022+ bear period (universe expansion)

Entry:   Close breaks BELOW Donchian N-period LOW  (shifted 1 bar, no lookahead)
Filter:  Close < EMA200  (mandatory bear regime gate)
Exit 1:  High >= active_stop (initial stop OR chandelier trailing, whichever fires first)
         initial_stop     = entry + ATR * atr_mult  (fixed, above entry)
         chandelier_stop  = lowest_low_since_entry + ATR_at_entry * atr_mult  (trailing)
         active_stop      = min(initial_stop, chandelier_stop)
Exit 2:  Close > Donchian(N//2) upper band  (trend-exit, checked after stop)

Chandelier note: SHORT-side Chandelier is the mirror of the long-side formula.
  LONG:  stop = highest_high_since_entry - ATR * mult
  SHORT: stop = lowest_low_since_entry  + ATR * mult  <- implemented here

Cost floor: 0.25R (Futures, Section 4.2 -- funding rate adjusted)
Stability zone: N in [15, 20, 25], PASS if >= 67% profitable
Year-by-year: 2019/2020/2021/2022/2023/2024/2025 for all PASS combos
  Flag: any single year > 50% of total R  (concentration)
  Flag: 2022 NOT best or second-best year  (unexpected for short system)

Output: data/research_donchianshort_futures_t1/
"""

from __future__ import annotations

import math
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# =============================================================================
# CONFIG
# =============================================================================

ROOT      = Path(__file__).resolve().parent
OHLCV_1D  = ROOT / "data" / "futures_universe" / "ohlcv_1d"
CACHE_4H  = ROOT / "data" / "futures_universe" / "ohlcv_4h"
CACHE_6H  = ROOT / "data" / "futures_universe" / "ohlcv_6h"
SYM_ALL   = ROOT / "data" / "futures_universe" / "all_symbols.csv"
SYM_PRE21 = ROOT / "data" / "futures_universe" / "symbols_pre2021.csv"
OUT_DIR   = ROOT / "data" / "research_donchianshort_futures_t1"

CACHE_4H.mkdir(parents=True, exist_ok=True)
CACHE_6H.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES  = ["1d", "4h", "6h"]
DONCHIAN_NS = [10, 15, 20, 25, 30, 40, 55]
ATR_MULTS   = [2.0, 3.0]
FILTER_MODE = "ema200_price_below"
ATR_N       = 14
EMA_N       = 200

STABILITY_ZONE_NS  = [15, 20, 25]
STABILITY_PASS_PCT = 67.0
COST_FLOOR_R       = 0.25          # Futures: 0.25R (funding rate overhead)
MIN_BARS           = 300           # warmup + minimum meaningful history

REPORT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Bars to fetch for intraday (approx 4 years of history)
TF_TARGET_BARS = {"1d": 2000, "4h": 8760, "6h": 5840}
TF_BAR_MS      = {"1d": 86_400_000, "4h": 14_400_000, "6h": 21_600_000}

EPS = 1e-12


def p(*a, **kw):
    kw.setdefault("flush", True)
    text = " ".join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), **kw)


# =============================================================================
# UNIVERSE
# =============================================================================

def load_universe(sym_file: Path) -> List[str]:
    if not sym_file.exists():
        return []
    df = pd.read_csv(sym_file)
    col = df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna()]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_1d(symbol: str) -> Optional[pd.DataFrame]:
    """Load 1D OHLCV from futures_universe/ohlcv_1d/{SYMBOL}_1d.csv."""
    path = OHLCV_1D / f"{symbol}_1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"].astype(float),
                                       unit="ms", utc=True)
        else:
            return None
        df = df.sort_index()
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return None
        df = df[needed].astype(float).dropna()
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


def _init_exchange():
    if not HAS_CCXT:
        return None
    try:
        ex = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        })
        ex.load_markets()
        return ex
    except Exception:
        return None


def _fetch_paginated(exchange, symbol_ccxt: str, tf: str,
                     target_bars: int) -> Optional[pd.DataFrame]:
    bar_ms   = TF_BAR_MS[tf]
    since_ms = int(time_mod.time() * 1000) - target_bars * bar_ms
    all_rows: List[list] = []
    max_pages = (target_bars // 1000) + 5

    for _ in range(max_pages):
        try:
            rows = exchange.fetch_ohlcv(
                symbol_ccxt, timeframe=tf, since=since_ms, limit=1000,
            )
            time_mod.sleep(exchange.rateLimit / 1000.0)
        except Exception:
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        since_ms = last_ts + bar_ms
        if since_ms > int(time_mod.time() * 1000):
            break
        if len(rows) < 1000:
            break

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows,
                      columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def load_or_download_intra(symbol: str, tf: str,
                            exchange) -> Optional[pd.DataFrame]:
    """Load cached 4H/6H OHLCV, or download from Binance Futures if missing."""
    cache_dir = CACHE_4H if tf == "4h" else CACHE_6H
    path = cache_dir / f"{symbol}_{tf}.csv"

    if path.exists():
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_index()
            needed = ["open", "high", "low", "close", "volume"]
            if all(c in df.columns for c in needed):
                df = df[needed].astype(float).dropna()
                min_needed = TF_TARGET_BARS[tf] // 3
                if len(df) >= min_needed:
                    return df
        except Exception:
            pass

    if exchange is None:
        return None

    # binanceusdm perpetual format: BTCUSDT -> BTC/USDT:USDT
    if not symbol.endswith("USDT"):
        return None
    base     = symbol[:-4]                      # strip trailing USDT
    sym_ccxt = f"{base}/USDT:USDT"             # perpetual futures
    if sym_ccxt not in exchange.symbols:
        # Fallback: linear perps sometimes listed without settlement suffix
        sym_ccxt = f"{base}/USDT"
        if sym_ccxt not in exchange.symbols:
            return None

    p(f"    Downloading {symbol} {tf}...", end=" ")
    df = _fetch_paginated(exchange, sym_ccxt, tf, TF_TARGET_BARS[tf])
    if df is None or len(df) < MIN_BARS:
        p("SKIP (insufficient data)")
        return None

    try:
        df.to_csv(path)
    except Exception:
        pass

    p(f"{len(df)} bars")
    return df


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame, don_n: int) -> pd.DataFrame:
    d = df.copy()
    don_exit = max(don_n // 2, 5)

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # Donchian bands (shifted 1 bar -- no lookahead)
    d["don_low"]      = d["low"].shift(1).rolling(don_n).min()
    d["don_high_exit"] = d["high"].shift(1).rolling(don_exit).max()

    d["ema200"] = d["close"].ewm(span=EMA_N, adjust=False).mean()

    return d, don_exit


# =============================================================================
# BACKTESTER — single symbol, SHORT side, WITH chandelier
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    don_n: int,
    atr_mult: float,
    tf: str,
) -> List[dict]:
    d, don_exit = add_indicators(df, don_n)
    warmup = max(don_n, don_exit, EMA_N, ATR_N) + 5

    pos    = None
    trades = []

    for i in range(warmup, len(d)):
        row          = d.iloc[i]
        atr          = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0:
            continue

        close        = float(row["close"])
        high         = float(row["high"])
        low          = float(row["low"])
        t            = d.index[i]
        don_low      = row.get("don_low")
        don_high_ex  = row.get("don_high_exit")
        ema200       = row.get("ema200")

        for v in [don_low, don_high_ex, ema200]:
            try:
                if not math.isfinite(float(v)):
                    don_low = None
                    break
            except (TypeError, ValueError):
                don_low = None
                break
        if don_low is None:
            continue

        if pos is not None:
            # Update chandelier trailing stop (uses this bar's low)
            pos["lowest_low"] = min(pos["lowest_low"], low)
            chandelier_stop   = pos["lowest_low"] + pos["atr_at_entry"] * atr_mult
            # Active stop: whichever is lower (fires sooner as price rises back up)
            active_stop = min(pos["initial_stop"], chandelier_stop)
            reason = "chandelier" if chandelier_stop < pos["initial_stop"] else "stop"

            # Exit 1: stop or chandelier hit (checked on this bar's HIGH)
            if high >= active_stop:
                net_r = (pos["entry"] - active_stop) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, active_stop, net_r, reason,
                    don_n, don_exit, atr_mult, tf,
                ))
                pos = None
                continue

            # Exit 2: Donchian N//2 upper band (checked on CLOSE)
            if close > float(don_high_ex):
                net_r = (pos["entry"] - close) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "donchian_exit",
                    don_n, don_exit, atr_mult, tf,
                ))
                pos = None
                continue

        if pos is None:
            bear_regime  = close < float(ema200)
            short_signal = close < float(don_low)
            if bear_regime and short_signal:
                risk = atr * atr_mult
                if risk > EPS:
                    pos = {
                        "entry_time":    t,
                        "entry":         close,
                        "initial_stop":  close + risk,
                        "risk":          risk,
                        "atr_at_entry":  atr,
                        "lowest_low":    low,
                    }

    return trades


def _make_trade(
    symbol: str, pos: dict, exit_time, exit_price: float,
    net_r: float, reason: str,
    don_n: int, don_exit: int, atr_mult: float, tf: str,
) -> dict:
    return {
        "symbol":       symbol,
        "side":         "SHORT",
        "timeframe":    tf,
        "filter_mode":  FILTER_MODE,
        "don_n":        don_n,
        "don_exit_n":   don_exit,
        "atr_mult":     atr_mult,
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
# YEAR-BY-YEAR ANALYSIS
# =============================================================================

def year_by_year(trades: List[dict]) -> Dict[int, dict]:
    """Compute per-year stats from a list of trades (keyed by entry_year)."""
    by_year: Dict[int, List[dict]] = {}
    for t in trades:
        yr = int(t.get("entry_year", 0))
        by_year.setdefault(yr, []).append(t)
    result = {}
    for yr in REPORT_YEARS:
        result[yr] = _stats(by_year.get(yr, []))
    return result


def _year_flags(yby: Dict[int, dict], total_r: float) -> List[str]:
    """Return list of flag strings for a given year-by-year breakdown."""
    flags = []

    # Flag 1: any year > 50% of total R
    for yr in REPORT_YEARS:
        yr_r = yby[yr]["total_r"]
        if abs(total_r) > EPS and abs(yr_r) > 0.5 * abs(total_r):
            pct = yr_r / total_r * 100
            flags.append(f"[CONCENTRATED]  {yr} = {yr_r:+.1f}R = {pct:.0f}% of total R")

    # Flag 2: 2022 should be best or second-best year (short system)
    year_rs = {yr: yby[yr]["total_r"] for yr in REPORT_YEARS
               if yby[yr]["trades"] > 0}
    if year_rs:
        ranked = sorted(year_rs.items(), key=lambda x: x[1], reverse=True)
        top2   = {yr for yr, _ in ranked[:2]}
        if 2022 in year_rs and 2022 not in top2:
            rank_2022 = next(i + 1 for i, (yr, _) in enumerate(ranked) if yr == 2022)
            flags.append(
                f"[!2022 NOT TOP2] 2022 ranks #{rank_2022}  "
                f"(best={ranked[0][0]} at {ranked[0][1]:+.1f}R, "
                f"2nd={ranked[1][0] if len(ranked) > 1 else 'n/a'})"
            )
        elif 2022 not in year_rs:
            flags.append("[!2022 NO TRADES]  No trades in 2022 -- check universe coverage")

    return flags


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, am), g in grid.groupby(["timeframe", "atr_mult"]):
        zone       = g[g["don_n"].isin(STABILITY_ZONE_NS)]
        n_zone     = len(zone)
        pf_gt1     = int((zone["pf"] > 1.0).sum())
        avgr_pos   = int((zone["avg_r"] > 0).sum())
        cf_pass    = int((zone["avg_r"] > COST_FLOOR_R).sum())
        pct        = 100.0 * pf_gt1 / max(n_zone, 1)
        verdict    = ("PASS" if pct >= STABILITY_PASS_PCT
                      else ("WARN" if pct >= 40 else "FAIL"))

        can20 = g[g["don_n"] == 20]
        rows.append(dict(
            timeframe=tf, filter_mode=FILTER_MODE, atr_mult=am,
            zone_total=n_zone, zone_pf_gt1=pf_gt1,
            zone_avg_r_positive=avgr_pos,
            zone_cost_floor_pass=cf_pass,
            zone_stability_pct=round(pct, 1),
            stability_verdict=verdict,
            can_n20_trades=int(can20["trades"].sum())     if not can20.empty else 0,
            can_n20_avg_r=round(float(can20["avg_r"].mean()), 4) if not can20.empty else 0.0,
            can_n20_pf=round(float(can20["pf"].mean()), 3)       if not can20.empty else 0.0,
            can_n20_max_dd_r=round(float(can20["max_dd_r"].mean()), 2) if not can20.empty else 0.0,
            can_n20_win_pct=round(float(can20["win_rate_pct"].mean()), 1) if not can20.empty else 0.0,
        ))

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct", "can_n20_pf"], ascending=False
    ).reset_index(drop=True)


# =============================================================================
# YEAR-BY-YEAR REPORT (PASS combos only)
# =============================================================================

def write_year_by_year_report(
    grid: pd.DataFrame,
    sa: pd.DataFrame,
    all_trades_df: pd.DataFrame,
) -> None:
    pass_rows = sa[sa["stability_verdict"] == "PASS"]
    if pass_rows.empty:
        return

    lines = [
        "YEAR-BY-YEAR ANALYSIS  (PASS combinations only)",
        "=" * 80,
        "",
        "Entry year attribution.  Flags:",
        "  [CONCENTRATED]     any single year > 50% of total R",
        "  [!2022 NOT TOP2]   2022 is not the best or second-best year",
        "  [!2022 NO TRADES]  no trades at all in 2022",
        "",
    ]

    for _, row in sa.iterrows():
        tf = row["timeframe"]
        am = row["atr_mult"]
        verdict = row["stability_verdict"]

        # Only show PASS combos at detail level; show brief row for WARN/FAIL
        if verdict != "PASS":
            continue

        lines += [
            f"{'=' * 70}",
            f"  [{verdict}]  Timeframe={tf}  ATR*{am:.1f}  "
            f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}  "
            f"({row['zone_stability_pct']:.0f}%)",
            f"{'=' * 70}",
        ]

        # Each N in the stability zone AND canonical N=20
        for don_n in sorted(set(STABILITY_ZONE_NS + [20])):
            trades_slice = all_trades_df[
                (all_trades_df["timeframe"] == tf) &
                (all_trades_df["atr_mult"]  == am) &
                (all_trades_df["don_n"]     == don_n)
            ]
            if trades_slice.empty:
                continue

            trade_list = trades_slice.to_dict("records")
            s = _stats(trade_list)
            yby = year_by_year(trade_list)
            total_r = s["total_r"]

            zone_tag = " [zone]" if don_n in STABILITY_ZONE_NS else ""
            can_tag  = " [canonical]"         if don_n == 20 else ""
            lines += [
                "",
                f"  N={don_n}  exit={don_n // 2}  "
                f"trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
                f"pf={s['pf']:.2f}  total_r={total_r:+.1f}R"
                f"{zone_tag}{can_tag}",
                "",
                f"  {'Year':>4}  {'Trades':>6}  {'WinRate':>7}  "
                f"{'TotalR':>8}  {'AvgR':>7}  {'PF':>5}  {'MaxDD':>7}",
                f"  {'-' * 56}",
            ]
            for yr in REPORT_YEARS:
                ys = yby[yr]
                if ys["trades"] == 0:
                    lines.append(f"  {yr:>4}  {'--':>6}  {'--':>7}  "
                                 f"{'--':>8}  {'--':>7}  {'--':>5}  {'--':>7}")
                else:
                    pct_of_total = ys["total_r"] / max(abs(total_r), EPS) * 100
                    conc = " <-- CONCENTRATED" if abs(pct_of_total) > 50 else ""
                    lines.append(
                        f"  {yr:>4}  {ys['trades']:>6}  {ys['win_rate_pct']:>6.1f}%  "
                        f"  {ys['total_r']:>+7.1f}R  {ys['avg_r']:>+6.3f}  "
                        f"{ys['pf']:>5.2f}  {ys['max_dd_r']:>+7.2f}"
                        + conc
                    )

            flags = _year_flags(yby, total_r)
            if flags:
                lines += ["", "  ** FLAGS:"]
                for fl in flags:
                    lines.append(f"     {fl}")

    out = OUT_DIR / "phase_t1_donchianshort_futures_year_by_year.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"Year-by-year report: {out}")


# =============================================================================
# MAIN TEXT REPORT
# =============================================================================

def write_report(
    grid: pd.DataFrame,
    sa: pd.DataFrame,
    sym_count: Dict[str, int],
    all_trades_df: pd.DataFrame,
) -> None:
    lines = [
        "PHASE T1 -- DonchianShort Futures Concept Discovery  (expanded universe)",
        "=" * 80,
        "",
        "Entry:   Close < Donchian N-period LOW (shifted 1 bar, no lookahead)",
        "Filter:  Close < EMA200  (mandatory bear regime gate)",
        "Stop:    Entry + ATR(14) * atr_mult  (above entry for short)",
        "Chandelier (SHORT-side mirror of long formula):",
        "  trail_stop = lowest_low_since_entry + ATR_at_entry * atr_mult",
        "  active_stop = min(initial_stop, chandelier_stop)",
        "Exit:    high >= active_stop  OR  close > Donchian(N//2) upper band",
        "",
        "Exchange:       Binance Futures (USD-M perpetuals)",
        f"Timeframes:     {TIMEFRAMES}",
        f"Donchian N:     {DONCHIAN_NS}  (exit = N // 2, upper band)",
        f"ATR mults:      {ATR_MULTS}",
        f"Filter:         {FILTER_MODE}",
        "",
        "Universe:",
        f"  1D :  all symbols with data in {OHLCV_1D.relative_to(ROOT)}",
        f"  4H/6H: symbols_pre2021.csv (58 symbols, bear-cycle coverage)",
        "",
        "Section 4.2 -- Futures cost floor adjustment:",
        f"  Short-side (Futures): avg_r > {COST_FLOOR_R}R  (funding rate overhead)",
        "",
        f"Stability zone:  N in {STABILITY_ZONE_NS}  (PASS if >={STABILITY_PASS_PCT:.0f}% profitable)",
    ]

    for tf in TIMEFRAMES:
        tf_sa   = sa[sa["timeframe"] == tf]
        tf_grid = grid[grid["timeframe"] == tf]
        n_syms  = sym_count.get(tf, 0)

        lines += [
            "",
            "=" * 80,
            f"TIMEFRAME: {tf}  (symbols loaded: {n_syms})",
            "=" * 80,
            "STABILITY RANKING",
            "-" * 50,
        ]

        for _, row in tf_sa.iterrows():
            v = row["stability_verdict"]
            cf = f"  cf={int(row['zone_cost_floor_pass'])}/{int(row['zone_total'])}"
            lines.append(
                f"  [{v:4s}] ATR*{row['atr_mult']:.1f}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])} profitable  "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: trades={row['can_n20_trades']}  "
                f"avg_r={row['can_n20_avg_r']:+.4f}  "
                f"pf={row['can_n20_pf']:.2f}  "
                f"dd={row['can_n20_max_dd_r']:+.2f}  "
                f"win%={row['can_n20_win_pct']:.1f}%"
                + cf
            )

        if not tf_sa.empty:
            best    = tf_sa.iloc[0]
            gate_42 = "PASS" if best["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
            lines += [
                "",
                f"  Best ({tf}):  ATR*{best['atr_mult']:.1f}  "
                f"stability={best['zone_stability_pct']:.0f}%  "
                f"({best['stability_verdict']})  "
                f"Sec4.2={gate_42} ({best['can_n20_avg_r']:.4f}R)",
            ]

        lines += [
            "",
            f"DONCHIAN N SENSITIVITY  ({tf}, {FILTER_MODE})",
            "-" * 50,
        ]
        for am in ATR_MULTS:
            g = tf_grid[tf_grid["atr_mult"] == am].sort_values("don_n")
            lines.append(f"\n  ATR*{am:.1f}")
            for _, row in g.iterrows():
                n    = int(row["don_n"])
                zone = " <- zone" if n in STABILITY_ZONE_NS else ""
                can  = " <- canonical" if n == 20 else ""
                sign = "[+]" if row["pf"] > 1.0 and row["avg_r"] > 0 else "[-]"
                cf   = " (cf PASS)" if row["avg_r"] > COST_FLOOR_R else ""
                lines.append(
                    f"    {sign} N={n:2d}  exit={n // 2:2d}  "
                    f"trades={int(row['trades']):4d}  "
                    f"avg_r={row['avg_r']:+.4f}  "
                    f"pf={row['pf']:.2f}  "
                    f"dd={row['max_dd_r']:+.2f}  "
                    f"win%={row['win_rate_pct']:.1f}%"
                    + cf + zone + can
                )

    # ---- 2022 highlight ----
    lines += [
        "",
        "=" * 80,
        "2022 BEAR MARKET CHECK  (best year for any short system)",
        "=" * 80,
    ]
    if "entry_year" in all_trades_df.columns:
        t2022 = all_trades_df[all_trades_df["entry_year"] == 2022]
        if not t2022.empty:
            for tf in TIMEFRAMES:
                tf_t = t2022[t2022["timeframe"] == tf]
                if tf_t.empty:
                    lines.append(f"  {tf}: no trades in 2022")
                    continue
                lines.append(f"  {tf}: {len(tf_t)} trades in 2022")
                for am in ATR_MULTS:
                    for n in DONCHIAN_NS:
                        sl = tf_t[(tf_t["atr_mult"] == am) & (tf_t["don_n"] == n)]
                        if sl.empty:
                            continue
                        s = _stats(sl.to_dict("records"))
                        zone = " [zone]" if n in STABILITY_ZONE_NS else ""
                        lines.append(
                            f"    ATR*{am:.1f} N={n:2d}: "
                            f"trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
                            f"pf={s['pf']:.2f}  total_r={s['total_r']:+.1f}R"
                            + zone
                        )
        else:
            lines.append("  No trades recorded in 2022 -- check universe coverage.")

    # ---- Overall best ----
    passes = sa[sa["stability_verdict"] == "PASS"]
    warns  = sa[sa["stability_verdict"] == "WARN"]
    best   = (passes.iloc[0] if not passes.empty
              else (warns.iloc[0] if not warns.empty
                    else (sa.iloc[0] if not sa.empty else None)))

    lines += [
        "",
        "=" * 80,
        "OVERALL BEST CANDIDATE",
        "=" * 80,
    ]
    if best is not None:
        gate_42 = "PASS" if best["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
        proceed = (
            "PROCEED TO T2"
            if best["stability_verdict"] in ("PASS", "WARN")
            and best["can_n20_avg_r"] > COST_FLOOR_R
            else "DO NOT PROCEED"
        )
        lines += [
            f"  Timeframe:   {best['timeframe']}",
            f"  ATR mult:    {best['atr_mult']:.1f}",
            f"  Stability:   {best['zone_stability_pct']:.0f}%  ({best['stability_verdict']})",
            f"  Sec 4.2:     {gate_42} ({best['can_n20_avg_r']:.4f}R vs >{COST_FLOOR_R}R)",
            "",
            f"  !! {proceed} -- awaiting human review.",
        ]
    else:
        lines.append("  No candidates found.")

    lines += [
        "",
        "=" * 80,
        "INTERPRETATION",
        "=" * 80,
        "",
        f"  PASS (>={STABILITY_PASS_PCT:.0f}%): Short-side edge robust -- proceed to T2.",
        "  WARN (40-67%): Edge exists but parameter-sensitive.",
        "  FAIL (<40%):  No consistent short edge -- do NOT proceed.",
        "",
        f"  Cost floor: avg_r > {COST_FLOOR_R}R (Futures funding rate adjustment).",
        "  Win rate 25-40% healthy for short trend-following (bearish periods rare).",
        "  IMPORTANT: Bear markets are shorter than bull markets in 4yr+ backtest.",
        "  Fewer trades than long-side expected -- structural, not a failure.",
        "",
        "  !! STOP HERE -- do not proceed to T2 until human review.",
    ]

    out = OUT_DIR / "phase_t1_donchianshort_futures_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    p(f"\nMain report: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    p("=" * 80)
    p("PHASE T1 -- DonchianShort Futures Concept Discovery  (expanded universe)")
    p("=" * 80)
    p(f"Timeframes:      {TIMEFRAMES}")
    p(f"Donchian N grid: {DONCHIAN_NS}  (exit = N // 2, upper band)")
    p(f"ATR mults:       {ATR_MULTS}")
    p(f"Filter:          {FILTER_MODE}")
    p(f"Cost floor:      avg_r > {COST_FLOOR_R}R  (Futures, Section 4.2)")
    p(f"Output:          {OUT_DIR}")
    p()

    # Load universe lists
    pre21_syms = load_universe(SYM_PRE21)
    all_syms   = load_universe(SYM_ALL)

    if not pre21_syms and not all_syms:
        p("[ERROR] No universe files found.")
        return 1

    p(f"Universe: {len(pre21_syms)} pre-2021 symbols  |  "
      f"{len(all_syms)} all symbols")

    # For 4H/6H: use pre-2021 symbols only (download feasibility)
    # For 1D:    use all symbols (already cached)
    intra_syms = pre21_syms if pre21_syms else all_syms
    d1_syms    = all_syms   if all_syms   else pre21_syms

    # Init exchange for intraday downloads
    exchange = None
    if not HAS_CCXT:
        p("[WARN] ccxt not installed -- 4H/6H download disabled, 1D only.")
    else:
        p("Initializing Binance Futures exchange...")
        exchange = _init_exchange()
        if exchange is None:
            p("[WARN] Could not connect to Binance Futures -- 4H/6H download disabled.")

    all_trades:   List[dict] = []
    summary_rows: List[dict] = []
    sym_count:    Dict[str, int] = {}

    total_combos = len(DONCHIAN_NS) * len(ATR_MULTS)

    for tf in TIMEFRAMES:
        p()
        p(f"{'=' * 60}")
        p(f"TIMEFRAME: {tf}")
        p(f"{'=' * 60}")

        # Select universe and data loader for this TF
        if tf == "1d":
            syms_for_tf = d1_syms
        else:
            syms_for_tf = intra_syms
            if exchange is None:
                p(f"  Skipping {tf} -- no exchange connection.")
                continue

        p(f"Loading data for {len(syms_for_tf)} symbols...")
        data: Dict[str, pd.DataFrame] = {}

        for sym in syms_for_tf:
            if tf == "1d":
                df = load_1d(sym)
            else:
                df = load_or_download_intra(sym, tf, exchange)

            if df is not None and len(df) >= MIN_BARS:
                data[sym] = df

        n_loaded = len(data)
        sym_count[tf] = n_loaded
        p(f"Loaded: {n_loaded} / {len(syms_for_tf)} symbols  "
          f"({n_loaded * 100 // max(len(syms_for_tf), 1)}%)")

        if not data:
            p(f"  [WARN] No data for {tf} -- skipping.")
            continue

        # Grid sweep
        p(f"Running grid: {total_combos} combos x {n_loaded} symbols...")
        done = 0
        for don_n in DONCHIAN_NS:
            for am in ATR_MULTS:
                combo_trades: List[dict] = []
                for sym, df in data.items():
                    combo_trades.extend(backtest_symbol(sym, df, don_n, am, tf))

                s = _stats(combo_trades)
                row = dict(
                    timeframe=tf, filter_mode=FILTER_MODE,
                    don_n=don_n, don_exit_n=don_n // 2,
                    atr_mult=am, symbols=n_loaded,
                )
                row.update(s)
                summary_rows.append(row)
                all_trades.extend(combo_trades)
                done += 1

                zone_tag = " [zone]" if don_n in STABILITY_ZONE_NS else ""
                cf_tag   = " [cf PASS]" if s["avg_r"] > COST_FLOOR_R else ""
                p(
                    f"  [{done:2d}/{total_combos}] N={don_n:2d}  ATR*{am:.1f}  "
                    f"trades={s['trades']:4d}  avg_r={s['avg_r']:+.4f}  "
                    f"pf={s['pf']:.2f}  win%={s['win_rate_pct']:.1f}%"
                    + cf_tag + zone_tag
                )

    if not summary_rows:
        p("[ERROR] No trades produced across any timeframe.")
        return 1

    grid = pd.DataFrame(summary_rows)
    sa   = stability_analysis(grid)

    # ---- Save CSVs ----
    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv(
        OUT_DIR / "phase_t1_donchianshort_futures_trades.csv", index=False)
    grid.to_csv(
        OUT_DIR / "phase_t1_donchianshort_futures_summary.csv", index=False)
    sa.to_csv(
        OUT_DIR / "phase_t1_donchianshort_futures_stability.csv", index=False)

    # ---- Reports ----
    write_report(grid, sa, sym_count, all_trades_df)
    write_year_by_year_report(grid, sa, all_trades_df)

    # ---- Console stability summary ----
    p()
    p("=" * 80)
    p("STABILITY RANKING SUMMARY")
    p("=" * 80)
    for tf in TIMEFRAMES:
        tf_sa = sa[sa["timeframe"] == tf]
        if tf_sa.empty:
            continue
        p(f"  [{tf}]")
        for _, row in tf_sa.iterrows():
            v = row["stability_verdict"]
            p(
                f"    [{v:4s}] ATR*{row['atr_mult']:.1f}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}  "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: avg_r={row['can_n20_avg_r']:+.4f}R  "
                f"pf={row['can_n20_pf']:.2f}"
            )

    passes = sa[sa["stability_verdict"] == "PASS"]
    warns  = sa[sa["stability_verdict"] == "WARN"]
    p()
    p(f"PASS combos: {len(passes)}  |  WARN: {len(warns)}")

    # ---- Year-by-year console preview for best combo ----
    if not passes.empty:
        best = passes.iloc[0]
        tf_b, am_b = best["timeframe"], best["atr_mult"]
        p()
        p(f"Best PASS combo: {tf_b}  ATR*{am_b:.1f}")
        for don_n in sorted(set(STABILITY_ZONE_NS)):
            slice_t = all_trades_df[
                (all_trades_df["timeframe"] == tf_b) &
                (all_trades_df["atr_mult"]  == am_b) &
                (all_trades_df["don_n"]     == don_n)
            ]
            if slice_t.empty:
                continue
            yby    = year_by_year(slice_t.to_dict("records"))
            s      = _stats(slice_t.to_dict("records"))
            flags  = _year_flags(yby, s["total_r"])
            p(f"  N={don_n}  total={s['total_r']:+.1f}R  avg={s['avg_r']:+.4f}R")
            for yr in REPORT_YEARS:
                ys = yby[yr]
                if ys["trades"] > 0:
                    fl = (" [CONC]" if any(str(yr) in f and "[CONCENTRATED]" in f
                                          for f in flags) else "")
                    p(f"    {yr}: {ys['trades']:3d} trades  "
                      f"avg_r={ys['avg_r']:+.4f}  "
                      f"total_r={ys['total_r']:+.1f}R"
                      + fl)
            for fl in flags:
                p(f"  !! {fl}")

    p()
    p("=" * 80)
    p("T1 COMPLETE -- awaiting human review before T2")
    p("=" * 80)
    p(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
