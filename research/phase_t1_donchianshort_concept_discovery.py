#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- DonchianShort Concept Discovery  (multi-timeframe: 4h, 6h, 1d)

Short-side mirror of the long-side Donchian T1.
Validates edge on Binance Futures SHORT trades across the universe.

Entry:   Close breaks BELOW Donchian N-period LOW (shifted, no lookahead)
Filter:  Close < EMA200 (bear regime gate -- mirror of long-side ema200_price)
Stop:    Entry_price + ATR(14) * atr_mult  (ABOVE entry for short)
Exit 1:  High >= stop  -> exit at stop  (stop hit)
Exit 2:  Close > Donchian_high(N//2)  -> exit at close  (upper half-band exit)

NOTE: Chandelier trailing stop is NOT used in T1.  It is introduced in T3B.
T1 exit = simple initial stop OR Donchian N//2 upper band.

Futures cost adjustment:
  Long-side cost floor (Spot):    avg_r > 0.15R
  Short-side cost floor (Futures): avg_r > 0.25R  (funding rate +0.05-0.15R added)
  See Section 13B of pipeline document.

Data source:
  1d: data/research_trend_t15/ohlcv_cache/       (reuse long-side cache)
  4h: data/research_donchianshort_t1/ohlcv_cache/ (downloaded here if missing)
  6h: data/research_donchianshort_t1/ohlcv_cache/ (downloaded here if missing)

Output: data/research_donchianshort_t1/
"""

from __future__ import annotations

import math
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("Install ccxt:  pip install ccxt")

# =============================================================================
# CONFIG
# =============================================================================

ROOT          = Path(__file__).resolve().parents[1]
CACHE_1D      = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
CACHE_INTRA   = ROOT / "data" / "research_donchianshort_t1" / "ohlcv_cache"
OUT_DIR       = ROOT / "data" / "research_donchianshort_t1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INTRA.mkdir(parents=True, exist_ok=True)

# Timeframes to test -- 1d reuses long-side cache; 4h/6h are downloaded
TIMEFRAMES = ["4h", "6h", "1d"]

# Target bars per timeframe (approx 4 years of data)
# 1d: 1500  4h: 4yr * 365 * 6 = 8760  6h: 4yr * 365 * 4 = 5840
TF_TARGET_BARS = {"1d": 1500, "4h": 8760, "6h": 5840}
TF_BAR_MS      = {"1d": 86_400_000, "4h": 14_400_000, "6h": 21_600_000}

DONCHIAN_NS       = [10, 15, 20, 25, 30, 40, 55]
ATR_MULTS         = [2.0, 3.0]
FILTER_MODE       = "ema200_price_below"
ATR_N             = 14
EMA_REGIME_N      = 200

# Stability zone -- same as long-side
STABILITY_ZONE_NS  = [15, 20, 25]
STABILITY_PASS_PCT = 67.0

# Futures cost floor (Section 13B)
COST_FLOOR_R = 0.25

# Minimum bars required after loading (warmup + meaningful history)
MIN_BARS = 500

# Universe exclusions (same as long-side)
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI", "WBTC", "BTTC",
}
EXCLUDE_KW = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")


# =============================================================================
# EXCHANGE / DOWNLOAD
# =============================================================================

def _init_exchange() -> "ccxt.binance":
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })


def _stem_to_symbol(stem: str, tf: str) -> Optional[str]:
    """BTC_USDT_1d -> BTC/USDT (after stripping tf suffix)."""
    parts = stem.split("_")
    if parts and parts[-1] == tf:
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    quote = parts[-1]
    base  = "_".join(parts[:-1])
    if quote != "USDT":
        return None
    if base in EXCLUDE_BASES:
        return None
    if any(kw in base for kw in EXCLUDE_KW):
        return None
    return f"{base}/{quote}"


def discover_symbols_1d() -> List[str]:
    syms = []
    for f in sorted(CACHE_1D.glob("*_1d.csv")):
        sym = _stem_to_symbol(f.stem, "1d")
        if sym:
            syms.append(sym)
    return syms


def _fetch_paginated(exchange: "ccxt.binance", symbol: str, tf: str,
                     target_bars: int) -> Optional[pd.DataFrame]:
    """Fetch up to target_bars of OHLCV history via paginated requests."""
    bar_ms    = TF_BAR_MS[tf]
    since_ms  = int(time_mod.time() * 1000) - target_bars * bar_ms
    all_rows: List[list] = []
    max_pages = (target_bars // 1000) + 3

    for _ in range(max_pages):
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=1000)
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

    df = pd.DataFrame(all_rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def load_or_download(symbol: str, tf: str,
                     exchange: Optional["ccxt.binance"] = None) -> Optional[pd.DataFrame]:
    """Load from cache; download if missing (sub-daily only)."""
    fname = symbol.replace("/", "_") + f"_{tf}.csv"

    if tf == "1d":
        path = CACHE_1D / fname
    else:
        path = CACHE_INTRA / fname

    # Try cache first
    if path.exists():
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.set_index("timestamp").sort_index()
            else:
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df = df.sort_index()
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.dropna()
            # Accept cache only if it covers enough history (>= 2 years equivalent)
            min_cached = TF_TARGET_BARS[tf] // 2
            if len(df) >= min_cached:
                return df.tail(TF_TARGET_BARS[tf])
        except Exception:
            pass

    # Download if sub-daily and exchange provided
    if tf == "1d" or exchange is None:
        return None

    df = _fetch_paginated(exchange, symbol, tf, TF_TARGET_BARS[tf])
    if df is None or len(df) < MIN_BARS:
        return None

    # Save to cache
    try:
        df.to_csv(path)
    except Exception:
        pass

    return df.tail(TF_TARGET_BARS[tf])


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame, don_n: int) -> pd.DataFrame:
    d = df.copy()
    don_exit = max(don_n // 2, 5)

    # ATR (Wilder simple rolling, consistent with long-side)
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # Donchian bands (shifted by 1 bar -- no lookahead)
    d["don_low"]  = d["low"].shift(1).rolling(don_n).min()   # entry band (break below)
    d["don_high"] = d["high"].shift(1).rolling(don_n).max()  # reference only

    # Exit band: Donchian upper N//2 (mirror of long-side lower N//2 exit)
    d["don_high_exit"] = d["high"].shift(1).rolling(don_exit).max()

    # EMA200 regime filter
    d["ema200"] = d["close"].ewm(span=EMA_REGIME_N, adjust=False).mean()

    return d, don_exit


# =============================================================================
# BACKTESTER (single symbol, SHORT-side)
# =============================================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    don_n: int,
    atr_mult: float,
    tf: str,
) -> List[dict]:
    d, don_exit = add_indicators(df, don_n)

    warmup = max(don_n, don_exit, EMA_REGIME_N, ATR_N) + 5

    pos    = None
    trades = []

    for i in range(warmup, len(d)):
        row   = d.iloc[i]
        atr   = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0:
            continue

        close        = float(row["close"])
        high         = float(row["high"])
        low          = float(row["low"])
        t            = d.index[i]
        don_low      = row.get("don_low")
        don_high_ex  = row.get("don_high_exit")
        ema200       = row.get("ema200")

        if not all(math.isfinite(float(x)) for x in [don_low, don_high_ex, ema200]):
            continue

        if pos is not None:
            # --- EXIT CHECKS ---

            # Exit 1: stop hit (high >= stop for shorts)
            if high >= pos["stop"]:
                net_r = (pos["entry"] - pos["stop"]) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, pos["stop"], net_r, "stop",
                    don_n, don_exit, atr_mult, tf,
                ))
                pos = None
                continue

            # Exit 2: Donchian N//2 upper band (close back above half-period high)
            if close > float(don_high_ex):
                net_r = (pos["entry"] - close) / pos["risk"]
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, "donchian_exit",
                    don_n, don_exit, atr_mult, tf,
                ))
                pos = None
                continue

        if pos is None:
            # --- ENTRY ---
            # Filter: close < EMA200 (bear regime)
            bear_regime = close < float(ema200)
            # Signal: close breaks below Donchian N-period low
            short_signal = close < float(don_low)

            if bear_regime and short_signal:
                risk = atr * atr_mult
                if risk > 0:
                    pos = {
                        "entry_time": t,
                        "entry":      close,
                        "stop":       close + risk,   # stop ABOVE entry for short
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
    don_n: int,
    don_exit: int,
    atr_mult: float,
    tf: str,
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
# STABILITY ANALYSIS
# =============================================================================

def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, am), g in grid.groupby(["timeframe", "atr_mult"]):
        zone          = g[g["don_n"].isin(STABILITY_ZONE_NS)]
        total_zone    = len(zone)
        pf_gt1        = int((zone["pf"] > 1.0).sum())
        avg_r_pos     = int((zone["avg_r"] > 0).sum())
        cost_floor    = int((zone["avg_r"] > COST_FLOOR_R).sum())
        pct           = 100.0 * pf_gt1 / max(total_zone, 1)

        can20 = g[g["don_n"] == 20]
        can_trades   = int(can20["trades"].sum())   if not can20.empty else 0
        can_avg_r    = float(can20["avg_r"].mean())  if not can20.empty else 0.0
        can_pf       = float(can20["pf"].mean())     if not can20.empty else 0.0
        can_dd       = float(can20["max_dd_r"].mean()) if not can20.empty else 0.0
        can_win      = float(can20["win_rate_pct"].mean()) if not can20.empty else 0.0

        rows.append(dict(
            timeframe=tf,
            filter_mode=FILTER_MODE,
            atr_mult=am,
            zone_total=total_zone,
            zone_pf_gt1=pf_gt1,
            zone_avg_r_positive=avg_r_pos,
            zone_cost_floor_pass=cost_floor,
            zone_stability_pct=round(pct, 1),
            stability_verdict=(
                "PASS" if pct >= STABILITY_PASS_PCT else
                ("WARN" if pct >= 40 else "FAIL")
            ),
            can_n20_trades=can_trades,
            can_n20_avg_r=round(can_avg_r, 4),
            can_n20_pf=round(can_pf, 3),
            can_n20_max_dd_r=round(can_dd, 2),
            can_n20_win_pct=round(can_win, 1),
        ))

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct", "can_n20_pf"], ascending=False
    ).reset_index(drop=True)


# =============================================================================
# REPORT
# =============================================================================

def write_report(grid: pd.DataFrame, sa: pd.DataFrame,
                 sym_count: Dict[str, int]) -> None:
    lines = [
        "PHASE T1 -- DonchianShort Concept Discovery  (multi-timeframe)",
        "=" * 80,
        "",
        "Short-side mirror of long-side Donchian T1.",
        "Entry:   Close < Donchian N-period LOW (shifted 1 bar, no lookahead)",
        "Filter:  Close < EMA200  (bear regime gate)",
        "Stop:    Entry + ATR(14) * atr_mult  (above entry for short)",
        "Exit:    High >= stop  OR  Close > Donchian(N//2) upper band",
        "",
        "Exchange:        Binance Futures (USD-M perpetuals)",
        f"Timeframes:      {TIMEFRAMES}",
        f"Donchian N grid: {DONCHIAN_NS}",
        f"ATR mults:       {ATR_MULTS}",
        f"Filter:          {FILTER_MODE}",
        "",
        "Section 13B -- Futures cost floor adjustment:",
        "  Long-side (Spot):     avg_r > 0.15R",
        f"  Short-side (Futures): avg_r > {COST_FLOOR_R}R  (funding rate ~0.05-0.15R overhead)",
        "",
        f"Stability zone:  N in {STABILITY_ZONE_NS}  (PASS if >={STABILITY_PASS_PCT:.0f}% profitable)",
    ]

    # ---- Per-timeframe sections ----
    for tf in TIMEFRAMES:
        tf_sa   = sa[sa["timeframe"] == tf]
        tf_grid = grid[grid["timeframe"] == tf]
        n_syms  = sym_count.get(tf, 0)
        target  = TF_TARGET_BARS[tf]
        yr_note = f"~4yr on {tf}"

        lines += [
            "",
            "=" * 80,
            f"TIMEFRAME: {tf}  (symbols={n_syms}, target bars={target} {yr_note})",
            "=" * 80,
            "STABILITY RANKING",
            "-" * 40,
        ]

        for _, row in tf_sa.iterrows():
            v = row["stability_verdict"]
            cf_note = f"  cost_floor_pass={int(row['zone_cost_floor_pass'])}/{int(row['zone_total'])}"
            lines.append(
                f"  [{v:4s}] ATR*{row['atr_mult']:.1f}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])} profitable "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: trades={row['can_n20_trades']}  "
                f"avg_r={row['can_n20_avg_r']:+.4f}  "
                f"pf={row['can_n20_pf']:.2f}  "
                f"dd={row['can_n20_max_dd_r']:+.2f}  "
                f"win%={row['can_n20_win_pct']:.1f}%"
                + cf_note
            )

        if not tf_sa.empty:
            best     = tf_sa.iloc[0]
            gate_42  = "PASS" if best["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
            lines += [
                "",
                "  Best candidate for this TF:",
                f"    Timeframe:     {tf}",
                f"    ATR mult:      {best['atr_mult']:.1f}",
                f"    Stability:     {best['zone_stability_pct']:.0f}%  ({best['stability_verdict']})",
                f"    Sec 4.2 gate:  {gate_42} ({best['can_n20_avg_r']:.4f}R vs >{COST_FLOOR_R}R)",
            ]

        lines += [
            "",
            f"DONCHIAN N SENSITIVITY (SHORT-SIDE, {tf}, ema200_price_below)",
            "-" * 40,
        ]
        for am in ATR_MULTS:
            g = tf_grid[tf_grid["atr_mult"] == am].sort_values("don_n")
            lines.append(f"\n  ATR*{am:.1f}")
            for _, row in g.iterrows():
                n    = int(row["don_n"])
                zone = " <- STABILITY ZONE" if n in STABILITY_ZONE_NS else ""
                can  = " <- CANONICAL"      if n == 20 else ""
                sign = "[+]" if row["pf"] > 1.0 and row["avg_r"] > 0 else "[-]"
                cf   = " (cost_floor PASS)" if row["avg_r"] > COST_FLOOR_R else ""
                lines.append(
                    f"    {sign} N={n:2d}  exit={n//2:2d}  trades={int(row['trades']):4d}  "
                    f"avg_r={row['avg_r']:+.3f}  pf={row['pf']:.2f}  "
                    f"dd={row['max_dd_r']:+.2f}  win%={row['win_rate_pct']:.1f}%"
                    + cf + zone + can
                )

    # ---- Overall best pick ----
    passes = sa[sa["stability_verdict"] == "PASS"]
    warns  = sa[sa["stability_verdict"] == "WARN"]
    best_overall = passes.iloc[0] if not passes.empty else (
        warns.iloc[0] if not warns.empty else (sa.iloc[0] if not sa.empty else None)
    )

    lines += [
        "",
        "=" * 80,
        "OVERALL BEST CANDIDATE (across all timeframes)",
        "=" * 80,
    ]
    if best_overall is not None:
        gate_42 = "PASS" if best_overall["can_n20_avg_r"] > COST_FLOOR_R else "FAIL"
        proceed = (
            "PROCEED TO T2" if best_overall["stability_verdict"] in ("PASS", "WARN")
            and best_overall["can_n20_avg_r"] > COST_FLOOR_R
            else "DO NOT PROCEED"
        )
        lines += [
            f"  Timeframe:     {best_overall['timeframe']}",
            f"  ATR mult:      {best_overall['atr_mult']:.1f}",
            f"  Stability:     {best_overall['zone_stability_pct']:.0f}%  ({best_overall['stability_verdict']})",
            f"  Sec 4.2 gate:  {gate_42} ({best_overall['can_n20_avg_r']:.4f}R)",
            "",
            f"  !! {proceed} -- awaiting human review.",
        ]
    else:
        lines.append("  No candidates found.")

    lines += [
        "",
        "=" * 80,
        "INTERPRETATION (Section 13B)",
        "=" * 80,
        "",
        "  PASS (>=67%): Short-side edge is robust -- proceed to T2.",
        "  WARN (40-67%): Edge exists but parameter-sensitive.",
        "  FAIL (<40%):  No consistent short-side edge.  Do NOT proceed.",
        "",
        f"  Cost floor: avg_r > {COST_FLOOR_R}R (Futures funding rate adjustment).",
        "  Win rate 30-45% healthy for trend-following.",
        "  IMPORTANT: Bear markets are shorter than bull markets in 4yr backtest.",
        "  Fewer trades expected vs long-side -- this is structural, not a failure.",
        "",
        "  !! STOP HERE -- do not proceed to T2 until human review.",
    ]

    report_path = OUT_DIR / "phase_t1_donchianshort_stability_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 80)
    print("PHASE T1 -- DonchianShort Concept Discovery  (multi-timeframe)")
    print("=" * 80)
    print(f"Timeframes:      {TIMEFRAMES}")
    print(f"Donchian N grid: {DONCHIAN_NS}  (exit = N // 2, upper band)")
    print(f"ATR mults:       {ATR_MULTS}")
    print(f"Filter:          {FILTER_MODE}  (close < EMA200)")
    print(f"Cost floor:      avg_r > {COST_FLOOR_R}R  (Futures, Section 13B)")
    print(f"1d cache:        {CACHE_1D}")
    print(f"Intra cache:     {CACHE_INTRA}")
    print()

    # Symbol universe: derive from 1d cache (same for all TFs)
    all_symbols = discover_symbols_1d()
    if not all_symbols:
        print(f"[ERROR] No symbols found in 1d cache: {CACHE_1D}")
        return 1
    print(f"Universe: {len(all_symbols)} symbols (from 1d cache)")

    # Init exchange for 4h/6h downloads (no API key required for public OHLCV)
    exchange = _init_exchange()

    all_trades:   List[dict] = []
    summary_rows: List[dict] = []
    sym_count:    Dict[str, int] = {}

    total_combos = len(DONCHIAN_NS) * len(ATR_MULTS)

    for tf in TIMEFRAMES:
        print()
        print(f"--- Timeframe: {tf} ---")
        target_bars = TF_TARGET_BARS[tf]

        # Load / download data for this TF
        data: Dict[str, pd.DataFrame] = {}
        print(f"Loading data (target {target_bars} bars per symbol)...")
        for sym in all_symbols:
            df = load_or_download(sym, tf, exchange if tf != "1d" else None)
            if df is not None and len(df) >= MIN_BARS:
                data[sym] = df
            else:
                if tf != "1d":
                    print(f"  SKIP {sym} ({tf})")

        n_loaded = len(data)
        sym_count[tf] = n_loaded
        print(f"Loaded: {n_loaded} / {len(all_symbols)} symbols")
        if not data:
            print(f"  [WARN] No data for {tf} -- skipping.")
            continue

        # Grid sweep for this TF
        done = 0
        print(f"Running grid: {total_combos} combos x {n_loaded} symbols")
        for don_n in DONCHIAN_NS:
            for am in ATR_MULTS:
                combo_trades: List[dict] = []
                for sym, df in data.items():
                    combo_trades.extend(backtest_symbol(sym, df, don_n, am, tf))

                s = _stats(combo_trades)
                row = dict(
                    timeframe=tf,
                    filter_mode=FILTER_MODE,
                    don_n=don_n,
                    don_exit_n=don_n // 2,
                    atr_mult=am,
                    symbols=n_loaded,
                )
                row.update(s)
                summary_rows.append(row)
                all_trades.extend(combo_trades)
                done += 1

                cf_tag = " [cost_floor PASS]" if s["avg_r"] > COST_FLOOR_R else ""
                print(
                    f"  [{done:2d}/{total_combos}] N={don_n:2d} exit={don_n//2:2d} "
                    f"ATR*{am:.1f}  trades={s['trades']:4d}  "
                    f"avg_r={s['avg_r']:+.3f}  pf={s['pf']:.2f}  "
                    f"win%={s['win_rate_pct']:.1f}%"
                    + cf_tag
                )

    if not summary_rows:
        print("[ERROR] No trades produced across any timeframe.")
        return 1

    grid = pd.DataFrame(summary_rows)
    sa   = stability_analysis(grid)

    # Save outputs
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(OUT_DIR / "phase_t1_donchianshort_trades.csv", index=False)
    grid.to_csv(OUT_DIR / "phase_t1_donchianshort_summary.csv", index=False)
    sa.to_csv(OUT_DIR / "phase_t1_donchianshort_stability.csv", index=False)

    write_report(grid, sa, sym_count)

    # Console summary
    print()
    print("=" * 80)
    print("STABILITY RANKING SUMMARY")
    print("=" * 80)
    for tf in TIMEFRAMES:
        tf_sa = sa[sa["timeframe"] == tf]
        if tf_sa.empty:
            continue
        print(f"  [{tf}]")
        for _, row in tf_sa.iterrows():
            v = row["stability_verdict"]
            print(
                f"    [{v:4s}] ATR*{row['atr_mult']:.1f}  "
                f"zone={int(row['zone_pf_gt1'])}/{int(row['zone_total'])}  "
                f"({row['zone_stability_pct']:.0f}%)  "
                f"N=20: avg_r={row['can_n20_avg_r']:+.4f}R  "
                f"pf={row['can_n20_pf']:.2f}"
            )

    print()
    print("=" * 80)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 80)
    print(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
