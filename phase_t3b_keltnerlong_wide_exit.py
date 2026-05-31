#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T3B -- KeltnerLong Wide-Exit / Chandelier Engineering

Sweeps chandelier stop parameters on the T2 canonical config to maximise
capture of large trending moves.

T2 canonical (locked):
  ema200_price / N=15 / km=3.0 / sm=2.0
  avg_r=+0.417R  PF=1.666  trades=167  avg_MFE=2.763R

Chandelier activation grid:   [1.0, 1.5, 2.0, 2.5, 3.0] R
Chandelier ATR-mult grid:     [2.0, 2.5, 3.0]
Total combos:                 15

Exit priority (in order):
  1. low <= initial_stop                       -> exit at stop
  2. chandelier active AND low <= chan_stop     -> exit at chan_stop
  3. chandelier inactive AND close < EMA(N)    -> midline exit
Chandelier is updated AFTER all exit checks (no lookahead).

Data:    data/research_trend_t15/ohlcv_cache/
Output:  data/research_keltnerlong_t3b/
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG -- T2 canonical (do not change)
# =============================================================================

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_keltnerlong_t3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME     = "1d"
BACKTEST_BARS = 1500

KELTNER_N    = 15
KELTNER_MULT = 3.0
STOP_MULT    = 2.0
FILTER_MODE  = "ema200_price"
EMA200_N     = 200

# T3B chandelier grid
CHAN_ACTIVATE_R_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
CHAN_ATR_MULT_GRID   = [2.0, 2.5, 3.0]

# T2 baseline (for comparison)
T2_AVG_R   = 0.4168
T2_PF      = 1.666
T2_TOTAL_R = 69.61
T2_TRADES  = 167

COST_FLOOR_R = 0.15
MIN_BARS     = 300

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
# INDICATORS  (precomputed once per symbol -- shared across combos)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(KELTNER_N).mean()
    ema_n = d["close"].ewm(span=KELTNER_N, adjust=False).mean()

    d["ema_n"]    = ema_n
    d["atr_n"]    = atr_n
    d["ema_n_s1"] = ema_n.shift(1)
    d["atr_n_s1"] = atr_n.shift(1)
    d["ema200"]   = d["close"].ewm(span=EMA200_N, adjust=False).mean()
    return d


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class Trade:
    trade_id:       int
    symbol:         str
    chan_activate_r: float
    chan_atr_mult:  float
    entry_time:     str
    exit_time:      str
    entry_price:    float
    exit_price:     float
    initial_stop:   float
    initial_risk:   float
    exit_reason:    str
    bars_held:      int
    net_r:          float
    mfe_r:          float
    mae_r:          float
    chan_activated: bool


# =============================================================================
# BACKTESTER  (single symbol, single combo)
# =============================================================================

def backtest_symbol(
    symbol: str,
    ind: pd.DataFrame,
    chan_activate_r: float,
    chan_atr_mult: float,
    first_id: int,
) -> List[Trade]:
    warmup = max(KELTNER_N, EMA200_N) + 5
    pos    = None
    trades = []
    tid    = first_id

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

        if not all(math.isfinite(x) for x in [ema_n, ema_n_s1, atr_n_s1, ema200]):
            continue

        upper_band = ema_n_s1 + atr_n_s1 * KELTNER_MULT

        if pos is not None:
            bars_held = i - pos["entry_i"]

            # Track MFE, MAE, HH (highest high since entry)
            pos["mfe_r"] = max(pos["mfe_r"], (high - pos["entry"]) / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (low  - pos["entry"]) / pos["risk"])
            pos["hh"]    = max(pos["hh"], high)

            # Determine effective stop (initial or chandelier, whichever is higher)
            if pos["chan_active"]:
                eff_stop = max(pos["stop"], pos["chan_stop"])
            else:
                eff_stop = pos["stop"]

            exit_px = None
            reason  = None

            # Exit 1: initial stop (always checked first)
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                reason  = "initial_stop"
            # Exit 2: chandelier stop
            elif pos["chan_active"] and low <= pos["chan_stop"]:
                exit_px = pos["chan_stop"]
                reason  = "chandelier_stop"
            # Exit 3: midline cross-under (only when chandelier inactive)
            elif not pos["chan_active"] and close < ema_n:
                exit_px = close
                reason  = "midline_exit"

            if exit_px is not None:
                net_r = (float(exit_px) - pos["entry"]) / pos["risk"]
                trades.append(Trade(
                    trade_id=tid,
                    symbol=symbol,
                    chan_activate_r=chan_activate_r,
                    chan_atr_mult=chan_atr_mult,
                    entry_time=str(pos["entry_time"]),
                    exit_time=str(t),
                    entry_price=float(pos["entry"]),
                    exit_price=float(exit_px),
                    initial_stop=float(pos["stop"]),
                    initial_risk=float(pos["risk"]),
                    exit_reason=reason,
                    bars_held=int(bars_held),
                    net_r=float(net_r),
                    mfe_r=float(pos["mfe_r"]),
                    mae_r=float(pos["mae_r"]),
                    chan_activated=bool(pos["chan_active"]),
                ))
                tid += 1
                pos = None
                continue

            # Update chandelier AFTER all exit checks
            if pos["mfe_r"] >= chan_activate_r:
                pos["chan_active"] = True
            if pos["chan_active"]:
                new_cs = pos["hh"] - atr_n * chan_atr_mult
                pos["chan_stop"] = max(pos["chan_stop"], new_cs)

        if pos is None:
            # Filter: close > EMA200
            if close <= ema200:
                continue
            # Entry: close breaks above Keltner upper band (previous bar)
            if close > upper_band:
                risk = atr_n * STOP_MULT
                if risk > 0:
                    pos = dict(
                        entry_time=t,
                        entry_i=i,
                        entry=close,
                        stop=close - risk,
                        risk=risk,
                        hh=close,
                        mfe_r=0.0,
                        mae_r=0.0,
                        chan_active=False,
                        chan_stop=close - risk,  # initialise at initial stop level
                    )

    return trades


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


def _stats(trades: List[Trade]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, avg_mfe_r=0.0)
    r = np.array([t.net_r for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, avg_mfe_r=0.0)
    mfe = np.array([t.mfe_r for t in trades], dtype=float)
    return dict(
        trades=int(len(r)),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        pf=_pf(r),
        max_dd_r=_dd(r),
        win_rate_pct=float(100.0 * (r > 0).mean()),
        avg_mfe_r=float(mfe.mean()),
        chan_activated_pct=float(100.0 * sum(t.chan_activated for t in trades) / len(trades)),
    )


def build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in trades_df.groupby("symbol"):
        r = g["net_r"].values
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        rows.append(dict(
            symbol=sym,
            trades=int(len(r)),
            total_r=float(r.sum()),
            avg_r=float(r.mean()),
            win_rate_pct=float(100.0 * (r > 0).mean()),
            profit_factor=_pf(r),
            max_dd_r=_dd(r),
            avg_mfe_r=float(g["mfe_r"].mean()),
        ))
    return pd.DataFrame(rows).sort_values("total_r", ascending=False).reset_index(drop=True)


def build_equity(trades_df: pd.DataFrame) -> pd.DataFrame:
    eq = trades_df.copy()
    eq["exit_time"] = pd.to_datetime(eq["exit_time"], utc=True, format="mixed")
    eq = eq.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)
    eq["equity_r"] = eq["net_r"].cumsum()
    eq["peak_r"]   = eq["equity_r"].cummax()
    eq["dd_r"]     = eq["equity_r"] - eq["peak_r"]
    return eq[["exit_time", "trade_id", "symbol",
               "net_r", "equity_r", "peak_r", "dd_r",
               "exit_reason", "chan_activated"]]


# =============================================================================
# REPORT
# =============================================================================

def write_report(variant_df: pd.DataFrame, best: pd.Series,
                 n_symbols: int) -> None:
    total_combos = len(CHAN_ACTIVATE_R_GRID) * len(CHAN_ATR_MULT_GRID)
    lines = [
        "PHASE T3B -- KeltnerLong Wide-Exit / Chandelier Engineering",
        "=" * 80,
        "",
        "T2 canonical:   ema200_price / N=15 / km=3.0 / sm=2.0",
        f"T2 baseline:    trades={T2_TRADES}  avg_r={T2_AVG_R:+.4f}R"
        f"  PF={T2_PF:.3f}  total_r={T2_TOTAL_R:+.2f}R",
        "",
        f"Chan activate grid: {CHAN_ACTIVATE_R_GRID}",
        f"Chan ATR-mult grid: {CHAN_ATR_MULT_GRID}",
        f"Total combos:       {total_combos}",
        f"Symbols:            {n_symbols}",
        "",
        "Exit priority:",
        "  1. low <= initial_stop",
        "  2. chandelier active AND low <= chan_stop",
        "  3. chandelier inactive AND close < EMA(N)  (midline)",
        "  Chandelier updated AFTER exit checks.",
        "",
        "=" * 80,
        "VARIANT RANKING  (sorted by total_r)",
        "=" * 80,
    ]

    for _, row in variant_df.iterrows():
        vs_t2_avg = row["avg_r"] - T2_AVG_R
        vs_t2_pf  = row["pf"]   - T2_PF
        tag = ""
        if row["avg_r"] > T2_AVG_R and row["pf"] > T2_PF:
            tag = "  [BETTER THAN T2]"
        elif row["avg_r"] > T2_AVG_R or row["pf"] > T2_PF:
            tag = "  [PARTIAL IMPROVEMENT]"
        lines.append(
            f"  act={row['chan_activate_r']:.1f}R  cm={row['chan_atr_mult']:.1f}"
            f"  trades={int(row['trades']):4d}"
            f"  total_r={row['total_r']:+7.2f}R"
            f"  avg_r={row['avg_r']:+.4f}R ({vs_t2_avg:+.4f} vs T2)"
            f"  pf={row['pf']:.3f} ({vs_t2_pf:+.3f} vs T2)"
            f"  win%={row['win_rate_pct']:.1f}%"
            f"  chan_act%={row['chan_activated_pct']:.0f}%"
            + tag
        )

    # Best combo detail
    vs_avg = best["avg_r"] - T2_AVG_R
    vs_pf  = best["pf"]    - T2_PF
    vs_tot = best["total_r"] - T2_TOTAL_R
    gate   = "PASS" if best["avg_r"] > COST_FLOOR_R else "FAIL"
    verdict = (
        "PROCEED TO T4" if best["avg_r"] > T2_AVG_R else "REVIEW BEFORE T4"
    )

    lines += [
        "",
        "-" * 80,
        "BEST COMBO (highest total_r)",
        "-" * 80,
        f"  Chan activation:  {best['chan_activate_r']:.1f}R",
        f"  Chan ATR mult:    {best['chan_atr_mult']:.1f}",
        f"  Trades:           {int(best['trades'])}",
        f"  Total R:          {best['total_r']:+.2f}R  ({vs_tot:+.2f}R vs T2)",
        f"  Avg R/trade:      {best['avg_r']:+.4f}R  ({vs_avg:+.4f}R vs T2)",
        f"  Profit factor:    {best['pf']:.3f}  ({vs_pf:+.3f} vs T2)",
        f"  Win rate:         {best['win_rate_pct']:.1f}%",
        f"  Max DD (R):       {best['max_dd_r']:+.2f}R",
        f"  Avg MFE:          {best['avg_mfe_r']:+.3f}R",
        f"  Chan activated %: {best['chan_activated_pct']:.1f}%",
        f"  Sec 4.2 gate:     {gate} ({best['avg_r']:.4f}R > {COST_FLOOR_R}R)",
        "",
        f"  !! {verdict} -- awaiting human review.",
    ]

    lines += [
        "",
        "=" * 80,
        "T3B INTERPRETATION",
        "=" * 80,
        "",
        "  Goal: chandelier stop should capture more of large trending moves",
        "  vs. the midline cross-under exit used in T2.",
        "",
        "  Lower activation R = more trades get chandelier (wider net)",
        "  Higher activation R = only strong trends get chandelier (tighter quality filter)",
        "  Lower chan_mult = tighter trailing stop (captures less but gives less back)",
        "  Higher chan_mult = looser trailing stop (captures more, but gives more back)",
        "",
        "  PROCEED TO T4 if best combo avg_r > T2 baseline.",
        "  DO NOT PROCEED if all combos regress vs T2.",
    ]

    path = OUT_DIR / "phase_t3b_keltnerlong_stability_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    total_combos = len(CHAN_ACTIVATE_R_GRID) * len(CHAN_ATR_MULT_GRID)
    print("=" * 80)
    print("PHASE T3B -- KeltnerLong Wide-Exit / Chandelier Engineering")
    print("=" * 80)
    print(f"T2 canonical:      ema200_price / N={KELTNER_N} / km={KELTNER_MULT} / sm={STOP_MULT}")
    print(f"T2 baseline:       avg_r={T2_AVG_R:+.4f}R  PF={T2_PF:.3f}  total_r={T2_TOTAL_R:+.2f}R")
    print(f"Chan activate grid: {CHAN_ACTIVATE_R_GRID}")
    print(f"Chan ATR-mult grid: {CHAN_ATR_MULT_GRID}")
    print(f"Total combos:      {total_combos}")
    print(f"Cache:             {CACHE_DIR}")
    print()

    symbols = discover_symbols()
    if not symbols:
        print(f"[ERROR] No symbols in cache: {CACHE_DIR}")
        return 1

    print(f"Loading OHLCV data ({len(symbols)} symbols found)...")
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

    # Precompute indicators once per symbol
    print("\nPrecomputing indicators...")
    ind_cache: Dict[str, pd.DataFrame] = {
        sym: add_indicators(df) for sym, df in data.items()
    }
    print(f"  {len(ind_cache)} symbols ready")

    # Grid sweep
    print(f"\nRunning {total_combos} combos x {len(data)} symbols...")
    variant_rows: List[dict] = []
    best_trades:  Optional[List[Trade]] = None
    best_total_r: float = float("-inf")
    done = 0

    for act_r in CHAN_ACTIVATE_R_GRID:
        for cm in CHAN_ATR_MULT_GRID:
            all_trades: List[Trade] = []
            tid = 1
            for sym, ind in ind_cache.items():
                trades = backtest_symbol(sym, ind, act_r, cm, first_id=tid)
                if trades:
                    tid = max(t.trade_id for t in trades) + 1
                all_trades.extend(trades)

            s = _stats(all_trades)
            row = dict(chan_activate_r=act_r, chan_atr_mult=cm)
            row.update(s)
            variant_rows.append(row)
            done += 1

            vs_avg = s["avg_r"] - T2_AVG_R
            tag    = " (*)" if s["total_r"] > T2_TOTAL_R else ""
            print(
                f"  [{done:2d}/{total_combos}] act={act_r:.1f}R cm={cm:.1f}"
                f"  trades={s['trades']:4d}"
                f"  total_r={s['total_r']:+7.2f}R"
                f"  avg_r={s['avg_r']:+.4f}R ({vs_avg:+.4f})"
                f"  pf={s['pf']:.3f}"
                f"  win%={s['win_rate_pct']:.1f}%"
                f"  chan%={s.get('chan_activated_pct', 0):.0f}%"
                + tag
            )

            if s["total_r"] > best_total_r:
                best_total_r = s["total_r"]
                best_trades  = list(all_trades)

    # Save variant summary
    variant_df = pd.DataFrame(variant_rows).sort_values("total_r", ascending=False)
    variant_df.to_csv(OUT_DIR / "phase_t3b_keltnerlong_variant_summary.csv", index=False)
    best_row = variant_df.iloc[0]

    # Save best-combo outputs
    if best_trades:
        best_df = pd.DataFrame([asdict(t) for t in best_trades])
        best_df["entry_time"] = pd.to_datetime(best_df["entry_time"], utc=True, format="mixed")
        best_df["exit_time"]  = pd.to_datetime(best_df["exit_time"],  utc=True, format="mixed")
        best_df = best_df.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)

        best_df.to_csv(OUT_DIR / "phase_t3b_keltnerlong_trades.csv", index=False)
        build_equity(best_df).to_csv(OUT_DIR / "phase_t3b_keltnerlong_equity.csv", index=False)
        build_asset_summary(best_df).to_csv(
            OUT_DIR / "phase_t3b_keltnerlong_asset_summary.csv", index=False)

    write_report(variant_df, best_row, len(data))

    # Console summary
    print()
    print("=" * 80)
    print("T3B SUMMARY")
    print("=" * 80)
    print(f"  T2 baseline:  trades={T2_TRADES}  avg_r={T2_AVG_R:+.4f}R"
          f"  PF={T2_PF:.3f}  total_r={T2_TOTAL_R:+.2f}R")
    print()
    print("  Top 5 combos by total_r:")
    for _, row in variant_df.head(5).iterrows():
        print(
            f"    act={row['chan_activate_r']:.1f}R  cm={row['chan_atr_mult']:.1f}"
            f"  total_r={row['total_r']:+7.2f}R"
            f"  avg_r={row['avg_r']:+.4f}R"
            f"  pf={row['pf']:.3f}"
            f"  win%={row['win_rate_pct']:.1f}%"
        )
    print()
    best = variant_df.iloc[0]
    vs   = "BETTER" if best["avg_r"] > T2_AVG_R else "WORSE"
    print(f"  Best combo: act={best['chan_activate_r']:.1f}R  cm={best['chan_atr_mult']:.1f}"
          f"  -> avg_r={best['avg_r']:+.4f}R  {vs} than T2")
    print()
    print("=" * 80)
    print("T3B COMPLETE -- awaiting human review before T4")
    print("=" * 80)
    print(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
