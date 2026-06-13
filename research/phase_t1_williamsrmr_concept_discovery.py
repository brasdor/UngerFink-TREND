#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- Williams %R MR Long Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology

Williams %R = (highest_high(N) - close) / (highest_high(N) - lowest_low(N)) * -100
Range: 0 (most overbought) to -100 (most oversold)
Oversold condition: WR < oversold_thresh  (e.g. WR < -80)

Entry : WR(N) < oversold_thresh AND not in position --> enter long at bar close
Exit  : fixed time exit after hold_bars (primary, same as RSI MR)
Safety: ATR x atr_mult stop below entry (backstop only)
Filter: none | ema200_price_above

Parameter grid:
  wr_n          : [5, 7, 10, 14, 20]
  oversold_thresh: [-70, -75, -80, -85, -90]  (more negative = more oversold)
  hold_bars     : [10, 15, 20, 25]
  atr_mult      : [2.0, 3.0]
  filter_mode   : [none, ema200_price_above]
  timeframe     : 1d and 2h (run both, report separately)

Stability zone:
  wr_n +/- 1 step in list AND oversold_thresh +/- 1 step in list
  PASS >= 67% of zone combos profitable

MR gates:
  win_rate : 50-70%
  avg_r    : > 0.10R
  min_trades: 80

Year-by-year mandatory -- 2021/2025 concentration flagged automatically.
STOPS after T1 for human review.
"""

from __future__ import annotations

import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"


def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_williamsrmr_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Grid ----------
WR_N            = [5, 7, 10, 14, 20]
OVERSOLD_THRESH = [-70, -75, -80, -85, -90]
HOLD_BARS       = [10, 15, 20, 25]
ATR_MULTS       = [2.0, 3.0]
FILTER_MODES    = ["none", "ema200_price_above"]
TIMEFRAMES      = ["1d", "2h"]

MAX_BARS = {"1d": 2000, "2h": 17520}
MIN_BARS = 200

MR_GATES = {
    "min_trades":    80,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "stability_min": 0.67,
}

SYMBOLS_1D = []  # filled by scan at startup
SYMBOLS_2H = []


# =============================================================================
# DATA
# =============================================================================

def scan_symbols(tf: str) -> list[str]:
    files = sorted(RAW_DIR.glob(f"*_{tf}.csv"))
    return [f.stem.replace(f"_{tf}", "") for f in files]


def load_ohlcv(symbol: str, tf: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return None
        limit = MAX_BARS.get(tf, 2000)
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr14"]  = tr.rolling(14).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    for n in WR_N:
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        denom = hh - ll
        # Avoid division by zero -- flat bars
        out[f"wr{n}"] = np.where(denom > 1e-9, (hh - close) / denom * -100, 0.0)

    return out


# =============================================================================
# BACKTEST (single symbol, single param combo)
# =============================================================================

def backtest_symbol(df: pd.DataFrame,
                    wr_n: int, oversold: float,
                    hold_bars: int, atr_mult: float,
                    filter_mode: str) -> list[dict]:
    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    wr     = df[f"wr{wr_n}"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(wr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            ema_ok = (close[i] > ema200[i]) if filter_mode == "ema200_price_above" else True
            if ema_ok and wr[i] < oversold:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price - atr_mult * atr[i]
        else:
            bars_held  = i - e_bar
            exit_price = None
            reason     = None

            if low_v[i] <= stop:
                exit_price = stop
                reason     = "atr_stop"
            elif bars_held >= hold_bars:
                exit_price = close[i]
                reason     = "time_exit"

            if reason is not None:
                risk = e_price - stop
                if risk > 1e-9:
                    r_mult   = (exit_price - e_price) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    trades.append({
                        "net_r": float(r_mult),
                        "year":  entry_dt.year,
                    })
                in_pos = False

    return trades


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd_r": 0.0, "total_r": 0.0}
    rs   = np.array([t["net_r"] for t in trades])
    wins = rs[rs > 0]; losses = np.abs(rs[rs < 0])
    pf   = wins.sum() / losses.sum() if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    cum  = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd   = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
    return {
        "n":        len(rs),
        "win_rate": float(len(wins) / len(rs)),
        "avg_r":    float(np.mean(rs)),
        "pf":       float(pf),
        "max_dd_r": dd,
        "total_r":  float(np.sum(rs)),
    }


def year_breakdown(trades: list[dict]) -> dict[int, dict]:
    by_year: dict[int, list] = {}
    for t in trades:
        by_year.setdefault(t["year"], []).append(t)
    return {yr: calc_metrics(ts) for yr, ts in by_year.items()}


def stability_score(grid_df: pd.DataFrame,
                    wr_n: int, oversold: float,
                    atr_mult: float, filter_mode: str) -> float:
    wr_ns = WR_N
    ovs   = OVERSOLD_THRESH
    try:
        ri = wr_ns.index(wr_n)
        oi = ovs.index(oversold)
    except ValueError:
        return 0.0
    n_nbrs = [wr_ns[j] for j in range(max(0, ri-1), min(len(wr_ns), ri+2))]
    o_nbrs = [ovs[j]   for j in range(max(0, oi-1), min(len(ovs),   oi+2))]
    zone   = grid_df[
        (grid_df["wr_n"].isin(n_nbrs)) &
        (grid_df["oversold"].isin(o_nbrs)) &
        (grid_df["atr_mult"] == atr_mult) &
        (grid_df["filter_mode"] == filter_mode)
    ]
    if zone.empty:
        return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


# =============================================================================
# RUN ONE TIMEFRAME
# =============================================================================

def run_timeframe(tf: str, symbols: list[str]) -> pd.DataFrame:
    p(f"\n{'='*70}")
    p(f"  Timeframe: {tf.upper()}  |  {len(symbols)} symbols")
    p(f"{'='*70}")

    # Load all data
    loaded: list[tuple[str, pd.DataFrame]] = []
    for sym in symbols:
        df = load_ohlcv(sym, tf)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"  Loaded: {len(loaded)}/{len(symbols)}")

    param_combos = list(itertools.product(
        WR_N, OVERSOLD_THRESH, HOLD_BARS, ATR_MULTS, FILTER_MODES
    ))
    p(f"  Param combos: {len(param_combos)}")

    combo_trades: dict[tuple, list[dict]] = {k: [] for k in param_combos}

    for idx, (sym, df) in enumerate(loaded, 1):
        if idx == 1 or idx % 20 == 0 or idx == len(loaded):
            p(f"  Processing {idx}/{len(loaded)} {sym}...")
        for combo in param_combos:
            combo_trades[combo].extend(
                backtest_symbol(df, *combo)
            )

    rows = []
    for (wn, ov, hb, am, fm) in param_combos:
        m = calc_metrics(combo_trades[(wn, ov, hb, am, fm)])
        rows.append({
            "timeframe":   tf,
            "wr_n":        wn,
            "oversold":    ov,
            "hold_bars":   hb,
            "atr_mult":    am,
            "filter_mode": fm,
            "num_trades":  m["n"],
            "win_rate":    round(m["win_rate"], 4),
            "avg_r":       round(m["avg_r"], 4),
            "pf":          round(m["pf"], 4),
            "max_dd_r":    round(m["max_dd_r"], 2),
            "total_r":     round(m["total_r"], 2),
        })

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / f"phase_t1_grid_{tf}.csv", index=False)
    p(f"  Grid saved: {len(grid)} rows")

    # ---------- Gates ----------
    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    p(f"\n  Combos >= {MR_GATES['min_trades']} trades : {len(viable)}")

    viable_wr = viable[
        (viable["win_rate"] >= MR_GATES["win_rate_min"]) &
        (viable["win_rate"] <= MR_GATES["win_rate_max"])
    ]
    p(f"  After WR 50-70%             : {len(viable_wr)}")

    viable_ar = viable_wr[viable_wr["avg_r"] >= MR_GATES["avg_r_min"]]
    p(f"  After avg_r >= 0.10R        : {len(viable_ar)}")

    if viable_ar.empty:
        p(f"\n  No combos pass WR + avg_r gates on {tf.upper()}.")
        p(f"  Best by avg_r (trade-qualified):")
        top = viable.nlargest(10, "avg_r")[
            ["wr_n","oversold","hold_bars","atr_mult","filter_mode",
             "num_trades","win_rate","avg_r","pf"]]
        p(top.to_string(index=False))
        return grid

    viable_ar = viable_ar.copy()
    viable_ar["stability"] = viable_ar.apply(
        lambda r: stability_score(grid, int(r["wr_n"]), float(r["oversold"]),
                                  float(r["atr_mult"]), str(r["filter_mode"])),
        axis=1
    )
    stable = viable_ar[viable_ar["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"  After stability >= 0.67     : {len(stable)}")

    gate_pass = not stable.empty
    if not gate_pass:
        p(f"  NOTE: using top-20 by stability for display")
        stable = viable_ar.nlargest(20, "stability")

    stable = stable.sort_values(["stability", "avg_r"], ascending=[False, False]).reset_index(drop=True)

    # ---------- Heatmap ----------
    pivot = grid.groupby(["wr_n", "oversold"])["avg_r"].mean().unstack("oversold")
    p(f"\n  avg_r heatmap (wr_n vs oversold) | all hold_bars/atr/filter averaged:")
    p(pivot.round(3).to_string())

    # ---------- Top combos table ----------
    p(f"\n  Top stable combos ({tf.upper()}):")
    p(f"  {'wrN':>4} {'OB':>4} {'hold':>5} {'atr':>5} {'filter':>20} "
      f"{'trades':>7} {'WR%':>6} {'AvgR':>8} {'PF':>6} {'stab':>6}")
    p("  " + "-"*78)
    for _, row in stable.head(12).iterrows():
        p(f"  {int(row['wr_n']):>4} {int(row['oversold']):>4} "
          f"{int(row['hold_bars']):>5} {row['atr_mult']:>5.1f} "
          f"{row['filter_mode']:>20} "
          f"{int(row['num_trades']):>7} {row['win_rate']*100:>5.1f}% "
          f"{row['avg_r']:>+7.4f} {row['pf']:>6.2f} {row['stability']:>6.2f}")

    # ---------- Year-by-year for top 5 ----------
    p(f"\n  Year-by-year for top 5 combos ({tf.upper()}):")
    for rank, (_, row) in enumerate(stable.head(5).iterrows(), 1):
        key    = (int(row["wr_n"]), float(row["oversold"]),
                  int(row["hold_bars"]), float(row["atr_mult"]), str(row["filter_mode"]))
        trades = combo_trades[key]
        yb     = year_breakdown(trades)

        total_r = sum(m["total_r"] for m in yb.values())
        concentration_years = {
            yr: (m["total_r"] / total_r) for yr, m in yb.items()
            if total_r > 0 and m["total_r"] / total_r > 0.50
        }

        p(f"\n  #{rank}: WR{int(row['wr_n'])}/ob{int(row['oversold'])}/h{int(row['hold_bars'])}/"
          f"atr{row['atr_mult']}/{row['filter_mode']}  "
          f"avg_r={row['avg_r']:+.4f}R  WR={row['win_rate']*100:.1f}%  "
          f"stab={row['stability']:.2f}")
        if concentration_years:
            for yr, pct in concentration_years.items():
                p(f"  *** CONCENTRATION FLAG: {yr} = {pct*100:.0f}% of total R ***")

        p(f"  {'Year':>5}  {'N':>4}  {'WR%':>6}  {'AvgR':>8}  "
          f"{'TotR':>8}  {'DD':>6}  Note")
        for yr in sorted(yb.keys()):
            ym   = yb[yr]
            note = ""
            if yr == 2022:
                note = "<< BEAR"
            elif yr in (2021, 2025) and total_r > 0 and ym["total_r"] / total_r > 0.40:
                note = "*** HIGH CONC"
            p(f"  {yr:>5}  {ym['n']:>4}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
              f"{ym['max_dd_r']:>5.2f}R  {note}")

    stable.to_csv(OUT_DIR / f"phase_t1_stability_ranking_{tf}.csv", index=False)
    p(f"\n  [OK] phase_t1_grid_{tf}.csv")
    p(f"  [OK] phase_t1_stability_ranking_{tf}.csv ({len(stable)} stable combos)")
    return grid


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Phase T1 -- Williams %R MR Long Concept Discovery")
    p("  WR = (HH - close) / (HH - LL) * -100  |  range 0 to -100")
    p("  Entry: WR(N) < oversold_thresh (e.g. WR < -80)")
    p("  Exit : fixed time exit (primary) + ATR safety stop")
    p("  MR gates: WR 50-70%, avg_r > 0.10R, stability >= 67%")
    p("=" * 70)

    global SYMBOLS_1D, SYMBOLS_2H
    SYMBOLS_1D = scan_symbols("1d")
    SYMBOLS_2H = scan_symbols("2h")
    p(f"\n  Symbols available: 1D={len(SYMBOLS_1D)}  2H={len(SYMBOLS_2H)}")

    all_grids: list[pd.DataFrame] = []
    for tf, syms in [("1d", SYMBOLS_1D), ("2h", SYMBOLS_2H)]:
        g = run_timeframe(tf, syms)
        all_grids.append(g)

    # ---------- Combined summary ----------
    combined = pd.concat(all_grids, ignore_index=True)

    # Best combos per timeframe that pass all gates
    p(f"\n{'='*70}")
    p("  COMBINED SUMMARY")
    p(f"{'='*70}")
    for tf in TIMEFRAMES:
        tg    = combined[combined["timeframe"] == tf]
        valid = tg[
            (tg["num_trades"] >= MR_GATES["min_trades"]) &
            (tg["win_rate"] >= MR_GATES["win_rate_min"]) &
            (tg["win_rate"] <= MR_GATES["win_rate_max"]) &
            (tg["avg_r"] >= MR_GATES["avg_r_min"])
        ]
        p(f"  {tf.upper()}: {len(valid)} combos pass WR+avg_r gates")
        if not valid.empty:
            best = valid.nlargest(1, "avg_r").iloc[0]
            p(f"    Best: WR({int(best['wr_n'])})/ob{int(best['oversold'])}/"
              f"h{int(best['hold_bars'])}/{best['filter_mode']}  "
              f"avg_r={best['avg_r']:+.4f}R  WR={best['win_rate']*100:.1f}%")

    # ---------- Write summary txt ----------
    with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Williams %R MR Long -- T1 Concept Discovery\n")
        f.write(f"Timeframes: {TIMEFRAMES}\n")
        f.write(f"MR gates: WR 50-70%, avg_r > 0.10R, stability >= 0.67\n\n")
        for tf in TIMEFRAMES:
            tg    = combined[combined["timeframe"] == tf]
            valid = tg[
                (tg["num_trades"] >= MR_GATES["min_trades"]) &
                (tg["win_rate"] >= MR_GATES["win_rate_min"]) &
                (tg["win_rate"] <= MR_GATES["win_rate_max"]) &
                (tg["avg_r"] >= MR_GATES["avg_r_min"])
            ]
            f.write(f"\n{tf.upper()}: {len(valid)} gate-passing combos\n")
            if not valid.empty:
                f.write(valid.nlargest(10, "avg_r")[
                    ["wr_n","oversold","hold_bars","atr_mult","filter_mode",
                     "num_trades","win_rate","avg_r","pf","total_r"]
                ].to_string(index=False))
                f.write("\n")

    p(f"\n[OK] phase_t1_summary.txt")
    p(f"\nSTOPPING -- awaiting human review of stability ranking before T2.")


if __name__ == "__main__":
    main()
