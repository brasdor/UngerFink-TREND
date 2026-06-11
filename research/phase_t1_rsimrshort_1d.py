#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- RSI MR Short 1D (Tests 1 and 2 combined)
UngerFink Pipeline / Andrea Unger Methodology

Test 1 (unfiltered)  : filter=none only
Test 2 (filtered)    : filter=ema200_price_below only
Run both, output separately, compare side-by-side.

Entry  : RSI(N) > overbought threshold on 1D close --> SHORT at bar close
Exit   : fixed time exit after hold_bars
Safety : ATR x atr_mult ABOVE entry (stop for shorts)

Unlike RSIShortBreakdown (which used Chandelier trailing stop), this is a
PURE MR system: fixed time exit, no trend-following exit. Expecting:
  win_rate 50-70% (MR profile, not TF 30-45%)
  avg_r > 0.15R   (Futures cost floor)
  2022 bear       : positive or near-zero (overbought spikes in bear recoveries)

Parameter grid:
  rsi_n       : [7, 10, 14]
  overbought  : [70, 75, 80, 85]
  hold_bars   : [10, 15, 20, 25]
  atr_mult    : [2.0, 3.0]
  Total per filter: 3 x 4 x 4 x 2 = 96 combos

Stability zone: rsi_n +/- 1 step AND overbought +/- 1 step, PASS >= 67%

HALT condition: if 2025 > 50% of total R for canonical combo --> HALT
FLAG condition: if 2025+2026 < 0 --> flag prominently

Output: data/research_rsimrshort_1d/        (Test 1, filter=none)
        data/research_rsimrshort_1d_filtered/ (Test 2, filter=ema200_price_below)
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

TIMEFRAME = "1d"
MAX_BARS  = 2000
MIN_BARS  = 200

RSI_N       = [7, 10, 14]
OVERBOUGHT  = [70, 75, 80, 85]
HOLD_BARS   = [10, 15, 20, 25]
ATR_MULTS   = [2.0, 3.0]

MR_GATES = {
    "min_trades":    80,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.15,   # Futures cost floor
    "stability_min": 0.67,
}

HALT_YEAR_MAX_PCT = 0.50   # halt if any single year > 50% of total R

SYMBOLS = [
    "AAVE_USDT", "ADA_USDT",  "ALT_USDT",  "APT_USDT",  "ARB_USDT",
    "ARKM_USDT", "ASTER_USDT","ATOM_USDT", "AVAX_USDT", "BCH_USDT",
    "BNB_USDT",  "BTC_USDT",  "CHZ_USDT",  "DASH_USDT", "DOGE_USDT",
    "DOT_USDT",  "EIGEN_USDT","ENA_USDT",  "ETH_USDT",  "FET_USDT",
    "FIL_USDT",  "GRT_USDT",  "HBAR_USDT", "ICP_USDT",  "INJ_USDT",
    "JTO_USDT",  "LINK_USDT", "LPT_USDT",  "LTC_USDT",  "MORPHO_USDT",
    "NEAR_USDT", "NIL_USDT",  "ONDO_USDT", "ORDI_USDT", "PENDLE_USDT",
    "PENGU_USDT","PEPE_USDT", "RENDER_USDT","SAGA_USDT","SEI_USDT",
    "SOL_USDT",  "SPK_USDT",  "SUI_USDT",  "TAO_USDT",  "TIA_USDT",
    "TON_USDT",  "TRX_USDT",  "UNI_USDT",  "WLD_USDT",  "XRP_USDT",
    "ZEC_USDT",  "ZEN_USDT",
]


# =============================================================================
# DATA / INDICATORS
# =============================================================================

def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{symbol}_{TIMEFRAME}.csv"
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
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:].reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    close = out["close"]; high = out["high"]; low = out["low"]
    tr = pd.concat([high-low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    out["atr14"]  = tr.rolling(14).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    for n in RSI_N:
        d  = close.diff()
        ag = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        out[f"rsi{n}"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    return out


# =============================================================================
# BACKTEST (SHORT side, fixed time exit)
# =============================================================================

def backtest_symbol(df: pd.DataFrame,
                    rsi_n: int, overbought: int,
                    hold_bars: int, atr_mult: float,
                    filter_mode: str) -> list[dict]:
    close  = df["close"].values
    high_v = df["high"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    rsi    = df[f"rsi{rsi_n}"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            ema_ok = (close[i] < ema200[i]) if filter_mode == "ema200_price_below" else True
            if ema_ok and rsi[i] > overbought:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price + atr_mult * atr[i]   # stop ABOVE entry for shorts
        else:
            bars_held  = i - e_bar
            exit_price = None
            reason     = None

            if high_v[i] >= stop:
                exit_price = stop
                reason     = "atr_stop"
            elif bars_held >= hold_bars:
                exit_price = close[i]
                reason     = "time_exit"

            if reason is not None:
                risk = stop - e_price           # always positive
                if risk > 1e-9:
                    r_mult   = (e_price - exit_price) / risk  # positive when price fell
                    entry_dt = pd.Timestamp(e_ts)
                    trades.append({
                        "net_r": float(r_mult),
                        "year":  entry_dt.year,
                    })
                in_pos = False

    return trades


# =============================================================================
# METRICS / STABILITY / HALT CHECK
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
    return {"n": len(rs), "win_rate": float(len(wins)/len(rs)),
            "avg_r": float(np.mean(rs)), "pf": float(pf),
            "max_dd_r": dd, "total_r": float(np.sum(rs))}


def year_breakdown(trades: list[dict]) -> dict[int, dict]:
    by_year: dict[int, list] = {}
    for t in trades:
        by_year.setdefault(t["year"], []).append(t)
    return {yr: calc_metrics(ts) for yr, ts in by_year.items()}


def stability_score(grid: pd.DataFrame, rsi_n: int, ob: int,
                    atr_mult: float) -> float:
    rn_list = RSI_N; ob_list = OVERBOUGHT
    try:
        ri = rn_list.index(rsi_n); oi = ob_list.index(ob)
    except ValueError:
        return 0.0
    rn_nbrs = [rn_list[j] for j in range(max(0,ri-1), min(len(rn_list),ri+2))]
    ob_nbrs = [ob_list[j] for j in range(max(0,oi-1), min(len(ob_list),oi+2))]
    zone = grid[
        (grid["rsi_n"].isin(rn_nbrs)) &
        (grid["overbought"].isin(ob_nbrs)) &
        (grid["atr_mult"] == atr_mult)
    ]
    if zone.empty:
        return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


def concentration_check(yb: dict[int, dict], total_r: float) -> list[str]:
    flags = []
    if total_r <= 0:
        return flags
    for yr, ym in yb.items():
        pct = ym["total_r"] / total_r
        if pct > HALT_YEAR_MAX_PCT:
            flags.append(f"HALT: {yr}={pct*100:.0f}% of total R (limit {HALT_YEAR_MAX_PCT*100:.0f}%)")
    r2025 = yb.get(2025, {}).get("total_r", 0.0)
    r2026 = yb.get(2026, {}).get("total_r", 0.0)
    if r2025 + r2026 < 0:
        flags.append(f"FLAG: 2025+2026 = {r2025+r2026:.2f}R (recent deterioration)")
    return flags


# =============================================================================
# RUN ONE FILTER VARIANT
# =============================================================================

def run_filter(loaded: list[tuple[str, pd.DataFrame]],
               filter_mode: str,
               out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    param_combos = list(itertools.product(RSI_N, OVERBOUGHT, HOLD_BARS, ATR_MULTS))
    combo_trades: dict[tuple, list[dict]] = {k: [] for k in param_combos}

    for sym, df in loaded:
        for combo in param_combos:
            combo_trades[combo].extend(
                backtest_symbol(df, *combo, filter_mode)
            )

    rows = []
    for (rn, ob, hb, am) in param_combos:
        m = calc_metrics(combo_trades[(rn, ob, hb, am)])
        rows.append({"rsi_n": rn, "overbought": ob, "hold_bars": hb,
                     "atr_mult": am, "filter_mode": filter_mode,
                     "num_trades": m["n"], "win_rate": round(m["win_rate"],4),
                     "avg_r": round(m["avg_r"],4), "pf": round(m["pf"],4),
                     "max_dd_r": round(m["max_dd_r"],2), "total_r": round(m["total_r"],2)})
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "phase_t1_grid_1d.csv", index=False)

    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    viable_wr = viable[(viable["win_rate"] >= MR_GATES["win_rate_min"]) &
                       (viable["win_rate"] <= MR_GATES["win_rate_max"])]
    viable_ar = viable_wr[viable_wr["avg_r"] >= MR_GATES["avg_r_min"]].copy()

    p(f"    Combos >= 80 trades     : {len(viable)}")
    p(f"    After WR 50-70%         : {len(viable_wr)}")
    p(f"    After avg_r >= 0.15R    : {len(viable_ar)}")

    if viable_ar.empty:
        p(f"    No combos pass gates. Best by avg_r:")
        p(viable.nlargest(8,"avg_r")[["rsi_n","overbought","hold_bars",
            "atr_mult","num_trades","win_rate","avg_r","pf"]].to_string(index=False))
        grid.to_csv(out_dir / "phase_t1_grid_1d.csv", index=False)
        with open(out_dir / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"RSI MR Short 1D | filter={filter_mode}\nT1 GATE: FAIL\n")
        return grid

    viable_ar["stability"] = viable_ar.apply(
        lambda r: stability_score(grid, int(r["rsi_n"]), int(r["overbought"]),
                                  float(r["atr_mult"])), axis=1)
    stable = viable_ar[viable_ar["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"    After stability >= 0.67 : {len(stable)}")

    if stable.empty:
        stable = viable_ar.nlargest(10, "stability")

    stable = stable.sort_values(["stability","avg_r"], ascending=[False,False]).reset_index(drop=True)
    stable.to_csv(out_dir / "phase_t1_stability_ranking.csv", index=False)

    # Heatmap
    pivot = grid.groupby(["rsi_n","overbought"])["avg_r"].mean().unstack("overbought")
    p(f"\n    avg_r heatmap (rsi_n vs overbought):")
    p(pivot.round(3).to_string())

    # Year-by-year + halt check for top combos
    halt_any = False
    p(f"\n    Year-by-year top {min(5,len(stable))} combos:")
    for rank, (_, row) in enumerate(stable.head(5).iterrows(), 1):
        key    = (int(row["rsi_n"]), int(row["overbought"]),
                  int(row["hold_bars"]), float(row["atr_mult"]))
        trades = combo_trades[key]
        yb     = year_breakdown(trades)
        m      = calc_metrics(trades)
        flags  = concentration_check(yb, m["total_r"])
        has_halt = any("HALT" in f for f in flags)
        if has_halt: halt_any = True
        status = "HALT" if has_halt else ("FLAG" if flags else "PASS")

        p(f"\n    #{rank}: RSI{int(row['rsi_n'])}/ob{int(row['overbought'])}/"
          f"h{int(row['hold_bars'])}/atr{row['atr_mult']}  "
          f"avg_r={row['avg_r']:+.4f}R  WR={row['win_rate']*100:.1f}%  "
          f"stab={row['stability']:.2f}  [{status}]")
        for f in flags:
            p(f"    --> {f}")

        p(f"    {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotR':>8}  {'Frac%':>7}")
        for yr in sorted(yb.keys()):
            ym   = yb[yr]
            frac = ym["total_r"] / m["total_r"] if m["total_r"] != 0 else 0.0
            mark = " <<BEAR" if yr == 2022 else (" ***CONC" if frac > 0.50 else "")
            p(f"    {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  {frac:>+6.0%}{mark}")

    gate_str = "HALT" if halt_any else ("PASS" if not stable.empty else "FAIL")
    p(f"\n    T1 gate filter={filter_mode}: {gate_str}")

    with open(out_dir / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"RSI MR Short 1D | filter={filter_mode}\n")
        f.write(f"Gates: WR 50-70%, avg_r > 0.15R (Futures), stability >= 0.67\n")
        f.write(f"Halt: single year > 50% total R\n\n")
        f.write(f"Stable combos: {len(stable)}\n")
        f.write(f"T1 GATE: {gate_str}\n\n")
        cols = ["rsi_n","overbought","hold_bars","atr_mult","num_trades",
                "win_rate","avg_r","pf","stability","total_r"]
        if all(c in stable.columns for c in cols):
            f.write(stable.head(10)[cols].to_string(index=False))
        f.write("\n\navg_r heatmap:\n")
        f.write(pivot.round(3).to_string())

    return grid


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Phase T1 -- RSI MR Short 1D (Tests 1 + 2 combined)")
    p("  Entry: RSI(N) > overbought --> SHORT at bar close")
    p("  Exit: fixed time exit (MR profile)")
    p("  S4.2: 0.15R Futures cost floor")
    p("  Halt: single year > 50% of total R")
    p("=" * 70)

    # Load once
    loaded: list[tuple[str, pd.DataFrame]] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"\n  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")

    # Test 1: filter=none
    out1 = ROOT / "data" / "research_rsimrshort_1d"
    p(f"\n{'='*70}")
    p(f"  TEST 1: filter=none  -->  {out1.name}/")
    p(f"{'='*70}")
    run_filter(loaded, "none", out1)

    # Test 2: filter=ema200_price_below
    out2 = ROOT / "data" / "research_rsimrshort_1d_filtered"
    p(f"\n{'='*70}")
    p(f"  TEST 2: filter=ema200_price_below  -->  {out2.name}/")
    p(f"{'='*70}")
    run_filter(loaded, "ema200_price_below", out2)

    p(f"\n[DONE] RSI MR Short 1D -- Tests 1 and 2 complete")
    p(f"STOPPING -- awaiting human review before T2.")


if __name__ == "__main__":
    main()
