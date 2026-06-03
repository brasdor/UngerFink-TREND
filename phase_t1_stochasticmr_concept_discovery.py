#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- Stochastic Oscillator MR Long Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology

FINAL MR OSCILLATOR INVESTIGATION -- strict early halt applied.

Stochastic %K(N) = (close - lowest_low(N)) / (highest_high(N) - lowest_low(N)) x 100
Range: 0 to 100  (oversold = below threshold, e.g. < 20)
%D = 3-bar SMA of %K  (standard smoothing, reduces noise)

Entry : %K (or %D) < oversold_thresh --> enter long at bar close
Exit  : fixed time exit after hold_bars (primary -- time exit validated for MR)
Safety: ATR x atr_mult below entry (backstop)
Filter: none | ema200_price_above

Parameter grid:
  stoch_n       : [5, 7, 10, 14, 20]
  oversold_thresh: [10, 15, 20, 25]
  smooth_d      : [1, 3]   (1 = raw %K, 3 = 3-bar %D)
  hold_bars     : [10, 15, 20, 25]
  atr_mult      : [2.0, 3.0]
  filter_mode   : [none, ema200_price_above]
  Total combos  : 5 x 4 x 2 x 4 x 2 x 2 = 640

Stability zone: stoch_n +/- 1 step AND oversold_thresh +/- 1 step, PASS >= 67%

MR gates:
  win_rate : 50-70%
  avg_r    : > 0.10R
  min_trades: 80

MANDATORY EARLY HALT CONDITIONS (applied before reporting T1 PASS):
  1. 2021 concentration: if 2021 > 60% of total R for canonical combo --> HALT
  2. Recent deterioration: if 2025+2026 combined < 0 --> FLAG
  3. Year distribution: require at least 4 of 6 years positive
  4. 2022 bear: must be >= -10R (not catastrophic)

Benchmark: RSI MR 1D frozen (filter=none, all years positive, 2022 bear = best year)

Output: data/research_stochasticmr_t1/
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


ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_stochasticmr_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "1d"
MAX_BARS  = 2000
MIN_BARS  = 200

STOCH_N        = [5, 7, 10, 14, 20]
OVERSOLD_THRESH= [10, 15, 20, 25]
SMOOTH_D       = [1, 3]
HOLD_BARS      = [10, 15, 20, 25]
ATR_MULTS      = [2.0, 3.0]
FILTER_MODES   = ["none", "ema200_price_above"]

MR_GATES = {
    "min_trades":    80,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "stability_min": 0.67,
}

# Early halt thresholds
HALT_2021_MAX_PCT   = 0.60   # 2021 must not be > 60% of total R
HALT_MIN_POS_YEARS  = 4      # at least 4 of 6 years positive
HALT_BEAR_FLOOR     = -10.0  # 2022 must be >= -10R
FLAG_RECENT_NEG     = True   # flag if 2025+2026 < 0

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
# DATA
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

    for n in STOCH_N:
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        denom = hh - ll
        k = np.where(denom > 1e-9, (close - ll) / denom * 100.0, 50.0)
        k_series = pd.Series(k, index=close.index)
        out[f"k{n}"]  = k_series
        out[f"d3_{n}"] = k_series.rolling(3).mean()   # %D = 3-bar SMA of %K

    return out


# =============================================================================
# BACKTEST (single symbol, single param combo)
# =============================================================================

def backtest_symbol(df: pd.DataFrame,
                    stoch_n: int, oversold: float,
                    smooth_d: int, hold_bars: int,
                    atr_mult: float, filter_mode: str) -> list[dict]:

    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    # Use %K or %D depending on smooth_d
    if smooth_d == 1:
        signal = df[f"k{stoch_n}"].values
    else:
        signal = df[f"d3_{stoch_n}"].values

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(signal[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            ema_ok = (close[i] > ema200[i]) if filter_mode == "ema200_price_above" else True
            if ema_ok and signal[i] < oversold:
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


# =============================================================================
# STABILITY
# =============================================================================

def stability_score(grid: pd.DataFrame,
                    stoch_n: int, oversold: float,
                    smooth_d: int, atr_mult: float,
                    filter_mode: str) -> float:
    sn_list = STOCH_N; ob_list = OVERSOLD_THRESH
    try:
        si = sn_list.index(stoch_n); oi = ob_list.index(int(oversold))
    except ValueError:
        return 0.0
    sn_nbrs = [sn_list[j] for j in range(max(0, si-1), min(len(sn_list), si+2))]
    ob_nbrs = [ob_list[j] for j in range(max(0, oi-1), min(len(ob_list), oi+2))]
    zone = grid[
        (grid["stoch_n"].isin(sn_nbrs)) &
        (grid["oversold"].isin(ob_nbrs)) &
        (grid["smooth_d"] == smooth_d) &
        (grid["atr_mult"] == atr_mult) &
        (grid["filter_mode"] == filter_mode)
    ]
    if zone.empty:
        return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


# =============================================================================
# EARLY HALT CHECK
# =============================================================================

def early_halt_check(yb: dict[int, dict], total_r: float,
                     label: str) -> tuple[bool, list[str]]:
    """
    Returns (halt, reasons_list).
    halt=True means this combo fails the strict year-distribution test.
    """
    reasons: list[str] = []
    halt = False
    all_years = sorted(yb.keys())

    # 1. 2021 concentration
    r2021 = yb.get(2021, {}).get("total_r", 0.0)
    if total_r > 0:
        pct2021 = r2021 / total_r
        if pct2021 > HALT_2021_MAX_PCT:
            reasons.append(f"HALT: 2021={pct2021*100:.0f}% of total R (limit {HALT_2021_MAX_PCT*100:.0f}%)")
            halt = True

    # 2. Positive years count
    pos_years = [yr for yr in all_years if yb[yr]["total_r"] > 0]
    if len(pos_years) < HALT_MIN_POS_YEARS:
        reasons.append(f"HALT: only {len(pos_years)}/6 years positive (need >= {HALT_MIN_POS_YEARS})")
        halt = True

    # 3. 2022 bear floor
    r2022 = yb.get(2022, {}).get("total_r", 0.0)
    if r2022 < HALT_BEAR_FLOOR:
        reasons.append(f"HALT: 2022 bear = {r2022:.1f}R (floor {HALT_BEAR_FLOOR}R)")
        halt = True

    # 4. Recent (2025+2026) flag
    r2025 = yb.get(2025, {}).get("total_r", 0.0)
    r2026 = yb.get(2026, {}).get("total_r", 0.0)
    recent = r2025 + r2026
    if recent < 0:
        reasons.append(f"FLAG: 2025+2026 combined = {recent:.2f}R (negative recent)")

    return halt, reasons


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    total_combos = (len(STOCH_N) * len(OVERSOLD_THRESH) * len(SMOOTH_D)
                    * len(HOLD_BARS) * len(ATR_MULTS) * len(FILTER_MODES))

    p("=" * 70)
    p("  Phase T1 -- Stochastic Oscillator MR Long Concept Discovery")
    p("  %K(N) = (close - LL_N) / (HH_N - LL_N) x 100  |  range 0-100")
    p("  Entry: %K (or %D) < oversold_thresh --> long at bar close")
    p("  FINAL MR OSCILLATOR -- strict early halt applied")
    p(f"  Total combos: {total_combos}  |  Universe: {len(SYMBOLS)} symbols")
    p("=" * 70)

    # Load & pre-compute indicators once per symbol
    loaded: list[tuple[str, pd.DataFrame]] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"\n  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")

    param_combos = list(itertools.product(
        STOCH_N, OVERSOLD_THRESH, SMOOTH_D, HOLD_BARS, ATR_MULTS, FILTER_MODES
    ))

    # Accumulate trades per combo
    combo_trades: dict[tuple, list[dict]] = {k: [] for k in param_combos}
    for idx, (sym, df) in enumerate(loaded, 1):
        if idx == 1 or idx % 15 == 0 or idx == len(loaded):
            p(f"  Processing {idx}/{len(loaded)} {sym}...")
        for combo in param_combos:
            combo_trades[combo].extend(backtest_symbol(df, *combo))

    # Build grid
    rows = []
    for (sn, ob, sd, hb, am, fm) in param_combos:
        m = calc_metrics(combo_trades[(sn, ob, sd, hb, am, fm)])
        rows.append({
            "stoch_n":     sn,
            "oversold":    ob,
            "smooth_d":    sd,
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
    grid.to_csv(OUT_DIR / "phase_t1_grid_1d.csv", index=False)
    p(f"  Grid saved: {len(grid)} rows")

    # ---------- Gate filters ----------
    p(f"\n--- Gate Analysis ---")
    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    p(f"  Combos >= {MR_GATES['min_trades']} trades : {len(viable)}")

    viable_wr = viable[
        (viable["win_rate"] >= MR_GATES["win_rate_min"]) &
        (viable["win_rate"] <= MR_GATES["win_rate_max"])
    ]
    p(f"  After WR 50-70%            : {len(viable_wr)}")

    viable_ar = viable_wr[viable_ar_mask := viable_wr["avg_r"] >= MR_GATES["avg_r_min"]]
    p(f"  After avg_r >= 0.10R       : {len(viable_ar)}")

    if viable_ar.empty:
        p(f"\n  No combos pass WR + avg_r gates.")
        p(f"  Best by avg_r (trade-qualified):")
        cols = ["stoch_n","oversold","smooth_d","hold_bars","atr_mult",
                "filter_mode","num_trades","win_rate","avg_r","pf"]
        p(viable.nlargest(12, "avg_r")[cols].to_string(index=False))
        p(f"\n  T1 GATE: FAIL -- no edge found")
        with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
            f.write("Stochastic MR T1 -- FAIL\n")
            f.write("No combos pass WR 50-70% AND avg_r > 0.10R\n\n")
            f.write(viable.nlargest(12,"avg_r")[cols].to_string(index=False))
        sys.exit(1)

    # Stability
    viable_ar = viable_ar.copy()
    viable_ar["stability"] = viable_ar.apply(
        lambda r: stability_score(grid, int(r["stoch_n"]), float(r["oversold"]),
                                  int(r["smooth_d"]), float(r["atr_mult"]),
                                  str(r["filter_mode"])),
        axis=1,
    )
    stable = viable_ar[viable_ar["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"  After stability >= 0.67    : {len(stable)}")

    if stable.empty:
        p(f"  NOTE: no stability-passing combos. Top by stability:")
        stable = viable_ar.nlargest(15, "stability")
    stable = stable.sort_values(["stability","avg_r"], ascending=[False,False]).reset_index(drop=True)

    # ---------- Heatmaps ----------
    p(f"\n  avg_r heatmap (%K only, smooth_d=1, all hold/atr/filter averaged):")
    g1 = grid[grid["smooth_d"]==1].groupby(["stoch_n","oversold"])["avg_r"].mean().unstack("oversold")
    p(g1.round(3).to_string())

    p(f"\n  avg_r heatmap (%D smooth_d=3, all hold/atr/filter averaged):")
    g3 = grid[grid["smooth_d"]==3].groupby(["stoch_n","oversold"])["avg_r"].mean().unstack("oversold")
    p(g3.round(3).to_string())

    # ---------- Top combos table ----------
    p(f"\n  Top stable combos:")
    p(f"  {'sN':>3} {'OB':>4} {'sD':>3} {'hold':>5} {'atr':>5} {'filter':>20} "
      f"{'trades':>7} {'WR%':>6} {'AvgR':>8} {'PF':>6} {'stab':>6}")
    p("  " + "-"*80)
    for _, row in stable.head(15).iterrows():
        p(f"  {int(row['stoch_n']):>3} {int(row['oversold']):>4} "
          f"{int(row['smooth_d']):>3} {int(row['hold_bars']):>5} "
          f"{row['atr_mult']:>5.1f} {row['filter_mode']:>20} "
          f"{int(row['num_trades']):>7} {row['win_rate']*100:>5.1f}% "
          f"{row['avg_r']:>+7.4f} {row['pf']:>6.2f} {row['stability']:>6.2f}")

    # ---------- Year-by-year + early halt for ALL stable combos ----------
    p(f"\n{'='*70}")
    p(f"  EARLY HALT CHECK -- Year-by-Year for ALL {len(stable)} stable combos")
    p(f"  Halt conditions: 2021>60% total R | <4 positive years | 2022<-10R")
    p(f"  Flag condition:  2025+2026 combined < 0")
    p(f"{'='*70}")

    halt_count    = 0
    pass_count    = 0
    flag_count    = 0
    passing_combos: list[tuple[dict, list[str]]] = []

    for rank, (_, row) in enumerate(stable.iterrows(), 1):
        key    = (int(row["stoch_n"]), int(row["oversold"]), int(row["smooth_d"]),
                  int(row["hold_bars"]), float(row["atr_mult"]), str(row["filter_mode"]))
        trades = combo_trades[key]
        yb     = year_breakdown(trades)
        m      = calc_metrics(trades)

        halt, reasons = early_halt_check(yb, m["total_r"], str(key))
        has_recent_flag = any("FLAG" in r for r in reasons)
        has_halt        = any("HALT" in r for r in reasons)

        if has_halt:
            halt_count += 1
        elif has_recent_flag:
            flag_count += 1
            passing_combos.append((row.to_dict(), reasons))
        else:
            pass_count += 1
            passing_combos.append((row.to_dict(), []))

        status = "HALT" if has_halt else ("FLAG" if has_recent_flag else "PASS")
        p(f"\n  #{rank}: K{int(row['stoch_n'])}/ob{int(row['oversold'])}/d{int(row['smooth_d'])}/"
          f"h{int(row['hold_bars'])}/atr{row['atr_mult']}/{row['filter_mode']}  "
          f"avg_r={row['avg_r']:+.4f}R  WR={row['win_rate']*100:.1f}%  "
          f"stab={row['stability']:.2f}  [{status}]")
        for reason in reasons:
            p(f"  --> {reason}")

        p(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotR':>8}  {'Frac':>6}")
        total_r_abs = abs(m["total_r"]) if m["total_r"] != 0 else 1.0
        for yr in sorted(yb.keys()):
            ym   = yb[yr]
            frac = ym["total_r"] / m["total_r"] if m["total_r"] != 0 else 0.0
            bear_mark = " <<BEAR" if yr == 2022 else ""
            conc_mark = " ***CONC" if frac > 0.50 else ""
            p(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
              f"{frac:>+5.0%}{bear_mark}{conc_mark}")

    p(f"\n{'='*70}")
    p(f"  EARLY HALT SUMMARY")
    p(f"{'='*70}")
    p(f"  Stable combos checked : {len(stable)}")
    p(f"  HALT (fail conditions): {halt_count}")
    p(f"  FLAG (recent warning) : {flag_count}")
    p(f"  PASS (clean)          : {pass_count}")

    overall_gate = pass_count > 0 or flag_count > 0
    final_status = "PASS" if pass_count > 0 else ("MARGINAL" if flag_count > 0 else "HALT")

    p(f"\n  T1 OVERALL: {final_status}")

    if passing_combos:
        best_row, best_flags = passing_combos[0]
        p(f"\n  Canonical recommendation:")
        p(f"    stoch_n={int(best_row['stoch_n'])}  oversold={int(best_row['oversold'])}  "
          f"smooth_d={int(best_row['smooth_d'])}  hold={int(best_row['hold_bars'])}  "
          f"atr={best_row['atr_mult']}  filter={best_row['filter_mode']}")
        p(f"    avg_r={best_row['avg_r']:+.4f}R  WR={best_row['win_rate']*100:.1f}%  "
          f"stab={best_row['stability']:.2f}")
        if best_flags:
            p(f"    Flags: {' | '.join(best_flags)}")
    else:
        p(f"\n  No combos survive early halt check.")
        p(f"  Stochastic MR: HALT -- same structural pattern as Williams %R and BollingerBandMR")

    # ---------- Save ----------
    stable.to_csv(OUT_DIR / "phase_t1_stability_ranking.csv", index=False)

    with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Stochastic Oscillator MR Long -- T1 Concept Discovery\n")
        f.write(f"Config: 1D | stoch_n x oversold x smooth_d x hold x atr x filter\n")
        f.write(f"MR gates: WR 50-70%, avg_r > 0.10R, stability >= 0.67\n")
        f.write(f"Early halt: 2021 <= 60% | >=4 pos years | 2022 >= -10R\n\n")
        f.write(f"Grid: {len(grid)} combos\n")
        f.write(f"Stable (gate pass): {len(stable)}\n")
        f.write(f"After early halt: HALT={halt_count} FLAG={flag_count} PASS={pass_count}\n")
        f.write(f"T1 OVERALL: {final_status}\n\n")
        if not stable.empty:
            cols = ["stoch_n","oversold","smooth_d","hold_bars","atr_mult",
                    "filter_mode","num_trades","win_rate","avg_r","pf","stability","total_r"]
            f.write("Stable combos:\n")
            f.write(stable[cols].to_string(index=False))
        f.write("\n\navg_r heatmap (%K, smooth_d=1):\n")
        f.write(g1.round(3).to_string())
        f.write("\n\navg_r heatmap (%D, smooth_d=3):\n")
        f.write(g3.round(3).to_string())

    p(f"\n[OK] phase_t1_grid_1d.csv       ({len(grid)} combos)")
    p(f"[OK] phase_t1_stability_ranking.csv ({len(stable)} stable)")
    p(f"[OK] phase_t1_summary.txt")
    p(f"\nSTOPPING -- awaiting human review. T1 OVERALL: {final_status}")
    sys.exit(0 if overall_gate else 1)


if __name__ == "__main__":
    main()
