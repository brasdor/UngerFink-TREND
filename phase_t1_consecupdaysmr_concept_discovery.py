#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- ConsecUpDays MR Short Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology

Mirror of ConsecDownDays MR Long -- SHORT side.

Entry  : N consecutive closes where close > prev_close --> SHORT at bar close
Exit   : fixed time exit after hold_bars (MR profile)
Safety : ATR x atr_mult ABOVE entry (stop for shorts)
Filter : none | ema200_price_above (short in bull trend, overbought bounce)
         | ema200_price_below (short in bear trend, dead-cat bounce)

Parameter grid:
  consec_n    : [3, 4, 5, 6, 7]
  hold_bars   : [5, 10, 15, 20]
  atr_mult    : [2.0, 3.0]
  filter_mode : [none, ema200_price_above, ema200_price_below]
  Total combos: 5 x 4 x 2 x 3 = 120

Stability zone: consec_n +/- 1 step in list, PASS >= 67%

MR gates (Futures cost floor):
  win_rate    : 50-70%
  avg_r       : > 0.15R
  min_trades  : 80

HALT: if any single year > 50% of total R for canonical combo
FLAG: if 2025+2026 combined < 0

Output: data/research_consecupdaysmr_t1/
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
OUT_DIR = ROOT / "data" / "research_consecupdaysmr_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "1d"
MAX_BARS  = 2000
MIN_BARS  = 200

CONSEC_N     = [3, 4, 5, 6, 7]
HOLD_BARS    = [5, 10, 15, 20]
ATR_MULTS    = [2.0, 3.0]
FILTER_MODES = ["none", "ema200_price_above", "ema200_price_below"]

MR_GATES = {
    "min_trades":    80,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.15,   # Futures cost floor
    "stability_min": 0.67,
}

HALT_YEAR_MAX_PCT = 0.50

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
    # Consecutive up closes: count how many bars back close > prev_close
    up = (close > close.shift(1)).astype(int)
    # Rolling sum of consecutive ups ending at each bar
    consec = []
    cnt = 0
    for val in up:
        if val == 1:
            cnt += 1
        else:
            cnt = 0
        consec.append(cnt)
    out["consec_up"] = consec
    return out


# =============================================================================
# BACKTEST (SHORT side, fixed time exit)
# =============================================================================

def backtest_symbol(df: pd.DataFrame,
                    consec_n: int, hold_bars: int,
                    atr_mult: float, filter_mode: str) -> list[dict]:
    close    = df["close"].values
    high_v   = df["high"].values
    atr      = df["atr14"].values
    ema200   = df["ema200"].values
    consec   = df["consec_up"].values
    ts       = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            if filter_mode == "ema200_price_above":
                ema_ok = close[i] > ema200[i]
            elif filter_mode == "ema200_price_below":
                ema_ok = close[i] < ema200[i]
            else:
                ema_ok = True

            if ema_ok and consec[i] >= consec_n:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                stop    = e_price + atr_mult * atr[i]   # stop above entry
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
                risk = stop - e_price    # always positive
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
# METRICS / STABILITY / HALT
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


def stability_score(grid: pd.DataFrame, consec_n: int,
                    atr_mult: float, filter_mode: str) -> float:
    cn_list = CONSEC_N
    try:
        ci = cn_list.index(consec_n)
    except ValueError:
        return 0.0
    cn_nbrs = [cn_list[j] for j in range(max(0,ci-1), min(len(cn_list),ci+2))]
    zone = grid[
        (grid["consec_n"].isin(cn_nbrs)) &
        (grid["atr_mult"] == atr_mult) &
        (grid["filter_mode"] == filter_mode)
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
            flags.append(f"HALT: {yr}={pct*100:.0f}% of total R")
    r2025 = yb.get(2025, {}).get("total_r", 0.0)
    r2026 = yb.get(2026, {}).get("total_r", 0.0)
    if r2025 + r2026 < 0:
        flags.append(f"FLAG: 2025+2026 = {r2025+r2026:.2f}R (negative recent)")
    return flags


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    total_combos = len(CONSEC_N) * len(HOLD_BARS) * len(ATR_MULTS) * len(FILTER_MODES)
    p("=" * 70)
    p("  Phase T1 -- ConsecUpDays MR Short Concept Discovery")
    p("  Entry: N consecutive UP closes --> SHORT at bar close")
    p("  Exit: fixed time exit (MR profile)")
    p("  S4.2: 0.15R Futures cost floor")
    p(f"  Total combos: {total_combos}  |  Universe: {len(SYMBOLS)} symbols")
    p("=" * 70)

    loaded: list[tuple[str, pd.DataFrame]] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"\n  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")

    param_combos = list(itertools.product(CONSEC_N, HOLD_BARS, ATR_MULTS, FILTER_MODES))
    combo_trades: dict[tuple, list[dict]] = {k: [] for k in param_combos}

    for idx, (sym, df) in enumerate(loaded, 1):
        if idx == 1 or idx % 15 == 0 or idx == len(loaded):
            p(f"  Processing {idx}/{len(loaded)} {sym}...")
        for combo in param_combos:
            combo_trades[combo].extend(backtest_symbol(df, *combo))

    rows = []
    for (cn, hb, am, fm) in param_combos:
        m = calc_metrics(combo_trades[(cn, hb, am, fm)])
        rows.append({"consec_n": cn, "hold_bars": hb, "atr_mult": am,
                     "filter_mode": fm, "num_trades": m["n"],
                     "win_rate": round(m["win_rate"],4), "avg_r": round(m["avg_r"],4),
                     "pf": round(m["pf"],4), "max_dd_r": round(m["max_dd_r"],2),
                     "total_r": round(m["total_r"],2)})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "phase_t1_grid_1d.csv", index=False)
    p(f"  Grid saved: {len(grid)} rows")

    # ---------- Gate filters ----------
    p(f"\n--- Gate Analysis ---")
    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    p(f"  Combos >= 80 trades : {len(viable)}")
    viable_wr = viable[(viable["win_rate"] >= MR_GATES["win_rate_min"]) &
                       (viable["win_rate"] <= MR_GATES["win_rate_max"])]
    p(f"  After WR 50-70%     : {len(viable_wr)}")
    viable_ar = viable_wr[viable_wr["avg_r"] >= MR_GATES["avg_r_min"]].copy()
    p(f"  After avg_r >= 0.15R: {len(viable_ar)}")

    if viable_ar.empty:
        p(f"\n  No combos pass gates. Best by avg_r:")
        p(viable.nlargest(10,"avg_r")[["consec_n","hold_bars","atr_mult",
            "filter_mode","num_trades","win_rate","avg_r","pf"]].to_string(index=False))
        p(f"\n  T1 GATE: FAIL")
        with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
            f.write("ConsecUpDays MR Short T1 -- FAIL\n")
            f.write("No combos pass WR 50-70% AND avg_r > 0.15R\n")
        sys.exit(1)

    viable_ar["stability"] = viable_ar.apply(
        lambda r: stability_score(grid, int(r["consec_n"]),
                                  float(r["atr_mult"]), str(r["filter_mode"])),
        axis=1)
    stable = viable_ar[viable_ar["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"  After stability >= 0.67: {len(stable)}")

    if stable.empty:
        stable = viable_ar.nlargest(10, "stability")
    stable = stable.sort_values(["stability","avg_r"], ascending=[False,False]).reset_index(drop=True)

    # Heatmap per filter
    p(f"\n  avg_r by (consec_n, filter_mode) | all hold/atr averaged:")
    pivot = grid.groupby(["consec_n","filter_mode"])["avg_r"].mean().unstack("filter_mode")
    p(pivot.round(3).to_string())

    # Top combos table
    p(f"\n  Top stable combos:")
    p(f"  {'cN':>3} {'hold':>5} {'atr':>5} {'filter':>22} "
      f"{'trades':>7} {'WR%':>6} {'AvgR':>8} {'PF':>6} {'stab':>6}")
    p("  " + "-"*70)
    for _, row in stable.head(12).iterrows():
        p(f"  {int(row['consec_n']):>3} {int(row['hold_bars']):>5} "
          f"{row['atr_mult']:>5.1f} {row['filter_mode']:>22} "
          f"{int(row['num_trades']):>7} {row['win_rate']*100:>5.1f}% "
          f"{row['avg_r']:>+7.4f} {row['pf']:>6.2f} {row['stability']:>6.2f}")

    # Year-by-year + halt
    p(f"\n{'='*70}")
    p(f"  HALT CHECK -- Year-by-Year top combos")
    p(f"{'='*70}")
    halt_any = False
    for rank, (_, row) in enumerate(stable.head(5).iterrows(), 1):
        key    = (int(row["consec_n"]), int(row["hold_bars"]),
                  float(row["atr_mult"]), str(row["filter_mode"]))
        trades = combo_trades[key]
        yb     = year_breakdown(trades)
        m      = calc_metrics(trades)
        flags  = concentration_check(yb, m["total_r"])
        has_halt = any("HALT" in f for f in flags)
        if has_halt: halt_any = True
        status = "HALT" if has_halt else ("FLAG" if flags else "PASS")

        p(f"\n  #{rank}: consec{int(row['consec_n'])}/h{int(row['hold_bars'])}/"
          f"atr{row['atr_mult']}/{row['filter_mode']}  "
          f"avg_r={row['avg_r']:+.4f}R  WR={row['win_rate']*100:.1f}%  "
          f"stab={row['stability']:.2f}  [{status}]")
        for f in flags:
            p(f"  --> {f}")

        p(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotR':>8}  {'Frac%':>7}")
        for yr in sorted(yb.keys()):
            ym   = yb[yr]
            frac = ym["total_r"] / m["total_r"] if m["total_r"] != 0 else 0.0
            mark = " <<BEAR" if yr == 2022 else (" ***CONC" if frac > 0.50 else "")
            p(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  {frac:>+6.0%}{mark}")

    gate_str = "HALT" if halt_any else ("PASS" if not stable.empty else "FAIL")
    p(f"\n  T1 GATE: {gate_str}")
    stable.to_csv(OUT_DIR / "phase_t1_stability_ranking.csv", index=False)

    with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write("ConsecUpDays MR Short T1\n")
        f.write("Gates: WR 50-70%, avg_r > 0.15R (Futures), stability >= 0.67\n")
        f.write(f"T1 GATE: {gate_str}\n\n")
        cols = ["consec_n","hold_bars","atr_mult","filter_mode","num_trades",
                "win_rate","avg_r","pf","stability","total_r"]
        if all(c in stable.columns for c in cols):
            f.write(stable.head(10)[cols].to_string(index=False))
        f.write("\n\navg_r heatmap:\n")
        f.write(pivot.round(3).to_string())

    p(f"\n[OK] phase_t1_grid_1d.csv         ({len(grid)} rows)")
    p(f"[OK] phase_t1_stability_ranking.csv ({len(stable)} stable)")
    p(f"[OK] phase_t1_summary.txt")
    p(f"\nSTOPPING -- awaiting human review. T1 GATE: {gate_str}")
    sys.exit(0 if not halt_any and not stable.empty else 1)


if __name__ == "__main__":
    main()
