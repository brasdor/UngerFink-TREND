#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- IntraweekBias Concept Discovery

Enter at bar OPEN of a specific day-of-week (UTC).
Exit at bar CLOSE after hold_bars days (no stop in T1 — pure calendar bias).
1R = ATR(14) of prior bar (volatility normalizer).

Day encoding: Monday=0, Tuesday=1, ..., Sunday=6

Grid:
  1D: 7 days × 5 hold_bars × 2 sides × 2 filters = 140 combos

Stability (ALL THREE required for PASS):
  1. Adjacent-day zone: ≥ 2/3 of [day-1, day, day+1] (mod 7) profitable
  2. Asset coverage: ≥ 30% of symbols have avg_r > 0
  3. Year stability: ≥ 3 of [2022, 2023, 2024, 2025] profitable
     (year counted only if ≥ 20 trades that year)

§4.2 cost floor: 0.15R
Output: data/research_intraweekbias_t1/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT     = Path(__file__).parent
D1_CACHE = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR  = ROOT / "data" / "research_intraweekbias_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_DAYS   = [0, 1, 2, 3, 4, 5, 6]          # Mon=0 … Sun=6
DAY_NAMES    = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
HOLD_BARS    = [1, 2, 3, 4, 5]
SIDES        = ["LONG", "SHORT"]
FILTER_MODES = ["none", "ema200_price"]

ATR_N        = 14
EMA_N        = 200
COST_FLOOR_R = 0.15

STABILITY_YEARS    = [2022, 2023, 2024, 2025]
MIN_YR_PASS        = 3
MIN_YR_TRADES      = 20
ASSET_COVERAGE_MIN = 0.30


# =============================================================================
# INDICATORS
# =============================================================================

def compute_ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(close)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1 : n + 1]))
        for i in range(n + 1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i - 1]):
                atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


# =============================================================================
# DATA
# =============================================================================

def get_symbols() -> List[str]:
    syms = []
    for f in sorted(D1_CACHE.glob("*_1d.csv")):
        stem = f.stem[: -len("_1d")]
        sym  = stem.replace("_", "/", 1)
        if sym.endswith("/USDT"):
            syms.append(sym)
    return syms


def load_1d(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = D1_CACHE / f"{clean}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time", "open", "high", "low", "close"]].dropna()


# =============================================================================
# PER-SYMBOL BACKTEST
# =============================================================================

def process_symbol(df: pd.DataFrame, symbol: str) -> Dict[Tuple, dict]:
    df = df.sort_values("time").reset_index(drop=True)
    n  = len(df)

    op     = df["open"].to_numpy(dtype=float)
    cl     = df["close"].to_numpy(dtype=float)
    hi     = df["high"].to_numpy(dtype=float)
    lo     = df["low"].to_numpy(dtype=float)
    ema200 = compute_ema(cl, EMA_N)
    atr14  = compute_atr(hi, lo, cl, ATR_N)

    dow   = df["time"].dt.dayofweek.to_numpy(dtype=int)   # Mon=0..Sun=6
    years = df["time"].dt.year.to_numpy(dtype=int)

    # Valid: warm-up done, prior-bar ATR & EMA available
    valid = np.zeros(n, dtype=bool)
    for i in range(EMA_N + 1, n):
        valid[i] = (
            np.isfinite(op[i]) and
            np.isfinite(atr14[i - 1]) and atr14[i - 1] > 0 and
            np.isfinite(ema200[i - 1])
        )

    results: Dict[Tuple, dict] = {}

    for entry_day in ENTRY_DAYS:
        day_mask = (dow == entry_day) & valid
        ei = np.where(day_mask)[0]
        if len(ei) == 0:
            continue

        for hold in HOLD_BARS:
            xi = ei + hold - 1
            ok = xi < n
            ei_v = ei[ok];  xi_v = xi[ok]
            if len(ei_v) == 0:
                continue

            entry_op  = op[ei_v]
            exit_cl   = cl[xi_v]
            atr_prior = atr14[ei_v - 1]
            ema_prior = ema200[ei_v - 1]
            yr_vals   = years[ei_v]

            long_pnl  = exit_cl - entry_op
            short_pnl = entry_op - exit_cl

            for side in SIDES:
                raw_r = (long_pnl if side == "LONG" else short_pnl) / atr_prior

                for filt in FILTER_MODES:
                    if filt == "ema200_price":
                        fmask = (entry_op > ema_prior) if side == "LONG" else (entry_op < ema_prior)
                    else:
                        fmask = np.ones(len(ei_v), dtype=bool)

                    r   = raw_r[fmask]
                    yrs = yr_vals[fmask]
                    if len(r) == 0:
                        continue

                    yr_stats: Dict[int, Tuple[int, float]] = {}
                    for y in np.unique(yrs):
                        ym = yrs == y
                        yr_stats[int(y)] = (int(ym.sum()), float(r[ym].sum()))

                    key = (entry_day, hold, side, filt)
                    results[key] = {
                        "n":         int(len(r)),
                        "total_r":   float(r.sum()),
                        "pos_count": int((r > 0).sum()),
                        "yr":        yr_stats,
                    }

    return results


# =============================================================================
# AGGREGATION & STABILITY
# =============================================================================

def aggregate_all(per_sym: Dict[str, Dict[Tuple, dict]]) -> pd.DataFrame:
    from collections import defaultdict

    combo_n:    Dict[Tuple, int]   = defaultdict(int)
    combo_tot:  Dict[Tuple, float] = defaultdict(float)
    combo_pos:  Dict[Tuple, int]   = defaultdict(int)
    combo_yr:   Dict[Tuple, Dict]  = defaultdict(dict)
    combo_syms: Dict[Tuple, List]  = defaultdict(list)

    for sym, sym_res in per_sym.items():
        for key, val in sym_res.items():
            combo_n[key]   += val["n"]
            combo_tot[key] += val["total_r"]
            combo_pos[key] += val["pos_count"]
            combo_syms[key].append(val["total_r"] / val["n"] if val["n"] > 0 else 0.0)
            for yr, (yn, ytr) in val["yr"].items():
                if yr in combo_yr[key]:
                    en, et = combo_yr[key][yr]
                    combo_yr[key][yr] = (en + yn, et + ytr)
                else:
                    combo_yr[key][yr] = (yn, ytr)

    rows = []
    for key in sorted(combo_n.keys()):
        entry_day, hold, side, filt = key
        nn    = combo_n[key]
        total = combo_tot[key]
        pos   = combo_pos[key]
        avg_r = total / nn if nn > 0 else 0.0
        win_r = pos / nn   if nn > 0 else 0.0

        yr_profitable = {}
        for yr in STABILITY_YEARS:
            if yr in combo_yr[key]:
                yn, ytr = combo_yr[key][yr]
                yr_profitable[yr] = (ytr / yn > 0) if yn >= MIN_YR_TRADES else None
            else:
                yr_profitable[yr] = None

        n_yr_pass    = sum(1 for v in yr_profitable.values() if v is True)
        n_yr_counted = sum(1 for v in yr_profitable.values() if v is not None)

        sym_avgs  = combo_syms[key]
        asset_cov = sum(1 for x in sym_avgs if x > 0) / len(sym_avgs) if sym_avgs else 0.0

        rows.append({
            "entry_day":     entry_day,
            "day_name":      DAY_NAMES[entry_day],
            "hold_bars":     hold,
            "side":          side,
            "filter_mode":   filt,
            "trades":        nn,
            "avg_r":         avg_r,
            "total_r":       total,
            "win_rate":      win_r,
            "asset_coverage": asset_cov,
            "n_yr_pass":     n_yr_pass,
            "n_yr_counted":  n_yr_counted,
            "yr_2022":       yr_profitable.get(2022),
            "yr_2023":       yr_profitable.get(2023),
            "yr_2024":       yr_profitable.get(2024),
            "yr_2025":       yr_profitable.get(2025),
            "pass_cost_floor": avg_r >= COST_FLOOR_R,
        })

    return pd.DataFrame(rows)


def apply_stability(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["zone_pass_rate"]      = 0.0
    df["stability_zone_pass"] = False
    df["stability_pass"]      = False

    for (hold, side, filt), grp in df.groupby(["hold_bars", "side", "filter_mode"]):
        pass_map = dict(zip(grp["entry_day"], grp["pass_cost_floor"]))
        for idx, row in grp.iterrows():
            d     = int(row["entry_day"])
            prev_d = (d - 1) % 7
            next_d = (d + 1) % 7
            zone  = [pass_map.get(prev_d, False), bool(row["pass_cost_floor"]),
                     pass_map.get(next_d, False)]
            rate  = sum(zone) / 3.0
            df.at[idx, "zone_pass_rate"]      = rate
            df.at[idx, "stability_zone_pass"] = rate >= 2 / 3

            yr_ok    = row["n_yr_pass"] >= MIN_YR_PASS and row["n_yr_counted"] >= MIN_YR_PASS
            asset_ok = row["asset_coverage"] >= ASSET_COVERAGE_MIN
            df.at[idx, "stability_pass"] = (
                bool(df.at[idx, "stability_zone_pass"]) and yr_ok and asset_ok
            )

    return df


# =============================================================================
# HEATMAP
# =============================================================================

def build_heatmap(
    per_sym: Dict[str, Dict[Tuple, dict]],
    hold: int, side: str, filt: str,
) -> pd.DataFrame:
    symbols = sorted(per_sym.keys())
    data = {}
    for sym in symbols:
        col = []
        sym_res = per_sym.get(sym, {})
        for d in ENTRY_DAYS:
            key = (d, hold, side, filt)
            val = sym_res.get(key)
            col.append(val["total_r"] / val["n"] if val and val["n"] > 0 else np.nan)
        data[sym] = col
    hm = pd.DataFrame(data, index=[DAY_NAMES[d] for d in ENTRY_DAYS])
    hm.index.name = "entry_day"
    return hm


# =============================================================================
# REPORT
# =============================================================================

def write_report(results: pd.DataFrame) -> None:
    lines = [
        "PHASE T1 -- IntraweekBias Concept Discovery",
        "=" * 70,
        "",
        "Entry:   Open of 1D bar on entry_day (UTC)",
        "Exit:    Close of bar at entry_idx + hold_bars - 1",
        "1R:      ATR(14) of prior bar",
        "Days:    Mon=0 Tue=1 Wed=2 Thu=3 Fri=4 Sat=5 Sun=6",
        "",
        "STABILITY (ALL THREE required):",
        "  1. Adjacent-day zone: >=2/3 of [day-1, day, day+1] pass §4.2",
        "  2. Asset coverage: >=30% of symbols have avg_r > 0",
        "  3. Year stability: >=3 of [2022,2023,2024,2025] profitable",
        f"§4.2 cost floor: {COST_FLOOR_R}R",
        "",
        "=" * 70, "RESULTS BY FILTER", "=" * 70, "",
    ]

    for filt in FILTER_MODES:
        lines += [f"Filter: {filt}", ""]
        fsub = results[results["filter_mode"] == filt]
        for side in SIDES:
            ssub = fsub[fsub["side"] == side]
            if ssub.empty:
                continue
            lines.append(
                f"  {side}  {'DAY':4s} {'HB':2s}  {'N':5s}  "
                f"{'AVG_R':7s}  {'WIN%':5s}  {'COV%':5s}  "
                f"{'YR_OK':5s}  PASS  STAB"
            )
            lines.append("  " + "-" * 62)
            for _, r in ssub.sort_values(["hold_bars", "entry_day"]).iterrows():
                yr_ok = r["n_yr_pass"] >= MIN_YR_PASS and r["n_yr_counted"] >= MIN_YR_PASS
                lines.append(
                    f"  {r['day_name']:4s} {int(r['hold_bars']):2d}"
                    f"  {int(r['trades']):5d}"
                    f"  {r['avg_r']:+.4f}"
                    f"  {r['win_rate']:.1%}"
                    f"  {r['asset_coverage']:.0%}"
                    f"  {'Y' if yr_ok else 'N':5s}"
                    f"  {'PASS' if r['pass_cost_floor'] else '    ':4s}"
                    f"  {'STAB' if r['stability_pass'] else '    ':4s}"
                )
            lines.append("")

    # Stability ranking
    stable = results[results["stability_pass"]].sort_values("avg_r", ascending=False)
    lines += ["=" * 70, "STABILITY RANKING (all 3 conditions + §4.2)", "=" * 70, ""]
    if stable.empty:
        lines.append("  No combo passes all three stability conditions AND §4.2.")
    else:
        for _, r in stable.iterrows():
            lines.append(
                f"  [{r['day_name']}|hb={int(r['hold_bars'])}|{r['side']:5s}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  win={r['win_rate']:.1%}"
                f"  cov={r['asset_coverage']:.0%}"
                f"  yr={int(r['n_yr_pass'])}/{int(r['n_yr_counted'])}"
            )

    # Top by avg_r (any stability)
    top = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False)
    lines += [
        "", "=" * 70,
        "TOP COMBOS BY avg_r (pass_cost_floor=True, any stability)",
        "=" * 70, "",
    ]
    if top.empty:
        lines.append("  None pass §4.2 cost floor.")
    else:
        for _, r in top.head(20).iterrows():
            lines.append(
                f"  [{r['day_name']}|hb={int(r['hold_bars'])}|{r['side']:5s}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  win={r['win_rate']:.1%}"
                f"  cov={r['asset_coverage']:.0%}"
                f"  yr={int(r['n_yr_pass'])}/{int(r['n_yr_counted'])}"
                f"  {'STAB' if r['stability_pass'] else '    '}"
            )

    lines += [
        "", "=" * 70, "NEXT STEPS", "=" * 70, "",
        "  T2 candidates: stability_pass=True AND avg_r >= 0.15R",
        "  Do not proceed to T2 without human review.",
    ]

    rpt = OUT_DIR / "phase_t1_intraweekbias_summary.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T1 -- IntraweekBias Concept Discovery")
    print("=" * 70)

    symbols = get_symbols()
    print(f"Universe: {len(symbols)} symbols  (1D cache)")
    print(f"Grid: 7 days x {HOLD_BARS} hold_bars x {SIDES} x {FILTER_MODES}")
    print(f"Output: {OUT_DIR}")
    print()

    per_sym: Dict[str, Dict[Tuple, dict]] = {}
    loaded = skipped = 0

    for sym in symbols:
        raw = load_1d(sym)
        if raw is None or len(raw) < EMA_N + 100:
            skipped += 1
            continue
        sym_res = process_symbol(raw, sym)
        if sym_res:
            per_sym[sym] = sym_res
            loaded += 1
        else:
            skipped += 1

    print(f"Loaded={loaded}  Skipped={skipped}")

    if not per_sym:
        print("[ERROR] No data loaded.")
        return 1

    results = apply_stability(aggregate_all(per_sym))
    results.to_csv(OUT_DIR / "phase_t1_intraweekbias_results.csv", index=False)

    # Heatmaps: hold=1 for raw daily signal scan
    for side in SIDES:
        for filt in ["none", "ema200_price"]:
            hm  = build_heatmap(per_sym, hold=1, side=side, filt=filt)
            tag = f"{side.lower()}_{filt}"
            hm.to_csv(OUT_DIR / f"phase_t1_heatmap_1d_{tag}_hold1.csv")
    print("[OK] Heatmaps written")

    write_report(results)

    n_pass = int(results["pass_cost_floor"].sum())
    n_stab = int(results["stability_pass"].sum())
    print(f"\nTotal combos : {len(results)}")
    print(f"Pass cost floor: {n_pass}")
    print(f"Full stability : {n_stab}")

    print("\n" + "=" * 70)
    print("TOP 10 by avg_r (pass_cost_floor=True)")
    print("=" * 70)
    top = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False).head(10)
    if top.empty:
        print("  None pass §4.2 cost floor.")
    else:
        for _, r in top.iterrows():
            print(
                f"  [{r['day_name']}|hb={int(r['hold_bars'])}|{r['side']:5s}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  win={r['win_rate']:.1%}"
                f"  cov={r['asset_coverage']:.0%}"
                f"  yr={int(r['n_yr_pass'])}/{int(r['n_yr_counted'])}"
                f"  {'STAB' if r['stability_pass'] else '    '}"
            )

    print()
    print("=" * 70)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
