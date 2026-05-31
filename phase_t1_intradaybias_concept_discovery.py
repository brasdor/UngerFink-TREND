#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- IntradayBias Concept Discovery

Enter at bar OPEN of a specific UTC hour.
Exit at bar CLOSE after hold_bars bars (no stop in T1 — pure time bias).
1R = ATR(14) of prior bar (volatility normalizer).

Grid:
  1H: 24 entry hours × 4 hold_bars × 2 sides × 3 filters = 576 combos
  2H: 12 entry hours × 4 × 2 × 3                         = 288 combos
  4H:  6 entry hours × 4 × 2 × 3                         =  144 combos
  Total: 1008 combos

Stability (ALL THREE required for PASS):
  1. Adjacent-hour zone: ≥ 2/3 of [hour-step, hour, hour+step] profitable
  2. Asset coverage: ≥ 30% of symbols have avg_r > 0
  3. Year stability: ≥ 3 of years [2022,2023,2024,2025] profitable
     (year counted only if ≥ 20 trades that year)

§4.2 cost floor: 0.15R
Output: data/research_intradaybias_t1/
"""

from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    sys.exit("ccxt not installed: pip install ccxt")


# =============================================================================
# CONFIG
# =============================================================================

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "research_intradaybias_t1" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_intradaybias_t1"
D1_CACHE  = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
for d in (CACHE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 4+ years per TF
TF_TARGET_BARS = {"1h": 40_000, "2h": 20_000, "4h": 10_000}
TF_MS          = {"1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000}

TF_ENTRY_HOURS = {
    "1h": list(range(24)),
    "2h": list(range(0, 24, 2)),
    "4h": list(range(0, 24, 4)),
}

HOLD_BARS    = [1, 2, 3, 4]
SIDES        = ["LONG", "SHORT"]
FILTER_MODES = ["none", "ema200_price", "weekday"]

ATR_N        = 14
EMA_N        = 200
COST_FLOOR_R = 0.15

STABILITY_YEARS    = [2022, 2023, 2024, 2025]
MIN_YR_PASS        = 3
MIN_YR_TRADES      = 20   # min trades in a year to count it
ASSET_COVERAGE_MIN = 0.30  # ≥ 30% of symbols profitable


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
# DATA LOADING
# =============================================================================

def get_symbols() -> List[str]:
    syms = []
    for f in sorted(D1_CACHE.glob("*_1d.csv")):
        stem = f.stem[: -len("_1d")]
        sym  = stem.replace("_", "/", 1)
        if sym.endswith("/USDT"):
            syms.append(sym)
    return syms


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    needed = {"open", "high", "low", "close", "volume"}
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df[["time", "open", "high", "low", "close", "volume"]].dropna(subset=["time"])


def _download(exchange, symbol: str, tf: str, target_bars: int) -> Optional[pd.DataFrame]:
    bar_ms  = TF_MS[tf]
    now_ms  = int(pd.Timestamp.utcnow().timestamp() * 1_000)
    since   = now_ms - target_bars * bar_ms
    limit   = 1_000
    rows: List = []

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        except Exception as exc:
            print(f"    [WARN] {symbol} {tf}: {exc}")
            time_mod.sleep(1.0)
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < limit or last_ts >= now_ms:
            break
        since = last_ts + 1
        time_mod.sleep(0.06)

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def get_ohlcv(symbol: str, tf: str, exchange) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = CACHE_DIR / f"{clean}_{tf}.csv"
    if path.exists():
        df = _load_csv(path)
        # Re-download if cached data is too short (< 50% of target)
        min_bars = TF_TARGET_BARS[tf] // 2
        if len(df) >= min_bars:
            return df
    df = _download(exchange, symbol, tf, TF_TARGET_BARS[tf])
    if df is not None and len(df) > 100:
        df.to_csv(path, index=False)
    return df


# =============================================================================
# PER-SYMBOL BACKTEST
# =============================================================================

def process_symbol(
    df: pd.DataFrame, symbol: str, tf: str
) -> Dict[Tuple, dict]:
    """
    Returns dict: key=(entry_hour, hold_bars, side, filter_mode)
                  val={"n": int, "total_r": float, "pos_r": float,
                       "yr": {year: (n, total_r)},
                       "pos_count": int}
    """
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)

    op    = df["open"].to_numpy(dtype=float)
    cl    = df["close"].to_numpy(dtype=float)
    hi    = df["high"].to_numpy(dtype=float)
    lo    = df["low"].to_numpy(dtype=float)
    ema200 = compute_ema(cl, EMA_N)
    atr14  = compute_atr(hi, lo, cl, ATR_N)

    hours   = df["time"].dt.hour.to_numpy(dtype=int)
    years   = df["time"].dt.year.to_numpy(dtype=int)
    weekday = df["time"].dt.dayofweek.to_numpy(dtype=int)

    # Valid bars: EMA warm-up done, ATR valid, prior-bar data available
    valid_base = np.zeros(n, dtype=bool)
    for i in range(EMA_N + 1, n):
        valid_base[i] = (
            np.isfinite(op[i]) and np.isfinite(atr14[i - 1]) and
            atr14[i - 1] > 0 and np.isfinite(ema200[i - 1])
        )

    results: Dict[Tuple, dict] = {}

    for entry_hour in TF_ENTRY_HOURS[tf]:
        # Mask: bars starting at entry_hour, past warm-up
        hour_mask = (hours == entry_hour) & valid_base
        ei = np.where(hour_mask)[0]
        if len(ei) == 0:
            continue

        for hold in HOLD_BARS:
            xi = ei + hold - 1           # exit bar indices (hold=1 → same bar)
            ok = xi < n
            ei_v = ei[ok]
            xi_v = xi[ok]
            if len(ei_v) == 0:
                continue

            entry_op   = op[ei_v]
            exit_cl    = cl[xi_v]
            atr_prior  = atr14[ei_v - 1]
            ema_prior  = ema200[ei_v - 1]
            yr_vals    = years[ei_v]
            wd_vals    = weekday[ei_v]

            # raw pnl in price units
            long_pnl  = exit_cl - entry_op
            short_pnl = entry_op - exit_cl

            for side in SIDES:
                raw_r = (long_pnl if side == "LONG" else short_pnl) / atr_prior

                for filt in FILTER_MODES:
                    if filt == "ema200_price":
                        if side == "LONG":
                            fmask = entry_op > ema_prior
                        else:
                            fmask = entry_op < ema_prior
                    elif filt == "weekday":
                        fmask = wd_vals < 5    # Mon–Fri
                    else:
                        fmask = np.ones(len(ei_v), dtype=bool)

                    r    = raw_r[fmask]
                    yrs  = yr_vals[fmask]

                    if len(r) == 0:
                        continue

                    yr_stats: Dict[int, Tuple[int, float]] = {}
                    for y in np.unique(yrs):
                        ym = yrs == y
                        yr_stats[int(y)] = (int(ym.sum()), float(r[ym].sum()))

                    key = (entry_hour, hold, side, filt)
                    results[key] = {
                        "n":         int(len(r)),
                        "total_r":   float(r.sum()),
                        "pos_count": int((r > 0).sum()),
                        "yr":        yr_stats,
                    }

    return results


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_all(
    per_sym: Dict[str, Dict[Tuple, dict]],
    tf: str,
) -> pd.DataFrame:
    """
    Collect per-symbol results into a DataFrame of combo-level stats.
    Also returns per-symbol avg_r for asset-coverage stability.
    """
    from collections import defaultdict

    combo_n:     Dict[Tuple, int]   = defaultdict(int)
    combo_total: Dict[Tuple, float] = defaultdict(float)
    combo_pos:   Dict[Tuple, int]   = defaultdict(int)
    combo_yr:    Dict[Tuple, Dict[int, Tuple[int, float]]] = defaultdict(dict)
    combo_syms:  Dict[Tuple, List[float]] = defaultdict(list)  # per-symbol avg_r

    for sym, sym_results in per_sym.items():
        for key, val in sym_results.items():
            combo_n[key]     += val["n"]
            combo_total[key] += val["total_r"]
            combo_pos[key]   += val["pos_count"]
            # Per-symbol avg_r for asset coverage check
            combo_syms[key].append(val["total_r"] / val["n"] if val["n"] > 0 else 0.0)
            # Accumulate per-year stats
            for yr, (yn, ytr) in val["yr"].items():
                if yr in combo_yr[key]:
                    existing_yn, existing_ytr = combo_yr[key][yr]
                    combo_yr[key][yr] = (existing_yn + yn, existing_ytr + ytr)
                else:
                    combo_yr[key][yr] = (yn, ytr)

    rows = []
    for key in sorted(combo_n.keys()):
        entry_hour, hold, side, filt = key
        nn    = combo_n[key]
        total = combo_total[key]
        pos   = combo_pos[key]
        avg_r = total / nn if nn > 0 else 0.0
        win_r = pos / nn   if nn > 0 else 0.0

        # Per-year profitability
        yr_profitable = {}
        for yr in STABILITY_YEARS:
            if yr in combo_yr[key]:
                yn, ytr = combo_yr[key][yr]
                if yn >= MIN_YR_TRADES:
                    yr_profitable[yr] = (ytr / yn) > 0
                else:
                    yr_profitable[yr] = None  # not enough trades
            else:
                yr_profitable[yr] = None

        n_yr_pass = sum(1 for v in yr_profitable.values() if v is True)
        n_yr_counted = sum(1 for v in yr_profitable.values() if v is not None)

        # Asset coverage
        sym_avgs = combo_syms[key]
        n_pos_sym = sum(1 for x in sym_avgs if x > 0)
        n_sym     = len(sym_avgs)
        asset_cov = n_pos_sym / n_sym if n_sym > 0 else 0.0

        rows.append({
            "timeframe":     tf,
            "entry_hour":    entry_hour,
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
    """
    Stability (ALL THREE required):
      1. Adjacent-hour zone: ≥ 2/3 of [prev, curr, next] hours profitable
      2. Asset coverage ≥ 30%
      3. Year stability: ≥ 3/4 years profitable (with ≥ MIN_YR_TRADES)
    """
    df = results.copy()
    df["zone_pass_rate"]      = 0.0
    df["stability_zone_pass"] = False
    df["stability_pass"]      = False

    for (tf, hold, side, filt), grp in df.groupby(
        ["timeframe", "hold_bars", "side", "filter_mode"]
    ):
        hours_list = TF_ENTRY_HOURS[tf]
        n_h = len(hours_list)
        pass_map = dict(zip(grp["entry_hour"], grp["pass_cost_floor"]))

        for idx, row in grp.iterrows():
            h_idx = hours_list.index(int(row["entry_hour"]))
            prev_h = hours_list[(h_idx - 1) % n_h]
            next_h = hours_list[(h_idx + 1) % n_h]
            zone_flags = [
                pass_map.get(prev_h, False),
                bool(row["pass_cost_floor"]),
                pass_map.get(next_h, False),
            ]
            zone_rate = sum(zone_flags) / 3.0
            df.at[idx, "zone_pass_rate"]      = zone_rate
            df.at[idx, "stability_zone_pass"] = zone_rate >= 2 / 3

            # Year stability
            yr_ok = row["n_yr_pass"] >= MIN_YR_PASS and row["n_yr_counted"] >= MIN_YR_PASS

            # Asset coverage
            asset_ok = row["asset_coverage"] >= ASSET_COVERAGE_MIN

            # Full stability
            df.at[idx, "stability_pass"] = (
                bool(df.at[idx, "stability_zone_pass"]) and yr_ok and asset_ok
            )

    return df


# =============================================================================
# HEATMAP
# =============================================================================

def build_heatmap(
    per_sym: Dict[str, Dict[Tuple, dict]],
    tf: str, hold: int, side: str, filt: str,
) -> pd.DataFrame:
    """Build avg_r heatmap: rows=entry_hours, cols=symbols."""
    hours = TF_ENTRY_HOURS[tf]
    symbols = sorted(per_sym.keys())
    data = {}
    for sym in symbols:
        col = []
        sym_res = per_sym.get(sym, {})
        for h in hours:
            key = (h, hold, side, filt)
            val = sym_res.get(key)
            if val and val["n"] > 0:
                col.append(val["total_r"] / val["n"])
            else:
                col.append(np.nan)
        data[sym] = col
    hm = pd.DataFrame(data, index=hours)
    hm.index.name = "entry_hour_utc"
    return hm


# =============================================================================
# REPORT
# =============================================================================

def write_report(results: pd.DataFrame) -> None:
    lines = [
        "PHASE T1 -- IntradayBias Concept Discovery",
        "=" * 70,
        "",
        "Entry:   Open of bar at entry_hour_utc",
        "Exit:    Close of bar at entry_idx + hold_bars - 1",
        "1R:      ATR(14) of prior bar (no stop applied in T1)",
        "Filter:  none | ema200_price | weekday (Mon–Fri only)",
        "",
        "STABILITY (ALL THREE required):",
        "  1. Adjacent-hour zone: ≥ 2/3 of [H-step, H, H+step] pass §4.2",
        "  2. Asset coverage: ≥ 30% of symbols have avg_r > 0",
        "  3. Year stability: ≥ 3 of [2022,2023,2024,2025] profitable",
        f"§4.2 cost floor: {COST_FLOOR_R}R",
        "",
    ]

    for tf in ["1h", "2h", "4h"]:
        sub = results[results["timeframe"] == tf]
        if sub.empty:
            continue
        lines += [f"{'='*70}", f"TIMEFRAME: {tf}", f"{'='*70}", ""]

        for filt in FILTER_MODES:
            lines += [f"  Filter: {filt}", ""]
            fsub = sub[sub["filter_mode"] == filt]
            for side in SIDES:
                ssub = fsub[fsub["side"] == side]
                if ssub.empty:
                    continue
                lines.append(
                    f"    {side}  "
                    f"{'HOUR':4s}  {'HB':2s}  {'N':6s}  "
                    f"{'AVG_R':7s}  {'WIN%':5s}  {'COV%':5s}  "
                    f"{'YR_OK':5s}  PASS  STAB"
                )
                lines.append("    " + "-" * 68)
                for _, r in ssub.sort_values(["hold_bars", "entry_hour"]).iterrows():
                    yr_ok = r["n_yr_pass"] >= MIN_YR_PASS and r["n_yr_counted"] >= MIN_YR_PASS
                    lines.append(
                        f"    {int(r['entry_hour']):4d}  {int(r['hold_bars']):2d}"
                        f"  {int(r['trades']):6d}"
                        f"  {r['avg_r']:+.4f}"
                        f"  {r['win_rate']:.1%}"
                        f"  {r['asset_coverage']:.0%}"
                        f"  {'Y' if yr_ok else 'N':5s}"
                        f"  {'PASS' if r['pass_cost_floor'] else '    ':4s}"
                        f"  {'STAB' if r['stability_pass'] else '    ':4s}"
                    )
                lines.append("")

    # Top combos
    stable = results[results["stability_pass"]].sort_values("avg_r", ascending=False)
    lines += ["=" * 70, "STABILITY RANKING (all 3 conditions met, avg_r ≥ 0.15R)", "=" * 70, ""]
    if stable.empty:
        lines.append("  No combo passes all three stability conditions AND §4.2.")
    else:
        for _, r in stable.head(20).iterrows():
            lines.append(
                f"  [{r['timeframe']:2s}|{r['side']:5s}|h={int(r['entry_hour']):02d}|hb={int(r['hold_bars'])}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  cov={r['asset_coverage']:.0%}"
                f"  yr={int(r['n_yr_pass'])}/{int(r['n_yr_counted'])}  zone={r['zone_pass_rate']:.0%}"
            )

    # Best avg_r regardless of full stability (diagnostics)
    lines += [
        "", "=" * 70,
        "TOP 20 BY avg_r (pass_cost_floor only, any stability)",
        "=" * 70, "",
    ]
    top_pass = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False)
    if top_pass.empty:
        lines.append("  None pass §4.2 cost floor.")
    else:
        for _, r in top_pass.head(20).iterrows():
            lines.append(
                f"  [{r['timeframe']:2s}|{r['side']:5s}|h={int(r['entry_hour']):02d}|hb={int(r['hold_bars'])}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  cov={r['asset_coverage']:.0%}"
                f"  yr={int(r['n_yr_pass'])}/{int(r['n_yr_counted'])}"
                f"  {'STAB' if r['stability_pass'] else '    '}"
            )

    lines += [
        "", "=" * 70, "NEXT STEPS", "=" * 70, "",
        "  T2 candidates: rows where stability_pass=True AND avg_r ≥ 0.15R",
        "  Review heatmap CSVs to see which hours show cross-asset patterns.",
        "  Do not proceed to T2 without human review.",
    ]

    rpt = OUT_DIR / "phase_t1_intradaybias_summary.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T1 -- IntradayBias Concept Discovery")
    print("=" * 70)

    symbols = get_symbols()
    print(f"Universe: {len(symbols)} symbols")
    print(f"Grid: TFs=1h/2h/4h  hours=24/12/6  hold={HOLD_BARS}  sides={SIDES}  filters={FILTER_MODES}")
    print(f"Output: {OUT_DIR}")
    print()

    exchange = ccxt.binance({"enableRateLimit": True})

    all_results: List[pd.DataFrame] = []

    for tf in ["1h", "2h", "4h"]:
        print(f"[TF={tf}] Downloading + processing...")
        per_sym: Dict[str, Dict[Tuple, dict]] = {}
        loaded = skipped = 0

        for sym in symbols:
            raw = get_ohlcv(sym, tf, exchange)
            if raw is None or len(raw) < EMA_N + 200:
                skipped += 1
                continue
            sym_res = process_symbol(raw, sym, tf)
            if sym_res:
                per_sym[sym] = sym_res
                loaded += 1
            else:
                skipped += 1

        print(f"  Loaded={loaded}  Skipped={skipped}")

        if not per_sym:
            print(f"  [WARN] No data for {tf}, skipping.")
            continue

        tf_results = aggregate_all(per_sym, tf)
        tf_results = apply_stability(tf_results)
        all_results.append(tf_results)

        # Heatmaps for this TF (hold=1, no filter)
        for side in SIDES:
            hm = build_heatmap(per_sym, tf, hold=1, side=side, filt="none")
            hm_path = OUT_DIR / f"phase_t1_heatmap_{tf}_{side.lower()}_hold1_none.csv"
            hm.to_csv(hm_path)
            print(f"  [OK] Heatmap: {hm_path.name}")

        # Also heatmap for hold=1, ema200_price filter, LONG (useful secondary view)
        hm_ema = build_heatmap(per_sym, tf, hold=1, side="LONG", filt="ema200_price")
        hm_ema.to_csv(OUT_DIR / f"phase_t1_heatmap_{tf}_long_hold1_ema200.csv")

    if not all_results:
        print("[ERROR] No results produced.")
        return 1

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_DIR / "phase_t1_intradaybias_results.csv", index=False)

    write_report(results)

    # Console summary
    n_pass = results["pass_cost_floor"].sum()
    n_stab = results["stability_pass"].sum()
    print(f"\nTotal combos: {len(results)}")
    print(f"Pass §4.2 (avg_r ≥ 0.15R): {n_pass}")
    print(f"Pass full stability:        {n_stab}")

    print("\n" + "=" * 70)
    print("TOP 10 by avg_r (pass_cost_floor=True)")
    print("=" * 70)
    top = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False).head(10)
    if top.empty:
        print("  None pass §4.2 cost floor.")
    else:
        for _, r in top.iterrows():
            print(
                f"  [{r['timeframe']:2s}|{r['side']:5s}|h={int(r['entry_hour']):02d}|hb={int(r['hold_bars'])}|{r['filter_mode']:12s}]"
                f"  avg_r={r['avg_r']:+.4f}  cov={r['asset_coverage']:.0%}"
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
