#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- ATRChannel Concept Discovery

Entry: close > SMA(N) + ATR(N) × atr_channel_mult  (prior bar, no lookahead)
Filter: ema200_price  (close > EMA200)
Exit:   close < SMA(N) (midline)  OR  initial stop (entry - ATR(N) × stop_mult)
Side:   LONG only (Binance Spot)

Parameter grid:
  N:               [20, 50, 100, 150, 200, 350]
  atr_channel_mult: [3.0, 5.0, 7.0, 9.0]
  stop_mult:        [2.0, 3.0]
  filter:           ema200_price

Stability zone: N ± 1 step in the grid; PASS if ≥ 67% of zone profitable.
§4.2 cost floor: 0.15R.
December 2024 concentration flag: top_month_r > 50% of total R.
Output: data/research_atrchannel_t1/
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

ROOT     = Path(__file__).resolve().parents[1]
D1_CACHE = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
OUT_DIR  = ROOT / "data" / "research_atrchannel_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES         = [20, 50, 100, 150, 200, 350]
ATR_CHANNEL_MULTS = [3.0, 5.0, 7.0, 9.0]
STOP_MULTS       = [2.0, 3.0]
FILTER_MODES     = ["ema200_price"]

EMA_N        = 200
COST_FLOOR_R = 0.15
MIN_TRADES   = 20
CONC_FLAG_PCT = 0.50   # flag if top month > 50% of total R


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


def compute_sma(close: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average via pandas for speed."""
    return pd.Series(close).rolling(n, min_periods=n).mean().to_numpy()


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothed ATR."""
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
    return df[["time", "open", "high", "low", "close"]].dropna().sort_values("time").reset_index(drop=True)


# =============================================================================
# BACKTEST (single combo)
# =============================================================================

def run_backtest(
    close: np.ndarray,
    low: np.ndarray,
    sma_n: np.ndarray,
    atr_n: np.ndarray,
    ema200: np.ndarray,
    atr_channel_mult: float,
    stop_mult: float,
    times: np.ndarray,
    symbol: str,
    n_val: int,
) -> List[dict]:
    trades: List[dict] = []
    pos: Optional[dict] = None

    for i in range(1, len(close)):
        if not (np.isfinite(sma_n[i - 1]) and np.isfinite(atr_n[i - 1]) and
                np.isfinite(ema200[i - 1]) and np.isfinite(close[i]) and
                np.isfinite(low[i])):
            continue

        upper_band = sma_n[i - 1] + atr_n[i - 1] * atr_channel_mult

        # --- EXIT ---
        if pos is not None:
            exit_px     = None
            exit_reason = None

            if low[i] <= pos["stop"]:
                exit_px     = float(pos["stop"])
                exit_reason = "initial_stop"
            elif close[i] < sma_n[i - 1]:
                exit_px     = float(close[i])
                exit_reason = "midline_exit"

            if exit_px is not None:
                risk = pos["entry"] - pos["stop"]
                if risk > 0:
                    trades.append({
                        "symbol":        symbol,
                        "n":             n_val,
                        "atr_ch_mult":   atr_channel_mult,
                        "stop_mult":     stop_mult,
                        "filter_mode":   "ema200_price",
                        "entry_time":    pos["entry_time"],
                        "exit_time":     times[i],
                        "entry_price":   pos["entry"],
                        "exit_price":    exit_px,
                        "initial_stop":  pos["stop"],
                        "exit_reason":   exit_reason,
                        "net_r":         (exit_px - pos["entry"]) / risk,
                    })
                pos = None

        # --- ENTRY ---
        if pos is None:
            if close[i] > ema200[i - 1] and close[i] > upper_band:
                stop_px = close[i] - atr_n[i - 1] * stop_mult
                if stop_px < close[i]:
                    pos = {
                        "entry":      float(close[i]),
                        "stop":       float(stop_px),
                        "entry_time": times[i],
                    }

    return trades


# =============================================================================
# AGGREGATION
# =============================================================================

def concentration_info(trades_df: pd.DataFrame) -> Tuple[str, float, bool]:
    """Returns (top_month, top_month_pct, dec2024_flag)."""
    if trades_df.empty:
        return ("", 0.0, False)
    month_r = trades_df.groupby("month")["net_r"].sum()
    total_r = month_r.sum()
    if total_r <= 0:
        return ("", 0.0, False)
    top_month = str(month_r.idxmax())
    top_pct   = float(month_r.max() / total_r)
    dec2024   = bool(top_month == "2024-12" and top_pct > CONC_FLAG_PCT)
    return (top_month, top_pct, dec2024)


def period_split_avgs(r: np.ndarray) -> Tuple[float, float]:
    """Return (first_half_avg_r, second_half_avg_r) split by trade count."""
    if len(r) < 2:
        return (float(r.mean()), float(r.mean()))
    mid = len(r) // 2
    return (float(r[:mid].mean()), float(r[mid:].mean()))


def aggregate(all_trades: List[dict]) -> pd.DataFrame:
    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df["month"]     = df["exit_time"].dt.to_period("M").astype(str)
    df = df.sort_values("exit_time").reset_index(drop=True)

    rows = []
    for key, g in df.groupby(["n", "atr_ch_mult", "stop_mult", "filter_mode"], sort=True):
        g = g.sort_values("exit_time")
        r = g["net_r"].to_numpy(dtype=float)
        gw = r[r > 0].sum();  gl = -r[r < 0].sum()
        pf = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
        eq = np.cumsum(r);  peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())

        top_month, top_pct, dec2024_flag = concentration_info(g)
        fh_avg, sh_avg = period_split_avgs(r)

        rows.append({
            "n":                    int(key[0]),
            "atr_ch_mult":          key[1],
            "stop_mult":            key[2],
            "filter_mode":          key[3],
            "trades":               int(len(r)),
            "total_r":              float(r.sum()),
            "avg_r":                float(r.mean()),
            "win_rate":             float((r > 0).mean()),
            "profit_factor":        pf,
            "max_dd_r":             dd,
            "first_half_avg_r":     fh_avg,
            "second_half_avg_r":    sh_avg,
            "second_half_negative": bool(sh_avg < 0),
            "top_month":            top_month,
            "top_month_pct":        top_pct,
            "dec2024_flag":         dec2024_flag,
            "pass_cost_floor":      bool(r.mean() >= COST_FLOOR_R and len(r) >= MIN_TRADES),
        })

    return pd.DataFrame(rows)


def stability_zones(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["zone_pass_rate"]      = 0.0
    df["stability_zone_pass"] = False

    for (atr_mult, stop_mult, filt), grp in df.groupby(
        ["atr_ch_mult", "stop_mult", "filter_mode"]
    ):
        pass_map = dict(zip(grp["n"], grp["pass_cost_floor"]))
        for idx, row in grp.iterrows():
            n_idx = N_VALUES.index(int(row["n"]))
            zone  = [N_VALUES[n_idx]]
            if n_idx > 0:
                zone.insert(0, N_VALUES[n_idx - 1])
            if n_idx < len(N_VALUES) - 1:
                zone.append(N_VALUES[n_idx + 1])
            rate = sum(pass_map.get(z, False) for z in zone) / len(zone)
            df.at[idx, "zone_pass_rate"]      = rate
            df.at[idx, "stability_zone_pass"] = rate >= 2 / 3

    return df


# =============================================================================
# REPORT
# =============================================================================

def write_report(results: pd.DataFrame, total_symbols: int) -> None:
    lines = [
        "PHASE T1 -- ATRChannel Concept Discovery",
        "=" * 70,
        "",
        "Method:  SMA(N) + ATR(N)*mult upper channel breakout",
        "Entry:   close > SMA(N)[i-1] + ATR(N)[i-1] * atr_channel_mult",
        "Filter:  ema200_price (close > EMA200)",
        "Exit:    close < SMA(N) (midline) OR initial stop (ATR(N)*stop_mult)",
        "Side:    LONG only",
        f"Symbols: {total_symbols}",
        "",
        f"§4.2 cost floor: {COST_FLOOR_R}R  (min {MIN_TRADES} trades)",
        "Dec-2024 flag:   top_month_r > 50% of total R AND month = 2024-12",
        "",
        "=" * 70,
        "FULL RESULTS TABLE",
        "=" * 70,
        "",
        f"  {'N':>4}  {'ch_m':>5}  {'sm':>4}  {'N_tr':>5}  {'avg_r':>7}  "
        f"{'PF':>5}  {'win%':>5}  {'DD':>7}  PASS  ZONE  TOP_MONTH  DEC24",
        "  " + "-" * 80,
    ]

    for _, r in results.sort_values(["n", "atr_ch_mult", "stop_mult"]).iterrows():
        dec_flag = " DEC24!" if r["dec2024_flag"] else ""
        lines.append(
            f"  {int(r['n']):>4}  {r['atr_ch_mult']:>5.1f}  {r['stop_mult']:>4.1f}"
            f"  {int(r['trades']):>5}  {r['avg_r']:>+7.4f}"
            f"  {r['profit_factor']:>5.3f}  {r['win_rate']:>5.1%}"
            f"  {r['max_dd_r']:>+7.2f}"
            f"  {'PASS' if r['pass_cost_floor'] else '    ':4s}"
            f"  {'ZONE' if r['stability_zone_pass'] else '    ':4s}"
            f"  {r['top_month']:>8s}  {r['top_month_pct']:>5.1%}{dec_flag}"
        )

    # Stability ranking
    stable = results[results["stability_zone_pass"] & results["pass_cost_floor"]].sort_values(
        "avg_r", ascending=False
    )
    lines += ["", "=" * 70, "STABILITY RANKING (§4.2 + zone_pass)", "=" * 70, ""]
    if stable.empty:
        lines.append("  No combo passes both §4.2 cost floor and stability zone.")
    else:
        for _, r in stable.iterrows():
            dec_flag = "  *** DEC2024 CONCENTRATION FLAG ***" if r["dec2024_flag"] else ""
            lines.append(
                f"  [N={int(r['n'])} / ch_mult={r['atr_ch_mult']:.1f} / sm={r['stop_mult']:.1f}]"
                f"  trades={int(r['trades'])}  avg_r={r['avg_r']:+.4f}"
                f"  PF={r['profit_factor']:.3f}  win={r['win_rate']:.1%}"
                f"  top_month={r['top_month']} ({r['top_month_pct']:.1%}){dec_flag}"
            )

    # Dec 2024 flagged combos
    dec_flagged = results[results["dec2024_flag"]]
    if not dec_flagged.empty:
        lines += ["", "=" * 70, "DECEMBER 2024 CONCENTRATION FLAGS", "=" * 70, ""]
        lines.append("  The following combos have >50% of total R in December 2024:")
        lines.append("  (Same failure pattern that halted KeltnerLong and LinReg at T4)")
        lines.append("")
        for _, r in dec_flagged.sort_values("avg_r", ascending=False).iterrows():
            lines.append(
                f"  [N={int(r['n'])} / ch_mult={r['atr_ch_mult']:.1f} / sm={r['stop_mult']:.1f}]"
                f"  dec2024_pct={r['top_month_pct']:.1%}  avg_r={r['avg_r']:+.4f}"
                f"  total_r={r['total_r']:+.2f}"
            )

    # T1 early-warning diagnostics for §4.2-passing combos
    passing = results[results["pass_cost_floor"]]
    lines += ["", "=" * 70, "EARLY-WARNING DIAGNOSTICS (§4.2-passing combos only)", "=" * 70, ""]
    if passing.empty:
        lines.append("  No combos pass §4.2 — diagnostics not applicable.")
    else:
        lines.append(
            f"  {'Config':30s}  {'fh_avg_r':>8s}  {'sh_avg_r':>8s}  "
            f"{'SH_NEG':6s}  {'top_month':>8s}  {'top_pct':>7s}  {'DEC24':5s}"
        )
        lines.append("  " + "-" * 78)
        for _, r in passing.sort_values("avg_r", ascending=False).iterrows():
            cfg = f"N={int(r['n'])}/ch={r['atr_ch_mult']:.1f}/sm={r['stop_mult']:.1f}"
            sh_warn = "  *** DECAY ***" if r["second_half_negative"] else ""
            dc_warn = "  *** DEC24 ***" if r["dec2024_flag"] else ""
            lines.append(
                f"  {cfg:30s}  {r['first_half_avg_r']:>+8.4f}  {r['second_half_avg_r']:>+8.4f}"
                f"  {'YES' if r['second_half_negative'] else 'no':6s}"
                f"  {r['top_month']:>8s}  {r['top_month_pct']:>7.1%}"
                f"  {'YES' if r['dec2024_flag'] else 'no':5s}"
                f"{sh_warn}{dc_warn}"
            )

    lines += [
        "",
        "=" * 70,
        "INTERPRETATION GUIDE",
        "=" * 70,
        "",
        "  PROCEED TO T2 if:",
        "  - ≥1 combo: stability_zone_pass=True AND avg_r ≥ 0.15R",
        "  - Dec 2024 NOT the dominant month (top_month_pct < 50%)",
        "  - Second-half avg_r positive (no structural decay)",
        "  - Win rate in 30–45% range (§4.1)",
        "",
        "  HALT if:",
        "  - No combo passes §4.2 cost floor",
        "  - All passing combos flagged for Dec 2024 concentration",
        "  - Second-half avg_r negative (same decay pattern as Keltner/LinReg T4)",
        "  - Win rate < 30% (§4.1) or > 55% (overfitting)",
        "",
        "  Do not proceed to T2 without human review.",
    ]

    rpt = OUT_DIR / "phase_t1_atrchannel_summary.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T1 -- ATRChannel Concept Discovery")
    print("=" * 70)

    symbols = get_symbols()
    print(f"Universe: {len(symbols)} symbols  (1D cache)")
    print(f"Grid: N={N_VALUES} x ch_mult={ATR_CHANNEL_MULTS} x sm={STOP_MULTS}")
    print(f"Total combos per symbol: {len(N_VALUES)*len(ATR_CHANNEL_MULTS)*len(STOP_MULTS)}")
    print(f"Output: {OUT_DIR}")
    print()

    all_trades: List[dict] = []
    loaded = skipped = 0

    for sym in symbols:
        raw = load_1d(sym)
        if raw is None or len(raw) < EMA_N + 50:
            skipped += 1
            continue

        close  = raw["close"].to_numpy(dtype=float)
        high   = raw["high"].to_numpy(dtype=float)
        low    = raw["low"].to_numpy(dtype=float)
        times  = raw["time"].to_numpy()
        ema200 = compute_ema(close, EMA_N)

        for n_val in N_VALUES:
            if len(close) < n_val + 50:
                continue
            sma_n = compute_sma(close, n_val)
            atr_n = compute_atr(high, low, close, n_val)

            for atr_mult in ATR_CHANNEL_MULTS:
                for sm in STOP_MULTS:
                    trades = run_backtest(
                        close, low, sma_n, atr_n, ema200,
                        atr_mult, sm, times, sym, n_val,
                    )
                    all_trades.extend(trades)

        loaded += 1

    print(f"Loaded={loaded}  Skipped={skipped}  Total_trades={len(all_trades)}")

    if not all_trades:
        print("[ERROR] No trades generated.")
        return 1

    results = stability_zones(aggregate(all_trades))
    results.to_csv(OUT_DIR / "phase_t1_atrchannel_stability_ranking.csv", index=False)

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(OUT_DIR / "phase_t1_atrchannel_trades.csv", index=False)

    write_report(results, loaded)

    # Console summary
    n_pass = int(results["pass_cost_floor"].sum())
    n_stab = int(results["stability_zone_pass"].sum())
    n_both = int((results["pass_cost_floor"] & results["stability_zone_pass"]).sum())
    n_dec  = int(results["dec2024_flag"].sum())

    print(f"\nTotal combos  : {len(results)}")
    print(f"Pass §4.2     : {n_pass}")
    print(f"Zone stable   : {n_stab}")
    print(f"Pass both     : {n_both}")
    print(f"Dec-2024 flag : {n_dec}")

    print("\n" + "=" * 70)
    print("TOP 10 by avg_r (pass_cost_floor=True)")
    print("=" * 70)
    top = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False).head(10)
    if top.empty:
        print("  None pass §4.2 cost floor.")
    else:
        for _, r in top.iterrows():
            dec = " ***DEC24***" if r["dec2024_flag"] else ""
            print(
                f"  [N={int(r['n']):>3}/ch={r['atr_ch_mult']:.1f}/sm={r['stop_mult']:.1f}]"
                f"  avg_r={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}"
                f"  win={r['win_rate']:.1%}  trades={int(r['trades'])}"
                f"  {'ZONE' if r['stability_zone_pass'] else '    '}{dec}"
            )

    print()
    print("=" * 70)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
