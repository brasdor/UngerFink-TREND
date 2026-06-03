#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- RSIMeanReversionShort Concept Discovery (REFINED)
UngerFink Pipeline / Andrea Unger Methodology  (Section 19A)

Refined grid based on T2 full-history findings:
  - ob=75 failed win-rate gate (40.8%) -- raise threshold
  - rsi_n=[2,3,5] weak in original heatmap -- drop
  - ema200_price_below mandatory -- single filter only
  - Use full 4H history 2021-2026 from ohlcv_cache

Entry : RSI(N) > overbought  [positional SHORT, price < EMA200]
Exit  : Fixed time exit after hold_bars
Stop  : ATR x atr_mult ABOVE entry (short stop-loss)

Grid:
  rsi_n        : [7, 10, 14]
  overbought   : [80, 85, 90]
  hold_bars    : [10, 15, 20, 25, 30]
  atr_mult     : [2.0, 3.0]
  filter       : ema200_price_below only (mandatory)

Critical outputs:
  - Year-by-year for top 5 combos (2021-2026)
  - 2025 concentration flag (>60% of total R in 2025)
  - HALT if no stable combos at ob >= 85

§4.2 cost floor: 0.15R (Futures)
Output: data/research_rsimrshort_t1/
"""

from __future__ import annotations
import argparse, itertools, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "research_rsimrshort_t1" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_rsimrshort_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "AAVE_USDT","ADA_USDT","APT_USDT","ARB_USDT","ASTER_USDT",
    "ATOM_USDT","AVAX_USDT","BCH_USDT","BNB_USDT","BTC_USDT",
    "CHZ_USDT","DASH_USDT","DOGE_USDT","DOT_USDT","EIGEN_USDT",
    "ENA_USDT","ETH_USDT","FET_USDT","FIL_USDT","GRT_USDT",
    "HBAR_USDT","ICP_USDT","INJ_USDT","JTO_USDT","LINK_USDT",
    "LPT_USDT","LTC_USDT","NEAR_USDT","ONDO_USDT","PENGU_USDT",
    "PEPE_USDT","RENDER_USDT","SEI_USDT","SOL_USDT","SUI_USDT",
    "TIA_USDT","TRX_USDT","UNI_USDT","WLD_USDT","XRP_USDT",
    "ZEC_USDT","ZEN_USDT",
]  # 42 symbols confirmed in ohlcv_cache

RSI_N      = [7, 10, 14]
OVERBOUGHT = [80, 85, 90]
HOLD_BARS  = [10, 15, 20, 25, 30]
ATR_MULTS  = [2.0, 3.0]
FILTERS    = ["ema200_price_below"]  # mandatory only

GATES = {
    "min_trades":    80,   # slightly lower for short system (fewer signals)
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.15,
    "stability_min": 0.67,
    "conc_2025_max": 0.60,  # flag if 2025 > 60% of total R
}

MIN_BARS = 500  # need 2022 coverage (~500 bars from Jan 2021 to Dec 2022)


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{sym}_4h.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open","high","low","close","volume"):
            if col not in df.columns:
                return None
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]; high = out["high"]; low = out["low"]
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    for n in RSI_N:
        d  = close.diff()
        ag = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        out[f"rsi{n}"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    return out


def backtest_symbol_all_combos(df: pd.DataFrame,
                                param_combos: list[tuple]) -> dict[tuple, list[dict]]:
    """SHORT: enter when RSI > overbought AND close < EMA200."""
    close  = df["close"].values
    high_v = df["high"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
    # ema200_price_below filter: close < ema200
    filt   = close < ema200

    results: dict[tuple, list[dict]] = {}
    for (rsi_n, overbought, hold_bars, atr_mult) in param_combos:
        rsi    = df[f"rsi{rsi_n}"].values
        trades = []
        in_pos = False; e_price = stop = 0.0; e_bar = 0; e_ts = None

        for i in range(len(df)):
            if np.isnan(rsi[i]) or np.isnan(atr[i]):
                continue
            if not in_pos:
                if rsi[i] > overbought and filt[i]:
                    in_pos  = True; e_price = close[i]; e_bar = i; e_ts = ts[i]
                    stop    = e_price + atr_mult * atr[i]  # stop ABOVE for short
            else:
                bars_held = i - e_bar; ep = reason = None
                if high_v[i] >= stop:
                    ep, reason = stop, "atr_stop"
                elif bars_held >= hold_bars:
                    ep, reason = close[i], "time_exit"
                if reason:
                    risk = stop - e_price
                    if risk > 1e-9:
                        rm = (e_price - ep) / risk  # positive when price falls
                        entry_dt = pd.Timestamp(e_ts)
                        trades.append({
                            "r_multiple": float(rm),
                            "bars_held":  bars_held,
                            "reason":     reason,
                            "year":       entry_dt.year,
                        })
                    in_pos = False

        results[(rsi_n, overbought, hold_bars, atr_mult)] = trades
    return results


def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0}
    rs = np.array([t["r_multiple"] for t in trades])
    w = rs[rs>0]; l = np.abs(rs[rs<0])
    pf = w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd  = float(np.max(peak-cum)) if len(cum) else 0.0
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":dd,"total_r":float(np.sum(rs))}


def year_breakdown_trades(trades: list[dict]) -> dict[int, dict]:
    by_year: dict[int, list[float]] = {}
    for t in trades:
        yr = t["year"]
        by_year.setdefault(yr, []).append(t["r_multiple"])
    return {yr: calc_metrics([{"r_multiple":r,"bars_held":0,"reason":"","year":yr}
                               for r in rs]) for yr, rs in by_year.items()}


def concentration_2025(trades: list[dict]) -> float:
    """Fraction of total positive R that comes from 2025."""
    total_r = sum(t["r_multiple"] for t in trades)
    if total_r <= 0:
        return 0.0
    r2025 = sum(t["r_multiple"] for t in trades if t["year"] == 2025)
    return r2025 / total_r


def stability_score(grid: pd.DataFrame, rsi_n: int, overbought: int,
                    hold_bars: int, atr_mult: float) -> float:
    rsi_ns = sorted(RSI_N); ovbs = sorted(OVERBOUGHT)
    try:
        ri = rsi_ns.index(rsi_n); oi = ovbs.index(overbought)
    except ValueError:
        return 0.0
    n_nbrs = [rsi_ns[j] for j in range(max(0,ri-1), min(len(rsi_ns),ri+2))]
    o_nbrs = [ovbs[j]   for j in range(max(0,oi-1), min(len(ovbs),oi+2))]
    zone = grid[(grid["rsi_n"].isin(n_nbrs)) & (grid["overbought"].isin(o_nbrs)) &
                (grid["hold_bars"]==hold_bars) & (grid["atr_mult"]==atr_mult)]
    if zone.empty: return 0.0
    return round((zone["avg_r"]>0).sum()/len(zone), 4)


def main() -> None:
    p("=" * 70)
    p("  Phase T1 -- RSIMeanReversionShort (REFINED GRID)")
    p("  Direction: SHORT  |  4H full history 2021-2026")
    p("  Grid: rsi_n=[7,10,14]  ob=[80,85,90]  hold=[10-30]  atr=[2,3]")
    p("  Filter: ema200_price_below ONLY (mandatory)")
    p("  Cost floor: 0.15R (Futures)")
    p("=" * 70)

    # Load data
    loaded = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"  Symbols loaded: {len(loaded)}  (min {MIN_BARS} bars required for 2022 coverage)")
    if not loaded:
        p(f"  ERROR: no data in {CACHE_DIR}"); sys.exit(1)

    # Verify 2022 coverage
    sample_sym, sample_df = loaded[0]
    first_date = sample_df["timestamp"].iloc[0].date() if "timestamp" in sample_df.columns else "unknown"
    p(f"  Data coverage: {first_date} onward  (2022 {'COVERED' if str(first_date) < '2022-01-01' else 'PARTIAL/MISSING'})")

    # Run grid
    param_combos = list(itertools.product(RSI_N, OVERBOUGHT, HOLD_BARS, ATR_MULTS))
    p(f"  Param combos: {len(param_combos)}  x  1 filter = {len(param_combos)} total")

    # Accumulate trades per combo
    combo_trades: dict[tuple, list[dict]] = {key: [] for key in param_combos}
    for sym_idx, (sym, df) in enumerate(loaded, 1):
        if sym_idx % 10 == 0 or sym_idx == 1 or sym_idx == len(loaded):
            p(f"  Processing {sym_idx}/{len(loaded)} {sym}...")
        for key, trades in backtest_symbol_all_combos(df, param_combos).items():
            combo_trades[key].extend(trades)

    # Build grid DataFrame
    rows = []
    for (rsi_n, overbought, hold_bars, atr_mult) in param_combos:
        trades = combo_trades[(rsi_n, overbought, hold_bars, atr_mult)]
        m = calc_metrics(trades)
        c2025 = concentration_2025(trades)
        rows.append({
            "rsi_n":rsi_n,"overbought":overbought,"hold_bars":hold_bars,"atr_mult":atr_mult,
            "num_trades":m["n"],"win_rate":round(m["win_rate"],4),
            "avg_r":round(m["avg_r"],4),"profit_factor":round(m["pf"],4),
            "max_dd_r":round(m["max_dd_r"],2),"total_r":round(m["total_r"],2),
            "conc_2025":round(c2025,3),
        })
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR/"phase_t1_grid_4h_refined.csv", index=False)
    p(f"  Grid saved: {len(grid)} rows")

    # Gate filters
    p(f"\n--- Results ---")
    viable = grid[grid["num_trades"] >= GATES["min_trades"]].copy()
    p(f"  Combos >= {GATES['min_trades']} trades: {len(viable)}")
    if viable.empty:
        p("  FAIL: no combos meet min_trades."); sys.exit(1)

    viable = viable[(viable["win_rate"] >= GATES["win_rate_min"]) &
                    (viable["win_rate"] <= GATES["win_rate_max"])]
    p(f"  After win_rate 50-70%: {len(viable)}")
    if viable.empty:
        top = grid[grid["num_trades"]>=GATES["min_trades"]].nlargest(10,"avg_r")[
            ["rsi_n","overbought","hold_bars","atr_mult","num_trades","win_rate","avg_r"]]
        p("  FAIL: no combos pass win_rate gate. Best by avg_r:")
        p(top.to_string(index=False))
        p("\n  HALT: ob=[80,85,90] + rsi_n=[7,10,14] still cannot meet 50% win rate floor.")
        p("  RSI MR Short on 4H crypto may be structurally incompatible with §4.1 gate.")
        sys.exit(1)

    viable = viable[viable["avg_r"] >= GATES["avg_r_min"]]
    p(f"  After avg_r >= {GATES['avg_r_min']}R: {len(viable)}")
    if viable.empty:
        p("  FAIL: no combos pass avg_r gate."); sys.exit(1)

    # Stability
    p("  Computing stability zones...")
    viable["stability"] = viable.apply(
        lambda r: stability_score(grid, int(r["rsi_n"]), int(r["overbought"]),
                                  int(r["hold_bars"]), float(r["atr_mult"])), axis=1)
    stable = viable[viable["stability"] >= GATES["stability_min"]].copy()
    p(f"  After stability >= {GATES['stability_min']}: {len(stable)}")

    # Check for ob=85/90 vs ob=80 distribution
    for ob in [80, 85, 90]:
        n = len(viable[viable["overbought"]==ob])
        n_stable = len(stable[stable["overbought"]==ob]) if not stable.empty else 0
        p(f"    ob={ob}: {n} viable, {n_stable} stable")

    if stable.empty:
        p("  HALT: no stable combos found at raised thresholds.")
        p("  RSI MR Short: structurally weak on 4H crypto -- cannot proceed to T2.")
        viable.nlargest(10,"stability").to_csv(OUT_DIR/"phase_t1_stability_ranking.csv",index=False)
        sys.exit(1)
    gate_pass = True

    stable = stable.sort_values(["stability","avg_r"],ascending=[False,False]).reset_index(drop=True)

    # 2025 concentration check
    flagged_2025 = stable[stable["conc_2025"] > GATES["conc_2025_max"]]
    clean_stable = stable[stable["conc_2025"] <= GATES["conc_2025_max"]]
    p(f"  2025 concentration > {GATES['conc_2025_max']*100:.0f}%: {len(flagged_2025)} flagged, {len(clean_stable)} clean")

    # Print top 10
    p(f"\n  Top 10 stable combos (SHORT / 4H / 2021-2026):")
    cols = ["rsi_n","overbought","hold_bars","atr_mult","num_trades","win_rate","avg_r","stability","conc_2025"]
    p(stable.head(10)[cols].to_string(index=False))

    # Heatmap
    pivot = grid.groupby(["rsi_n","overbought"])["avg_r"].mean().unstack("overbought")
    p(f"\n  avg_r heatmap (rsi_n vs overbought) | filter=ema200_price_below (all hold/atr):")
    p(pivot.round(3).to_string())

    # Year-by-year for top 5 combos
    p(f"\n  Year-by-year for top 5 stable combos:")
    for rank, (_, row) in enumerate(stable.head(5).iterrows(), 1):
        key = (int(row["rsi_n"]), int(row["overbought"]), int(row["hold_bars"]), float(row["atr_mult"]))
        trades = combo_trades[key]
        yb = year_breakdown_trades(trades)
        conc_flag = " <<< 2025 CONCENTRATED" if row["conc_2025"] > GATES["conc_2025_max"] else ""
        p(f"\n  #{rank}: rsi{int(row['rsi_n'])}/ob{int(row['overbought'])}/h{int(row['hold_bars'])}/atr{row['atr_mult']}  "
          f"avg_r={row['avg_r']:+.4f}R  stability={row['stability']:.2f}  2025_conc={row['conc_2025']*100:.0f}%{conc_flag}")
        p(f"  {'Year':>5}  {'N':>4}  {'WR%':>6}  {'AvgR':>8}  {'TotR':>8}  Note")
        for yr in sorted(yb.keys()):
            ym = yb[yr]
            note = "BEAR" if yr==2022 else ("BULL" if yr in (2021,2024) else "")
            neg  = " WEAK" if ym["avg_r"]<0 else ""
            p(f"  {yr:>5}  {ym['n']:>4}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  {note}{neg}")

    # Save outputs
    stable.to_csv(OUT_DIR/"phase_t1_stability_ranking.csv", index=False)
    with open(OUT_DIR/"phase_t1_summary.txt","w",encoding="utf-8") as f:
        f.write("RSIMeanReversionShort T1 REFINED -- 4H full history 2021-2026\n")
        f.write(f"Grid: rsi_n={RSI_N}  ob={OVERBOUGHT}  hold={HOLD_BARS}  atr={ATR_MULTS}\n")
        f.write(f"Combos: {len(grid)}  Viable: {len(viable)}  Stable: {len(stable)}\n")
        f.write(f"2025-concentrated (>{GATES['conc_2025_max']*100:.0f}%): {len(flagged_2025)}\n")
        f.write(f"2025-clean stable: {len(clean_stable)}\n\n")
        f.write("Top 20 stable combos:\n")
        f.write(stable.head(20)[cols].to_string(index=False))
        f.write("\n\navg_r heatmap:\n"); f.write(pivot.round(3).to_string())
        f.write(f"\n\nT1 GATE: {'PASS' if gate_pass else 'FAIL'}\n")

    p(f"\n[OK] phase_t1_grid_4h_refined.csv  ({len(grid)} rows)")
    p(f"[OK] phase_t1_stability_ranking.csv ({len(stable)} stable combos)")
    p(f"[OK] phase_t1_summary.txt")
    p(f"\nT1 GATE: PASS  ({len(stable)} stable combos, {len(clean_stable)} without 2025 concentration)")
    sys.exit(0)


if __name__ == "__main__":
    main()
