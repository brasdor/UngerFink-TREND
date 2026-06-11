#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- BollingerBandMR Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology

Entry : close < SMA(bb_n) - std_mult * StdDev(close, bb_n)  [positional]
Exit  : fixed time exit after hold_bars
Safety stop: ATR x atr_mult below entry close

Usage:
    python phase_t1_bollingerband_mr_concept_discovery.py --timeframe 1d
    python phase_t1_bollingerband_mr_concept_discovery.py  # all TFs
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

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_bollingerbandmr_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "AAVE_USDT","ADA_USDT","ALT_USDT","APT_USDT","ARB_USDT","ARKM_USDT",
    "ASTER_USDT","ATOM_USDT","AVAX_USDT","BCH_USDT","BNB_USDT","BTC_USDT",
    "CHZ_USDT","DASH_USDT","DOGE_USDT","DOT_USDT","EIGEN_USDT","ENA_USDT",
    "ETH_USDT","FET_USDT","FIL_USDT","GRT_USDT","HBAR_USDT","ICP_USDT",
    "INJ_USDT","JTO_USDT","LINK_USDT","LPT_USDT","LTC_USDT","MORPHO_USDT",
    "NEAR_USDT","NIL_USDT","ONDO_USDT","ORDI_USDT","PENDLE_USDT","PENGU_USDT",
    "PEPE_USDT","RENDER_USDT","SAGA_USDT","SEI_USDT","SOL_USDT","SPK_USDT",
    "SUI_USDT","TAO_USDT","TIA_USDT","TON_USDT","TRX_USDT","UNI_USDT",
    "WLD_USDT","XRP_USDT","ZEC_USDT","ZEN_USDT",
]

MAX_BARS = {"1d": 2000, "2h": 17520, "4h": 6000}
MIN_BARS = 200

BB_N        = [10, 15, 20, 25, 30]
STD_MULT    = [1.5, 2.0, 2.5, 3.0]
HOLD_BARS   = [10, 15, 20, 25]
ATR_MULTS   = [2.0, 3.0, 4.0]
FILTERS     = ["none", "ema200_price_above"]

MR_GATES = {
    "min_trades":    100,
    "win_rate_min":  0.50,
    "win_rate_max":  0.70,
    "avg_r_min":     0.10,
    "stability_min": 0.67,
}


def p_flush(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)


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
        for col in ("open","high","low","close","volume"):
            if col not in df.columns:
                return None
        limit = MAX_BARS.get(tf, 2000)
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all BB variants, ATR14, EMA200 once per symbol."""
    out = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]

    # ATR(14)
    tr  = pd.concat([high - low, (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()

    # EMA200
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    # Bollinger bands for each (bb_n, std_mult)
    for n in BB_N:
        sma = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=0)
        out[f"sma{n}"] = sma
        out[f"std{n}"] = std
        for m in STD_MULT:
            m_str = str(m).replace(".", "p")
            out[f"lower_{n}_{m_str}"] = sma - m * std

    return out


def backtest_symbol_all_combos(
    df: pd.DataFrame,
    filter_mode: str,
    param_combos: list[tuple],
) -> dict[tuple, list[dict]]:
    """Run all (bb_n, std_mult, hold_bars, atr_mult) combos for one symbol."""
    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values

    filt = close > ema200 if filter_mode == "ema200_price_above" else np.ones(len(df), bool)

    results: dict[tuple, list[dict]] = {}
    for (bb_n, std_mult, hold_bars, atr_mult) in param_combos:
        m_str  = str(std_mult).replace(".", "p")
        lb_col = f"lower_{bb_n}_{m_str}"
        if lb_col not in df.columns:
            results[(bb_n, std_mult, hold_bars, atr_mult)] = []
            continue

        lower = df[lb_col].values
        trades: list[dict] = []
        in_pos  = False
        e_price = 0.0
        stop    = 0.0
        e_bar   = 0

        for i in range(len(df)):
            if np.isnan(lower[i]) or np.isnan(atr[i]):
                continue
            if not in_pos:
                if close[i] < lower[i] and filt[i]:
                    in_pos  = True
                    e_price = close[i]
                    e_bar   = i
                    stop    = e_price - atr_mult * atr[i]
            else:
                bars_held = i - e_bar
                ep = None
                if low_v[i] <= stop:
                    ep = stop
                elif bars_held >= hold_bars:
                    ep = close[i]
                if ep is not None:
                    risk = e_price - stop
                    if risk > 1e-9:
                        trades.append({"r_multiple": (ep - e_price) / risk,
                                       "bars_held": bars_held,
                                       "reason": "atr_stop" if low_v[i] <= stop else "time_exit"})
                    in_pos = False

        results[(bb_n, std_mult, hold_bars, atr_mult)] = trades
    return results


def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0}
    rs = np.array([t["r_multiple"] for t in trades])
    wins = rs[rs>0]; losses = np.abs(rs[rs<0])
    pf   = wins.sum()/losses.sum() if losses.sum()>0 else (99.0 if wins.sum()>0 else 0.0)
    cum  = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd   = float(np.max(peak - cum)) if len(cum) else 0.0
    return {"n":len(rs),"win_rate":float(len(wins)/len(rs)),
            "avg_r":float(np.mean(rs)),"pf":float(pf),
            "max_dd_r":dd,"total_r":float(np.sum(rs))}


def stability_score(grid: pd.DataFrame, tf: str, filt: str,
                    bb_n: int, std_mult: float,
                    hold_bars: int, atr_mult: float) -> float:
    bb_ns  = sorted(BB_N)
    stds   = sorted(STD_MULT)
    try:
        bi = bb_ns.index(bb_n)
        si = stds.index(std_mult)
    except ValueError:
        return 0.0
    n_nbrs = [bb_ns[j]  for j in range(max(0,bi-1), min(len(bb_ns),bi+2))]
    s_nbrs = [stds[j]   for j in range(max(0,si-1), min(len(stds),si+2))]
    zone = grid[
        (grid["timeframe"]==tf) & (grid["filter_mode"]==filt) &
        (grid["bb_n"].isin(n_nbrs)) & (grid["std_mult"].isin(s_nbrs)) &
        (grid["hold_bars"]==hold_bars) & (grid["atr_mult"]==atr_mult)
    ]
    if zone.empty: return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


def run_grid_for_tf(tf: str) -> pd.DataFrame:
    param_combos = list(itertools.product(BB_N, STD_MULT, HOLD_BARS, ATR_MULTS))
    p(f"  Param combos per filter: {len(param_combos)} x {len(FILTERS)} filters = {len(param_combos)*len(FILTERS)}")

    loaded = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym, tf)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")
    if not loaded:
        return pd.DataFrame()

    combo_trades: dict[tuple, list[dict]] = {}
    for filt in FILTERS:
        for key in param_combos:
            combo_trades[(filt, *key)] = []

    for sym_idx, (sym, df) in enumerate(loaded, 1):
        if sym_idx % 10 == 0 or sym_idx == 1 or sym_idx == len(loaded):
            p(f"  Processing symbol {sym_idx}/{len(loaded)} {sym}...")
        for filt in FILTERS:
            for key, trades in backtest_symbol_all_combos(df, filt, param_combos).items():
                combo_trades[(filt, *key)].extend(trades)

    rows = []
    for filt in FILTERS:
        for (bb_n, std_mult, hold_bars, atr_mult) in param_combos:
            trades = combo_trades[(filt, bb_n, std_mult, hold_bars, atr_mult)]
            m = calc_metrics(trades)
            rows.append({
                "timeframe": tf, "filter_mode": filt,
                "bb_n": bb_n, "std_mult": std_mult,
                "hold_bars": hold_bars, "atr_mult": atr_mult,
                "num_trades": m["n"], "win_rate": round(m["win_rate"],4),
                "avg_r": round(m["avg_r"],4), "profit_factor": round(m["pf"],4),
                "max_dd_r": round(m["max_dd_r"],2), "total_r": round(m["total_r"],2),
            })
    return pd.DataFrame(rows)


def report(grid: pd.DataFrame, tf_label: str) -> bool:
    p(f"\n--- Results for {tf_label} ---")
    viable = grid[grid["num_trades"] >= MR_GATES["min_trades"]].copy()
    p(f"  Combos >= {MR_GATES['min_trades']} trades: {len(viable)}")
    if viable.empty:
        top = grid.nlargest(10, "num_trades")[
            ["timeframe","filter_mode","bb_n","std_mult","hold_bars","atr_mult","num_trades","avg_r"]]
        p("  FAIL: no combos meet min_trades. Top by trade count:")
        p(top.to_string(index=False))
        return False
    viable = viable[(viable["win_rate"]>=MR_GATES["win_rate_min"]) &
                    (viable["win_rate"]<=MR_GATES["win_rate_max"])]
    p(f"  After win_rate gate: {len(viable)}")
    viable = viable[viable["avg_r"] >= MR_GATES["avg_r_min"]]
    p(f"  After avg_r gate: {len(viable)}")
    if viable.empty:
        cands = grid[grid["num_trades"]>=MR_GATES["min_trades"]].nlargest(10,"avg_r")[
            ["timeframe","filter_mode","bb_n","std_mult","hold_bars","atr_mult","num_trades","win_rate","avg_r"]]
        p("  FAIL: no combos pass win_rate + avg_r. Best by avg_r:")
        p(cands.to_string(index=False)); return False

    p("  Computing stability zones...")
    viable["stability"] = viable.apply(
        lambda r: stability_score(grid, r["timeframe"], r["filter_mode"],
            int(r["bb_n"]), float(r["std_mult"]), int(r["hold_bars"]), float(r["atr_mult"])), axis=1)
    stable = viable[viable["stability"] >= MR_GATES["stability_min"]].copy()
    p(f"  After stability >= {MR_GATES['stability_min']}: {len(stable)}")
    if stable.empty:
        p("  NOTE: no stability-passing combos -- showing best by stability:")
        stable = viable.nlargest(30, "stability")
        gate_pass = False
    else:
        gate_pass = True

    stable = stable.sort_values(["stability","avg_r"], ascending=[False,False]).reset_index(drop=True)

    no_filt  = viable[viable["filter_mode"]=="none"]
    ema_filt = viable[viable["filter_mode"]=="ema200_price_above"]
    p(f"\n  EMA200 filter impact:")
    p(f"    no filter    : {no_filt['avg_r'].mean():.4f}R  ({len(no_filt)} combos)")
    p(f"    ema200 above : {ema_filt['avg_r'].mean():.4f}R  ({len(ema_filt)} combos)")

    top = stable.head(10)
    p(f"\n  Top 10 stable combos:")
    p(top[["timeframe","filter_mode","bb_n","std_mult","hold_bars","atr_mult",
           "num_trades","win_rate","avg_r","stability"]].to_string(index=False))

    # Heatmap: avg_r by (bb_n, std_mult)
    best_tf   = stable.iloc[0]["timeframe"]
    best_filt = stable.iloc[0]["filter_mode"]
    pivot = grid[(grid["timeframe"]==best_tf)&(grid["filter_mode"]==best_filt)]\
        .groupby(["bb_n","std_mult"])["avg_r"].mean().unstack("std_mult")
    p(f"\n  avg_r heatmap (bb_n vs std_mult) | TF={best_tf} filter={best_filt}:")
    p(pivot.round(3).to_string())

    stable.to_csv(OUT_DIR / "phase_t1_stability_ranking.csv", index=False)
    with open(OUT_DIR / "phase_t1_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"BollingerBandMR T1 -- {tf_label}\n")
        f.write(f"Grid: {len(grid)}  Viable: {len(viable)}  Stable: {len(stable)}\n\n")
        f.write(f"EMA200 no_filter avg_r: {no_filt['avg_r'].mean():.4f}R ({len(no_filt)} combos)\n")
        f.write(f"EMA200 ema_filter avg_r: {ema_filt['avg_r'].mean():.4f}R ({len(ema_filt)} combos)\n\n")
        f.write("Top 20 stable combos:\n")
        f.write(stable.head(20)[["timeframe","filter_mode","bb_n","std_mult","hold_bars",
            "atr_mult","num_trades","win_rate","avg_r","profit_factor","max_dd_r",
            "stability","total_r"]].to_string(index=False))
        f.write("\n\navg_r heatmap:\n"); f.write(pivot.round(3).to_string())
        f.write(f"\n\nT1 GATE: {'PASS' if gate_pass else 'MARGINAL'}\n")
    return gate_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default=None, help="1d, 2h, 4h, or omit for all")
    args = parser.parse_args()

    tfs = [args.timeframe.lower()] if args.timeframe else list(MAX_BARS.keys())

    p("=" * 65)
    p("  Phase T1 -- BollingerBandMR Concept Discovery")
    p(f"  Timeframes: {tfs}  |  Symbols: {len(SYMBOLS)}")
    p(f"  BB_N: {BB_N}  |  STD_MULT: {STD_MULT}")
    p(f"  HOLD_BARS: {HOLD_BARS}  |  ATR_MULTS: {ATR_MULTS}")
    p("=" * 65)

    available = sum(1 for sym in SYMBOLS for tf in tfs
                    if (RAW_DIR/f"{sym}_{tf}.csv").exists())
    p(f"Data files found: {available}")
    if not available:
        p(f"ERROR: no data in {RAW_DIR}"); sys.exit(1)

    all_grids = []
    for tf in tfs:
        p(f"\n{'='*50}\n  Running timeframe: {tf}\n{'='*50}")
        g = run_grid_for_tf(tf)
        if not g.empty:
            g.to_csv(OUT_DIR / f"phase_t1_grid_{tf}.csv", index=False)
            p(f"  Grid saved: {len(g)} rows")
            all_grids.append(g)

    if not all_grids:
        p("ERROR: no results."); sys.exit(1)

    combined = pd.concat(all_grids, ignore_index=True)
    gate_pass = report(combined, "+".join(tfs))

    p(f"\n[OK] phase_t1_stability_ranking.csv")
    p(f"[OK] phase_t1_summary.txt")
    p(f"\nT1 GATE: {'PASS' if gate_pass else 'FAIL'}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
