#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T3MR -- MeanReversionRSI Exit Engineering
UngerFink Pipeline / Andrea Unger Methodology

Tests five exit variants against the canonical T2 config.
Must beat T2 baseline: avg_r=+0.223R, PF=2.16, win_rate 50-70%.

Variants:
  A  RSI exit only     -- exit when RSI >= exit_rsi (no stop, no time)
  B  ATR stop only     -- exit when price <= entry - atr_mult*ATR (no RSI, no time)
  C  Combined          -- RSI exit OR ATR stop (T2 canonical, baseline)
  D  Profit target     -- exit at +1.0R fixed OR ATR stop (no RSI exit)
  E  Time exit only    -- exit after time_exit bars (no RSI, no ATR)

Usage:
    python phase_t3mr_meanreversionrsi_exit_engineering.py \
        --timeframe 1d --rsi-n 14 --oversold 25 \
        --exit-rsi 50 --atr-mult 3.0 --time-exit 20 \
        --profit-target 1.0

Output: data/research_meanreversionrsi_t3mr_1d/
    phase_t3mr_variant_summary.csv
    phase_t3mr_trades_{variant}.csv
    phase_t3mr_report.txt
"""

from __future__ import annotations

import argparse
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


# =============================================================================
# PATHS / SYMBOLS
# =============================================================================

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"

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

MAX_BARS = {"1d": 2000, "2h": 17520, "4h": 6000, "6h": 4000, "8h": 3000}
MIN_BARS = 200

# T2 baseline to beat
T2_BASELINE = {"avg_r": 0.2233, "pf": 2.16, "win_rate": 0.633}

MR_GATES = {"win_rate_min": 0.50, "win_rate_max": 0.70, "avg_r_min": 0.10}


# =============================================================================
# DATA / INDICATORS
# =============================================================================

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
        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy() if len(df) >= MIN_BARS else None
    except Exception:
        return None


def add_indicators(df: pd.DataFrame, rsi_n: int) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    delta = close.diff()
    avg_g = delta.clip(lower=0).ewm(alpha=1/rsi_n, min_periods=rsi_n, adjust=False).mean()
    avg_l = (-delta.clip(upper=0)).ewm(alpha=1/rsi_n, min_periods=rsi_n, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


# =============================================================================
# BACKTEST PER VARIANT
# =============================================================================

def backtest_variant(df: pd.DataFrame, symbol: str,
                     rsi_n: int, oversold: int, exit_rsi: int,
                     atr_mult: float, time_exit: int,
                     profit_target_r: float, variant: str) -> list[dict]:
    """
    variant: 'A' | 'B' | 'C' | 'D' | 'E'
    A = RSI exit only
    B = ATR stop only
    C = RSI OR ATR stop (combined, T2 canonical)
    D = profit target (+profit_target_r * risk) OR ATR stop
    E = time exit only
    """
    df = df.copy()
    df = add_indicators(df, rsi_n)
    df = df.dropna(subset=["rsi", "atr"]).reset_index(drop=True)
    if len(df) < 50:
        return []

    close = df["close"].values
    low_v = df["low"].values
    rsi   = df["rsi"].values
    atr   = df["atr"].values
    ts    = df["timestamp"].values

    trades: list[dict] = []
    in_pos  = False
    e_price = 0.0
    stop    = 0.0
    tgt     = 0.0
    e_bar   = 0
    e_ts    = None

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]):
            continue

        if not in_pos:
            if rsi[i] < oversold:
                in_pos  = True
                e_price = close[i]
                e_bar   = i
                e_ts    = ts[i]
                risk    = atr_mult * atr[i]
                stop    = e_price - risk
                tgt     = e_price + profit_target_r * risk
        else:
            bars_held  = i - e_bar
            exit_price = None
            reason     = None

            if variant == "A":
                # RSI exit only — no stop, no time
                if rsi[i] >= exit_rsi:
                    exit_price = close[i]
                    reason     = "rsi_exit"

            elif variant == "B":
                # ATR stop only — no RSI exit, no time
                if low_v[i] <= stop:
                    exit_price = stop
                    reason     = "atr_stop"

            elif variant == "C":
                # Combined: RSI OR ATR stop (T2 canonical)
                if low_v[i] <= stop:
                    exit_price = stop
                    reason     = "atr_stop"
                elif rsi[i] >= exit_rsi:
                    exit_price = close[i]
                    reason     = "rsi_exit"
                elif bars_held >= time_exit:
                    exit_price = close[i]
                    reason     = "time_exit"

            elif variant == "D":
                # Profit target OR ATR stop
                if low_v[i] <= stop:
                    exit_price = stop
                    reason     = "atr_stop"
                elif close[i] >= tgt:
                    exit_price = tgt
                    reason     = "profit_target"
                elif bars_held >= time_exit:
                    exit_price = close[i]
                    reason     = "time_exit"

            elif variant == "E":
                # Time exit only
                if bars_held >= time_exit:
                    exit_price = close[i]
                    reason     = "time_exit"

            if reason is not None:
                risk_actual = e_price - stop
                if risk_actual > 1e-9:
                    r_mult = (exit_price - e_price) / risk_actual
                    trades.append({
                        "symbol":      symbol,
                        "entry_time":  pd.Timestamp(e_ts),
                        "exit_time":   pd.Timestamp(ts[i]),
                        "net_r":       round(float(r_mult), 4),
                        "win":         int(r_mult > 0),
                        "bars_held":   bars_held,
                        "exit_reason": reason,
                        "year":        pd.Timestamp(e_ts).year,
                    })
                in_pos = False

    return trades


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0,
                "max_dd_r": 0.0, "total_r": 0.0, "avg_bars": 0.0}
    rs   = np.array([t["net_r"] for t in trades])
    wins = rs[rs > 0]
    loss = np.abs(rs[rs < 0])
    pf   = wins.sum() / loss.sum() if loss.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    cum  = np.cumsum(rs)
    dd   = float(np.max(np.maximum.accumulate(cum) - cum))
    return {
        "n":        len(rs),
        "win_rate": float(len(wins) / len(rs)),
        "avg_r":    float(np.mean(rs)),
        "pf":       float(pf),
        "max_dd_r": dd,
        "total_r":  float(np.sum(rs)),
        "avg_bars": float(np.mean([t["bars_held"] for t in trades])),
    }


def year_breakdown(trades: list[dict]) -> dict[int, dict]:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    result = {}
    for yr, grp in df.groupby("year"):
        result[int(yr)] = calc_metrics(grp.to_dict("records"))
    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase T3MR Exit Engineering")
    parser.add_argument("--timeframe",       required=True)
    parser.add_argument("--rsi-n",           type=int,   required=True)
    parser.add_argument("--oversold",        type=int,   required=True)
    parser.add_argument("--exit-rsi",        type=int,   required=True)
    parser.add_argument("--atr-mult",        type=float, required=True)
    parser.add_argument("--time-exit",       type=int,   required=True)
    parser.add_argument("--profit-target",   type=float, default=1.0,
                        help="R multiple for variant D profit target (default 1.0)")
    args = parser.parse_args()

    tf           = args.timeframe.lower()
    rsi_n        = args.rsi_n
    oversold     = args.oversold
    exit_rsi     = args.exit_rsi
    atr_mult     = args.atr_mult
    time_exit    = args.time_exit
    profit_tgt   = args.profit_target

    out_dir = ROOT / f"data/research_meanreversionrsi_t3mr_{tf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p("=" * 70)
    p("  Phase T3MR -- MeanReversionRSI Exit Engineering")
    p(f"  Timeframe    : {tf}")
    p(f"  Entry config : rsi{rsi_n}/os{oversold}")
    p(f"  ATR mult     : {atr_mult}  |  Profit target: +{profit_tgt}R")
    p(f"  Exit RSI     : {exit_rsi}  |  Time exit    : {time_exit} bars")
    p(f"  Symbols      : {len(SYMBOLS)}")
    p(f"  T2 baseline  : avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}  WR={T2_BASELINE['win_rate']*100:.1f}%")
    p("=" * 70)

    # Load and prep all symbol data once
    loaded_data: list[tuple[str, pd.DataFrame]] = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym, tf)
        if df is not None:
            loaded_data.append((sym, df))
    p(f"  Symbols loaded: {len(loaded_data)}/{len(SYMBOLS)}")

    variants = {
        "A": "RSI exit only",
        "B": "ATR stop only",
        "C": "Combined (T2 canonical)",
        "D": f"Profit target +{profit_tgt}R OR ATR stop",
        "E": f"Time exit only ({time_exit} bars)",
    }

    all_results: list[dict] = []
    all_trades_by_variant: dict[str, list[dict]] = {}

    for var_key, var_desc in variants.items():
        p(f"\n  Running Variant {var_key}: {var_desc}...")
        var_trades: list[dict] = []

        for sym, df in loaded_data:
            trades = backtest_variant(
                df, sym, rsi_n, oversold, exit_rsi,
                atr_mult, time_exit, profit_tgt, var_key
            )
            var_trades.extend(trades)

        m = calc_metrics(var_trades)
        yb = year_breakdown(var_trades)
        y2022 = yb.get(2022, {})

        wr_pass   = MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"]
        avgr_pass = m["avg_r"] >= MR_GATES["avg_r_min"]
        beats_t2  = m["avg_r"] > T2_BASELINE["avg_r"] and m["pf"] > T2_BASELINE["pf"]
        bear_ok   = y2022.get("total_r", 0) >= -20.0 if y2022 else True

        gate = "PASS" if (wr_pass and avgr_pass) else "FAIL"
        flag = "BEATS T2" if beats_t2 else ("PASS" if (wr_pass and avgr_pass) else "FAIL")

        p(f"    Trades={m['n']:4d}  WR={m['win_rate']*100:5.1f}%  avg_r={m['avg_r']:+.4f}R  "
          f"PF={m['pf']:.2f}  DD={m['max_dd_r']:.2f}R  total={m['total_r']:+.1f}R  "
          f"bars={m['avg_bars']:.1f}  => {gate}")
        if y2022:
            p(f"    2022: n={y2022['n']} wr={y2022['win_rate']*100:.1f}% total={y2022['total_r']:+.2f}R  "
              f"{'OK' if bear_ok else 'FAIL-BEAR'}")

        all_results.append({
            "variant":     var_key,
            "description": var_desc,
            "n":           m["n"],
            "win_rate":    round(m["win_rate"], 4),
            "avg_r":       round(m["avg_r"], 4),
            "pf":          round(m["pf"], 2),
            "max_dd_r":    round(m["max_dd_r"], 2),
            "total_r":     round(m["total_r"], 2),
            "avg_bars":    round(m["avg_bars"], 1),
            "y2022_total": round(y2022.get("total_r", float("nan")), 2) if y2022 else None,
            "y2022_wr":    round(y2022.get("win_rate", float("nan")), 3) if y2022 else None,
            "wr_gate":     "PASS" if wr_pass else "FAIL",
            "avgr_gate":   "PASS" if avgr_pass else "FAIL",
            "bear_gate":   "PASS" if bear_ok else "FAIL",
            "overall":     flag,
        })
        all_trades_by_variant[var_key] = var_trades

    # Save variant summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_dir / "phase_t3mr_variant_summary.csv", index=False)

    # Save per-variant trade logs
    for var_key, trades in all_trades_by_variant.items():
        if trades:
            pd.DataFrame(trades).to_csv(
                out_dir / f"phase_t3mr_trades_{var_key}.csv", index=False)

    # Print comparison table
    p("\n" + "=" * 70)
    p("  T3MR EXIT VARIANT COMPARISON")
    p(f"  T2 Baseline: avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}  WR={T2_BASELINE['win_rate']*100:.1f}%")
    p("=" * 70)
    p(f"  {'Var':<3} {'Description':<36} {'N':>5} {'WR%':>6} {'AvgR':>7} "
      f"{'PF':>5} {'MaxDD':>6} {'TotR':>7} {'Bars':>5} {'2022':>7} {'Gate'}")
    p("  " + "-" * 96)
    for r in all_results:
        y22 = f"{r['y2022_total']:+.1f}R" if r['y2022_total'] is not None else "  N/A "
        beat = " ***" if r["overall"] == "BEATS T2" else ""
        p(f"  {r['variant']:<3} {r['description']:<36} {r['n']:>5} "
          f"{r['win_rate']*100:>5.1f}% {r['avg_r']:>+7.4f} "
          f"{r['pf']:>5.2f} {r['max_dd_r']:>6.2f}R {r['total_r']:>+7.1f}R "
          f"{r['avg_bars']:>5.1f} {y22:>7} {r['wr_gate']}/{r['avgr_gate']}{beat}")
    p("  " + "-" * 96)
    p(f"  {'BAS':<3} {'C = T2 canonical (reference)':<36}", end="")

    # Year-by-year for each variant
    p("\n\n  Year-by-year detail per variant:")
    for var_key, trades in all_trades_by_variant.items():
        if not trades:
            continue
        yb = year_breakdown(trades)
        p(f"\n  Variant {var_key}:")
        for yr in sorted(yb.keys()):
            ym = yb[yr]
            bear = " <<< BEAR" if yr == 2022 else ""
            p(f"    {yr}  n={ym['n']:4d}  wr={ym['win_rate']*100:5.1f}%  "
              f"avg_r={ym['avg_r']:+.3f}R  total={ym['total_r']:+7.2f}R  "
              f"dd={ym['max_dd_r']:.2f}R{bear}")

    # Best variant recommendation
    passing = [r for r in all_results if r["wr_gate"] == "PASS" and r["avgr_gate"] == "PASS"]
    if passing:
        best = max(passing, key=lambda x: x["avg_r"])
        p(f"\n  Best passing variant: {best['variant']} -- {best['description']}")
        p(f"  avg_r={best['avg_r']}R  PF={best['pf']}  WR={best['win_rate']*100:.1f}%")
        beats = best["avg_r"] > T2_BASELINE["avg_r"]
        p(f"  {'BEATS T2 baseline' if beats else 'Does NOT beat T2 baseline'}")

    # Write full report
    with open(out_dir / "phase_t3mr_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Phase T3MR -- MeanReversionRSI Exit Engineering\n")
        f.write(f"Timeframe : {tf}  |  Config: rsi{rsi_n}/os{oversold}/exit{exit_rsi}/atr{atr_mult}/t{time_exit}\n")
        f.write(f"T2 Baseline: avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}  WR={T2_BASELINE['win_rate']*100:.1f}%\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nYear-by-year per variant:\n")
        for var_key, trades in all_trades_by_variant.items():
            if not trades:
                continue
            yb = year_breakdown(trades)
            f.write(f"\nVariant {var_key} -- {variants[var_key]}:\n")
            for yr in sorted(yb.keys()):
                ym = yb[yr]
                bear = " <<< BEAR" if yr == 2022 else ""
                f.write(f"  {yr}  n={ym['n']:4d}  wr={ym['win_rate']*100:5.1f}%  "
                        f"avg_r={ym['avg_r']:+.3f}R  total={ym['total_r']:+7.2f}R  "
                        f"dd={ym['max_dd_r']:.2f}R{bear}\n")

    p(f"\n[OK] phase_t3mr_variant_summary.csv")
    p(f"[OK] phase_t3mr_trades_A/B/C/D/E.csv")
    p(f"[OK] phase_t3mr_report.txt")
    sys.exit(0)


if __name__ == "__main__":
    main()
