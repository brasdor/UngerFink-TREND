#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T3MR -- ConsecDownDaysMR Exit Engineering
UngerFink Pipeline / Andrea Unger Methodology

Tests exit variants against the T2 canonical: consec5/ema200/hold20/atr2.

Variants:
  A  RSI exit         -- exit when RSI(14) crosses above 50 (no time, no ATR)
  C  Combined         -- RSI > 50 OR ATR stop OR time exit (defensive)
  E  Time exit 20     -- hold exactly 20 bars (T2 canonical)
  F  Time exit 25     -- hold 25 bars (test longer hold on high avg_r system)

T2 baseline: avg_r=+0.470R  PF=2.31  WR=51.4%  (consec5/ema200/hold20/atr2)
Output: data/research_consecdowndays_mr_t3mr/
"""

from __future__ import annotations
import os, sys, warnings
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
OUT_DIR = ROOT / "data" / "research_consecdowndays_mr_t3mr"
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

CONSEC_N  = 5
HOLD_BARS = 20
ATR_MULT  = 2.0
FILTER    = "ema200_price_above"
RSI_EXIT  = 50
RSI_N     = 14

T2_BASELINE = {"avg_r": 0.4696, "pf": 2.31, "win_rate": 0.514}
MR_GATES    = {"win_rate_min": 0.50, "win_rate_max": 0.70, "avg_r_min": 0.10}

VARIANTS = {
    "A": "RSI(14) exit above 50",
    "C": "Combined: RSI>50 OR ATR stop OR time20",
    "E": "Time exit 20 bars (T2 canonical)",
    "F": "Time exit 25 bars",
}


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{sym}_1d.csv"
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
        if len(df) > 2000:
            df = df.iloc[-2000:].reset_index(drop=True)
        return df if len(df) >= 200 else None
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]; high = df["high"]; low = df["low"]
    # ATR
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # EMA200
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    # RSI14
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    df["rsi14"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    # Consecutive down streak
    down = (close < close.shift(1)).astype(int)
    streak = [0] * len(down)
    for i in range(1, len(down)):
        streak[i] = streak[i-1] + 1 if down.iloc[i] else 0
    df["streak"] = streak
    return df


def backtest_symbol(df: pd.DataFrame, sym: str, variant: str) -> list[dict]:
    close  = df["close"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    rsi    = df["rsi14"].values
    streak = df["streak"].values
    ts     = df["timestamp"].values

    filt = close > ema200  # ema200_price_above always on for this config

    trades: list[dict] = []
    in_pos = False
    e_price = stop = 0.0
    e_bar = 0; e_ts = None

    for i in range(len(df)):
        if np.isnan(atr[i]) or np.isnan(rsi[i]):
            continue

        if not in_pos:
            if int(streak[i]) == CONSEC_N and filt[i]:
                in_pos = True
                e_price = close[i]; e_bar = i; e_ts = ts[i]
                stop = e_price - ATR_MULT * atr[i]
        else:
            bars_held = i - e_bar
            ep = reason = None

            if variant == "A":
                # RSI exit only
                if rsi[i] >= RSI_EXIT:
                    ep, reason = close[i], "rsi_exit"

            elif variant == "C":
                # RSI OR ATR stop OR time20
                if low_v[i] <= stop:
                    ep, reason = stop, "atr_stop"
                elif rsi[i] >= RSI_EXIT:
                    ep, reason = close[i], "rsi_exit"
                elif bars_held >= HOLD_BARS:
                    ep, reason = close[i], "time_exit"

            elif variant == "E":
                if low_v[i] <= stop:
                    ep, reason = stop, "atr_stop"
                elif bars_held >= HOLD_BARS:
                    ep, reason = close[i], "time_exit"

            elif variant == "F":
                if low_v[i] <= stop:
                    ep, reason = stop, "atr_stop"
                elif bars_held >= 25:
                    ep, reason = close[i], "time_exit"

            if reason:
                risk = e_price - stop
                if risk > 1e-9:
                    rm = (ep - e_price) / risk
                    entry_dt = pd.Timestamp(e_ts)
                    trades.append({
                        "symbol":      sym,
                        "entry_time":  entry_dt,
                        "exit_time":   pd.Timestamp(ts[i]),
                        "net_r":       round(float(rm), 4),
                        "win":         int(rm > 0),
                        "bars_held":   bars_held,
                        "exit_reason": reason,
                        "year":        entry_dt.year,
                    })
                in_pos = False

    return trades


def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0,"avg_bars":0.0}
    rs = np.array([t["net_r"] for t in trades])
    w = rs[rs>0]; l = np.abs(rs[rs<0])
    pf = w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum))
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":dd,"total_r":float(np.sum(rs)),
            "avg_bars":float(np.mean([t["bars_held"] for t in trades]))}


def year_breakdown(trades: list[dict]) -> dict[int, dict]:
    df = pd.DataFrame(trades)
    if df.empty: return {}
    return {int(yr): calc_metrics(grp.to_dict("records"))
            for yr, grp in df.groupby("year")}


def main() -> None:
    p("=" * 70)
    p("  Phase T3MR -- ConsecDownDaysMR Exit Engineering")
    p(f"  Entry: consec_n={CONSEC_N}  filter={FILTER}  atr_mult={ATR_MULT}")
    p(f"  T2 baseline: avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}  WR={T2_BASELINE['win_rate']*100:.1f}%")
    p("=" * 70)

    loaded = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"  Symbols loaded: {len(loaded)}/{len(SYMBOLS)}")

    all_results = []
    all_trades_by_var: dict[str, list[dict]] = {}

    for var, desc in VARIANTS.items():
        p(f"\n  Running Variant {var}: {desc}...")
        var_trades: list[dict] = []
        for sym, df in loaded:
            var_trades.extend(backtest_symbol(df, sym, var))
        m  = calc_metrics(var_trades)
        yb = year_breakdown(var_trades)
        y2022 = yb.get(2022, {})

        wr_pass = MR_GATES["win_rate_min"] <= m["win_rate"] <= MR_GATES["win_rate_max"]
        ar_pass = m["avg_r"] >= MR_GATES["avg_r_min"]
        gate    = "PASS" if (wr_pass and ar_pass) else "FAIL"
        beats   = m["avg_r"] > T2_BASELINE["avg_r"] and m["pf"] > T2_BASELINE["pf"]

        p(f"    n={m['n']:4d}  WR={m['win_rate']*100:5.1f}%  avg_r={m['avg_r']:+.4f}R  "
          f"PF={m['pf']:.2f}  DD={m['max_dd_r']:.2f}R  total={m['total_r']:+.1f}R  "
          f"bars={m['avg_bars']:.1f}  {gate}{'  *** BEATS T2' if beats else ''}")
        if y2022:
            p(f"    2022: n={y2022['n']}  wr={y2022['win_rate']*100:.1f}%  "
              f"total={y2022['total_r']:+.2f}R  {'OK' if y2022['total_r']>=-20 else 'FAIL-BEAR'}")

        all_results.append({"variant":var,"description":desc,"n":m["n"],
            "win_rate":round(m["win_rate"],4),"avg_r":round(m["avg_r"],4),
            "pf":round(m["pf"],2),"max_dd_r":round(m["max_dd_r"],2),
            "total_r":round(m["total_r"],2),"avg_bars":round(m["avg_bars"],1),
            "y2022_total":round(y2022.get("total_r",float("nan")),2) if y2022 else None,
            "wr_gate":"PASS" if wr_pass else "FAIL",
            "avgr_gate":"PASS" if ar_pass else "FAIL",
            "beats_t2":beats})
        all_trades_by_var[var] = var_trades

        if var_trades:
            pd.DataFrame(var_trades).to_csv(OUT_DIR/f"phase_t3mr_trades_{var}.csv", index=False)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUT_DIR/"phase_t3mr_variant_summary.csv", index=False)

    # Comparison table
    p()
    p("=" * 70)
    p("  T3MR EXIT VARIANT COMPARISON")
    p(f"  T2 Baseline: avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}  WR={T2_BASELINE['win_rate']*100:.1f}%")
    p("=" * 70)
    p(f"  {'Var':<3} {'Description':<35} {'N':>5} {'WR%':>6} {'AvgR':>8} {'PF':>5} {'DD':>7} {'TotR':>8} {'2022':>7} Gate")
    p("  " + "-" * 90)
    for r in all_results:
        y22 = f"{r['y2022_total']:+.1f}R" if r["y2022_total"] is not None else "  N/A"
        beat = " ***" if r["beats_t2"] else ""
        p(f"  {r['variant']:<3} {r['description']:<35} {r['n']:>5} "
          f"{r['win_rate']*100:>5.1f}% {r['avg_r']:>+8.4f} {r['pf']:>5.2f} "
          f"{r['max_dd_r']:>6.2f}R {r['total_r']:>+8.1f}R {y22:>7} "
          f"{r['wr_gate']}/{r['avgr_gate']}{beat}")

    # Year-by-year per variant
    p()
    for var in VARIANTS:
        trades = all_trades_by_var[var]
        if not trades: continue
        yb = year_breakdown(trades)
        p(f"\n  Variant {var}:")
        for yr in sorted(yb.keys()):
            ym = yb[yr]
            bear = " <<< BEAR" if yr==2022 else ""
            p(f"    {yr}  n={ym['n']:4d}  wr={ym['win_rate']*100:5.1f}%  "
              f"avg_r={ym['avg_r']:+.3f}R  total={ym['total_r']:+7.2f}R  dd={ym['max_dd_r']:.2f}R{bear}")

    # Best passing variant
    passing = [r for r in all_results if r["wr_gate"]=="PASS" and r["avgr_gate"]=="PASS"]
    best_var = max(passing, key=lambda x: x["avg_r"]) if passing else None

    p()
    if best_var:
        p(f"  Best passing variant: {best_var['variant']} -- {best_var['description']}")
        p(f"  avg_r={best_var['avg_r']}R  PF={best_var['pf']}  WR={best_var['win_rate']*100:.1f}%")
        # Save best variant label for T4
        (OUT_DIR/"best_variant.txt").write_text(best_var["variant"], encoding="utf-8")

    with open(OUT_DIR/"phase_t3mr_report.txt","w",encoding="utf-8") as f:
        f.write("T3MR ConsecDownDaysMR Exit Engineering\n")
        f.write(f"Config: consec{CONSEC_N}/ema200/hold{HOLD_BARS}/atr{ATR_MULT}\n")
        f.write(f"T2 Baseline: avg_r={T2_BASELINE['avg_r']}R  PF={T2_BASELINE['pf']}\n\n")
        f.write(summary_df.to_string(index=False))

    p(f"\n[OK] phase_t3mr_variant_summary.csv")
    p(f"[OK] phase_t3mr_trades_A/C/E/F.csv")
    p(f"[OK] best_variant.txt = {best_var['variant'] if best_var else 'N/A'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
