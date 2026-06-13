#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- RSIShortBreakdown Concept Discovery
UngerFink Pipeline / Andrea Unger Methodology (Section 19A reclassified)

System character: SHORT-SIDE TREND FOLLOWING
  NOT mean reversion -- RSI overbought on 4H crypto with price<EMA200
  produces 30-45% WR with large winners = trend following profile.

Entry : RSI(N) > overbought AND close < EMA(200) --> SHORT at close
Exit  : Chandelier trailing stop (ACT=4R, trail=3xATR above lowest low)
        OR time backstop after 30 bars
        Let winners run -- do NOT use fixed time exit as primary

Initial stop: ATR x atr_mult ABOVE entry (short stop-loss)
Chandelier:   activated when unrealised profit reaches ACT_R
              stop = lowest_low_since_entry + trail_mult x ATR
              ratchets DOWN only -- protects accrued profit
              exit when HIGH >= chandelier stop

Parameters tested:
  rsi_n        : [7, 10, 14]
  overbought   : [80, 85, 90]
  atr_mult     : [2.0, 3.0]   (initial stop width)
  chandelier   : ACT=4R, trail=3.0 (fixed -- canonical Donchian params)
  time_backstop: 30 bars (fixed -- prevents runaway open positions)
  filter       : ema200_price_below only (mandatory)

TREND FOLLOWING gates (NOT MR):
  §4.1 win rate : 30-45%  (TF range, not MR 50-70%)
  §4.2 avg_r    : > 0.25R (Futures cost floor, higher than spot MR)
  §2.1 stability: >= 67%

Data: full 4H 2021-2026 from ohlcv_cache (2022 bear market covered)
Output: data/research_rsishortbreakdown_t1/
"""

from __future__ import annotations
import itertools, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)

ROOT      = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "research_rsimrshort_t1" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_rsishortbreakdown_t1"
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
]

RSI_N        = [7, 10, 14]
OVERBOUGHT   = [80, 85, 90]
ATR_MULTS    = [2.0, 3.0]

# Chandelier params (fixed)
CHANDELIER_ACT   = 4.0   # activate when unrealised profit >= 4R
CHANDELIER_TRAIL = 3.0   # trail = lowest_low + 3 x ATR
TIME_BACKSTOP    = 30    # exit after 30 bars regardless

EMA_N = 200
MIN_BARS = 500  # need 2022 coverage

TF_GATES = {
    "min_trades":    80,
    "win_rate_min":  0.30,   # trend following range
    "win_rate_max":  0.45,
    "avg_r_min":     0.25,   # Futures cost floor
    "stability_min": 0.67,
}


# =============================================================================
# DATA
# =============================================================================

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
    out   = df.copy()
    close = out["close"]; high = out["high"]; low = out["low"]
    tr    = pd.concat([high-low,(high-close.shift()).abs(),
                       (low-close.shift()).abs()],axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["ema200"] = close.ewm(span=EMA_N, adjust=False).mean()
    for n in RSI_N:
        d  = close.diff()
        ag = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        out[f"rsi{n}"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    return out


# =============================================================================
# BACKTEST  (SHORT with Chandelier exit)
# =============================================================================

def backtest_symbol(df: pd.DataFrame,
                    rsi_n: int, overbought: int, atr_mult: float) -> list[dict]:
    """
    SHORT entry: RSI > overbought AND close < EMA200
    Exit logic (priority order):
      1. Initial stop hit: high >= entry + atr_mult*ATR (before Chandelier active)
      2. Chandelier stop hit: high >= lowest_low + trail*ATR (after activation)
      3. Time backstop: exit at close after TIME_BACKSTOP bars
    """
    close  = df["close"].values
    high_v = df["high"].values
    low_v  = df["low"].values
    atr    = df["atr14"].values
    ema200 = df["ema200"].values
    rsi    = df[f"rsi{rsi_n}"].values
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades: list[dict] = []
    in_pos          = False
    e_price         = 0.0
    initial_stop    = 0.0
    initial_risk    = 0.0
    chandelier_stop = 0.0
    chandelier_active = False
    lowest_low      = 0.0
    e_bar           = 0
    e_ts            = None

    for i in range(len(df)):
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            # Entry: RSI overbought + price below EMA200
            if rsi[i] > overbought and close[i] < ema200[i]:
                in_pos            = True
                e_price           = close[i]
                e_bar             = i
                e_ts              = ts[i]
                initial_risk      = atr_mult * atr[i]
                initial_stop      = e_price + initial_risk
                chandelier_stop   = initial_stop
                chandelier_active = False
                lowest_low        = close[i]
        else:
            bars_held = i - e_bar

            # Update lowest low (for short, track how far price has fallen)
            if close[i] < lowest_low:
                lowest_low = close[i]
            if low_v[i] < lowest_low:
                lowest_low = low_v[i]

            # Check Chandelier activation
            unrealised_r = (e_price - close[i]) / max(initial_risk, 1e-9)
            if not chandelier_active and unrealised_r >= CHANDELIER_ACT:
                chandelier_active = True

            # Compute current stop
            if chandelier_active:
                new_ch_stop = lowest_low + CHANDELIER_TRAIL * atr[i]
                # Chandelier only ratchets DOWN for shorts (lock more profit as price falls)
                chandelier_stop = min(chandelier_stop, new_ch_stop)
                current_stop = chandelier_stop
            else:
                current_stop = initial_stop

            # Exit checks
            ep = reason = None

            if high_v[i] >= current_stop:
                ep     = current_stop
                reason = "chandelier" if chandelier_active else "initial_stop"
            elif bars_held >= TIME_BACKSTOP:
                ep     = close[i]
                reason = "time_backstop"

            if reason:
                r_mult = (e_price - ep) / max(initial_risk, 1e-9)  # positive when price fell
                if initial_risk > 1e-9:
                    entry_dt = pd.Timestamp(e_ts)
                    trades.append({
                        "r_multiple":   float(r_mult),
                        "bars_held":    bars_held,
                        "reason":       reason,
                        "year":         entry_dt.year,
                        "chandelier":   chandelier_active,
                    })
                in_pos = False

    return trades


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,
                "max_dd_r":0.0,"total_r":0.0,"avg_bars":0.0}
    rs   = np.array([t["r_multiple"] for t in trades])
    w    = rs[rs>0]; l = np.abs(rs[rs<0])
    pf   = w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum  = np.cumsum(rs); peak = np.maximum.accumulate(cum)
    dd   = float(np.max(peak-cum)) if len(cum) else 0.0
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),
            "avg_r":float(np.mean(rs)),"pf":float(pf),
            "max_dd_r":dd,"total_r":float(np.sum(rs)),
            "avg_bars":float(np.mean([t["bars_held"] for t in trades]))}


def year_breakdown(trades: list[dict]) -> dict[int, dict]:
    by_year: dict[int, list[dict]] = {}
    for t in trades:
        by_year.setdefault(t["year"], []).append(t)
    return {yr: calc_metrics(ts) for yr, ts in by_year.items()}


def exit_breakdown(trades: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        counts[t["reason"]] = counts.get(t["reason"], 0) + 1
    return counts


def stability_score(grid: pd.DataFrame, rsi_n: int, overbought: int,
                    atr_mult: float) -> float:
    rsi_ns = sorted(RSI_N); ovbs = sorted(OVERBOUGHT)
    try:
        ri = rsi_ns.index(rsi_n); oi = ovbs.index(overbought)
    except ValueError:
        return 0.0
    n_nbrs = [rsi_ns[j] for j in range(max(0,ri-1), min(len(rsi_ns),ri+2))]
    o_nbrs = [ovbs[j]   for j in range(max(0,oi-1), min(len(ovbs),oi+2))]
    zone = grid[(grid["rsi_n"].isin(n_nbrs)) & (grid["overbought"].isin(o_nbrs)) &
                (grid["atr_mult"]==atr_mult)]
    if zone.empty: return 0.0
    return round((zone["avg_r"] > 0).sum() / len(zone), 4)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 70)
    p("  Phase T1 -- RSIShortBreakdown Concept Discovery")
    p("  Character: SHORT-SIDE TREND FOLLOWING (not MR)")
    p("  Entry: RSI(N) > overbought + price < EMA200 --> SHORT")
    p(f"  Exit : Chandelier ACT={CHANDELIER_ACT}R trail={CHANDELIER_TRAIL}xATR OR {TIME_BACKSTOP}-bar backstop")
    p("  Gates: WR 30-45%, avg_r > 0.25R  (TREND FOLLOWING, not MR)")
    p("=" * 70)

    # Load data
    loaded = []
    for sym in SYMBOLS:
        df = load_ohlcv(sym)
        if df is not None:
            loaded.append((sym, add_indicators(df)))
    p(f"  Symbols loaded: {len(loaded)}")
    if not loaded:
        p(f"  ERROR: no data in {CACHE_DIR}"); sys.exit(1)

    # Coverage check
    first_df = loaded[0][1]
    if "timestamp" in first_df.columns:
        first_date = str(first_df["timestamp"].iloc[0].date())
        p(f"  Data from: {first_date}  (2022 {'COVERED' if first_date < '2022-01-01' else 'PARTIAL'})")

    param_combos = list(itertools.product(RSI_N, OVERBOUGHT, ATR_MULTS))
    p(f"  Param combos: {len(param_combos)}  (rsi x ob x atr)")

    # Accumulate trades
    combo_trades: dict[tuple, list[dict]] = {k: [] for k in param_combos}
    for idx, (sym, df) in enumerate(loaded, 1):
        if idx % 10 == 0 or idx == 1 or idx == len(loaded):
            p(f"  Processing {idx}/{len(loaded)} {sym}...")
        for (rsi_n, ob, atr_m) in param_combos:
            combo_trades[(rsi_n, ob, atr_m)].extend(
                backtest_symbol(df, rsi_n, ob, atr_m))

    # Build grid
    rows = []
    for (rsi_n, ob, atr_m) in param_combos:
        trades = combo_trades[(rsi_n, ob, atr_m)]
        m = calc_metrics(trades)
        eb = exit_breakdown(trades)
        rows.append({
            "rsi_n":rsi_n,"overbought":ob,"atr_mult":atr_m,
            "num_trades":m["n"],"win_rate":round(m["win_rate"],4),
            "avg_r":round(m["avg_r"],4),"pf":round(m["pf"],4),
            "max_dd_r":round(m["max_dd_r"],2),"total_r":round(m["total_r"],2),
            "avg_bars":round(m["avg_bars"],1),
            "pct_chandelier":round(eb.get("chandelier",0)/max(m["n"],1)*100,1),
            "pct_init_stop":round(eb.get("initial_stop",0)/max(m["n"],1)*100,1),
            "pct_backstop":round(eb.get("time_backstop",0)/max(m["n"],1)*100,1),
        })
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR/"phase_t1_grid_4h.csv", index=False)
    p(f"  Grid saved: {len(grid)} rows")

    # Gate filters
    p(f"\n--- Gate Analysis ---")
    viable = grid[grid["num_trades"] >= TF_GATES["min_trades"]].copy()
    p(f"  Combos >= {TF_GATES['min_trades']} trades: {len(viable)}")
    if viable.empty:
        p("  FAIL: no combos meet min_trades"); sys.exit(1)

    viable_wr = viable[(viable["win_rate"] >= TF_GATES["win_rate_min"]) &
                       (viable["win_rate"] <= TF_GATES["win_rate_max"])]
    p(f"  After win_rate 30-45%: {len(viable_wr)}")

    viable_ar = viable_wr[viable_wr["avg_r"] >= TF_GATES["avg_r_min"]]
    p(f"  After avg_r >= {TF_GATES['avg_r_min']}R: {len(viable_ar)}")

    if viable_ar.empty:
        p("\n  No combos pass both WR and avg_r gates.")
        p("  Showing best by avg_r (all trade-qualified):")
        top = viable.nlargest(10,"avg_r")[
            ["rsi_n","overbought","atr_mult","num_trades","win_rate","avg_r","pf"]]
        p(top.to_string(index=False))
        sys.exit(1)

    p("  Computing stability zones...")
    viable_ar = viable_ar.copy()
    viable_ar["stability"] = viable_ar.apply(
        lambda r: stability_score(grid, int(r["rsi_n"]), int(r["overbought"]),
                                  float(r["atr_mult"])), axis=1)
    stable = viable_ar[viable_ar["stability"] >= TF_GATES["stability_min"]].copy()
    p(f"  After stability >= {TF_GATES['stability_min']}: {len(stable)}")

    if stable.empty:
        p("  NOTE: no stability-passing combos -- showing best by stability:")
        stable = viable_ar.nlargest(20,"stability")
        gate_pass = False
    else:
        gate_pass = True

    stable = stable.sort_values(["stability","avg_r"],ascending=[False,False]).reset_index(drop=True)

    # Heatmap: avg_r by (rsi_n, overbought)
    pivot = grid.groupby(["rsi_n","overbought"])["avg_r"].mean().unstack("overbought")
    p(f"\n  avg_r heatmap (rsi_n vs overbought) | all atr_mults averaged:")
    p(pivot.round(3).to_string())

    # Exit reason breakdown for top combos
    p(f"\n  Exit reason analysis (top combos):")
    p(f"  {'rsi':>4} {'ob':>4} {'atr':>5} {'trades':>7} {'WR%':>6} {'AvgR':>8} "
      f"{'chan%':>6} {'stop%':>6} {'back%':>6} {'stab':>6}")
    p("  " + "-"*72)
    for _, row in stable.head(10).iterrows():
        p(f"  {int(row['rsi_n']):>4} {int(row['overbought']):>4} {row['atr_mult']:>5.1f} "
          f"{int(row['num_trades']):>7} {row['win_rate']*100:>5.1f}% "
          f"{row['avg_r']:>+7.4f} {row['pct_chandelier']:>5.1f}% "
          f"{row['pct_init_stop']:>5.1f}% {row['pct_backstop']:>5.1f}% "
          f"{row['stability']:>6.2f}")

    # Year-by-year for top 5 combos
    p(f"\n  Year-by-year for top 5 combos:")
    for rank, (_, row) in enumerate(stable.head(5).iterrows(), 1):
        key    = (int(row["rsi_n"]), int(row["overbought"]), float(row["atr_mult"]))
        trades = combo_trades[key]
        yb     = year_breakdown(trades)
        p(f"\n  #{rank}: RSI({int(row['rsi_n'])})>ob{int(row['overbought'])}/atr{row['atr_mult']}  "
          f"avg_r={row['avg_r']:+.4f}R  WR={row['win_rate']*100:.1f}%  "
          f"PF={row['pf']:.2f}  stability={row['stability']:.2f}")
        p(f"  {'Year':>5}  {'N':>4}  {'WR%':>6}  {'AvgR':>8}  {'TotR':>8}  {'DD':>7}  Note")
        for yr in sorted(yb.keys()):
            ym   = yb[yr]
            note = "<<< BEAR (filter most active)" if yr==2022 else \
                   "(bull yr, few entries)" if yr in (2021,2024) else ""
            neg  = " WEAK" if ym["avg_r"] < 0 else ""
            p(f"  {yr:>5}  {ym['n']:>4}  {ym['win_rate']*100:>5.1f}%  "
              f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
              f"{ym['max_dd_r']:>6.2f}R  {note}{neg}")

    # Save outputs
    stable.to_csv(OUT_DIR/"phase_t1_stability_ranking.csv", index=False)
    with open(OUT_DIR/"phase_t1_summary.txt","w",encoding="utf-8") as f:
        f.write("RSIShortBreakdown T1 -- SHORT-SIDE TREND FOLLOWING\n")
        f.write(f"Chandelier ACT={CHANDELIER_ACT}R trail={CHANDELIER_TRAIL}xATR backstop={TIME_BACKSTOP}bars\n")
        f.write(f"TF Gates: WR 30-45%, avg_r>{TF_GATES['avg_r_min']}R, stability>={TF_GATES['stability_min']}\n\n")
        f.write(f"Grid: {len(grid)}  Viable (trades): {len(viable)}  "
                f"Viable (WR+R): {len(viable_ar)}  Stable: {len(stable)}\n\n")
        f.write("Top 15 stable combos:\n")
        cols = ["rsi_n","overbought","atr_mult","num_trades","win_rate","avg_r","pf",
                "max_dd_r","avg_bars","pct_chandelier","pct_backstop","stability","total_r"]
        f.write(stable.head(15)[cols].to_string(index=False))
        f.write("\n\navg_r heatmap (rsi_n vs overbought):\n")
        f.write(pivot.round(3).to_string())
        f.write(f"\n\nT1 GATE: {'PASS' if gate_pass else 'MARGINAL'}\n")

    p(f"\n[OK] phase_t1_grid_4h.csv         ({len(grid)} combos)")
    p(f"[OK] phase_t1_stability_ranking.csv ({len(stable)} stable)")
    p(f"[OK] phase_t1_summary.txt")
    p(f"\nT1 GATE: {'PASS' if gate_pass else 'MARGINAL -- review required'}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
