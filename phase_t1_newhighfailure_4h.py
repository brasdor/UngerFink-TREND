"""
Phase T1 -- New High Failure Short on 4H
Entry: high > max(prev N highs) AND close < open (bearish candle) AND close < EMA200
Exit: fixed time exit after hold_bars (4H bars)
Stop: entry_bar_high + ATR * atr_mult ABOVE entry
§4.2: 0.25R
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR = Path("data/futures_universe/ohlcv_4h")
OUT_DIR  = Path("data/research_newhighfailure_4h_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_N = [5, 10, 15, 20, 30]
HOLD_BARS  = [5, 10, 15, 20]
ATR_MULT   = [2.0, 3.0]
COMBOS     = list(product(LOOKBACK_N, HOLD_BARS, ATR_MULT))

ATR_PERIOD     = 14
EMA_PERIOD     = 200
ATR_PCT_WINDOW = 252 * 6
ATR_PCT_MIN    = 50
WARMUP         = 300
COST_FLOOR     = 0.25
CONCENTRATION_THR = 0.40

def compute_atr(df):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    atr = np.full(len(tr), np.nan)
    atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
    alpha = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr

def precompute_symbol(df):
    n = len(df)
    atr = compute_atr(df)
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    opens  = df["open"].values.astype(float)
    ema = np.full(n, np.nan)
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    d = df.copy()
    d["atr"]       = atr
    d["ema200"]    = ema
    d["is_bearish"] = closes < opens   # bearish candle
    d["bear_regime"] = closes < ema
    return d

def backtest_symbol(symbol, d, warmup, ln, hb, am):
    timestamps = d["timestamp"].values
    opens  = d["open"].values.astype(float)
    closes = d["close"].values.astype(float)
    highs  = d["high"].values.astype(float)
    atrs   = d["atr"].values.astype(float)
    is_bearish = d["is_bearish"].values
    bear = d["bear_regime"].values

    # Rolling max of prev ln highs (exclusive of current bar)
    prev_high_max = np.full(len(highs), np.nan)
    for i in range(ln, len(highs)):
        prev_high_max[i] = highs[i - ln : i].max()

    trades = []
    in_trade = False
    entry_price = stop = risk = bars_held = 0

    for i in range(warmup, len(closes)):
        if in_trade:
            bars_held += 1
            if highs[i] >= stop:
                r = (entry_price - stop) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
                continue
            if bars_held >= hb:
                exit_p = opens[i]
                r = (entry_price - exit_p) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
            continue

        if np.isnan(prev_high_max[i]) or np.isnan(atrs[i]):
            continue
        if not bear[i] or not is_bearish[i]:
            continue
        # New high: current bar's high > max of prev ln highs
        if highs[i] > prev_high_max[i]:
            entry_price = closes[i]          # enter at close of signal bar
            stop = highs[i] + atrs[i] * am  # stop above the new high bar
            risk = stop - entry_price
            if risk <= 0:
                continue
            in_trade = True
            bars_held = 0

    return trades

def run_grid():
    files = sorted(DATA_DIR.glob("*_4h.csv"))
    if not files:
        print("ERROR: No 4H data. Run download_ohlcv_4h.py first.")
        import sys; sys.exit(1)
    symbols = [f.stem.replace("_4h", "") for f in files]
    print(f"Universe: {len(symbols)} symbols")
    print(f"Combos: {len(LOOKBACK_N)}x{len(HOLD_BARS)}x{len(ATR_MULT)} = {len(COMBOS)}")

    pre2021_syms = set()
    pre2021_csv = Path("data/futures_universe/symbols_pre2021.csv")
    if pre2021_csv.exists():
        pre2021_syms = set(pd.read_csv(pre2021_csv)["symbol"].tolist())

    all_rows = []
    for idx, symbol in enumerate(symbols, 1):
        df = pd.read_csv(DATA_DIR / f"{symbol}_4h.csv")
        if len(df) < WARMUP + 20:
            continue
        d = precompute_symbol(df)
        for ln, hb, am in COMBOS:
            trades = backtest_symbol(symbol, d, WARMUP, ln, hb, am)
            for t in trades:
                ts_val = float(t["ts"])
                ts_dt = pd.Timestamp(ts_val, unit="ms", tz="UTC")
                all_rows.append({
                    "symbol": symbol, "ln": ln, "hb": hb, "am": am,
                    "year": ts_dt.year, "r": t["r"]
                })
        if idx % 30 == 0:
            print(f"  [{idx}/{len(symbols)}]")

    print(f"Total trades: {len(all_rows)}")
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "grid_trades.csv", index=False)
    return df_all, pre2021_syms, len(symbols)

def stability_zone(df_all, ln, hb, am):
    ln_steps = sorted(LOOKBACK_N)
    hb_steps = sorted(HOLD_BARS)
    li = ln_steps.index(ln); hi = hb_steps.index(hb)
    results = []
    for dli in [-1, 0, 1]:
        nli = li + dli
        if not (0 <= nli < len(ln_steps)): continue
        nln = ln_steps[nli]
        for dhi in [-1, 0, 1]:
            nhi = hi + dhi
            if not (0 <= nhi < len(hb_steps)): continue
            nhb = hb_steps[nhi]
            if dli == 0 and dhi == 0: continue
            sub = df_all[(df_all.ln==nln)&(df_all.hb==nhb)&(df_all.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_all[(df_all.ln==ln)&(df_all.hb==hb)&(df_all.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

def analyze(df_all, pre2021_syms, n_symbols):
    lines = []
    lines.append("=" * 80)
    lines.append("Phase T1 -- New High Failure Short on 4H")
    lines.append("=" * 80)
    lines.append(f"\nEntry: high > max(prev N highs) AND bearish candle AND close < EMA200")
    lines.append(f"       Enter SHORT at close. Stop = bar_high + ATR*mult.")
    lines.append(f"\nUniverse: {n_symbols} symbols")
    lines.append(f"lookback_n: {LOOKBACK_N}")
    lines.append(f"hold_bars:  {HOLD_BARS}")
    lines.append(f"atr_mult:   {ATR_MULT}")
    lines.append(f"Combos:     {len(COMBOS)}")
    lines.append(f"\nStability: N±1 AND hb±1, PASS if >=67%")
    lines.append(f"§4.2 floor: avg_r > {COST_FLOOR}R")
    lines.append(f"PRIMARY GATE: 2022 must be POSITIVE")
    lines.append(f"FLAG: 2025 > 40% of total R")

    best_r = -np.inf
    best_combo = None
    all_pass_cost = []

    for am in ATR_MULT:
        sub_am = df_all[df_all.am == am]
        pass_list, warn_list, fail_list = [], [], []
        for ln, hb in product(LOOKBACK_N, HOLD_BARS):
            sub = sub_am[(sub_am.ln==ln)&(sub_am.hb==hb)]
            if len(sub) == 0: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_all, ln, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else ("WARN" if pct >= 0.50 else "FAIL")
            e = {"ln": ln, "hb": hb, "am": am, "avg_r": avg_r,
                 "n_trades": len(sub), "zone": f"{n_p}/{n_t}", "zone_pct": pct,
                 "cost_ok": cost_ok, "verdict": verdict}
            if verdict == "PASS": pass_list.append(e)
            elif verdict == "WARN": warn_list.append(e)
            else: fail_list.append(e)
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        lines.append(f"\n{'='*80}")
        lines.append(f"ATR mult={am}  |  PASS: {len(pass_list)}  WARN: {len(warn_list)}  FAIL: {len(fail_list)}")
        lines.append("=" * 80)
        for lst, label in [(pass_list, "PASS"), (warn_list, "WARN")]:
            if lst:
                top = sorted(lst, key=lambda x: -x["avg_r"])[:8]
                lines.append(f"TOP {label} COMBOS")
                lines.append("-" * 60)
                for e in top:
                    lines.append(f"  [{e['verdict']}] N={e['ln']:2d}  hb={e['hb']:2d}  "
                                 f"zone={e['zone']} ({e['zone_pct']:.0%})  "
                                 f"t={e['n_trades']}  avg_r={e['avg_r']:+.4f}  "
                                 f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Heatmap
    lines.append(f"\navg_r HEATMAP (ATR*2.0)  * = §4.2")
    lines.append("-" * 50)
    lines.append(f"  N\\hb  " + "  ".join(f"hb={h:2d}" for h in HOLD_BARS))
    for ln in LOOKBACK_N:
        row = []
        for hb in HOLD_BARS:
            sub = df_all[(df_all.ln==ln)&(df_all.hb==hb)&(df_all.am==2.0)]
            r = sub["r"].mean() if len(sub) else np.nan
            star = "*" if not np.isnan(r) and r > COST_FLOOR else " "
            row.append(f"{r:+.3f}{star}" if not np.isnan(r) else "   nan ")
        lines.append(f"  N={ln:2d}: " + "  ".join(row))

    # 2022 gate
    lines.append(f"\n{'='*80}")
    lines.append("2022 BEAR MARKET CHECK")
    lines.append("=" * 80)
    if not all_pass_cost:
        lines.append(f"  No combos pass stability+§4.2. Best avg_r={best_r:+.4f}R")
    else:
        for e in sorted(all_pass_cost, key=lambda x: -x["avg_r"])[:3]:
            sub = df_all[(df_all.ln==e["ln"])&(df_all.hb==e["hb"])&(df_all.am==e["am"])]
            sub_pre = sub[sub["symbol"].isin(pre2021_syms)] if pre2021_syms else sub
            yr2022 = sub_pre[sub_pre.year == 2022]["r"].sum()
            lines.append(f"  N={e['ln']} hb={e['hb']} am={e['am']}  avg_r={e['avg_r']:+.4f}  "
                        f"2022_r={yr2022:+.2f}  {'GATE_PASS' if yr2022 > 0 else 'GATE_FAIL'}")

    # Year-by-year
    lines.append(f"\n{'='*80}")
    lines.append("YEAR-BY-YEAR  (best combo)")
    lines.append("=" * 80)
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.ln==bc["ln"])&(df_all.hb==bc["hb"])&(df_all.am==bc["am"])]
        lines.append(f"  N={bc['ln']}  hb={bc['hb']}  ATR*{bc['am']}  verdict={bc['verdict']}")
        lines.append(f"  {'Year':>6}  {'Trades':>6}  {'TotalR':>8}  {'AvgR':>8}  {'Win%':>6}")
        total_r = 0; year_rs = {}
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            tr = y["r"].sum(); ar = y["r"].mean()
            wp = (y["r"] > 0).mean() * 100
            lines.append(f"  {yr:>6}  {len(y):>6}  {tr:>+8.2f}  {ar:>+8.4f}  {wp:>5.1f}%")
            total_r += tr; year_rs[yr] = tr
        lines.append("")
        for yr, yr_r in year_rs.items():
            pct = yr_r / total_r if total_r != 0 else 0
            if abs(pct) > CONCENTRATION_THR and yr_r > 0:
                lines.append(f"  [CONCENTRATED]  {yr}={yr_r:+.1f}R={pct:.0%}"
                            + (" <-- RECENCY" if yr == 2025 else ""))

    lines.append(f"\n{'='*80}")
    lines.append("OVERALL VERDICT")
    lines.append("=" * 80)
    if all_pass_cost:
        lines.append(f"  {len(all_pass_cost)} combo(s) pass stability + §4.2.")
        lines.append(f"  Best avg_r: {max(e['avg_r'] for e in all_pass_cost):+.4f}R")
        lines.append(f"  VERDICT: REVIEW -- check 2022 gate and year-by-year.")
    else:
        lines.append(f"  NO combos pass stability + §4.2.  Best avg_r = {best_r:+.4f}R")
        lines.append(f"  VERDICT: HALT T1 -- insufficient edge.")

    report = "\n".join(lines)
    print(report)
    with open(OUT_DIR / "phase_t1_newhighfailure_4h_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {OUT_DIR}/")

if __name__ == "__main__":
    print("=" * 80)
    print("Phase T1 -- New High Failure Short on 4H")
    print("=" * 80)
    df_all, pre2021_syms, n_syms = run_grid()
    analyze(df_all, pre2021_syms, n_syms)
    print("\nDone. Do not proceed to T2 until you review the report.")
