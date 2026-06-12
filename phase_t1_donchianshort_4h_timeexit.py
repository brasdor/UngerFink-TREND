"""
Phase T1 -- Donchian Short Time Exit on 4H
Entry: close < Donchian N-period low + EMA200 below + ATR pct >= 50
Exit: fixed time exit after hold_bars (4H bars)
Stop: ATR * atr_mult ABOVE entry
§4.2: 0.25R
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR   = Path("data/futures_universe/ohlcv_4h")
OUT_DIR    = Path("data/research_donchianshort_4h_timeexit_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DONCHIAN_N   = [10, 15, 20, 25, 30]
HOLD_BARS    = [10, 20, 30, 40, 60]
ATR_MULT     = [2.0, 3.0]
COMBOS       = list(product(DONCHIAN_N, HOLD_BARS, ATR_MULT))

ATR_PERIOD       = 14
EMA_PERIOD       = 200
ATR_PCT_WINDOW   = 252 * 6   # ~1 year of 4H bars
ATR_PCT_MIN_BARS = 50
ATR_PCT_FLOOR    = 50        # must be >= 50th percentile
WARMUP           = 300
COST_FLOOR       = 0.25
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
    # ATR percentile rank (vectorized numpy loop)
    pct_rank = np.full(n, np.nan)
    for i in range(ATR_PCT_MIN_BARS, n):
        start = max(0, i - ATR_PCT_WINDOW)
        window = atr[start:i]
        if len(window) < 2: continue
        pct_rank[i] = float((window <= atr[i]).mean() * 100)
    # EMA200
    closes = df["close"].values.astype(float)
    ema = np.full(n, np.nan)
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)

    d = df.copy()
    d["atr"]       = atr
    d["atr_pct"]   = pct_rank
    d["ema200"]    = ema
    d["atr_ok"]    = pct_rank >= ATR_PCT_FLOOR
    d["bear_regime"] = closes < ema
    return d

def backtest_symbol(symbol, d, warmup, dn, hb, am):
    timestamps = d["timestamp"].values
    opens  = d["open"].values.astype(float)
    closes = d["close"].values.astype(float)
    lows   = d["low"].values.astype(float)
    atrs   = d["atr"].values.astype(float)
    atr_ok = d["atr_ok"].values
    bear   = d["bear_regime"].values

    # Donchian low: rolling min of close over previous dn bars (exclusive)
    don_low = np.full(len(closes), np.nan)
    for i in range(dn, len(closes)):
        don_low[i] = closes[i - dn : i].min()

    trades = []
    in_trade = False
    entry_price = stop = bars_held = 0

    for i in range(warmup, len(closes)):
        if in_trade:
            bars_held += 1
            # check stop (high of bar vs stop)
            if d["high"].values[i] >= stop:
                exit_p = stop
                r = (entry_price - exit_p) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r, "exit": "stop"})
                in_trade = False
                continue
            if bars_held >= hb:
                exit_p = opens[i]
                r = (entry_price - exit_p) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r, "exit": "time"})
                in_trade = False
            continue

        if np.isnan(don_low[i]) or np.isnan(atrs[i]) or not atr_ok[i] or not bear[i]:
            continue
        # entry: close < donchian low
        if closes[i] < don_low[i]:
            entry_price = opens[i + 1] if i + 1 < len(opens) else closes[i]
            stop = entry_price + atrs[i] * am
            risk = stop - entry_price
            if risk <= 0:
                continue
            in_trade = True
            bars_held = 0

    return trades

def run_grid():
    files = sorted(DATA_DIR.glob("*_4h.csv"))
    if not files:
        print("ERROR: No 4H data found. Run download_ohlcv_4h.py first.")
        sys.exit(1)

    symbols = [f.stem.replace("_4h", "") for f in files]
    print(f"Universe: {len(symbols)} symbols")
    print(f"Combos: {len(DONCHIAN_N)}x{len(HOLD_BARS)}x{len(ATR_MULT)} = {len(COMBOS)}")

    pre2021_syms = set()
    pre2021_csv = Path("data/futures_universe/symbols_pre2021.csv")
    if pre2021_csv.exists():
        pre2021_syms = set(pd.read_csv(pre2021_csv)["symbol"].tolist())

    all_rows = []
    for idx, symbol in enumerate(symbols, 1):
        df = pd.read_csv(DATA_DIR / f"{symbol}_4h.csv")
        if len(df) < WARMUP + 60:
            continue
        d = precompute_symbol(df)
        for dn, hb, am in COMBOS:
            trades = backtest_symbol(symbol, d, WARMUP, dn, hb, am)
            for t in trades:
                ts = pd.Timestamp(t["ts"], unit="ms", tz="UTC") if isinstance(t["ts"], (int, float)) else pd.Timestamp(t["ts"])
                all_rows.append({
                    "symbol": symbol, "dn": dn, "hb": hb, "am": am,
                    "year": ts.year, "r": t["r"], "exit": t["exit"]
                })
        if idx % 30 == 0:
            print(f"  [{idx}/{len(symbols)}]")

    print(f"Total trades: {len(all_rows)}")
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "grid_trades.csv", index=False)
    return df_all, pre2021_syms, symbols

def stability_zone(df_all, dn, hb, am):
    dn_steps = sorted(DONCHIAN_N)
    hb_steps = sorted(HOLD_BARS)
    neighbors = []
    for ddn in [-1, 0, 1]:
        ndn = dn_steps[dn_steps.index(dn) + ddn] if 0 <= dn_steps.index(dn) + ddn < len(dn_steps) else None
        for dhb in [-1, 0, 1]:
            nhb = hb_steps[hb_steps.index(hb) + dhb] if 0 <= hb_steps.index(hb) + dhb < len(hb_steps) else None
            if ndn is None or nhb is None or (ddn == 0 and dhb == 0):
                continue
            neighbors.append((ndn, nhb))
    results = []
    center = df_all[(df_all.dn == dn) & (df_all.hb == hb) & (df_all.am == am)]
    center_r = center["r"].mean() if len(center) else 0.0
    results.append(center_r > 0)
    for ndn, nhb in neighbors:
        sub = df_all[(df_all.dn == ndn) & (df_all.hb == nhb) & (df_all.am == am)]
        results.append(sub["r"].mean() > 0 if len(sub) else False)
    if not results:
        return 0, 0
    return sum(results), len(results)

def analyze(df_all, pre2021_syms, n_symbols):
    lines = []
    lines.append("=" * 80)
    lines.append("Phase T1 -- Donchian Short Time Exit on 4H")
    lines.append("=" * 80)
    lines.append(f"\nEntry:  close < Donchian(N) low  +  EMA200 below  +  ATR pct >= 50")
    lines.append(f"Exit:   fixed time exit after hold_bars (4H bars)")
    lines.append(f"Stop:   ATR * atr_mult ABOVE entry")
    lines.append(f"\nUniverse: {n_symbols} symbols")
    lines.append(f"Combos:   {len(COMBOS)}")
    lines.append(f"donchian_n: {DONCHIAN_N}")
    lines.append(f"hold_bars:  {HOLD_BARS}")
    lines.append(f"atr_mult:   {ATR_MULT}")
    lines.append(f"\nStability: N±1 AND hb±1, PASS if >=67%")
    lines.append(f"§4.2 floor: avg_r > {COST_FLOOR}R")
    lines.append(f"PRIMARY GATE: 2022 must be POSITIVE")
    lines.append(f"FLAG: 2025 > 40% of total R")

    best_combo = None
    best_r = -np.inf

    for am in ATR_MULT:
        sub_am = df_all[df_all.am == am]
        pass_list, warn_list, fail_list = [], [], []

        for dn, hb in product(DONCHIAN_N, HOLD_BARS):
            sub = sub_am[(sub_am.dn == dn) & (sub_am.hb == hb)]
            if len(sub) == 0:
                continue
            avg_r = sub["r"].mean()
            n_pass, n_total = stability_zone(df_all, dn, hb, am)
            pct = n_pass / n_total if n_total else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else ("WARN" if pct >= 0.50 else "FAIL")
            entry = {"dn": dn, "hb": hb, "am": am, "avg_r": avg_r,
                     "n_trades": len(sub), "zone": f"{n_pass}/{n_total}",
                     "zone_pct": pct, "cost_ok": cost_ok, "verdict": verdict}
            if verdict == "PASS": pass_list.append(entry)
            elif verdict == "WARN": warn_list.append(entry)
            else: fail_list.append(entry)
            if avg_r > best_r:
                best_r = avg_r
                best_combo = entry

        lines.append(f"\n{'='*80}")
        lines.append(f"ATR mult={am}  |  PASS: {len(pass_list)}  WARN: {len(warn_list)}  FAIL: {len(fail_list)}")
        lines.append("=" * 80)

        for lst, label in [(pass_list, "PASS"), (warn_list, "WARN")]:
            if lst:
                top = sorted(lst, key=lambda x: -x["avg_r"])[:10]
                lines.append(f"TOP {label} COMBOS")
                lines.append("-" * 60)
                for e in top:
                    f42 = "PASS" if e["cost_ok"] else "FAIL"
                    lines.append(f"  [{e['verdict']}] N={e['dn']:2d}  hb={e['hb']:2d}  "
                                 f"zone={e['zone']} ({e['zone_pct']:.0%})  "
                                 f"trades={e['n_trades']}  avg_r={e['avg_r']:+.4f}  §4.2={f42}")

    # Heatmap
    lines.append(f"\navg_r HEATMAP (ATR*2.0)  * = passes §4.2")
    lines.append("-" * 60)
    lines.append(f"  N\\hb  " + "  ".join(f"hb={h:2d}" for h in HOLD_BARS))
    for dn in DONCHIAN_N:
        row = []
        for hb in HOLD_BARS:
            sub = df_all[(df_all.dn == dn) & (df_all.hb == hb) & (df_all.am == 2.0)]
            r = sub["r"].mean() if len(sub) else np.nan
            star = "*" if r > COST_FLOOR else " "
            row.append(f"{r:+.3f}{star}")
        lines.append(f"  N={dn:2d}: " + "  ".join(row))

    # 2022 check
    lines.append(f"\n{'='*80}")
    lines.append("2022 BEAR MARKET CHECK  (PASS + §4.2 combos only)")
    lines.append("=" * 80)
    lines.append(f"  {len(pre2021_syms)} symbols have pre-2021 data (2022 gate applies to these).")

    passed_cost = [e for e in pass_list if e["cost_ok"]] if pass_list else []
    # re-collect across all am
    all_pass_cost = []
    for am in ATR_MULT:
        for dn, hb in product(DONCHIAN_N, HOLD_BARS):
            sub = df_all[(df_all.dn == dn) & (df_all.hb == hb) & (df_all.am == am)]
            if len(sub) == 0: continue
            n_p, n_t = stability_zone(df_all, dn, hb, am)
            pct = n_p / n_t if n_t else 0
            avg_r = sub["r"].mean()
            if pct >= 0.67 and avg_r > COST_FLOOR:
                all_pass_cost.append({"dn": dn, "hb": hb, "am": am, "avg_r": avg_r})

    if not all_pass_cost:
        lines.append(f"  No combos pass stability + §4.2.  Best avg_r = {best_r:+.4f}R")
    else:
        for e in sorted(all_pass_cost, key=lambda x: -x["avg_r"])[:3]:
            sub = df_all[(df_all.dn == e["dn"]) & (df_all.hb == e["hb"]) & (df_all.am == e["am"])]
            # filter to pre2021 symbols for 2022 gate
            sub_pre = sub[sub["symbol"].isin(pre2021_syms)] if pre2021_syms else sub
            yr2022 = sub_pre[sub_pre.year == 2022]["r"].sum()
            lines.append(f"  N={e['dn']} hb={e['hb']} am={e['am']}  avg_r={e['avg_r']:+.4f}  2022 total_r={yr2022:+.2f}  {'PASS' if yr2022 > 0 else 'FAIL'}")

    # Year-by-year for best combo
    lines.append(f"\n{'='*80}")
    lines.append("YEAR-BY-YEAR  (best combo by avg_r)")
    lines.append("=" * 80)
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.dn == bc["dn"]) & (df_all.hb == bc["hb"]) & (df_all.am == bc["am"])]
        lines.append(f"  N={bc['dn']}  hb={bc['hb']}  ATR*{bc['am']}  verdict={bc['verdict']}  §4.2={'PASS' if bc['cost_ok'] else 'FAIL'}")
        lines.append(f"  {'Year':>6}  {'Trades':>6}  {'TotalR':>8}  {'AvgR':>8}  {'Win%':>6}  {'PF':>5}")
        total_r = 0
        year_rs = {}
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            tr = y["r"].sum()
            ar = y["r"].mean()
            wp = (y["r"] > 0).mean() * 100
            wins = y[y["r"] > 0]["r"].sum()
            losses = abs(y[y["r"] <= 0]["r"].sum())
            pf = wins / losses if losses > 0 else np.inf
            lines.append(f"  {yr:>6}  {len(y):>6}  {tr:>+8.2f}  {ar:>+8.4f}  {wp:>5.1f}%  {pf:>5.2f}")
            total_r += tr
            year_rs[yr] = tr
        lines.append("")
        for yr, yr_r in year_rs.items():
            pct_of_total = yr_r / total_r if total_r != 0 else 0
            if abs(pct_of_total) > CONCENTRATION_THR and yr_r > 0:
                lines.append(f"  [CONCENTRATED]  {yr} = {yr_r:+.1f}R = {pct_of_total:.0%} of total R"
                             + (" <-- 2025 RECENCY" if yr == 2025 else ""))

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
    with open(OUT_DIR / "phase_t1_donchianshort_4h_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {OUT_DIR}/phase_t1_donchianshort_4h_report.txt")

if __name__ == "__main__":
    print("=" * 80)
    print("Phase T1 -- Donchian Short Time Exit on 4H")
    print("=" * 80)
    df_all, pre2021_syms, symbols = run_grid()
    analyze(df_all, pre2021_syms, len(symbols))
    print("\nDone. Do not proceed to T2 until you review the report.")
