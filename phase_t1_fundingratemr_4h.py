"""
Phase T1 -- Funding Rate MR Short on 4H
Funding rates are 8H (00:00, 08:00, 16:00 UTC). Each 8H period covers 2 x 4H bars.
Entry: funding_rate for current 8H period > entry_threshold -> short at NEXT 4H bar open
Exit: funding_rate < exit_threshold OR max hold_bars 4H bars
Filter: ema200_below mandatory
§4.2: 0.25R
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR    = Path("data/futures_universe/ohlcv_4h")
FUND_DIR    = Path("data/futures_universe/funding_rates")
OUT_DIR     = Path("data/research_fundingratemr_4h_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_THR  = [0.0003, 0.0005, 0.0007, 0.001]   # decimal (0.0003 = 0.03%/8h)
EXIT_THR   = [0.0001, 0.0002]
HOLD_BARS  = [3, 6, 12, 24]                      # 4H bars
ATR_MULT   = [2.0, 3.0]
COMBOS     = list(product(ENTRY_THR, EXIT_THR, HOLD_BARS, ATR_MULT))

ATR_PERIOD   = 14
EMA_PERIOD   = 200
WARMUP       = 250
COST_FLOOR   = 0.25
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
    closes = df["close"].values.astype(float)
    n = len(closes)
    atr = compute_atr(df)
    ema = np.full(n, np.nan)
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    d = df.copy()
    d["atr"]    = atr
    d["ema200"] = ema
    d["bear"]   = closes < ema
    return d

def build_funding_dict(symbol):
    """Build dict: 8H period start (UTC datetime) -> funding_rate."""
    fp = FUND_DIR / f"{symbol}_funding.csv"
    if not fp.exists():
        return {}
    fund = pd.read_csv(fp)
    if "funding_time" not in fund.columns or len(fund) == 0:
        return {}
    fund["dt"] = pd.to_datetime(fund["funding_time"], unit="ms", utc=True)
    # Map: each 8H period start (00, 08, 16) -> rate
    # key = "YYYY-MM-DD HH" where HH in {00,08,16}
    d = {}
    for _, row in fund.iterrows():
        key = row["dt"].strftime("%Y-%m-%d %H")
        d[key] = float(row["funding_rate"])
    return d

def bar_funding_key(ts_ms):
    """Given 4H bar open timestamp (ms), return the 8H funding key it falls in."""
    dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
    # 8H periods start at 00, 08, 16
    hour = dt.hour
    period_start = (hour // 8) * 8
    return dt.replace(hour=period_start, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H")

def backtest_symbol(symbol, d, warmup, fund_dict, et, xt, hb, am):
    if not fund_dict:
        return []
    timestamps = d["timestamp"].values
    opens  = d["open"].values.astype(float)
    closes = d["close"].values.astype(float)
    highs  = d["high"].values.astype(float)
    atrs   = d["atr"].values.astype(float)
    bear   = d["bear"].values

    # Precompute funding rate per bar
    fund_per_bar = np.zeros(len(timestamps))
    for i, ts in enumerate(timestamps):
        key = bar_funding_key(float(ts))
        fund_per_bar[i] = fund_dict.get(key, 0.0)

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
            curr_fr = fund_per_bar[i]
            if curr_fr < xt or bars_held >= hb:
                exit_p = opens[i]
                r = (entry_price - exit_p) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
            continue

        if np.isnan(atrs[i]) or not bear[i]:
            continue
        prev_fr = fund_per_bar[i - 1] if i > 0 else 0.0
        if prev_fr > et:
            if i + 1 >= len(opens):
                continue
            entry_price = opens[i + 1]
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
        print("ERROR: No 4H data. Run download_ohlcv_4h.py first.")
        import sys; sys.exit(1)
    symbols = [f.stem.replace("_4h", "") for f in files]

    fund_syms = [f.stem.replace("_funding", "") for f in FUND_DIR.glob("*_funding.csv")]
    valid = [s for s in symbols if s in set(fund_syms)]
    print(f"4H universe: {len(symbols)} symbols")
    print(f"With funding data: {len(valid)} symbols")
    print(f"Combos: {len(ENTRY_THR)}x{len(EXIT_THR)}x{len(HOLD_BARS)}x{len(ATR_MULT)} = {len(COMBOS)}")

    pre2021_syms = set()
    pre2021_csv = Path("data/futures_universe/symbols_pre2021.csv")
    if pre2021_csv.exists():
        pre2021_syms = set(pd.read_csv(pre2021_csv)["symbol"].tolist())

    all_rows = []
    for idx, symbol in enumerate(valid, 1):
        df = pd.read_csv(DATA_DIR / f"{symbol}_4h.csv")
        if len(df) < WARMUP + 24:
            continue
        d = precompute_symbol(df)
        fund_dict = build_funding_dict(symbol)
        if not fund_dict:
            continue
        for et, xt, hb, am in COMBOS:
            trades = backtest_symbol(symbol, d, WARMUP, fund_dict, et, xt, hb, am)
            for t in trades:
                ts_val = float(t["ts"])
                ts_dt = pd.Timestamp(ts_val, unit="ms", tz="UTC")
                all_rows.append({
                    "symbol": symbol, "et": et, "xt": xt, "hb": hb, "am": am,
                    "year": ts_dt.year, "r": t["r"]
                })
        if idx % 20 == 0:
            print(f"  [{idx}/{len(valid)}]")

    print(f"Total trades: {len(all_rows)}")
    df_all = pd.DataFrame(all_rows)
    if len(df_all):
        df_all.to_csv(OUT_DIR / "grid_trades.csv", index=False)
    return df_all, pre2021_syms, len(valid)

def stability_zone(df_all, et, xt, hb, am):
    et_steps = sorted(ENTRY_THR)
    hb_steps = sorted(HOLD_BARS)
    ei = et_steps.index(et)
    hi = hb_steps.index(hb)
    results = []
    for dei in [-1, 0, 1]:
        nei = ei + dei
        if not (0 <= nei < len(et_steps)): continue
        net = et_steps[nei]
        for dhi in [-1, 0, 1]:
            nhi = hi + dhi
            if not (0 <= nhi < len(hb_steps)): continue
            nhb = hb_steps[nhi]
            if dei == 0 and dhi == 0: continue
            sub = df_all[(df_all.et == net) & (df_all.xt == xt) & (df_all.hb == nhb) & (df_all.am == am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_all[(df_all.et == et) & (df_all.xt == xt) & (df_all.hb == hb) & (df_all.am == am)]
    center_ok = center["r"].mean() > 0 if len(center) else False
    return sum([center_ok] + results), len(results) + 1

def analyze(df_all, pre2021_syms, n_valid):
    lines = []
    lines.append("=" * 80)
    lines.append("Phase T1 -- Funding Rate MR Short on 4H")
    lines.append("=" * 80)
    lines.append(f"\nEntry:  prev 8H funding rate > entry_threshold  (ema200_below mandatory)")
    lines.append(f"        Enter SHORT at next 4H bar open.")
    lines.append(f"Exit:   curr funding rate < exit_threshold  OR  max hold_bars 4H bars")
    lines.append(f"Stop:   ATR * atr_mult ABOVE entry")
    lines.append(f"\nUniverse: {n_valid} symbols with funding data")
    lines.append(f"entry_thr: {[f'{x*100:.3f}%' for x in ENTRY_THR]}")
    lines.append(f"exit_thr:  {[f'{x*100:.3f}%' for x in EXIT_THR]}")
    lines.append(f"hold_bars: {HOLD_BARS} (4H bars)")
    lines.append(f"atr_mult:  {ATR_MULT}")
    lines.append(f"\nStability: entry_thr±1 AND hb±1, PASS if >=67%")
    lines.append(f"§4.2 floor: avg_r > {COST_FLOOR}R")
    lines.append(f"PRIMARY GATE: 2022 must be POSITIVE")
    lines.append(f"FLAG: 2025 > 40% of total R")

    if len(df_all) == 0:
        lines.append("\nWARNING: Zero trades generated.")
        report = "\n".join(lines)
        print(report)
        with open(OUT_DIR / "phase_t1_fundingratemr_4h_report.txt", "w") as f:
            f.write(report)
        return

    best_r = -np.inf
    best_combo = None
    all_pass_cost = []

    for am in ATR_MULT:
        sub_am = df_all[df_all.am == am]
        pass_list, warn_list = [], []
        for et, xt, hb in product(ENTRY_THR, EXIT_THR, HOLD_BARS):
            sub = sub_am[(sub_am.et == et) & (sub_am.xt == xt) & (sub_am.hb == hb)]
            if len(sub) == 0: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_all, et, xt, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else ("WARN" if pct >= 0.50 else "FAIL")
            e = {"et": et, "xt": xt, "hb": hb, "am": am, "avg_r": avg_r,
                 "n_trades": len(sub), "zone": f"{n_p}/{n_t}", "zone_pct": pct,
                 "cost_ok": cost_ok, "verdict": verdict}
            if verdict == "PASS": pass_list.append(e)
            elif verdict == "WARN": warn_list.append(e)
            if avg_r > best_r:
                best_r = avg_r; best_combo = e
            if verdict == "PASS" and cost_ok:
                all_pass_cost.append(e)

        lines.append(f"\n{'='*80}")
        lines.append(f"ATR mult={am}  |  PASS: {len(pass_list)}  WARN: {len(warn_list)}")
        lines.append("=" * 80)
        for lst, label in [(pass_list, "PASS"), (warn_list, "WARN")]:
            if lst:
                top = sorted(lst, key=lambda x: -x["avg_r"])[:5]
                lines.append(f"TOP {label} COMBOS")
                lines.append("-" * 60)
                for e in top:
                    lines.append(f"  [{e['verdict']}] et={e['et']*100:.3f}%  xt={e['xt']*100:.3f}%  "
                                 f"hb={e['hb']:2d}  zone={e['zone']} ({e['zone_pct']:.0%})  "
                                 f"trades={e['n_trades']}  avg_r={e['avg_r']:+.4f}  "
                                 f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Heatmap
    lines.append(f"\navg_r HEATMAP (xt=0.01%  ATR*2.0)")
    lines.append("-" * 60)
    xt0 = EXIT_THR[0]
    lines.append(f"  et\\hb  " + "  ".join(f"hb={h:2d}" for h in HOLD_BARS))
    for et in ENTRY_THR:
        row = []
        for hb in HOLD_BARS:
            sub = df_all[(df_all.et == et) & (df_all.xt == xt0) & (df_all.hb == hb) & (df_all.am == 2.0)]
            r = sub["r"].mean() if len(sub) else np.nan
            star = "*" if not np.isnan(r) and r > COST_FLOOR else " "
            row.append(f"{r:+.3f}{star}" if not np.isnan(r) else "   nan ")
        lines.append(f"  et={et*100:.3f}%: " + "  ".join(row))

    # 2022 gate
    lines.append(f"\n{'='*80}")
    lines.append("2022 BEAR MARKET CHECK")
    lines.append("=" * 80)
    if not all_pass_cost:
        lines.append(f"  No combos pass stability + §4.2.  Best avg_r: {best_r:+.4f}R")
    else:
        for e in sorted(all_pass_cost, key=lambda x: -x["avg_r"])[:3]:
            sub = df_all[(df_all.et==e["et"])&(df_all.xt==e["xt"])&(df_all.hb==e["hb"])&(df_all.am==e["am"])]
            sub_pre = sub[sub["symbol"].isin(pre2021_syms)] if pre2021_syms else sub
            yr2022 = sub_pre[sub_pre.year == 2022]["r"].sum()
            lines.append(f"  et={e['et']*100:.3f}% hb={e['hb']} am={e['am']}  avg_r={e['avg_r']:+.4f}  "
                        f"2022_r={yr2022:+.2f}  {'GATE_PASS' if yr2022 > 0 else 'GATE_FAIL'}")

    # Year-by-year
    lines.append(f"\n{'='*80}")
    lines.append("YEAR-BY-YEAR  (best combo)")
    lines.append("=" * 80)
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.et==bc["et"])&(df_all.xt==bc["xt"])&(df_all.hb==bc["hb"])&(df_all.am==bc["am"])]
        lines.append(f"  et={bc['et']*100:.3f}%  xt={bc['xt']*100:.3f}%  hb={bc['hb']}  ATR*{bc['am']}")
        lines.append(f"  {'Year':>6}  {'Trades':>6}  {'TotalR':>8}  {'AvgR':>8}  {'Win%':>6}")
        total_r = 0
        year_rs = {}
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
                lines.append(f"  [CONCENTRATED]  {yr} = {yr_r:+.1f}R = {pct:.0%} of total R"
                            + (" <-- RECENCY" if yr == 2025 else ""))

    lines.append(f"\n{'='*80}")
    lines.append("OVERALL VERDICT")
    lines.append("=" * 80)
    if all_pass_cost:
        lines.append(f"  {len(all_pass_cost)} combo(s) pass stability + §4.2.")
        lines.append(f"  VERDICT: REVIEW -- check 2022 gate.")
    else:
        lines.append(f"  NO combos pass stability + §4.2.  Best avg_r = {best_r:+.4f}R")
        lines.append(f"  VERDICT: HALT T1 -- insufficient edge.")

    report = "\n".join(lines)
    print(report)
    with open(OUT_DIR / "phase_t1_fundingratemr_4h_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {OUT_DIR}/phase_t1_fundingratemr_4h_report.txt")

if __name__ == "__main__":
    print("=" * 80)
    print("Phase T1 -- Funding Rate MR Short on 4H")
    print("=" * 80)
    df_all, pre2021_syms, n_valid = run_grid()
    analyze(df_all, pre2021_syms, n_valid)
    print("\nDone.")
