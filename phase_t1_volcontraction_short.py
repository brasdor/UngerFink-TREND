"""
Phase T1 -- Volatility Contraction Short (4H and 1D)
Entry (ALL must be true):
  - ATR(14) has been below the Nth percentile for at least compression_bars consecutive bars
  - Price breaks BELOW the lowest low of the compression period
  - Price below EMA200
Exit: fixed time exit after hold_bars
Stop: ATR * atr_mult ABOVE entry
§4.2: 0.25R
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_4H = Path("data/futures_universe/ohlcv_4h")
DATA_1D = Path("data/futures_universe/ohlcv_1d")
OUT_DIR = Path("data/research_volcontraction_short_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPRESSION_BARS = [5, 10, 15, 20]
ATR_PERCENTILE   = [20, 25, 30]
HOLD_BARS        = [5, 10, 15, 20]
ATR_MULT         = [2.0, 3.0]
TIMEFRAMES       = ["4H", "1D"]
COMBOS           = list(product(COMPRESSION_BARS, ATR_PERCENTILE, HOLD_BARS, ATR_MULT))

ATR_PERIOD     = 14
EMA_PERIOD     = 200
ATR_PCT_WINDOW = 252 * 6  # ~1yr of 4H bars; for 1D this becomes ~252 bars (we cap at available)
ATR_PCT_WINDOW_1D = 252
ATR_PCT_MIN_BARS = 50
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

def precompute_symbol(df, tf):
    n = len(df)
    atr = compute_atr(df)
    window = ATR_PCT_WINDOW if tf == "4H" else ATR_PCT_WINDOW_1D
    # ATR percentile rank
    pct_rank = np.full(n, np.nan)
    for i in range(ATR_PCT_MIN_BARS, n):
        start = max(0, i - window)
        w = atr[start:i]
        if len(w) < 10: continue
        pct_rank[i] = float((w <= atr[i]).mean() * 100)
    closes = df["close"].values.astype(float)
    ema = np.full(n, np.nan)
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    d = df.copy()
    d["atr"]      = atr
    d["atr_pct"]  = pct_rank
    d["ema200"]   = ema
    d["bear"]     = closes < ema
    return d

def backtest_symbol(symbol, d, warmup, cb, ap, hb, am):
    """
    cb = compression_bars (min consecutive bars ATR must be below percentile)
    ap = atr_percentile threshold
    hb = hold_bars
    am = atr_mult
    """
    timestamps = d["timestamp"].values
    opens  = d["open"].values.astype(float)
    closes = d["close"].values.astype(float)
    lows   = d["low"].values.astype(float)
    highs  = d["high"].values.astype(float)
    atrs   = d["atr"].values.astype(float)
    pct    = d["atr_pct"].values
    bear   = d["bear"].values

    # Precompute: for each bar i, how many consecutive bars (ending at i) have atr_pct < ap
    consec_low_atr = np.zeros(len(atrs), dtype=int)
    for i in range(1, len(atrs)):
        if np.isnan(pct[i]):
            consec_low_atr[i] = 0
        elif pct[i] < ap:
            consec_low_atr[i] = consec_low_atr[i-1] + 1
        else:
            consec_low_atr[i] = 0

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

        if np.isnan(atrs[i]) or not bear[i]:
            continue
        # Need at least cb bars of low ATR before this bar
        # (i.e., bars i-cb through i-1 must all have atr_pct < ap)
        if consec_low_atr[i - 1] < cb:
            continue
        # Entry: close < lowest low of the compression window
        comp_start = max(0, i - cb)
        compression_low = lows[comp_start:i].min()
        if closes[i] < compression_low:
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

def run_grid_tf(tf):
    data_dir = DATA_4H if tf == "4H" else DATA_1D
    suffix   = "_4h" if tf == "4H" else "_1d"
    files    = sorted(data_dir.glob(f"*{suffix}.csv"))
    if not files:
        print(f"  No {tf} data found, skipping.")
        return pd.DataFrame(), set(), 0
    symbols = [f.stem.replace(suffix, "") for f in files]
    print(f"\n  {tf}: {len(symbols)} symbols, {len(COMBOS)} combos")

    pre2021_syms = set()
    pre2021_csv = Path("data/futures_universe/symbols_pre2021.csv")
    if pre2021_csv.exists():
        pre2021_syms = set(pd.read_csv(pre2021_csv)["symbol"].tolist())

    all_rows = []
    for idx, symbol in enumerate(symbols, 1):
        df = pd.read_csv(data_dir / f"{symbol}{suffix}.csv")
        if len(df) < WARMUP + 20:
            continue
        d = precompute_symbol(df, tf)
        for cb, ap, hb, am in COMBOS:
            trades = backtest_symbol(symbol, d, WARMUP, cb, ap, hb, am)
            for t in trades:
                ts_val = float(t["ts"])
                ts_dt  = pd.Timestamp(ts_val, unit="ms", tz="UTC")
                all_rows.append({
                    "symbol": symbol, "cb": cb, "ap": ap, "hb": hb, "am": am,
                    "year": ts_dt.year, "r": t["r"], "tf": tf
                })
        if idx % 30 == 0:
            print(f"    [{idx}/{len(symbols)}]")
    return pd.DataFrame(all_rows), pre2021_syms, len(symbols)

def stability_zone(df_tf, cb, ap, hb, am):
    cb_steps = sorted(COMPRESSION_BARS)
    hb_steps = sorted(HOLD_BARS)
    ci = cb_steps.index(cb); hi = hb_steps.index(hb)
    results = []
    for dcb in [-1, 0, 1]:
        nci = ci + dcb
        if not (0 <= nci < len(cb_steps)): continue
        ncb = cb_steps[nci]
        for dhb in [-1, 0, 1]:
            nhi = hi + dhb
            if not (0 <= nhi < len(hb_steps)): continue
            nhb = hb_steps[nhi]
            if dcb == 0 and dhb == 0: continue
            sub = df_tf[(df_tf.cb==ncb)&(df_tf.ap==ap)&(df_tf.hb==nhb)&(df_tf.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_tf[(df_tf.cb==cb)&(df_tf.ap==ap)&(df_tf.hb==hb)&(df_tf.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

def analyze_tf(df_tf, pre2021_syms, n_syms, tf):
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"TIMEFRAME: {tf}")
    lines.append("=" * 80)
    lines.append(f"  Universe: {n_syms} symbols")
    lines.append(f"  Total trades: {len(df_tf)}")

    best_r = -np.inf
    best_combo = None
    all_pass_cost = []

    for am in ATR_MULT:
        sub_am = df_tf[df_tf.am == am]
        pass_list, warn_list = [], []
        for cb, ap, hb in product(COMPRESSION_BARS, ATR_PERCENTILE, HOLD_BARS):
            sub = sub_am[(sub_am.cb==cb)&(sub_am.ap==ap)&(sub_am.hb==hb)]
            if len(sub) == 0: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_tf, cb, ap, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else ("WARN" if pct >= 0.50 else "FAIL")
            e = {"cb": cb, "ap": ap, "hb": hb, "am": am, "avg_r": avg_r,
                 "n_trades": len(sub), "zone": f"{n_p}/{n_t}", "zone_pct": pct,
                 "cost_ok": cost_ok, "verdict": verdict}
            if verdict == "PASS": pass_list.append(e)
            elif verdict == "WARN": warn_list.append(e)
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        lines.append(f"\n  ATR mult={am}  |  PASS: {len(pass_list)}  WARN: {len(warn_list)}")
        for lst, label in [(pass_list, "PASS"), (warn_list, "WARN")]:
            if lst:
                top = sorted(lst, key=lambda x: -x["avg_r"])[:5]
                lines.append(f"  TOP {label}:")
                for e in top:
                    lines.append(f"    [{e['verdict']}] cb={e['cb']:2d}  ap={e['ap']:2d}  hb={e['hb']:2d}  "
                                 f"zone={e['zone']} ({e['zone_pct']:.0%})  "
                                 f"t={e['n_trades']}  avg_r={e['avg_r']:+.4f}  "
                                 f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Heatmap: cb vs hb for best ap, am=2.0
    best_ap = ATR_PERCENTILE[1]  # middle value (25)
    lines.append(f"\n  avg_r HEATMAP (ap={best_ap}  ATR*2.0)  * = §4.2")
    lines.append("  " + "  ".join(f"hb={h:2d}" for h in HOLD_BARS))
    for cb in COMPRESSION_BARS:
        row = []
        for hb in HOLD_BARS:
            sub = df_tf[(df_tf.cb==cb)&(df_tf.ap==best_ap)&(df_tf.hb==hb)&(df_tf.am==2.0)]
            r = sub["r"].mean() if len(sub) else np.nan
            star = "*" if not np.isnan(r) and r > COST_FLOOR else " "
            row.append(f"{r:+.3f}{star}" if not np.isnan(r) else "  nan  ")
        lines.append(f"  cb={cb:2d}: " + "  ".join(row))

    # 2022 gate
    lines.append(f"\n  2022 GATE:")
    if not all_pass_cost:
        lines.append(f"    No combos pass stability+§4.2. Best avg_r={best_r:+.4f}R")
    else:
        for e in sorted(all_pass_cost, key=lambda x: -x["avg_r"])[:3]:
            sub = df_tf[(df_tf.cb==e["cb"])&(df_tf.ap==e["ap"])&(df_tf.hb==e["hb"])&(df_tf.am==e["am"])]
            sub_pre = sub[sub["symbol"].isin(pre2021_syms)] if pre2021_syms else sub
            yr2022 = sub_pre[sub_pre.year == 2022]["r"].sum()
            lines.append(f"    cb={e['cb']} ap={e['ap']} hb={e['hb']} am={e['am']}  "
                        f"avg_r={e['avg_r']:+.4f}  2022_r={yr2022:+.2f}  "
                        f"{'GATE_PASS' if yr2022 > 0 else 'GATE_FAIL'}")

    # Year-by-year best combo
    if best_combo:
        bc = best_combo
        sub = df_tf[(df_tf.cb==bc["cb"])&(df_tf.ap==bc["ap"])&(df_tf.hb==bc["hb"])&(df_tf.am==bc["am"])]
        lines.append(f"\n  YEAR-BY-YEAR (best combo: cb={bc['cb']} ap={bc['ap']} hb={bc['hb']} ATR*{bc['am']}):")
        total_r = 0; year_rs = {}
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            tr = y["r"].sum(); ar = y["r"].mean()
            wp = (y["r"] > 0).mean() * 100
            lines.append(f"    {yr}: t={len(y):4d}  total={tr:+7.2f}R  avg={ar:+.4f}R  win={wp:.1f}%")
            total_r += tr; year_rs[yr] = tr
        for yr, yr_r in year_rs.items():
            pct = yr_r / total_r if total_r != 0 else 0
            if abs(pct) > CONCENTRATION_THR and yr_r > 0:
                lines.append(f"  [CONCENTRATED] {yr}={yr_r:+.1f}R={pct:.0%}"
                            + (" <-- RECENCY" if yr == 2025 else ""))

    verdict_str = "REVIEW" if all_pass_cost else "HALT"
    lines.append(f"\n  VERDICT [{tf}]: {verdict_str} -- best avg_r={best_r:+.4f}R")
    return "\n".join(lines), best_r, len(all_pass_cost)

if __name__ == "__main__":
    print("=" * 80)
    print("Phase T1 -- Volatility Contraction Short (4H + 1D)")
    print("=" * 80)
    print(f"compression_bars: {COMPRESSION_BARS}")
    print(f"atr_percentile:   {ATR_PERCENTILE}")
    print(f"hold_bars:        {HOLD_BARS}")
    print(f"atr_mult:         {ATR_MULT}")
    print(f"Combos per TF:    {len(COMBOS)}")

    all_sections = []
    summary = []

    for tf in TIMEFRAMES:
        print(f"\nRunning {tf}...")
        df_tf, pre2021_syms, n_syms = run_grid_tf(tf)
        if len(df_tf) == 0:
            all_sections.append(f"\n[{tf}] No data / no trades.")
            summary.append(f"  {tf}: NO DATA")
            continue
        df_tf.to_csv(OUT_DIR / f"grid_trades_{tf.lower()}.csv", index=False)
        section, best_r, n_pass = analyze_tf(df_tf, pre2021_syms, n_syms, tf)
        all_sections.append(section)
        verdict = "REVIEW" if n_pass > 0 else "HALT"
        summary.append(f"  {tf}: best_avg_r={best_r:+.4f}R  pass_combos={n_pass}  verdict={verdict}")

    header = (
        "=" * 80 + "\n"
        "Phase T1 -- Volatility Contraction Short\n"
        "=" * 80 + "\n"
        "Entry: ATR < Nth pct for >= compression_bars bars THEN close < compression_period_low AND close < EMA200\n"
        "Exit:  fixed time exit (hold_bars)\n"
        f"§4.2 floor: {COST_FLOOR}R\n"
    )
    summary_block = "\nSUMMARY:\n" + "\n".join(summary)
    report = header + summary_block + "\n".join(all_sections)
    print(report)
    with open(OUT_DIR / "phase_t1_volcontraction_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {OUT_DIR}/phase_t1_volcontraction_report.txt")
    print("\nDone. Do not proceed to T2 until you review the report.")
