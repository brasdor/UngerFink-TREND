"""
Batch 2B — Linear Regression Slope Short (T1)
Entry: linreg slope of close over N bars is negative AND steeper than threshold
       AND close < EMA200
Exit: fixed time exit after hold_bars
Stop: ATR × mult ABOVE entry (short stop)
Timeframes: 4H, 1D
§4.2 floor: 0.15R (Futures)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_4H = Path("data/futures_universe/ohlcv_4h")
DATA_1D = Path("data/futures_universe/ohlcv_1d")
OUT_DIR = Path("data/research_linregshort_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINREG_N         = [10, 15, 20, 30, 40]
SLOPE_THRESHOLD  = [-0.001, -0.002, -0.005]  # normalized: slope_pct per bar < threshold
HOLD_BARS        = [5, 10, 15, 20]
ATR_MULT         = [2.0, 3.0]
TIMEFRAMES       = ["4H", "1D"]

ATR_PERIOD       = 14
EMA_PERIOD       = 200
WARMUP           = 250
COST_FLOOR       = 0.15

def compute_atr(df):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    atr = np.full(len(tr), np.nan)
    atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
    a = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, len(tr)):
        atr[i] = atr[i-1] * (1 - a) + tr[i] * a
    return atr

def compute_ema200(closes):
    n = len(closes)
    ema = np.full(n, np.nan)
    if n < EMA_PERIOD: return ema
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    return ema

def compute_linreg_slopes(closes, n):
    """Vectorized OLS slope for rolling window of length n.
    Returns array of slope / close (normalized slope as fraction per bar).
    """
    m = len(closes)
    slopes = np.full(m, np.nan)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_denom = ((x - x_mean) ** 2).sum()
    if x_denom == 0:
        return slopes
    for i in range(n - 1, m):
        y = closes[i - n + 1: i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_denom
        # normalize by current price to get pct-per-bar
        if closes[i] > 0:
            slopes[i] = slope / closes[i]
    return slopes

def precompute_symbol(df, max_n):
    closes = df["close"].values.astype(float)
    atr    = compute_atr(df)
    ema    = compute_ema200(closes)
    # Precompute slopes for all required N values
    slopes = {}
    for n in LINREG_N:
        if n <= max_n:
            slopes[n] = compute_linreg_slopes(closes, n)
    return {
        "timestamps": df["timestamp"].values.astype(np.int64),
        "opens":  df["open"].values.astype(float),
        "closes": closes,
        "highs":  df["high"].values.astype(float),
        "lows":   df["low"].values.astype(float),
        "atrs":   atr,
        "ema200": ema,
        "slopes": slopes,
    }

def backtest_symbol(d, warmup, ln, st, hb, am):
    timestamps = d["timestamps"]
    opens  = d["opens"];  closes = d["closes"]
    highs  = d["highs"];  lows   = d["lows"]
    atrs   = d["atrs"];   ema200 = d["ema200"]
    slope_arr = d["slopes"].get(ln)
    if slope_arr is None:
        return []

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
                r = (entry_price - opens[i]) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
            continue

        if np.isnan(atrs[i]) or np.isnan(ema200[i]) or np.isnan(slope_arr[i]):
            continue
        # Filter: bear regime
        if closes[i] >= ema200[i]:
            continue
        # Entry: slope < threshold (steeply negative)
        if slope_arr[i] >= st:  # st is negative, so this means slope too shallow
            continue
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

def stability_zone(df_tf, ln, st, hb, am):
    ln_steps = sorted(LINREG_N); hb_steps = sorted(HOLD_BARS)
    st_steps = sorted(SLOPE_THRESHOLD)
    li = ln_steps.index(ln); hi = hb_steps.index(hb); si = st_steps.index(st)
    results = []
    for dln in [-1, 0, 1]:
        nli = li + dln
        if not (0 <= nli < len(ln_steps)): continue
        nln = ln_steps[nli]
        for dsi in [-1, 0, 1]:
            nsi = si + dsi
            if not (0 <= nsi < len(st_steps)): continue
            nst = st_steps[nsi]
            sub = df_tf[(df_tf.ln==nln)&(df_tf.st==nst)&(df_tf.hb==hb)&(df_tf.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_tf[(df_tf.ln==ln)&(df_tf.st==st)&(df_tf.hb==hb)&(df_tf.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

def run_tf(tf):
    data_dir = DATA_4H if tf == "4H" else DATA_1D
    suffix   = "_4h" if tf == "4H" else "_1d"
    files    = sorted(data_dir.glob(f"*{suffix}.csv"))
    if not files:
        return pd.DataFrame()

    max_n = max(LINREG_N)
    combos = list(product(LINREG_N, SLOPE_THRESHOLD, HOLD_BARS, ATR_MULT))
    print(f"  {tf}: {len(files)} symbols, {len(combos)} combos")

    all_rows = []
    for idx, f in enumerate(files, 1):
        sym = f.stem.replace(suffix, "")
        df  = pd.read_csv(f)
        if len(df) < WARMUP + max_n + 10:
            continue
        d = precompute_symbol(df, max_n)
        for ln, st, hb, am in combos:
            trades = backtest_symbol(d, WARMUP, ln, st, hb, am)
            for t in trades:
                ts_dt = pd.Timestamp(int(t["ts"]), unit="ms", tz="UTC")
                all_rows.append({"sym": sym, "ln": ln, "st": st, "hb": hb, "am": am,
                                  "year": ts_dt.year, "r": t["r"], "tf": tf})
        if idx % 50 == 0:
            print(f"    [{idx}/{len(files)}]")

    return pd.DataFrame(all_rows)

def analyze_tf(df_tf, tf):
    lines = [f"\n{'='*60}", f"TIMEFRAME: {tf}", "="*60,
             f"  Total trades: {len(df_tf)}"]

    best_r = -np.inf; best_combo = None; all_pass_cost = []
    for am in ATR_MULT:
        sub_am = df_tf[df_tf.am == am]
        pass_list = []
        for ln, st, hb in product(LINREG_N, SLOPE_THRESHOLD, HOLD_BARS):
            sub = sub_am[(sub_am.ln==ln)&(sub_am.st==st)&(sub_am.hb==hb)]
            if len(sub) < 3: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_tf, ln, st, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else "FAIL"
            e = {"ln": ln, "st": st, "hb": hb, "am": am, "avg_r": avg_r,
                 "n_trades": len(sub), "zone_pct": pct, "verdict": verdict, "cost_ok": cost_ok}
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS": pass_list.append(e)
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        top = sorted(pass_list, key=lambda x: -x["avg_r"])[:5]
        lines.append(f"\n  ATR*{am}  PASS: {len(pass_list)}")
        for e in top:
            lines.append(f"    [PASS] ln={e['ln']:2d} st={e['st']:+.3f} hb={e['hb']:2d} "
                         f"zone={e['zone_pct']:.0%} t={e['n_trades']:5d} avg_r={e['avg_r']:+.4f} "
                         f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Year-by-year best combo
    if best_combo:
        bc = best_combo
        sub = df_tf[(df_tf.ln==bc["ln"])&(df_tf.st==bc["st"])&(df_tf.hb==bc["hb"])&(df_tf.am==bc["am"])]
        lines.append(f"\n  Year-by-year (best: ln={bc['ln']} st={bc['st']:+.3f} hb={bc['hb']} ATR*{bc['am']}):")
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            lines.append(f"    {yr}: t={len(y):4d}  total={y['r'].sum():+7.2f}R  avg={y['r'].mean():+.4f}R")

    verdict = "BREAKTHROUGH" if all_pass_cost else "HALT"
    lines.append(f"\n  VERDICT [{tf}]: {verdict} — best avg_r={best_r:+.4f}R  pass_§4.2={len(all_pass_cost)}")
    return "\n".join(lines), best_r, len(all_pass_cost)

if __name__ == "__main__":
    print("="*60)
    print("Batch 2B — Linear Regression Slope Short (T1)")
    print(f"linreg_n={LINREG_N}  slope_thr={SLOPE_THRESHOLD}")
    print(f"hold_bars={HOLD_BARS}  atr_mult={ATR_MULT}")
    print(f"§4.2 floor: {COST_FLOOR}R")
    print("="*60)

    sections = []; summary = []
    for tf in TIMEFRAMES:
        print(f"\nRunning {tf}...")
        df_tf = run_tf(tf)
        if len(df_tf) == 0:
            sections.append(f"\n[{tf}] No data.")
            summary.append(f"  {tf}: NO DATA")
            continue
        df_tf.to_csv(OUT_DIR / f"grid_trades_{tf.lower()}.csv", index=False)
        sec, best_r, n_pass = analyze_tf(df_tf, tf)
        sections.append(sec)
        summary.append(f"  {tf}: best_avg_r={best_r:+.4f}R  pass_§4.2={n_pass}  "
                        f"verdict={'BREAKTHROUGH' if n_pass>0 else 'HALT'}")

    header = ("="*60 + "\nBatch 2B — Linear Regression Slope Short T1\n"
              f"§4.2 floor: {COST_FLOOR}R\n")
    report = header + "\nSUMMARY:\n" + "\n".join(summary) + "\n" + "\n".join(sections)
    print(report)
    with open(OUT_DIR / "phase_t1_linregshort_report.txt", "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_DIR}/phase_t1_linregshort_report.txt")
