"""
Batch 2C — MA Crossover Short (T1)
Entry: fast EMA crosses below slow EMA AND close < EMA200
Exit: reverse crossover (fast crosses back above slow) OR max 30 bars
Stop: ATR × mult ABOVE entry
Timeframes: 4H, 1D
§4.2 floor: 0.15R
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_4H  = Path("data/futures_universe/ohlcv_4h")
DATA_1D  = Path("data/futures_universe/ohlcv_1d")
OUT_DIR  = Path("data/research_macrossshort_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_EMA   = [10, 20, 30, 50]
SLOW_EMA   = [50, 100, 200]
ATR_MULT   = [2.0, 3.0]
MAX_HOLD   = 30
TIMEFRAMES = ["4H", "1D"]
COST_FLOOR = 0.15
ATR_PERIOD = 14
EMA200_N   = 200
WARMUP     = 250

# Valid pairs only (fast < slow)
PAIRS = [(f, s) for f in FAST_EMA for s in SLOW_EMA if f < s]

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

def compute_ema(closes, n):
    m = len(closes)
    ema = np.full(m, np.nan)
    if m < n: return ema
    ema[n - 1] = closes[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, m):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    return ema

def precompute_symbol(df):
    closes = df["close"].values.astype(float)
    atr    = compute_atr(df)
    ema200 = compute_ema(closes, EMA200_N)
    emas   = {}
    for n in set(FAST_EMA + SLOW_EMA):
        emas[n] = compute_ema(closes, n)
    return {
        "timestamps": df["timestamp"].values.astype(np.int64),
        "opens":  df["open"].values.astype(float),
        "closes": closes,
        "highs":  df["high"].values.astype(float),
        "lows":   df["low"].values.astype(float),
        "atrs":   atr,
        "ema200": ema200,
        "emas":   emas,
    }

def backtest_symbol(d, warmup, fast_n, slow_n, am):
    timestamps = d["timestamps"]
    opens  = d["opens"];  closes = d["closes"]
    highs  = d["highs"]
    atrs   = d["atrs"];   ema200 = d["ema200"]
    fast_arr = d["emas"].get(fast_n)
    slow_arr = d["emas"].get(slow_n)
    if fast_arr is None or slow_arr is None:
        return []

    trades = []
    in_trade = False
    entry_price = stop = risk = bars_held = 0

    for i in range(warmup, len(closes)):
        if in_trade:
            bars_held += 1
            # Exit conditions
            if highs[i] >= stop:
                r = (entry_price - stop) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
                continue
            # Reverse cross: fast crosses back above slow → exit
            if (not np.isnan(fast_arr[i]) and not np.isnan(slow_arr[i]) and
                    fast_arr[i] > slow_arr[i]):
                r = (entry_price - opens[i]) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
                continue
            if bars_held >= MAX_HOLD:
                r = (entry_price - opens[i]) / risk if risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
            continue

        # Entry conditions
        if i < 1: continue
        if (np.isnan(atrs[i]) or np.isnan(ema200[i]) or
                np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]) or
                np.isnan(fast_arr[i-1]) or np.isnan(slow_arr[i-1])):
            continue
        # Bear regime filter
        if closes[i] >= ema200[i]:
            continue
        # Cross: fast was above slow at i-1, now below at i
        cross_down = fast_arr[i-1] >= slow_arr[i-1] and fast_arr[i] < slow_arr[i]
        if not cross_down:
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

def stability_zone(df_tf, fast_n, slow_n, am):
    # Stability: adjacent fast_n and slow_n values
    f_steps = sorted(FAST_EMA); s_steps = sorted(SLOW_EMA)
    if fast_n not in f_steps or slow_n not in s_steps:
        return 0, 1
    fi = f_steps.index(fast_n); si = s_steps.index(slow_n)
    results = []
    for df_ in [-1, 0, 1]:
        nfi = fi + df_
        if not (0 <= nfi < len(f_steps)): continue
        nf = f_steps[nfi]
        for ds in [-1, 0, 1]:
            nsi = si + ds
            if not (0 <= nsi < len(s_steps)): continue
            ns = s_steps[nsi]
            if df_ == 0 and ds == 0: continue
            if nf >= ns: continue  # invalid pair
            sub = df_tf[(df_tf.fast==nf)&(df_tf.slow==ns)&(df_tf.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_tf[(df_tf.fast==fast_n)&(df_tf.slow==slow_n)&(df_tf.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

def run_tf(tf):
    data_dir = DATA_4H if tf == "4H" else DATA_1D
    suffix   = "_4h" if tf == "4H" else "_1d"
    files    = sorted(data_dir.glob(f"*{suffix}.csv"))
    if not files: return pd.DataFrame()

    combos = list(product(PAIRS, ATR_MULT))
    print(f"  {tf}: {len(files)} symbols, {len(PAIRS)} pairs × {len(ATR_MULT)} ATR = {len(combos)} combos")

    all_rows = []
    for idx, f in enumerate(files, 1):
        sym = f.stem.replace(suffix, "")
        df  = pd.read_csv(f)
        if len(df) < WARMUP + max(SLOW_EMA) + 10:
            continue
        d = precompute_symbol(df)
        for (fast_n, slow_n), am in combos:
            trades = backtest_symbol(d, WARMUP, fast_n, slow_n, am)
            for t in trades:
                ts_dt = pd.Timestamp(int(t["ts"]), unit="ms", tz="UTC")
                all_rows.append({"sym": sym, "fast": fast_n, "slow": slow_n, "am": am,
                                  "year": ts_dt.year, "r": t["r"], "tf": tf})
        if idx % 50 == 0:
            print(f"    [{idx}/{len(files)}]")

    return pd.DataFrame(all_rows)

def analyze_tf(df_tf, tf):
    lines = [f"\n{'='*60}", f"TIMEFRAME: {tf}", "="*60,
             f"  Pairs: {PAIRS}  ATR_mult: {ATR_MULT}",
             f"  Total trades: {len(df_tf)}  Max hold: {MAX_HOLD} bars"]

    best_r = -np.inf; best_combo = None; all_pass_cost = []
    for am in ATR_MULT:
        sub_am = df_tf[df_tf.am == am]
        pass_list = []
        for fast_n, slow_n in PAIRS:
            sub = sub_am[(sub_am.fast==fast_n)&(sub_am.slow==slow_n)]
            if len(sub) < 3: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_tf, fast_n, slow_n, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else "FAIL"
            e = {"fast": fast_n, "slow": slow_n, "am": am, "avg_r": avg_r,
                 "n_trades": len(sub), "zone_pct": pct, "verdict": verdict, "cost_ok": cost_ok}
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS": pass_list.append(e)
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        top = sorted(pass_list, key=lambda x: -x["avg_r"])[:5]
        lines.append(f"\n  ATR*{am}  PASS: {len(pass_list)}")
        for e in top:
            lines.append(f"    [PASS] fast={e['fast']:3d}/slow={e['slow']:3d} "
                         f"zone={e['zone_pct']:.0%} t={e['n_trades']:5d} avg_r={e['avg_r']:+.4f} "
                         f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    if best_combo:
        bc = best_combo
        sub = df_tf[(df_tf.fast==bc["fast"])&(df_tf.slow==bc["slow"])&(df_tf.am==bc["am"])]
        lines.append(f"\n  Year-by-year (best: fast={bc['fast']} slow={bc['slow']} ATR*{bc['am']}):")
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            lines.append(f"    {yr}: t={len(y):4d}  total={y['r'].sum():+7.2f}R  avg={y['r'].mean():+.4f}R")

    verdict = "BREAKTHROUGH" if all_pass_cost else "HALT"
    lines.append(f"\n  VERDICT [{tf}]: {verdict} — best avg_r={best_r:+.4f}R  pass_§4.2={len(all_pass_cost)}")
    return "\n".join(lines), best_r, len(all_pass_cost)

if __name__ == "__main__":
    print("="*60)
    print("Batch 2C — MA Crossover Short (T1)")
    print(f"Pairs: {PAIRS}  ATR: {ATR_MULT}  Max hold: {MAX_HOLD} bars")
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

    header = "="*60 + "\nBatch 2C — MA Crossover Short T1\n" + f"§4.2 floor: {COST_FLOOR}R\n"
    report = header + "\nSUMMARY:\n" + "\n".join(summary) + "\n" + "\n".join(sections)
    print(report)
    with open(OUT_DIR / "phase_t1_macrossshort_report.txt", "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_DIR}/phase_t1_macrossshort_report.txt")
