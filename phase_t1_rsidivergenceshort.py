"""
Batch 2D — RSI Bearish Divergence Short (T1)
Entry: price makes higher close over N bars BUT RSI(14) makes lower high
       AND close < EMA200 (bear regime filter)
Proxy: close[i] >= max(close[i-N:i]) AND RSI[i] < RSI[argmax(close[i-N:i])]
Exit: fixed time exit after hold_bars
Stop: ATR × mult ABOVE entry
Timeframe: 1D only
§4.2 floor: 0.15R
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_1D    = Path("data/futures_universe/ohlcv_1d")
OUT_DIR    = Path("data/research_rsidivergenceshort_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_N  = [5, 10, 15, 20]
HOLD_BARS   = [5, 10, 15, 20]
ATR_MULT    = [2.0, 3.0]
ATR_PERIOD  = 14
RSI_N       = 14
EMA_PERIOD  = 200
WARMUP      = 250
COST_FLOOR  = 0.15

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
    n = len(closes); ema = np.full(n, np.nan)
    if n < EMA_PERIOD: return ema
    ema[EMA_PERIOD - 1] = closes[:EMA_PERIOD].mean()
    alpha = 2.0 / (EMA_PERIOD + 1)
    for i in range(EMA_PERIOD, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    return ema

def compute_rsi(closes, n=RSI_N):
    m = len(closes); rsi = np.full(m, np.nan)
    if m < n + 1: return rsi
    diff = np.diff(closes, prepend=closes[0])
    gains = np.maximum(diff, 0)
    losses = np.abs(np.minimum(diff, 0))
    avg_gain = np.full(m, np.nan); avg_loss = np.full(m, np.nan)
    avg_gain[n] = gains[1:n+1].mean()
    avg_loss[n] = losses[1:n+1].mean()
    for i in range(n + 1, m):
        avg_gain[i] = (avg_gain[i-1] * (n - 1) + gains[i]) / n
        avg_loss[i] = (avg_loss[i-1] * (n - 1) + losses[i]) / n
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi[:n] = np.nan
    return rsi

def precompute_symbol(df):
    closes = df["close"].values.astype(float)
    atr    = compute_atr(df)
    ema200 = compute_ema200(closes)
    rsi    = compute_rsi(closes)
    return {
        "timestamps": df["timestamp"].values.astype(np.int64),
        "opens":  df["open"].values.astype(float),
        "closes": closes,
        "highs":  df["high"].values.astype(float),
        "lows":   df["low"].values.astype(float),
        "atrs":   atr,
        "ema200": ema200,
        "rsi":    rsi,
    }

def backtest_symbol(d, warmup, lb, hb, am):
    """
    lb = lookback period for divergence detection
    hb = hold bars
    am = ATR stop multiplier
    """
    timestamps = d["timestamps"]
    opens  = d["opens"];  closes = d["closes"]
    highs  = d["highs"]
    atrs   = d["atrs"];   ema200 = d["ema200"]
    rsi    = d["rsi"]

    trades = []
    in_trade = False
    entry_price = stop = risk = bars_held = 0

    for i in range(max(warmup, lb + 1), len(closes)):
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

        if np.isnan(atrs[i]) or np.isnan(ema200[i]) or np.isnan(rsi[i]):
            continue
        # Bear regime filter
        if closes[i] >= ema200[i]:
            continue
        # Look back lb bars
        window_cl  = closes[i - lb: i]  # lb bars before current
        window_rsi = rsi[i - lb: i]
        if np.any(np.isnan(window_rsi)):
            continue
        # Find peak in lookback window
        peak_idx_local = int(np.argmax(window_cl))
        peak_close = window_cl[peak_idx_local]
        peak_rsi   = window_rsi[peak_idx_local]
        # Bearish divergence: price makes new high (close[i] >= peak in window)
        # but RSI is lower than at the peak bar
        if closes[i] < peak_close:
            continue  # price not at new high relative to window
        if rsi[i] >= peak_rsi:
            continue  # RSI not lower → no divergence
        # Confirmed bearish divergence + EMA200 bear
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

def stability_zone(df_tf, lb, hb, am):
    lb_steps = sorted(LOOKBACK_N); hb_steps = sorted(HOLD_BARS)
    li = lb_steps.index(lb); hi = hb_steps.index(hb)
    results = []
    for dlb in [-1, 0, 1]:
        nli = li + dlb
        if not (0 <= nli < len(lb_steps)): continue
        nlb = lb_steps[nli]
        for dhb in [-1, 0, 1]:
            nhi = hi + dhb
            if not (0 <= nhi < len(hb_steps)): continue
            nhb = hb_steps[nhi]
            if dlb == 0 and dhb == 0: continue
            sub = df_tf[(df_tf.lb==nlb)&(df_tf.hb==nhb)&(df_tf.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_tf[(df_tf.lb==lb)&(df_tf.hb==hb)&(df_tf.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

if __name__ == "__main__":
    print("="*60)
    print("Batch 2D — RSI Bearish Divergence Short (1D)")
    print(f"lookback_n={LOOKBACK_N}  hold_bars={HOLD_BARS}  atr_mult={ATR_MULT}")
    print(f"§4.2 floor: {COST_FLOOR}R")
    print("="*60)

    files = sorted(DATA_1D.glob("*_1d.csv"))
    combos = list(product(LOOKBACK_N, HOLD_BARS, ATR_MULT))
    print(f"1D: {len(files)} symbols, {len(combos)} combos")

    all_rows = []
    for idx, f in enumerate(files, 1):
        sym = f.stem.replace("_1d", "")
        df  = pd.read_csv(f)
        if len(df) < WARMUP + max(LOOKBACK_N) + 10:
            continue
        d = precompute_symbol(df)
        for lb, hb, am in combos:
            trades = backtest_symbol(d, WARMUP, lb, hb, am)
            for t in trades:
                ts_dt = pd.Timestamp(int(t["ts"]), unit="ms", tz="UTC")
                all_rows.append({"sym": sym, "lb": lb, "hb": hb, "am": am,
                                  "year": ts_dt.year, "r": t["r"]})
        if idx % 50 == 0:
            print(f"  [{idx}/{len(files)}]")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "grid_trades_1d.csv", index=False)
    print(f"\nTotal trades: {len(df_all)}")

    # Analysis
    lines = ["Batch 2D — RSI Bearish Divergence Short (1D)",
             f"§4.2 floor: {COST_FLOOR}R  |  EMA200 bear filter",
             f"Total trades: {len(df_all)}", ""]

    best_r = -np.inf; best_combo = None; all_pass_cost = []
    for am in ATR_MULT:
        sub_am = df_all[df_all.am == am]
        pass_list = []
        for lb, hb in product(LOOKBACK_N, HOLD_BARS):
            sub = sub_am[(sub_am.lb==lb)&(sub_am.hb==hb)]
            if len(sub) < 3: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_all, lb, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else "FAIL"
            e = {"lb": lb, "hb": hb, "am": am, "avg_r": avg_r, "n_trades": len(sub),
                 "zone_pct": pct, "verdict": verdict, "cost_ok": cost_ok}
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS": pass_list.append(e)
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        top = sorted(pass_list, key=lambda x: -x["avg_r"])[:5]
        lines.append(f"ATR*{am}  PASS: {len(pass_list)}")
        for e in top:
            lines.append(f"  [PASS] lb={e['lb']:2d} hb={e['hb']:2d} zone={e['zone_pct']:.0%} "
                         f"t={e['n_trades']:4d} avg_r={e['avg_r']:+.4f} "
                         f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Year-by-year best combo
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.lb==bc["lb"])&(df_all.hb==bc["hb"])&(df_all.am==bc["am"])]
        lines.append(f"\nYear-by-year (best: lb={bc['lb']} hb={bc['hb']} ATR*{bc['am']}):")
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            lines.append(f"  {yr}: t={len(y):4d}  total={y['r'].sum():+7.2f}R  avg={y['r'].mean():+.4f}R")

    win_rate_check = None
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.lb==bc["lb"])&(df_all.hb==bc["hb"])&(df_all.am==bc["am"])]
        wr = (sub["r"] > 0).mean() * 100
        win_rate_check = f"Win rate: {wr:.1f}% (MR target: 50-70%)"

    verdict = "BREAKTHROUGH" if all_pass_cost else "HALT"
    lines.append(f"\nBest avg_r: {best_r:+.4f}R")
    if win_rate_check: lines.append(win_rate_check)
    lines.append(f"VERDICT: {verdict}")

    report = "\n".join(lines)
    print(report)
    with open(OUT_DIR / "phase_t1_rsidivergenceshort_report.txt", "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_DIR}/phase_t1_rsidivergenceshort_report.txt")
    print(f"SUMMARY_2D: best_avg_r={best_r:+.4f}R  pass_§4.2={len(all_pass_cost)}  verdict={verdict}")
