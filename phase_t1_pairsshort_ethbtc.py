"""
Batch 2E — Pairs Short ETH/BTC (1D)
Market-neutral: short ETH, long BTC simultaneously when ETH/BTC ratio
crosses below its N-day MA AND ratio has been declining for 3 consecutive bars.
P&L = return from ETH/BTC ratio declining (ETH underperforms BTC).
Risk = ATR of ratio × mult (volatility of spread).
§4.2 floor: 0.15R (market neutral, lower threshold)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_1D    = Path("data/futures_universe/ohlcv_1d")
OUT_DIR    = Path("data/research_pairsshort_ethbtc_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MA_N       = [10, 20, 30, 50]
HOLD_BARS  = [5, 10, 15, 20]
ATR_MULT   = [1.5, 2.0, 3.0]
ATR_PERIOD = 14
WARMUP     = 100
COST_FLOOR = 0.15   # market neutral — lower threshold
CONSEC_DEC = 3      # require 3 consecutive declining ratio bars

def compute_atr_series(series):
    """ATR for a price series (here, ratio series)."""
    s = series
    n = len(s); atr = np.full(n, np.nan)
    if n < ATR_PERIOD + 1: return atr
    # True range for ratio: just use |diff| as proxy (no high/low)
    diff = np.abs(np.diff(s, prepend=s[0]))
    atr[ATR_PERIOD - 1] = diff[:ATR_PERIOD].mean()
    a = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, n):
        atr[i] = atr[i-1] * (1 - a) + diff[i] * a
    return atr

def compute_ema(series, n):
    m = len(series); ema = np.full(m, np.nan)
    if m < n: return ema
    ema[n - 1] = series[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, m):
        ema[i] = series[i] * alpha + ema[i-1] * (1 - alpha)
    return ema

def backtest_pairs(ratio, timestamps, warmup, ma_n, hb, am):
    """
    ratio: ETH_close / BTC_close
    Entry: ratio crosses below MA AND declining 3 consecutive bars
    Exit: hold_bars OR ratio crosses back above MA
    P&L: (entry_ratio - exit_ratio) / (atr × am) in R units
    """
    n   = len(ratio)
    ma  = compute_ema(ratio, ma_n)
    atr = compute_atr_series(ratio)

    trades = []
    in_trade = False
    entry_ratio = stop_ratio = atr_risk = bars_held = 0

    for i in range(max(warmup, ma_n + CONSEC_DEC + 1), n):
        if in_trade:
            bars_held += 1
            # Stop: ratio rises above stop (trade loses)
            if ratio[i] >= stop_ratio:
                r = (entry_ratio - stop_ratio) / atr_risk if atr_risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
                continue
            # Exit: ratio crosses back above MA
            if not np.isnan(ma[i]) and ratio[i] > ma[i]:
                r = (entry_ratio - ratio[i]) / atr_risk if atr_risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
                continue
            if bars_held >= hb:
                r = (entry_ratio - ratio[i]) / atr_risk if atr_risk > 0 else 0.0
                trades.append({"ts": timestamps[i], "r": r})
                in_trade = False
            continue

        if np.isnan(ma[i]) or np.isnan(atr[i]): continue
        if atr[i] <= 0: continue
        # Cross below MA
        if i < 1: continue
        if np.isnan(ma[i-1]): continue
        cross_below = ratio[i-1] >= ma[i-1] and ratio[i] < ma[i]
        if not cross_below: continue
        # Consecutive declining check
        if i < CONSEC_DEC: continue
        declining = all(ratio[i - k] < ratio[i - k - 1] for k in range(CONSEC_DEC))
        if not declining: continue
        if i + 1 >= n: continue
        entry_ratio = ratio[i + 1] if i + 1 < n else ratio[i]
        atr_risk    = atr[i] * am
        stop_ratio  = entry_ratio + atr_risk   # stop above entry (ratio goes up = loss)
        in_trade = True
        bars_held = 0

    return trades

def stability_zone(df_all, ma_n, hb, am):
    ma_steps = sorted(MA_N); hb_steps = sorted(HOLD_BARS)
    mi = ma_steps.index(ma_n); hi = hb_steps.index(hb)
    results = []
    for dm in [-1, 0, 1]:
        nmi = mi + dm
        if not (0 <= nmi < len(ma_steps)): continue
        nma = ma_steps[nmi]
        for dh in [-1, 0, 1]:
            nhi = hi + dh
            if not (0 <= nhi < len(hb_steps)): continue
            nhb = hb_steps[nhi]
            if dm == 0 and dh == 0: continue
            sub = df_all[(df_all.ma_n==nma)&(df_all.hb==nhb)&(df_all.am==am)]
            results.append(sub["r"].mean() > 0 if len(sub) else False)
    center = df_all[(df_all.ma_n==ma_n)&(df_all.hb==hb)&(df_all.am==am)]
    c_ok = center["r"].mean() > 0 if len(center) else False
    return sum([c_ok] + results), len(results) + 1

if __name__ == "__main__":
    print("="*60)
    print("Batch 2E — Pairs Short ETH/BTC (1D)")
    print(f"MA_N={MA_N}  hold_bars={HOLD_BARS}  atr_mult={ATR_MULT}")
    print(f"§4.2 floor: {COST_FLOOR}R  |  Market neutral")
    print("="*60)

    # Load ETH and BTC data, align by timestamp
    eth_path = DATA_1D / "ETHUSDT_1d.csv"
    btc_path = DATA_1D / "BTCUSDT_1d.csv"
    if not eth_path.exists() or not btc_path.exists():
        print("ERROR: ETHUSDT_1d.csv or BTCUSDT_1d.csv not found")
        exit(1)

    eth_df = pd.read_csv(eth_path).set_index("timestamp")
    btc_df = pd.read_csv(btc_path).set_index("timestamp")
    common_ts = eth_df.index.intersection(btc_df.index)
    eth_cl = eth_df.loc[common_ts, "close"].values.astype(float)
    btc_cl = btc_df.loc[common_ts, "close"].values.astype(float)
    timestamps = np.array(common_ts, dtype=np.int64)

    # Compute ETH/BTC ratio
    ratio = eth_cl / btc_cl
    print(f"Aligned bars: {len(ratio)}  ({pd.Timestamp(timestamps[0], unit='ms', tz='UTC').date()} to {pd.Timestamp(timestamps[-1], unit='ms', tz='UTC').date()})")

    combos = list(product(MA_N, HOLD_BARS, ATR_MULT))
    print(f"Combos: {len(combos)}\n")

    all_rows = []
    for ma_n, hb, am in combos:
        trades = backtest_pairs(ratio, timestamps, WARMUP, ma_n, hb, am)
        for t in trades:
            ts_dt = pd.Timestamp(int(t["ts"]), unit="ms", tz="UTC")
            all_rows.append({"ma_n": ma_n, "hb": hb, "am": am, "year": ts_dt.year, "r": t["r"]})

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / "grid_trades_1d.csv", index=False)
    print(f"Total trades: {len(df_all)}\n")

    # Analysis
    lines = ["Batch 2E — Pairs Short ETH/BTC (1D)",
             f"§4.2 floor: {COST_FLOOR}R  |  Market neutral  |  consec_dec={CONSEC_DEC} bars",
             f"Total trades: {len(df_all)}", ""]

    best_r = -np.inf; best_combo = None; all_pass_cost = []
    for am in ATR_MULT:
        sub_am = df_all[df_all.am == am]
        pass_list = []
        for ma_n, hb in product(MA_N, HOLD_BARS):
            sub = sub_am[(sub_am.ma_n==ma_n)&(sub_am.hb==hb)]
            if len(sub) < 3: continue
            avg_r = sub["r"].mean()
            n_p, n_t = stability_zone(df_all, ma_n, hb, am)
            pct = n_p / n_t if n_t else 0
            cost_ok = avg_r > COST_FLOOR
            verdict = "PASS" if pct >= 0.67 else "FAIL"
            e = {"ma_n": ma_n, "hb": hb, "am": am, "avg_r": avg_r, "n_trades": len(sub),
                 "zone_pct": pct, "verdict": verdict, "cost_ok": cost_ok}
            if avg_r > best_r: best_r = avg_r; best_combo = e
            if verdict == "PASS": pass_list.append(e)
            if verdict == "PASS" and cost_ok: all_pass_cost.append(e)

        top = sorted(pass_list, key=lambda x: -x["avg_r"])[:5]
        lines.append(f"ATR*{am}  PASS: {len(pass_list)}")
        for e in top:
            lines.append(f"  [PASS] ma_n={e['ma_n']:2d} hb={e['hb']:2d} zone={e['zone_pct']:.0%} "
                         f"t={e['n_trades']:4d} avg_r={e['avg_r']:+.4f} "
                         f"§4.2={'PASS' if e['cost_ok'] else 'FAIL'}")

    # Year-by-year
    if best_combo:
        bc = best_combo
        sub = df_all[(df_all.ma_n==bc["ma_n"])&(df_all.hb==bc["hb"])&(df_all.am==bc["am"])]
        lines.append(f"\nYear-by-year (best: ma_n={bc['ma_n']} hb={bc['hb']} ATR*{bc['am']}):")
        for yr in sorted(sub.year.unique()):
            y = sub[sub.year == yr]
            lines.append(f"  {yr}: t={len(y):4d}  total={y['r'].sum():+7.2f}R  avg={y['r'].mean():+.4f}R")

    # Ratio stats
    lines.append(f"\nETH/BTC ratio stats: mean={ratio.mean():.4f}  std={ratio.std():.4f}")
    lines.append(f"Bear years check: 2022 bear market should show ETH underperformance")

    verdict = "BREAKTHROUGH" if all_pass_cost else "HALT"
    lines.append(f"\nBest avg_r: {best_r:+.4f}R  §4.2 passes: {len(all_pass_cost)}")
    lines.append(f"VERDICT: {verdict}")

    report = "\n".join(lines)
    print(report)
    with open(OUT_DIR / "phase_t1_pairsshort_ethbtc_report.txt", "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_DIR}/phase_t1_pairsshort_ethbtc_report.txt")
    print(f"SUMMARY_2E: best_avg_r={best_r:+.4f}R  pass_§4.2={len(all_pass_cost)}  verdict={verdict}")
