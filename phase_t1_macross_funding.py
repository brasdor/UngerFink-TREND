"""
TEST 3 -- MA Crossover Short + Funding Gate (all timeframes).
Entry: fast EMA crosses below slow EMA
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)
NOTE: atr_mult [2.0, 3.0] added as the stop/risk unit (R requires a stop;
      mirrors all other sweep tests).
Previous best without gate: +0.123R.
Output: data/research_macross_funding_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_macross_funding_t1")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["2h", "4h", "6h", "8h", "1d"]
FAST_EMA   = [10, 20, 30]
SLOW_EMA   = [50, 100, 200]
HOLD_BARS  = [10, 20, 30]
ATR_MULT   = [2.0, 3.0]

print("TEST 3: MA Crossover Short + funding gate", flush=True)
all_trades = []
grid_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe {tf.upper()} ===", flush=True)
    uni = C.load_universe(tf)
    if not uni:
        print(f"  [{tf}] no data -- skipping", flush=True)
        continue
    fund = {s: C.load_funding(s) for s in uni}
    # precompute all needed EMAs once per symbol
    periods = sorted(set(FAST_EMA + SLOW_EMA))
    emas = {s: {p: C.compute_ema(d["close"], p) for p in periods} for s, d in uni.items()}
    for fast in FAST_EMA:
        for slow in SLOW_EMA:
            sig = {}
            for s, d in uni.items():
                f_, s_ = emas[s][fast], emas[s][slow]
                f_prev = np.roll(f_, 1); s_prev = np.roll(s_, 1)
                mask = d["bear"] & (f_ < s_) & (f_prev >= s_prev)
                mask &= ~np.isnan(f_) & ~np.isnan(s_) & ~np.isnan(f_prev) & ~np.isnan(s_prev)
                mask[0] = False
                idx = np.where(mask)[0]
                gated = [i for i in idx
                         if (fr := C.get_rate_at(fund[s], int(d["ts"][i]))) is not None
                         and fr >= C.FUNDING_THRESHOLD]
                sig[s] = np.array(gated, dtype=int)
            for hb in HOLD_BARS:
                for am in ATR_MULT:
                    trades = []
                    for s, d in uni.items():
                        trades += C.simulate(s, d, sig[s], hb, am, fund[s])
                    if not trades:
                        continue
                    dft = C.add_time_cols(pd.DataFrame(trades))
                    stats = C.combo_stats(dft)
                    grid_rows.append({"tf": tf, "fast_ema": fast, "slow_ema": slow,
                                      "hold_bars": hb, "atr_mult": am, **stats})
                    dft = dft.drop(columns=["dt"])
                    dft["tf"] = tf; dft["fast_ema"] = fast; dft["slow_ema"] = slow
                    dft["hold_bars"] = hb; dft["atr_mult"] = am
                    all_trades.append(dft)
        print(f"  [{tf}] fast_ema={fast} done", flush=True)
    dfg = pd.DataFrame(grid_rows)
    sub = dfg[(dfg["tf"] == tf) & (dfg["n"] >= C.MIN_TRADES_COMBO)]
    if len(sub):
        b = sub.iloc[sub["avg_r"].values.argmax()]
        print(f"  [{tf}] PARTIAL BEST: fast={b['fast_ema']} slow={b['slow_ema']} "
              f"hb={b['hold_bars']} am={b['atr_mult']} n={int(b['n'])} "
              f"avg_r={b['avg_r']:+.4f}R", flush=True)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL -- check data", flush=True)
    raise SystemExit(1)

C.write_report(OUT, "MA Crossover Short + Funding Gate (2H/4H/6H/8H/1D)",
               df_grid, ["fast_ema", "slow_ema", "hold_bars", "atr_mult"],
               ["fast_ema", "slow_ema"],
               reference_line="No-gate baseline: +0.123R | System 7 reference +0.304R")
