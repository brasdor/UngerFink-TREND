"""
TEST 2 -- New High Failure Short + Funding Gate (all timeframes).
Entry: high makes new N-bar high (high > max of prior N highs)
       AND bearish close (close < open)
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)
Previous best without gate: +0.108R.
Output: data/research_nhf_funding_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_nhf_funding_t1")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["2h", "4h", "6h", "8h", "1d"]
LOOKBACK_N = [5, 10, 15, 20]
HOLD_BARS  = [5, 10, 15, 20]
ATR_MULT   = [2.0, 3.0]

print("TEST 2: New High Failure + funding gate", flush=True)
all_trades = []
grid_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe {tf.upper()} ===", flush=True)
    uni = C.load_universe(tf)
    if not uni:
        print(f"  [{tf}] no data -- skipping", flush=True)
        continue
    fund = {s: C.load_funding(s) for s in uni}
    for n_lb in LOOKBACK_N:
        sig = {}
        for s, d in uni.items():
            rmax = C.rolling_max_prior(d["high"], n_lb)
            mask = d["bear"] & (d["high"] > rmax) & (d["close"] < d["open"])
            mask &= ~np.isnan(rmax)
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
                grid_rows.append({"tf": tf, "lookback_n": n_lb, "hold_bars": hb,
                                  "atr_mult": am, **stats})
                dft = dft.drop(columns=["dt"])
                dft["tf"] = tf; dft["lookback_n"] = n_lb
                dft["hold_bars"] = hb; dft["atr_mult"] = am
                all_trades.append(dft)
        print(f"  [{tf}] lookback_n={n_lb} done", flush=True)
    # partial best after each timeframe
    dfg = pd.DataFrame(grid_rows)
    sub = dfg[(dfg["tf"] == tf) & (dfg["n"] >= C.MIN_TRADES_COMBO)]
    if len(sub):
        b = sub.iloc[sub["avg_r"].values.argmax()]
        print(f"  [{tf}] PARTIAL BEST: lb={b['lookback_n']} hb={b['hold_bars']} "
              f"am={b['atr_mult']} n={int(b['n'])} avg_r={b['avg_r']:+.4f}R", flush=True)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL -- check data", flush=True)
    raise SystemExit(1)

C.write_report(OUT, "New High Failure + Funding Gate (2H/4H/6H/8H/1D)",
               df_grid, ["lookback_n", "hold_bars", "atr_mult"], ["lookback_n"],
               reference_line="No-gate baseline: +0.108R | System 7 reference +0.304R")
