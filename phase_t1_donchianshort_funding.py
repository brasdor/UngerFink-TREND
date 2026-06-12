"""
TEST 6 -- Donchian Short + Funding Gate (all timeframes). FINAL comprehensive test.
Entry: close below Donchian N-period low (close < min of prior N lows)
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)
Previous best without gate: +0.096R (4 failed attempts).
If this fails WITH the funding gate, Donchian Short is PERMANENTLY closed.
Output: data/research_donchianshort_funding_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_donchianshort_funding_t1")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["2h", "4h", "6h", "8h", "1d"]
DONCHIAN_N = [10, 15, 20, 25, 30]
HOLD_BARS  = [10, 15, 20, 25, 30]
ATR_MULT   = [2.0, 3.0]

print("TEST 6: Donchian Short + funding gate (final attempt)", flush=True)
all_trades = []
grid_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe {tf.upper()} ===", flush=True)
    uni = C.load_universe(tf)
    if not uni:
        print(f"  [{tf}] no data -- skipping", flush=True)
        continue
    fund = {s: C.load_funding(s) for s in uni}
    for dn in DONCHIAN_N:
        sig = {}
        for s, d in uni.items():
            rmin = C.rolling_min_prior(d["low"], dn)
            mask = d["bear"] & (d["close"] < rmin)
            mask &= ~np.isnan(rmin)
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
                grid_rows.append({"tf": tf, "donchian_n": dn, "hold_bars": hb,
                                  "atr_mult": am, **stats})
                dft = dft.drop(columns=["dt"])
                dft["tf"] = tf; dft["donchian_n"] = dn
                dft["hold_bars"] = hb; dft["atr_mult"] = am
                all_trades.append(dft)
        print(f"  [{tf}] donchian_n={dn} done", flush=True)
    dfg = pd.DataFrame(grid_rows)
    sub = dfg[(dfg["tf"] == tf) & (dfg["n"] >= C.MIN_TRADES_COMBO)]
    if len(sub):
        b = sub.iloc[sub["avg_r"].values.argmax()]
        print(f"  [{tf}] PARTIAL BEST: dn={b['donchian_n']} hb={b['hold_bars']} "
              f"am={b['atr_mult']} n={int(b['n'])} avg_r={b['avg_r']:+.4f}R", flush=True)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL -- check data", flush=True)
    raise SystemExit(1)

C.write_report(OUT, "Donchian Short + Funding Gate (2H/4H/6H/8H/1D) -- FINAL ATTEMPT",
               df_grid, ["donchian_n", "hold_bars", "atr_mult"], ["donchian_n"],
               reference_line="No-gate baseline: +0.096R (4 failed attempts) | System 7 reference +0.304R")
