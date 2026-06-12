"""
TEST 1 -- VolContraction Short 8H + Funding Gate (System 7 logic on 8H bars).
Entry: ATR(14) pct < atr_percentile for >= compression_bars consecutive 8H bars
       AND close breaks below compression-period low
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)
Reference: System 7 (4H) avg_r=+0.304R. No-gate baseline: +0.106R.
Output: data/research_volcontraction_8h_funding_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_volcontraction_8h_funding_t1")
OUT.mkdir(parents=True, exist_ok=True)

TF = "8h"
COMPRESSION_BARS = [5, 10, 15]
ATR_PERCENTILE   = [15, 20, 25]
HOLD_BARS        = [10, 15, 20]
ATR_MULT         = [2.0, 3.0]

print("TEST 1: VolContraction 8H + funding gate", flush=True)
print("Loading 8H universe (with ATR percentile ranks -- slow step)...", flush=True)
uni = C.load_universe(TF, with_atr_pct=True)
fund = {s: C.load_funding(s) for s in uni}

def consec_below(pct, ap):
    n = len(pct)
    c = np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isnan(pct[i]):
            c[i] = 0
        elif pct[i] < ap:
            c[i] = c[i-1] + 1
        else:
            c[i] = 0
    return c

all_trades = []
grid_rows = []
n_combos = len(COMPRESSION_BARS) * len(ATR_PERCENTILE) * len(HOLD_BARS) * len(ATR_MULT)
done = 0
for ap in ATR_PERCENTILE:
    consec = {s: consec_below(d["atr_pct"], ap) for s, d in uni.items()}
    for cb in COMPRESSION_BARS:
        # signal: consec at i-1 >= cb AND close < min(low of prior cb bars) AND bear
        sig = {}
        for s, d in uni.items():
            cprev = np.roll(consec[s], 1); cprev[0] = 0
            rmin = C.rolling_min_prior(d["low"], cb)
            mask = d["bear"] & (cprev >= cb) & (d["close"] < rmin)
            mask &= ~np.isnan(rmin)
            idx = np.where(mask)[0]
            # pre-apply funding gate (identical outcome, faster across combos)
            gated = [i for i in idx
                     if (fr := C.get_rate_at(fund[s], int(d["ts"][i]))) is not None
                     and fr >= C.FUNDING_THRESHOLD]
            sig[s] = np.array(gated, dtype=int)
        for hb in HOLD_BARS:
            for am in ATR_MULT:
                trades = []
                for s, d in uni.items():
                    trades += C.simulate(s, d, sig[s], hb, am, fund[s])
                done += 1
                if not trades:
                    print(f"  [{done}/{n_combos}] cb={cb} ap={ap} hb={hb} am={am}: 0 trades", flush=True)
                    continue
                dft = C.add_time_cols(pd.DataFrame(trades))
                stats = C.combo_stats(dft)
                grid_rows.append({"tf": TF, "cb": cb, "ap": ap, "hb": hb, "am": am, **stats})
                print(f"  [{done}/{n_combos}] cb={cb} ap={ap} hb={hb} am={am}: "
                      f"n={stats['n']} avg_r={stats['avg_r']:+.4f}R", flush=True)
                dft = dft.drop(columns=["dt"])
                dft["tf"] = TF; dft["cb"] = cb; dft["ap"] = ap; dft["hb"] = hb; dft["am"] = am
                all_trades.append(dft)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL -- check data", flush=True)
    raise SystemExit(1)

C.write_report(OUT, "VolContraction 8H + Funding Gate",
               df_grid, ["cb", "ap", "hb", "am"], ["cb"],
               reference_line="Reference: System 7 (4H+gate) +0.304R | no-gate baseline +0.106R")
