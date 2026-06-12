"""
TEST 4 PROXY -- Volume Spike Short (OI proxy) + Funding Gate.
Real OI history is unavailable (Binance API: ~30 days only). Hypothesis:
high volume = new positions being built = OI spike proxy.

Entry: volume > 3x average of prior 20 bars
       AND close breaks below N-bar low (directional confirmation)
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)

If this passes 4.2 AND 2022 positive: flag as candidate for re-run with real
Coinglass OI data (validates concept before buying the subscription).
Output: data/research_volspike_proxy_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_volspike_proxy_t1")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["4h", "8h", "1d"]
LOOKBACK_N = [5, 10, 15]
HOLD_BARS  = [10, 15, 20, 30]
ATR_MULT   = [2.0, 3.0]
VOL_MULT   = 3.0
VOL_AVG_N  = 20

print("TEST 4 PROXY: Volume Spike Short (OI proxy) + funding gate", flush=True)
all_trades = []
grid_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe {tf.upper()} ===", flush=True)
    uni = C.load_universe(tf, log_every=50)
    fund = {s: C.load_funding(s) for s in uni}
    vols = {}
    for s in uni:
        df = pd.read_csv(C.DATA_ROOT / f"ohlcv_{tf}" / f"{s}_{tf}.csv")
        vols[s] = df["volume"].values.astype(float)
    for n_lb in LOOKBACK_N:
        sig = {}
        for s, d in uni.items():
            v = vols[s]
            v_avg = pd.Series(v).rolling(VOL_AVG_N).mean().shift(1).values
            rmin = C.rolling_min_prior(d["low"], n_lb)
            mask = (d["bear"] & (v > VOL_MULT * v_avg) & (d["close"] < rmin))
            mask &= ~np.isnan(v_avg) & ~np.isnan(rmin)
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

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL", flush=True)
    raise SystemExit(1)

best = C.write_report(OUT, "Volume Spike Short (OI PROXY) + Funding Gate (4H/8H/1D)",
                      df_grid, ["lookback_n", "hold_bars", "atr_mult"], ["lookback_n"],
                      reference_line="PROXY for Test 4 (OI Spike) -- real OI history unavailable. "
                                     "vol > 3x avg(20) as OI proxy")

with open(OUT / "t1_summary.txt", "a", encoding="ascii", errors="replace") as f:
    note = ("\n\nNOTE: This is a PROXY for OI Spike Short. If 4.2 + 2022 pass here, "
            "flag for re-run with real Coinglass OI data (~$50/month subscription) "
            "before any T2. Proxy validation de-risks the data purchase.\n")
    f.write(note)
    print(note, flush=True)
