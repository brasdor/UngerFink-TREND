"""
TEST 5 PROXY -- RSI Overbought Short (L/S ratio proxy) + Funding Gate.
Real long/short ratio history is unavailable (Binance API: ~30 days only).
Hypothesis: RSI(14) overbought = retail likely long-biased = L/S proxy.

Entry: RSI(14) > overbought_threshold
       AND close < EMA200 AND funding_rate >= 0.01%/8h
Exit:  fixed time exit after hold_bars (stop = ATR x atr_mult above entry)

NOTE: RSI overbought short was tested before WITHOUT the funding gate and
failed (avg_r +0.055R). This tests whether the gate unlocks it (same
pattern as MA Cross +0.123R -> +0.255R).
Output: data/research_rsioverbought_proxy_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_rsioverbought_proxy_t1")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES  = ["4h", "8h", "1d"]
OB_THRESH   = [65, 70, 75, 80]
HOLD_BARS   = [10, 15, 20, 30]
ATR_MULT    = [2.0, 3.0]
RSI_N       = 14


def compute_rsi(closes, n=RSI_N):
    """Wilder RSI, SMA-seeded (consistent with engine ATR seeding)."""
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    m = len(closes)
    rsi = np.full(m, np.nan)
    if m < n + 1:
        return rsi
    avg_g = gain[1:n+1].mean()
    avg_l = loss[1:n+1].mean()
    alpha = 1.0 / n
    for i in range(n + 1, m):
        avg_g = avg_g * (1 - alpha) + gain[i] * alpha
        avg_l = avg_l * (1 - alpha) + loss[i] * alpha
        rsi[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return rsi


print("TEST 5 PROXY: RSI Overbought Short (L/S proxy) + funding gate", flush=True)
all_trades = []
grid_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== Timeframe {tf.upper()} ===", flush=True)
    uni = C.load_universe(tf, log_every=50)
    fund = {s: C.load_funding(s) for s in uni}
    rsis = {s: compute_rsi(d["close"]) for s, d in uni.items()}
    for ob in OB_THRESH:
        sig = {}
        for s, d in uni.items():
            mask = d["bear"] & (rsis[s] > ob)
            mask &= ~np.isnan(rsis[s])
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
                grid_rows.append({"tf": tf, "ob_thresh": ob, "hold_bars": hb,
                                  "atr_mult": am, **stats})
                dft = dft.drop(columns=["dt"])
                dft["tf"] = tf; dft["ob_thresh"] = ob
                dft["hold_bars"] = hb; dft["atr_mult"] = am
                all_trades.append(dft)
        print(f"  [{tf}] ob_thresh={ob} done", flush=True)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
df_grid = pd.DataFrame(grid_rows)
if len(df_grid) == 0:
    print("NO TRADES AT ALL (note: RSI>65 while below EMA200 is rare -- "
          "check signal counts)", flush=True)
    raise SystemExit(1)

C.write_report(OUT, "RSI Overbought Short (L/S PROXY) + Funding Gate (4H/8H/1D)",
               df_grid, ["ob_thresh", "hold_bars", "atr_mult"], ["ob_thresh"],
               reference_line="PROXY for Test 5 (L/S Ratio) -- real L/S history unavailable. "
                              "Prior no-gate RSI OB short: +0.055R FAIL")

with open(OUT / "t1_summary.txt", "a", encoding="ascii", errors="replace") as f:
    note = ("\n\nNOTE: This is a PROXY for L/S Ratio Extreme Short. If 4.2 + 2022 "
            "pass here, flag for re-run with real Coinglass L/S data (~$50/month) "
            "before any T2. Proxy validation de-risks the data purchase.\n")
    f.write(note)
    print(note, flush=True)
