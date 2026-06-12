"""
TEST 7 -- Multi-Signal Confluence Short (4H and 8H).

Signals (bar-level, all on signal bar close):
  A: VolContraction breakdown (System 7 logic; 4H uses System 7 frozen
     params cb=15 ap=20; 8H uses Test 1 best cb/ap from its grid)
  B: New High Failure (Test 2 best lookback_n for that timeframe)
  C: OI spike -- EXCLUDED (Binance API only exposes 30 days of OI history;
     see data/research_oispike_funding_t1/t1_feasibility_report.txt)
  D: funding_rate >= 0.02%/8h (higher-crowding threshold)

Confluence levels (3 available signals): L2 = any 2 of A/B/D, L3 = all 3.
Mandatory base gates on every entry: close < EMA200 AND funding >= 0.01%/8h.
Exit: System 7-inherited -- stop = ATR x 2.0 above entry, time exit after
hold_bars = 30 (4H) / 15 (8H) (~5 calendar days).

Output: data/research_confluence_short_t1/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_confluence_short_t1")
OUT.mkdir(parents=True, exist_ok=True)

D_THRESHOLD = 0.0002
EXIT_CFG = {"4h": {"hb": 30, "am": 2.0}, "8h": {"hb": 15, "am": 2.0}}
VC_CFG_4H = {"cb": 15, "ap": 20}   # System 7 frozen entry params


def best_from_grid(csv_path, tf, param_cols):
    p = Path(csv_path)
    if not p.exists():
        return None
    g = pd.read_csv(p)
    sub = g[(g["tf"] == tf) & (g["n"] >= C.MIN_TRADES_COMBO)]
    if len(sub) == 0:
        sub = g[g["tf"] == tf]
    if len(sub) == 0:
        return None
    b = sub.iloc[sub["avg_r"].values.argmax()]
    return {c: b[c] for c in param_cols}


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


print("TEST 7: Multi-Signal Confluence Short (signals A/B/D; C excluded -- no OI history)", flush=True)
all_trades = []
grid_rows = []
lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

for tf in ["4h", "8h"]:
    pr(f"\n=== Timeframe {tf.upper()} ===")
    # resolve signal configs
    if tf == "4h":
        vc = VC_CFG_4H
    else:
        vc = best_from_grid("data/research_volcontraction_8h_funding_t1/t1_grid_results.csv",
                            tf, ["cb", "ap"])
        if vc is None:
            vc = {"cb": 10, "ap": 20}
            pr(f"  [WARN] Test 1 grid missing -- fallback VC params {vc}")
    nhf = best_from_grid("data/research_nhf_funding_t1/t1_grid_results.csv",
                         tf, ["lookback_n"])
    if nhf is None:
        nhf = {"lookback_n": 10}
        pr(f"  [WARN] Test 2 grid missing -- fallback NHF params {nhf}")
    cb, ap = int(vc["cb"]), int(vc["ap"])
    lb = int(nhf["lookback_n"])
    hb, am = EXIT_CFG[tf]["hb"], EXIT_CFG[tf]["am"]
    pr(f"  Signal A: VolContraction cb={cb} ap={ap}")
    pr(f"  Signal B: NHF lookback_n={lb}")
    pr(f"  Signal D: funding >= 0.02%/8h")
    pr(f"  Exit: hold_bars={hb}  atr_mult={am}  (System 7-inherited)")

    uni = C.load_universe(tf, with_atr_pct=True)
    fund = {s: C.load_funding(s) for s in uni}

    for level in [2, 3]:
        trades = []
        for s, d in uni.items():
            n = d["n"]
            cons = consec_below(d["atr_pct"], ap)
            cprev = np.roll(cons, 1); cprev[0] = 0
            rmin = C.rolling_min_prior(d["low"], cb)
            sigA = (cprev >= cb) & (d["close"] < rmin) & ~np.isnan(rmin)
            rmax = C.rolling_max_prior(d["high"], lb)
            sigB = (d["high"] > rmax) & (d["close"] < d["open"]) & ~np.isnan(rmax)
            # candidate bars: bear + at least one price signal (D alone can't reach L2)
            cand = np.where(d["bear"] & (sigA | sigB))[0]
            idx = []
            for i in cand:
                fr = C.get_rate_at(fund[s], int(d["ts"][i]))
                if fr is None or fr < C.FUNDING_THRESHOLD:
                    continue  # mandatory base gate
                count = int(sigA[i]) + int(sigB[i]) + int(fr >= D_THRESHOLD)
                if count >= level:
                    idx.append(i)
            trades += C.simulate(s, d, np.array(idx, dtype=int), hb, am, fund[s])
        if not trades:
            pr(f"\n  L{level}: 0 trades")
            grid_rows.append({"tf": tf, "level": level, "n": 0})
            continue
        dft = C.add_time_cols(pd.DataFrame(trades))
        stats = C.combo_stats(dft)
        grid_rows.append({"tf": tf, "level": level, "cb": cb, "ap": ap,
                          "lookback_n": lb, "hb": hb, "am": am, **stats})
        pr(f"\n  L{level} (any {level} of A/B/D): n={stats['n']}  "
           f"avg_r={stats['avg_r']:+.4f}R  total={stats['total_r']:+.1f}R  "
           f"wr={stats['win_rate']*100:.1f}%  pf={stats['pf']:.2f}")
        pr(f"    4.2 (0.25R): {'[PASS]' if stats['avg_r'] > C.COST_FLOOR else '[FAIL]'}   "
           f"2022: {stats['y2022_total']:+.2f}R {'[PASS]' if stats['y2022_positive'] else '[FAIL]'}   "
           f"2025 conc: {stats['r2025_share']*100 if pd.notna(stats['r2025_share']) else float('nan'):.1f}% "
           f"{'[FLAG]' if stats['flag_2025_conc'] else '[OK]'}")
        pr(f"    year-by-year:")
        for yr in range(2019, 2027):
            if stats[f"y{yr}_n"] > 0:
                pr(f"      {yr}: n={int(stats[f'y{yr}_n']):4d}  "
                   f"total={stats[f'y{yr}_total']:+8.2f}R  avg={stats[f'y{yr}_avg']:+.4f}R")
        if stats["avg_r"] > C.COST_FLOOR and stats["y2022_positive"]:
            pr(f"    *** BREAKTHROUGH CANDIDATE ***")
        dft = dft.drop(columns=["dt"])
        dft["tf"] = tf; dft["level"] = level
        all_trades.append(dft)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "t1_trades.csv", index=False)
pd.DataFrame(grid_rows).to_csv(OUT / "t1_grid_results.csv", index=False)
with open(OUT / "t1_summary.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUT}/t1_grid_results.csv, t1_summary.txt", flush=True)
