"""
ACTION 3 -- T15 PARAMETER STABILITY: MA Cross Short V2 (re-canonicalized).
Canonical: fast=20 slow=30 hb=35 am=2.0 funding>=0.01%, 4H.
Grid: fast [10,20,30] x slow [20,30,50] x hb [30,35,40]
      x funding_threshold [0.005%, 0.01%, 0.02%]   (fast < slow only)
Target: >= 67% of zone (one-step adjacent) combos pass 4.2 (0.25R).
Output: data/research_macross_short_v2_t15/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_macross_short_v2_t15")
OUT.mkdir(parents=True, exist_ok=True)

TF, AM = "4h", 2.0
FASTS = [10, 20, 30]
SLOWS = [20, 30, 50]
HOLDS = [30, 35, 40]
THRS  = [0.00005, 0.0001, 0.0002]
CANON = {"fast": 20, "slow": 30, "hb": 35, "thr": 0.0001}

print("T15 V2: MA Cross Short re-canonicalized stability", flush=True)
uni = C.load_universe(TF, log_every=50)
fund = {s: C.load_funding(s) for s in uni}
periods = sorted(set(FASTS + SLOWS))
emas = {s: {p: C.compute_ema(d["close"], p) for p in periods} for s, d in uni.items()}

rows = []
for fast in FASTS:
    for slow in SLOWS:
        if fast >= slow:
            continue
        sig = {}
        for s, d in uni.items():
            f_, s_ = emas[s][fast], emas[s][slow]
            f_prev = np.roll(f_, 1); s_prev = np.roll(s_, 1)
            mask = d["bear"] & (f_ < s_) & (f_prev >= s_prev)
            mask &= ~np.isnan(f_) & ~np.isnan(s_) & ~np.isnan(f_prev) & ~np.isnan(s_prev)
            mask[0] = False
            sig[s] = np.where(mask)[0]
        for hb in HOLDS:
            for thr in THRS:
                trades = []
                for s, d in uni.items():
                    trades += C.simulate(s, d, sig[s], hb, AM, fund[s],
                                         funding_threshold=thr)
                if not trades:
                    continue
                dft = C.add_time_cols(pd.DataFrame(trades))
                st = C.combo_stats(dft)
                rows.append({"fast": fast, "slow": slow, "hb": hb, "thr": thr, **st})
        print(f"  fast={fast} slow={slow} done", flush=True)

g = pd.DataFrame(rows)
g.to_csv(OUT / "t15_grid.csv", index=False)

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

pr("=" * 74)
pr("T15 V2: MA Cross Short stability around NEW canonical (20,30,35,0.01%)")
pr("=" * 74)
pr(f"\n  fast slow  hb   thr%      n     avg_r    2022R   4.2")
for _, x in g.sort_values(["fast", "slow", "hb", "thr"]).iterrows():
    is_c = (x["fast"], x["slow"], x["hb"], x["thr"]) == (20, 30, 35, 0.0001)
    pr(f"  {int(x['fast']):<4} {int(x['slow']):<5} {int(x['hb']):<4} "
       f"{x['thr']*100:.3f}  {int(x['n']):6d}  {x['avg_r']:+.4f}  "
       f"{x['y2022_total']:+7.1f}  {'PASS' if x['avg_r'] > 0.25 else 'fail'}"
       f"{'   <-- CANONICAL' if is_c else ''}")

def at(fast, slow, hb, thr):
    m = g[(g["fast"] == fast) & (g["slow"] == slow) & (g["hb"] == hb) & (g["thr"] == thr)]
    return m.iloc[0] if len(m) else None

adj = []
for dim, vals in [("fast", FASTS), ("slow", SLOWS), ("hb", HOLDS), ("thr", THRS)]:
    ci = vals.index(CANON[dim])
    for step in (-1, 1):
        ni = ci + step
        if ni < 0 or ni >= len(vals):
            continue
        p = dict(CANON); p[dim] = vals[ni]
        if p["fast"] >= p["slow"]:
            continue
        row = at(p["fast"], p["slow"], p["hb"], p["thr"])
        if row is not None:
            adj.append((dim, p[dim], row))

pr(f"\n--- Adjacent zone ---")
n_pass = 0
for dim, val, row in adj:
    ok = row["avg_r"] > 0.25
    n_pass += ok
    vs = f"{val*100:.3f}%" if dim == "thr" else str(val)
    pr(f"  {dim}={vs:<7} n={int(row['n']):5d}  avg_r={row['avg_r']:+.4f}R  "
       f"{'[PASS]' if ok else '[fail]'}")
canon_row = at(20, 30, 35, 0.0001)
canon_ok = canon_row["avg_r"] > 0.25
frac = (n_pass + canon_ok) / (len(adj) + 1)
pr(f"\n  Canonical: avg_r={canon_row['avg_r']:+.4f}R [{'PASS' if canon_ok else 'fail'}]")
pr(f"  Zone: {n_pass + canon_ok}/{len(adj) + 1} = {frac*100:.0f}%  (gate >= 67%)  "
   f"{'[PASS]' if frac >= 0.67 else '[FAIL]'}")
pr(f"\n  Within design level (thr=0.01%): "
   f"{(g.loc[g['thr'] == 0.0001, 'avg_r'] > 0.25).mean()*100:.0f}% of "
   f"{len(g[g['thr'] == 0.0001])} combos pass 4.2 "
   f"(System 7 precedent: 74%)")
pr(f"  Profitability (avg_r>0) across full grid: {(g['avg_r'] > 0).mean()*100:.0f}%")

with open(OUT / "t15_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
