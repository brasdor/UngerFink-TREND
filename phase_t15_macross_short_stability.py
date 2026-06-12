"""
TRACK 3 -- T15 PARAMETER STABILITY: MA Cross Short 4H + Funding Gate.

Canonical: fast=20 slow=50 hb=30 am=2.0 funding>=0.01%
Grid:      fast [10,20,30] x slow [30,50,100] x hb [25,30,35]
           x funding_threshold [0.005%, 0.01%, 0.02%]   (fast < slow only)
PASS if >= 67% of combos adjacent to canonical pass 4.2 (avg_r > 0.25R).
Also: flag if the funding threshold is the only dimension driving the edge.
Output: data/research_macross_short_t15/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_macross_short_t15")
OUT.mkdir(parents=True, exist_ok=True)

TF = "4h"
AM = 2.0
FASTS = [10, 20, 30]
SLOWS = [30, 50, 100]
HOLDS = [25, 30, 35]
THRS  = [0.00005, 0.0001, 0.0002]   # 0.005% / 0.01% / 0.02%
CANON = {"fast": 20, "slow": 50, "hb": 30, "thr": 0.0001}

print("T15: MA Cross Short parameter stability", flush=True)
print("Loading 4H universe...", flush=True)
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
            sig[s] = np.where(mask)[0]   # ungated -- gate varies per thr
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
pr("T15 PARAMETER STABILITY: MA Cross Short 4H")
pr(f"Canonical: fast=20 slow=50 hb=30 thr=0.01%  |  4.2 floor 0.25R")
pr("=" * 74)
pr(f"\n  fast slow  hb   thr%      n     avg_r    2022R   4.2")
for _, x in g.sort_values(["fast", "slow", "hb", "thr"]).iterrows():
    is_canon = (x["fast"] == CANON["fast"] and x["slow"] == CANON["slow"]
                and x["hb"] == CANON["hb"] and x["thr"] == CANON["thr"])
    pr(f"  {int(x['fast']):<4} {int(x['slow']):<5} {int(x['hb']):<4} "
       f"{x['thr']*100:.3f}  {int(x['n']):6d}  {x['avg_r']:+.4f}  "
       f"{x['y2022_total']:+7.1f}  {'PASS' if x['avg_r'] > 0.25 else 'fail'}"
       f"{'   <-- CANONICAL' if is_canon else ''}")

# adjacent zone: one-step move in ONE dimension, others at canonical
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
        row = at(p["fast"], p["slow"], p["hb"], p["thr"])
        if row is not None:
            adj.append((dim, p[dim], row))

pr(f"\n--- Adjacent zone (one-step from canonical) ---")
n_pass = 0
for dim, val, row in adj:
    ok = row["avg_r"] > 0.25
    n_pass += ok
    vs = f"{val*100:.3f}%" if dim == "thr" else str(val)
    pr(f"  {dim}={vs:<7} n={int(row['n']):5d}  avg_r={row['avg_r']:+.4f}R  "
       f"{'[PASS]' if ok else '[fail]'}")
canon_row = at(**CANON)
canon_ok = canon_row["avg_r"] > 0.25
frac = (n_pass + canon_ok) / (len(adj) + 1)
pr(f"\n  Canonical: avg_r={canon_row['avg_r']:+.4f}R {'[PASS]' if canon_ok else '[fail]'}")
pr(f"  Zone: {n_pass + canon_ok}/{len(adj) + 1} pass 4.2 = {frac*100:.0f}%  "
   f"(gate >= 67%)  {'[PASS]' if frac >= 0.67 else '[FAIL]'}")

# funding-threshold dependence check
pr(f"\n--- Funding threshold dependence ---")
pr(f"  At canonical fast/slow/hb, avg_r by threshold:")
for thr in THRS:
    row = at(CANON["fast"], CANON["slow"], CANON["hb"], thr)
    if row is not None:
        pr(f"    thr={thr*100:.3f}%: n={int(row['n']):5d}  avg_r={row['avg_r']:+.4f}R")
sub_nothr = g[(g["thr"] == 0.00005)]
frac_pass_low = (sub_nothr["avg_r"] > 0.25).mean() if len(sub_nothr) else 0
pr(f"  At LOWEST threshold (0.005%), {frac_pass_low*100:.0f}% of all combos pass 4.2")
pr(f"  -> if edge survives at 0.005% across fast/slow/hb, the MA cross signal")
pr(f"     contributes structure beyond the funding gate alone")

with open(OUT / "t15_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
