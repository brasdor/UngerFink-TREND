"""
TRACK 1 -- T4 ROBUSTNESS: MA Cross Short 4H + Funding Gate (System 8 candidate).
Identical battery to System 7 T4.
Input:  data/research_macross_short_t2/t2_trades.csv (T2 PASS config)
Output: data/research_macross_short_t4/
Benchmark: System 7 avg_r=+0.304R, t=10.3, MC p05=+455R (bs=50), IS->OOS 24.4%
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/research_macross_short_t4")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

N_MC = 2000
BLOCK_SIZES = [1, 5, 10, 20, 50]
EXTRA_COSTS = [0.00, 0.05, 0.10, 0.15, 0.20]

tr = pd.read_csv("data/research_macross_short_t2/t2_trades.csv")
tr = tr.sort_values("entry_ts").reset_index(drop=True)
r = tr["r"].values
n = len(r)

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

def pf_of(x):
    w = x[x > 0].sum(); l = abs(x[x <= 0].sum())
    return w / l if l > 0 else np.inf

def max_dd(x):
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    return (eq - peak).min()

pr("=" * 74)
pr("T4 ROBUSTNESS: MA Cross Short 4H + Funding Gate")
pr("Config: fast=20 slow=50 hb=30 am=2.0  gate>=0.01%/8h + EMA200 bear")
pr("=" * 74)

# 1. Baseline
avg = r.mean(); std = r.std(ddof=1)
t_score = avg / (std / np.sqrt(n))
wr = (r > 0).mean()
# losing streak
streak = mx = 0
for x in r:
    streak = streak + 1 if x <= 0 else 0
    mx = max(mx, streak)
pr(f"\n--- 1. Baseline ---")
pr(f"  trades={n}  avg_r={avg:+.4f}R  wr={wr*100:.1f}%  pf={pf_of(r):.3f}")
pr(f"  total={r.sum():+.1f}R  max_dd={max_dd(r):+.1f}R  max losing streak={mx}")
pr(f"  t-score={t_score:.1f}  (System 7 benchmark: 10.3)")

# 2. Block bootstrap MC
pr(f"\n--- 2. Block Bootstrap Monte Carlo ({N_MC} runs) ---")
pr(f"  bs   totR_p05   totR_p50   totR_p95   prob_pos   PF_p05   DD_p05")
mc_ok_all = True
mc_rows = []
for bs in BLOCK_SIZES:
    n_blocks = int(np.ceil(n / bs))
    tot = np.empty(N_MC); pfs = np.empty(N_MC); dds = np.empty(N_MC)
    for k in range(N_MC):
        starts = rng.integers(0, max(1, n - bs + 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(bs)[None, :]).ravel()[:n]
        x = r[idx]
        tot[k] = x.sum(); pfs[k] = pf_of(x); dds[k] = max_dd(x)
    p05, p50, p95 = np.percentile(tot, [5, 50, 95])
    prob_pos = (tot > 0).mean()
    pf05 = np.percentile(pfs, 5)
    dd05 = np.percentile(dds, 5)
    mc_ok_all &= p05 > 0
    mc_rows.append({"block_size": bs, "tot_p05": p05, "tot_p50": p50, "tot_p95": p95,
                    "prob_pos": prob_pos, "pf_p05": pf05, "dd_p05": dd05})
    pr(f"  {bs:<4} {p05:+9.1f}  {p50:+9.1f}  {p95:+9.1f}    {prob_pos*100:6.2f}%   "
       f"{pf05:6.3f}  {dd05:+8.1f}")
pd.DataFrame(mc_rows).to_csv(OUT / "t4_montecarlo.csv", index=False)
pr(f"  MC p05 positive at ALL block sizes: {'[PASS]' if mc_ok_all else '[FAIL]'}")
pr(f"  (System 7: MC p05 bs=50 = +455R)")

# 3. Cost stress
pr(f"\n--- 3. Cost Stress ---")
cost_rows = []
for c in EXTRA_COSTS:
    x = r - c
    cost_rows.append({"extra_cost": c, "avg_r": x.mean(), "total_r": x.sum(), "pf": pf_of(x)})
    pr(f"  +{c:.2f}R: avg_r={x.mean():+.4f}R  total={x.sum():+8.1f}R  pf={pf_of(x):.3f}")
pd.DataFrame(cost_rows).to_csv(OUT / "t4_cost_stress.csv", index=False)
c05_pf = pf_of(r - 0.05)
pr(f"  0.05R stress PF > 1.0: {'[PASS]' if c05_pf > 1.0 else '[FAIL]'}")

# 4. Period splits
pr(f"\n--- 4. Period Splits ---")
h = n // 2
first_c, second_c = r[:h], r[h:]
pr(f"  By trade count: first half avg={first_c.mean():+.4f}R (n={h})  "
   f"second half avg={second_c.mean():+.4f}R (n={n-h})")
tr["dt"] = pd.to_datetime(tr["entry_ts"], unit="ms", utc=True)
t_mid = tr["dt"].min() + (tr["dt"].max() - tr["dt"].min()) / 2
first_t = tr.loc[tr["dt"] < t_mid, "r"]; second_t = tr.loc[tr["dt"] >= t_mid, "r"]
pr(f"  By time (mid {t_mid.date()}): first avg={first_t.mean():+.4f}R (n={len(first_t)})  "
   f"second avg={second_t.mean():+.4f}R (n={len(second_t)})")
second_ok = second_c.mean() > 0 and second_t.mean() > 0
pr(f"  Second half not negative: {'[PASS]' if second_ok else '[FAIL]'}")
last500 = r[-500:]
last500_flag = last500.mean() < 0.10
pr(f"  Last 500 trades: avg_r={last500.mean():+.4f}R  total={last500.sum():+.1f}R  "
   f"{'[FLAG < 0.10R]' if last500_flag else '[OK]'}")
pr(f"  (System 7 caveat was last-500 avg +0.092R)")

# 5. Remove best assets
pr(f"\n--- 5. Remove Best Assets ---")
by_asset = tr.groupby("symbol")["r"].sum().sort_values(ascending=False)
ra_ok = True
for k in [1, 3, 5]:
    excl = set(by_asset.index[:k])
    x = tr.loc[~tr["symbol"].isin(excl), "r"]
    pr(f"  remove top-{k} ({', '.join(list(excl)[:3])}{'...' if k>3 else ''}): "
       f"avg_r={x.mean():+.4f}R  total={x.sum():+8.1f}R  n={len(x)}")
    if k == 1:
        ra_ok = x.mean() > 0.10
pr(f"  Remove top-1 asset avg_r > 0.10R: {'[PASS]' if ra_ok else '[FAIL]'}")

# 6. Remove best months
pr(f"\n--- 6. Remove Best Months ---")
by_month = tr.groupby("month")["r"].sum().sort_values(ascending=False)
rm_ok = True
for k in [1, 2, 3]:
    excl = set(by_month.index[:k])
    x = tr.loc[~tr["month"].isin(excl), "r"]
    pr(f"  remove top-{k} ({', '.join(sorted(excl))}): avg_r={x.mean():+.4f}R  "
       f"total={x.sum():+8.1f}R")
    if k == 1:
        rm_ok = x.mean() > 0.10
pr(f"  Remove top-1 month avg_r > 0.10R: {'[PASS]' if rm_ok else '[FAIL]'}")

# 7. Concentration
pr(f"\n--- 7. Asset Concentration ---")
total = r.sum()
for k in [1, 3, 5]:
    pct = by_asset.head(k).sum() / total * 100
    pr(f"  top-{k}: {by_asset.head(k).sum():+8.1f}R  ({pct:.1f}% of total)")
conc_ok = by_asset.iloc[0] / total < 0.50

# 8. Year-by-year + remove best years
pr(f"\n--- 8. Year-by-Year ---")
for yr in sorted(tr["year"].unique()):
    yd = tr.loc[tr["year"] == yr, "r"]
    pr(f"  {yr}: n={len(yd):4d}  total={yd.sum():+8.2f}R  avg={yd.mean():+.4f}R")
yr2026 = tr.loc[tr["year"] == 2026, "r"]
y2026_ok = yr2026.sum() > 0
pr(f"  2026 partial positive: {'[PASS]' if y2026_ok else '[FAIL]'} ({yr2026.sum():+.1f}R)")
pr(f"\n--- Remove Best Years ---")
rem_year_ok = {}
for yr in [2022, 2024]:
    x = tr.loc[tr["year"] != yr, "r"]
    rem_year_ok[yr] = x.mean() > 0 and x.sum() > 0
    pr(f"  remove {yr}: avg_r={x.mean():+.4f}R  total={x.sum():+8.1f}R  "
       f"{'[PASS]' if rem_year_ok[yr] else '[FAIL]'}")

# Critical checks
checks = {
    "MC p05 positive all block sizes": mc_ok_all,
    "Cost stress 0.05R PF > 1.0":      c05_pf > 1.0,
    "Remove top-1 asset > 0.10R":      ra_ok,
    "Remove top-1 month > 0.10R":      rm_ok,
    "Remove 2022 profitable":          rem_year_ok[2022],
    "Remove 2024 profitable":          rem_year_ok[2024],
    "Second half not negative":        second_ok,
    "2026 partial positive":           y2026_ok,
    "Concentration top-1 < 50%":       conc_ok,
}
n_pass = sum(checks.values())
pr(f"\n--- T4 SCORECARD ---")
for k, v in checks.items():
    pr(f"  {'[PASS]' if v else '[FAIL]'} {k}")
if last500_flag:
    pr(f"  [FLAG] last 500 trades avg_r {last500.mean():+.4f}R < 0.10R")
pr(f"\n  Passed {n_pass}/{len(checks)}")
pr(f"\n--- BENCHMARK vs SYSTEM 7 ---")
bs50 = [m for m in mc_rows if m['block_size'] == 50][0]
pr(f"  metric        System 7      MA Cross")
pr(f"  avg_r         +0.304R       {avg:+.3f}R")
pr(f"  t-score       10.3          {t_score:.1f}")
pr(f"  MC p05 bs50   +455R         {bs50['tot_p05']:+.0f}R")
pr(f"  IS->OOS       24.4%         4.1%")
verdict = "PASS -- proceed to T5" if n_pass == len(checks) else (
          "REVIEW" if n_pass >= len(checks) - 2 else "HALT")
pr(f"\n  T4 VERDICT: {verdict}")

with open(OUT / "t4_master_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
