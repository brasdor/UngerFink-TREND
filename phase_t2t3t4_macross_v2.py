"""
ACTION 1 (part 1) -- MA Cross Short V2 re-canonicalized: T2 + T3 + T4.
New canonical: fast=20 slow=30 hold=35 am=2.0, EMA200 bear, funding>=0.01%, 4H.

T3: hold_bars sweep [25,30,35,40,45] (canonical hb=35 regardless; flag edge)
T2: IS 2019-2022 / OOS 2023-2026, full gates
T4: identical battery to original MA Cross T4

Writes verdict to data/research_macross_short_v2_t2/verdict.txt:
  PROCEED  -> all critical gates pass (2026-partial soft patch carried as
              documented caveat, consistent with System 7 acceptance)
  HALT     -> anything else fails
Outputs: data/research_macross_short_v2_t2/, data/research_macross_short_v2_t4/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT2 = Path("data/research_macross_short_v2_t2"); OUT2.mkdir(parents=True, exist_ok=True)
OUT4 = Path("data/research_macross_short_v2_t4"); OUT4.mkdir(parents=True, exist_ok=True)

TF = "4h"
FAST, SLOW, HB, AM = 20, 30, 35, 2.0
T3_HOLDS = [25, 30, 35, 40, 45]
IS_YEARS, OOS_YEARS = {2019, 2020, 2021, 2022}, {2023, 2024, 2025, 2026}
rng = np.random.default_rng(42)


def simulate_with_exits(symbol, d, signal_idx, hb, am, funding):
    """Same mechanics as C.simulate but records exit_ts (for T5 cap replay)."""
    ts, opens, highs = d["ts"], d["open"], d["high"]
    atrs = d["atr"]; n = d["n"]
    trades = []; next_free = 0
    for i in signal_idx:
        if i < next_free or i < C.WARMUP or i + 1 >= n:
            continue
        fr = C.get_rate_at(funding, int(ts[i]))
        if fr is None or fr < C.FUNDING_THRESHOLD:
            continue
        if np.isnan(atrs[i]) or atrs[i] <= 0:
            continue
        entry = opens[i + 1]; stop = entry + atrs[i] * am; risk = stop - entry
        if risk <= 0:
            continue
        exit_bar = None; r = None
        last = min(i + hb, n - 1)
        for j in range(i + 1, last + 1):
            if highs[j] >= stop:
                r = (entry - stop) / risk; exit_bar = j; break
            if j == i + hb:
                r = (entry - opens[j]) / risk; exit_bar = j; break
        if exit_bar is None:
            continue
        trades.append({"symbol": symbol, "entry_ts": int(ts[i]),
                       "exit_ts": int(ts[exit_bar]), "r": r, "funding": fr})
        next_free = exit_bar + 1
    return trades


lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

print("Loading 4H universe...", flush=True)
uni = C.load_universe(TF, log_every=50)
fund = {s: C.load_funding(s) for s in uni}
sig = {}
for s, d in uni.items():
    f_ = C.compute_ema(d["close"], FAST)
    s_ = C.compute_ema(d["close"], SLOW)
    f_prev = np.roll(f_, 1); s_prev = np.roll(s_, 1)
    mask = d["bear"] & (f_ < s_) & (f_prev >= s_prev)
    mask &= ~np.isnan(f_) & ~np.isnan(s_) & ~np.isnan(f_prev) & ~np.isnan(s_prev)
    mask[0] = False
    sig[s] = np.where(mask)[0]

# ------------------------------------------------------------------ T3
pr("=" * 74)
pr("T3 EXIT SWEEP: MA Cross Short V2 (fast=20 slow=30 am=2.0, 4H)")
pr("=" * 74)
pr(f"  hb     n     avg_r    total_r    wr%    pf    2022R    2025conc")
t3_rows = []
trades_by_hb = {}
for hb in T3_HOLDS:
    trades = []
    for s, d in uni.items():
        trades += simulate_with_exits(s, d, sig[s], hb, AM, fund[s])
    dft = C.add_time_cols(pd.DataFrame(trades))
    trades_by_hb[hb] = dft
    st = C.combo_stats(dft)
    t3_rows.append({"hold_bars": hb, **st})
    pr(f"  {hb:<4} {st['n']:6d}  {st['avg_r']:+.4f}  {st['total_r']:+9.1f}  "
       f"{st['win_rate']*100:5.1f}  {st['pf']:5.2f}  {st['y2022_total']:+7.1f}  "
       f"{st['r2025_share']*100:6.1f}%")
df_t3 = pd.DataFrame(t3_rows)
df_t3.to_csv(OUT2 / "t3_exit_sweep.csv", index=False)
best_hb = int(df_t3.iloc[df_t3["avg_r"].values.argmax()]["hold_bars"])
edge_flag = best_hb in (T3_HOLDS[0], T3_HOLDS[-1])
pr(f"  Best hb={best_hb}  {'[FLAG grid edge]' if edge_flag else '[interior]'}  "
   f"-- canonical stays hb={HB}")

tr = trades_by_hb[HB].copy()
tr["month"] = tr["dt"].dt.to_period("M").astype(str)
tr.drop(columns=["dt"]).to_csv(OUT2 / "t2_trades.csv", index=False)
r = tr.sort_values("entry_ts")["r"].values
n = len(r)

# ------------------------------------------------------------------ T2
pr("\n" + "=" * 74)
pr(f"T2 WALK-FORWARD: MA Cross Short V2 (fast={FAST} slow={SLOW} hb={HB} am={AM})")
pr(f"Universe: {tr['symbol'].nunique()} symbols   Total trades: {n}")
pr("=" * 74)
df_is  = tr[tr["year"].isin(IS_YEARS)]
df_oos = tr[tr["year"].isin(OOS_YEARS)]
is_avg, oos_avg = df_is["r"].mean(), df_oos["r"].mean()
degrade = (is_avg - oos_avg) / abs(is_avg)
pr(f"  IS  (2019-2022): n={len(df_is):5d}  avg_r={is_avg:+.4f}R")
pr(f"  OOS (2023-2026): n={len(df_oos):5d}  avg_r={oos_avg:+.4f}R")
pr(f"  Degradation: {degrade*100:.1f}% (max 40%)")
is_ok = is_avg > C.COST_FLOOR
deg_ok = degrade < 0.40 and oos_avg > 0
avg_r = tr["r"].mean()
wr = (tr["r"] > 0).mean()
wins, losses = tr.loc[tr["r"] > 0, "r"], tr.loc[tr["r"] <= 0, "r"]
pf = wins.sum() / abs(losses.sum())
cost_ok = avg_r > C.COST_FLOOR
by_asset = tr.groupby("symbol")["r"].sum().sort_values(ascending=False)
total_r = tr["r"].sum()
top1_sym, top1_r = by_asset.index[0], by_asset.iloc[0]
conc_ok = top1_r / total_r < 0.50
pr(f"  avg_r={avg_r:+.4f}R [{'PASS' if cost_ok else 'FAIL'}]  wr={wr*100:.1f}% "
   f"{'[in gate]' if 0.30 <= wr <= 0.45 else '[FLAG -- structural, S7=45.6%]'}  pf={pf:.2f}")
pr(f"  4.7 top-1: {top1_sym} {top1_r/total_r*100:.1f}% [{'PASS' if conc_ok else 'FAIL'}]")
pr(f"\n  Year-by-year:")
yr_totals = {}
for yr in sorted(tr["year"].unique()):
    yd = tr[tr["year"] == yr]
    yr_totals[yr] = yd["r"].sum()
    pr(f"    {yr}: n={len(yd):4d}  total={yd['r'].sum():+8.2f}R  avg={yd['r'].mean():+.4f}R")
rank_2022 = sorted(yr_totals.values(), reverse=True).index(yr_totals[2022]) + 1
y2022_ok = rank_2022 <= 2
pr(f"  2022 rank: #{rank_2022} [{'PASS' if y2022_ok else 'FAIL'}]")
by_month = tr.groupby("month")["r"].sum().sort_values(ascending=False)
avg_no_bm = tr.loc[tr["month"] != by_month.index[0], "r"].mean()
avg_no_ba = tr.loc[tr["symbol"] != top1_sym, "r"].mean()
bm_ok, ba_ok = avg_no_bm > 0.10, avg_no_ba > 0.10
pr(f"  Remove best month ({by_month.index[0]}): {avg_no_bm:+.4f}R [{'PASS' if bm_ok else 'FAIL'}]")
pr(f"  Remove best asset ({top1_sym}): {avg_no_ba:+.4f}R [{'PASS' if ba_ok else 'FAIL'}]")
t2_checks = {"IS>0.25R": is_ok, "degradation<40%": deg_ok, "4.2": cost_ok,
             "4.7": conc_ok, "2022 top-2": y2022_ok, "rm month": bm_ok, "rm asset": ba_ok}
t2_pass = all(t2_checks.values())
pr(f"\n  T2: {'PASS' if t2_pass else 'FAIL'} ({sum(t2_checks.values())}/{len(t2_checks)})")

# ------------------------------------------------------------------ T4
def pf_of(x):
    w = x[x > 0].sum(); l = abs(x[x <= 0].sum())
    return w / l if l > 0 else np.inf

def max_dd(x):
    eq = np.cumsum(x); return (eq - np.maximum.accumulate(eq)).min()

pr("\n" + "=" * 74)
pr("T4 ROBUSTNESS: MA Cross Short V2")
pr("=" * 74)
std = r.std(ddof=1); t_score = r.mean() / (std / np.sqrt(n))
streak = mx = 0
for x in r:
    streak = streak + 1 if x <= 0 else 0
    mx = max(mx, streak)
pr(f"  baseline: n={n} avg={r.mean():+.4f}R wr={wr*100:.1f}% pf={pf_of(r):.3f} "
   f"t={t_score:.1f} maxDD={max_dd(r):+.1f}R streak={mx}")

pr(f"\n  MC (2000 runs):  bs   p05      p50      p95     prob+   PF_p05")
mc_ok = True; mc_rows = []
for bs in [1, 5, 10, 20, 50]:
    n_blocks = int(np.ceil(n / bs))
    tot = np.empty(2000); pfs = np.empty(2000)
    for k in range(2000):
        starts = rng.integers(0, max(1, n - bs + 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(bs)[None, :]).ravel()[:n]
        x = r[idx]; tot[k] = x.sum(); pfs[k] = pf_of(x)
    p05, p50, p95 = np.percentile(tot, [5, 50, 95])
    mc_ok &= p05 > 0
    mc_rows.append({"bs": bs, "p05": p05, "p50": p50, "p95": p95,
                    "prob_pos": (tot > 0).mean(), "pf_p05": np.percentile(pfs, 5)})
    pr(f"                 {bs:<4} {p05:+8.1f} {p50:+8.1f} {p95:+8.1f}  "
       f"{(tot>0).mean()*100:6.2f}%  {np.percentile(pfs,5):.3f}")
pd.DataFrame(mc_rows).to_csv(OUT4 / "t4_montecarlo.csv", index=False)
pr(f"  MC p05 all positive: [{'PASS' if mc_ok else 'FAIL'}]")

c05_pf = pf_of(r - 0.05)
pr(f"\n  cost stress: " + "  ".join(
    f"+{c:.2f}R pf={pf_of(r-c):.2f}" for c in [0.0, 0.05, 0.10, 0.15, 0.20]))
pr(f"  0.05R PF>1: [{'PASS' if c05_pf > 1.0 else 'FAIL'}]")

h = n // 2
sec_ok = r[h:].mean() > 0
last500 = r[-500:]
last500_flag = last500.mean() < 0.10
pr(f"\n  halves: first {r[:h].mean():+.4f}R / second {r[h:].mean():+.4f}R "
   f"[{'PASS' if sec_ok else 'FAIL'}]")
pr(f"  last 500: {last500.mean():+.4f}R {'[FLAG]' if last500_flag else '[OK]'}")

tr_s = tr.sort_values("entry_ts")
ra1 = tr_s.loc[tr_s["symbol"] != top1_sym, "r"].mean()
ra_ok = ra1 > 0.10
for k in [1, 3, 5]:
    excl = set(by_asset.index[:k])
    x = tr_s.loc[~tr_s["symbol"].isin(excl), "r"]
    pr(f"  rm top-{k} assets: avg={x.mean():+.4f}R total={x.sum():+8.1f}R")
rm1 = tr_s.loc[tr_s["month"] != by_month.index[0], "r"].mean()
rm_ok = rm1 > 0.10
for k in [1, 2, 3]:
    excl = set(by_month.index[:k])
    x = tr_s.loc[~tr_s["month"].isin(excl), "r"]
    pr(f"  rm top-{k} months: avg={x.mean():+.4f}R total={x.sum():+8.1f}R")
pr(f"  concentration: top1 {by_asset.iloc[0]/total_r*100:.1f}%  "
    f"top3 {by_asset.head(3).sum()/total_r*100:.1f}%  "
    f"top5 {by_asset.head(5).sum()/total_r*100:.1f}%")

rem_yr = {}
for yr in [2022, 2024]:
    x = tr_s.loc[tr_s["year"] != yr, "r"]
    rem_yr[yr] = x.sum() > 0
    pr(f"  rm {yr}: avg={x.mean():+.4f}R total={x.sum():+8.1f}R "
       f"[{'PASS' if rem_yr[yr] else 'FAIL'}]")
y2026 = tr_s.loc[tr_s["year"] == 2026, "r"].sum()
y2026_ok = y2026 > 0
pr(f"  2026 partial: {y2026:+.1f}R [{'PASS' if y2026_ok else 'FAIL -- caveat class, monitor in T9B'}]")

t4_checks = {"MC p05 all bs": mc_ok, "cost 0.05R": c05_pf > 1.0,
             "rm top-1 asset >0.10R": ra_ok, "rm top-1 month >0.10R": rm_ok,
             "rm 2022": rem_yr[2022], "rm 2024": rem_yr[2024],
             "second half": sec_ok, "2026 partial": y2026_ok,
             "conc <50%": conc_ok}
pr(f"\n  T4 SCORECARD:")
for k, v in t4_checks.items():
    pr(f"    [{'PASS' if v else 'FAIL'}] {k}")
n4 = sum(t4_checks.values())
pr(f"    {n4}/{len(t4_checks)}")

# verdict: PROCEED if T2 all pass and T4 fails nothing except possibly 2026-partial
hard_t4 = {k: v for k, v in t4_checks.items() if k != "2026 partial"}
proceed = t2_pass and all(hard_t4.values())
pr(f"\n  VERDICT: {'PROCEED to T5-T8' if proceed else 'HALT for review'}")
if proceed and not y2026_ok:
    pr(f"  (2026 partial {y2026:+.1f}R carried as documented caveat -- System 7 precedent)")

pr(f"\n--- COMPARISON ---")
pr(f"  metric        System 7    MACross v1   MACross V2")
pr(f"  avg_r         +0.304R     +0.255R      {avg_r:+.3f}R")
pr(f"  trades        3214        2910         {n}")
pr(f"  t-score       10.3        9.2          {t_score:.1f}")
bs50 = mc_rows[-1]
pr(f"  MC p05 bs50   +455R       +350R        {bs50['p05']:+.0f}R")
pr(f"  IS->OOS       24.4%       4.1%         {degrade*100:.1f}%")
pr(f"  last 500      +0.092R     +0.292R      {last500.mean():+.3f}R")

with open(OUT2 / "t2_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
with open(OUT4 / "t4_master_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
with open(OUT2 / "verdict.txt", "w", encoding="ascii") as f:
    f.write("PROCEED" if proceed else "HALT")
print(f"\nSaved. Verdict: {'PROCEED' if proceed else 'HALT'}", flush=True)
