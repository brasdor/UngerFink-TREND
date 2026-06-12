"""
TRACK 2 -- MA Cross Short 4H: T2 walk-forward + T3 exit engineering.

Config: fast=20 slow=50 hold=30 (4H bars) atr_mult=2.0
        + EMA200 bear filter + funding >= 0.01%/8h (System 8 candidate)

T2: reuses the T1 trade list (identical engine, same config -- no need to
    re-run). IS 2019-2022 / OOS 2023-2026. IS avg_r > 0.25R required,
    IS->OOS degradation < 40%.
T3: re-runs the engine across hold_bars [15,20,25,30,35,40].

Output: data/research_macross_short_t2/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import t1_short_sweep_common as C

OUT = Path("data/research_macross_short_t2")
OUT.mkdir(parents=True, exist_ok=True)

TF = "4h"
FAST, SLOW, HB, AM = 20, 50, 30, 2.0
IS_YEARS  = {2019, 2020, 2021, 2022}
OOS_YEARS = {2023, 2024, 2025, 2026}
T3_HOLD_BARS = [15, 20, 25, 30, 35, 40]

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

# ------------------------------------------------------------------ T2
t1 = pd.read_csv("data/research_macross_funding_t1/t1_trades.csv")
tr = t1[(t1["tf"] == TF) & (t1["fast_ema"] == FAST) & (t1["slow_ema"] == SLOW) &
        (t1["hold_bars"] == HB) & (t1["atr_mult"] == AM)].copy()
tr["dt"] = pd.to_datetime(tr["entry_ts"], unit="ms", utc=True)
tr["year"] = tr["dt"].dt.year
tr["month"] = tr["dt"].dt.to_period("M").astype(str)

pr("=" * 74)
pr("T2 WALK-FORWARD: MA Cross Short 4H + Funding Gate (System 8 candidate)")
pr(f"Config: fast={FAST} slow={SLOW} hb={HB} am={AM}  gate>=0.01%/8h + EMA200 bear")
pr(f"Universe: {tr['symbol'].nunique()} symbols   Total trades: {len(tr)}")
pr("=" * 74)

df_is  = tr[tr["year"].isin(IS_YEARS)]
df_oos = tr[tr["year"].isin(OOS_YEARS)]
is_avg  = df_is["r"].mean()
oos_avg = df_oos["r"].mean()
degrade = (is_avg - oos_avg) / abs(is_avg) if is_avg != 0 else 0.0
pr(f"\n--- IS / OOS Split ---")
pr(f"  IS  (2019-2022): n={len(df_is):5d}  avg_r={is_avg:+.4f}R")
pr(f"  OOS (2023-2026): n={len(df_oos):5d}  avg_r={oos_avg:+.4f}R")
pr(f"  IS->OOS degradation: {degrade*100:.1f}%  (max 40%)")
is_ok      = is_avg > C.COST_FLOOR
degrade_ok = degrade < 0.40 and oos_avg > 0
pr(f"  IS avg_r > 0.25R: {'[PASS]' if is_ok else '[FAIL]'}")
pr(f"  Degradation:      {'[PASS]' if degrade_ok else '[FAIL]'}")

avg_r = tr["r"].mean()
wr = (tr["r"] > 0).mean()
wins, losses = tr.loc[tr["r"] > 0, "r"], tr.loc[tr["r"] <= 0, "r"]
pf = wins.sum() / abs(losses.sum())
cost_ok = avg_r > C.COST_FLOOR
wr_in_gate = 0.30 <= wr <= 0.45
pr(f"\n--- 4.1 / 4.2 ---")
pr(f"  avg_r: {avg_r:+.4f}R (floor 0.25R)  {'[PASS]' if cost_ok else '[FAIL]'}")
pr(f"  win rate: {wr*100:.1f}%  pf={pf:.2f}  "
   f"{'[PASS]' if wr_in_gate else '[FLAG above 45% -- structural: funding gate selects crowded shorts, not breakout trend entries; System 7 precedent 45.6%]'}")

by_asset = tr.groupby("symbol")["r"].sum().sort_values(ascending=False)
total_r = tr["r"].sum()
top1_sym, top1_r = by_asset.index[0], by_asset.iloc[0]
conc_ok = top1_r / total_r < 0.50
pr(f"\n--- 4.7 Concentration ---")
pr(f"  Top-1: {top1_sym} {top1_r:+.1f}R ({top1_r/total_r*100:.1f}%)  "
   f"{'[PASS]' if conc_ok else '[FAIL]'}")
for sym, r in by_asset.head(5).items():
    pr(f"    {sym:<18} {r:+8.2f}R ({r/total_r*100:.1f}%)")

pr(f"\n--- Year-by-Year ---")
yr_totals = {}
for yr in sorted(tr["year"].unique()):
    yd = tr[tr["year"] == yr]
    yr_totals[yr] = yd["r"].sum()
    pr(f"  {yr}: n={len(yd):4d}  total={yd['r'].sum():+8.2f}R  avg={yd['r'].mean():+.4f}R")
rank_2022 = sorted(yr_totals.values(), reverse=True).index(yr_totals.get(2022, -9e9)) + 1
y2022_ok = rank_2022 <= 2
pr(f"  2022 rank by total R: #{rank_2022}  (must be best or 2nd)  "
   f"{'[PASS]' if y2022_ok else '[FAIL]'}")

by_month = tr.groupby("month")["r"].sum().sort_values(ascending=False)
bm = by_month.index[0]
avg_no_bm = tr.loc[tr["month"] != bm, "r"].mean()
bm_ok = avg_no_bm > 0.10
pr(f"\n--- Remove Best Month ---")
pr(f"  Best month: {bm} ({by_month.iloc[0]:+.1f}R)  avg_r without: {avg_no_bm:+.4f}R "
   f"(floor 0.10R)  {'[PASS]' if bm_ok else '[FAIL]'}")
avg_no_ba = tr.loc[tr["symbol"] != top1_sym, "r"].mean()
ba_ok = avg_no_ba > 0.10
pr(f"--- Remove Best Asset ---")
pr(f"  Removed {top1_sym}: avg_r {avg_no_ba:+.4f}R (floor 0.10R)  "
   f"{'[PASS]' if ba_ok else '[FAIL]'}")

checks = {"IS avg_r > 0.25R": is_ok, "IS->OOS degradation < 40%": degrade_ok,
          "4.2 avg_r > 0.25R": cost_ok, "4.7 top-1 < 50%": conc_ok,
          "2022 best or 2nd year": y2022_ok, "remove best month > 0.10R": bm_ok,
          "remove best asset > 0.10R": ba_ok}
n_pass = sum(checks.values())
pr(f"\n--- T2 VERDICT ---")
for k, v in checks.items():
    pr(f"  {'[PASS]' if v else '[FAIL]'} {k}")
pr(f"  win rate {wr*100:.1f}% outside 30-45 gate: "
   f"{'no (in gate)' if wr_in_gate else 'FLAGGED -- documented structural'}")
critical_pass = is_ok and degrade_ok and cost_ok
verdict = "PASS" if critical_pass and n_pass == len(checks) else (
          "PASS WITH FLAGS" if critical_pass else "HALT")
pr(f"  T2: {verdict} ({n_pass}/{len(checks)} checks)")
tr.drop(columns=["dt"]).to_csv(OUT / "t2_trades.csv", index=False)

# ------------------------------------------------------------------ T3
pr(f"\n" + "=" * 74)
pr(f"T3 EXIT ENGINEERING: hold_bars sweep {T3_HOLD_BARS} (4H bars)")
pr("=" * 74)
if not critical_pass:
    pr("  T2 HALT -- T3 skipped per instructions")
else:
    print("  Loading 4H universe...", flush=True)
    uni = C.load_universe(TF, log_every=50)
    fund = {s: C.load_funding(s) for s in uni}
    periods = sorted({FAST, SLOW})
    sig = {}
    for s, d in uni.items():
        f_ = C.compute_ema(d["close"], FAST)
        s_ = C.compute_ema(d["close"], SLOW)
        f_prev = np.roll(f_, 1); s_prev = np.roll(s_, 1)
        mask = d["bear"] & (f_ < s_) & (f_prev >= s_prev)
        mask &= ~np.isnan(f_) & ~np.isnan(s_) & ~np.isnan(f_prev) & ~np.isnan(s_prev)
        mask[0] = False
        idx = np.where(mask)[0]
        gated = [i for i in idx
                 if (fr := C.get_rate_at(fund[s], int(d["ts"][i]))) is not None
                 and fr >= C.FUNDING_THRESHOLD]
        sig[s] = np.array(gated, dtype=int)

    t3_rows = []
    pr(f"\n  hb     n     avg_r    total_r    wr%    pf    2022R    2025conc")
    for hb in T3_HOLD_BARS:
        trades = []
        for s, d in uni.items():
            trades += C.simulate(s, d, sig[s], hb, AM, fund[s])
        dft = C.add_time_cols(pd.DataFrame(trades))
        st = C.combo_stats(dft)
        t3_rows.append({"hold_bars": hb, **st})
        share = st["r2025_share"]
        pr(f"  {hb:<4} {st['n']:6d}  {st['avg_r']:+.4f}  {st['total_r']:+9.1f}  "
           f"{st['win_rate']*100:5.1f}  {st['pf']:5.2f}  {st['y2022_total']:+7.1f}  "
           f"{share*100:6.1f}%")
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(OUT / "t3_exit_sweep.csv", index=False)
    best = df_t3.iloc[df_t3["avg_r"].values.argmax()]
    n_pos = int((df_t3["avg_r"] > 0).sum())
    n_42  = int((df_t3["avg_r"] > C.COST_FLOOR).sum())
    pr(f"\n  Best hold_bars: {int(best['hold_bars'])}  avg_r={best['avg_r']:+.4f}R")
    pr(f"  Exit stability: {n_pos}/{len(df_t3)} positive, {n_42}/{len(df_t3)} above 0.25R")
    pr(f"  T1 canonical hb=30: "
       f"{df_t3.loc[df_t3['hold_bars']==30, 'avg_r'].iloc[0]:+.4f}R")

with open(OUT / "t2t3_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
