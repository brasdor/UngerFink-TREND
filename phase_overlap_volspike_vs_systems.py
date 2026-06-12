"""
TRACK 2 -- Volume Spike Short 4H overlap vs System 7 AND MA Cross.

Volume Spike best combo: tf=4h lb=15 hb=30 am=2.0 (window = entry + 120h)
System 7:  trades_t2.csv (hb=30 4H) -- window = entry + 120h
MA Cross:  t2_trades.csv (hb=30 4H) -- window = entry + 120h
Overlap: VS entry within [other_entry - 2d, other_exit + 2d], same symbol.

Decision: overlap vs either > 60% -> same edge family, skip T2
          overlap vs either < 40% -> independent, proceed to T2
          unique avg_r > 0.25R -> worth pursuing even with partial overlap
Output: data/research_volspike_overlap/
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/research_volspike_overlap")
OUT.mkdir(parents=True, exist_ok=True)

H4_MS = 4 * 3600 * 1000
DAY_MS = 24 * 3600 * 1000
BUF = 2 * DAY_MS

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

vs = pd.read_csv("data/research_volspike_proxy_t1/t1_trades.csv")
vs = vs[(vs["tf"] == "4h") & (vs["lookback_n"] == 15) &
        (vs["hold_bars"] == 30) & (vs["atr_mult"] == 2.0)].copy()
vs["dt"] = pd.to_datetime(vs["entry_ts"], unit="ms", utc=True)
vs["year"] = vs["dt"].dt.year

s7 = pd.read_csv("data/research_volcontraction_funding_t2_v2/trades_t2.csv")
s7["exit_ts"] = s7["entry_ts"] + 30 * H4_MS
mc = pd.read_csv("data/research_macross_short_t2/t2_trades.csv")
mc["exit_ts"] = mc["entry_ts"] + 30 * H4_MS

def windows_by_sym(df):
    out = {}
    for sym, grp in df.groupby("symbol"):
        out[sym] = list(zip(grp["entry_ts"].values - BUF, grp["exit_ts"].values + BUF))
    return out

w7 = windows_by_sym(s7)
wm = windows_by_sym(mc)

def hit(wins, sym, ts):
    for lo, hi in wins.get(sym, []):
        if lo <= ts <= hi:
            return True
    return False

vs["ov_s7"] = [hit(w7, t.symbol, t.entry_ts) for t in vs.itertuples()]
vs["ov_mc"] = [hit(wm, t.symbol, t.entry_ts) for t in vs.itertuples()]
vs["ov_any"] = vs["ov_s7"] | vs["ov_mc"]

pr("=" * 74)
pr("OVERLAP: Volume Spike Short 4H (proxy T1 best) vs System 7 + MA Cross")
pr("=" * 74)
pr(f"Volume Spike trades: {len(vs)}   System 7: {len(s7)}   MA Cross: {len(mc)}")
p7 = vs["ov_s7"].mean(); pm = vs["ov_mc"].mean(); pa = vs["ov_any"].mean()
pr(f"\n  Overlap vs System 7:        {int(vs['ov_s7'].sum()):4d}/{len(vs)} = {p7*100:.1f}%")
pr(f"  Overlap vs MA Cross:        {int(vs['ov_mc'].sum()):4d}/{len(vs)} = {pm*100:.1f}%")
pr(f"  Overlap vs EITHER:          {int(vs['ov_any'].sum()):4d}/{len(vs)} = {pa*100:.1f}%")

uniq = vs[~vs["ov_any"]]
pr(f"\n--- Unique trades (not captured by either system) ---")
pr(f"  n={len(uniq)}  avg_r={uniq['r'].mean():+.4f}R  total={uniq['r'].sum():+.1f}R  "
   f"wr={(uniq['r']>0).mean()*100:.1f}%")
pr(f"  Unique avg_r > 0.25R: {'[PASS]' if uniq['r'].mean() > 0.25 else '[FAIL]'}")
pr(f"  Unique year-by-year:")
for yr in sorted(uniq["year"].unique()):
    yd = uniq[uniq["year"] == yr]
    pr(f"    {yr}: n={len(yd):4d}  total={yd['r'].sum():+8.2f}R  avg={yd['r'].mean():+.4f}R")

# weekly P&L correlation (entry-date attribution)
def weekly(df):
    s = df.copy()
    s["date"] = pd.to_datetime(s["entry_ts"], unit="ms", utc=True)
    return s.set_index("date")["r"].resample("W").sum()

w_vs, w_s7, w_mc = weekly(vs), weekly(s7), weekly(mc)
idx = w_vs.index.union(w_s7.index).union(w_mc.index)
a = w_vs.reindex(idx).fillna(0); b = w_s7.reindex(idx).fillna(0); c = w_mc.reindex(idx).fillna(0)
pr(f"\n--- Weekly P&L correlation (entry-date attribution) ---")
pr(f"  vs System 7: {a.corr(b):+.3f}")
pr(f"  vs MA Cross: {a.corr(c):+.3f}")

pr(f"\n--- DECISION ---")
worst = max(p7, pm)
uniq_strong = uniq["r"].mean() > 0.25
if uniq_strong:
    rec = (f"UNIQUE TRADES STRONG (avg_r {uniq['r'].mean():+.3f}R > 0.25R) -- worth "
           f"pursuing even with partial overlap ({pa*100:.0f}% vs either).")
elif worst > 0.60:
    rec = f"SAME EDGE FAMILY (overlap {worst*100:.0f}% > 60%) -- skip T2."
elif worst < 0.40:
    rec = f"GENUINELY INDEPENDENT (max overlap {worst*100:.0f}% < 40%) -- proceed to T2."
else:
    rec = (f"GRAY ZONE (max overlap {worst*100:.0f}%, unique avg_r "
           f"{uniq['r'].mean():+.3f}R) -- judgement call.")
pr(f"  {rec}")

vs.drop(columns=["dt"]).to_csv(OUT / "trades_volspike_with_overlap_flags.csv", index=False)
with open(OUT / "overlap_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
