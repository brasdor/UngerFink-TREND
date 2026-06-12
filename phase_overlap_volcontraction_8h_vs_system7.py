"""
TRACK 1 -- VolContraction 8H vs System 7 (4H) trade overlap analysis.

System 7 trades: data/research_volcontraction_funding_t2_v2/trades_t2.csv
  (frozen config cb=15 ap=20 hb=30 am=2.0, 4H -> position window = entry + 120h)
8H candidate:    data/research_volcontraction_8h_funding_t1/t1_trades.csv
  filtered to best combo cb=15 ap=20 hb=15 am=2.0 (window = entry + 120h)

For each 8H trade: overlap = System 7 had a position on the same symbol whose
[entry - 2 days, exit + 2 days] window contains the 8H entry.
Exit times approximated as entry + hold_bars (stops can only shorten the
window; the +/-2 day buffer absorbs that).

Daily P&L correlation: trade R attributed to ENTRY date (exit ts not stored
in T1 trade files) -- documented approximation; also reported weekly.

Decision rule: overlap > 60% -> same system, skip T2
               overlap < 40% -> genuinely different, proceed to T2

Output: data/research_volcontraction_8h_overlap/
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/research_volcontraction_8h_overlap")
OUT.mkdir(parents=True, exist_ok=True)

S7_CSV = Path("data/research_volcontraction_funding_t2_v2/trades_t2.csv")
H8_CSV = Path("data/research_volcontraction_8h_funding_t1/t1_trades.csv")

H4_MS = 4 * 3600 * 1000
H8_MS = 8 * 3600 * 1000
DAY_MS = 24 * 3600 * 1000
BUFFER_MS = 2 * DAY_MS

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

s7 = pd.read_csv(S7_CSV)
s7["exit_ts"] = s7["entry_ts"] + 30 * H4_MS   # hb=30 4H bars

h8 = pd.read_csv(H8_CSV)
h8 = h8[(h8["cb"] == 15) & (h8["ap"] == 20) & (h8["hb"] == 15) & (h8["am"] == 2.0)].copy()
h8["exit_ts"] = h8["entry_ts"] + 15 * H8_MS   # hb=15 8H bars
h8["dt"] = pd.to_datetime(h8["entry_ts"], unit="ms", utc=True)
h8["year"] = h8["dt"].dt.year

pr("=" * 74)
pr("OVERLAP ANALYSIS: VolContraction 8H (T1 best) vs System 7 4H (frozen)")
pr("=" * 74)
pr(f"System 7 trades: {len(s7)}   8H trades: {len(h8)}")
pr(f"Symbols: S7={s7['symbol'].nunique()}  8H={h8['symbol'].nunique()}")

# build per-symbol S7 interval lists
s7_by_sym = {}
for sym, grp in s7.groupby("symbol"):
    s7_by_sym[sym] = list(zip(grp["entry_ts"].values - BUFFER_MS,
                              grp["exit_ts"].values + BUFFER_MS))

def overlaps(sym, ts):
    for lo, hi in s7_by_sym.get(sym, []):
        if lo <= ts <= hi:
            return True
    return False

h8["overlap"] = [overlaps(r.symbol, r.entry_ts) for r in h8.itertuples()]
n_ov = int(h8["overlap"].sum())
pct_ov = n_ov / len(h8)

pr(f"\n--- Overlap (8H entry within S7 position window +/- 2 days, same symbol) ---")
pr(f"  Overlapping 8H trades: {n_ov}/{len(h8)} = {pct_ov*100:.1f}%")

uniq = h8[~h8["overlap"]]
ovl  = h8[h8["overlap"]]
pr(f"\n--- Unique 8H trades (System 7 would NOT have been in the market) ---")
pr(f"  n={len(uniq)}  avg_r={uniq['r'].mean():+.4f}R  total={uniq['r'].sum():+.1f}R  "
   f"wr={(uniq['r']>0).mean()*100:.1f}%")
pr(f"  Overlapping subset: n={len(ovl)}  avg_r={ovl['r'].mean():+.4f}R")
pr(f"  Unique trades vs 4.2 floor (0.25R): "
   f"{'[PASS]' if uniq['r'].mean() > 0.25 else '[FAIL]'}")
pr(f"  Unique year-by-year:")
for yr in sorted(uniq["year"].unique()):
    yd = uniq[uniq["year"] == yr]
    pr(f"    {yr}: n={len(yd):4d}  total={yd['r'].sum():+8.2f}R  avg={yd['r'].mean():+.4f}R")

# daily P&L correlation (entry-date attribution)
s7["date"] = pd.to_datetime(s7["entry_ts"], unit="ms", utc=True).dt.date
h8["date"] = h8["dt"].dt.date
d7 = s7.groupby("date")["r"].sum()
d8 = h8.groupby("date")["r"].sum()
idx = pd.date_range(min(d7.index.min(), d8.index.min()),
                    max(d7.index.max(), d8.index.max()), freq="D").date
a = pd.Series(d7, index=idx).fillna(0.0)
b = pd.Series(d8, index=idx).fillna(0.0)
corr_d = a.corr(b)
aw = a.groupby(pd.to_datetime(a.index).to_period("W")).sum()
bw = b.groupby(pd.to_datetime(b.index).to_period("W")).sum()
corr_w = aw.corr(bw)
pr(f"\n--- P&L correlation (R attributed to entry date) ---")
pr(f"  Daily:  {corr_d:+.3f}")
pr(f"  Weekly: {corr_w:+.3f}")

pr(f"\n--- RECOMMENDATION ---")
if pct_ov > 0.60:
    rec = "SAME SYSTEM -- overlap > 60%. Skip T2. 8H is System 7 on coarser bars."
elif pct_ov < 0.40:
    rec = "GENUINELY DIFFERENT -- overlap < 40%. Proceed to T2."
else:
    rec = ("GRAY ZONE (40-60% overlap). Judgement call: review unique-trade "
           "economics and P&L correlation above.")
pr(f"  Overlap = {pct_ov*100:.1f}% -> {rec}")

h8.drop(columns=["dt"]).to_csv(OUT / "trades_8h_with_overlap_flag.csv", index=False)
with open(OUT / "overlap_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
