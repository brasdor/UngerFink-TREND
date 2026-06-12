"""
ACTION 2 -- Overlap analysis: System 7 (VolContraction 4H) vs System 8 (MACross 4H).

System 7 trades: data/research_volcontraction_funding_t2_v2/trades_t2.csv
                 (frozen config, hold 30 x 4H bars)
System 8 trades: data/research_macross_short_v2_t2/t2_trades.csv
                 (frozen config, exit_ts recorded)

Reports:
  - Overlap %: same symbol, S8 entry within S7 position window +/- 2 days
    (and the reverse direction)
  - Weekly P&L correlation
  - Average and PEAK concurrent open positions (both systems combined,
    plus same-symbol simultaneous holdings)
  - Recommendation: independent vs shared heat limit

Output: data/research_s7_s8_overlap/
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/research_s7_s8_overlap")
OUT.mkdir(parents=True, exist_ok=True)

H4_MS = 4 * 3600 * 1000
DAY_MS = 24 * 3600 * 1000
BUF = 2 * DAY_MS

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

s7 = pd.read_csv("data/research_volcontraction_funding_t2_v2/trades_t2.csv")
s7["exit_ts"] = s7["entry_ts"] + 30 * H4_MS
s8 = pd.read_csv("data/research_macross_short_v2_t2/t2_trades.csv")
# s8 has real exit_ts

pr("=" * 74)
pr("OVERLAP: System 7 (VolContraction 4H) vs System 8 (MACross 4H)")
pr("=" * 74)
pr(f"System 7 trades: {len(s7)} ({s7['symbol'].nunique()} symbols)")
pr(f"System 8 trades: {len(s8)} ({s8['symbol'].nunique()} symbols)")

def windows_by_sym(df):
    out = {}
    for sym, grp in df.groupby("symbol"):
        out[sym] = list(zip(grp["entry_ts"].values - BUF, grp["exit_ts"].values + BUF))
    return out

def hit(wins, sym, ts):
    for lo, hi in wins.get(sym, []):
        if lo <= ts <= hi:
            return True
    return False

w7 = windows_by_sym(s7)
w8 = windows_by_sym(s8)
s8["ov"] = [hit(w7, t.symbol, t.entry_ts) for t in s8.itertuples()]
s7["ov"] = [hit(w8, t.symbol, t.entry_ts) for t in s7.itertuples()]

pr(f"\n--- Trade overlap (same symbol, within position window +/- 2 days) ---")
pr(f"  S8 trades overlapping an S7 position: {int(s8['ov'].sum()):4d}/{len(s8)} = "
   f"{s8['ov'].mean()*100:.1f}%")
pr(f"  S7 trades overlapping an S8 position: {int(s7['ov'].sum()):4d}/{len(s7)} = "
   f"{s7['ov'].mean()*100:.1f}%")
u8 = s8[~s8["ov"]]
pr(f"  S8 unique trades: n={len(u8)}  avg_r={u8['r'].mean():+.4f}R "
   f"(overlapping subset: {s8.loc[s8['ov'], 'r'].mean():+.4f}R)")

# weekly P&L correlation (entry-date attribution)
def weekly(df):
    s = df.copy()
    s["date"] = pd.to_datetime(s["entry_ts"], unit="ms", utc=True)
    return s.set_index("date")["r"].resample("W").sum()

a, b = weekly(s7), weekly(s8)
idx = a.index.union(b.index)
corr_w = a.reindex(idx).fillna(0).corr(b.reindex(idx).fillna(0))
pr(f"\n--- Weekly P&L correlation: {corr_w:+.3f} ---")

# concurrent positions on a 4H grid
t0 = int(min(s7["entry_ts"].min(), s8["entry_ts"].min()))
t1 = int(max(s7["exit_ts"].max(), s8["exit_ts"].max()))
grid = np.arange(t0, t1 + H4_MS, H4_MS)

def open_count(df, grid):
    starts = np.sort(df["entry_ts"].values)
    ends = np.sort(df["exit_ts"].values)
    return (np.searchsorted(starts, grid, side="right")
            - np.searchsorted(ends, grid, side="right"))

n7 = open_count(s7, grid)
n8 = open_count(s8, grid)
comb = n7 + n8
active = comb > 0
pr(f"\n--- Concurrent open positions (4H grid, {len(grid)} points) ---")
pr(f"  S7 alone:    avg {n7.mean():.1f}   peak {n7.max()}")
pr(f"  S8 alone:    avg {n8.mean():.1f}   peak {n8.max()}")
pr(f"  COMBINED:    avg {comb.mean():.1f}   avg-when-active {comb[active].mean():.1f}   "
   f"PEAK {comb.max()}")
both_open = (n7 > 0) & (n8 > 0)
pr(f"  Both systems open simultaneously: {both_open.mean()*100:.1f}% of all bars")

# same-symbol simultaneous holdings
def intervals_by_sym(df):
    out = {}
    for sym, grp in df.groupby("symbol"):
        out[sym] = list(zip(grp["entry_ts"].values, grp["exit_ts"].values))
    return out

i7, i8 = intervals_by_sym(s7), intervals_by_sym(s8)
shared_ms = 0
for sym in set(i7) & set(i8):
    for a0, a1 in i7[sym]:
        for b0, b1 in i8[sym]:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                shared_ms += hi - lo
pr(f"  Same-symbol simultaneous holding time: {shared_ms/DAY_MS:.0f} symbol-days total")

# heat implications at 0.25% risk, $150 ceiling
peak_combined = int(comb.max())
pr(f"\n--- Heat implications (paper: $25-150 risk per position) ---")
pr(f"  Peak combined positions: {peak_combined}")
pr(f"  Worst-case combined heat at $25/pos (start): ${peak_combined*25:,.0f}")
pr(f"  Worst-case combined heat at $150/pos (ceiling): ${peak_combined*150:,.0f}")
pr(f"  Proposed shared cap: $1,000 (5% of $20k Futures pool)")
pr(f"  -> cap binds at {int(1000/25)} positions ($25 risk) / "
   f"{int(1000/150)} positions ($150 risk)")

pr(f"\n--- RECOMMENDATION ---")
ov_max = max(s8["ov"].mean(), s7["ov"].mean())
pr(f"  Trade overlap {ov_max*100:.0f}% and weekly P&L correlation {corr_w:+.2f}:")
pr(f"  The systems are correlated short-side harvesters of the same funding-")
pr(f"  gated regime. They can run as separate engines (different entry logic,")
pr(f"  validated independently) BUT they MUST share a heat limit -- peak")
pr(f"  combined exposure of {peak_combined} positions would otherwise breach any")
pr(f"  sane Futures pool risk budget in a clustered selloff.")
pr(f"  IMPLEMENTED: Rule 6 in signal_arbitrator.py -- S7+S8 combined open")
pr(f"  risk <= $1,000 (5% of $20k pool). Cross-system dedup (same symbol)")
pr(f"  already enforced via t9b_shared + arbitrator Rules 1/2.")

with open(OUT / "overlap_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT}/", flush=True)
