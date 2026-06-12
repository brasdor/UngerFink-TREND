"""
ACTION 1 (part 2) -- MA Cross Short V2: T5 portfolio filter -> T6 capital
execution -> T7 asset robustness -> T8 config freeze.
Runs only if data/research_macross_short_v2_t2/verdict.txt == PROCEED.
Process identical to System 7 (T5 caps incl. uncapped; T6 $10k, 0.25% risk,
$150 ceiling, 35% kill switch; T7 remove top assets; T8 freeze + scorecard).
Outputs: data/research_macross_short_v2_t5t6t7/, data/research_macross_short_v2_t8/
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

VERDICT = Path("data/research_macross_short_v2_t2/verdict.txt").read_text().strip()
if VERDICT != "PROCEED":
    print(f"T2/T4 verdict is {VERDICT} -- aborting T5-T8", flush=True)
    raise SystemExit(1)

OUT567 = Path("data/research_macross_short_v2_t5t6t7"); OUT567.mkdir(parents=True, exist_ok=True)
OUT8 = Path("data/research_macross_short_v2_t8"); OUT8.mkdir(parents=True, exist_ok=True)

CAPS = [("uncapped", None), ("max3", 3), ("max5", 5), ("max8", 8), ("max10", 10)]
INITIAL_CAPITAL = 10_000.0
RISK_PCT = 0.0025
CEILING = 150.0
KILL_DD = 0.35

tr = pd.read_csv("data/research_macross_short_v2_t2/t2_trades.csv")
tr = tr.sort_values("entry_ts").reset_index(drop=True)

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

def pf_of(x):
    w = x[x > 0].sum(); l = abs(x[x <= 0].sum())
    return w / l if l > 0 else np.inf

def max_dd_r(x):
    eq = np.cumsum(x); return (eq - np.maximum.accumulate(eq)).min()

# ------------------------------------------------------------------ T5
pr("=" * 74)
pr("T5 PORTFOLIO FILTER: MA Cross Short V2 (fast=20 slow=30 hb=35 am=2.0)")
pr("=" * 74)
pr(f"  variant   trades  accept%   totalR    avgR     pf     ddR")
t5 = {}
for name, cap in CAPS:
    if cap is None:
        acc = tr
    else:
        open_exits = []
        keep = []
        for t in tr.itertuples():
            open_exits = [e for e in open_exits if e > t.entry_ts]
            if len(open_exits) < cap:
                keep.append(t.Index)
                open_exits.append(t.exit_ts)
        acc = tr.loc[keep]
    x = acc["r"].values
    t5[name] = acc
    pr(f"  {name:<9} {len(acc):6d}  {len(acc)/len(tr)*100:6.1f}%  {x.sum():+8.1f}  "
       f"{x.mean():+.4f}  {pf_of(x):5.2f}  {max_dd_r(x):+7.1f}")
    acc.to_csv(OUT567 / f"t5_trades_{name}.csv", index=False)
pr(f"\n  Canonical (System 7 precedent): UNCAPPED -- caps cut total R; avg_r")
pr(f"  uplift from caps does not compensate (same crowding-event profile)")
CANON_VARIANT = "uncapped"
canon = t5[CANON_VARIANT]

# ------------------------------------------------------------------ T6
def run_t6(df):
    eq = INITIAL_CAPITAL; peak = eq; max_dd = 0.0; killed = False
    curve = []
    for t in df.itertuples():
        risk = min(eq * RISK_PCT, CEILING)
        eq += t.r * risk
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
        if dd <= -KILL_DD:
            killed = True
        curve.append({"entry_ts": t.entry_ts, "equity": eq})
    return eq, max_dd, killed, pd.DataFrame(curve)

pr(f"\n" + "=" * 74)
pr(f"T6 CAPITAL EXECUTION: ${INITIAL_CAPITAL:,.0f}, risk {RISK_PCT*100}%, "
   f"ceiling ${CEILING:.0f}, kill switch -{KILL_DD*100:.0f}%")
pr("=" * 74)
t6_rows = []
for name, _ in CAPS:
    eq, mdd, killed, curve = run_t6(t5[name])
    yrs = (pd.to_datetime(t5[name]["exit_ts"].max(), unit="ms") -
           pd.to_datetime(t5[name]["entry_ts"].min(), unit="ms")).days / 365.25
    cagr = (eq / INITIAL_CAPITAL) ** (1 / yrs) - 1
    t6_rows.append({"variant": name, "final_eq": eq, "return_pct": eq/INITIAL_CAPITAL*100-100,
                    "cagr": cagr, "max_dd_pct": mdd*100, "killed": killed})
    pr(f"  {name:<9} final=${eq:>10,.0f}  return={eq/INITIAL_CAPITAL*100-100:+8.1f}%  "
       f"CAGR={cagr*100:+6.1f}%  maxDD={mdd*100:+6.2f}%  kill={'YES' if killed else 'no'}")
    if name == CANON_VARIANT:
        curve.to_csv(OUT567 / "t6_equity_canonical.csv", index=False)
pd.DataFrame(t6_rows).to_csv(OUT567 / "t6_variant_summary.csv", index=False)
canon_t6 = [x for x in t6_rows if x["variant"] == CANON_VARIANT][0]
kill_ok = not canon_t6["killed"]
pr(f"\n  Kill switch (canonical): {'[PASS] never fired' if kill_ok else '[FAIL] FIRED'}")

# ------------------------------------------------------------------ T7
pr(f"\n" + "=" * 74)
pr("T7 ASSET ROBUSTNESS (canonical variant, full T6 re-sim)")
pr("=" * 74)
by_asset = canon.groupby("symbol")["r"].sum().sort_values(ascending=False)
t7_ok = True
for k in [1, 3, 5]:
    excl = set(by_asset.index[:k])
    sub = canon[~canon["symbol"].isin(excl)]
    eq, mdd, killed, _ = run_t6(sub)
    profitable = eq > INITIAL_CAPITAL
    pr(f"  remove top-{k}: final=${eq:>10,.0f}  return={eq/INITIAL_CAPITAL*100-100:+8.1f}%  "
       f"maxDD={mdd*100:+6.2f}%  {'[PASS]' if profitable else '[FAIL]'}")
    if k == 1:
        t7_ok = profitable

# ------------------------------------------------------------------ T8
r = canon["r"].values
n = len(r)
wr = (r > 0).mean()
t_score = r.mean() / (r.std(ddof=1) / np.sqrt(n))
all_ok = kill_ok and t7_ok
pr(f"\n" + "=" * 74)
pr("T8 CONFIG FREEZE" if all_ok else "T8 HALTED -- gate failure above")
pr("=" * 74)

if all_ok:
    frozen = f'''# FROZEN CONFIG -- DO NOT MODIFY AFTER T8
# Method  : MACrossShort_FundingGate (System 8)
# Frozen  : {datetime.now(timezone.utc).date()}
# Research: T1 sweep -> T2 -> T3 -> T4 (8/9) -> T15 -> re-canonicalized
#           -> T2v2 (7/7) -> T3v2 -> T4v2 (8/9) -> T5 -> T6 -> T7 -> T8
# CAVEAT  : 2026 partial -31.6R (monitor in T9B); win rate 47.2% above 45%
#           gate -- structural (funding gate selects crowded shorts, S7=45.6%)

METHOD             = "MACrossShort_FundingGate"
TIMEFRAME          = "4H"
EXCHANGE           = "binance_futures_usdm"
SIDE               = "SHORT"
LEVERAGE           = 1.0

FAST_EMA           = 20        # fast EMA crosses below slow EMA on bar close
SLOW_EMA           = 30
EMA_N              = 200       # close must be BELOW EMA200
FUNDING_THRESHOLD  = 0.0001    # 0.01% per 8h -- discovery constraint (System 7 class)
ATR_N              = 14
ATR_STOP_MULT      = 2.0       # stop = ATR x 2.0 ABOVE entry
HOLD_BARS          = 35        # time exit after 35 x 4H bars (~5.8 calendar days)

PORTFOLIO_CAP      = None      # uncapped (System 7 precedent -- caps cut total R)
RISK_PER_TRADE_PCT = 0.0025
CAPITAL_CEILING    = 150.0
INITIAL_CAPITAL    = 10_000.0
KILL_SWITCH_DD_PCT = 35.0

UNIVERSE_SIZE      = {canon["symbol"].nunique()}
DATA_PATH_4H       = "data/futures_universe/ohlcv_4h/"
DATA_PATH_FUNDING  = "data/futures_universe/funding_rates/"

PERF_TRADES        = {n}
PERF_AVG_R         = {r.mean():.4f}
PERF_WIN_RATE      = {wr:.4f}
PERF_PROFIT_FACTOR = {pf_of(r):.4f}
PERF_T_STAT        = {t_score:.1f}
PERF_TOTAL_R       = {r.sum():.1f}
PERF_CAGR          = {canon_t6["cagr"]:.4f}
PERF_MAX_DD        = {canon_t6["max_dd_pct"]/100:.4f}
PERF_FINAL_EQ      = {canon_t6["final_eq"]:.2f}
'''
    with open(OUT8 / "phase_t8_frozen_config.py", "w", encoding="ascii") as f:
        f.write(frozen)
    pr(f"  Frozen config written: {OUT8}/phase_t8_frozen_config.py")

pr(f"\n--- T8 FINAL SCORECARD ---")
pr(f"  METHOD:           MACrossShort_FundingGate (System 8)")
pr(f"  CONFIG:           4H / fast=20 slow=30 / hb=35 / ATRx2.0 / EMA200 bear / funding>=0.01%")
pr(f"  TRADES:           {n}")
pr(f"  AVG R:            {r.mean():+.4f}R   [gate >0.25R]")
pr(f"  WIN RATE:         {wr*100:.1f}%   [FLAG above 45% -- structural]")
pr(f"  PROFIT FACTOR:    {pf_of(r):.3f}")
pr(f"  T-SCORE:          {t_score:.1f}")
pr(f"  TOTAL R:          {r.sum():+.1f}R")
pr(f"  CAGR:             {canon_t6['cagr']*100:+.1f}%")
pr(f"  MAX DD:           {canon_t6['max_dd_pct']:+.2f}%")
pr(f"  FINAL EQUITY:     ${canon_t6['final_eq']:,.0f} (from $10,000)")
pr(f"  KILL SWITCH:      {'never fired' if kill_ok else 'FIRED'}")
pr(f"  REMOVE TOP-1:     {'PASS' if t7_ok else 'FAIL'}")
pr(f"  STATUS:           {'FROZEN -- ready for T9B paper trading' if all_ok else 'HALTED'}")

pr(f"\n--- FINAL COMPARISON ---")
pr(f"  metric        System 7      MACross v1    MACross V2 (System 8)")
pr(f"  avg_r         +0.304R       +0.255R       {r.mean():+.3f}R")
pr(f"  trades        3214          2910          {n}")
pr(f"  t-score       10.3          9.2           {t_score:.1f}")
pr(f"  win rate      45.6%         49.6%         {wr*100:.1f}%")
pr(f"  CAGR          +41.7%        --            {canon_t6['cagr']*100:+.1f}%")
pr(f"  max DD        -18.8%        --            {canon_t6['max_dd_pct']:+.1f}%")
pr(f"  last 500      +0.092R       +0.292R       {r[-500:].mean():+.3f}R")

with open(OUT8 / "phase_t8_final_scorecard.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
with open(OUT567 / "t5t6t7_report.txt", "w", encoding="ascii", errors="replace") as f:
    f.write("\n".join(lines))
print(f"\nSaved to {OUT567}/ and {OUT8}/", flush=True)
