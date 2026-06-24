"""
Step 17 -- T1 Concept Discovery: S1 DonchianLong on Binance Futures 290-symbol universe
IS period: 2020-01-01 to 2022-12-31. OOS 2023-2026 sealed.
Exchange: Binance Futures USD-M. LONG only.

Entry: close > EMA200 AND close == Donchian_high(N) -> enter long at next open
Exit:  Donchian_low(N//2) OR Chandelier (ATR x 5.0 from HH, activation +4R)
Stop:  ATR x atr_mult below entry
Funding: actual per-symbol 8h rates deducted from net_r
"""
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_1D     = Path("data/futures_universe/ohlcv_1d")
FUNDING_DIR = Path("data/futures_universe/funding_rates")
OUT_DIR     = Path("data/research_donchian_futures_290sym_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2020-01-01")
IS_END   = pd.Timestamp("2022-12-31 23:59:59")

DONCHIAN_NS = [15, 20, 25, 30]
ATR_MULTS   = [1.5, 2.0, 2.5]
CHANDELIER_MULT = 5.0
CHANDELIER_ACTIVATE_R = 4.0
COST_FLOOR_R = 0.25

# ── Helpers ──────────────────────────────────────────────────────────
def load_funding(sym):
    p = FUNDING_DIR / f"{sym}_funding.csv"
    if not p.exists(): return None
    try:
        df = pd.read_csv(p)
        return sorted(zip(df["funding_time"].values.astype(np.int64),
                          df["funding_rate"].values.astype(float)))
    except: return None

def get_funding_cost_r(pairs, entry_ts_ms, hold_days, entry_price, risk):
    if not pairs or risk <= 0: return 0.0
    times = [p[0] for p in pairs]
    rates = [p[1] for p in pairs]
    period_8h_ms = 8 * 3600 * 1000
    total_cost_pct = 0.0
    n_payments = hold_days * 3
    for k in range(n_payments):
        ts = entry_ts_ms + k * period_8h_ms
        idx = bisect.bisect_right(times, ts) - 1
        if idx >= 0:
            total_cost_pct += rates[idx]
    return total_cost_pct * entry_price / risk

def compute_ema(close, n):
    ema = np.full(len(close), np.nan)
    if len(close) < n: return ema
    ema[n-1] = close[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, len(close)):
        ema[i] = ema[i-1] * (1 - alpha) + close[i] * alpha
    return ema

def compute_atr(hi, lo, cl, n=14):
    nb = len(cl)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = np.nanmean(tr[1:n+1])
        for i in range(n+1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i-1]):
                atr[i] = (atr[i-1]*(n-1)+tr[i])/n
    return atr

def donchian_high(hi, n):
    """Donchian upper band: highest high of previous N bars (shifted, no lookahead)."""
    out = np.full(len(hi), np.nan)
    for i in range(n, len(hi)):
        out[i] = np.max(hi[i-n:i])
    return out

def donchian_low(lo, n):
    """Donchian lower band: lowest low of previous N bars (shifted, no lookahead)."""
    out = np.full(len(lo), np.nan)
    for i in range(n, len(lo)):
        out[i] = np.min(lo[i-n:i])
    return out

# ── Load universe ────────────────────────────────────────────────────
files = sorted(DATA_1D.glob("*_1d.csv"))
symbols = [f.stem.replace("_1d", "") for f in files]
print(f"Universe: {len(symbols)} Futures symbols")

sym_data = {}
warmup = pd.Timedelta(days=250)
for sym in symbols:
    df = pd.read_csv(DATA_1D / f"{sym}_1d.csv")
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[(df["time"] >= IS_START - warmup) & (df["time"] <= IS_END)]
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 50: continue
    fp = load_funding(sym)
    if fp is None: continue
    sym_data[sym] = (df, fp)

print(f"Loaded: {len(sym_data)} symbols with IS data + funding")

# ── Backtest ─────────────────────────────────────────────────────────
def run_backtest(entry_n, atr_mult):
    exit_n = max(entry_n // 2, 5)
    all_trades = []
    for sym, (df, fp) in sym_data.items():
        cl = df["close"].values.astype(float)
        hi = df["high"].values.astype(float)
        lo = df["low"].values.astype(float)
        ts_arr = df["timestamp"].values.astype(np.int64)
        times = df["time"].values

        ema200 = compute_ema(cl, 200)
        atr = compute_atr(hi, lo, cl)
        dc_hi = donchian_high(hi, entry_n)
        dc_lo = donchian_low(lo, exit_n)

        in_pos = False
        e_price = 0; stop = 0; e_bar = 0; e_ts_ms = 0; e_ts = None
        highest_high = 0; chandelier_active = False

        for i in range(1, len(df)):
            if np.isnan(ema200[i]) or np.isnan(atr[i]): continue
            if np.isnan(dc_hi[i]) or np.isnan(dc_lo[i]): continue
            t = pd.Timestamp(times[i])
            if t < IS_START or t > IS_END: continue

            if not in_pos:
                if cl[i] > ema200[i] and cl[i] >= dc_hi[i]:
                    in_pos = True
                    e_price = cl[i]; e_bar = i; e_ts = times[i]
                    e_ts_ms = int(ts_arr[i])
                    stop = e_price - atr_mult * atr[i]
                    highest_high = hi[i]
                    chandelier_active = False
            else:
                bars_held = i - e_bar
                if hi[i] > highest_high:
                    highest_high = hi[i]

                risk = e_price - (e_price - atr_mult * atr[e_bar])
                if risk > 1e-9:
                    mfe_r = (highest_high - e_price) / risk
                    if mfe_r >= CHANDELIER_ACTIVATE_R:
                        chandelier_active = True

                chandelier_stop = highest_high - CHANDELIER_MULT * atr[i] if chandelier_active else -1e18
                effective_stop = max(stop, chandelier_stop)

                ep = reason = None
                if lo[i] <= effective_stop:
                    ep = effective_stop; reason = "stop"
                elif cl[i] <= dc_lo[i]:
                    ep = cl[i]; reason = "donchian_exit"

                if reason:
                    risk = e_price - stop
                    if risk > 1e-9:
                        raw_r = (ep - e_price) / risk
                        hold_days = bars_held
                        fc_r = get_funding_cost_r(fp, e_ts_ms, hold_days, e_price, risk)
                        net_r = raw_r - fc_r
                        entry_dt = pd.Timestamp(e_ts)
                        all_trades.append({
                            "symbol": sym, "raw_r": raw_r, "funding_cost_r": fc_r,
                            "net_r": net_r, "year": entry_dt.year,
                            "exit_reason": reason, "bars_held": bars_held,
                        })
                    in_pos = False
                    highest_high = 0; chandelier_active = False
    return all_trades

def compute_metrics(trades):
    if not trades:
        return {"n":0,"avg_r":0,"avg_r_raw":0,"avg_fc":0,"total_r":0,
                "sharpe":0,"win_rate":0,"pf":0,"y2022_r":0,"y2022_n":0}
    rs = np.array([t["net_r"] for t in trades])
    rs_raw = np.array([t["raw_r"] for t in trades])
    fc = np.array([t["funding_cost_r"] for t in trades])
    n=len(rs); avg_r=rs.mean(); total_r=rs.sum()
    std_r=rs.std() if n>1 else 1
    sharpe=avg_r/std_r*np.sqrt(n) if std_r>0 else 0
    wins=rs[rs>0]; losses=rs[rs<=0]
    wr=len(wins)/n if n>0 else 0
    pf=wins.sum()/abs(losses.sum()) if len(losses)>0 and losses.sum()!=0 else float('inf')
    y2022=[t["net_r"] for t in trades if t["year"]==2022]
    return {"n":n,"avg_r":avg_r,"avg_r_raw":rs_raw.mean(),"avg_fc":fc.mean(),
            "total_r":total_r,"sharpe":sharpe,"win_rate":wr,"pf":pf,
            "y2022_r":sum(y2022),"y2022_n":len(y2022)}

# ── Grid search ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 17 -- T1: S1 DONCHIAN LONG ON FUTURES (290-SYMBOL)")
print(f"IS: {IS_START.date()} to {IS_END.date()}")
print("="*70)

grid = list(product(DONCHIAN_NS, ATR_MULTS))
print(f"Grid: {len(grid)} combinations\n")

results = []
for idx, (dn, am) in enumerate(grid, 1):
    trades = run_backtest(dn, am)
    m = compute_metrics(trades)
    results.append({"donchian_n":dn, "atr_mult":am, **m})
    print(f"  [{idx}/{len(grid)}] N={dn} ATR={am} n={m['n']} "
          f"avg_r={m['avg_r']:+.4f}R raw={m['avg_r_raw']:+.4f}R "
          f"fc={m['avg_fc']:+.4f}R 2022={m['y2022_r']:+.1f}R")

results_df = pd.DataFrame(results)
results_df = results_df[results_df["n"] > 0].copy()

# Stability check
def check_stability(row, df):
    zone = df[
        (df["donchian_n"].between(row["donchian_n"]-5, row["donchian_n"]+5)) &
        (df["atr_mult"].between(row["atr_mult"]-0.5, row["atr_mult"]+0.5))
    ]
    if len(zone)==0: return False, 0
    pct = (zone["avg_r"]>0).sum()/len(zone)
    return pct>=0.67, pct

zp=[]; zpct=[]
for _,r in results_df.iterrows():
    p,pct = check_stability(r, results_df)
    zp.append(p); zpct.append(pct)
results_df["zone_pass"]=zp; results_df["zone_pct"]=zpct

results_df["gate_avgr"] = results_df["avg_r"] > COST_FLOOR_R
results_df["gate_2022"] = results_df["y2022_r"] > -20.0
results_df["all_gates"] = results_df["zone_pass"] & results_df["gate_avgr"] & results_df["gate_2022"]

results_df.to_csv(OUT_DIR / "step17_t1_stability_grid.csv", index=False)

passing = results_df[results_df["all_gates"]]
if len(passing) > 0:
    canonical = passing.sort_values("avg_r", ascending=False).iloc[0]
    gate_result = "PASS"
else:
    canonical = results_df.sort_values("avg_r", ascending=False).iloc[0]
    gate_result = "FAIL"

# Run canonical for detailed trades
can_trades = run_backtest(int(canonical["donchian_n"]), canonical["atr_mult"])

# ── Report ───────────────────────────────────────────────────────────
lines = []
def pr(s=""): print(s); lines.append(s)

pr("="*70)
pr("STEP 17 -- T1 REPORT: S1 DONCHIAN LONG ON FUTURES (290-SYMBOL)")
pr("="*70)
pr(f"\nIS: {IS_START.date()} to {IS_END.date()}")
pr(f"Universe: {len(sym_data)} Futures symbols with funding data")
pr(f"Entry: close > EMA200 AND close >= Donchian_high(N)")
pr(f"Exit: Donchian_low(N//2) OR Chandelier (ATR x{CHANDELIER_MULT}, activate +{CHANDELIER_ACTIVATE_R}R)")
pr(f"Funding: actual per-symbol 8h rates deducted")
pr(f"Cost floor: {COST_FLOOR_R}R (Futures long)")
pr(f"Grid: {len(results_df)} valid combinations")
pr(f"Passing all gates: {len(passing)}")
pr()

pr("TOP 10 BY avg_r (net of funding):")
pr(f"  {'N':>4} {'am':>5} {'n':>6} {'avg_r':>8} {'raw':>8} {'fc':>7} {'2022':>7} "
   f"{'zone':>5} {'gates':>6}")
for _, r in results_df.sort_values("avg_r", ascending=False).head(10).iterrows():
    pr(f"  {r['donchian_n']:>4.0f} {r['atr_mult']:>5.1f} {r['n']:>6.0f} "
       f"{r['avg_r']:>+7.4f}R {r['avg_r_raw']:>+7.4f}R {r['avg_fc']:>+6.4f}R "
       f"{r['y2022_r']:>+6.1f}R {r['zone_pct']*100:>4.0f}% "
       f"{'PASS' if r['all_gates'] else 'FAIL':>6}")
pr()

pr("CANONICAL CONFIG:")
pr(f"  donchian_n={int(canonical['donchian_n'])}  atr_mult={canonical['atr_mult']}")
pr()

can_m = compute_metrics(can_trades)
pr("CANONICAL RESULTS:")
pr(f"  Trades:    {can_m['n']}")
pr(f"  avg_r:     {can_m['avg_r']:+.4f}R  (net of funding)")
pr(f"  avg_r raw: {can_m['avg_r_raw']:+.4f}R  (before funding)")
pr(f"  avg fc:    {can_m['avg_fc']:+.4f}R per trade")
pr(f"  total_r:   {can_m['total_r']:+.1f}R")
pr(f"  Win rate:  {can_m['win_rate']*100:.1f}%")
pr(f"  PF:        {can_m['pf']:.2f}")
pr(f"  Sharpe:    {can_m['sharpe']:.2f}")
pr()

pr("2022 PERFORMANCE (KEY TEST -- Step 2 failed at -35.1R on 20 symbols):")
pr(f"  2022 total R (net): {can_m['y2022_r']:+.1f}R  ({can_m['y2022_n']} trades)")
if can_m['y2022_r'] < -20:
    pr(f"  2022 gate: FAIL (< -20R -- same failure mode as Step 2)")
elif can_m['y2022_r'] > 0:
    pr(f"  2022 gate: STRONG PASS (positive in bear market)")
else:
    pr(f"  2022 gate: PASS (> -20R threshold)")
pr()

pr("  Year-by-year:")
for yr in range(2020, 2023):
    yr_t = [t for t in can_trades if t["year"]==yr]
    if not yr_t:
        pr(f"    {yr}: -- no trades")
        continue
    yr_raw = np.array([t["raw_r"] for t in yr_t])
    yr_net = np.array([t["net_r"] for t in yr_t])
    yr_fc = np.array([t["funding_cost_r"] for t in yr_t])
    pr(f"    {yr}: {len(yr_t):>4} trades  raw={yr_raw.sum():>+7.1f}R  "
       f"net={yr_net.sum():>+7.1f}R  fc={yr_fc.sum():>+6.1f}R")
pr()

pr("FUNDING COST ANALYSIS:")
pr(f"  Average funding cost per trade: {can_m['avg_fc']:+.4f}R")
if can_m['avg_r_raw'] != 0:
    pr(f"  Funding degradation: {(can_m['avg_r_raw'] - can_m['avg_r']) / can_m['avg_r_raw'] * 100:.1f}% of raw avg_r")
pr(f"  avg_r WITHOUT funding: {can_m['avg_r_raw']:+.4f}R")
pr(f"  avg_r WITH funding:    {can_m['avg_r']:+.4f}R")
pr()

# Concentration check
sym_r = {}
for t in can_trades:
    sym_r[t["symbol"]] = sym_r.get(t["symbol"], 0) + t["net_r"]
sorted_syms = sorted(sym_r.items(), key=lambda x: x[1], reverse=True)
total_abs = sum(abs(v) for v in sym_r.values()) if sym_r else 1

pr("CONCENTRATION CHECK (top 5 symbols by total net R):")
for s, r in sorted_syms[:5]:
    n_s = sum(1 for t in can_trades if t["symbol"] == s)
    pct = abs(r) / total_abs * 100 if total_abs > 0 else 0
    pr(f"  {s:<20} {r:>+8.1f}R  ({n_s} trades, {pct:.1f}% of |total R|)")
top1_pct = abs(sorted_syms[0][1]) / can_m['total_r'] * 100 if sorted_syms and can_m['total_r'] != 0 else 0
pr(f"  Top-1 concentration: {top1_pct:.1f}%  [{'FAIL >50%' if top1_pct > 50 else 'PASS'}]")
pr()

pr("="*70)
pr("GATE CHECKS:")
pr(f"  avg_r > {COST_FLOOR_R}R:     {can_m['avg_r']:+.4f}R  "
   f"[{'PASS' if can_m['avg_r'] > COST_FLOOR_R else 'FAIL'}]")
pr(f"  2022 > -20R:        {can_m['y2022_r']:+.1f}R  "
   f"[{'PASS' if can_m['y2022_r'] > -20 else 'FAIL'}]")
pr(f"  Stability >= 67%:   {canonical['zone_pct']*100:.0f}%  "
   f"[{'PASS' if canonical['zone_pass'] else 'FAIL'}]")
pr(f"\n  GATE DECISION: {gate_result}")
pr()
pr("COMPARISON TO S1 SPOT:")
pr(f"  S1 Spot: avg_r=+0.9263R, CAGR=+14.3%, Max DD=-3.4% (24-symbol universe)")
pr(f"  S1 Futures: avg_r={can_m['avg_r']:+.4f}R ({len(sym_data)}-symbol universe)")
pr("="*70)

results_df.to_csv(OUT_DIR / "step17_t1_stability_grid.csv", index=False)
pd.DataFrame(can_trades).to_csv(OUT_DIR / "step17_t1_canonical_trades.csv", index=False)
(OUT_DIR / "step17_t1_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved to: {OUT_DIR / 'step17_t1_report.txt'}")
