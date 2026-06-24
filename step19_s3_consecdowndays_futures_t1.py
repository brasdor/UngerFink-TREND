"""
Step 19 -- T1: S3 ConsecDownDays on Binance Futures 290-symbol universe
IS period: 2020-01-01 to 2022-12-31. OOS 2023-2026 sealed.
Exchange: Binance Futures USD-M. LONG only.

Entry: 5 consecutive down closes AND close > EMA200 -> enter long at next open
Exit:  Fixed time exit after 20 bars. Safety stop: ATR x 2.0 below entry.
EMA200 filter RETAINED (Step 5 confirmed removing it = -39.5R in 2022).
Funding: actual per-symbol 8h rates deducted from net_r.
"""
import bisect
import numpy as np
import pandas as pd
from pathlib import Path

DATA_1D     = Path("data/futures_universe/ohlcv_1d")
FUNDING_DIR = Path("data/futures_universe/funding_rates")
OUT_DIR     = Path("data/research_s3_futures_290sym_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2020-01-01")
IS_END   = pd.Timestamp("2022-12-31 23:59:59")

CONSEC_N     = 5
HOLD_BARS    = 20
ATR_MULT     = 2.0
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

def count_consec_down(cl):
    """Count consecutive down closes ending at each bar."""
    out = np.zeros(len(cl), dtype=int)
    for i in range(1, len(cl)):
        if cl[i] < cl[i-1]:
            out[i] = out[i-1] + 1
        else:
            out[i] = 0
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
def run_backtest():
    all_trades = []
    for sym, (df, fp) in sym_data.items():
        cl = df["close"].values.astype(float)
        hi = df["high"].values.astype(float)
        lo = df["low"].values.astype(float)
        ts_arr = df["timestamp"].values.astype(np.int64)
        times = df["time"].values

        ema200 = compute_ema(cl, 200)
        atr = compute_atr(hi, lo, cl)
        consec = count_consec_down(cl)

        in_pos = False; e_price = 0; stop = 0; e_bar = 0; e_ts_ms = 0; e_ts = None
        for i in range(1, len(df)):
            if np.isnan(ema200[i]) or np.isnan(atr[i]): continue
            t = pd.Timestamp(times[i])
            if t < IS_START or t > IS_END: continue

            if not in_pos:
                if consec[i] >= CONSEC_N and cl[i] > ema200[i]:
                    in_pos = True
                    e_price = cl[i]; e_bar = i; e_ts = times[i]
                    e_ts_ms = int(ts_arr[i])
                    stop = e_price - ATR_MULT * atr[i]
            else:
                bars_held = i - e_bar
                ep = reason = None
                if lo[i] <= stop:
                    ep = stop; reason = "atr_stop"
                elif bars_held >= HOLD_BARS:
                    ep = cl[i]; reason = "time_exit"

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
    return all_trades

trades = run_backtest()
print(f"Total trades: {len(trades)}")

# ── Report ───────────────────────────────────────────────────────────
lines = []
def pr(s=""): print(s); lines.append(s)

pr("="*70)
pr("STEP 19 -- T1 REPORT: S3 CONSEC DOWN DAYS ON FUTURES (290-SYMBOL)")
pr("="*70)
pr(f"\nIS: {IS_START.date()} to {IS_END.date()}")
pr(f"Universe: {len(sym_data)} Futures symbols")
pr(f"Config: {CONSEC_N} consecutive down closes, EMA200 filter, "
   f"hold {HOLD_BARS} bars, ATR x{ATR_MULT}")
pr(f"Funding: actual per-symbol 8h rates deducted")
pr(f"Cost floor: {COST_FLOOR_R}R (Futures long)")
pr()

if not trades:
    pr("NO TRADES GENERATED -- FAIL")
    pr("="*70)
    (OUT_DIR / "step19_t1_report.txt").write_text("\n".join(lines), encoding="utf-8")
    exit()

rs_raw = np.array([t["raw_r"] for t in trades])
rs_net = np.array([t["net_r"] for t in trades])
fc_arr = np.array([t["funding_cost_r"] for t in trades])

n = len(trades)
avg_r_raw = rs_raw.mean()
avg_r_net = rs_net.mean()
avg_fc = fc_arr.mean()
total_r_raw = rs_raw.sum()
total_r_net = rs_net.sum()

wins_net = rs_net[rs_net > 0]
losses_net = rs_net[rs_net <= 0]
wr = len(wins_net) / n
pf = wins_net.sum() / abs(losses_net.sum()) if len(losses_net) > 0 and losses_net.sum() != 0 else float('inf')
std_r = rs_net.std() if n > 1 else 1
t_stat = avg_r_net / (std_r / np.sqrt(n)) if std_r > 0 else 0

pr("OVERALL RESULTS:")
pr(f"  Trades:          {n}")
pr(f"  avg_r (raw):     {avg_r_raw:+.4f}R  (before funding costs)")
pr(f"  avg_r (net):     {avg_r_net:+.4f}R  (after funding costs)")
pr(f"  avg funding cost: {avg_fc:+.4f}R per trade")
if avg_r_raw != 0:
    pr(f"  Funding degradation: {(avg_r_raw - avg_r_net) / avg_r_raw * 100:.1f}% of raw avg_r")
pr(f"  total_r (net):   {total_r_net:+.1f}R")
pr(f"  Win rate:        {wr*100:.1f}%")
pr(f"  PF:              {pf:.2f}")
pr(f"  t-stat:          {t_stat:.2f}")
pr()

pr("YEAR-BY-YEAR BREAKDOWN (critical -- watch 2021 funding costs):")
pr(f"  {'Year':>6} {'Trades':>7} {'avg_r_raw':>10} {'avg_r_net':>10} {'avg_fc':>8} {'total_r_net':>12}")
for yr in range(2020, 2023):
    yr_t = [t for t in trades if t["year"] == yr]
    if not yr_t:
        pr(f"  {yr:>6} {'--':>7}")
        continue
    yr_raw = np.array([t["raw_r"] for t in yr_t])
    yr_net = np.array([t["net_r"] for t in yr_t])
    yr_fc = np.array([t["funding_cost_r"] for t in yr_t])
    pr(f"  {yr:>6} {len(yr_t):>7} {yr_raw.mean():>+9.4f}R {yr_net.mean():>+9.4f}R "
       f"{yr_fc.mean():>+7.4f}R {yr_net.sum():>+11.1f}R")
pr()

pr("2021 FUNDING COST DETAIL (S3 is a bull specialist -- most trades here):")
yr2021 = [t for t in trades if t["year"] == 2021]
if yr2021:
    fc_2021 = np.array([t["funding_cost_r"] for t in yr2021])
    raw_2021 = np.array([t["raw_r"] for t in yr2021])
    net_2021 = np.array([t["net_r"] for t in yr2021])
    pr(f"  2021 trades: {len(yr2021)}")
    pr(f"  2021 avg funding cost: {fc_2021.mean():+.4f}R")
    pr(f"  2021 max funding cost: {fc_2021.max():+.4f}R")
    pr(f"  2021 avg_r raw:        {raw_2021.mean():+.4f}R")
    pr(f"  2021 avg_r net:        {net_2021.mean():+.4f}R")
    if raw_2021.mean() != 0:
        pct_deg = (raw_2021.mean() - net_2021.mean()) / raw_2021.mean() * 100
        pr(f"  2021 funding degradation: {pct_deg:.1f}%")
        if pct_deg > 30:
            pr(f"  ** WARNING: 2021 funding degrades avg_r by >{30}% -- structural concern **")
else:
    pr("  No 2021 trades")
pr()

pr("2022 PERFORMANCE (EMA200 filter should keep system mostly inactive):")
yr2022 = [t for t in trades if t["year"] == 2022]
if yr2022:
    net_2022 = np.array([t["net_r"] for t in yr2022])
    pr(f"  2022 trades: {len(yr2022)}")
    pr(f"  2022 total R (net): {net_2022.sum():+.1f}R")
    pr(f"  2022 gate: {'PASS' if net_2022.sum() > 0 else 'ACCEPTABLE' if net_2022.sum() > -5 else 'FAIL'}")
else:
    pr("  2022: 0 trades (EMA200 filter active -- expected)")
    pr("  2022 gate: PASS (no activity in bear)")
pr()

# Concentration check
sym_r = {}
for t in trades:
    sym_r[t["symbol"]] = sym_r.get(t["symbol"], 0) + t["net_r"]
sorted_syms = sorted(sym_r.items(), key=lambda x: x[1], reverse=True)

pr("TOP 10 SYMBOLS BY TOTAL NET R:")
for s, r in sorted_syms[:10]:
    n_s = sum(1 for t in trades if t["symbol"] == s)
    pr(f"  {s:<20} {r:>+8.1f}R  ({n_s} trades)")
if sorted_syms and total_r_net != 0:
    top1_pct = abs(sorted_syms[0][1]) / abs(total_r_net) * 100
    pr(f"  Top-1 concentration: {top1_pct:.1f}%  [{'FAIL >50%' if top1_pct > 50 else 'PASS'}]")
pr()

# Stability -- single config, check if avg_r zone is "stable" by proxy
# Since frozen config has no grid, stability = does it pass gates
stability_pass = avg_r_net > COST_FLOOR_R
yr2022_total = sum(t["net_r"] for t in trades if t["year"] == 2022)

pr("="*70)
pr("GATE CHECKS:")
pr(f"  avg_r > {COST_FLOOR_R}R:    {avg_r_net:+.4f}R  "
   f"[{'PASS' if avg_r_net > COST_FLOOR_R else 'FAIL'}]")
pr(f"  2022 positive:     {yr2022_total:+.1f}R  "
   f"[{'PASS' if yr2022_total >= 0 else 'MARGINAL' if yr2022_total > -5 else 'FAIL'}]")
pr()

if avg_r_net > COST_FLOOR_R:
    pr(f"  RESULT: PASS -- S3 can migrate to Futures")
    pr(f"  DECISION: S3 can run on Futures (avg_r {avg_r_net:+.4f}R > {COST_FLOOR_R}R)")
elif avg_r_net > 0.15:
    pr(f"  RESULT: MARGINAL -- S3 avg_r between 0.15R and {COST_FLOOR_R}R on Futures")
    pr(f"  DECISION: S3 should stay on Spot")
else:
    pr(f"  RESULT: FAIL -- S3 avg_r {avg_r_net:+.4f}R below floor")
    pr(f"  DECISION: S3 stays on Spot (FAIL on Futures)")
pr()

pr("COMPARISON TO S3 SPOT:")
pr(f"  S3 Spot: avg_r=+0.470R, CAGR=+4.03% (52-symbol universe)")
pr(f"  S3 Futures: avg_r={avg_r_net:+.4f}R ({len(sym_data)}-symbol universe)")
pr("="*70)

pd.DataFrame(trades).to_csv(OUT_DIR / "step19_t1_trades.csv", index=False)
(OUT_DIR / "step19_t1_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved to: {OUT_DIR / 'step19_t1_report.txt'}")
