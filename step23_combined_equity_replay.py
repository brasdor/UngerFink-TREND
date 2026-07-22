"""
Step 23 -- Combined Portfolio Historical Equity Replay (Prompt K16)

Performance attribution -- NOT a new validation step. Uses already-validated
T8 trade files and frozen configs. No system parameters are modified.

KEY DIFFERENCE FROM STEP 20:
  Step 20 compounded each trade independently on a single curve without
  normalizing for concurrent positions -> $600M+ unrealistic absolute result.
  Step 23 runs all 8 systems sharing the same $60k equity base:
    - risk_amount = system_alloc x regime_weight x 0.0025 (fixed dollar risk
      per system allocation, scaled by regime multiplier at entry)
    - portfolio equity updates ONLY on trade close events
    - per-system dollar P&L attribution by year

REGIME WEIGHTS:
  regime_allocator_v4 logic (hysteresis + EMA smoothing + 5-day min hold),
  computed walk-forward day by day -- at date T only data[<= T] is used.
  SCHEME_C table uses the CORRECTED S8 weights from engines/regime_daily.py
  (commit cb1afe5): the S8=0.00 placeholders in the older step20 script
  predated the S8 engine and starved the bear amplifier in BEAR regimes.
  regime_weight applied to sizing = w_regime(sys) / w_static_dollar(sys),
  i.e. the relative multiplier vs the static Scheme C dollar split
  (Section B semantics: "BEAR: S6/S7 get 1.4x, S1 gets 0.35x").
"""
import warnings; warnings.filterwarnings("ignore")
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

SIM_START = pd.Timestamp("2019-01-01")
SIM_END   = pd.Timestamp("2026-06-19")
IS_END    = pd.Timestamp("2022-12-31")

STARTING_CAPITAL = 60000.0
RISK_PER_TRADE   = 0.0025

DATA_1D = Path("data/futures_universe/ohlcv_1d")
DB_PATH = Path("data/futures.db")
OUT_DIR = Path("data/research_combined_equity_replay")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scheme C dollar allocations (Section B / K16)
SYSTEM_ALLOC = {
    "S1": 8000.0,   # DonchianLong (Spot)
    "S2": 4800.0,   # RSI MR Futures (40% of $12k MR pool)
    "S3": 10000.0,  # ConsecDownDays (Spot)
    "S5": 8000.0,   # Momentum Factor (Futures)
    "S6": 11000.0,  # VolContraction Short (Futures)
    "S7": 11000.0,  # MACross Short (Futures)
    "S8": 7200.0,   # RSI MR + funding gate (60% of $12k MR pool)
}
STATIC_FRAC = {k: v / STARTING_CAPITAL for k, v in SYSTEM_ALLOC.items()}

# Corrected Scheme C regime weights (engines/regime_daily.py, commit cb1afe5)
SCHEME_C = {
    "BULL":  {"S1": 0.25, "S2": 0.06, "S3": 0.15, "S5": 0.25,
              "S6": 0.10, "S7": 0.10, "S8": 0.09},
    "BEAR":  {"S1": 0.05, "S2": 0.10, "S3": 0.05, "S5": 0.15,
              "S6": 0.25, "S7": 0.25, "S8": 0.15},
    "MIXED": {"S1": 0.125, "S2": 0.125, "S3": 0.125, "S5": 0.125,
              "S6": 0.125, "S7": 0.125, "S8": 0.125},
}

# v4 anti-oscillation parameters (identical to regime_allocator_v4 / step20)
FUNDING_ENTER_HIGH = 0.00025
FUNDING_EXIT_HIGH  = 0.00010
FUNDING_ENTER_LOW  = 0.00005
FUNDING_EXIT_LOW   = 0.00010
BREADTH_ENTER_BULL = 58
BREADTH_EXIT_BULL  = 48
BREADTH_ENTER_BEAR = 42
BREADTH_EXIT_BEAR  = 48
MIN_HOLD_DAYS      = 5

SYSTEMS = ["S1", "S2", "S3", "S5", "S6", "S7", "S8"]

# ── Load trade files (entry AND exit timestamps required) ────────────
print("Loading trade files...")

def load_trades_s1():
    df = pd.read_csv("data/research_donchianlong_universev2/phase_c_t5_portfolio_trades.csv")
    df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_localize(None)
    df["exit_dt"]  = pd.to_datetime(df["exit_time"], utc=True).dt.tz_localize(None)
    df["r"] = df["net_r"]
    df["system"] = "S1"
    return df[["system", "entry_dt", "exit_dt", "r"]].copy()

def load_trades_s2():
    df = pd.read_csv("data/research_s2_futures_costcheck/s2_futures_trades.csv")
    df["entry_dt"] = pd.to_datetime(df["entry_time"])
    # No exit_time column: 1D crypto bars trade 24/7, so bars_held == calendar days
    assert (df["bars_held"] > 0).all() and (df["bars_held"] <= 25).all(), \
        "S2 bars_held out of expected range (frozen config: 20-bar time exit) -- check exit reconstruction"
    df["exit_dt"] = df["entry_dt"] + pd.to_timedelta(df["bars_held"], unit="D")
    df["r"] = df["net_r"]
    df["system"] = "S2"
    return df[["system", "entry_dt", "exit_dt", "r"]].copy()

def load_trades_s3():
    df = pd.read_csv("data/research_consecdowndays_mr_t6/phase_t6_equity_trades.csv")
    df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_localize(None)
    df["exit_dt"]  = pd.to_datetime(df["exit_time"], utc=True).dt.tz_localize(None)
    df["r"] = df["net_r"]
    df["system"] = "S3"
    return df[["system", "entry_dt", "exit_dt", "r"]].copy()

def _load_ts_file(path, system):
    df = pd.read_csv(path)
    df["entry_dt"] = pd.to_datetime(df["entry_ts"], unit="ms", utc=True).dt.tz_localize(None)
    df["exit_dt"]  = pd.to_datetime(df["exit_ts"], unit="ms", utc=True).dt.tz_localize(None)
    df["system"] = system
    return df[["system", "entry_dt", "exit_dt", "r"]].copy()

all_trades = pd.concat([
    load_trades_s1(), load_trades_s2(), load_trades_s3(),
    _load_ts_file("data/research_momentum_factor_t8/s5_regenerated_trades.csv", "S5"),
    _load_ts_file("data/research_volcontraction_funding_t8/s6_canonical_trades.csv", "S6"),
    _load_ts_file("data/research_macross_short_v2_t5t6t7/t5_trades_uncapped.csv", "S7"),
    _load_ts_file("data/research_rsi_mr_futures_fundinggate_t5t6t7t8/step4_t8_frozen_trades.csv", "S8"),
], ignore_index=True)

nan_count = all_trades["r"].isna().sum()
if nan_count > 0:
    print(f"  Dropped {nan_count} trades with NaN R-multiples")
    all_trades = all_trades.dropna(subset=["r"]).reset_index(drop=True)

# Restrict to simulation window on ENTRY date. Trades still open at SIM_END
# cannot credit P&L inside the window -- exclude them but report the count.
in_window = (all_trades["entry_dt"] >= SIM_START) & (all_trades["entry_dt"] <= SIM_END)
open_at_end = in_window & (all_trades["exit_dt"] > SIM_END)
if open_at_end.sum() > 0:
    print(f"  NOTE: {open_at_end.sum()} trades still open at {SIM_END.date()} excluded "
          f"({all_trades.loc[open_at_end, 'system'].value_counts().to_dict()})")
all_trades = all_trades[in_window & ~open_at_end].copy()

print(f"Total trades in simulation window: {len(all_trades)}")
for s in SYSTEMS:
    sub = all_trades[all_trades["system"] == s]
    print(f"  {s}: {len(sub):>5d} trades  ({sub['entry_dt'].min().date()} .. {sub['exit_dt'].max().date()})")

# ── Precompute regime inputs (walk-forward safe) ─────────────────────
print("\nPrecomputing regime data...")

btc = pd.read_csv(DATA_1D / "BTCUSDT_1d.csv")
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
btc = btc.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
btc_close = btc["close"].astype(float)
btc_ema200 = btc_close.ewm(span=200, adjust=False).mean()

print("  Computing breadth panel...")
all_closes = {}
for f in sorted(DATA_1D.glob("*_1d.csv")):
    sym = f.stem.replace("_1d", "")
    d = pd.read_csv(f, usecols=["timestamp", "close"])
    d["date"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    d = d.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    all_closes[sym] = d["close"].astype(float)
print(f"  Breadth universe: {len(all_closes)} symbols")
assert len(all_closes) >= 250, \
    f"Breadth universe only {len(all_closes)} symbols -- expected ~290 (finding #11: verify universe completeness)"

close_panel = pd.DataFrame(all_closes).sort_index()
ema200_panel = close_panel.ewm(span=200, min_periods=20, adjust=False).mean()
above_ema = (close_panel > ema200_panel).sum(axis=1)
total_syms = close_panel.notna().sum(axis=1)
breadth = (above_ema / total_syms * 100).fillna(50)

print("  Loading funding from futures.db...")
conn = sqlite3.connect(str(DB_PATH))
funding_db = pd.read_sql("SELECT timestamp, symbol, funding_rate FROM funding_history_coinglass", conn)
oi_db = pd.read_sql("SELECT timestamp, symbol, oi_usd FROM oi_history WHERE interval='1d'", conn)
conn.close()

funding_db["date"] = pd.to_datetime(funding_db["timestamp"], unit="ms").dt.normalize()
oi_db["date"] = pd.to_datetime(oi_db["timestamp"], unit="ms").dt.normalize()

monthly_top20 = {}
for m in pd.date_range("2019-01-01", SIM_END, freq="MS"):
    month_oi = oi_db[(oi_db["date"] >= m) & (oi_db["date"] < m + pd.DateOffset(months=1))]
    if len(month_oi) > 0:
        monthly_top20[m] = set(month_oi.groupby("symbol")["oi_usd"].mean().nlargest(20).index)

fund_by_date = {d: g for d, g in funding_db.groupby("date")}
daily_funding = {}
for d in pd.date_range("2019-01-01", SIM_END, freq="D"):
    top20 = monthly_top20.get(d.replace(day=1), set())
    if not top20:
        daily_funding[d] = 0.0001
        continue
    day_fund = fund_by_date.get(d)
    if day_fund is None:
        daily_funding[d] = 0.0001
        continue
    sel = day_fund[day_fund["symbol"].isin(top20)]
    daily_funding[d] = sel["funding_rate"].mean() if len(sel) > 0 else 0.0001
funding_series = pd.Series(daily_funding).sort_index()

btc_hi, btc_lo = btc["high"].astype(float), btc["low"].astype(float)
tr = pd.concat([btc_hi - btc_lo, (btc_hi - btc_close.shift()).abs(),
                (btc_lo - btc_close.shift()).abs()], axis=1).max(axis=1)
btc_atr14 = tr.rolling(14).mean()
btc_atr_pct = btc_atr14.rolling(252).apply(lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False)

breadth_smoothed = breadth.ewm(span=3, adjust=False).mean()
funding_smoothed = funding_series.ewm(span=5, adjust=False).mean()

# ── Daily walk-forward regime series (chronological, stateful) ───────
# Each day's classification uses only inputs computed from data <= that day
# (rolling/EWM series are causal). Precomputing chronologically preserves
# the hysteresis + min-hold state machine ordering.
print("  Building daily regime series (walk-forward)...")
regime_by_day = {}
cur_trend, cur_funding = "MIXED", "NEUTRAL"
last_trend_sw = last_fund_sw = pd.Timestamp("2018-01-01")

for day in pd.date_range(SIM_START, SIM_END, freq="D"):
    avail = btc_close.index[btc_close.index <= day]
    if len(avail) > 0:
        d = avail[-1]
        btc_above = float(btc_close.loc[d]) > float(btc_ema200.loc[d])
        br = float(breadth_smoothed.get(d, 50))

        if (day - last_trend_sw).days >= MIN_HOLD_DAYS:
            new_trend = cur_trend
            if cur_trend != "BULL" and btc_above and br > BREADTH_ENTER_BULL:
                new_trend = "BULL"
            elif cur_trend == "BULL" and (not btc_above or br < BREADTH_EXIT_BULL):
                new_trend = "MIXED"
            elif cur_trend != "BEAR" and not btc_above and br < BREADTH_ENTER_BEAR:
                new_trend = "BEAR"
            elif cur_trend == "BEAR" and (btc_above or br > BREADTH_EXIT_BEAR):
                new_trend = "MIXED"
            if new_trend != cur_trend:
                last_trend_sw = day
                cur_trend = new_trend

    fr_avail = funding_smoothed.index[funding_smoothed.index <= day]
    fr = float(funding_smoothed.loc[fr_avail[-1]]) if len(fr_avail) > 0 else 0.0001
    if (day - last_fund_sw).days >= MIN_HOLD_DAYS:
        new_funding = cur_funding
        if cur_funding != "HIGH" and fr > FUNDING_ENTER_HIGH:
            new_funding = "HIGH"
        elif cur_funding == "HIGH" and fr < FUNDING_EXIT_HIGH:
            new_funding = "NEUTRAL"
        elif cur_funding != "LOW" and fr < FUNDING_ENTER_LOW:
            new_funding = "LOW"
        elif cur_funding == "LOW" and fr > FUNDING_EXIT_LOW:
            new_funding = "NEUTRAL"
        if new_funding != cur_funding:
            last_fund_sw = day
            cur_funding = new_funding

    vp_avail = btc_atr_pct.index[btc_atr_pct.index <= day]
    vp = float(btc_atr_pct.loc[vp_avail[-1]]) if len(vp_avail) > 0 else 50
    if np.isnan(vp): vp = 50
    vol_mult = 0.75 if vp < 30 else (1.25 if vp > 70 else 1.00)

    regime_by_day[day.normalize()] = (cur_trend, cur_funding, vol_mult)

def get_regime_weights(trend, funding_regime):
    base = dict(SCHEME_C[trend])
    if funding_regime == "HIGH":
        for s in ["S6", "S7"]: base[s] = base.get(s, 0) + 0.05
        for s in ["S1", "S5"]: base[s] = max(0, base.get(s, 0) - 0.05)
    elif funding_regime == "LOW":
        for s in ["S2", "S3"]: base[s] = base.get(s, 0) + 0.05
        for s in ["S6", "S7"]: base[s] = max(0, base.get(s, 0) - 0.05)
    total = sum(base.values())
    if total > 0:
        base = {k: v / total for k, v in base.items()}
    return base

print("  Regime data ready.")

# ── Equity replay: shared $60k base, P&L applied on trade CLOSE ──────
print("\nRunning combined equity replay...")

# Compute each trade's risk_amount at ENTRY (regime at entry date, no lookahead)
def regime_at(dt):
    key = pd.Timestamp(dt).normalize()
    if key in regime_by_day:
        return regime_by_day[key]
    prior = [k for k in regime_by_day if k <= key]
    return regime_by_day[max(prior)] if prior else ("MIXED", "NEUTRAL", 1.0)

trades = all_trades.sort_values("entry_dt").reset_index(drop=True)
risk_amounts, regime_mults, entry_regimes = [], [], []
for _, row in trades.iterrows():
    trend, funding_reg, vol_mult = regime_at(row["entry_dt"])
    w = get_regime_weights(trend, funding_reg)
    sysname = row["system"]
    regime_mult = (w.get(sysname, 0.0) / STATIC_FRAC[sysname]) * vol_mult
    risk_amounts.append(SYSTEM_ALLOC[sysname] * regime_mult * RISK_PER_TRADE)
    regime_mults.append(regime_mult)
    entry_regimes.append(trend)
trades["risk_amount"] = risk_amounts
trades["regime_mult"] = regime_mults
trades["entry_regime"] = entry_regimes
trades["pnl"] = trades["risk_amount"] * trades["r"]
trades["exit_year"] = trades["exit_dt"].dt.year

# Replay on exit events chronologically
events = trades.sort_values("exit_dt").reset_index(drop=True)
equity = STARTING_CAPITAL
eq_rows = []
per_sys_pnl = {s: {} for s in SYSTEMS}       # system -> year -> $
per_sys_cum = {s: 0.0 for s in SYSTEMS}      # running per-system cum P&L
per_sys_peak = {s: 0.0 for s in SYSTEMS}
per_sys_maxdd_frac = {s: 0.0 for s in SYSTEMS}  # vs system alloc
per_sys_trades = {s: {} for s in SYSTEMS}

for _, ev in events.iterrows():
    equity += ev["pnl"]
    s, yr = ev["system"], ev["exit_year"]
    per_sys_pnl[s][yr] = per_sys_pnl[s].get(yr, 0.0) + ev["pnl"]
    per_sys_trades[s][yr] = per_sys_trades[s].get(yr, 0) + 1
    per_sys_cum[s] += ev["pnl"]
    per_sys_peak[s] = max(per_sys_peak[s], per_sys_cum[s])
    dd = (per_sys_cum[s] - per_sys_peak[s]) / SYSTEM_ALLOC[s]
    per_sys_maxdd_frac[s] = min(per_sys_maxdd_frac[s], dd)
    eq_rows.append({"date": ev["exit_dt"].normalize(), "equity": equity})

# Daily equity curve (forward-filled between close events)
eq_df = pd.DataFrame(eq_rows).groupby("date")["equity"].last()
daily_idx = pd.date_range(SIM_START, SIM_END, freq="D")
daily_eq = eq_df.reindex(daily_idx).ffill().fillna(STARTING_CAPITAL)
peak = daily_eq.cummax()
daily_dd = (daily_eq - peak) / peak * 100

daily_out = pd.DataFrame({"date": daily_idx, "equity": daily_eq.values,
                          "drawdown_from_peak_pct": daily_dd.values})
daily_out.to_csv(OUT_DIR / "daily_equity_curve.csv", index=False)

# ── Annual summaries ─────────────────────────────────────────────────
years = list(range(2019, 2027))

# Dominant regime per year from daily regime series
dominant_regime = {}
reg_days = pd.Series({d: v[0] for d, v in regime_by_day.items()})
for yr in years:
    sub = reg_days[(reg_days.index.year == yr)]
    dominant_regime[yr] = sub.value_counts().idxmax() if len(sub) else "n/a"

annual_rows = []
for yr in years:
    yr_eq = daily_eq[daily_eq.index.year == yr]
    if len(yr_eq) == 0:
        continue
    start_eq = daily_eq[daily_eq.index < pd.Timestamp(f"{yr}-01-01")]
    start_eq = start_eq.iloc[-1] if len(start_eq) else STARTING_CAPITAL
    end_eq = yr_eq.iloc[-1]
    yr_peak = yr_eq.cummax()
    yr_dd = ((yr_eq - yr_peak) / yr_peak).min() * 100
    n_tr = int((events["exit_year"] == yr).sum())
    annual_rows.append({"year": yr, "start_eq": round(start_eq, 2),
                        "end_eq": round(end_eq, 2),
                        "return_pct": round((end_eq / start_eq - 1) * 100, 2),
                        "max_dd_pct": round(yr_dd, 2), "n_trades": n_tr,
                        "dominant_regime": dominant_regime.get(yr, "n/a")})
annual_df = pd.DataFrame(annual_rows)
annual_df.to_csv(OUT_DIR / "annual_summary.csv", index=False)

per_sys_rows = []
for s in SYSTEMS:
    for yr in years:
        pnl = per_sys_pnl[s].get(yr, 0.0)
        n = per_sys_trades[s].get(yr, 0)
        if n == 0 and pnl == 0.0:
            continue
        per_sys_rows.append({"system": s, "year": yr, "alloc": SYSTEM_ALLOC[s],
                             "pnl_usd": round(pnl, 2),
                             "return_on_alloc_pct": round(pnl / SYSTEM_ALLOC[s] * 100, 2),
                             "n_trades": n,
                             "notes": "partial yr" if yr == 2026 else ""})
per_sys_df = pd.DataFrame(per_sys_rows)
per_sys_df.to_csv(OUT_DIR / "per_system_annual_summary.csv", index=False)

# ── Period metrics ───────────────────────────────────────────────────
def period_metrics(start, end, label):
    eq = daily_eq[(daily_eq.index >= start) & (daily_eq.index <= end)]
    if len(eq) < 2:
        return None
    start_val = daily_eq[daily_eq.index < start]
    start_val = start_val.iloc[-1] if len(start_val) else STARTING_CAPITAL
    end_val = eq.iloc[-1]
    n_years = (end - start).days / 365.25
    cagr = ((end_val / start_val) ** (1 / n_years) - 1) * 100
    rets = eq.pct_change().dropna()  # all calendar days incl. flat holding days
    sharpe = rets.mean() / rets.std() * np.sqrt(365) if len(rets) > 1 and rets.std() > 0 else 0
    pk = eq.cummax()
    max_dd = ((eq - pk) / pk).min() * 100
    return {"label": label, "start_eq": start_val, "end_eq": end_val,
            "total_ret": (end_val / start_val - 1) * 100,
            "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}

m_is   = period_metrics(SIM_START, IS_END, "IS 2019-2022")
m_oos  = period_metrics(pd.Timestamp("2023-01-01"), SIM_END, "OOS 2023-2026")
m_full = period_metrics(SIM_START, SIM_END, "Full 2019-2026")

# ── Report ───────────────────────────────────────────────────────────
lines = []
def pr(s=""): print(s); lines.append(s)

pr("=" * 78)
pr("STEP 23 -- COMBINED PORTFOLIO HISTORICAL EQUITY REPLAY ($60k, 2019-2026)")
pr("=" * 78)
pr(f"Simulation: {SIM_START.date()} to {SIM_END.date()}  |  risk 0.25% of system alloc")
pr(f"Regime allocator: v4 logic, corrected S8 weights (regime_daily.py), walk-forward")
pr(f"Total trades replayed: {len(events)}")
pr()

pr("ANNUAL SUMMARY (combined portfolio):")
pr(f"  {'Year':<6}{'Start eq':>12}{'End eq':>12}{'Return':>9}{'MaxDD':>8}{'Trades':>8}  Regime")
pr("  " + "-" * 66)
for _, r in annual_df.iterrows():
    yl = f"{int(r['year'])}*" if r["year"] == 2026 else f"{int(r['year'])}"
    pr(f"  {yl:<6}{r['start_eq']:>12,.0f}{r['end_eq']:>12,.0f}"
       f"{r['return_pct']:>+8.1f}%{r['max_dd_pct']:>7.1f}%{r['n_trades']:>8d}  {r['dominant_regime']}")
pr()

pr("CROSS-SYSTEM YEAR TABLE (dollar P&L by exit year):")
hdr = "  Year  " + "".join(f"{s:>9}" for s in SYSTEMS) + f"{'Total':>10}"
pr(hdr)
pr("  " + "-" * (len(hdr) - 2))
for yr in years:
    vals = [per_sys_pnl[s].get(yr, 0.0) for s in SYSTEMS]
    if all(v == 0 for v in vals):
        continue
    yl = f"{yr}*" if yr == 2026 else str(yr)
    pr(f"  {yl:<6}" + "".join(f"{v:>+9,.0f}" for v in vals) + f"{sum(vals):>+10,.0f}")
pr()

pr("PER-SYSTEM MAX DRAWDOWN (cum P&L vs own allocation):")
for s in SYSTEMS:
    pr(f"  {s}: {per_sys_maxdd_frac[s] * 100:>+6.2f}%  (alloc ${SYSTEM_ALLOC[s]:,.0f}, "
       f"total P&L ${per_sys_cum[s]:+,.0f})")
pr()

pr("PERIOD SUMMARIES:")
for m in [m_is, m_oos, m_full]:
    if m is None:
        continue
    pr(f"  {m['label']:<16}  ${m['start_eq']:>9,.0f} -> ${m['end_eq']:>9,.0f}"
       f"  total {m['total_ret']:>+7.1f}%  CAGR {m['cagr']:>+6.2f}%"
       f"  Sharpe {m['sharpe']:>5.2f}  MaxDD {m['max_dd']:>6.2f}%")
pr()

pr("SANITY CHECKS:")
y2022 = annual_df[annual_df["year"] == 2022]["return_pct"]
c2022 = float(y2022.iloc[0]) if len(y2022) else float("nan")
pr(f"  2022 positive (S6/S7/S8 earned): {c2022:+.1f}%  "
   f"[{'PASS' if c2022 > 0 else 'FAIL -- INVESTIGATE'}]")
worst_over = annual_df[annual_df["return_pct"] > 200]
pr(f"  No year > +200% (normalization): "
   f"[{'PASS' if worst_over.empty else 'FAIL -- ' + str(list(worst_over['year']))}]")
in_range = annual_df[(annual_df["return_pct"] < 5) | (annual_df["return_pct"] > 60)]
if not in_range.empty:
    pr(f"  Years outside +5%..+60% expected band: "
       + ", ".join(f"{int(r['year'])} ({r['return_pct']:+.1f}%)" for _, r in in_range.iterrows()))
pr()

pr("S5 NOTE: s5_regenerated_trades.csv carries R-multiples in column 'r'")
pr("  (verified: same convention as Step 20). Standard risk_amount x net_r used.")
pr("  Finding #23 caveat applies: S5's 2022 contribution is understated.")
pr("FUNDING NOTE: net_r in all trade files is funding-adjusted upstream at T8")
pr("  (sign conventions validated in Steps 16/18 and T8 freezes; not recomputed here).")
pr()

pr("=" * 78)
pr("PROJECTED AT FULL POST-OCTOBER SIZING -- ILLUSTRATIVE ONLY, NOT VALIDATED")
pr("(each year's dollar P&L x4 as proxy for 1.0% MR / 0.50% S1 sizing)")
pr("=" * 78)
proj_eq = STARTING_CAPITAL
pr(f"  {'Year':<6}{'P&L x4':>12}{'End eq':>12}{'Return':>9}")
pr("  " + "-" * 40)
for _, r in annual_df.iterrows():
    yr_pnl = (r["end_eq"] - r["start_eq"]) * 4.0
    start = proj_eq
    proj_eq += yr_pnl
    yl = f"{int(r['year'])}*" if r["year"] == 2026 else f"{int(r['year'])}"
    pr(f"  {yl:<6}{yr_pnl:>+12,.0f}{proj_eq:>12,.0f}{yr_pnl / start * 100:>+8.1f}%")
pr()
pr("Per-system full-sizing P&L x4 by year: multiply cross-system table by 4.")
pr("=" * 78)

(OUT_DIR / "combined_replay_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved: {OUT_DIR}/daily_equity_curve.csv, annual_summary.csv,")
print(f"       per_system_annual_summary.csv, combined_replay_report.txt")
