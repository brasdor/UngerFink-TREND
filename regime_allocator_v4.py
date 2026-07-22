"""
Step 12 v2 -- Regime Allocator v4 with Anti-Oscillation
Three improvements over v3:
  1. Hysteresis bands: entry/exit thresholds with dead band
  2. EMA smoothing: 5-day on funding, 3-day on breadth before thresholding
  3. Minimum 5-day holding period per axis before re-switch

Motivation: v3 produced 316 switches in 3.5 years (~90/year) because the
funding axis threshold sat at the median signal magnitude. See finding #24.

Validation gates:
  - Switch count < 50/year (vs v3's ~90/year)
  - 2022 BEAR classification >= 90% of days
  - CAGR >= v3's +21.8%
"""
import warnings; warnings.filterwarnings("ignore")
import sqlite3, json, numpy as np, pandas as pd
from pathlib import Path

DATA_1D = Path("data/futures_universe/ohlcv_1d")
DB_PATH = Path("data/futures.db")
OUT_DIR = Path("data/research_regime_allocator_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BT_START = pd.Timestamp("2021-01-01")
BT_END   = pd.Timestamp("2026-06-15")

# S2/S8 are a single MR pool split 40/60 (finding #18). The former combined
# S2 weight is divided accordingly — the old S8=0.00 placeholders predated the
# S8 engine and starved the bear amplifier in BULL/BEAR regimes (2022 was S8's
# best year). Must stay in sync with engines/regime_daily.py — enforced by
# tools/check_regime_weights_sync.py (CI).
SCHEME_C = {
    "BULL":  {"S1": 0.25, "S2": 0.06, "S3": 0.15, "S5": 0.25,
              "S6": 0.10, "S7": 0.10, "S8": 0.09},
    "BEAR":  {"S1": 0.05, "S2": 0.10, "S3": 0.05, "S5": 0.15,
              "S6": 0.25, "S7": 0.25, "S8": 0.15},
    "MIXED": {"S1": 0.125, "S2": 0.125, "S3": 0.125, "S5": 0.125,
              "S6": 0.125, "S7": 0.125, "S8": 0.125},
}

SYSTEM_ANNUAL_PROFILES = {
    2021: {"S1": 30.0, "S2": 2.65, "S3": 4.03, "S5": 47.0,
           "S6": 41.7, "S7": 46.4, "S8": 10.3},
    2022: {"S1": -10.0, "S2": 4.95, "S3": -3.2, "S5": -20.0,
           "S6": 80.0, "S7": 80.0, "S8": 32.1},
    2023: {"S1": 16.0, "S2": 2.65, "S3": 4.03, "S5": 47.0,
           "S6": 41.7, "S7": 46.4, "S8": 10.3},
    2024: {"S1": 25.0, "S2": 2.65, "S3": 16.0, "S5": 60.0,
           "S6": 41.7, "S7": 46.4, "S8": 10.3},
    2025: {"S1": 16.0, "S2": 2.65, "S3": 4.03, "S5": 47.0,
           "S6": 41.7, "S7": 46.4, "S8": 10.3},
    2026: {"S1": 5.0, "S2": -1.5, "S3": 1.0, "S5": 10.0,
           "S6": 10.0, "S7": -5.0, "S8": 5.0},
}

# ── Hysteresis thresholds ────────────────────────────────────────────
FUNDING_ENTER_HIGH = 0.00025   # 0.025%/8h
FUNDING_EXIT_HIGH  = 0.00010   # 0.010%/8h
FUNDING_ENTER_LOW  = 0.00005   # 0.005%/8h (below this = LOW)
FUNDING_EXIT_LOW   = 0.00010   # above this = exit LOW

BREADTH_ENTER_BULL = 58        # was 55
BREADTH_EXIT_BULL  = 48
BREADTH_ENTER_BEAR = 42        # was 45 (inverted: below this = BEAR)
BREADTH_EXIT_BEAR  = 48

MIN_HOLD_DAYS = 5

# ── Load BTC data ────────────────────────────────────────────────────
print("Loading BTC data...")
btc = pd.read_csv(DATA_1D / "BTCUSDT_1d.csv")
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms")
btc = btc.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
btc_close = btc["close"].astype(float)
btc_ema200 = btc_close.ewm(span=200, adjust=False).mean()

# Breadth
print("Computing breadth...")
all_closes = {}
for f in sorted(DATA_1D.glob("*_1d.csv")):
    sym = f.stem.replace("_1d", "")
    df = pd.read_csv(f, usecols=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    all_closes[sym] = df["close"].astype(float)

close_panel = pd.DataFrame(all_closes)
ema200_panel = close_panel.ewm(span=200, min_periods=20, adjust=False).mean()
above_ema = (close_panel > ema200_panel).sum(axis=1)
total_syms = close_panel.notna().sum(axis=1)
raw_breadth = (above_ema / total_syms * 100).fillna(50)
breadth_ema3 = raw_breadth.ewm(span=3, adjust=False).mean()

# Funding from DB
print("Loading funding data...")
conn = sqlite3.connect(str(DB_PATH))
funding_db = pd.read_sql("SELECT timestamp, symbol, funding_rate FROM funding_history_coinglass", conn)
oi_db = pd.read_sql("SELECT timestamp, symbol, oi_usd FROM oi_history WHERE interval='1d'", conn)
conn.close()

funding_db["date"] = pd.to_datetime(funding_db["timestamp"], unit="ms").dt.normalize()
oi_db["date"] = pd.to_datetime(oi_db["timestamp"], unit="ms").dt.normalize()

monthly_top20 = {}
for m in pd.date_range(BT_START - pd.DateOffset(months=1), BT_END, freq="MS"):
    month_oi = oi_db[(oi_db["date"] >= m) & (oi_db["date"] < m + pd.DateOffset(months=1))]
    if len(month_oi) > 0:
        avg_oi = month_oi.groupby("symbol")["oi_usd"].mean().nlargest(20)
        monthly_top20[m] = set(avg_oi.index)

print("Computing daily funding...")
dates = pd.date_range(BT_START, BT_END, freq="D")
daily_funding_raw = {}
for d in dates:
    month_key = d.replace(day=1)
    top20 = monthly_top20.get(month_key, set())
    if not top20:
        daily_funding_raw[d] = 0.0001
        continue
    day_fund = funding_db[(funding_db["date"] == d) & (funding_db["symbol"].isin(top20))]
    daily_funding_raw[d] = day_fund["funding_rate"].mean() if len(day_fund) > 0 else 0.0001

funding_raw = pd.Series(daily_funding_raw).sort_index()
funding_ema5 = funding_raw.ewm(span=5, adjust=False).mean()

# Vol axis (unchanged from v3)
print("Computing volatility regime...")
btc_hi = btc["high"].astype(float)
btc_lo = btc["low"].astype(float)
tr = pd.concat([btc_hi - btc_lo, (btc_hi - btc_close.shift()).abs(),
                (btc_lo - btc_close.shift()).abs()], axis=1).max(axis=1)
btc_atr14 = tr.rolling(14).mean()
btc_atr_pct = btc_atr14.rolling(252).apply(lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False)

# ── Regime classification with hysteresis + min hold ─────────────────
print("Classifying regimes with anti-oscillation...")

current_trend = "MIXED"
current_funding = "NEUTRAL"
last_trend_switch = BT_START - pd.Timedelta(days=MIN_HOLD_DAYS + 1)
last_funding_switch = BT_START - pd.Timedelta(days=MIN_HOLD_DAYS + 1)

def get_weights(trend, funding_regime, vol_mult):
    base = dict(SCHEME_C[trend])
    if funding_regime == "HIGH":
        for s in ["S6", "S7"]:  base[s] = base.get(s, 0) + 0.05
        for s in ["S1", "S5"]:  base[s] = max(0, base.get(s, 0) - 0.05)
    elif funding_regime == "LOW":
        for s in ["S2", "S3"]:  base[s] = base.get(s, 0) + 0.05
        for s in ["S6", "S7"]:  base[s] = max(0, base.get(s, 0) - 0.05)
    total = sum(base.values())
    if total > 0:
        base = {k: v / total for k, v in base.items()}
    return base

auto_equity = 60000.0
manual_equity = 60000.0
auto_daily_rets = []
manual_daily_rets = []
regime_log = []
switch_count = 0

for d in dates:
    yr = d.year
    if yr not in SYSTEM_ANNUAL_PROFILES:
        continue

    # Trend axis with hysteresis + EMA-smoothed breadth + min hold
    btc_above = (btc_close.get(d, None) is not None and
                 btc_ema200.get(d, None) is not None and
                 float(btc_close.get(d, 0)) > float(btc_ema200.get(d, 0)))
    br = float(breadth_ema3.get(d, 50))

    can_switch_trend = (d - last_trend_switch).days >= MIN_HOLD_DAYS
    new_trend = current_trend
    if can_switch_trend:
        if current_trend != "BULL" and btc_above and br > BREADTH_ENTER_BULL:
            new_trend = "BULL"
        elif current_trend == "BULL" and (not btc_above or br < BREADTH_EXIT_BULL):
            new_trend = "MIXED"
        elif current_trend != "BEAR" and not btc_above and br < BREADTH_ENTER_BEAR:
            new_trend = "BEAR"
        elif current_trend == "BEAR" and (btc_above or br > BREADTH_EXIT_BEAR):
            new_trend = "MIXED"

    if new_trend != current_trend:
        switch_count += 1
        last_trend_switch = d
        current_trend = new_trend

    # Funding axis with hysteresis + EMA-smoothed + min hold
    fr = float(funding_ema5.get(d, 0.0001))
    can_switch_funding = (d - last_funding_switch).days >= MIN_HOLD_DAYS
    new_funding = current_funding
    if can_switch_funding:
        if current_funding != "HIGH" and fr > FUNDING_ENTER_HIGH:
            new_funding = "HIGH"
        elif current_funding == "HIGH" and fr < FUNDING_EXIT_HIGH:
            new_funding = "NEUTRAL"
        elif current_funding != "LOW" and fr < FUNDING_ENTER_LOW:
            new_funding = "LOW"
        elif current_funding == "LOW" and fr > FUNDING_EXIT_LOW:
            new_funding = "NEUTRAL"

    if new_funding != current_funding:
        switch_count += 1
        last_funding_switch = d
        current_funding = new_funding

    # Vol axis (unchanged)
    vp = btc_atr_pct.get(d, 50) if d in btc_atr_pct.index else 50
    if np.isnan(vp): vp = 50
    if vp < 30:   vol_mult = 0.75
    elif vp > 70: vol_mult = 1.25
    else:         vol_mult = 1.00

    # Compute returns
    sys_daily = {}
    for sys_name, ann_ret in SYSTEM_ANNUAL_PROFILES[yr].items():
        sys_daily[sys_name] = (1 + ann_ret / 100) ** (1/365) - 1

    auto_weights = get_weights(current_trend, current_funding, vol_mult)
    auto_ret = sum(auto_weights.get(s, 0) * sys_daily.get(s, 0) for s in sys_daily)
    auto_ret *= vol_mult
    auto_equity *= (1 + auto_ret)
    auto_daily_rets.append(auto_ret)

    manual_weights = SCHEME_C["MIXED"]
    manual_ret = sum(manual_weights.get(s, 0) * sys_daily.get(s, 0) for s in sys_daily)
    manual_equity *= (1 + manual_ret)
    manual_daily_rets.append(manual_ret)

    regime_log.append({
        "date": d, "trend": current_trend, "funding": current_funding,
        "vol_mult": vol_mult, "auto_equity": auto_equity, "manual_equity": manual_equity,
        "breadth_ema3": br, "funding_ema5": fr,
    })

regime_df = pd.DataFrame(regime_log)
regime_df.to_csv(OUT_DIR / "regime_backtest_daily_v4.csv", index=False)

# ── Compute metrics ──────────────────────────────────────────────────
auto_rets = np.array(auto_daily_rets)
manual_rets = np.array(manual_daily_rets)
auto_sharpe = auto_rets.mean() / auto_rets.std() * np.sqrt(365) if auto_rets.std() > 0 else 0
manual_sharpe = manual_rets.mean() / manual_rets.std() * np.sqrt(365) if manual_rets.std() > 0 else 0
auto_cagr = (auto_equity / 60000) ** (365 / len(auto_daily_rets)) - 1
manual_cagr = (manual_equity / 60000) ** (365 / len(manual_daily_rets)) - 1

n_years = (BT_END - BT_START).days / 365.25
switches_per_year = switch_count / n_years

trend_counts = regime_df["trend"].value_counts()
funding_counts = regime_df["funding"].value_counts()

# 2022 BEAR classification check
yr2022 = regime_df[regime_df["date"].dt.year == 2022]
bear_2022_days = (yr2022["trend"] == "BEAR").sum()
bear_2022_pct = bear_2022_days / len(yr2022) * 100 if len(yr2022) > 0 else 0

# Count switches per year
yearly_switches = {}
prev_state = None
for _, row in regime_df.iterrows():
    state = (row["trend"], row["funding"])
    yr = row["date"].year
    if state != prev_state:
        yearly_switches[yr] = yearly_switches.get(yr, 0) + 1
        prev_state = state

# Also load v3 results for comparison
v3_report_path = Path("data/research_regime_allocator_v3/regime_backtest_daily.csv")
v3_switches = None
if v3_report_path.exists():
    v3_df = pd.read_csv(v3_report_path)
    prev = None
    v3_sw = 0
    for _, r in v3_df.iterrows():
        s = (r["trend"], r["funding"])
        if s != prev: v3_sw += 1; prev = s
    v3_switches = v3_sw

# ── Report ───────────────────────────────────────────────────────────
lines = []
def pr(s=""): print(s); lines.append(s)

pr("="*70)
pr("STEP 12 V2 -- REGIME ALLOCATOR v4 (ANTI-OSCILLATION)")
pr("="*70)
pr(f"\nBacktest: {BT_START.date()} to {BT_END.date()} ({len(regime_log)} days)")
pr(f"Three-axis detection with hysteresis + EMA smoothing + min hold")
pr()

pr("ANTI-OSCILLATION PARAMETERS:")
pr(f"  Funding hysteresis: enter HIGH >{FUNDING_ENTER_HIGH*100:.3f}%/8h, "
   f"exit HIGH <{FUNDING_EXIT_HIGH*100:.3f}%/8h")
pr(f"  Breadth hysteresis: enter BULL >{BREADTH_ENTER_BULL}%, "
   f"exit BULL <{BREADTH_EXIT_BULL}%")
pr(f"  EMA smoothing: funding 5-day, breadth 3-day")
pr(f"  Min hold period: {MIN_HOLD_DAYS} days per axis")
pr()

pr("SWITCH COUNT (KEY METRIC):")
pr(f"  v4 total switches: {switch_count}")
pr(f"  v4 switches/year:  {switches_per_year:.1f}")
if v3_switches:
    v3_per_year = v3_switches / n_years
    pr(f"  v3 total switches: {v3_switches}")
    pr(f"  v3 switches/year:  {v3_per_year:.1f}")
    pr(f"  Reduction:         {(1 - switches_per_year/v3_per_year)*100:.0f}%")
pr(f"  Target:            <50/year")
pr(f"  Gate:              [{'PASS' if switches_per_year < 50 else 'FAIL'}]")
pr()

pr("SWITCHES PER YEAR:")
for yr in sorted(yearly_switches.keys()):
    pr(f"  {yr}: {yearly_switches[yr]} switches")
pr()

pr("REGIME DISTRIBUTION:")
for regime in ["BULL", "BEAR", "MIXED"]:
    ct = trend_counts.get(regime, 0)
    pct = ct / len(regime_df) * 100
    pr(f"  {regime:>6s}: {ct:>5d} days ({pct:.1f}%)")
pr()
pr("FUNDING DISTRIBUTION:")
for regime in ["HIGH", "NEUTRAL", "LOW"]:
    ct = funding_counts.get(regime, 0)
    pct = ct / len(regime_df) * 100
    pr(f"  {regime:>8s}: {ct:>5d} days ({pct:.1f}%)")
pr()

pr("2022 BEAR CLASSIFICATION (CRITICAL CHECK):")
pr(f"  Days classified BEAR: {bear_2022_days}/{len(yr2022)} ({bear_2022_pct:.1f}%)")
pr(f"  Gate (>= 90%):        [{'PASS' if bear_2022_pct >= 90 else 'FAIL'}]")
pr()

pr("PERFORMANCE COMPARISON:")
pr(f"  {'Metric':<20s} {'v4 Auto':>14s} {'Manual (MIXED)':>14s}")
pr(f"  {'-'*50}")
pr(f"  {'Final equity':<20s} ${auto_equity:>13,.0f} ${manual_equity:>13,.0f}")
pr(f"  {'CAGR':<20s} {auto_cagr*100:>+13.1f}% {manual_cagr*100:>+13.1f}%")
pr(f"  {'Sharpe':<20s} {auto_sharpe:>14.3f} {manual_sharpe:>14.3f}")
pr()

cagr_gate = auto_cagr * 100 >= 21.8
pr("CAGR GATE:")
pr(f"  v4 CAGR: {auto_cagr*100:+.1f}%  (threshold >= +21.8% from v3)")
pr(f"  Gate:    [{'PASS' if cagr_gate else 'FAIL'}]")
pr()

pr("YEAR-BY-YEAR:")
for yr in range(2021, 2027):
    yr_data = regime_df[regime_df["date"].dt.year == yr]
    if len(yr_data) == 0: continue
    yr_auto = yr_data["auto_equity"].iloc[-1] / yr_data["auto_equity"].iloc[0] - 1
    yr_manual = yr_data["manual_equity"].iloc[-1] / yr_data["manual_equity"].iloc[0] - 1
    yr_regimes = yr_data["trend"].value_counts().to_dict()
    pr(f"  {yr}: auto={yr_auto*100:+.1f}%  manual={yr_manual*100:+.1f}%  "
       f"regimes={yr_regimes}")
pr()

all_pass = switches_per_year < 50 and bear_2022_pct >= 90 and cagr_gate
pr("="*70)
pr(f"OVERALL: {'ALL GATES PASS -- v4 ready for Step 20' if all_pass else 'GATE FAILURE -- investigate'}")
pr("="*70)

report_text = "\n".join(lines)
(OUT_DIR / "v4_vs_v3_comparison.txt").write_text(report_text, encoding="utf-8")

# Write regime_state.json (latest state for T9B)
latest = regime_df.iloc[-1]
regime_state = {
    "date": str(latest["date"].date()),
    "trend_regime": latest["trend"],
    "funding_regime": latest["funding"],
    "vol_multiplier": latest["vol_mult"],
    "weights": get_weights(latest["trend"], latest["funding"], latest["vol_mult"]),
    "allocator_version": "v4",
}
with open(OUT_DIR / "regime_state_v4.json", "w") as f:
    json.dump(regime_state, f, indent=2)

print(f"\nReport saved to: {OUT_DIR / 'v4_vs_v3_comparison.txt'}")
