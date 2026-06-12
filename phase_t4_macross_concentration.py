"""
T4 Concentration Check — MA Cross Short 1D
Best combo from T1: fast=10, slow=200 (EMA10/EMA200), ATR*2.0
Checks:
  - Remove top-1 month: avg_r must stay > 0.15R
  - Remove top-2 months
  - Remove top-3 months
  - Remove 2025 entirely: is system still profitable?
  - Year-by-year without 2025
  - IS/OOS split (train 2020-2023, test 2024-2026)
Verdict: T2 candidate if remove-top-1-month keeps avg_r > 0.15R
         HALT if removing 2025 inverts system (same as BollingerBand MR failure)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_1D  = Path("data/futures_universe/ohlcv_1d")
OUT_DIR  = Path("data/research_macross_t4_concentration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_N     = 10
SLOW_N     = 200
ATR_MULT   = 2.0
MAX_HOLD   = 30
COST_FLOOR = 0.15
ATR_PERIOD = 14
EMA200_N   = 200
WARMUP     = 250
TRAIN_YEARS = {2020, 2021, 2022, 2023}
TEST_YEARS  = {2024, 2025, 2026}

# ── Backtest engine ────────────────────────────────────────────────────
def compute_atr(df):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    atr = np.full(len(tr), np.nan)
    atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
    a = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, len(tr)):
        atr[i] = atr[i-1] * (1 - a) + tr[i] * a
    return atr

def compute_ema(closes, n):
    m = len(closes)
    ema = np.full(m, np.nan)
    if m < n:
        return ema
    ema[n - 1] = closes[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, m):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    return ema

def backtest_symbol(symbol, df, fast_n, slow_n, am):
    if len(df) < WARMUP + slow_n + 10:
        return []
    closes     = df["close"].values.astype(float)
    opens      = df["open"].values.astype(float)
    highs      = df["high"].values.astype(float)
    timestamps = df["timestamp"].values.astype(np.int64)
    atr        = compute_atr(df)
    ema200     = compute_ema(closes, EMA200_N)
    fast_arr   = compute_ema(closes, fast_n)
    slow_arr   = compute_ema(closes, slow_n)

    trades = []
    in_trade   = False
    entry_price = stop = risk = bars_held = 0
    entry_ts   = 0

    for i in range(WARMUP, len(closes)):
        if in_trade:
            bars_held += 1
            if highs[i] >= stop:
                r = (entry_price - stop) / risk if risk > 0 else 0.0
                trades.append({"symbol": symbol, "entry_ts": entry_ts, "r": r})
                in_trade = False
                continue
            if (not np.isnan(fast_arr[i]) and not np.isnan(slow_arr[i])
                    and fast_arr[i] > slow_arr[i]):
                r = (entry_price - opens[i]) / risk if risk > 0 else 0.0
                trades.append({"symbol": symbol, "entry_ts": entry_ts, "r": r})
                in_trade = False
                continue
            if bars_held >= MAX_HOLD:
                r = (entry_price - opens[i]) / risk if risk > 0 else 0.0
                trades.append({"symbol": symbol, "entry_ts": entry_ts, "r": r})
                in_trade = False
            continue

        if i < 1:
            continue
        if (np.isnan(atr[i]) or np.isnan(ema200[i]) or
                np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]) or
                np.isnan(fast_arr[i-1]) or np.isnan(slow_arr[i-1])):
            continue
        if closes[i] >= ema200[i]:
            continue
        cross_down = fast_arr[i-1] >= slow_arr[i-1] and fast_arr[i] < slow_arr[i]
        if not cross_down:
            continue
        if i + 1 >= len(opens):
            continue
        entry_price = opens[i + 1]
        stop = entry_price + atr[i] * am
        risk = stop - entry_price
        if risk <= 0:
            continue
        in_trade  = True
        bars_held = 0
        entry_ts  = int(timestamps[i])

    return trades

# ── Run backtest ───────────────────────────────────────────────────────
print("=" * 60)
print(f"T4 Concentration: MA Cross 1D  fast={FAST_N}/slow={SLOW_N}  ATR*{ATR_MULT}")
print("=" * 60)

files = sorted(DATA_1D.glob("*_1d.csv"))
print(f"Symbols: {len(files)}")

all_trades = []
for idx, f in enumerate(files, 1):
    sym = f.stem.replace("_1d", "")
    df  = pd.read_csv(f)
    trs = backtest_symbol(sym, df, FAST_N, SLOW_N, ATR_MULT)
    all_trades.extend(trs)
    if idx % 50 == 0:
        print(f"  [{idx}/{len(files)}]")

df_all = pd.DataFrame(all_trades)
if len(df_all) == 0:
    print("ERROR: zero trades")
    raise SystemExit(1)

df_all["dt"]    = pd.to_datetime(df_all["entry_ts"], unit="ms", utc=True)
df_all["year"]  = df_all["dt"].dt.year
df_all["month"] = df_all["dt"].dt.to_period("M").astype(str)
df_all.to_csv(OUT_DIR / "trades_winner.csv", index=False)
print(f"\nTotal trades: {len(df_all)}")
print(f"All-time avg_r: {df_all['r'].mean():+.4f}R")

# ── T4 Analysis ────────────────────────────────────────────────────────
lines = []
def pr(s=""):
    print(s)
    lines.append(s)

pr("=" * 60)
pr(f"T4 CONCENTRATION CHECK: MA Cross 1D  EMA{FAST_N}/EMA{SLOW_N}  ATR*{ATR_MULT}")
pr(f"Total trades: {len(df_all)}   §4.2 floor: {COST_FLOOR}R")
pr("=" * 60)

avg_all = df_all["r"].mean()
pr(f"\nBaseline avg_r: {avg_all:+.4f}R")

# Year-by-year full
pr(f"\n--- Year-by-Year (all years) ---")
for yr in sorted(df_all["year"].unique()):
    yd = df_all[df_all["year"] == yr]
    period = "IS " if yr in TRAIN_YEARS else "OOS"
    pct_of_total = yd["r"].sum() / df_all["r"].sum() * 100
    pr(f"  {yr} [{period}]: n={len(yd):4d}  total={yd['r'].sum():+7.2f}R  "
       f"avg={yd['r'].mean():+.4f}R  ({pct_of_total:+.0f}% of total R)")

# Top months by total R
pr(f"\n--- Top 10 Months by Total R ---")
by_month = df_all.groupby("month")["r"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
for i, (m, row) in enumerate(by_month.head(10).iterrows()):
    pr(f"  #{i+1} {m}: total={row['sum']:+.2f}R  n={int(row['count'])}  avg={row['mean']:+.4f}R")

# Remove best month
m1 = by_month.index[0]
m1_r = by_month.iloc[0]["sum"]
df_no_m1   = df_all[df_all["month"] != m1]
avg_no_m1  = df_no_m1["r"].mean() if len(df_no_m1) > 0 else 0.0
m1_ok      = avg_no_m1 > COST_FLOOR
pr(f"\n--- Remove Top-1 Month ({m1}, total={m1_r:+.2f}R) ---")
pr(f"  avg_r without: {avg_no_m1:+.4f}R  {'[PASS]' if m1_ok else '[FAIL]'}")

# Remove top-2 months
m2 = by_month.index[1]
m2_r = by_month.iloc[1]["sum"]
df_no_m2   = df_all[~df_all["month"].isin([m1, m2])]
avg_no_m2  = df_no_m2["r"].mean() if len(df_no_m2) > 0 else 0.0
m2_ok      = avg_no_m2 > COST_FLOOR
pr(f"\n--- Remove Top-2 Months ({m1}, {m2}) ---")
pr(f"  avg_r without: {avg_no_m2:+.4f}R  {'[PASS]' if m2_ok else '[FAIL]'}")

# Remove top-3 months
m3 = by_month.index[2]
m3_r = by_month.iloc[2]["sum"]
df_no_m3   = df_all[~df_all["month"].isin([m1, m2, m3])]
avg_no_m3  = df_no_m3["r"].mean() if len(df_no_m3) > 0 else 0.0
m3_ok      = avg_no_m3 > COST_FLOOR
pr(f"\n--- Remove Top-3 Months ({m1}, {m2}, {m3}) ---")
pr(f"  avg_r without: {avg_no_m3:+.4f}R  {'[PASS]' if m3_ok else '[FAIL]'}")

# Remove 2025 entirely
df_no_2025  = df_all[df_all["year"] != 2025]
avg_no_2025 = df_no_2025["r"].mean() if len(df_no_2025) > 0 else 0.0
no_2025_ok  = avg_no_2025 > COST_FLOOR
pr(f"\n--- Remove 2025 Entirely ---")
pr(f"  Trades without 2025: {len(df_no_2025)}")
pr(f"  avg_r without 2025: {avg_no_2025:+.4f}R  {'[PASS]' if no_2025_ok else '[FAIL - system needs 2025]'}")

pr(f"\n  Year-by-year without 2025:")
for yr in sorted(df_no_2025["year"].unique()):
    yd = df_no_2025[df_no_2025["year"] == yr]
    pr(f"    {yr}: n={len(yd):4d}  avg={yd['r'].mean():+.4f}R")

# 2025 concentration: what % of total R comes from 2025?
total_r  = df_all["r"].sum()
r_2025   = df_all[df_all["year"] == 2025]["r"].sum()
pct_2025 = r_2025 / total_r * 100 if total_r != 0 else 0.0
pr(f"\n  2025 concentration: {pct_2025:.0f}% of total R")
if pct_2025 > 80:
    pr(f"  [WARN] 2025 accounts for {pct_2025:.0f}% of total R — same failure pattern as BollingerBand MR")

# IS/OOS split
df_is   = df_all[df_all["year"].isin(TRAIN_YEARS)]
df_oos  = df_all[df_all["year"].isin(TEST_YEARS)]
is_avg  = df_is["r"].mean()  if len(df_is)  > 0 else 0.0
oos_avg = df_oos["r"].mean() if len(df_oos) > 0 else 0.0
degrade = (is_avg - oos_avg) / abs(is_avg) if is_avg != 0 else 0.0
pr(f"\n--- IS / OOS Split ---")
pr(f"  IS  (2020-2023): n={len(df_is):4d}  avg_r={is_avg:+.4f}R")
pr(f"  OOS (2024-2026): n={len(df_oos):4d}  avg_r={oos_avg:+.4f}R")
pr(f"  IS->OOS degradation: {degrade*100:.1f}%")

# Win rate
wr = (df_all["r"] > 0).sum() / len(df_all)
pr(f"\n--- Win Rate ---")
pr(f"  Win rate: {wr*100:.1f}%  (gate: 30-45%)")
wr_ok = 0.30 <= wr <= 0.45

# Concentration — top-1 asset
by_asset = df_all.groupby("symbol")["r"].sum().sort_values(ascending=False)
top1_sym = by_asset.index[0]
top1_pct = by_asset.iloc[0] / total_r * 100
pr(f"\n--- Concentration (top-1 asset) ---")
pr(f"  {top1_sym}: {top1_pct:.1f}% of total R  "
   f"{'[PASS]' if top1_pct < 50 else '[FAIL] >50%'}")

# Overall T4 verdict
pr(f"\n--- T4 VERDICT ---")
if not m1_ok:
    verdict = "HALT"
    reason  = "system fails remove-top-1-month (same pattern as BollingerBand MR)"
elif not no_2025_ok and pct_2025 > 80:
    verdict = "HALT"
    reason  = f"2025 accounts for {pct_2025:.0f}% of total R — year dependency confirmed"
elif m1_ok and no_2025_ok:
    verdict = "T2 CANDIDATE"
    reason  = "survives remove-top-1-month AND positive without 2025"
elif m1_ok and not no_2025_ok:
    verdict = "REVIEW"
    reason  = "survives remove-top-1-month but negative without 2025 — proceed with caution"
else:
    verdict = "HALT"
    reason  = "concentration failure"

pr(f"  {verdict}: {reason}")
pr(f"\n  Check summary:")
pr(f"    remove top-1 month:  {'[PASS]' if m1_ok else '[FAIL]'} avg_r={avg_no_m1:+.4f}R")
pr(f"    remove top-2 months: {'[PASS]' if m2_ok else '[FAIL]'} avg_r={avg_no_m2:+.4f}R")
pr(f"    remove top-3 months: {'[PASS]' if m3_ok else '[FAIL]'} avg_r={avg_no_m3:+.4f}R")
pr(f"    remove 2025:         {'[PASS]' if no_2025_ok else '[FAIL]'} avg_r={avg_no_2025:+.4f}R")
pr(f"    IS->OOS stable:      {'[PASS]' if degrade < 0.40 else '[FAIL]'} ({degrade*100:.0f}% degrade)")
pr(f"    win rate 30-45%:     {'[PASS]' if wr_ok else '[FAIL]'} ({wr*100:.1f}%)")

report = "\n".join(lines)
with open(OUT_DIR / "t4_concentration_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nSaved to {OUT_DIR}/t4_concentration_report.txt")
