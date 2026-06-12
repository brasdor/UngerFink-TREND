"""
T2 Full Validation + T3 Exit Engineering
Vol Contraction Short + Funding Gate >= 0.01%
Config: cb=15, ap=20, hb=20, am=2.0   Universe: 290 Binance Futures 4H

T2: IS 2020-2023 vs OOS 2024-2026
T3: hold_bars in [10, 15, 20, 25, 30] — run if T2 passes
Single symbol-loop: precompute once, run all hb variants per symbol.
"""
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_4H      = Path("data/futures_universe/ohlcv_4h")
FUNDING_DIR  = Path("data/futures_universe/funding_rates")
GRID_CSV     = Path("data/research_volcontraction_option2/grid_trades_with_funding.csv")
OUT_DIR      = Path("data/research_volcontraction_funding_t2_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed entry params
CB   = 15
AP   = 20
AM   = 2.0
FUNDING_THR  = 0.0001     # 0.01% per 8h
T3_HB_VALUES = [10, 15, 20, 25, 30]
T2_HB        = 20

COST_FLOOR_HARD  = 0.25   # §4.2 Futures floor
COST_FLOOR_SOFT  = 0.10   # remove-best checks
WIN_MIN, WIN_MAX = 0.30, 0.45
DEGRADE_LIMIT    = 0.40
TRAIN_YEARS      = {2020, 2021, 2022, 2023}
TEST_YEARS       = {2024, 2025, 2026}

ATR_PERIOD       = 14
EMA_PERIOD       = 200
ATR_PCT_WINDOW   = 252 * 6
ATR_PCT_MIN_BARS = 50
WARMUP           = 300

# ── Helpers ───────────────────────────────────────────────────────────
def load_funding_rates(symbol):
    path = FUNDING_DIR / f"{symbol}_funding.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "funding_time" not in df.columns or "funding_rate" not in df.columns:
            return None
        pairs = sorted(zip(df["funding_time"].values.astype(np.int64),
                           df["funding_rate"].values.astype(float)))
        return pairs
    except Exception:
        return None

def get_rate_at(pairs, ts_ms):
    if not pairs:
        return None
    times = [p[0] for p in pairs]
    idx = bisect.bisect_right(times, ts_ms) - 1
    return pairs[idx][1] if idx >= 0 else None

def compute_atr(df):
    hi = df["high"].values; lo = df["low"].values; cl = df["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr  = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
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

def precompute(df):
    n   = len(df)
    atr = compute_atr(df)
    pct = np.full(n, np.nan)
    for i in range(ATR_PCT_MIN_BARS, n):
        s = max(0, i - ATR_PCT_WINDOW)
        w = atr[s:i]
        if len(w) >= 10:
            pct[i] = float((w <= atr[i]).mean() * 100)
    closes = df["close"].values.astype(float)
    ema200 = compute_ema(closes, EMA_PERIOD)
    # Precompute consecutive low-ATR counter for cb=15 only
    cla = np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isnan(pct[i]):
            cla[i] = 0
        elif pct[i] < AP:
            cla[i] = cla[i-1] + 1
        else:
            cla[i] = 0
    return {
        "ts":    df["timestamp"].values.astype(np.int64),
        "o":     df["open"].values.astype(float),
        "c":     closes,
        "h":     df["high"].values.astype(float),
        "l":     df["low"].values.astype(float),
        "atr":   atr,
        "bear":  closes < ema200,
        "cla":   cla,        # consecutive low-ATR count
    }

def find_entries(d, funding_pairs):
    """Identify all valid entry signals regardless of hold_bars."""
    ts = d["ts"]; o = d["o"]; c = d["c"]; l = d["l"]
    atr = d["atr"]; bear = d["bear"]; cla = d["cla"]
    entries = []
    for i in range(WARMUP, len(c)):
        if np.isnan(atr[i]) or not bear[i]:
            continue
        if cla[i - 1] < CB:
            continue
        comp_start = max(0, i - CB)
        if c[i] >= l[comp_start:i].min():
            continue
        if i + 1 >= len(o):
            continue
        ts_ms = int(ts[i])
        fr = get_rate_at(funding_pairs, ts_ms) if funding_pairs else None
        if fr is None or fr < FUNDING_THR:
            continue
        ep  = o[i + 1]
        stp = ep + atr[i] * AM
        rsk = stp - ep
        if rsk <= 0:
            continue
        entries.append((i + 1, ts_ms, ep, stp, rsk, fr))   # (bar_idx, ts, entry, stop, risk, fr)
    return entries

def simulate_hb(entries, d, hb):
    """
    Simulate trades for a given hold_bars using pre-identified entries.
    Entries are applied sequentially; once in a trade, skip overlapping entries.
    """
    h = d["h"]; o = d["o"]; ts = d["ts"]
    trades = []
    in_trade  = False
    exit_bar  = -1
    for (bar_idx, entry_ts, ep, stp, rsk, fr) in entries:
        if in_trade and bar_idx <= exit_bar:
            continue
        in_trade = False
        # Simulate forward from bar_idx
        n = len(h)
        for j in range(bar_idx, min(bar_idx + hb, n)):
            if h[j] >= stp:
                r = (ep - stp) / rsk
                trades.append({"entry_ts": entry_ts, "r": r, "fr": fr})
                in_trade = True
                exit_bar = j
                break
            if j == bar_idx + hb - 1 or j == n - 1:
                r = (ep - o[j]) / rsk if rsk > 0 else 0.0
                trades.append({"entry_ts": entry_ts, "r": r, "fr": fr})
                in_trade = True
                exit_bar = j
                break
    return trades

# ── Main loop: one pass over all symbols ─────────────────────────────
print("=" * 70)
print(f"T2+T3: Vol Contraction Short + Funding Gate >= {FUNDING_THR*100:.2f}%")
print(f"Entry config: cb={CB} ap={AP} am={AM}   Hold variants: {T3_HB_VALUES}")
print("=" * 70)

files   = sorted(DATA_4H.glob("*_4h.csv"))
symbols = [f.stem.replace("_4h", "") for f in files]
print(f"Symbols: {len(symbols)}")

# accumulate trades per hb
all_trades = {hb: [] for hb in T3_HB_VALUES}
sym_col    = {hb: [] for hb in T3_HB_VALUES}

for idx, symbol in enumerate(symbols, 1):
    df = pd.read_csv(DATA_4H / f"{symbol}_4h.csv")
    if len(df) < WARMUP + 50:
        continue
    d  = precompute(df)
    fp = load_funding_rates(symbol)
    entries = find_entries(d, fp)
    if not entries:
        continue
    for hb in T3_HB_VALUES:
        trs = simulate_hb(entries, d, hb)
        for t in trs:
            t["symbol"] = symbol
        all_trades[hb].extend(trs)
    if idx % 50 == 0:
        print(f"  [{idx}/{len(symbols)}]")

print(f"Done.  hb=20: {len(all_trades[20])} trades")

# Build DataFrames
dfs = {}
for hb in T3_HB_VALUES:
    df_hb = pd.DataFrame(all_trades[hb])
    if len(df_hb) == 0:
        continue
    df_hb["dt"]    = pd.to_datetime(df_hb["entry_ts"], unit="ms", utc=True)
    df_hb["year"]  = df_hb["dt"].dt.year
    df_hb["month"] = df_hb["dt"].dt.to_period("M").astype(str)
    dfs[hb] = df_hb

dfs[20].to_csv(OUT_DIR / "trades_t2.csv", index=False)

# ── Baseline comparison from T1 grid ─────────────────────────────────
print("\nLoading T1 grid for baseline comparison...")
df_grid = pd.read_csv(GRID_CSV)
df_base = df_grid[(df_grid.cb==CB)&(df_grid.ap==AP)&(df_grid.hb==20)&(df_grid.am==AM)]
base_avg = df_base["r"].mean()
base_2022 = df_base[df_base["year"]==2022]["r"].sum()
base_2026 = df_base[df_base["year"]==2026]["r"].sum()
funded_base = df_base[df_base["funding"].notna() & (df_base["funding"] >= FUNDING_THR)]
print(f"Baseline (cb={CB} ap={AP} hb=20 am={AM}): n={len(df_base)} avg_r={base_avg:+.4f}R")

# ── T2 Analysis ───────────────────────────────────────────────────────
lines = []
def pr(s=""):
    print(s); lines.append(s)

df = dfs.get(T2_HB)
if df is None or len(df) == 0:
    pr("ERROR: no T2 trades"); raise SystemExit(1)

pr("\n" + "=" * 70)
pr(f"T2 WALK-FORWARD: Vol Contraction Short + Funding Gate >= {FUNDING_THR*100:.2f}%")
pr(f"Config: cb={CB} ap={AP} hb={T2_HB} am={AM}   §4.2 floor: {COST_FLOOR_HARD}R")
pr(f"Universe: {df['symbol'].nunique()} symbols   Total trades: {len(df)}")
pr("=" * 70)

# Baseline comparison table
pr("\n--- Baseline vs Funding Gate Comparison ---")
pr(f"  {'Metric':<25} {'No gate':<20} {'0.01% gate'}")
pr(f"  {'-'*60}")
pr(f"  {'Total trades':<25} {len(df_base):<20} {len(df)}")
pr(f"  {'avg_r':<25} {base_avg:+.4f}R{'':<14} {df['r'].mean():+.4f}R")
base_2022_avg = df_base[df_base['year']==2022]['r'].mean()
gate_2022_avg = df[df['year']==2022]['r'].mean() if 2022 in df['year'].values else float('nan')
pr(f"  {'2022 avg_r':<25} {base_2022_avg:+.4f}R{'':<14} {gate_2022_avg:+.4f}R")
base_2026_avg = df_base[df_base['year']==2026]['r'].mean()
gate_2026_avg = df[df['year']==2026]['r'].mean() if 2026 in df['year'].values else float('nan')
pr(f"  {'2026 avg_r':<25} {base_2026_avg:+.4f}R{'':<14} {gate_2026_avg:+.4f}R")
base_2026_tot = df_base[df_base['year']==2026]['r'].sum()
gate_2026_tot = df[df['year']==2026]['r'].sum() if 2026 in df['year'].values else 0.0
pr(f"  {'2026 total R':<25} {base_2026_tot:+.2f}R{'':<13} {gate_2026_tot:+.2f}R")

# Year-by-year
pr(f"\n--- Year-by-Year ---")
for yr in sorted(df["year"].unique()):
    yd = df[df["year"] == yr]
    period = "IS " if yr in TRAIN_YEARS else "OOS"
    total_r_all = df["r"].sum()
    pct = yd["r"].sum() / total_r_all * 100 if total_r_all != 0 else 0.0
    pr(f"  {yr} [{period}]: n={len(yd):5d}  total={yd['r'].sum():+8.2f}R  "
       f"avg={yd['r'].mean():+.5f}R  ({pct:+.0f}% of total R)")

# IS / OOS
df_is  = df[df["year"].isin(TRAIN_YEARS)]
df_oos = df[df["year"].isin(TEST_YEARS)]
is_avg  = df_is["r"].mean()  if len(df_is)  > 0 else 0.0
oos_avg = df_oos["r"].mean() if len(df_oos) > 0 else 0.0
if is_avg > 0 and oos_avg > 0:
    degrade = (is_avg - oos_avg) / is_avg
elif is_avg <= 0:
    degrade = 1.0
else:
    degrade = 0.0
degrade_ok = degrade < DEGRADE_LIMIT and oos_avg > 0

pr(f"\n--- IS / OOS Split ---")
pr(f"  IS  (2020-2023): n={len(df_is):5d}  avg_r={is_avg:+.5f}R")
pr(f"  OOS (2024-2026): n={len(df_oos):5d}  avg_r={oos_avg:+.5f}R")
pr(f"  IS->OOS degradation: {degrade*100:.1f}%  (limit: {DEGRADE_LIMIT*100:.0f}%)")
pr(f"  IS->OOS: {'[PASS]' if degrade_ok else '[FAIL]'}")

# First half vs second half
mid = len(df) // 2
df_first = df.iloc[:mid]; df_second = df.iloc[mid:]
first_avg = df_first["r"].mean(); second_avg = df_second["r"].mean()
halves_ok = first_avg > 0 and second_avg > 0
pr(f"\n--- Period Split (first vs second half by trade count) ---")
pr(f"  First  {len(df_first)} trades: avg_r={first_avg:+.5f}R  {'[PASS]' if first_avg > 0 else '[FAIL]'}")
pr(f"  Second {len(df_second)} trades: avg_r={second_avg:+.5f}R  {'[PASS]' if second_avg > 0 else '[FAIL]'}")

# Win rate
wr = (df["r"] > 0).sum() / len(df)
wr_ok = WIN_MIN <= wr <= WIN_MAX
pr(f"\n--- §4.1 Win Rate ---")
pr(f"  Win rate: {wr*100:.1f}%  (gate: {WIN_MIN*100:.0f}%-{WIN_MAX*100:.0f}%)")
pr(f"  Win rate: {'[PASS]' if wr_ok else '[FAIL]'}")

# §4.2 avg_r
avg_r = df["r"].mean()
cost_ok = avg_r > COST_FLOOR_HARD
pr(f"\n--- §4.2 Cost Floor ---")
pr(f"  avg_r: {avg_r:+.5f}R  (floor: {COST_FLOOR_HARD}R)")
pr(f"  §4.2: {'[PASS]' if cost_ok else '[FAIL]'}")

# 2022 gate
yr_2022 = df[df["year"] == 2022]
avg_2022 = yr_2022["r"].mean() if len(yr_2022) > 0 else 0.0
gate_2022_ok = avg_2022 > 0
pr(f"\n--- 2022 Primary Gate ---")
pr(f"  2022: n={len(yr_2022)}  avg_r={avg_2022:+.5f}R")
pr(f"  2022 positive: {'[PASS]' if gate_2022_ok else '[FAIL]'}")

# 2021 catastrophe check
yr_2021 = df[df["year"] == 2021]
avg_2021 = yr_2021["r"].mean() if len(yr_2021) > 0 else 0.0
cat_2021_ok = avg_2021 > -0.50
pr(f"\n--- 2021 Catastrophe Check ---")
pr(f"  2021: n={len(yr_2021)}  avg_r={avg_2021:+.5f}R")
pr(f"  2021 not catastrophic (>-0.5R): {'[PASS]' if cat_2021_ok else '[FAIL]'}")

# §4.7 Concentration — top-1 asset
total_r  = df["r"].sum()
by_asset = df.groupby("symbol")["r"].sum().sort_values(ascending=False)
top1_sym = by_asset.index[0]; top1_r = by_asset.iloc[0]
top1_pct = top1_r / total_r * 100 if total_r > 0 else 0.0
conc_ok  = top1_pct < 50.0
pr(f"\n--- §4.7 Concentration ---")
pr(f"  Top-1 asset: {top1_sym}  {top1_r:+.2f}R  ({top1_pct:.1f}%)")
pr(f"  Top-5 assets:")
for sym, r in by_asset.head(5).items():
    pr(f"    {sym:<22}  {r:+8.2f}R  ({r/total_r*100:.1f}%)")
pr(f"  Concentration: {'[PASS]' if conc_ok else '[FAIL]'} (<50%)")

# Remove best month
by_month = df.groupby("month")["r"].sum().sort_values(ascending=False)
m1 = by_month.index[0]; m1r = by_month.iloc[0]
df_no_m1 = df[df["month"] != m1]
avg_no_m1 = df_no_m1["r"].mean() if len(df_no_m1) > 0 else 0.0
bm_ok = avg_no_m1 > COST_FLOOR_SOFT
pr(f"\n--- Remove Best Month ---")
pr(f"  Best month: {m1}  total={m1r:+.2f}R")
pr(f"  avg_r without: {avg_no_m1:+.5f}R  (soft floor: {COST_FLOOR_SOFT}R)")
pr(f"  Remove best month: {'[PASS]' if bm_ok else '[FAIL]'}")

# Remove best asset
df_no_ba = df[df["symbol"] != top1_sym]
avg_no_ba = df_no_ba["r"].mean() if len(df_no_ba) > 0 else 0.0
ba_ok = avg_no_ba > COST_FLOOR_SOFT
pr(f"\n--- Remove Best Asset ---")
pr(f"  Removed: {top1_sym}")
pr(f"  avg_r without: {avg_no_ba:+.5f}R  (soft floor: {COST_FLOOR_SOFT}R)")
pr(f"  Remove best asset: {'[PASS]' if ba_ok else '[FAIL]'}")

# Overall T2 verdict
checks = {
    "IS->OOS degradation < 40%":    degrade_ok,
    "OOS avg_r positive":            oos_avg > 0,
    "§4.1 win rate 30-45%":          wr_ok,
    "§4.2 avg_r > 0.25R":            cost_ok,
    "2022 positive (primary gate)":  gate_2022_ok,
    "2021 not catastrophic":         cat_2021_ok,
    "§4.7 concentration < 50%":      conc_ok,
    "remove best month > 0.10R":     bm_ok,
    "remove best asset > 0.10R":     ba_ok,
    "both halves positive":          halves_ok,
}
n_pass = sum(checks.values()); n_total = len(checks)
critical = ["OOS avg_r positive", "2022 positive (primary gate)",
            "§4.2 avg_r > 0.25R", "IS->OOS degradation < 40%"]
critical_pass = all(checks[k] for k in critical)

pr(f"\n--- T2 VERDICT ---")
for name, ok in checks.items():
    pr(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
pr(f"\n  Passed {n_pass}/{n_total} checks")

t2_verdict = "BREAKTHROUGH" if critical_pass and n_pass >= 7 else (
             "REVIEW"       if critical_pass and n_pass >= 5 else "HALT")
pr(f"\n  T2 VERDICT: {t2_verdict}")

# ── T3 Exit Engineering (runs regardless — show even if T2 is REVIEW) ─
pr(f"\n\n{'='*70}")
pr(f"T3 EXIT ENGINEERING: hold_bars in {T3_HB_VALUES}")
pr(f"Entry fixed: cb={CB} ap={AP} am={AM} funding>={FUNDING_THR*100:.2f}%")
pr(f"{'='*70}")
pr(f"\n{'hb':>4} {'n_trades':>9} {'avg_r':>10} {'win%':>7} "
   f"{'IS avg_r':>10} {'OOS avg_r':>10} {'2022 avg_r':>12} {'2026 avg_r':>12} §4.2")
pr("-" * 90)

t3_rows = []
for hb in T3_HB_VALUES:
    dh = dfs.get(hb)
    if dh is None or len(dh) == 0:
        pr(f"{hb:>4}  NO DATA")
        continue
    n      = len(dh)
    avg    = dh["r"].mean()
    win    = (dh["r"] > 0).mean() * 100
    is_a   = dh[dh["year"].isin(TRAIN_YEARS)]["r"].mean()
    oos_a  = dh[dh["year"].isin(TEST_YEARS)]["r"].mean()
    y22    = dh[dh["year"]==2022]["r"].mean() if 2022 in dh["year"].values else float("nan")
    y26    = dh[dh["year"]==2026]["r"].mean() if 2026 in dh["year"].values else float("nan")
    ok     = "[PASS]" if avg > COST_FLOOR_HARD else "[FAIL]"
    star   = " <-- T2" if hb == T2_HB else ""
    pr(f"{hb:>4} {n:>9} {avg:>+10.4f}R {win:>6.1f}% "
       f"{is_a:>+10.4f}R {oos_a:>+10.4f}R {y22:>+12.4f}R {y26:>+12.4f}R {ok}{star}")
    t3_rows.append({"hb": hb, "n": n, "avg_r": avg, "win_pct": win,
                    "is_avg": is_a, "oos_avg": oos_a, "avg_2022": y22, "avg_2026": y26})

# Best T3 config
best_t3 = max(t3_rows, key=lambda x: x["avg_r"]) if t3_rows else None
if best_t3:
    pr(f"\n  Best exit: hold_bars={best_t3['hb']}  avg_r={best_t3['avg_r']:+.4f}R")
    if best_t3["hb"] != T2_HB:
        pr(f"  [NOTE] hb={best_t3['hb']} outperforms hb={T2_HB} — consider for live config")
    else:
        pr(f"  [CONFIRM] Original hb={T2_HB} is the best exit")

# Year-by-year for best T3 exit
pr(f"\n--- Year-by-Year for Best Exit (hb={best_t3['hb']}) ---")
dh_best = dfs.get(best_t3["hb"])
if dh_best is not None:
    for yr in sorted(dh_best["year"].unique()):
        yd = dh_best[dh_best["year"] == yr]
        period = "IS " if yr in TRAIN_YEARS else "OOS"
        pr(f"  {yr} [{period}]: n={len(yd):5d}  total={yd['r'].sum():+8.2f}R  "
           f"avg={yd['r'].mean():+.5f}R")

pr(f"\n{'='*70}")
pr(f"FINAL RECOMMENDATION")
pr(f"{'='*70}")
if t2_verdict in ("BREAKTHROUGH", "REVIEW"):
    pr(f"  T2: {t2_verdict}")
    pr(f"  Best entry config: cb={CB} ap={AP} am={AM} funding>={FUNDING_THR*100:.2f}%")
    pr(f"  Best exit: hold_bars={best_t3['hb']}  avg_r={best_t3['avg_r']:+.4f}R")
    pr(f"  Status: ready for T4 robustness checks")
else:
    pr(f"  T2: HALT")
    failed = [k for k, v in checks.items() if not v]
    pr(f"  Failed checks: {failed}")
    if not critical_pass:
        failed_critical = [k for k in critical if not checks[k]]
        pr(f"  Critical failures: {failed_critical}")

report = "\n".join(lines)
with open(OUT_DIR / "t2t3_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nSaved to {OUT_DIR}/t2t3_report.txt")
