"""
Regenerate S6 VolContractionShort canonical trades from frozen config.
Covers 2020-01-01 through 2026-06-19 (IS + OOS).
4H bars, 290-symbol Futures universe.

Frozen config (from phase_t9b_volcontraction_short_paper_engine.py):
  cb=15, ap=20, hb=30, am=2.0, EMA200, funding >= 0.01%/8h
  Entry: at close of signal bar (matching codebase convention)
  Exit:  fixed time exit after 30 4H bars OR ATR stop
  Side:  SHORT
"""
import bisect
import numpy as np
import pandas as pd
from pathlib import Path

DATA_4H     = Path("data/futures_universe/ohlcv_4h")
FUNDING_DIR = Path("data/futures_universe/funding_rates")
OUT_DIR     = Path("data/research_volcontraction_funding_t8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = pd.Timestamp("2020-01-01")
FULL_END   = pd.Timestamp("2026-06-19 23:59:59")
IS_END     = pd.Timestamp("2023-12-31 23:59:59")

COMPRESSION_BARS = 15
ATR_PERCENTILE   = 20
ATR_N            = 14
ATR_STOP_MULT    = 2.0
HOLD_BARS        = 30
EMA_N            = 200
FUNDING_THRESHOLD = 0.0001

DOCUMENTED_IS_AVG_R = 0.304
DOCUMENTED_IS_TRADES_142SYM = 3214

# ── Helpers ──────────────────────────────────────────────────────────
def load_funding(sym):
    p = FUNDING_DIR / f"{sym}_funding.csv"
    if not p.exists(): return None
    try:
        df = pd.read_csv(p)
        return sorted(zip(df["funding_time"].values.astype(np.int64),
                          df["funding_rate"].values.astype(float)))
    except: return None

def get_rate(pairs, ts_ms):
    if not pairs: return None
    times = [p[0] for p in pairs]
    idx = bisect.bisect_right(times, ts_ms) - 1
    return pairs[idx][1] if idx >= 0 else None

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

def compute_ema(close, n):
    ema = np.full(len(close), np.nan)
    if len(close) < n: return ema
    ema[n-1] = close[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, len(close)):
        ema[i] = ema[i-1] * (1 - alpha) + close[i] * alpha
    return ema


# ── Load universe ────────────────────────────────────────────────────
files = sorted(DATA_4H.glob("*_4h.csv"))
symbols = [f.stem.replace("_4h", "") for f in files]
print(f"Universe: {len(symbols)} Futures symbols (4H)")

warmup = pd.Timedelta(days=400)
sym_data = {}
for sym in symbols:
    df = pd.read_csv(DATA_4H / f"{sym}_4h.csv")
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[(df["time"] >= FULL_START - warmup) & (df["time"] <= FULL_END)]
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 500: continue
    fp = load_funding(sym)
    if fp is None: continue
    sym_data[sym] = (df, fp)

print(f"Loaded: {len(sym_data)} symbols with data + funding")

# ── Backtest ─────────────────────────────────────────────────────────
all_trades = []
for sym, (df, fp) in sym_data.items():
    cl = df["close"].values.astype(float)
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    ts_arr = df["timestamp"].values.astype(np.int64)
    times = df["time"].values

    atr = compute_atr(hi, lo, cl, ATR_N)
    ema200 = compute_ema(cl, EMA_N)

    # ATR percentile RANK (not threshold) — matches original T1 script
    atr_pct_rank = np.full(len(cl), np.nan)
    pct_window = 252 * 6  # ~1yr of 4H bars
    for i in range(50, len(cl)):
        start = max(0, i - pct_window)
        w = atr[start:i]
        w = w[np.isfinite(w)]
        if len(w) >= 10:
            atr_pct_rank[i] = float((w <= atr[i]).sum() / len(w) * 100)

    # Consecutive low-ATR bars: count at each bar how many consecutive
    # bars ending at bar i have atr_pct_rank < ATR_PERCENTILE
    consec = np.zeros(len(cl), dtype=int)
    for i in range(1, len(cl)):
        if np.isfinite(atr_pct_rank[i]) and atr_pct_rank[i] < ATR_PERCENTILE:
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 0

    op = df["open"].values.astype(float)
    in_pos = False; e_price = 0; stop = 0; e_bar = 0; e_ts = None; e_ts_ms = 0
    bars_held_count = 0
    warmup_bars = max(EMA_N, 300)
    for i in range(warmup_bars, len(df)):
        if np.isnan(ema200[i]) or np.isnan(atr[i]): continue
        t = pd.Timestamp(times[i])
        if t < FULL_START or t > FULL_END: continue

        if in_pos:
            bars_held_count += 1
            if hi[i] >= stop:
                r_mult = (e_price - stop) / (stop - e_price) if (stop - e_price) != 0 else -1.0
                r_mult = (e_price - stop) / risk_val
                all_trades.append({
                    "symbol": sym,
                    "entry_ts": int(e_ts_ms),
                    "exit_ts": int(ts_arr[i]),
                    "r": r_mult,
                    "year": pd.Timestamp(e_ts).year,
                    "side": "short",
                })
                in_pos = False
                continue
            if bars_held_count >= HOLD_BARS:
                exit_p = op[i]
                r_mult = (e_price - exit_p) / risk_val
                all_trades.append({
                    "symbol": sym,
                    "entry_ts": int(e_ts_ms),
                    "exit_ts": int(ts_arr[i]),
                    "r": r_mult,
                    "year": pd.Timestamp(e_ts).year,
                    "side": "short",
                })
                in_pos = False
            continue

        # Signal check (matches original T1 logic exactly):
        # 1. Previous bar must have consec >= COMPRESSION_BARS
        if consec[i - 1] < COMPRESSION_BARS: continue
        # 2. Must be below EMA200
        if cl[i] >= ema200[i]: continue
        # 3. Close breaks below lowest low of prior compression window
        comp_start = max(0, i - COMPRESSION_BARS)
        compression_low = lo[comp_start:i].min()
        if cl[i] >= compression_low: continue
        # 4. Funding gate
        ts_ms = int(ts_arr[i])
        fr = get_rate(fp, ts_ms)
        if fr is None or fr < FUNDING_THRESHOLD: continue
        # 5. Entry at next bar's open
        if i + 1 >= len(df): continue
        e_price = op[i + 1]
        stop = e_price + ATR_STOP_MULT * atr[i]
        risk_val = stop - e_price
        if risk_val <= 0: continue
        in_pos = True
        e_bar = i + 1
        e_ts = times[i + 1]
        e_ts_ms = int(ts_arr[i + 1])
        bars_held_count = 0

trades_df = pd.DataFrame(all_trades)
trades_df.to_csv(OUT_DIR / "s6_canonical_trades.csv", index=False)

print(f"\nTotal trades (full period): {len(trades_df)}")

# ── IS Verification ──────────────────────────────────────────────────
is_trades = trades_df[trades_df["year"] <= 2023].copy() if not trades_df.empty else pd.DataFrame()
n_is = len(is_trades)
is_avg_r = is_trades["r"].mean() if n_is > 0 else 0

print("\n" + "="*60)
print("S6 REGENERATION — IS VERIFICATION")
print("="*60)
print(f"IS period: 2020-01-01 to 2023-12-31")
print(f"IS trades: {n_is}")
print(f"IS avg_r:  {is_avg_r:+.4f}R")
print(f"Documented IS avg_r (142-sym): +{DOCUMENTED_IS_AVG_R}R")
print(f"Documented IS trades (142-sym): ~{DOCUMENTED_IS_TRADES_142SYM}")
print(f"Expected: more trades on 290-sym, avg_r within 20% of +{DOCUMENTED_IS_AVG_R}R")
print()

gate_pass = is_avg_r >= 0.240
print(f"avg_r >= 0.240R: {is_avg_r:+.4f}R  [{'PASS' if gate_pass else 'FAIL'}]")
if not gate_pass:
    print("STOPPING — IS verification failed. Do not use for Step 20.")
else:
    print(f"PASS — S6 regeneration verified.")
    print(f"\nYear breakdown:")
    for yr in sorted(trades_df["year"].unique()):
        yr_t = trades_df[trades_df["year"] == yr]
        print(f"  {yr}: {len(yr_t)} trades  avg_r={yr_t['r'].mean():+.4f}R  total_r={yr_t['r'].sum():+.1f}R")

print(f"\nSaved to: {OUT_DIR / 's6_canonical_trades.csv'}")
print(f"Columns: {list(trades_df.columns)}")
print(f"Rows: {len(trades_df)}")
print("="*60)
