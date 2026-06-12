"""
Shared engine for Short-Side Sweep V2 (funding gate applied to all tests).

Mechanics mirror System 7 (phase_t2_volcontraction_funding.py) exactly:
  - ATR(14) Wilder-style, SMA-seeded
  - EMA200 SMA-seeded
  - Bear filter: close < EMA200 (mandatory)
  - Funding gate: most recent funding_rate at signal bar ts >= threshold (mandatory)
  - Entry: next bar open after signal close
  - Stop: entry + ATR(signal_bar) * atr_mult (short -> stop above)
  - Exit: stop hit (high >= stop, filled at stop) OR time exit at open of
    bar signal+hold_bars. No overlapping trades per symbol.
  - r = (entry - exit) / (stop - entry)

Section 4.2 cost floor: 0.25R (Futures). 2022 must be positive.
Flag if 2025 > 40 percent of total R.
"""
import bisect
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT   = Path("data/futures_universe")
FUNDING_DIR = DATA_ROOT / "funding_rates"

FUNDING_THRESHOLD = 0.0001   # 0.01% per 8h -- the breakthrough gate
COST_FLOOR        = 0.25
ATR_PERIOD        = 14
EMA_PERIOD        = 200
WARMUP            = 300
MIN_TRADES_COMBO  = 30       # min trades for a combo to be canonical candidate

# bars per ~252 trading days, used for ATR percentile windows
PCT_WINDOW = {"2h": 252 * 12, "4h": 252 * 6, "6h": 252 * 4, "8h": 252 * 3, "1d": 252}
ATR_PCT_MIN_BARS = 50


# ---------------------------------------------------------------- indicators
def compute_atr(df):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    atr = np.full(len(tr), np.nan)
    if len(tr) < ATR_PERIOD:
        return atr
    atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
    alpha = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


def compute_ema(closes, period):
    n = len(closes)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period - 1] = closes[:period].mean()
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1 - alpha)
    return ema


def compute_atr_percentile(atr, window):
    n = len(atr)
    pct = np.full(n, np.nan)
    for i in range(ATR_PCT_MIN_BARS, n):
        start = max(0, i - window)
        w = atr[start:i]
        if len(w) < 10:
            continue
        pct[i] = float((w <= atr[i]).mean() * 100)
    return pct


def rolling_max_prior(arr, n):
    """max of arr[i-n:i] (prior n bars, excluding current). nan-padded head."""
    s = pd.Series(arr)
    return s.rolling(n).max().shift(1).values


def rolling_min_prior(arr, n):
    s = pd.Series(arr)
    return s.rolling(n).min().shift(1).values


# ---------------------------------------------------------------- data loading
def load_funding(symbol):
    path = FUNDING_DIR / f"{symbol}_funding.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "funding_time" not in df.columns or "funding_rate" not in df.columns:
            return None
        df = df.sort_values("funding_time")
        return (df["funding_time"].values.astype(np.int64),
                df["funding_rate"].values.astype(float))
    except Exception:
        return None


def get_rate_at(funding, ts_ms):
    if funding is None:
        return None
    times, rates = funding
    idx = bisect.bisect_right(times, ts_ms) - 1
    return rates[idx] if idx >= 0 else None


def funding_symbols():
    return sorted([f.stem.replace("_funding", "") for f in FUNDING_DIR.glob("*_funding.csv")])


def load_symbol(tf, symbol, with_atr_pct=False):
    """Load one symbol at timeframe tf, precompute indicators. None if missing/short."""
    path = DATA_ROOT / f"ohlcv_{tf}" / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < WARMUP + 20:
        return None
    closes = df["close"].values.astype(float)
    atr = compute_atr(df)
    ema200 = compute_ema(closes, EMA_PERIOD)
    d = {
        "ts":    df["timestamp"].values.astype(np.int64),
        "open":  df["open"].values.astype(float),
        "high":  df["high"].values.astype(float),
        "low":   df["low"].values.astype(float),
        "close": closes,
        "atr":   atr,
        "ema200": ema200,
        "bear":  closes < ema200,
        "n":     len(df),
    }
    if with_atr_pct:
        d["atr_pct"] = compute_atr_percentile(atr, PCT_WINDOW[tf])
    return d


def load_universe(tf, with_atr_pct=False, log_every=25):
    """Load all funding-data symbols at tf. Returns dict symbol -> arrays."""
    syms = funding_symbols()
    uni = {}
    for i, sym in enumerate(syms, 1):
        d = load_symbol(tf, sym, with_atr_pct=with_atr_pct)
        if d is not None:
            uni[sym] = d
        if log_every and i % log_every == 0:
            print(f"  [{tf}] loaded {i}/{len(syms)} symbols...", flush=True)
    print(f"  [{tf}] universe ready: {len(uni)} symbols usable", flush=True)
    return uni


# ---------------------------------------------------------------- simulation
def simulate(symbol, d, signal_idx, hb, am, funding,
             funding_threshold=FUNDING_THRESHOLD):
    """Simulate non-overlapping short trades from sorted signal bar indices.

    Mirrors System 7: signal on bar close i -> funding gate at ts[i] ->
    entry open[i+1], stop = entry + atr[i]*am, stop checked from bar i+1,
    time exit at open[i+hb]. Bar of exit is consumed (next signal >= exit+1).
    """
    ts, opens, highs = d["ts"], d["open"], d["high"]
    atrs = d["atr"]
    n = d["n"]
    trades = []
    next_free = 0
    for i in signal_idx:
        if i < next_free or i < WARMUP or i + 1 >= n:
            continue
        fr = get_rate_at(funding, int(ts[i]))
        if fr is None or fr < funding_threshold:
            continue
        if np.isnan(atrs[i]) or atrs[i] <= 0:
            continue
        entry = opens[i + 1]
        stop = entry + atrs[i] * am
        risk = stop - entry
        if risk <= 0:
            continue
        exit_bar = None
        r = None
        last = min(i + hb, n - 1)
        for j in range(i + 1, last + 1):
            if highs[j] >= stop:
                r = (entry - stop) / risk
                exit_bar = j
                break
            if j == i + hb:
                r = (entry - opens[j]) / risk
                exit_bar = j
                break
        if exit_bar is None:
            continue  # trade still open at end of data -- drop (System 7 same)
        trades.append({"symbol": symbol, "entry_ts": int(ts[i]),
                       "r": r, "funding": fr})
        next_free = exit_bar + 1
    return trades


# ---------------------------------------------------------------- analysis
def combo_stats(df_trades):
    """Aggregate stats for one combo's trades DataFrame (needs r, year cols)."""
    n = len(df_trades)
    if n == 0:
        return None
    r = df_trades["r"]
    wins = r[r > 0]
    losses = r[r <= 0]
    total_r = r.sum()
    stats = {
        "n": n,
        "avg_r": r.mean(),
        "total_r": total_r,
        "win_rate": (r > 0).mean(),
        "pf": (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
    }
    for yr in range(2019, 2027):
        yd = df_trades[df_trades["year"] == yr]
        stats[f"y{yr}_n"] = len(yd)
        stats[f"y{yr}_total"] = yd["r"].sum() if len(yd) else 0.0
        stats[f"y{yr}_avg"] = yd["r"].mean() if len(yd) else np.nan
    stats["y2022_positive"] = stats["y2022_total"] > 0 and stats["y2022_n"] > 0
    stats["r2025_share"] = (stats["y2025_total"] / total_r) if total_r > 0 else np.nan
    stats["flag_2025_conc"] = bool(total_r > 0 and stats["y2025_total"] / total_r > 0.40)
    return stats


def add_time_cols(df_trades):
    df_trades["dt"] = pd.to_datetime(df_trades["entry_ts"], unit="ms", utc=True)
    df_trades["year"] = df_trades["dt"].dt.year
    return df_trades


def stability_check(df_grid, canonical_row, primary_params, all_params):
    """Zone = vary each primary param by +/-1 grid step (others fixed at canonical).
    Includes canonical. PASS if >= 67% of zone combos have avg_r > 0.
    df_grid: per-combo stats with param columns. Returns (frac, n_zone, pass)."""
    zone_mask = pd.Series(False, index=df_grid.index)
    # canonical itself
    canon_mask = pd.Series(True, index=df_grid.index)
    for p in all_params:
        canon_mask &= df_grid[p] == canonical_row[p]
    zone_mask |= canon_mask
    for p in primary_params:
        vals = sorted(df_grid[p].unique())
        ci = vals.index(canonical_row[p])
        for step in (-1, 1):
            ni = ci + step
            if ni < 0 or ni >= len(vals):
                continue
            m = pd.Series(True, index=df_grid.index)
            for q in all_params:
                m &= df_grid[q] == (vals[ni] if q == p else canonical_row[q])
            zone_mask |= m
    zone = df_grid[zone_mask]
    if len(zone) == 0:
        return 0.0, 0, False
    frac = (zone["avg_r"] > 0).mean()
    return frac, len(zone), frac >= 0.67


def write_report(out_dir, test_name, df_grid, grid_param_cols, primary_params,
                 reference_line=""):
    """Rank grid, run gates per timeframe, write summary. Returns best rows dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_grid = df_grid.sort_values("avg_r", ascending=False).reset_index(drop=True)
    df_grid.to_csv(out_dir / "t1_grid_results.csv", index=False)

    lines = []
    def pr(s=""):
        print(s, flush=True)
        lines.append(s)

    pr("=" * 78)
    pr(f"T1 RESULTS: {test_name}")
    pr(f"Funding gate >= 0.01%/8h + EMA200 bear filter (mandatory on all combos)")
    pr(f"Cost floor (4.2): {COST_FLOOR}R   Min trades per combo: {MIN_TRADES_COMBO}")
    if reference_line:
        pr(reference_line)
    pr("=" * 78)

    best_per_tf = {}
    for tf in sorted(df_grid["tf"].unique()):
        sub = df_grid[df_grid["tf"] == tf]
        valid = sub[sub["n"] >= MIN_TRADES_COMBO]
        pr(f"\n--- Timeframe {tf.upper()} ---")
        pr(f"  combos: {len(sub)}  with >= {MIN_TRADES_COMBO} trades: {len(valid)}")
        if len(valid) == 0:
            pr("  NO VALID COMBOS (insufficient trades)")
            continue
        best = valid.iloc[valid["avg_r"].values.argmax()]
        best_per_tf[tf] = best
        param_str = " ".join(f"{p}={best[p]}" for p in grid_param_cols)
        pr(f"  BEST: {param_str}")
        pr(f"    n={int(best['n'])}  avg_r={best['avg_r']:+.4f}R  total={best['total_r']:+.1f}R  "
           f"wr={best['win_rate']*100:.1f}%  pf={best['pf']:.2f}")
        frac, nz, st_ok = stability_check(sub, best, primary_params, grid_param_cols)
        pr(f"    stability zone (+/-1 on {','.join(primary_params)}): "
           f"{frac*100:.0f}% of {nz} profitable  {'[PASS]' if st_ok else '[FAIL]'}")
        c42 = best["avg_r"] > COST_FLOOR
        pr(f"    4.2 cost floor: avg_r {best['avg_r']:+.4f}R vs {COST_FLOOR}R  "
           f"{'[PASS]' if c42 else '[FAIL]'}")
        pr(f"    2022: n={int(best['y2022_n'])} total={best['y2022_total']:+.2f}R  "
           f"{'[PASS]' if best['y2022_positive'] else '[FAIL]'}")
        share = best["r2025_share"]
        share_s = f"{share*100:.1f}%" if pd.notna(share) else "n/a"
        pr(f"    2025 concentration: {share_s} of total R  "
           f"{'[FLAG >40%]' if best['flag_2025_conc'] else '[OK]'}")
        pr(f"    year-by-year:")
        for yr in range(2019, 2027):
            if best[f"y{yr}_n"] > 0:
                pr(f"      {yr}: n={int(best[f'y{yr}_n']):4d}  total={best[f'y{yr}_total']:+8.2f}R  "
                   f"avg={best[f'y{yr}_avg']:+.4f}R")
        breakthrough = c42 and best["y2022_positive"] and st_ok
        if c42 and best["y2022_positive"]:
            pr(f"    *** BREAKTHROUGH CANDIDATE: avg_r >= 0.25R AND 2022 positive ***")
        pr(f"    verdict: {'BREAKTHROUGH' if breakthrough else ('REVIEW' if c42 else 'FAIL 4.2')}")

    pr("\n--- TOP 15 COMBOS OVERALL (n >= 30) ---")
    top = df_grid[df_grid["n"] >= MIN_TRADES_COMBO].head(15)
    hdr = "  " + "tf".ljust(4) + " ".join(p.ljust(7) for p in grid_param_cols) + \
          "n".rjust(6) + "avg_r".rjust(9) + "total".rjust(9) + "wr%".rjust(6) + "  2022R" + "  25conc"
    pr(hdr)
    for _, row in top.iterrows():
        share = row["r2025_share"]
        share_s = f"{share*100:5.1f}%" if pd.notna(share) else "  n/a "
        pr("  " + str(row["tf"]).ljust(4) +
           " ".join(str(row[p]).ljust(7) for p in grid_param_cols) +
           f"{int(row['n']):6d}{row['avg_r']:+9.4f}{row['total_r']:+9.1f}"
           f"{row['win_rate']*100:6.1f}  {row['y2022_total']:+6.1f}  {share_s}")

    report = "\n".join(lines)
    with open(out_dir / "t1_summary.txt", "w", encoding="ascii", errors="replace") as f:
        f.write(report)
    print(f"\nSaved: {out_dir}/t1_grid_results.csv, t1_summary.txt", flush=True)
    return best_per_tf
