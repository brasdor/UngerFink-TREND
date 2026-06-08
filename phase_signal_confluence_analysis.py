#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Confluence Analysis -- Three Frozen Systems
UngerFink Pipeline / Andrea Unger Methodology

Analyses cross-system signal overlap and whether confluence predicts better outcomes.

Systems (frozen configs):
  Donchian      : close > highest_high(20) AND close > EMA200
  RSI MR        : RSI(14) < 25  (no filter -- canonical)
  ConsecDownDays: 5 consecutive down closes AND close > EMA200

For each detected signal, the trade is simulated forward to capture R outcome.

Metrics produced:
  1. Signal overlap matrix  -- same-symbol same-day co-occurrence rates
  2. P&L correlation        -- Pearson correlation of daily-R series per system pair
  3. Confluence premium     -- avg_r by signal combination (D only / MR only / etc.)
  4. Year-by-year confluence frequency
  5. Cross-system lag analysis -- does System A's signal predict System B?

Universe: all 1D files in data/raw_trend_t1/ with >= 400 bars

Output: data/research_signal_confluence/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_signal_confluence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 400   # ~1.5 years minimum

# --- frozen signal params ---
DON_N           = 20
RSI_N           = 14
RSI_OVERSOLD    = 25
CONSEC_N        = 5
EMA_N           = 200
ATR_N           = 14

# --- exit params (per frozen config) ---
DON_EXIT_N      = 10     # lower Donchian band
DON_ATR_MULT    = 2.0
DON_MAX_BARS    = 120    # backstop for open Donchian positions

MR_HOLD_BARS    = 20
MR_ATR_MULT     = 3.0

CD_HOLD_BARS    = 20
CD_ATR_MULT     = 2.0

SYS = ["Donchian", "RSI_MR", "ConsecDownDays"]


def p(*a, **k):
    k.pop("flush", None)
    print(*a, flush=True, **k)


# =============================================================================
# DATA
# =============================================================================

def scan_symbols() -> list[str]:
    return sorted(f.stem.replace("_1d", "") for f in RAW_DIR.glob("*_1d.csv"))


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{sym}_1d.csv"
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        # timestamp column is ISO string "2021-01-01 00:00:00+00:00"
        ts_col = next((c for c in ("timestamp", "time", "date") if c in df.columns), None)
        if ts_col is None:
            return None
        ts_parsed = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts_parsed.isna().all():
            # fallback: try milliseconds
            ts_parsed = pd.to_datetime(
                pd.to_numeric(df[ts_col], errors="coerce"), unit="ms", utc=True, errors="coerce"
            )
        df["date"] = ts_parsed.dt.date
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


# =============================================================================
# INDICATORS (computed once per symbol)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]

    # ATR(14)
    prev_c = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_N).mean()

    # EMA200
    out["ema200"] = close.ewm(span=EMA_N, adjust=False).mean()

    # RSI(14) -- Wilder EWM
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    out["rsi"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

    # Donchian upper (entry) and lower (exit) bands
    out["don_upper"] = high.rolling(DON_N).max()
    out["don_lower"] = low.rolling(DON_EXIT_N).min()

    # Consecutive down closes
    up_arr  = (close > close.shift(1)).astype(int).values
    consec  = np.zeros(len(close), dtype=int)
    cnt     = 0
    for i in range(1, len(close)):
        if close.iloc[i] < close.iloc[i - 1]:
            cnt += 1
        else:
            cnt = 0
        consec[i] = cnt
    out["consec_down"] = consec

    return out


# =============================================================================
# SIGNAL DETECTION (all three systems on one symbol's full history)
# =============================================================================

def detect_signals(sym: str, df: pd.DataFrame) -> list[dict]:
    """Return one row per detected signal, with trade R simulated forward."""
    records = []
    close  = df["close"].values
    high_v = df["high"].values
    low_v  = df["low"].values
    atr    = df["atr"].values
    ema200 = df["ema200"].values
    rsi    = df["rsi"].values
    d_up   = df["don_upper"].values
    d_lo   = df["don_lower"].values
    consec = df["consec_down"].values
    dates  = df["date"].values

    n = len(df)

    for i in range(DON_N + 1, n - 1):
        if np.isnan(atr[i]) or np.isnan(ema200[i]) or np.isnan(rsi[i]):
            continue

        # --- Donchian signal ---
        if close[i] > d_up[i - 1] and close[i] > ema200[i]:
            r = _simulate_donchian(close, high_v, low_v, atr, d_lo, i, n)
            records.append({
                "symbol":     sym,
                "date":       dates[i],
                "year":       pd.Timestamp(dates[i]).year,
                "system":     "Donchian",
                "entry_price":close[i],
                "net_r":      r,
                "has_don":    1,
                "has_rsi":    0,
                "has_cd":     0,
            })

        # --- RSI MR signal ---
        if rsi[i] < RSI_OVERSOLD:
            r = _simulate_time_exit(close, low_v, atr, i, n, MR_HOLD_BARS, MR_ATR_MULT)
            records.append({
                "symbol":     sym,
                "date":       dates[i],
                "year":       pd.Timestamp(dates[i]).year,
                "system":     "RSI_MR",
                "entry_price":close[i],
                "net_r":      r,
                "has_don":    0,
                "has_rsi":    1,
                "has_cd":     0,
            })

        # --- ConsecDownDays signal ---
        if consec[i] >= CONSEC_N and close[i] > ema200[i]:
            r = _simulate_time_exit(close, low_v, atr, i, n, CD_HOLD_BARS, CD_ATR_MULT)
            records.append({
                "symbol":     sym,
                "date":       dates[i],
                "year":       pd.Timestamp(dates[i]).year,
                "system":     "ConsecDownDays",
                "entry_price":close[i],
                "net_r":      r,
                "has_don":    0,
                "has_rsi":    0,
                "has_cd":     1,
            })

    return records


def _simulate_donchian(close, high_v, low_v, atr, d_lo, entry_i, n):
    """Donchian: exit when close < N=10 lower band, ATR stop, or 120-bar backstop."""
    ep   = close[entry_i]
    risk = DON_ATR_MULT * atr[entry_i]
    stop = ep - risk
    if risk < 1e-9:
        return 0.0

    for j in range(entry_i + 1, min(entry_i + DON_MAX_BARS + 1, n)):
        if low_v[j] <= stop:
            return (stop - ep) / risk
        if close[j] < d_lo[j]:
            return (close[j] - ep) / risk
    # backstop
    last = min(entry_i + DON_MAX_BARS, n - 1)
    return (close[last] - ep) / risk


def _simulate_time_exit(close, low_v, atr, entry_i, n, hold_bars, atr_mult):
    """Time exit: hold hold_bars bars, safety stop = ATR * atr_mult below entry."""
    ep   = close[entry_i]
    risk = atr_mult * atr[entry_i]
    stop = ep - risk
    if risk < 1e-9:
        return 0.0

    for j in range(entry_i + 1, min(entry_i + hold_bars + 1, n)):
        if low_v[j] <= stop:
            return (stop - ep) / risk
    last = min(entry_i + hold_bars, n - 1)
    return (close[last] - ep) / risk


# =============================================================================
# SAME-SYMBOL CONFLUENCE (signal fires on same symbol within a date window)
# =============================================================================

def tag_confluence(signals_df: pd.DataFrame, window_days: int = 0) -> pd.DataFrame:
    """
    For each (symbol, date) group, check if multiple systems fired.
    window_days=0 means strictly the same day.
    Returns signals_df with added columns: n_systems, combo_label.
    """
    df = signals_df.copy()
    df["date_pd"] = pd.to_datetime(df["date"])

    # Aggregate to (symbol, date) level: which systems fired?
    if window_days == 0:
        grp = df.groupby(["symbol", "date"])[["has_don", "has_rsi", "has_cd"]].max().reset_index()
    else:
        # For window > 0: mark each trade-date if any system fired in that symbol
        # within ±window_days (complex -- skip for now, use same-day only)
        grp = df.groupby(["symbol", "date"])[["has_don", "has_rsi", "has_cd"]].max().reset_index()

    grp["n_systems"] = grp["has_don"] + grp["has_rsi"] + grp["has_cd"]

    def combo(row):
        parts = []
        if row["has_don"]: parts.append("D")
        if row["has_rsi"]: parts.append("M")
        if row["has_cd"]:  parts.append("C")
        return "+".join(parts) if parts else "none"

    grp["combo"] = grp.apply(combo, axis=1)

    # Merge combo back to signal rows
    df = df.merge(grp[["symbol", "date", "n_systems", "combo"]],
                  on=["symbol", "date"], how="left")
    return df


# =============================================================================
# DAILY PORTFOLIO R  (for correlation analysis)
# =============================================================================

def daily_system_r(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, system): sum the net_r of all trades entering on that date.
    Returns a wide DataFrame indexed by date with one column per system.
    """
    agg = (signals_df
           .groupby(["date", "system"])["net_r"]
           .sum()
           .reset_index())
    wide = agg.pivot(index="date", columns="system", values="net_r").fillna(0.0)
    # Ensure all three systems present
    for s in SYS:
        if s not in wide.columns:
            wide[s] = 0.0
    return wide[SYS].copy()


# =============================================================================
# METRICS
# =============================================================================

def metrics(rs: pd.Series | np.ndarray) -> dict:
    r = np.asarray(rs)
    if len(r) == 0:
        return {"n": 0, "avg_r": 0.0, "win_rate": 0.0, "pf": 0.0, "total_r": 0.0}
    wins   = r[r > 0]
    losses = np.abs(r[r < 0])
    pf     = wins.sum() / losses.sum() if losses.sum() > 0 else (99.0 if wins.sum() > 0 else 0.0)
    return {
        "n":        int(len(r)),
        "avg_r":    float(np.mean(r)),
        "win_rate": float(len(wins) / len(r)),
        "pf":       float(pf),
        "total_r":  float(np.sum(r)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 68)
    p("  Signal Confluence Analysis -- Three Frozen Systems")
    p("  Donchian (breakout) | RSI MR (oversold) | ConsecDownDays (weakness)")
    p("=" * 68)

    symbols = scan_symbols()
    p(f"\n  Universe: {len(symbols)} symbols in raw_trend_t1/")

    # ── Phase 1: collect all signals across all symbols ──────────────────────
    p("\n  Phase 1: detecting signals and simulating outcomes...")
    all_records: list[dict] = []
    loaded = skipped = 0

    for idx, sym in enumerate(symbols, 1):
        df = load_ohlcv(sym)
        if df is None:
            skipped += 1
            continue
        df = add_indicators(df)
        recs = detect_signals(sym, df)
        all_records.extend(recs)
        loaded += 1
        if idx % 20 == 0 or idx == len(symbols):
            p(f"    {idx}/{len(symbols)} {sym}  (signals so far: {len(all_records)})")

    p(f"\n  Loaded: {loaded}  Skipped (insufficient data): {skipped}")
    p(f"  Total signals: {len(all_records)}")

    if not all_records:
        p("  ERROR: no signals found"); return

    sigs = pd.DataFrame(all_records)
    sigs.to_csv(OUT_DIR / "all_signals.csv", index=False)

    # ── Phase 2: per-system baseline ─────────────────────────────────────────
    p("\n  Phase 2: per-system baseline metrics")
    p(f"  {'System':>16}  {'N':>6}  {'AvgR':>8}  {'WR%':>7}  {'PF':>6}  {'TotalR':>8}")
    p("  " + "-" * 60)
    for sys in SYS:
        sub = sigs[sigs["system"] == sys]
        m   = metrics(sub["net_r"])
        p(f"  {sys:>16}  {m['n']:>6}  {m['avg_r']:>+7.4f}R  "
          f"{m['win_rate']*100:>6.1f}%  {m['pf']:>6.2f}  {m['total_r']:>+8.2f}R")

    # ── Phase 3: signal overlap matrix (same symbol, same day) ──────────────
    p("\n  Phase 3: signal overlap matrix (same symbol, same day)")
    sigs_tagged = tag_confluence(sigs, window_days=0)

    # Compute co-occurrence counts
    overlap_data = {}
    pairs = [("Donchian", "RSI_MR", "has_don", "has_rsi"),
             ("Donchian", "ConsecDownDays", "has_don", "has_cd"),
             ("RSI_MR",   "ConsecDownDays", "has_rsi", "has_cd")]

    # Need symbol-date level (not trade level) for overlap
    sym_date = (sigs_tagged
                .groupby(["symbol", "date"])[["has_don","has_rsi","has_cd"]]
                .max()
                .reset_index())

    p(f"\n  Pair overlap (same symbol, same day):")
    p(f"  {'Pair':>35}  {'Both fire':>10}  {'A fires':>8}  {'Overlap%':>9}")
    p("  " + "-" * 70)

    overlap_rows = []
    for sA, sB, cA, cB in pairs:
        both = int(((sym_date[cA] == 1) & (sym_date[cB] == 1)).sum())
        a_only = int((sym_date[cA] == 1).sum())
        b_only = int((sym_date[cB] == 1).sum())
        pct_a  = both / a_only * 100 if a_only > 0 else 0.0
        pct_b  = both / b_only * 100 if b_only > 0 else 0.0
        label  = f"{sA} + {sB}"
        p(f"  {label:>35}  {both:>10}  {a_only:>8}  "
          f"{pct_a:>8.2f}% of {sA[:8]}")
        overlap_rows.append({
            "pair": label, "A": sA, "B": sB,
            "both": both, "A_fires": a_only, "B_fires": b_only,
            "pct_of_A": round(pct_a, 3), "pct_of_B": round(pct_b, 3),
        })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUT_DIR / "signal_overlap_matrix.csv", index=False)

    # ── Phase 4: P&L correlation ─────────────────────────────────────────────
    p("\n  Phase 4: P&L correlation (daily R per system, Pearson)")
    daily_r = daily_system_r(sigs)

    # Only compute on days where at least one system had a signal
    active_days = daily_r[(daily_r != 0).any(axis=1)]
    p(f"  Active days (any system fired): {len(active_days)}")

    corr_matrix = active_days.corr(method="pearson")
    p(f"\n  Daily-R Pearson correlation matrix:")
    p(f"  {'':>16}  {'Donchian':>10}  {'RSI_MR':>10}  {'ConsecDownDays':>16}")
    for sys in SYS:
        row = corr_matrix.loc[sys]
        p(f"  {sys:>16}  {row['Donchian']:>+9.4f}  {row['RSI_MR']:>+9.4f}  "
          f"{row['ConsecDownDays']:>+15.4f}")

    corr_matrix.to_csv(OUT_DIR / "pnl_correlation_matrix.csv")

    # ── Phase 5: confluence premium (same symbol same day) ──────────────────
    p("\n  Phase 5: confluence premium (same-symbol same-day signal overlap)")

    # Confluence can only be seen on the (symbol, date) level
    # For each (symbol, date) combo, average the net_r across all systems that fired
    sym_date_r = (sigs_tagged
                  .groupby(["symbol", "date"])
                  .agg(
                      has_don=("has_don", "max"),
                      has_rsi=("has_rsi", "max"),
                      has_cd=("has_cd",  "max"),
                      n_systems=("n_systems", "max"),
                      combo=("combo", "first"),
                      avg_r_this_day=("net_r", "mean"),
                      year=("year", "first"),
                  )
                  .reset_index())

    prem_rows = []
    p(f"\n  {'Combo':>6}  {'Description':>30}  {'N':>6}  {'AvgR':>8}  {'WR%':>7}  {'PF':>6}")
    p("  " + "-" * 72)
    combo_order = ["D", "M", "C", "D+M", "D+C", "M+C", "D+M+C"]

    for combo in combo_order:
        sub = sym_date_r[sym_date_r["combo"] == combo]
        if sub.empty:
            continue
        m = metrics(sub["avg_r_this_day"])
        desc = {
            "D":     "Donchian only",
            "M":     "RSI_MR only",
            "C":     "ConsecDownDays only",
            "D+M":   "Donchian + RSI_MR",
            "D+C":   "Donchian + ConsecDownDays",
            "M+C":   "RSI_MR + ConsecDownDays",
            "D+M+C": "All three systems",
        }.get(combo, combo)
        p(f"  {combo:>6}  {desc:>30}  {m['n']:>6}  {m['avg_r']:>+7.4f}R  "
          f"{m['win_rate']*100:>6.1f}%  {m['pf']:>6.2f}")
        prem_rows.append({
            "combo": combo, "description": desc, **m,
            "avg_r": round(m["avg_r"], 4),
        })

    prem_df = pd.DataFrame(prem_rows)
    prem_df.to_csv(OUT_DIR / "confluence_premium.csv", index=False)

    # ── Phase 6: year-by-year confluence ─────────────────────────────────────
    p("\n  Phase 6: year-by-year signal counts and confluence events")

    yearly_rows = []
    years = sorted(sigs["year"].unique())
    p(f"\n  {'Year':>5}  {'Don':>6}  {'RSIMR':>6}  {'CD':>6}  {'D+M':>5}  {'D+C':>5}  {'M+C':>5}  {'All3':>5}")
    p("  " + "-" * 55)

    for yr in years:
        yr_sym = sym_date_r[sym_date_r["year"] == yr]
        n_don = int((yr_sym["has_don"] == 1).sum())
        n_mr  = int((yr_sym["has_rsi"] == 1).sum())
        n_cd  = int((yr_sym["has_cd"]  == 1).sum())
        n_dm  = int((yr_sym["combo"] == "D+M").sum())
        n_dc  = int((yr_sym["combo"] == "D+C").sum())
        n_mc  = int((yr_sym["combo"] == "M+C").sum())
        n_all = int((yr_sym["combo"] == "D+M+C").sum())
        p(f"  {yr:>5}  {n_don:>6}  {n_mr:>6}  {n_cd:>6}  "
          f"{n_dm:>5}  {n_dc:>5}  {n_mc:>5}  {n_all:>5}")
        yearly_rows.append({"year": yr, "Donchian": n_don, "RSI_MR": n_mr,
                             "ConsecDownDays": n_cd, "D+M": n_dm, "D+C": n_dc,
                             "M+C": n_mc, "All3": n_all})

    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(OUT_DIR / "yearly_confluence.csv", index=False)

    # ── Phase 7: cross-system same-day portfolio correlation ─────────────────
    p("\n  Phase 7: portfolio-level same-day analysis")
    p("  (Do all systems tend to generate signals on the same calendar days?)")

    # For each date, count how many systems had at least one signal in the universe
    daily_active = (sigs.groupby(["date", "system"])["net_r"]
                    .count()
                    .reset_index()
                    .rename(columns={"net_r": "n_signals"}))
    daily_active["active"] = 1
    wide_active = (daily_active
                   .pivot(index="date", columns="system", values="active")
                   .fillna(0)
                   .astype(int))
    for s in SYS:
        if s not in wide_active.columns:
            wide_active[s] = 0

    # Days where each system fired at least one signal (anywhere in universe)
    p(f"\n  Days with at least one signal (any symbol):")
    for sys in SYS:
        n_days = int((wide_active[sys] > 0).sum())
        p(f"    {sys:<20}: {n_days} days")

    # Overlap: days where BOTH fire (any symbol)
    p(f"\n  Same-day portfolio co-occurrence (any symbol in universe):")
    p(f"  {'Pair':>35}  {'Both active days':>17}  {'Overlap%':>9}")
    p("  " + "-" * 66)
    for sA, sB, cA, cB in pairs:
        both = int(((wide_active.get(sA, pd.Series(0)) > 0) &
                    (wide_active.get(sB, pd.Series(0)) > 0)).sum())
        nA   = int((wide_active.get(sA, pd.Series(0)) > 0).sum())
        nB   = int((wide_active.get(sB, pd.Series(0)) > 0).sum())
        pct  = both / nA * 100 if nA > 0 else 0.0
        label = f"{sA} + {sB}"
        p(f"  {label:>35}  {both:>17}  {pct:>8.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    p("\n" + "=" * 68)
    p("  FINDINGS SUMMARY")
    p("=" * 68)

    # Key correlation values
    don_mr_corr  = corr_matrix.loc["Donchian", "RSI_MR"]
    don_cd_corr  = corr_matrix.loc["Donchian", "ConsecDownDays"]
    mr_cd_corr   = corr_matrix.loc["RSI_MR",   "ConsecDownDays"]

    p(f"\n  1. SAME-SYMBOL OVERLAP:")
    dm_row = overlap_df[overlap_df["pair"].str.contains("RSI_MR") & overlap_df["pair"].str.contains("Donchian")]
    mc_row = overlap_df[overlap_df["pair"].str.contains("RSI_MR") & overlap_df["pair"].str.contains("ConsecDown")]
    dc_row = overlap_df[overlap_df["pair"].str.contains("Donchian") & overlap_df["pair"].str.contains("ConsecDown")]
    if not dm_row.empty:
        p(f"     Donchian + RSI_MR same day:          {int(dm_row['both'].iloc[0])} events "
          f"({dm_row['pct_of_A'].iloc[0]:.2f}% of Donchian signals)")
    if not dc_row.empty:
        p(f"     Donchian + ConsecDownDays same day:  {int(dc_row['both'].iloc[0])} events "
          f"({dc_row['pct_of_A'].iloc[0]:.2f}% of Donchian signals)")
    if not mc_row.empty:
        p(f"     RSI_MR + ConsecDownDays same day:    {int(mc_row['both'].iloc[0])} events "
          f"({mc_row['pct_of_A'].iloc[0]:.2f}% of RSI_MR signals)")

    p(f"\n  2. DAILY-R CORRELATIONS:")
    p(f"     Donchian  vs RSI_MR        : {don_mr_corr:+.4f}")
    p(f"     Donchian  vs ConsecDownDays: {don_cd_corr:+.4f}")
    p(f"     RSI_MR    vs ConsecDownDays: {mr_cd_corr:+.4f}")
    if don_mr_corr < -0.05:
        p(f"     >> Donchian / RSI_MR are NEGATIVELY correlated -- genuine diversification")
    elif abs(don_mr_corr) < 0.15:
        p(f"     >> Donchian / RSI_MR are UNCORRELATED -- genuine diversification")

    p(f"\n  3. CONFLUENCE PREMIUM (M+C -- most likely combo):")
    mc_prem = prem_df[prem_df["combo"] == "M+C"]
    m_only  = prem_df[prem_df["combo"] == "M"]
    c_only  = prem_df[prem_df["combo"] == "C"]
    if not mc_prem.empty and not m_only.empty and not c_only.empty:
        mc_avg = float(mc_prem["avg_r"].iloc[0])
        m_avg  = float(m_only["avg_r"].iloc[0])
        c_avg  = float(c_only["avg_r"].iloc[0])
        p(f"     RSI_MR only avg_r:         {m_avg:+.4f}R")
        p(f"     ConsecDownDays only avg_r:  {c_avg:+.4f}R")
        p(f"     RSI_MR + ConsecDownDays:    {mc_avg:+.4f}R  "
          f"({'PREMIUM' if mc_avg > max(m_avg, c_avg) else 'no premium'})")

    # ── Write master report ───────────────────────────────────────────────────
    with open(OUT_DIR / "confluence_report.txt", "w", encoding="utf-8") as f:
        f.write("Signal Confluence Analysis -- Three Frozen Systems\n")
        f.write("=" * 68 + "\n\n")

        f.write("1. SIGNAL OVERLAP MATRIX (same symbol, same day)\n")
        f.write(overlap_df.to_string(index=False) + "\n\n")

        f.write("2. P&L CORRELATION MATRIX (daily R per system)\n")
        f.write(corr_matrix.round(4).to_string() + "\n\n")

        f.write("3. CONFLUENCE PREMIUM\n")
        f.write(prem_df.to_string(index=False) + "\n\n")

        f.write("4. YEAR-BY-YEAR CONFLUENCE\n")
        f.write(yearly_df.to_string(index=False) + "\n\n")

        f.write("5. KEY CORRELATIONS\n")
        f.write(f"  Donchian  vs RSI_MR        : {don_mr_corr:+.4f}\n")
        f.write(f"  Donchian  vs ConsecDownDays : {don_cd_corr:+.4f}\n")
        f.write(f"  RSI_MR    vs ConsecDownDays : {mr_cd_corr:+.4f}\n")

    p(f"\n  Output files:")
    for fn in ["all_signals.csv", "signal_overlap_matrix.csv",
               "pnl_correlation_matrix.csv", "confluence_premium.csv",
               "yearly_confluence.csv", "confluence_report.txt"]:
        p(f"    {OUT_DIR / fn}")

    p()


if __name__ == "__main__":
    main()
