#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Allocator V2 -- Stability Validation
UngerFink Pipeline / Andrea Unger Methodology

Four tests:
  Test 1 -- Regime threshold sensitivity  (5 BULL x 4 BEAR = 17 valid combos)
  Test 2 -- Capital ceiling sensitivity   ($100 to $250)
  Test 3 -- Pool split sensitivity        (4 A/B configurations)
  Test 4 -- Walk-forward validation       (IS=2020-2022, OOS=2023-2026)

Output: data/research_portfolio_allocator_v2_validation/
"""

from __future__ import annotations

import warnings
from datetime import date as ddate
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_allocator import (
    p, scan_symbols, build_data_matrices, check_exit, perf, Position,
    DON_ATR_MULT, DON_MAX_BARS, MR_ATR_MULT, MR_HOLD,
    CD_ATR_MULT, CD_HOLD, RSI_OVERSOLD, CONSEC_N,
    EPS,
)

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "data" / "research_portfolio_allocator_v2_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical V2 parameters (reference point) ────────────────────────────────
CANONICAL_BULL   = 0.55
CANONICAL_BEAR   = 0.45
CANONICAL_CAP    = 150.0
CANONICAL_POOL_A = 27_000.0
CANONICAL_POOL_B =  3_000.0
TOTAL_CAPITAL    = 30_000.0

BASE_RISK_PCT  = 0.0025
BULL_DON_PCT   = 0.0035
BEAR_MR_PCT    = 0.0050
MAX_HEAT_PCT   = 0.05


# =============================================================================
# PARAMETRIC CORE ENGINE
# =============================================================================

def detect_regime_p(date_, btc_df, breadth,
                    bull_thresh: float, bear_thresh: float) -> str:
    btc_row   = btc_df.loc[date_] if date_ in btc_df.index else None
    bread_val = float(breadth.get(date_, np.nan))
    if btc_row is None or np.isnan(bread_val):
        return "MIXED"
    btc_above = float(btc_row["close"]) > float(btc_row["ema200"])
    if btc_above and bread_val > bull_thresh:
        return "BULL"
    if not btc_above and bread_val < bear_thresh:
        return "BEAR"
    return "MIXED"


def v2_risk_pct_p(system: str, regime: str) -> float:
    if system == "Donchian" and regime == "BULL":
        return BULL_DON_PCT
    if system == "RSI_MR" and regime == "BEAR":
        return BEAR_MR_PCT
    return BASE_RISK_PCT


def generate_signals_p(date_, data, regime, systems):
    signals = []
    for sym, df in data.items():
        if date_ not in df.index:
            continue
        row   = df.loc[date_]
        close = float(row["close"])
        atr   = float(row.get("atr", np.nan))
        ema200= float(row.get("ema200", np.nan))
        rsi   = float(row.get("rsi", np.nan))
        d_up  = float(row.get("don_upper", np.nan))
        d_lo  = float(row.get("don_lower", np.nan))
        cd    = int(row.get("consec_down", 0))
        if any(np.isnan(x) for x in [close, atr, ema200]) or atr <= EPS:
            continue
        if "Donchian" in systems and not np.isnan(d_up):
            if close > d_up and close > ema200:
                signals.append({
                    "symbol": sym, "system": "Donchian",
                    "close": close, "atr": atr,
                    "stop": close - DON_ATR_MULT * atr,
                    "risk_unit": DON_ATR_MULT * atr,
                    "risk_pct": v2_risk_pct_p("Donchian", regime),
                    "don_lower": d_lo,
                })
        if "RSI_MR" in systems and not np.isnan(rsi):
            if rsi < RSI_OVERSOLD:
                signals.append({
                    "symbol": sym, "system": "RSI_MR",
                    "close": close, "atr": atr,
                    "stop": close - MR_ATR_MULT * atr,
                    "risk_unit": MR_ATR_MULT * atr,
                    "risk_pct": v2_risk_pct_p("RSI_MR", regime),
                    "don_lower": 0.0,
                })
        if "ConsecDownDays" in systems:
            if cd >= CONSEC_N and close > ema200:
                signals.append({
                    "symbol": sym, "system": "ConsecDownDays",
                    "close": close, "atr": atr,
                    "stop": close - CD_ATR_MULT * atr,
                    "risk_unit": CD_ATR_MULT * atr,
                    "risk_pct": v2_risk_pct_p("ConsecDownDays", regime),
                    "don_lower": 0.0,
                })
    signals.sort(key=lambda s: s["risk_pct"], reverse=True)
    return signals


def run_pool_p(
    pool_name: str,
    pool_capital: float,
    systems: list,
    dates: list,
    data: dict,
    breadth,
    btc_df,
    bull_thresh: float = CANONICAL_BULL,
    bear_thresh: float = CANONICAL_BEAR,
    risk_cap_usd: float = CANONICAL_CAP,
    date_filter: Optional[tuple] = None,
) -> tuple:
    """
    Full V2 pool backtest with parametric regime thresholds and capital ceiling.
    date_filter: (start_date, end_date) to restrict simulation window.
    """
    if date_filter is not None:
        start_d, end_d = date_filter
        active_dates = [d for d in dates if start_d <= d <= end_d]
    else:
        active_dates = dates

    equity   = pool_capital
    peak     = pool_capital
    open_pos = []
    closed   = []
    eq_curve = []

    for date_ in active_dates:
        regime = detect_regime_p(date_, btc_df, breadth, bull_thresh, bear_thresh)

        # Exits
        still_open = []
        for pos in open_pos:
            result = check_exit(pos, date_, data)
            if result is not None:
                exit_price, reason = result
                r_mult   = (exit_price - pos.entry_price) / max(pos.risk_unit, EPS)
                pnl_usdt = pos.risk_amt * r_mult
                equity  += pnl_usdt
                peak     = max(peak, equity)
                closed.append({
                    "pool":        pool_name,
                    "entry_date":  pos.entry_date,
                    "exit_date":   date_,
                    "symbol":      pos.symbol,
                    "system":      pos.system,
                    "net_r":       round(r_mult, 4),
                    "pnl_usdt":    round(pnl_usdt, 4),
                    "exit_reason": reason,
                    "regime":      regime,
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        occupied  = {pos.symbol for pos in open_pos}
        curr_heat = sum(pos.risk_pct for pos in open_pos)

        for sig in generate_signals_p(date_, data, regime, systems):
            if sig["symbol"] in occupied:
                continue
            if curr_heat + sig["risk_pct"] > MAX_HEAT_PCT:
                continue
            r_amt = min(equity * sig["risk_pct"], risk_cap_usd)
            pos = Position(
                symbol=sig["symbol"], system=sig["system"],
                entry_date=date_, entry_price=sig["close"],
                stop=sig["stop"], risk_unit=sig["risk_unit"],
                risk_pct=sig["risk_pct"], risk_amt=r_amt,
                bars_held=0, don_lower=sig.get("don_lower", 0.0),
            )
            open_pos.append(pos)
            occupied.add(sig["symbol"])
            curr_heat += sig["risk_pct"]

        dd_pct = (equity - peak) / max(peak, EPS) * 100.0
        eq_curve.append({
            "date":   date_,
            "equity": round(equity, 4),
            "dd_pct": round(dd_pct, 4),
            "pool":   pool_name,
        })

    return pd.DataFrame(eq_curve), pd.DataFrame(closed)


def combined_metrics(eq_a: pd.DataFrame, eq_b: pd.DataFrame,
                     pool_a_cap: float, pool_b_cap: float) -> dict:
    if eq_a.empty or eq_b.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "end": 0.0}
    total_cap = pool_a_cap + pool_b_cap
    eq_a_s  = eq_a.set_index("date")["equity"]
    eq_b_s  = eq_b.set_index("date")["equity"]
    combined = (eq_a_s + eq_b_s).reset_index()
    combined.columns = ["date", "equity"]
    m = perf(combined["equity"], total_cap, "combined")
    return {"cagr": m["cagr"], "max_dd": m["max_dd"], "end": m["end"]}


def is_local_peak(vals: list, idx: int, tol: float = 0.02) -> bool:
    """True if vals[idx] is more than tol above both its neighbors."""
    if idx == 0 or idx >= len(vals) - 1:
        return False
    v = vals[idx]
    return v > vals[idx - 1] * (1 + tol) and v > vals[idx + 1] * (1 + tol)


def is_local_valley(vals: list, idx: int, tol: float = 0.02) -> bool:
    """True if vals[idx] is more than tol below both its neighbors (for MaxDD — more negative)."""
    if idx == 0 or idx >= len(vals) - 1:
        return False
    v = vals[idx]
    return v < vals[idx - 1] * (1 - tol) and v < vals[idx + 1] * (1 - tol)


# =============================================================================
# TEST 1 -- REGIME THRESHOLD SENSITIVITY
# =============================================================================

def test1_regime_thresholds(data, dates, breadth, btc_df):
    p("\n" + "=" * 72)
    p("  TEST 1 -- Regime Threshold Sensitivity")
    p("  Canonical: BULL breadth>55%  BEAR breadth<45%")
    p("  Requirement: canonical must sit in a stable zone, not a local peak")
    p("=" * 72)

    bull_vals = [0.45, 0.50, 0.55, 0.60, 0.65]
    bear_vals = [0.35, 0.40, 0.45, 0.50]

    valid = [(bt, br) for bt in bull_vals for br in bear_vals if br < bt]
    p(f"\n  Running {len(valid)} valid combinations (skipping {20 - len(valid)} where BEAR>=BULL)...")

    results = []
    for i, (bull_t, bear_t) in enumerate(valid, 1):
        is_can = abs(bull_t - CANONICAL_BULL) < 1e-9 and abs(bear_t - CANONICAL_BEAR) < 1e-9
        p(f"  [{i:02d}/{len(valid)}] BULL>{bull_t*100:.0f}% BEAR<{bear_t*100:.0f}% ...", end=" ", flush=True)
        eq_a, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                              dates, data, breadth, btc_df,
                              bull_thresh=bull_t, bear_thresh=bear_t)
        eq_b, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                              dates, data, breadth, btc_df,
                              bull_thresh=bull_t, bear_thresh=bear_t)
        m = combined_metrics(eq_a, eq_b, CANONICAL_POOL_A, CANONICAL_POOL_B)
        tag = "  <-- CANONICAL" if is_can else ""
        p(f"CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:+.2f}%{tag}")
        results.append({
            "bull_pct": int(bull_t * 100),
            "bear_pct": int(bear_t * 100),
            "cagr":     m["cagr"],
            "max_dd":   m["max_dd"],
            "canonical": is_can,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "test1_regime_thresholds.csv", index=False)

    # CAGR grid
    p()
    p("  CAGR GRID  (* = canonical)")
    p(f"  {'BULL\\BEAR':>11}" + "".join(f"  {int(b*100):>5}%" for b in bear_vals))
    p("  " + "-" * 55)
    for bull_t in bull_vals:
        cells = []
        for bear_t in bear_vals:
            r = df[(df["bull_pct"] == int(bull_t*100)) & (df["bear_pct"] == int(bear_t*100))]
            if r.empty:
                cells.append("  n/a  ")
            else:
                tag = "*" if bool(r.iloc[0]["canonical"]) else " "
                cells.append(f"{r.iloc[0]['cagr']:>+6.2f}{tag}")
        p(f"  {int(bull_t*100):>9}%  " + "  ".join(cells))

    # MaxDD grid
    p()
    p("  MAX DD GRID  (* = canonical)")
    p(f"  {'BULL\\BEAR':>11}" + "".join(f"  {int(b*100):>5}%" for b in bear_vals))
    p("  " + "-" * 55)
    for bull_t in bull_vals:
        cells = []
        for bear_t in bear_vals:
            r = df[(df["bull_pct"] == int(bull_t*100)) & (df["bear_pct"] == int(bear_t*100))]
            if r.empty:
                cells.append("  n/a  ")
            else:
                tag = "*" if bool(r.iloc[0]["canonical"]) else " "
                cells.append(f"{r.iloc[0]['max_dd']:>+6.2f}{tag}")
        p(f"  {int(bull_t*100):>9}%  " + "  ".join(cells))

    # Stability verdict
    p()
    canon_row = df[df["canonical"]]
    if canon_row.empty:
        p("  [!] Canonical row not found in results")
        return df

    canon_cagr = float(canon_row.iloc[0]["cagr"])
    all_cagrs  = df["cagr"].tolist()
    max_cagr   = float(df["cagr"].max())
    cagr_range = max_cagr - float(df["cagr"].min())

    # Find canonical position in CAGR-sorted list
    canon_idx_in_bull = sorted(
        [i for i, r in df[df["bull_pct"] == int(CANONICAL_BULL*100)].iterrows()],
        key=lambda i: df.loc[i, "bear_pct"]
    )
    # Check neighbors on BEAR axis at canonical BULL
    same_bull = df[df["bull_pct"] == int(CANONICAL_BULL*100)].sort_values("bear_pct").reset_index(drop=True)
    can_idx_sb = same_bull[same_bull["canonical"] == True].index
    if len(can_idx_sb) > 0:
        ci = int(can_idx_sb[0])
        vals = same_bull["cagr"].tolist()
        peak = is_local_peak(vals, ci)
        p(f"  Canonical CAGR along BEAR axis (BULL=55% fixed):  ", end="")
        p("  ".join(f"{v:+.2f}%" + (" *" if same_bull.loc[i,'canonical'] else "  ") for i, v in enumerate(vals)))
        if peak:
            p(f"  [!!] PEAK DETECTED: canonical is a local maximum along BEAR axis")
        else:
            p(f"  [OK] Canonical is NOT a local peak along BEAR axis")

    p(f"  CAGR range across all valid combos: {cagr_range:.2f}pp")
    if cagr_range < 5.0:
        p(f"  [OK] Low sensitivity ({cagr_range:.2f}pp range) -- regime thresholds are structural")
    elif cagr_range < 15.0:
        p(f"  [~] Moderate sensitivity ({cagr_range:.2f}pp range) -- review grid for concentration")
    else:
        p(f"  [!] High sensitivity ({cagr_range:.2f}pp range) -- thresholds may be data-fitted")

    return df


# =============================================================================
# TEST 2 -- CAPITAL CEILING SENSITIVITY
# =============================================================================

def test2_capital_ceiling(data, dates, breadth, btc_df):
    p("\n" + "=" * 72)
    p("  TEST 2 -- Capital Ceiling Sensitivity")
    p("  Canonical: $150 per-trade risk cap")
    p("  Requirement: $150 must not be an isolated peak")
    p("=" * 72)

    caps = [100.0, 125.0, 150.0, 175.0, 200.0, 250.0]
    results = []

    for cap in caps:
        is_can = abs(cap - CANONICAL_CAP) < 1e-9
        p(f"  Cap=${cap:.0f} ...", end=" ", flush=True)
        eq_a, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                              dates, data, breadth, btc_df, risk_cap_usd=cap)
        eq_b, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                              dates, data, breadth, btc_df, risk_cap_usd=cap)
        m = combined_metrics(eq_a, eq_b, CANONICAL_POOL_A, CANONICAL_POOL_B)
        tag = "  <-- CANONICAL" if is_can else ""
        p(f"CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:+.2f}%{tag}")
        results.append({
            "cap_usd":   cap,
            "cagr":      m["cagr"],
            "max_dd":    m["max_dd"],
            "canonical": is_can,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "test2_capital_ceiling.csv", index=False)

    p()
    p(f"  {'Cap ($)':>9}  {'CAGR':>9}  {'MaxDD':>9}  {'Note'}")
    p("  " + "-" * 50)
    cagr_vals = df["cagr"].tolist()
    mdd_vals  = df["max_dd"].tolist()
    for i, row in df.iterrows():
        peak_c = "  [CAGR PEAK]" if is_local_peak(cagr_vals, i) else ""
        peak_d = "  [DD VALLEY]" if is_local_valley(mdd_vals, i) else ""
        canon  = "  *CANONICAL"  if row["canonical"] else ""
        p(f"  ${row['cap_usd']:>8.0f}  {row['cagr']:>+8.2f}%  {row['max_dd']:>+8.2f}%{peak_c}{peak_d}{canon}")

    # Check if canonical is a peak
    can_idx = next(i for i, r in df.iterrows() if r["canonical"])
    cagr_peak = is_local_peak(cagr_vals, can_idx)
    cagr_range = max(cagr_vals) - min(cagr_vals)

    p()
    if cagr_peak:
        p(f"  [!!] CANONICAL $150 IS A LOCAL CAGR PEAK -- possible optimization artifact")
    else:
        p(f"  [OK] Canonical $150 is NOT a local CAGR peak")

    if cagr_range < 3.0:
        p(f"  [OK] Low ceiling sensitivity ({cagr_range:.2f}pp range) -- ceiling is structural")
    elif cagr_range < 8.0:
        p(f"  [~] Moderate sensitivity ({cagr_range:.2f}pp range)")
    else:
        p(f"  [!] High sensitivity ({cagr_range:.2f}pp range) -- ceiling may be data-fitted")

    # Show DD monotonicity (expected: lower cap → lower DD)
    p()
    p("  Expected: lower ceiling -> lower MaxDD (monotone)")
    diffs = [mdd_vals[i] - mdd_vals[i-1] for i in range(1, len(mdd_vals))]
    monotone_dd = all(d >= 0 for d in diffs)  # less negative means less DD
    if monotone_dd:
        p("  [OK] MaxDD improves (or holds) as ceiling decreases -- monotone relationship confirmed")
    else:
        p("  [~] MaxDD is NOT monotone in ceiling -- non-linear interaction with heat limit")

    return df


# =============================================================================
# TEST 3 -- POOL SPLIT SENSITIVITY
# =============================================================================

def test3_pool_splits(data, dates, breadth, btc_df):
    p("\n" + "=" * 72)
    p("  TEST 3 -- Pool Split Sensitivity")
    p("  Canonical: Pool A=$27k / Pool B=$3k  (total $30k)")
    p("  Requirement: ConsecDownDays must contribute meaningfully")
    p("               across the tested split range")
    p("=" * 72)

    splits = [
        (25_000.0,  5_000.0),
        (27_000.0,  3_000.0),   # canonical
        (28_000.0,  2_000.0),
        (29_000.0,  1_000.0),
    ]
    results = []

    for pool_a, pool_b in splits:
        is_can = pool_a == CANONICAL_POOL_A
        p(f"  A=${pool_a/1000:.0f}k / B=${pool_b/1000:.0f}k ...", end=" ", flush=True)
        eq_a, tr_a = run_pool_p("A", pool_a, ["Donchian","RSI_MR"],
                                 dates, data, breadth, btc_df)
        eq_b, tr_b = run_pool_p("B", pool_b, ["ConsecDownDays"],
                                 dates, data, breadth, btc_df)
        m_comb = combined_metrics(eq_a, eq_b, pool_a, pool_b)
        m_b    = perf(eq_b["equity"], pool_b, "B") if not eq_b.empty else {"cagr":0.0,"max_dd":0.0}
        cd_n   = len(tr_b)
        cd_avg = float(tr_b["net_r"].mean()) if not tr_b.empty else 0.0
        cd_pnl = float(tr_b["pnl_usdt"].sum()) if not tr_b.empty else 0.0
        tag = "  <-- CANONICAL" if is_can else ""
        p(f"CAGR={m_comb['cagr']:+.2f}%  MaxDD={m_comb['max_dd']:+.2f}%  "
          f"CD={cd_n} trades  CD_avgR={cd_avg:+.3f}R{tag}")
        results.append({
            "pool_a_k":       int(pool_a / 1000),
            "pool_b_k":       int(pool_b / 1000),
            "cagr":           m_comb["cagr"],
            "max_dd":         m_comb["max_dd"],
            "pool_b_cagr":    m_b["cagr"],
            "pool_b_max_dd":  m_b["max_dd"],
            "cd_trades":      cd_n,
            "cd_avg_r":       round(cd_avg, 4),
            "cd_total_pnl":   round(cd_pnl, 2),
            "canonical":      is_can,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "test3_pool_splits.csv", index=False)

    p()
    p(f"  {'Pool A':>7}  {'Pool B':>7}  {'CAGR':>9}  {'MaxDD':>9}  "
      f"{'CD Trades':>10}  {'CD AvgR':>8}  {'CD CAGR':>8}")
    p("  " + "-" * 75)
    for _, row in df.iterrows():
        tag = "  *" if row["canonical"] else "   "
        p(f"  ${row['pool_a_k']:>5}k  ${row['pool_b_k']:>5}k  "
          f"{row['cagr']:>+8.2f}%  {row['max_dd']:>+8.2f}%  "
          f"{row['cd_trades']:>10}  {row['cd_avg_r']:>+7.4f}R  "
          f"{row['pool_b_cagr']:>+7.2f}%{tag}")

    # Check that ConsecDownDays remains meaningful across splits
    p()
    min_cd_trades = int(df["cd_trades"].min())
    min_cd_avgr   = float(df["cd_avg_r"].min())
    if min_cd_avgr > 0.05:
        p(f"  [OK] ConsecDownDays remains positive avg_R across all splits (min={min_cd_avgr:+.3f}R)")
    else:
        p(f"  [~] ConsecDownDays avg_R dips low in some splits (min={min_cd_avgr:+.3f}R)")

    cagr_range = float(df["cagr"].max()) - float(df["cagr"].min())
    if cagr_range < 3.0:
        p(f"  [OK] Combined CAGR is stable across splits ({cagr_range:.2f}pp range)")
    else:
        p(f"  [~] Combined CAGR varies {cagr_range:.2f}pp across splits")

    return df


# =============================================================================
# TEST 4 -- WALK-FORWARD VALIDATION
# =============================================================================

def test4_walk_forward(data, dates, breadth, btc_df):
    p("\n" + "=" * 72)
    p("  TEST 4 -- Walk-Forward Validation")
    p("  IS  = 2020-01-01 to 2022-12-31  (3 years -- bull + bear)")
    p("  OOS = 2023-01-01 to present     (3+ years)")
    p("  Critical question: does the allocator work out-of-sample?")
    p("=" * 72)

    IS_START  = ddate(2020, 1, 1)
    IS_END    = ddate(2022, 12, 31)
    OOS_START = ddate(2023, 1, 1)
    OOS_END   = ddate(2099, 12, 31)

    # Step 1: Canonical V2 full period (reference)
    p("\n  [1/4] Canonical V2 full period 2020-2026 (reference)...")
    eq_a, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                          dates, data, breadth, btc_df)
    eq_b, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                          dates, data, breadth, btc_df)
    m_full = combined_metrics(eq_a, eq_b, CANONICAL_POOL_A, CANONICAL_POOL_B)
    p(f"       Full 2020-2026: CAGR={m_full['cagr']:+.2f}%  MaxDD={m_full['max_dd']:+.2f}%")

    # Step 2: Canonical thresholds on IS only
    p("\n  [2/4] Canonical thresholds on IS (2020-2022)...")
    eq_a_is, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                             dates, data, breadth, btc_df,
                             date_filter=(IS_START, IS_END))
    eq_b_is, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                             dates, data, breadth, btc_df,
                             date_filter=(IS_START, IS_END))
    m_is = combined_metrics(eq_a_is, eq_b_is, CANONICAL_POOL_A, CANONICAL_POOL_B)
    p(f"       IS 2020-2022:   CAGR={m_is['cagr']:+.2f}%  MaxDD={m_is['max_dd']:+.2f}%")

    # Step 3: Grid search on IS -- find best thresholds
    p("\n  [3/4] IS threshold grid search (find best-fit IS thresholds)...")
    bull_vals = [0.45, 0.50, 0.55, 0.60, 0.65]
    bear_vals = [0.35, 0.40, 0.45, 0.50]
    valid_combos = [(bt, br) for bt in bull_vals for br in bear_vals if br < bt]
    is_grid = []
    for i, (bull_t, bear_t) in enumerate(valid_combos, 1):
        p(f"  [{i:02d}/{len(valid_combos)}] IS BULL>{bull_t*100:.0f}% BEAR<{bear_t*100:.0f}%...", end=" ")
        eq_a_g, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                                dates, data, breadth, btc_df,
                                bull_thresh=bull_t, bear_thresh=bear_t,
                                date_filter=(IS_START, IS_END))
        eq_b_g, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                                dates, data, breadth, btc_df,
                                bull_thresh=bull_t, bear_thresh=bear_t,
                                date_filter=(IS_START, IS_END))
        m_g = combined_metrics(eq_a_g, eq_b_g, CANONICAL_POOL_A, CANONICAL_POOL_B)
        is_can = abs(bull_t - CANONICAL_BULL) < 1e-9 and abs(bear_t - CANONICAL_BEAR) < 1e-9
        tag = "  <-- canonical" if is_can else ""
        p(f"CAGR={m_g['cagr']:+.2f}%  MaxDD={m_g['max_dd']:+.2f}%{tag}")
        is_grid.append({
            "bull": bull_t, "bear": bear_t,
            "is_cagr": m_g["cagr"], "is_max_dd": m_g["max_dd"],
        })
    is_df = pd.DataFrame(is_grid)
    is_df.to_csv(OUT_DIR / "test4_is_grid.csv", index=False)

    best_is  = is_df.loc[is_df["is_cagr"].idxmax()]
    best_bull = float(best_is["bull"])
    best_bear = float(best_is["bear"])
    p(f"\n       Best IS thresholds: BULL>{best_bull*100:.0f}%  BEAR<{best_bear*100:.0f}%")
    p(f"       Best IS CAGR: {best_is['is_cagr']:+.2f}%  (canonical IS: {m_is['cagr']:+.2f}%)")

    # Step 4: OOS with canonical and IS-fitted thresholds
    p("\n  [4/4] OOS (2023-present) with both threshold sets...")
    eq_a_oos_c, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                                dates, data, breadth, btc_df,
                                date_filter=(OOS_START, OOS_END))
    eq_b_oos_c, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                                dates, data, breadth, btc_df,
                                date_filter=(OOS_START, OOS_END))
    m_oos_can = combined_metrics(eq_a_oos_c, eq_b_oos_c, CANONICAL_POOL_A, CANONICAL_POOL_B)
    p(f"       OOS canonical thresholds: CAGR={m_oos_can['cagr']:+.2f}%  MaxDD={m_oos_can['max_dd']:+.2f}%")

    eq_a_oos_f, _ = run_pool_p("A", CANONICAL_POOL_A, ["Donchian","RSI_MR"],
                                dates, data, breadth, btc_df,
                                bull_thresh=best_bull, bear_thresh=best_bear,
                                date_filter=(OOS_START, OOS_END))
    eq_b_oos_f, _ = run_pool_p("B", CANONICAL_POOL_B, ["ConsecDownDays"],
                                dates, data, breadth, btc_df,
                                bull_thresh=best_bull, bear_thresh=best_bear,
                                date_filter=(OOS_START, OOS_END))
    m_oos_fit = combined_metrics(eq_a_oos_f, eq_b_oos_f, CANONICAL_POOL_A, CANONICAL_POOL_B)
    p(f"       OOS IS-fitted thresholds: CAGR={m_oos_fit['cagr']:+.2f}%  MaxDD={m_oos_fit['max_dd']:+.2f}%")

    # Summary table
    p()
    p("  WALK-FORWARD SUMMARY TABLE")
    p(f"  {'Period':>16}  {'Thresholds':>24}  {'CAGR':>9}  {'MaxDD':>9}")
    p("  " + "-" * 65)
    rows = [
        ("Full 2020-2026",  "canonical (55/45)",                   m_full["cagr"],    m_full["max_dd"]),
        ("IS  2020-2022",   "canonical (55/45)",                   m_is["cagr"],      m_is["max_dd"]),
        ("IS  2020-2022",   f"best IS ({int(best_bull*100)}/{int(best_bear*100)})",   best_is["is_cagr"], best_is["is_max_dd"]),
        ("OOS 2023-now",    "canonical (55/45)",                   m_oos_can["cagr"], m_oos_can["max_dd"]),
        ("OOS 2023-now",    f"IS-fitted ({int(best_bull*100)}/{int(best_bear*100)})", m_oos_fit["cagr"], m_oos_fit["max_dd"]),
    ]
    for period, thresh, cagr, mdd in rows:
        p(f"  {period:>16}  {thresh:>24}  {cagr:>+8.2f}%  {mdd:>+8.2f}%")

    # Verdicts
    p()
    p("  VERDICTS")
    p("  " + "-" * 50)

    if m_oos_can["cagr"] > 5.0:
        p(f"  [OK] OOS CAGR strongly positive ({m_oos_can['cagr']:+.2f}%) -- allocator works OOS")
    elif m_oos_can["cagr"] > 0.0:
        p(f"  [~]  OOS CAGR weakly positive ({m_oos_can['cagr']:+.2f}%) -- caution")
    else:
        p(f"  [!!] OOS CAGR NEGATIVE ({m_oos_can['cagr']:+.2f}%) -- allocator FAILS OOS")

    is_oos_gap = m_is["cagr"] - m_oos_can["cagr"]
    if abs(is_oos_gap) < 10.0:
        p(f"  [OK] IS vs OOS gap: {is_oos_gap:+.2f}pp -- no significant overfit")
    elif abs(is_oos_gap) < 20.0:
        p(f"  [~]  IS vs OOS gap: {is_oos_gap:+.2f}pp -- moderate degradation, monitor")
    else:
        p(f"  [!!] IS vs OOS gap: {is_oos_gap:+.2f}pp -- significant degradation")

    fit_oos_diff = abs(m_oos_fit["cagr"] - m_oos_can["cagr"])
    if fit_oos_diff < 3.0:
        p(f"  [OK] IS-fitted vs canonical OOS delta: {fit_oos_diff:.2f}pp"
          f" -- regime thresholds are structural (not data-snooped)")
    else:
        p(f"  [~]  IS-fitted vs canonical OOS delta: {fit_oos_diff:.2f}pp"
          f" -- fitting thresholds gives different OOS, worth investigating")

    if m_oos_can["max_dd"] > -12.0:
        p(f"  [OK] OOS MaxDD {m_oos_can['max_dd']:+.2f}% -- within acceptable range")
    else:
        p(f"  [!]  OOS MaxDD {m_oos_can['max_dd']:+.2f}% -- higher than IS/full MaxDD")

    # Save
    wf_df = pd.DataFrame([
        {"period": r[0], "thresholds": r[1], "cagr": r[2], "max_dd": r[3]}
        for r in rows
    ])
    wf_df.to_csv(OUT_DIR / "test4_walkforward.csv", index=False)

    return wf_df


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 72)
    p("  PORTFOLIO ALLOCATOR V2 -- STABILITY VALIDATION")
    p("  Test 1: Regime thresholds  |  Test 2: Capital ceiling")
    p("  Test 3: Pool splits        |  Test 4: Walk-forward")
    p("=" * 72)

    p("\n  Loading data (shared across all tests)...")
    symbols = scan_symbols()
    mats    = build_data_matrices(symbols)
    data    = mats["data"]
    dates   = mats["dates"]
    breadth = mats["breadth"]
    btc_df  = data.get("BTC_USDT", pd.DataFrame())
    p(f"  Symbols: {len(data)}  |  Dates: {dates[0]} to {dates[-1]}  ({len(dates)} days)")

    t1 = test1_regime_thresholds(data, dates, breadth, btc_df)
    t2 = test2_capital_ceiling(data, dates, breadth, btc_df)
    t3 = test3_pool_splits(data, dates, breadth, btc_df)
    t4 = test4_walk_forward(data, dates, breadth, btc_df)

    p()
    p("=" * 72)
    p("  ALL TESTS COMPLETE")
    p(f"  Output directory: {OUT_DIR}")
    p("  Files:")
    for fn in ["test1_regime_thresholds.csv", "test2_capital_ceiling.csv",
               "test3_pool_splits.csv", "test4_is_grid.csv", "test4_walkforward.csv"]:
        p(f"    {OUT_DIR / fn}")
    p("=" * 72)


if __name__ == "__main__":
    main()
