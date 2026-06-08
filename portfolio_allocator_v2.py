#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-Aware Portfolio Allocator V2 -- Three Frozen Systems
UngerFink Pipeline / Andrea Unger Methodology

Improvements over V1:
  1. Capital ceiling: risk_amount = min(equity*0.25%, base_capital*0.5%)
     caps position growth as equity compounds -- prevents DD amplification
  2. Split capital pools:
       Pool A ($27k): Donchian + RSI_MR, regime-aware, 5% heat
       Pool B ($3k) : ConsecDownDays only, unconstrained by pool A, 5% heat
  3. BEAR regime enhancement:
       RSI_MR in BEAR: risk = 0.50% (vs 0.25% base)
       Reflects validated finding: RSI_MR made +22.6R in 2022 bear

Comparison output: V1 results (hardcoded) vs V2 (re-run).
Output: data/research_portfolio_allocator_v2/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse all data loading from V1 (import rather than duplicate)
from portfolio_allocator import (
    p, scan_symbols, load_ohlcv, add_indicators, build_data_matrices,
    detect_regime, check_exit, perf, Position,
    DON_N, DON_EXIT_N, DON_ATR_MULT, DON_MAX_BARS,
    RSI_N, RSI_OVERSOLD, MR_HOLD, MR_ATR_MULT,
    CONSEC_N, CD_HOLD, CD_ATR_MULT,
    EMA_N, ATR_N, MIN_BARS, EPS,
)

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "data" / "research_portfolio_allocator_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# V2 CONSTANTS
# =============================================================================

BASE_CAPITAL   = 30_000.0   # original starting capital (ceiling reference)
POOL_A_CAPITAL = 27_000.0   # Donchian + RSI_MR
POOL_B_CAPITAL =  3_000.0   # ConsecDownDays (bull specialist)
TOTAL_CAPITAL  = POOL_A_CAPITAL + POOL_B_CAPITAL   # $30k

# Position sizing with ceiling
BASE_RISK_PCT  = 0.0025     # 0.25%
BULL_DON_PCT   = 0.0035     # 0.35%  (Donchian in BULL: base + 0.10%)
BEAR_MR_PCT    = 0.0050     # 0.50%  (RSI_MR in BEAR: bear specialist bonus)
RISK_CAP_USD   = BASE_CAPITAL * 0.005   # $150 hard cap per trade

MAX_HEAT_PCT   = 0.05       # 5% of pool equity

# V1 results for comparison (hardcoded from prior run)
V1_RESULTS = {
    "cagr":     22.49,
    "max_dd":  -9.68,
    "years":    {2021: 67.04, 2022: -9.16, 2023: -1.95,
                 2024: 105.28, 2025: 70.18, 2026: -31.55},
}


# =============================================================================
# V2 RISK SIZING
# =============================================================================

def risk_amount_v2(equity: float, risk_pct: float) -> float:
    """
    Position size with capital ceiling.
    Never risks more than RISK_CAP_USD ($150) per trade regardless of how
    much equity has compounded -- prevents drawdown amplification.
    """
    return min(equity * risk_pct, RISK_CAP_USD)


def v2_risk_pct(system: str, regime: str) -> float:
    """Return risk fraction for this system × regime combo under V2 rules."""
    if system == "Donchian" and regime == "BULL":
        return BULL_DON_PCT    # 0.35%
    if system == "RSI_MR" and regime == "BEAR":
        return BEAR_MR_PCT     # 0.50%
    return BASE_RISK_PCT       # 0.25%


# =============================================================================
# V2 SIGNAL GENERATION
# =============================================================================

def generate_signals_v2(date: object, data: dict, regime: str,
                         systems: list[str]) -> list[dict]:
    """
    Generate signals for the given systems list.
    Signals sorted by risk_pct descending (higher-priority regime signals first).
    """
    signals = []

    for sym, df in data.items():
        if date not in df.index:
            continue
        row   = df.loc[date]
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
                rpct = v2_risk_pct("Donchian", regime)
                signals.append({
                    "symbol": sym, "system": "Donchian",
                    "close": close, "atr": atr,
                    "stop": close - DON_ATR_MULT * atr,
                    "risk_unit": DON_ATR_MULT * atr,
                    "risk_pct": rpct, "don_lower": d_lo,
                })

        if "RSI_MR" in systems and not np.isnan(rsi):
            if rsi < RSI_OVERSOLD:
                rpct = v2_risk_pct("RSI_MR", regime)
                signals.append({
                    "symbol": sym, "system": "RSI_MR",
                    "close": close, "atr": atr,
                    "stop": close - MR_ATR_MULT * atr,
                    "risk_unit": MR_ATR_MULT * atr,
                    "risk_pct": rpct, "don_lower": 0.0,
                })

        if "ConsecDownDays" in systems:
            if cd >= CONSEC_N and close > ema200:
                rpct = v2_risk_pct("ConsecDownDays", regime)
                signals.append({
                    "symbol": sym, "system": "ConsecDownDays",
                    "close": close, "atr": atr,
                    "stop": close - CD_ATR_MULT * atr,
                    "risk_unit": CD_ATR_MULT * atr,
                    "risk_pct": rpct, "don_lower": 0.0,
                })

    signals.sort(key=lambda s: s["risk_pct"], reverse=True)
    return signals


# =============================================================================
# SINGLE POOL BACKTEST ENGINE
# =============================================================================

def run_pool(pool_name: str, pool_capital: float, systems: list[str],
             dates: list, data: dict, breadth: pd.Series,
             btc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-driven backtest for one capital pool.
    Returns (equity_curve_df, trades_df).
    """
    equity    = pool_capital
    peak      = pool_capital
    open_pos: list[Position] = []
    closed:   list[dict]     = []
    eq_curve: list[dict]     = []

    for date in dates:
        regime, _ = detect_regime(date, btc_df, breadth)

        # ── Exits ────────────────────────────────────────────────────────────
        still_open = []
        for pos in open_pos:
            result = check_exit(pos, date, data)
            if result is not None:
                exit_price, reason = result
                r_mult   = (exit_price - pos.entry_price) / max(pos.risk_unit, EPS)
                pnl_usdt = pos.risk_amt * r_mult
                equity  += pnl_usdt
                peak     = max(peak, equity)
                closed.append({
                    "pool":        pool_name,
                    "entry_date":  pos.entry_date,
                    "exit_date":   date,
                    "symbol":      pos.symbol,
                    "system":      pos.system,
                    "entry_price": round(pos.entry_price, 8),
                    "exit_price":  round(exit_price, 8),
                    "bars_held":   pos.bars_held,
                    "net_r":       round(r_mult, 4),
                    "pnl_usdt":    round(pnl_usdt, 4),
                    "exit_reason": reason,
                    "regime":      regime,
                    "risk_pct":    round(pos.risk_pct * 100, 3),
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        # ── Current heat ─────────────────────────────────────────────────────
        occupied   = {p.symbol for p in open_pos}
        curr_heat  = sum(p.risk_pct for p in open_pos)

        # ── New signals ───────────────────────────────────────────────────────
        signals = generate_signals_v2(date, data, regime, systems)
        for sig in signals:
            if sig["symbol"] in occupied:
                continue
            new_heat = curr_heat + sig["risk_pct"]
            if new_heat > MAX_HEAT_PCT:
                continue
            r_amt = risk_amount_v2(equity, sig["risk_pct"])
            pos = Position(
                symbol      = sig["symbol"],
                system      = sig["system"],
                entry_date  = date,
                entry_price = sig["close"],
                stop        = sig["stop"],
                risk_unit   = sig["risk_unit"],
                risk_pct    = sig["risk_pct"],
                risk_amt    = r_amt,
                bars_held   = 0,
                don_lower   = sig.get("don_lower", 0.0),
            )
            open_pos.append(pos)
            occupied.add(sig["symbol"])
            curr_heat += sig["risk_pct"]

        dd_pct = (equity - peak) / max(peak, EPS) * 100.0
        eq_curve.append({
            "date":     date,
            "equity":   round(equity, 4),
            "peak":     round(peak, 4),
            "dd_pct":   round(dd_pct, 4),
            "open_pos": len(open_pos),
            "heat_pct": round(curr_heat * 100, 3),
            "regime":   regime,
            "pool":     pool_name,
        })

    return pd.DataFrame(eq_curve), pd.DataFrame(closed)


# =============================================================================
# PERFORMANCE HELPERS
# =============================================================================

def eq_to_annual_pct(eq_df: pd.DataFrame, start_cap: float) -> dict:
    """Year-by-year return % using equity-relative (not original-capital) denominator."""
    eq = eq_df.set_index("date")["equity"].copy()
    eq.index = pd.to_datetime(eq.index)
    results = {}
    for yr in range(eq.index.year.min(), eq.index.year.max() + 1):
        yr_eq = eq[eq.index.year == yr]
        if yr_eq.empty:
            continue
        prior = eq[eq.index.year < yr]
        beg   = float(prior.iloc[-1]) if not prior.empty else start_cap
        end   = float(yr_eq.iloc[-1])
        results[yr] = round((end - beg) / max(beg, EPS) * 100, 2)
    return results


def trade_stats(trades_df: pd.DataFrame, label: str) -> dict:
    if trades_df.empty:
        return {"label": label, "n": 0, "avg_r": 0.0, "win_rate": 0.0}
    rs = trades_df["net_r"].values
    return {
        "label":    label,
        "n":        len(rs),
        "avg_r":    round(float(np.mean(rs)), 4),
        "win_rate": round(float((rs > 0).mean() * 100), 1),
        "total_r":  round(float(np.sum(rs)), 2),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 68)
    p("  Portfolio Allocator V2 -- Three Improvements")
    p("  Ceiling cap | Split pools | BEAR RSI_MR enhancement")
    p("=" * 68)
    p(f"\n  Capital structure:")
    p(f"    Pool A: ${POOL_A_CAPITAL:,.0f}  Donchian + RSI_MR  (regime-aware)")
    p(f"    Pool B: ${POOL_B_CAPITAL:,.0f}   ConsecDownDays    (bull specialist)")
    p(f"    Risk cap: min(equity x 0.25%, ${RISK_CAP_USD:.0f}) per trade")
    p(f"    BEAR RSI_MR: {BEAR_MR_PCT*100:.2f}% risk (bear specialist bonus)")

    # ── Load data ────────────────────────────────────────────────────────────
    p("\n  [1/4] Loading data...")
    symbols = scan_symbols()
    mats    = build_data_matrices(symbols)
    data    = mats["data"]
    dates   = mats["dates"]
    breadth = mats["breadth"]
    btc_df  = data.get("BTC_USDT", pd.DataFrame())
    p(f"        Symbols: {len(data)}  |  Dates: {dates[0]} to {dates[-1]}  ({len(dates)} days)")

    # ── Run Pool A ────────────────────────────────────────────────────────────
    p("\n  [2/4] Pool A: Donchian + RSI_MR ($27k, regime-aware, ceiling cap)...")
    eq_a, tr_a = run_pool("A", POOL_A_CAPITAL, ["Donchian","RSI_MR"],
                           dates, data, breadth, btc_df)
    p(f"        Trades closed: {len(tr_a)}  |  Final equity: ${eq_a['equity'].iloc[-1]:,.2f}")

    # ── Run Pool B ────────────────────────────────────────────────────────────
    p("\n  [3/4] Pool B: ConsecDownDays ($3k, independent)...")
    eq_b, tr_b = run_pool("B", POOL_B_CAPITAL, ["ConsecDownDays"],
                           dates, data, breadth, btc_df)
    p(f"        Trades closed: {len(tr_b)}  |  Final equity: ${eq_b['equity'].iloc[-1]:,.2f}")

    # ── Combined V2 equity ────────────────────────────────────────────────────
    eq_a_s = eq_a.set_index("date")["equity"]
    eq_b_s = eq_b.set_index("date")["equity"]
    eq_combined = (eq_a_s + eq_b_s).reset_index()
    eq_combined.columns = ["date", "equity"]
    eq_combined["dd_pct"] = (
        (eq_combined["equity"] - eq_combined["equity"].cummax())
        / eq_combined["equity"].cummax().replace(0, np.nan) * 100
    ).fillna(0.0).round(4)

    tr_all = pd.concat([tr_a, tr_b], ignore_index=True)

    # ── Performance metrics ───────────────────────────────────────────────────
    p("\n  [4/4] Computing metrics...")

    m_v2_total = perf(eq_combined["equity"], TOTAL_CAPITAL, "V2 Combined")
    m_v2_a     = perf(eq_a["equity"],        POOL_A_CAPITAL, "V2 Pool A (Don+MR)")
    m_v2_b     = perf(eq_b["equity"],        POOL_B_CAPITAL, "V2 Pool B (ConsecDown)")

    yr_v2  = eq_to_annual_pct(eq_combined, TOTAL_CAPITAL)
    yr_v2a = eq_to_annual_pct(eq_a, POOL_A_CAPITAL)
    yr_v2b = eq_to_annual_pct(eq_b, POOL_B_CAPITAL)

    # Regime breakdown for Pool A
    reg_map = dict(zip(eq_a["date"], eq_a["regime"]))
    tr_a_reg = tr_a.copy()
    tr_a_reg["regime_at_entry"] = tr_a_reg["entry_date"].map(reg_map)

    # ── Print report ─────────────────────────────────────────────────────────
    p()
    p("=" * 68)
    p("  RESULTS")
    p("=" * 68)

    p("\n  PERFORMANCE COMPARISON  V1 vs V2")
    p(f"  {'':>30}  {'CAGR':>8}  {'MaxDD':>8}  {'End Equity':>12}")
    p("  " + "-" * 62)
    p(f"  {'V1 (original allocator)':>30}  "
      f"{V1_RESULTS['cagr']:>+7.2f}%  "
      f"{V1_RESULTS['max_dd']:>+7.2f}%  "
      f"${V1_RESULTS['cagr']/100*30000+30000:>10,.0f}  (approx)")
    for m in [m_v2_total, m_v2_a, m_v2_b]:
        p(f"  {m['label']:>30}  {m['cagr']:>+7.2f}%  {m['max_dd']:>+7.2f}%  "
          f"${m['end']:>10,.0f}")

    p("\n  YEAR-BY-YEAR COMPARISON (equity-relative %)")
    years = sorted(set(list(V1_RESULTS["years"].keys()) + list(yr_v2.keys())))
    p(f"  {'Year':>5}  {'V1':>8}  {'V2 Total':>9}  {'V2 PoolA':>9}  {'V2 PoolB':>9}")
    p("  " + "-" * 50)
    for yr in years:
        v1  = V1_RESULTS["years"].get(yr, 0.0)
        v2t = yr_v2.get(yr, 0.0)
        v2a = yr_v2a.get(yr, 0.0)
        v2b = yr_v2b.get(yr, 0.0)
        flag = "  <- bear" if yr == 2022 else ("  <- partial" if yr == 2026 else "")
        p(f"  {yr:>5}  {v1:>+7.2f}%  {v2t:>+8.2f}%  {v2a:>+8.2f}%  {v2b:>+8.2f}%{flag}")

    p("\n  REGIME DISTRIBUTION")
    regime_counts = eq_a["regime"].value_counts()
    total_days    = len(eq_a)
    for regime in ["BULL","MIXED","BEAR"]:
        cnt = int(regime_counts.get(regime, 0))
        p(f"    {regime:>5}: {cnt:>4} days  ({cnt/total_days*100:.1f}%)")

    p("\n  REGIME PREMIUM  (Pool A)")
    p(f"  {'Regime':>6}  {'N':>5}  {'AvgR':>8}  {'WR%':>7}  {'Risk':>7}")
    p("  " + "-" * 40)
    for regime in ["BULL","MIXED","BEAR"]:
        sub = tr_a_reg[tr_a_reg["regime_at_entry"] == regime]
        if sub.empty:
            continue
        rs   = sub["net_r"].values
        wr   = float((rs > 0).mean() * 100) if len(rs) > 0 else 0.0
        avg  = float(np.mean(rs))
        r_pct= float(sub["risk_pct"].mean())
        p(f"  {regime:>6}  {len(sub):>5}  {avg:>+7.4f}R  {wr:>6.1f}%  {r_pct:>5.2f}%")

    p("\n  SYSTEM BREAKDOWN  (all trades V2)")
    p(f"  {'System':>16}  {'Pool':>6}  {'N':>5}  {'AvgR':>8}  {'WR%':>7}  {'TotalR':>8}")
    p("  " + "-" * 58)
    for sys, pool_trades in [
        ("Donchian",       tr_a[tr_a["system"]=="Donchian"]),
        ("RSI_MR",         tr_a[tr_a["system"]=="RSI_MR"]),
        ("ConsecDownDays", tr_b),
    ]:
        if pool_trades.empty:
            continue
        rs   = pool_trades["net_r"].values
        wr   = float((rs > 0).mean() * 100)
        pool = "A" if sys != "ConsecDownDays" else "B"
        p(f"  {sys:>16}  {pool:>6}  {len(rs):>5}  {float(np.mean(rs)):>+7.4f}R  "
          f"{wr:>6.1f}%  {float(np.sum(rs)):>+7.2f}R")

    p("\n  CAPITAL CEILING EFFECT")
    if not tr_all.empty and "risk_pct" in tr_all.columns:
        # Show how many trades hit the cap vs uncapped
        tr_capped = tr_a.copy()
        tr_capped["would_be_risk"] = tr_capped["risk_pct"] / 100  # approximate
        p(f"    Risk cap: ${RISK_CAP_USD:.0f} per trade")
        p(f"    Pool A trades: {len(tr_a)}  Pool B trades: {len(tr_b)}")
        avg_heat_a = float(eq_a["heat_pct"].mean())
        max_heat_a = float(eq_a["heat_pct"].max())
        avg_heat_b = float(eq_b["heat_pct"].mean())
        max_heat_b = float(eq_b["heat_pct"].max())
        p(f"    Pool A heat: avg {avg_heat_a:.2f}%  max {max_heat_a:.2f}%")
        p(f"    Pool B heat: avg {avg_heat_b:.2f}%  max {max_heat_b:.2f}%")

    # ── Save outputs ─────────────────────────────────────────────────────────
    eq_a.to_csv(OUT_DIR / "pool_a_equity.csv", index=False)
    eq_b.to_csv(OUT_DIR / "pool_b_equity.csv", index=False)
    eq_combined.to_csv(OUT_DIR / "combined_equity.csv", index=False)
    tr_a.to_csv(OUT_DIR / "pool_a_trades.csv", index=False)
    tr_b.to_csv(OUT_DIR / "pool_b_trades.csv", index=False)
    tr_all.to_csv(OUT_DIR / "all_trades.csv", index=False)

    # Comparison CSV
    compare_rows = []
    for yr in years:
        compare_rows.append({
            "year":      yr,
            "v1_pct":    V1_RESULTS["years"].get(yr, 0.0),
            "v2_pct":    yr_v2.get(yr, 0.0),
            "v2_pool_a": yr_v2a.get(yr, 0.0),
            "v2_pool_b": yr_v2b.get(yr, 0.0),
        })
    pd.DataFrame(compare_rows).to_csv(OUT_DIR / "v1_vs_v2_comparison.csv", index=False)

    # Report
    with open(OUT_DIR / "v2_report.txt", "w", encoding="utf-8") as f:
        f.write("Portfolio Allocator V2 Report\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"V1: CAGR={V1_RESULTS['cagr']:+.2f}%  MaxDD={V1_RESULTS['max_dd']:+.2f}%\n")
        f.write(f"V2: CAGR={m_v2_total['cagr']:+.2f}%  MaxDD={m_v2_total['max_dd']:+.2f}%\n\n")
        f.write("Year-by-year (equity-relative %):\n")
        for row in compare_rows:
            f.write(f"  {row['year']}: V1={row['v1_pct']:+.2f}%  "
                    f"V2={row['v2_pct']:+.2f}%  "
                    f"PoolA={row['v2_pool_a']:+.2f}%  PoolB={row['v2_pool_b']:+.2f}%\n")
        f.write("\nSystem breakdown:\n")
        for sys, pool_trades in [("Donchian",tr_a[tr_a["system"]=="Donchian"]),
                                   ("RSI_MR",tr_a[tr_a["system"]=="RSI_MR"]),
                                   ("ConsecDownDays",tr_b)]:
            if not pool_trades.empty:
                rs = pool_trades["net_r"].values
                f.write(f"  {sys}: n={len(rs)}  avg_r={float(np.mean(rs)):+.4f}R  "
                        f"win={float((rs>0).mean()*100):.1f}%\n")

    p("\n  Files written:")
    for fn in ["pool_a_equity.csv","pool_b_equity.csv","combined_equity.csv",
               "pool_a_trades.csv","pool_b_trades.csv",
               "v1_vs_v2_comparison.csv","v2_report.txt"]:
        p(f"    {OUT_DIR / fn}")
    p()


if __name__ == "__main__":
    main()
