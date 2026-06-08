#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-Aware Portfolio Allocator -- Three Frozen Systems
UngerFink Pipeline / Andrea Unger Methodology

Combines DonchianLong, RSI_MR, and ConsecDownDays into one shared capital pool
with regime-adjusted position sizing and a 5% max portfolio heat limit.

Regime Detection (daily):
  BULL : BTC above EMA200  AND  breadth > 55%
  BEAR : BTC below EMA200  AND  breadth < 45%
  MIXED: everything else

Position sizing:
  Base   : 0.25% risk per trade
  Bonus  : +0.10% if signal aligns with regime
           (Donchian in BULL, RSI_MR in BEAR, ConsecDownDays in BULL)
  Max heat: 5% of current equity at any time

Comparison:
  Combined  : all three systems share one $30k capital pool
  Independent: each system runs on its own $10k (total $30k deployed)

Output: data/research_portfolio_allocator/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_portfolio_allocator"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def p(*a, **k):
    k.pop("flush", None)
    print(*a, flush=True, **k)


# =============================================================================
# FROZEN CONFIGS
# =============================================================================

DON_N           = 20      # Donchian entry lookback
DON_EXIT_N      = 10      # Donchian exit (lower band)
DON_ATR_MULT    = 2.0     # Donchian initial stop
DON_MAX_BARS    = 120     # Donchian backstop

RSI_N           = 14
RSI_OVERSOLD    = 25
MR_HOLD         = 20
MR_ATR_MULT     = 3.0

CONSEC_N        = 5
CD_HOLD         = 20
CD_ATR_MULT     = 2.0

EMA_N           = 200
ATR_N           = 14

# Portfolio params
INITIAL_CAPITAL_PER_SYS = 10_000.0
N_SYSTEMS               = 3
COMBINED_CAPITAL        = INITIAL_CAPITAL_PER_SYS * N_SYSTEMS   # $30k
INDEPENDENT_CAPITAL     = INITIAL_CAPITAL_PER_SYS               # $10k each

BASE_RISK_PCT    = 0.0025   # 0.25%
BONUS_RISK_PCT   = 0.0010   # +0.10% for regime alignment
MAX_HEAT_PCT     = 0.05     # 5% max total open risk

# Independent system caps (frozen configs)
DON_MAX_POS     = 8
MR_MAX_POS      = 10
CD_MAX_POS      = 999       # uncapped

MIN_BARS        = 400
EPS             = 1e-12


# =============================================================================
# POSITION DATACLASS
# =============================================================================

@dataclass
class Position:
    symbol:       str
    system:       str         # "Donchian" | "RSI_MR" | "ConsecDownDays"
    entry_date:   object      # date
    entry_price:  float
    stop:         float       # initial stop price
    risk_unit:    float       # absolute price risk = entry - stop
    risk_pct:     float       # fraction of capital at risk (0.0025 or 0.0035)
    risk_amt:     float       # $ at risk at entry
    bars_held:    int = 0
    # Donchian-specific (updated daily)
    don_lower:    float = 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def scan_symbols() -> list[str]:
    return sorted(f.stem.replace("_1d", "") for f in RAW_DIR.glob("*_1d.csv"))


def load_ohlcv(sym: str) -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"{sym}_1d.csv"
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("timestamp","time","date") if c in df.columns), None)
        if ts_col is None:
            return None
        df["date"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.date
        for col in ("open","high","low","close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


# =============================================================================
# INDICATORS (computed once per symbol, stored in dict by date)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]

    prev_c = close.shift(1)
    tr  = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    out["atr"]    = tr.rolling(ATR_N).mean()
    out["ema200"] = close.ewm(span=EMA_N, adjust=False).mean()

    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/RSI_N, min_periods=RSI_N, adjust=False).mean()
    out["rsi"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

    out["don_upper"] = high.rolling(DON_N).max().shift(1)   # yesterday's high (no lookahead)
    out["don_lower"] = low.rolling(DON_EXIT_N).min()

    cd_arr = np.zeros(len(close), dtype=int)
    cnt    = 0
    for i in range(1, len(close)):
        cnt = cnt + 1 if close.iloc[i] < close.iloc[i-1] else 0
        cd_arr[i] = cnt
    out["consec_down"] = cd_arr

    return out


# =============================================================================
# PRECOMPUTE DATE-INDEXED MATRICES FOR REGIME + SIGNALS
# =============================================================================

def build_data_matrices(symbols: list[str]) -> dict:
    """
    Returns:
      data[sym] = df with indicators (date as index)
      dates     = sorted list of all trading dates
      breadth   = pd.Series (date) -> fraction of symbols with close > EMA200
      avg_rsi   = pd.Series (date) -> mean RSI across universe
    """
    data: dict = {}
    all_dates: set = set()

    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None:
            continue
        df = add_indicators(df)
        df = df.set_index("date")
        data[sym] = df
        all_dates.update(df.index)

    dates = sorted(all_dates)

    # Breadth: % above EMA200 per day
    above_ema = {sym: (data[sym]["close"] > data[sym]["ema200"]).astype(int)
                 for sym in data}
    breadth_df = pd.DataFrame(above_ema).reindex(dates)
    breadth     = breadth_df.mean(axis=1, skipna=True)

    # Average RSI per day
    rsi_df = pd.DataFrame({sym: data[sym]["rsi"] for sym in data}).reindex(dates)
    avg_rsi = rsi_df.mean(axis=1, skipna=True)

    return {"data": data, "dates": dates, "breadth": breadth, "avg_rsi": avg_rsi}


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(date: object, btc_df: pd.DataFrame,
                  breadth: pd.Series) -> tuple[str, dict]:
    """
    Returns (regime_label, regime_detail_dict).
    BULL : BTC above EMA200 AND breadth > 55%
    BEAR : BTC below EMA200 AND breadth < 45%
    MIXED: everything else
    """
    btc_row    = btc_df.loc[date] if date in btc_df.index else None
    bread_val  = float(breadth.get(date, np.nan))

    if btc_row is None or np.isnan(bread_val):
        return "MIXED", {}

    btc_close  = float(btc_row["close"])
    btc_ema    = float(btc_row["ema200"])
    btc_above  = btc_close > btc_ema

    if btc_above and bread_val > 0.55:
        regime = "BULL"
    elif not btc_above and bread_val < 0.45:
        regime = "BEAR"
    else:
        regime = "MIXED"

    return regime, {
        "btc_close": round(btc_close, 4),
        "btc_ema200": round(btc_ema, 4),
        "btc_above_ema": btc_above,
        "breadth_pct": round(bread_val * 100, 1),
    }


def regime_alignment_bonus(system: str, regime: str) -> float:
    """
    Returns extra risk fraction if signal aligns with current regime.
    Donchian      -> best in BULL (trend following)
    RSI_MR        -> best in BEAR (extreme oversold events)
    ConsecDownDays-> best in BULL (dip buying in uptrend)
    """
    aligned = (
        (system == "Donchian"       and regime == "BULL") or
        (system == "RSI_MR"         and regime == "BEAR") or
        (system == "ConsecDownDays" and regime == "BULL")
    )
    return BONUS_RISK_PCT if aligned else 0.0


# =============================================================================
# SIGNAL GENERATION FOR ONE DATE
# =============================================================================

def generate_signals(date: object, data: dict, regime: str) -> list[dict]:
    """Return list of signal dicts for all systems, ranked by risk_pct desc."""
    signals = []

    for sym, df in data.items():
        if date not in df.index:
            continue
        row = df.loc[date]
        close = float(row["close"])
        atr   = float(row.get("atr", np.nan))
        ema200= float(row.get("ema200", np.nan))
        rsi   = float(row.get("rsi", np.nan))
        d_up  = float(row.get("don_upper", np.nan))
        d_lo  = float(row.get("don_lower", np.nan))
        cd    = int(row.get("consec_down", 0))

        if any(np.isnan(x) for x in [close, atr, ema200]) or atr <= EPS:
            continue

        # --- Donchian ---
        if not np.isnan(d_up) and close > d_up and close > ema200:
            bonus   = regime_alignment_bonus("Donchian", regime)
            risk_pct= BASE_RISK_PCT + bonus
            stop    = close - DON_ATR_MULT * atr
            signals.append({
                "symbol":   sym, "system": "Donchian",
                "close":    close, "atr": atr, "ema200": ema200,
                "stop":     stop, "risk_unit": DON_ATR_MULT * atr,
                "risk_pct": risk_pct, "bonus": bonus > 0,
                "don_lower": d_lo,
            })

        # --- RSI MR ---
        if not np.isnan(rsi) and rsi < RSI_OVERSOLD:
            bonus   = regime_alignment_bonus("RSI_MR", regime)
            risk_pct= BASE_RISK_PCT + bonus
            stop    = close - MR_ATR_MULT * atr
            signals.append({
                "symbol":   sym, "system": "RSI_MR",
                "close":    close, "atr": atr, "ema200": ema200,
                "stop":     stop, "risk_unit": MR_ATR_MULT * atr,
                "risk_pct": risk_pct, "bonus": bonus > 0,
                "don_lower": 0.0,
            })

        # --- ConsecDownDays ---
        if cd >= CONSEC_N and close > ema200:
            bonus   = regime_alignment_bonus("ConsecDownDays", regime)
            risk_pct= BASE_RISK_PCT + bonus
            stop    = close - CD_ATR_MULT * atr
            signals.append({
                "symbol":   sym, "system": "ConsecDownDays",
                "close":    close, "atr": atr, "ema200": ema200,
                "stop":     stop, "risk_unit": CD_ATR_MULT * atr,
                "risk_pct": risk_pct, "bonus": bonus > 0,
                "don_lower": 0.0,
            })

    # Higher risk_pct = regime-aligned = higher priority
    signals.sort(key=lambda s: s["risk_pct"], reverse=True)
    return signals


# =============================================================================
# EXIT CHECK FOR ONE POSITION ON ONE DATE
# =============================================================================

def check_exit(pos: Position, date: object, data: dict) -> Optional[tuple[float, str]]:
    """
    Returns (exit_price, reason) or None if no exit.
    Updates pos.bars_held and pos.don_lower in place.
    """
    sym = pos.symbol.replace("/", "_").replace(":", "_")
    # Try to find the symbol in data with either format
    df_key = next((k for k in data if k == sym or k == pos.symbol.replace("/","_")), None)
    if df_key is None or date not in data[df_key].index:
        pos.bars_held += 1
        return None

    row       = data[df_key].loc[date]
    bar_low   = float(row["low"])
    bar_close = float(row["close"])
    d_lo      = float(row.get("don_lower", np.nan))

    # Update Donchian lower for Donchian positions
    if pos.system == "Donchian" and not np.isnan(d_lo):
        pos.don_lower = d_lo

    # Priority 1: stop hit (low pierced stop)
    if bar_low <= pos.stop:
        return (pos.stop, "stop")

    # Priority 2: system-specific exit
    if pos.system == "Donchian":
        if not np.isnan(pos.don_lower) and bar_close < pos.don_lower:
            return (bar_close, "don_lower")
        if pos.bars_held >= DON_MAX_BARS:
            return (bar_close, "backstop")
    elif pos.system == "RSI_MR":
        if pos.bars_held >= MR_HOLD:
            return (bar_close, "time_exit")
    elif pos.system == "ConsecDownDays":
        if pos.bars_held >= CD_HOLD:
            return (bar_close, "time_exit")

    pos.bars_held += 1
    return None


# =============================================================================
# COMBINED ALLOCATOR BACKTEST
# =============================================================================

def run_combined_allocator(dates: list, data: dict, breadth: pd.Series,
                            avg_rsi: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (equity_curve_df, trades_df, regime_log_df)
    """
    btc_key = "BTC_USDT"
    btc_df  = data.get(btc_key, pd.DataFrame())

    equity       = COMBINED_CAPITAL
    peak         = COMBINED_CAPITAL
    open_pos:    list[Position] = []
    closed:      list[dict]     = []
    eq_curve:    list[dict]     = []
    regime_log:  list[dict]     = []

    for date in dates:
        regime, regime_detail = detect_regime(date, btc_df, breadth)

        # ── exits ────────────────────────────────────────────────────────────
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
                    "regime_entry": pos.system,  # placeholder
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        # ── current heat ─────────────────────────────────────────────────────
        occupied_syms = {p.symbol for p in open_pos}
        current_heat  = sum(p.risk_pct for p in open_pos)

        # ── new signals ───────────────────────────────────────────────────────
        signals = generate_signals(date, data, regime)
        entries_today = 0
        for sig in signals:
            sym_key = sig["symbol"]
            if sym_key in occupied_syms:
                continue
            new_heat = current_heat + sig["risk_pct"]
            if new_heat > MAX_HEAT_PCT:
                continue
            risk_amt = equity * sig["risk_pct"]
            pos = Position(
                symbol      = sym_key,
                system      = sig["system"],
                entry_date  = date,
                entry_price = sig["close"],
                stop        = sig["stop"],
                risk_unit   = sig["risk_unit"],
                risk_pct    = sig["risk_pct"],
                risk_amt    = risk_amt,
                bars_held   = 0,
                don_lower   = sig.get("don_lower", 0.0),
            )
            open_pos.append(pos)
            occupied_syms.add(sym_key)
            current_heat += sig["risk_pct"]
            entries_today += 1

        dd_pct = (equity - peak) / max(peak, EPS) * 100.0
        eq_curve.append({
            "date":     date,
            "equity":   round(equity, 4),
            "peak":     round(peak, 4),
            "dd_pct":   round(dd_pct, 4),
            "open_pos": len(open_pos),
            "heat_pct": round(current_heat * 100, 3),
            "regime":   regime,
        })
        regime_log.append({
            "date":          date,
            "regime":        regime,
            "breadth_pct":   regime_detail.get("breadth_pct", np.nan),
            "btc_above_ema": regime_detail.get("btc_above_ema", np.nan),
            "entries":       entries_today,
            "open_pos":      len(open_pos),
        })

    return (pd.DataFrame(eq_curve),
            pd.DataFrame(closed),
            pd.DataFrame(regime_log))


# =============================================================================
# INDEPENDENT SYSTEMS BACKTEST (three separate capital pools)
# =============================================================================

def run_independent_system(system: str, dates: list, data: dict,
                            max_positions: int) -> pd.DataFrame:
    """Run one system independently with its own $10k capital."""
    equity    = INDEPENDENT_CAPITAL
    peak      = INDEPENDENT_CAPITAL
    open_pos: list[Position] = []
    closed:   list[dict]     = []
    eq_curve: list[dict]     = []

    for date in dates:
        # Exits
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
                    "exit_date": date, "symbol": pos.symbol,
                    "net_r": round(r_mult, 4), "pnl_usdt": round(pnl_usdt, 4),
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        occupied = {p.symbol for p in open_pos}

        # New signals (no regime bonus, fixed 0.25% risk)
        signals = generate_signals(date, data, "MIXED")  # no regime bonus
        for sig in signals:
            if sig["system"] != system:
                continue
            if sig["symbol"] in occupied:
                continue
            if len(open_pos) >= max_positions:
                break
            risk_amt = equity * BASE_RISK_PCT
            pos = Position(
                symbol=sig["symbol"], system=system,
                entry_date=date, entry_price=sig["close"],
                stop=sig["stop"], risk_unit=sig["risk_unit"],
                risk_pct=BASE_RISK_PCT, risk_amt=risk_amt,
                bars_held=0, don_lower=sig.get("don_lower",0.0),
            )
            open_pos.append(pos)
            occupied.add(sig["symbol"])

        dd_pct = (equity - peak) / max(peak, EPS) * 100.0
        eq_curve.append({
            "date":   date,
            "equity": round(equity, 4),
            "dd_pct": round(dd_pct, 4),
        })

    df = pd.DataFrame(eq_curve).set_index("date")
    return df


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def perf(equity_series: pd.Series, start_cap: float, label: str) -> dict:
    rs    = equity_series.values
    start = rs[0] if rs[0] > 0 else start_cap
    end   = rs[-1]
    n_days= len(rs)
    years = n_days / 365.25
    cagr  = ((end / start) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    # Max drawdown
    cum  = pd.Series(rs)
    peak = cum.cummax()
    dd   = (cum - peak) / peak.replace(0, np.nan) * 100.0
    max_dd = float(dd.min())

    return {
        "label":   label,
        "start":   round(start, 2),
        "end":     round(end, 2),
        "cagr":    round(cagr, 2),
        "max_dd":  round(max_dd, 2),
        "n_days":  n_days,
    }


def year_r(trades_df: pd.DataFrame, capital: float) -> dict:
    """Annual return % for a trades DataFrame with pnl_usdt column."""
    if trades_df.empty or "pnl_usdt" not in trades_df.columns:
        return {}
    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year
    annual = trades_df.groupby("year")["pnl_usdt"].sum()
    return {yr: round(float(pnl) / capital * 100, 2) for yr, pnl in annual.items()}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 68)
    p("  Regime-Aware Portfolio Allocator -- Three Frozen Systems")
    p("  DonchianLong | RSI_MR | ConsecDownDays  (shared capital)")
    p("=" * 68)

    # ── Load data ────────────────────────────────────────────────────────────
    p("\n  [1/5] Loading data and computing indicators...")
    symbols = scan_symbols()
    mats    = build_data_matrices(symbols)
    data    = mats["data"]
    dates   = mats["dates"]
    breadth = mats["breadth"]
    avg_rsi = mats["avg_rsi"]
    p(f"       Symbols loaded : {len(data)}")
    p(f"       Date range     : {dates[0]} to {dates[-1]}  ({len(dates)} days)")

    # ── Combined allocator backtest ──────────────────────────────────────────
    p("\n  [2/5] Running combined regime-aware allocator ($30k capital)...")
    eq_df, trades_df, regime_df = run_combined_allocator(dates, data, breadth, avg_rsi)
    p(f"       Trades closed  : {len(trades_df)}")
    p(f"       Final equity   : ${eq_df['equity'].iloc[-1]:,.2f}")

    eq_df.to_csv(OUT_DIR / "allocator_equity_curve.csv", index=False)
    trades_df.to_csv(OUT_DIR / "allocator_trades.csv", index=False)
    regime_df.to_csv(OUT_DIR / "allocator_regime_log.csv", index=False)

    # ── Independent systems backtest ─────────────────────────────────────────
    p("\n  [3/5] Running independent system backtests ($10k each)...")
    ind_don = run_independent_system("Donchian",       dates, data, DON_MAX_POS)
    p(f"       Donchian done:       ${ind_don['equity'].iloc[-1]:,.2f}")
    ind_mr  = run_independent_system("RSI_MR",         dates, data, MR_MAX_POS)
    p(f"       RSI_MR done:         ${ind_mr['equity'].iloc[-1]:,.2f}")
    ind_cd  = run_independent_system("ConsecDownDays", dates, data, CD_MAX_POS)
    p(f"       ConsecDownDays done: ${ind_cd['equity'].iloc[-1]:,.2f}")

    # Combined independent equity (sum of three systems)
    ind_combined = (ind_don["equity"] + ind_mr["equity"] + ind_cd["equity"]).fillna(method="ffill")
    ind_don.to_csv(OUT_DIR / "independent_donchian.csv")
    ind_mr.to_csv(OUT_DIR  / "independent_rsimr.csv")
    ind_cd.to_csv(OUT_DIR  / "independent_consecdowndays.csv")

    # ── Performance metrics ──────────────────────────────────────────────────
    p("\n  [4/5] Computing performance metrics...")

    m_combined = perf(eq_df.set_index("date")["equity"], COMBINED_CAPITAL, "Combined Allocator")
    m_don      = perf(ind_don["equity"], INDEPENDENT_CAPITAL, "Donchian (indep)")
    m_mr       = perf(ind_mr["equity"],  INDEPENDENT_CAPITAL, "RSI_MR (indep)")
    m_cd       = perf(ind_cd["equity"],  INDEPENDENT_CAPITAL, "ConsecDownDays (indep)")
    m_ind_sum  = perf(ind_combined,      COMBINED_CAPITAL,    "Independent Sum")

    # Year-by-year for combined allocator
    alloc_yr   = year_r(trades_df, COMBINED_CAPITAL)

    # Reconstruct independent year-by-year from individual equity curves
    ind_don_trades = pd.DataFrame()   # approximation via equity delta
    def eq_to_annual_pct(eq_series: pd.Series, cap: float) -> dict:
        eq = eq_series.copy()
        eq.index = pd.to_datetime(eq.index)
        start_cap = float(eq.iloc[0])
        results   = {}
        for yr in range(eq.index.year.min(), eq.index.year.max() + 1):
            yr_eq = eq[eq.index.year == yr]
            if yr_eq.empty:
                continue
            beg = float(eq[eq.index.year <  yr].iloc[-1]) if any(eq.index.year < yr) else start_cap
            end = float(yr_eq.iloc[-1])
            results[yr] = round((end - beg) / max(beg, EPS) * 100, 2)
        return results

    don_yr  = eq_to_annual_pct(ind_don["equity"], INDEPENDENT_CAPITAL)
    mr_yr   = eq_to_annual_pct(ind_mr["equity"],  INDEPENDENT_CAPITAL)
    cd_yr   = eq_to_annual_pct(ind_cd["equity"],  INDEPENDENT_CAPITAL)
    ind_yr  = eq_to_annual_pct(ind_combined,      COMBINED_CAPITAL)

    # ── Regime distribution ──────────────────────────────────────────────────
    regime_counts = regime_df["regime"].value_counts()
    total_days    = len(regime_df)

    # ── Print report ─────────────────────────────────────────────────────────
    p("\n" + "=" * 68)
    p("  RESULTS")
    p("=" * 68)

    p("\n  PERFORMANCE COMPARISON")
    p(f"  {'System':>28}  {'Start':>8}  {'End':>10}  {'CAGR':>7}  {'MaxDD':>7}")
    p("  " + "-" * 68)
    for m in [m_combined, m_ind_sum, m_don, m_mr, m_cd]:
        p(f"  {m['label']:>28}  ${m['start']:>7,.0f}  ${m['end']:>9,.0f}  "
          f"{m['cagr']:>+6.2f}%  {m['max_dd']:>+6.2f}%")

    # Year-by-year comparison
    p("\n  YEAR-BY-YEAR RETURN (%)")
    years = sorted(set(list(alloc_yr.keys()) + list(ind_yr.keys())))
    p(f"  {'Year':>5}  {'Allocator':>10}  {'Indep Sum':>10}  {'Donchian':>10}  "
      f"{'RSI_MR':>8}  {'ConsecDown':>11}")
    p("  " + "-" * 65)
    for yr in years:
        p(f"  {yr:>5}  {alloc_yr.get(yr,0.0):>+9.2f}%  {ind_yr.get(yr,0.0):>+9.2f}%  "
          f"{don_yr.get(yr,0.0):>+9.2f}%  {mr_yr.get(yr,0.0):>+7.2f}%  "
          f"{cd_yr.get(yr,0.0):>+10.2f}%")

    p("\n  REGIME DISTRIBUTION")
    for regime in ["BULL","MIXED","BEAR"]:
        cnt = int(regime_counts.get(regime, 0))
        pct = cnt / total_days * 100 if total_days > 0 else 0.0
        p(f"    {regime:>5}: {cnt:>4} days  ({pct:.1f}%)")

    # Regime premium: avg_r by regime in combined allocator
    if not trades_df.empty and "exit_date" in trades_df.columns:
        p("\n  REGIME PREMIUM (combined allocator)")
        trades_yr = trades_df.copy()
        trades_yr["exit_date_pd"] = pd.to_datetime(trades_yr["exit_date"])
        # Merge regime at entry date
        reg_map = dict(zip(regime_df["date"], regime_df["regime"]))
        trades_yr["regime_at_entry"] = trades_yr["entry_date"].map(reg_map)
        for regime in ["BULL","MIXED","BEAR"]:
            sub = trades_yr[trades_yr["regime_at_entry"] == regime]
            if sub.empty:
                continue
            avg_r = float(sub["net_r"].mean())
            p(f"    {regime:>5}: n={len(sub):4}  avg_r={avg_r:+.4f}R")

        p("\n  SYSTEM PERFORMANCE WITHIN ALLOCATOR")
        p(f"  {'System':>16}  {'N':>5}  {'AvgR':>8}  {'WR%':>7}  {'Bonus%':>7}")
        p("  " + "-" * 50)
        for sys in ["Donchian","RSI_MR","ConsecDownDays"]:
            sub = trades_yr[trades_yr["system"] == sys]
            if sub.empty:
                continue
            rs = sub["net_r"].values
            wr = float((rs > 0).mean() * 100) if len(rs) > 0 else 0.0
            p(f"  {sys:>16}  {len(sub):>5}  {float(rs.mean()):>+7.4f}R  "
              f"{wr:>6.1f}%")

    p("\n  HEAT AND POSITION UTILISATION")
    avg_heat = float(eq_df["heat_pct"].mean())
    max_heat = float(eq_df["heat_pct"].max())
    avg_pos  = float(eq_df["open_pos"].mean())
    max_pos  = int(eq_df["open_pos"].max())
    p(f"    Avg portfolio heat : {avg_heat:.2f}%  (max {max_heat:.2f}%  limit {MAX_HEAT_PCT*100:.0f}%)")
    p(f"    Avg open positions : {avg_pos:.1f}  (max {max_pos})")

    # ── Save comparison CSV ──────────────────────────────────────────────────
    p("\n  [5/5] Writing output files...")
    summary_rows = []
    for yr in years:
        summary_rows.append({
            "year": yr,
            "allocator_pct": alloc_yr.get(yr, 0.0),
            "independent_sum_pct": ind_yr.get(yr, 0.0),
            "donchian_pct": don_yr.get(yr, 0.0),
            "rsimr_pct":    mr_yr.get(yr, 0.0),
            "consecdown_pct": cd_yr.get(yr, 0.0),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "yearly_comparison.csv", index=False)

    with open(OUT_DIR / "allocator_report.txt", "w", encoding="utf-8") as f:
        f.write("Regime-Aware Portfolio Allocator Report\n")
        f.write("=" * 68 + "\n\n")
        f.write("PERFORMANCE SUMMARY\n")
        for m in [m_combined, m_ind_sum, m_don, m_mr, m_cd]:
            f.write(f"  {m['label']:>28}  CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:+.2f}%"
                    f"  End=${m['end']:,.0f}\n")
        f.write("\nREGIME DISTRIBUTION\n")
        for regime in ["BULL","MIXED","BEAR"]:
            cnt = int(regime_counts.get(regime, 0))
            f.write(f"  {regime}: {cnt} days ({cnt/total_days*100:.1f}%)\n")
        f.write("\nYEAR-BY-YEAR\n")
        for row in summary_rows:
            f.write(f"  {row['year']}: allocator={row['allocator_pct']:+.2f}%  "
                    f"independent={row['independent_sum_pct']:+.2f}%\n")

    for fn in ["allocator_equity_curve.csv","allocator_trades.csv",
               "allocator_regime_log.csv","yearly_comparison.csv","allocator_report.txt"]:
        p(f"    {OUT_DIR / fn}")
    p()


if __name__ == "__main__":
    main()
