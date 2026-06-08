#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 -- Portfolio Integration Test
A: Allocator V2 alone            ($30k)
B: Momentum Factor alone         ($10k)
C: A + B combined                ($40k, independent)
D: A + B regime-coordinated      ($40k, BEAR bk 10->15, BULL tk 20->25)

Output: data/research_momentum_portfolio/
"""

from __future__ import annotations

import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "research_momentum_portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from portfolio_allocator import (
    scan_symbols, build_data_matrices, check_exit, Position,
    DON_ATR_MULT, MR_ATR_MULT, CD_ATR_MULT,
    RSI_OVERSOLD, CONSEC_N,
    EPS,
)

# ── Constants ─────────────────────────────────────────────────────────────

CANONICAL_BULL  = 0.55
CANONICAL_BEAR  = 0.45
CANONICAL_CAP   = 150.0
POOL_A_CAP      = 27_000.0
POOL_B_CAP      =  3_000.0
BASE_RISK_PCT   = 0.0025
BULL_DON_PCT    = 0.0035
BEAR_MR_PCT     = 0.0050
MAX_HEAT_PCT    = 0.05

MOM_CAPITAL     = 10_000.0
MOM_LB          = 20
MOM_TK          = 20
MOM_BK          = 10
MOM_FILTER_LONG = 'ema200_above'
MOM_COST_RT     = 0.0015

D_BULL_TK       = 25
D_BEAR_BK       = 15

FUTURES_OHLCV   = ROOT / "data" / "futures_universe" / "ohlcv_1d"
FUTURES_SYMS    = ROOT / "data" / "futures_universe" / "all_symbols.csv"


# ── Utility ───────────────────────────────────────────────────────────────

def p(*a, **kw):
    kw.setdefault('flush', True)
    text = ' '.join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode(), **kw)


def detect_regime_p(date_, btc_df, breadth,
                    bull_thresh=CANONICAL_BULL, bear_thresh=CANONICAL_BEAR) -> str:
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


def get_regime_at_date(regime_by_date: dict, date_key) -> str:
    """
    Look up regime; regime_by_date is keyed by datetime.date.
    date_key may be str "YYYY-MM-DD" or datetime.date.
    Falls back to nearest earlier date if exact key missing.
    """
    if isinstance(date_key, str):
        d = datetime.date.fromisoformat(date_key)
    elif isinstance(date_key, pd.Timestamp):
        d = date_key.date()
    else:
        d = date_key
    if d in regime_by_date:
        return regime_by_date[d]
    earlier = sorted(k for k in regime_by_date if k <= d)
    return regime_by_date[earlier[-1]] if earlier else "MIXED"


# ── Metrics ───────────────────────────────────────────────────────────────

def calc_cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return 0.0 if years <= 0 else float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1) * 100


def calc_maxdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak.clip(lower=EPS)).min()) * 100


def calc_sharpe(equity: pd.Series) -> float:
    daily = equity.pct_change().dropna()
    if len(daily) < 10:
        return 0.0
    mu, s = daily.mean(), daily.std(ddof=1)
    return 0.0 if s < EPS else float(mu / s * np.sqrt(252))


def year_by_year(equity: pd.Series) -> dict:
    out = {}
    equity = equity.dropna()
    if equity.empty:
        return out
    for yr in range(equity.index.min().year, equity.index.max().year + 1):
        yr_eq = equity[equity.index.year == yr].dropna()
        if yr_eq.empty:
            continue
        prior = equity[equity.index < yr_eq.index[0]].dropna()
        beg   = float(prior.iloc[-1]) if not prior.empty else float(yr_eq.iloc[0])
        out[yr] = round((float(yr_eq.iloc[-1]) - beg) / max(beg, EPS) * 100, 2)
    return out


def portfolio_metrics(equity: pd.Series, initial: float) -> dict:
    eq = equity.dropna()
    if eq.empty:
        return {'cagr': 0.0, 'max_dd': 0.0, 'sharpe': 0.0, 'total_return': 0.0, 'yby': {}}
    return {
        'cagr':         round(calc_cagr(eq),  2),
        'max_dd':       round(calc_maxdd(eq),  2),
        'sharpe':       round(calc_sharpe(eq), 3),
        'total_return': round((eq.iloc[-1] / initial - 1) * 100, 2),
        'yby':          year_by_year(eq),
    }


# ── Allocator V2 Engine ───────────────────────────────────────────────────

def _v2_risk_pct(system: str, regime: str) -> float:
    if system == "Donchian"      and regime == "BULL": return BULL_DON_PCT
    if system == "RSI_MR"        and regime == "BEAR": return BEAR_MR_PCT
    return BASE_RISK_PCT


def _generate_signals(date_, data, regime, systems):
    signals = []
    for sym, df in data.items():
        if date_ not in df.index:
            continue
        row    = df.loc[date_]
        close  = float(row["close"])
        atr    = float(row.get("atr",      np.nan))
        ema200 = float(row.get("ema200",   np.nan))
        rsi    = float(row.get("rsi",      np.nan))
        d_up   = float(row.get("don_upper", np.nan))
        d_lo   = float(row.get("don_lower", np.nan))
        cd     = int(row.get("consec_down", 0))
        if any(np.isnan(x) for x in [close, atr, ema200]) or atr <= EPS:
            continue
        if "Donchian" in systems and not np.isnan(d_up):
            if close > d_up and close > ema200:
                signals.append(dict(symbol=sym, system="Donchian",
                    close=close, atr=atr,
                    stop=close - DON_ATR_MULT * atr,
                    risk_unit=DON_ATR_MULT * atr,
                    risk_pct=_v2_risk_pct("Donchian", regime),
                    don_lower=d_lo))
        if "RSI_MR" in systems and not np.isnan(rsi):
            if rsi < RSI_OVERSOLD:
                signals.append(dict(symbol=sym, system="RSI_MR",
                    close=close, atr=atr,
                    stop=close - MR_ATR_MULT * atr,
                    risk_unit=MR_ATR_MULT * atr,
                    risk_pct=_v2_risk_pct("RSI_MR", regime),
                    don_lower=0.0))
        if "ConsecDownDays" in systems:
            if cd >= CONSEC_N and close > ema200:
                signals.append(dict(symbol=sym, system="ConsecDownDays",
                    close=close, atr=atr,
                    stop=close - CD_ATR_MULT * atr,
                    risk_unit=CD_ATR_MULT * atr,
                    risk_pct=_v2_risk_pct("ConsecDownDays", regime),
                    don_lower=0.0))
    signals.sort(key=lambda s: s["risk_pct"], reverse=True)
    return signals


def _run_pool(pool_name, pool_capital, systems, dates, data, breadth, btc_df):
    """
    Single-pool V2 backtest.
    Returns: equity_series (daily, Timestamp index), closed_df, regime_by_date dict.
    """
    equity   = pool_capital
    peak     = pool_capital
    open_pos = []
    eq_daily = {}
    closed   = []
    regime_by_date = {}

    for date_ in dates:
        regime = detect_regime_p(date_, btc_df, breadth)
        regime_by_date[date_] = regime

        still_open = []
        for pos in open_pos:
            result = check_exit(pos, date_, data)
            if result is not None:
                exit_price, reason = result
                r_mult   = (exit_price - pos.entry_price) / max(pos.risk_unit, EPS)
                pnl_usdt = pos.risk_amt * r_mult
                equity  += pnl_usdt
                peak     = max(peak, equity)
                closed.append(dict(
                    pool=pool_name, system=pos.system,
                    entry_date=pos.entry_date, exit_date=date_,
                    symbol=pos.symbol,
                    net_r=round(r_mult, 4),
                    pnl_usdt=round(pnl_usdt, 4),
                    exit_reason=reason,
                    regime=regime,
                ))
            else:
                still_open.append(pos)
        open_pos = still_open

        occupied  = {pos.symbol for pos in open_pos}
        curr_heat = sum(pos.risk_pct for pos in open_pos)

        for sig in _generate_signals(date_, data, regime, systems):
            if sig["symbol"] in occupied:
                continue
            if curr_heat + sig["risk_pct"] > MAX_HEAT_PCT:
                continue
            r_amt = min(equity * sig["risk_pct"], CANONICAL_CAP)
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

        eq_daily[date_] = equity

    eq_series = pd.Series(eq_daily)
    eq_series.index = pd.to_datetime(list(eq_series.keys()))
    return eq_series, pd.DataFrame(closed) if closed else pd.DataFrame(), regime_by_date


def run_allocator_v2(dates, data, breadth, btc_df):
    p("    Pool A (Donchian + RSI_MR)...")
    eq_a, closed_a, regime_dates = _run_pool("A", POOL_A_CAP, ["Donchian","RSI_MR"],
                                             dates, data, breadth, btc_df)
    p("    Pool B (ConsecDownDays)...")
    eq_b, closed_b, _            = _run_pool("B", POOL_B_CAP, ["ConsecDownDays"],
                                             dates, data, breadth, btc_df)
    all_idx = eq_a.index.union(eq_b.index)
    eq_a    = eq_a.reindex(all_idx).ffill()
    eq_b    = eq_b.reindex(all_idx).ffill()
    combined = (eq_a + eq_b).sort_index()

    all_closed = pd.concat([closed_a, closed_b], ignore_index=True) if (
        not closed_a.empty or not closed_b.empty) else pd.DataFrame()

    return combined, all_closed, regime_dates


# ── Momentum Factor Engine ─────────────────────────────────────────────────

def _load_futures_close() -> pd.DataFrame:
    syms = pd.read_csv(FUTURES_SYMS)['symbol'].tolist()
    frames = {}
    for sym in syms:
        path = FUTURES_OHLCV / f"{sym}_1d.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates('date').set_index('date').sort_index()
            frames[sym] = df['close'].astype(float)
        except Exception:
            pass
    return pd.DataFrame(frames).sort_index()


def _biweekly_fridays(close: pd.DataFrame) -> list[str]:
    fridays = close.index[close.index.dayofweek == 4]
    return [str(d.date()) for d in fridays[::2]]


def run_momentum_factor(close: pd.DataFrame, ema200: pd.DataFrame,
                        capital: float, tk: int, bk: int, filter_long: str,
                        regime_by_date: dict | None = None,
                        bull_tk: int | None = None,
                        bear_bk: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Single-config Momentum Factor backtest with daily MTM equity.
    Regime coordination: in BULL use bull_tk, in BEAR use bear_bk.
    Returns: (daily equity Series, period stats DataFrame)
    """
    rebal_dates = _biweekly_fridays(close)
    returns_n   = close / close.shift(MOM_LB) - 1

    ret_idx_set   = set(returns_n.index.strftime('%Y-%m-%d'))
    close_idx_set = set(close.index.strftime('%Y-%m-%d'))

    equity       = capital
    eq_by_date   = {}
    period_stats = []
    prev_longs: set  = set()
    prev_shorts: set = set()

    for i in range(len(rebal_dates) - 1):
        t  = rebal_dates[i]
        t1 = rebal_dates[i + 1]
        if t not in close_idx_set or t1 not in close_idx_set:
            continue
        if t not in ret_idx_set:
            continue

        # Regime coordination
        regime    = get_regime_at_date(regime_by_date, t) if regime_by_date else "MIXED"
        period_tk = (bull_tk if regime == "BULL" and bull_tk else tk)
        period_bk = (bear_bk if regime == "BEAR" and bear_bk else bk)

        t_ts  = pd.Timestamp(t)
        t1_ts = pd.Timestamp(t1)

        if t_ts not in returns_n.index:
            continue
        sig = returns_n.loc[t_ts].dropna()
        if sig.empty:
            continue

        # Long-side universe filter
        if filter_long == 'ema200_above' and t_ts in ema200.index:
            close_t = close.loc[t_ts]
            ema_t   = ema200.loc[t_ts]
            above   = [s for s in sig.index
                       if (close_t.get(s, np.nan) or np.nan) > (ema_t.get(s, np.nan) or np.nan)]
            sig_long = sig.reindex(above).dropna()
        else:
            sig_long = sig

        sig_short = sig  # no filter on short side

        if len(sig_long) < period_tk or len(sig_short) < period_bk:
            continue

        longs  = set(sig_long.nlargest(period_tk).index)
        shorts = set(sig_short.nsmallest(period_bk).index)

        # Daily MTM within [t, t1]
        close_t_row = close.loc[t_ts]
        period_days = close.index[(close.index >= t_ts) & (close.index <= t1_ts)]

        for d in period_days[1:]:
            close_d = close.loc[d]
            d_str   = str(d.date())

            l_val = [s for s in longs
                     if pd.notna(close_t_row.get(s, np.nan)) and (close_t_row.get(s, 0) or 0) > 0
                     and pd.notna(close_d.get(s, np.nan)) and (close_d.get(s, 0) or 0) > 0]
            s_val = [s for s in shorts
                     if pd.notna(close_t_row.get(s, np.nan)) and (close_t_row.get(s, 0) or 0) > 0
                     and pd.notna(close_d.get(s, np.nan)) and (close_d.get(s, 0) or 0) > 0]

            r_l = float((close_d[l_val] / close_t_row[l_val] - 1).mean()) if l_val else 0.0
            r_s = float(-(close_d[s_val] / close_t_row[s_val] - 1).mean()) if s_val else 0.0
            eq_by_date[d_str] = equity * (1.0 + 0.5 * r_l + 0.5 * r_s)

        # End-of-period cost-adjusted equity update
        if t1_ts not in close.index:
            continue
        close_t1 = close.loc[t1_ts]
        l_val = [s for s in longs
                 if pd.notna(close_t_row.get(s, np.nan)) and (close_t_row.get(s, 0) or 0) > 0
                 and pd.notna(close_t1.get(s, np.nan)) and (close_t1.get(s, 0) or 0) > 0]
        s_val = [s for s in shorts
                 if pd.notna(close_t_row.get(s, np.nan)) and (close_t_row.get(s, 0) or 0) > 0
                 and pd.notna(close_t1.get(s, np.nan)) and (close_t1.get(s, 0) or 0) > 0]

        r_l = float((close_t1[l_val] / close_t_row[l_val] - 1).mean()) if l_val else 0.0
        r_s = float(-(close_t1[s_val] / close_t_row[s_val] - 1).mean()) if s_val else 0.0

        turn_l = len(longs.symmetric_difference(prev_longs))  / max(len(longs | prev_longs), 1)
        turn_s = len(shorts.symmetric_difference(prev_shorts)) / max(len(shorts | prev_shorts), 1)
        cost   = ((turn_l + turn_s) / 2.0) * MOM_COST_RT

        net_ret = 0.5 * r_l + 0.5 * r_s - cost
        equity *= (1.0 + net_ret)
        eq_by_date[t1] = equity   # override with cost-adjusted value

        period_stats.append(dict(
            date=t1, regime=regime,
            r_long=round(r_l * 100, 3), r_short=round(r_s * 100, 3),
            net_ret=round(net_ret * 100, 3),
            tk=period_tk, bk=period_bk,
        ))
        prev_longs  = longs
        prev_shorts = shorts

    eq_series = pd.Series(eq_by_date)
    eq_series.index = pd.to_datetime(eq_series.index)
    eq_series = eq_series.sort_index()
    return eq_series, pd.DataFrame(period_stats) if period_stats else pd.DataFrame()


# ── Bear-regime performance helper ────────────────────────────────────────

def bear_equity_cagr(equity: pd.Series, regime_by_date: dict) -> tuple[float, int]:
    """
    Compute CAGR of the equity curve only during BEAR dates.
    Returns (cagr%, n_bear_dates).
    """
    bear_dates = sorted(
        d for d in equity.index
        if get_regime_at_date(regime_by_date, str(d.date())) == "BEAR"
    )
    if len(bear_dates) < 2:
        return 0.0, len(bear_dates)
    bear_eq = equity.reindex(bear_dates).dropna()
    if bear_eq.empty or bear_eq.iloc[0] <= 0:
        return 0.0, 0
    years = (bear_dates[-1] - bear_dates[0]).days / 365.25
    if years <= 0:
        return 0.0, len(bear_dates)
    return float((bear_eq.iloc[-1] / bear_eq.iloc[0]) ** (1.0 / years) - 1) * 100, len(bear_dates)


# ── Report ────────────────────────────────────────────────────────────────

def _fmt(v, fmt="+.2f", unit="%"):
    return f"{v:{fmt}}{unit}" if pd.notna(v) else "  n/a"


def print_report(portfolios, regime_by_date, stats_b, stats_d,
                 bear_r_a, corr_ab):
    p()
    p("=" * 100)
    p("  PHASE 3 RESULTS -- PORTFOLIO INTEGRATION TEST")
    p("=" * 100)

    # ── Main comparison table ──
    p()
    p(f"  {'Portfolio':<35}  {'Init $':>8}  {'CAGR':>8}  {'MaxDD':>8}  "
      f"{'Sharpe':>7}  {'TotalRet':>10}")
    p("  " + "-" * 85)
    for label, init_cap, m in portfolios:
        p(f"  {label:<35}  ${init_cap:>7,}  {_fmt(m['cagr']):>9}  {_fmt(m['max_dd']):>9}  "
          f"{m['sharpe']:>7.3f}  {_fmt(m['total_return']):>11}")

    # ── Year-by-year ──
    p()
    p("  YEAR-BY-YEAR RETURNS")
    hdr = f"  {'Portfolio':<35}"
    for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        hdr += f"  {yr}"
    p(hdr)
    p("  " + "-" * 95)
    for label, _, m in portfolios:
        row = f"  {label:<35}"
        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            v = m['yby'].get(yr, np.nan)
            row += f"  {_fmt(v, '+.1f'):>7}" if pd.notna(v) else "     n/a"
        p(row)

    # ── BEAR regime performance ──
    p()
    p("  BEAR REGIME PERFORMANCE")
    p("  " + "-" * 85)
    p(f"  Portfolio A  --  Allocator V2 avg_R in BEAR trades:  {bear_r_a:>+.4f}R  "
      f"(target >+0.40R, prev validation +0.14R)")

    for reg_label, stats in [("B  --  Momentum Factor (canonical)",       stats_b),
                              ("D  --  Momentum Factor (regime-coord)", stats_d)]:
        if stats is None or stats.empty:
            p(f"  Portfolio {reg_label}:  no data")
            continue
        bear = stats[stats['regime'] == 'BEAR']
        mixed = stats[stats['regime'] == 'MIXED']
        bull  = stats[stats['regime'] == 'BULL']
        for sub, name in [(bull, 'BULL'), (mixed, 'MIXED'), (bear, 'BEAR')]:
            if sub.empty:
                continue
            avg  = float(sub['net_ret'].mean())
            n    = len(sub)
            note = "  <-- BEAR target" if name == "BEAR" else ""
            p(f"  Portfolio {reg_label} -- {name:<5}  avg period ret: "
              f"{avg:>+.3f}%  ({n:>3} periods){note}")

    # ── Correlation ──
    p()
    p("  CORRELATION  (Allocator V2 daily P&L  vs  Momentum Factor daily P&L)")
    if pd.notna(corr_ab):
        tag = ("LOW -- excellent diversification" if abs(corr_ab) < 0.2
               else "MODERATE -- acceptable" if abs(corr_ab) < 0.4
               else "HIGH -- limited diversification benefit")
        p(f"  Pearson r = {corr_ab:>+.4f}  ->  {tag}")
    else:
        p("  Pearson r = n/a (insufficient overlapping dates)")

    # ── Key question ──
    p()
    p("  KEY QUESTION: Does adding Momentum Factor close the BEAR regime gap?")
    p(f"  Allocator V2 alone:          BEAR avg_R = {bear_r_a:>+.4f}R  (target >+0.40R)")
    if stats_b is not None and not stats_b.empty:
        bear_b = stats_b[stats_b['regime'] == 'BEAR']
        if not bear_b.empty:
            p(f"  Momentum (Portfolio B):      BEAR avg period = {float(bear_b['net_ret'].mean()):>+.3f}%  "
              f"per 2-week period ({len(bear_b)} periods)")
    if stats_d is not None and not stats_d.empty:
        bear_d = stats_d[stats_d['regime'] == 'BEAR']
        if not bear_d.empty:
            p(f"  Momentum regime-coord (D):   BEAR avg period = {float(bear_d['net_ret'].mean()):>+.3f}%  "
              f"per 2-week period ({len(bear_d)} periods) [bk {MOM_BK}->{D_BEAR_BK}]")

    p()
    p("=" * 100)
    p("  PHASE 3 COMPLETE")
    p(f"  Output: data/research_momentum_portfolio/")
    p("=" * 100)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p("=" * 72)
    p("  PHASE 3 -- Portfolio Integration Test")
    p("  Portfolios A / B / C / D  |  2019-2026 full period")
    p("=" * 72)

    # 1. Load spot universe (Allocator V2)
    p("\n  [1/6] Loading Spot Universe (Allocator V2)...")
    symbols = scan_symbols()
    result  = build_data_matrices(symbols)
    data    = result['data']
    dates   = result['dates']
    breadth = result['breadth']
    btc_key = next((k for k in data if 'BTC' in k.upper()), None)
    if btc_key is None:
        raise RuntimeError("BTC data not found in spot universe")
    btc_df = data[btc_key]
    p(f"         {len(symbols)} symbols  |  {dates[0]} to {dates[-1]}")

    # 2. Load futures universe (Momentum Factor)
    p("\n  [2/6] Loading Futures Universe (Momentum Factor)...")
    close_f  = _load_futures_close()
    ema200_f = close_f.ewm(span=200, min_periods=20, adjust=False).mean()
    p(f"         {close_f.shape[1]} symbols  |  "
      f"{close_f.index[0].date()} to {close_f.index[-1].date()}")

    # 3. Portfolio A -- Allocator V2 alone
    p("\n  [3/6] Portfolio A -- Allocator V2 ($30k)...")
    eq_a, closed_a, regime_dates = run_allocator_v2(dates, data, breadth, btc_df)
    bear_r_a = float(closed_a[closed_a['regime'] == 'BEAR']['net_r'].mean()) \
               if not closed_a.empty else 0.0
    p(f"         Done.  CAGR={calc_cagr(eq_a):>+.2f}%  MaxDD={calc_maxdd(eq_a):>+.2f}%  "
      f"BEAR avg_R={bear_r_a:>+.4f}R")

    # 4. Portfolio B -- Momentum Factor alone
    p("\n  [4/6] Portfolio B -- Momentum Factor ($10k, canonical config)...")
    eq_b, stats_b = run_momentum_factor(
        close_f, ema200_f, MOM_CAPITAL,
        tk=MOM_TK, bk=MOM_BK, filter_long=MOM_FILTER_LONG,
        regime_by_date=regime_dates)
    p(f"         Done.  CAGR={calc_cagr(eq_b):>+.2f}%  MaxDD={calc_maxdd(eq_b):>+.2f}%  "
      f"Periods={len(stats_b)}")

    # 5. Portfolio C -- A + B combined (independent)
    p("\n  [5/6] Portfolio C -- A+B Combined ($40k, independent)...")
    all_idx = eq_a.index.union(eq_b.index).sort_values()
    eq_c    = (eq_a.reindex(all_idx).ffill() + eq_b.reindex(all_idx).ffill())
    p(f"         Done.  CAGR={calc_cagr(eq_c):>+.2f}%  MaxDD={calc_maxdd(eq_c):>+.2f}%")

    # 6. Portfolio D -- A + regime-coordinated Momentum
    p("\n  [6/6] Portfolio D -- A + Regime-Coordinated Momentum ($40k)...")
    p(f"         BULL tk {MOM_TK}->{D_BULL_TK}  |  BEAR bk {MOM_BK}->{D_BEAR_BK}")
    eq_d_mom, stats_d = run_momentum_factor(
        close_f, ema200_f, MOM_CAPITAL,
        tk=MOM_TK, bk=MOM_BK, filter_long=MOM_FILTER_LONG,
        regime_by_date=regime_dates,
        bull_tk=D_BULL_TK, bear_bk=D_BEAR_BK)
    all_idx_d = eq_a.index.union(eq_d_mom.index).sort_values()
    eq_d      = (eq_a.reindex(all_idx_d).ffill() + eq_d_mom.reindex(all_idx_d).ffill())
    p(f"         Done.  CAGR={calc_cagr(eq_d):>+.2f}%  MaxDD={calc_maxdd(eq_d):>+.2f}%")

    # Correlation
    pnl_a  = eq_a.diff().dropna()
    pnl_b  = eq_b.diff().dropna()
    common = pnl_a.index.intersection(pnl_b.index)
    corr_ab = float(pnl_a.loc[common].corr(pnl_b.loc[common])) if len(common) > 10 else np.nan

    # Compute all metrics
    init_a, init_c = POOL_A_CAP + POOL_B_CAP, POOL_A_CAP + POOL_B_CAP + MOM_CAPITAL
    m_a = portfolio_metrics(eq_a,   init_a)
    m_b = portfolio_metrics(eq_b,   MOM_CAPITAL)
    m_c = portfolio_metrics(eq_c,   init_c)
    m_d = portfolio_metrics(eq_d,   init_c)

    portfolios = [
        ("A: Allocator V2 alone",       init_a,  m_a),
        ("B: Momentum Factor alone",     MOM_CAPITAL, m_b),
        ("C: A+B combined",             init_c,  m_c),
        ("D: A+B regime-coordinated",   init_c,  m_d),
    ]

    # Save outputs
    summary_rows = []
    for label, init_cap, m in portfolios:
        row = {'portfolio': label, 'initial': init_cap,
               'cagr': m['cagr'], 'max_dd': m['max_dd'], 'sharpe': m['sharpe'],
               'total_return': m['total_return']}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            row[f'yr_{yr}'] = m['yby'].get(yr, np.nan)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "portfolio_comparison.csv", index=False)

    # Save equity curves
    eq_out = pd.DataFrame({
        'equity_A': eq_a,
        'equity_B': eq_b.reindex(eq_a.index).ffill(),
        'equity_C': eq_c.reindex(eq_a.index).ffill(),
        'equity_D': eq_d.reindex(eq_a.index).ffill(),
    })
    eq_out.index.name = 'date'
    eq_out.to_csv(OUT_DIR / "equity_curves.csv")

    if not stats_b.empty:
        stats_b.to_csv(OUT_DIR / "momentum_b_period_stats.csv", index=False)
    if not stats_d.empty:
        stats_d.to_csv(OUT_DIR / "momentum_d_period_stats.csv", index=False)

    print_report(portfolios, regime_dates, stats_b, stats_d, bear_r_a, corr_ab)


if __name__ == "__main__":
    main()
