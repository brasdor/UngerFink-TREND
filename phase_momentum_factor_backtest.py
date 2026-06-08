#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 -- Momentum Factor Backtest Long + Short
UngerFink Pipeline / Andrea Unger Methodology

Reads Phase 1 output from data/futures_universe/ohlcv_1d/
Signals on Friday close, holds until next rebalancing date.
Parameter grid: lookback x top_k x bottom_k x rebalance x filter_long x filter_short

Cost model: 0.15% round-trip applied to turnover fraction per period.

Output: data/research_momentum_factor/
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent
OHLCV_DIR  = ROOT / "data" / "futures_universe" / "ohlcv_1d"
SYM_FILE   = ROOT / "data" / "futures_universe" / "all_symbols.csv"
OUT_DIR    = ROOT / "data" / "research_momentum_factor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
COST_RT         = 0.0015    # 0.15% round-trip (fee + slippage)
LONG_WEIGHT     = 0.50      # 50% of capital on long basket
SHORT_WEIGHT    = 0.50      # 50% of capital on short basket

# Parameter grid
LOOKBACKS     = [10, 20, 30, 60, 90]
TOP_KS        = [5, 10, 15, 20]
BOTTOM_KS     = [5, 10, 15, 20]
REBALANCES    = ['weekly', 'biweekly']
FILTER_LONGS  = ['none', 'ema200_above']
FILTER_SHORTS = ['none', 'ema200_below']

TOTAL_COMBOS  = (len(LOOKBACKS) * len(TOP_KS) * len(BOTTOM_KS)
                 * len(REBALANCES) * len(FILTER_LONGS) * len(FILTER_SHORTS))

REPORT_TOP_N = 10
EPS = 1e-12


def p(*a, **kw):
    kw.setdefault('flush', True)
    text = ' '.join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode(), **kw)


# =============================================================================
# METRICS
# =============================================================================

def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1) * 100


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.clip(lower=EPS)
    return float(dd.min()) * 100


def _sharpe(period_rets: list[float], periods_per_year: float = 52.0) -> float:
    if len(period_rets) < 4:
        return 0.0
    arr = np.array(period_rets, dtype=float)
    mu, sigma = arr.mean(), arr.std(ddof=1)
    if sigma < EPS:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def _calmar(equity: pd.Series) -> float:
    mdd = _max_dd(equity)
    if abs(mdd) < EPS:
        return 0.0
    return round(_cagr(equity) / abs(mdd), 3)


def _year_by_year(equity: pd.Series) -> dict[int, float]:
    out = {}
    for yr in range(equity.index.min().year, equity.index.max().year + 1):
        yr_eq = equity[equity.index.year == yr]
        if yr_eq.empty:
            continue
        prior  = equity[equity.index < yr_eq.index[0]]
        beg    = float(prior.iloc[-1]) if not prior.empty else float(equity.iloc[0])
        end    = float(yr_eq.iloc[-1])
        out[yr] = round((end - beg) / max(beg, EPS) * 100, 2)
    return out


# =============================================================================
# DATA LOADING
# =============================================================================

def load_universe() -> pd.DataFrame:
    """Load all symbol CSVs into a single close-price DataFrame (date × symbol)."""
    if not SYM_FILE.exists():
        raise FileNotFoundError(f"Symbol list not found: {SYM_FILE}  -- run Phase 1 first")

    syms = pd.read_csv(SYM_FILE)['symbol'].tolist()
    p(f"  Loading {len(syms)} symbols from {OHLCV_DIR}...")

    frames: dict[str, pd.Series] = {}
    skipped = 0
    for sym in syms:
        path = OHLCV_DIR / f"{sym}_1d.csv"
        if not path.exists():
            skipped += 1
            continue
        try:
            df = pd.read_csv(path, usecols=['date', 'open', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates('date').set_index('date').sort_index()
            frames[sym] = df
        except Exception as exc:
            p(f"    Warning: could not load {sym}: {exc}")
            skipped += 1

    if skipped:
        p(f"  Skipped {skipped} files (missing/corrupt)")

    # Build matrices
    close = pd.DataFrame({sym: df['close'] for sym, df in frames.items()})
    open_ = pd.DataFrame({sym: df['open']  for sym, df in frames.items()})
    close = close.sort_index().astype(float)
    open_ = open_.sort_index().astype(float)

    p(f"  Loaded: {len(frames)} symbols  |  "
      f"{close.index[0].date()} to {close.index[-1].date()}  "
      f"({len(close)} dates)")
    return close, open_


def compute_ema200(close: pd.DataFrame) -> pd.DataFrame:
    return close.ewm(span=200, min_periods=20, adjust=False).mean()


def compute_returns(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """N-bar price return: close[t] / close[t-n] - 1."""
    return close / close.shift(n) - 1


# =============================================================================
# REBALANCING DATE SCHEDULE
# =============================================================================

def get_rebal_dates(close: pd.DataFrame, rebalance: str) -> list[str]:
    """Return list of rebalancing date strings (Fridays, or every-other Friday)."""
    fridays = close.index[close.index.dayofweek == 4]
    if rebalance == 'weekly':
        dates = fridays
    else:  # biweekly: every other Friday
        dates = fridays[::2]
    return [str(d.date()) for d in dates]


# =============================================================================
# SINGLE-COMBO BACKTEST ENGINE
# =============================================================================

def _backtest_combo(
    close:        pd.DataFrame,
    returns_n:    pd.DataFrame,
    ema200:       pd.DataFrame,
    rebal_dates:  list[str],
    top_k:        int,
    bottom_k:     int,
    filter_long:  str,
    filter_short: str,
) -> dict | None:
    """
    Vectorised weekly/biweekly long-short momentum backtest.
    Returns None if fewer than 4 periods traded.
    """
    capital     = INITIAL_CAPITAL
    equity_pts  = {}
    period_rets = []
    long_rets   = []
    short_rets  = []
    turnovers   = []
    prev_longs: set  = set()
    prev_shorts: set = set()

    close_idx = set(str(d.date()) for d in close.index)
    ret_idx   = set(str(d.date()) for d in returns_n.index)

    for i in range(len(rebal_dates) - 1):
        t  = rebal_dates[i]
        t1 = rebal_dates[i + 1]

        if t not in close_idx or t1 not in close_idx or t not in ret_idx:
            continue

        # ---------- signal: N-day momentum at t ----------
        sig = returns_n.loc[t].dropna()
        if sig.empty:
            continue

        syms_at_t  = set(sig.index) & set(close.columns)
        close_at_t = close.loc[t, list(syms_at_t)].dropna()
        ema_at_t   = ema200.loc[t, list(syms_at_t)].dropna() if t in ema200.index else pd.Series(dtype=float)

        # ---------- universe filter for long ----------
        long_universe = set(sig.index)
        if filter_long == 'ema200_above':
            eligible = ema_at_t[close_at_t.reindex(ema_at_t.index) > ema_at_t]
            long_universe &= set(eligible.index)

        # ---------- universe filter for short ----------
        short_universe = set(sig.index)
        if filter_short == 'ema200_below':
            eligible = ema_at_t[close_at_t.reindex(ema_at_t.index) <= ema_at_t]
            short_universe &= set(eligible.index)

        sig_long  = sig.reindex(list(long_universe)).dropna()
        sig_short = sig.reindex(list(short_universe)).dropna()

        if len(sig_long) < top_k or len(sig_short) < bottom_k:
            continue

        longs  = set(sig_long.nlargest(top_k).index)
        shorts = set(sig_short.nsmallest(bottom_k).index)

        # ---------- holding period returns (close[t] to close[t1]) ----------
        close_t  = close.loc[t]
        close_t1 = close.loc[t1]

        l_valid = [s for s in longs
                   if s in close.columns
                   and pd.notna(close_t.get(s)) and close_t.get(s, 0) > 0
                   and pd.notna(close_t1.get(s)) and close_t1.get(s, 0) > 0]
        s_valid = [s for s in shorts
                   if s in close.columns
                   and pd.notna(close_t.get(s)) and close_t.get(s, 0) > 0
                   and pd.notna(close_t1.get(s)) and close_t1.get(s, 0) > 0]

        if not l_valid and not s_valid:
            continue

        # Period returns per basket
        p_long  = float((close_t1[l_valid] / close_t[l_valid] - 1).mean()) if l_valid else 0.0
        p_short = float(-(close_t1[s_valid] / close_t[s_valid] - 1).mean()) if s_valid else 0.0

        # Turnover cost
        turn_l = (len(longs.symmetric_difference(prev_longs))
                  / max(len(longs | prev_longs), 1))
        turn_s = (len(shorts.symmetric_difference(prev_shorts))
                  / max(len(shorts | prev_shorts), 1))
        avg_turn = (turn_l + turn_s) / 2.0
        cost = avg_turn * COST_RT

        # Portfolio return (50% long + 50% short, net of cost)
        net_ret = LONG_WEIGHT * p_long + SHORT_WEIGHT * p_short - cost

        capital *= (1.0 + net_ret)
        equity_pts[t1] = capital
        period_rets.append(net_ret)
        long_rets.append(p_long)
        short_rets.append(p_short)
        turnovers.append(avg_turn)

        prev_longs  = longs
        prev_shorts = shorts

    if len(period_rets) < 4:
        return None

    equity = pd.Series(equity_pts)
    equity.index = pd.to_datetime(equity.index)
    equity = equity.sort_index()

    return {
        'equity':       equity,
        'period_rets':  period_rets,
        'long_rets':    long_rets,
        'short_rets':   short_rets,
        'turnovers':    turnovers,
        'n_periods':    len(period_rets),
    }


def _metrics_from_result(r: dict) -> dict:
    eq = r['equity']
    return {
        'cagr':         round(_cagr(eq), 2),
        'max_dd':       round(_max_dd(eq), 2),
        'sharpe':       round(_sharpe(r['period_rets']), 3),
        'calmar':       _calmar(eq),
        'total_return': round((eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100, 2),
        'n_periods':    r['n_periods'],
        'avg_turnover': round(np.mean(r['turnovers']) * 100, 1),
        'avg_long_ret': round(np.mean(r['long_rets']) * 100, 3),
        'avg_short_ret':round(np.mean(r['short_rets'])* 100, 3),
    }


# =============================================================================
# MAIN BACKTEST LOOP
# =============================================================================

def run_grid(close: pd.DataFrame, ema200: pd.DataFrame) -> pd.DataFrame:
    p(f"\n  Running {TOTAL_COMBOS} parameter combinations...")
    p(f"  Lookbacks: {LOOKBACKS}  top_k: {TOP_KS}  bottom_k: {BOTTOM_KS}")
    p(f"  Rebalances: {REBALANCES}  Filters: {FILTER_LONGS} x {FILTER_SHORTS}")
    p()

    # Precompute N-day returns for each unique lookback
    returns_cache: dict[int, pd.DataFrame] = {
        n: compute_returns(close, n) for n in LOOKBACKS
    }

    # Precompute rebalancing dates for each rebalance type
    rebal_cache: dict[str, list[str]] = {
        r: get_rebal_dates(close, r) for r in REBALANCES
    }

    all_rows = []
    done = 0

    combos = list(itertools.product(
        LOOKBACKS, TOP_KS, BOTTOM_KS, REBALANCES, FILTER_LONGS, FILTER_SHORTS
    ))

    for (lb, tk, bk, reb, fl, fs) in combos:
        done += 1
        r = _backtest_combo(
            close        = close,
            returns_n    = returns_cache[lb],
            ema200       = ema200,
            rebal_dates  = rebal_cache[reb],
            top_k        = tk,
            bottom_k     = bk,
            filter_long  = fl,
            filter_short = fs,
        )

        if r is None:
            row = {'lookback': lb, 'top_k': tk, 'bottom_k': bk,
                   'rebalance': reb, 'filter_long': fl, 'filter_short': fs,
                   'cagr': np.nan, 'max_dd': np.nan, 'sharpe': np.nan,
                   'calmar': np.nan, 'total_return': np.nan,
                   'n_periods': 0, 'avg_turnover': np.nan,
                   'avg_long_ret': np.nan, 'avg_short_ret': np.nan}
        else:
            m = _metrics_from_result(r)
            yby = _year_by_year(r['equity'])
            row = {
                'lookback': lb, 'top_k': tk, 'bottom_k': bk,
                'rebalance': reb, 'filter_long': fl, 'filter_short': fs,
                **m,
            }
            for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
                row[f'yr_{yr}'] = yby.get(yr, np.nan)

        all_rows.append(row)

        if done % 50 == 0 or done == len(combos):
            p(f"  Progress: {done}/{len(combos)}  "
              f"(last: lb={lb} tk={tk} bk={bk} {reb} {fl}/{fs}  "
              f"CAGR={row.get('cagr', '?'):>+.1f}%  "
              f"Sharpe={row.get('sharpe', '?'):>+.3f})", end="\r")

    p()   # newline after progress bar
    return pd.DataFrame(all_rows)


# =============================================================================
# STABILITY CHECK
# =============================================================================

def stability_check(df: pd.DataFrame, best_row: pd.Series) -> bool:
    """
    Check if the best combo is profitable at adjacent lookback steps.
    Adjacent = neighbouring values in the LOOKBACKS grid.
    """
    lb    = best_row['lookback']
    lb_list = sorted(LOOKBACKS)
    idx   = lb_list.index(lb)
    neighbors = []
    if idx > 0:
        neighbors.append(lb_list[idx - 1])
    if idx < len(lb_list) - 1:
        neighbors.append(lb_list[idx + 1])

    p()
    p("  STABILITY CHECK  (best lookback must be profitable at adjacent steps)")
    p(f"  Canonical lookback: {lb}d")
    stable = True
    for nb in neighbors:
        sub = df[
            (df['lookback'] == nb)
            & (df['top_k']        == best_row['top_k'])
            & (df['bottom_k']     == best_row['bottom_k'])
            & (df['rebalance']    == best_row['rebalance'])
            & (df['filter_long']  == best_row['filter_long'])
            & (df['filter_short'] == best_row['filter_short'])
        ]
        if sub.empty:
            p(f"    lb={nb}d : n/a")
            continue
        r_cagr = sub.iloc[0]['cagr']
        r_sharpe = sub.iloc[0]['sharpe']
        ok = pd.notna(r_cagr) and r_cagr > 0 and pd.notna(r_sharpe) and r_sharpe > 0.3
        tag = "[OK]" if ok else "[!!FAIL]"
        p(f"    lb={nb}d :  CAGR={r_cagr:>+.2f}%  Sharpe={r_sharpe:>+.3f}  {tag}")
        if not ok:
            stable = False

    p(f"  Result: {'STABLE -- canonical is not a peak' if stable else 'UNSTABLE -- canonical may be a lucky peak'}")
    return stable


# =============================================================================
# REPORT
# =============================================================================

def print_report(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=['sharpe'])
    if valid.empty:
        p("  No valid results -- check data coverage")
        return

    top = valid.nlargest(REPORT_TOP_N, 'sharpe')

    p()
    p("=" * 90)
    p(f"  PHASE 2 RESULTS -- Top {REPORT_TOP_N} Combos by Sharpe")
    p("=" * 90)
    hdr = (f"  {'#':>2}  {'LB':>4}  {'TK':>4}  {'BK':>4}  "
           f"{'Reb':>9}  {'FL':>13}  {'FS':>13}  "
           f"{'CAGR':>7}  {'MaxDD':>7}  {'Sharpe':>7}  {'WinRate':>8}  {'Turn%':>6}")
    p(hdr)
    p("  " + "-" * 86)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        p(f"  {rank:>2}  {int(row['lookback']):>4}  {int(row['top_k']):>4}  "
          f"{int(row['bottom_k']):>4}  {row['rebalance']:>9}  "
          f"{row['filter_long']:>13}  {row['filter_short']:>13}  "
          f"{row['cagr']:>+6.2f}%  {row['max_dd']:>+6.2f}%  "
          f"{row['sharpe']:>7.3f}  "
          f"{'n/a':>8}  "
          f"{row['avg_turnover']:>5.1f}%")

    # Year-by-year for top 5
    p()
    p("  YEAR-BY-YEAR  (top 5 combos by Sharpe)")
    p(f"  {'Rank':<5}  {'Config':<40}", end="")
    years = [yr for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]]
    for yr in years:
        print(f"  {yr}", end="")
    print()
    p("  " + "-" * 120)
    for rank, (_, row) in enumerate(top.head(5).iterrows(), 1):
        cfg = f"lb={int(row['lookback'])} tk={int(row['top_k'])} bk={int(row['bottom_k'])} {row['rebalance'][:4]}"
        print(f"  {rank:<5}  {cfg:<40}", end="")
        for yr in years:
            v = row.get(f'yr_{yr}', np.nan)
            if pd.notna(v):
                print(f"  {v:>+5.1f}%", end="")
            else:
                print(f"  {'n/a':>6}", end="")
        print()

    # Long vs Short attribution for best combo
    p()
    p("  LONG vs SHORT ATTRIBUTION  (best combo by Sharpe)")
    best = top.iloc[0]
    p(f"  Config: lb={int(best['lookback'])}  tk={int(best['top_k'])}  "
      f"bk={int(best['bottom_k'])}  {best['rebalance']}")
    p(f"  Avg long  side return:  {best['avg_long_ret']:>+.3f}% per period")
    p(f"  Avg short side return:  {best['avg_short_ret']:>+.3f}% per period")
    p(f"  Avg turnover:           {best['avg_turnover']:>.1f}% of positions replaced per period")

    # Overall distribution
    p()
    p("  DISTRIBUTION OF PROFITABLE COMBOS")
    n_profitable = (valid['cagr'] > 0).sum()
    n_sharpe_pos = (valid['sharpe'] > 0.5).sum()
    p(f"  CAGR > 0:      {n_profitable:>4} / {len(valid)} combos  ({n_profitable/len(valid)*100:.1f}%)")
    p(f"  Sharpe > 0.5:  {n_sharpe_pos:>4} / {len(valid)} combos  ({n_sharpe_pos/len(valid)*100:.1f}%)")

    # Lookback breakdown (best Sharpe per lookback)
    p()
    p("  BEST SHARPE PER LOOKBACK  (all rebalance / filter combos)")
    p(f"  {'Lookback':>9}  {'BestSharpe':>11}  {'BestCAGR':>10}  {'AvgSharpe':>11}")
    p("  " + "-" * 50)
    for lb in LOOKBACKS:
        sub = valid[valid['lookback'] == lb]
        if sub.empty:
            continue
        p(f"  {lb:>8}d  {sub['sharpe'].max():>+10.3f}  "
          f"{sub['cagr'].max():>+9.2f}%  {sub['sharpe'].mean():>+10.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    p("=" * 72)
    p("  PHASE 2 -- Momentum Factor Backtest  Long + Short")
    p(f"  Parameter grid: {TOTAL_COMBOS} combinations")
    p(f"  Cost model: {COST_RT*100:.2f}% round-trip on turnover fraction")
    p("=" * 72)

    # Load data
    p("\n  [1/4] Loading data...")
    close, open_ = load_universe()

    # Compute EMA200 once
    p("  [2/4] Computing EMA200 filter...")
    ema200 = compute_ema200(close)
    p(f"         EMA200 matrix: {ema200.shape}")

    # Run grid
    p("  [3/4] Running parameter grid...")
    results_df = run_grid(close, ema200)

    # Save full results
    results_df.to_csv(OUT_DIR / "all_combos.csv", index=False)
    p(f"\n  Full results: {OUT_DIR / 'all_combos.csv'}")

    # Report
    p("\n  [4/4] Generating report...")
    print_report(results_df)

    # Save top combos
    valid = results_df.dropna(subset=['sharpe'])
    if not valid.empty:
        top10 = valid.nlargest(REPORT_TOP_N, 'sharpe')
        top10.to_csv(OUT_DIR / "top10_combos.csv", index=False)
        p(f"\n  Top 10 saved: {OUT_DIR / 'top10_combos.csv'}")

        # Stability check on #1
        p()
        best = valid.nlargest(1, 'sharpe').iloc[0]
        stability_check(results_df, best)

    p()
    p("=" * 72)
    p("  PHASE 2 COMPLETE")
    p(f"  Output: {OUT_DIR}")
    p("=" * 72)
    p()
    p("  >>> Pause: review Phase 2 results before proceeding to Phase 3 <<<")


if __name__ == "__main__":
    main()
