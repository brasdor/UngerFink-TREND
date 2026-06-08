"""
M3 Backtest Engine
==================
Signal on close[t], execute on open[t+1].  Never uses close[t] as entry price.

Position structure
------------------
Long basket  (Q5) : 65% of equity × leverage, equal-weighted
Short basket (Q1) : 35% of equity × leverage, equal-weighted (skipped in long_only mode)

Costs
-----
Round-trip fee + tiered slippage + actual funding rate (defaults to 0 if DB has no data).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn
from m3_backtest.costs import (
    FEE_MAKER, SLIP_TIER1, SLIP_TIER2, FUNDING_INTERVAL, total_cost,
)
from m3_backtest.metrics import (
    sharpe, max_drawdown, cagr, calmar, win_rate,
    factor_attribution, print_summary,
)

INITIAL_CAPITAL   = 10_000.0
LONG_ALLOC        = 0.65
SHORT_ALLOC       = 0.35
_LONG_Q           = 5
_SHORT_Q          = 1
_MIN_RANK_DATES   = 4   # trigger M2 backfill if fewer ranking dates found
_OHLCV_BUFFER_MS  = 14 * 86_400_000   # 14-day buffer after end_date for exit prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def _next_date(after: str, sorted_dates: list[str]) -> str | None:
    """First date in sorted_dates strictly greater than `after`."""
    for d in sorted_dates:
        if d > after:
            return d
    return None


def _ensure_rankings(conn, start_date: str, end_date: str) -> None:
    """Auto-trigger M2 historical backfill when rankings are sparse."""
    n = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM weekly_rankings WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchone()[0]
    if n < _MIN_RANK_DATES:
        logger.info(f"Only {n} ranking date(s) in [{start_date}, {end_date}] — running M2 backfill...")
        from m2_factor_engine.engine import run_historical_rankings
        run_historical_rankings(start_date, end_date)


def _load_data(
    conn,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    rankings_df, opens_df, closes_df, avg_vol_df, daily_funding_df
    All DataFrames use string date index ("YYYY-MM-DD") where applicable.
    """
    rankings = pd.read_sql_query(
        "SELECT * FROM weekly_rankings WHERE date >= ? AND date <= ? ORDER BY date",
        conn, params=[start_date, end_date],
    )
    if rankings.empty:
        return rankings, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    symbols = rankings["symbol"].unique().tolist()

    # OHLCV — load with a 35-day pre-buffer (rolling vol window) and 14-day post-buffer
    pre_ms  = _date_ms(start_date) - 35 * 86_400_000
    post_ms = _date_ms(end_date)   + _OHLCV_BUFFER_MS
    ph = ",".join("?" * len(symbols))

    ohlcv_raw = pd.read_sql_query(
        f"SELECT symbol, open_time, open, close, quote_volume FROM ohlcv "
        f"WHERE symbol IN ({ph}) AND open_time >= ? AND open_time <= ? ORDER BY open_time",
        conn, params=symbols + [pre_ms, post_ms],
    )

    if ohlcv_raw.empty:
        return rankings, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ohlcv_raw["date"] = (
        pd.to_datetime(ohlcv_raw["open_time"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d")
    )

    opens_df = ohlcv_raw.pivot_table(
        index="date", columns="symbol", values="open", aggfunc="last"
    ).reindex(columns=symbols)

    closes_df = ohlcv_raw.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).reindex(columns=symbols)

    # 30-day rolling average volume — used for slippage tier determination
    avg_vol_df = (
        ohlcv_raw.pivot_table(
            index="date", columns="symbol", values="quote_volume", aggfunc="last"
        )
        .reindex(columns=symbols)
        .rolling(30, min_periods=5)
        .mean()
    )

    # Funding rates — convert to daily mean per symbol
    funding_raw = pd.read_sql_query(
        f"SELECT symbol, funding_time, funding_rate FROM funding_rates "
        f"WHERE symbol IN ({ph}) AND funding_time >= ? AND funding_time <= ? ORDER BY funding_time",
        conn, params=symbols + [pre_ms, post_ms],
    )

    if not funding_raw.empty:
        funding_raw["date"] = (
            pd.to_datetime(funding_raw["funding_time"], unit="ms", utc=True)
            .dt.strftime("%Y-%m-%d")
        )
        daily_funding = (
            funding_raw.groupby(["date", "symbol"])["funding_rate"]
            .mean()
            .unstack("symbol")
            .reindex(columns=symbols)
            .fillna(0.0)
        )
    else:
        daily_funding = pd.DataFrame(0.0, index=opens_df.index, columns=symbols)

    return rankings, opens_df, closes_df, avg_vol_df, daily_funding


# ---------------------------------------------------------------------------
# Vectorised cost computation (same formula as costs.total_cost)
# ---------------------------------------------------------------------------

def _vec_costs(
    avg_vols: pd.Series,
    mean_frs: pd.Series,
    hold_days: int,
    notional_each: float,
) -> pd.Series:
    """Return absolute cost Series indexed by symbol."""
    slips = avg_vols.apply(
        lambda v: SLIP_TIER1 if pd.notna(v) and v >= 50_000_000 else SLIP_TIER2
    )
    funding = mean_frs * (24.0 / FUNDING_INTERVAL) * hold_days
    cost_frac = FEE_MAKER * 2 + slips * 2 + funding
    return cost_frac * notional_each


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    start_date: str,
    end_date: str,
    leverage: int = 3,
    long_only: bool = False,
) -> dict:
    """
    Vectorised weekly backtest.

    Entry rule  : open[signal_date + 1 calendar day]  (enforced via _next_date)
    Exit rule   : open[next_signal_date + 1 calendar day]
    MTM (intra) : daily close prices

    Returns dict with keys:
        equity_curve   (pd.Series, daily, DatetimeIndex)
        trade_log      (pd.DataFrame, one row per symbol-period)
        period_returns (pd.Series, one per rebalancing period)
        summary_stats  (dict)
        rankings       (pd.DataFrame, the loaded weekly_rankings slice)
    """
    with get_conn() as conn:
        _ensure_rankings(conn, start_date, end_date)

        # Survivorship bias guard: warn whenever backtest predates the earliest
        # universe snapshot so the limitation is never silently forgotten.
        earliest_snap = conn.execute(
            "SELECT MIN(date) FROM universe_history"
        ).fetchone()[0]
        if earliest_snap is None:
            logger.warning(
                "WARNING: universe_history is empty.  All backtest results are "
                "subject to survivorship bias -- rankings used today's universe."
            )
        elif start_date < earliest_snap:
            logger.warning(
                f"WARNING: universe_history has no snapshots before {earliest_snap}.  "
                f"Backtest results before this date are subject to survivorship bias "
                f"and should be treated as indicative only."
            )

        rankings, opens_df, closes_df, avg_vol_df, daily_funding = _load_data(
            conn, start_date, end_date
        )

    if rankings.empty:
        logger.error("No rankings available for this date range.")
        return {}
    if opens_df.empty:
        logger.error("No OHLCV data available for this date range.")
        return {}

    rebal_dates = sorted(rankings["date"].unique())
    if len(rebal_dates) < 2:
        logger.error(f"Need >= 2 rebalancing dates; found {len(rebal_dates)}.")
        return {}

    all_dates = sorted(opens_df.index.tolist())

    capital         = INITIAL_CAPITAL
    equity_series   = {}
    trade_frames: list[pd.DataFrame] = []
    period_rets: list[float]         = []
    skipped_trades  = 0

    for i, signal_date in enumerate(rebal_dates[:-1]):
        next_signal = rebal_dates[i + 1]

        entry_date = _next_date(signal_date, all_dates)
        exit_date  = _next_date(next_signal,  all_dates)

        if entry_date is None or exit_date is None:
            continue
        if entry_date not in opens_df.index or exit_date not in opens_df.index:
            continue

        # Enforce: signal date must strictly precede entry date
        assert signal_date < entry_date, \
            f"Lookahead: signal {signal_date} >= entry {entry_date}"

        # ── Positions for this period ─────────────────────────────────────────
        period_rank = rankings[rankings["date"] == signal_date]
        long_cands  = period_rank[period_rank["quintile"] == _LONG_Q]["symbol"].tolist()
        short_cands = (
            [] if long_only
            else period_rank[period_rank["quintile"] == _SHORT_Q]["symbol"].tolist()
        )

        def _filter_valid(syms: list[str], label: str) -> list[str]:
            valid = []
            for s in syms:
                if s not in opens_df.columns:
                    continue
                ep = opens_df.at[entry_date, s] if s in opens_df.columns else np.nan
                xp = opens_df.at[exit_date,  s] if s in opens_df.columns else np.nan
                if pd.notna(ep) and ep > 0 and pd.notna(xp) and xp > 0:
                    valid.append(s)
                else:
                    nonlocal skipped_trades
                    skipped_trades += 1
            return valid

        valid_long  = _filter_valid(long_cands,  "long")
        valid_short = _filter_valid(short_cands, "short")

        if not valid_long and not valid_short:
            continue

        n_long  = len(valid_long)
        n_short = len(valid_short)
        hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days or 1

        long_notional  = (capital * LONG_ALLOC  * leverage / n_long)  if n_long  else 0.0
        short_notional = (capital * SHORT_ALLOC * leverage / n_short) if n_short else 0.0

        # Entry and exit prices (vectorised Series)
        ep_long  = opens_df.loc[entry_date, valid_long]  if valid_long  else pd.Series(dtype=float)
        xp_long  = opens_df.loc[exit_date,  valid_long]  if valid_long  else pd.Series(dtype=float)
        ep_short = opens_df.loc[entry_date, valid_short] if valid_short else pd.Series(dtype=float)
        xp_short = opens_df.loc[exit_date,  valid_short] if valid_short else pd.Series(dtype=float)

        # Mean funding rate during hold period (vectorised)
        hold_dates = [d for d in daily_funding.index if entry_date <= d < exit_date]
        def _mean_fr(syms, sign=1.0) -> pd.Series:
            if not syms or not hold_dates:
                return pd.Series(0.0, index=syms)
            avail = [s for s in syms if s in daily_funding.columns]
            fr = daily_funding.loc[hold_dates, avail].mean() if hold_dates else pd.Series(0.0, index=avail)
            return fr.reindex(syms, fill_value=0.0) * sign

        fr_long  = _mean_fr(valid_long,  sign= 1.0)  # longs pay positive funding
        fr_short = _mean_fr(valid_short, sign=-1.0)  # shorts receive positive funding

        # 30-day avg volume at entry date
        def _avg_vol(syms) -> pd.Series:
            if not syms or entry_date not in avg_vol_df.index:
                return pd.Series(0.0, index=syms)
            return avg_vol_df.loc[entry_date, syms].fillna(0.0)

        av_long  = _avg_vol(valid_long)
        av_short = _avg_vol(valid_short)

        # ── P&L (all vectorised) ──────────────────────────────────────────────
        period_net = 0.0

        def _make_trades(
            syms, ep, xp, notional, direction, av, fr, cap
        ) -> pd.DataFrame | None:
            if not syms:
                return None
            ret = xp / ep - 1                            # raw price return
            gross = ret * notional * direction            # $ gross P&L
            cost_abs = _vec_costs(av, fr, hold_days, notional)
            net = gross - cost_abs
            return pd.DataFrame({
                "signal_date":      signal_date,
                "entry_date":       entry_date,
                "exit_date":        exit_date,
                "symbol":           syms,
                "direction":        direction,
                "entry_price":      ep.values,
                "exit_price":       xp.values,
                "notional":         notional,
                "capital_at_entry": cap,
                "gross_pnl":        gross.values,
                "cost":             cost_abs.values,
                "net_pnl":          net.values,
                "hold_days":        hold_days,
                "return_pct":       (ret * direction * 100).values,
            })

        df_long  = _make_trades(valid_long,  ep_long,  xp_long,  long_notional,  1, av_long,  fr_long,  capital)
        df_short = _make_trades(valid_short, ep_short, xp_short, short_notional, -1, av_short, fr_short, capital)

        for df in [df_long, df_short]:
            if df is not None and not df.empty:
                trade_frames.append(df)
                period_net += df["net_pnl"].sum()

        period_ret = period_net / capital
        period_rets.append(period_ret)
        capital += period_net

        # ── Daily MTM equity curve (within hold period, using closes) ─────────
        mtm_dates = [d for d in all_dates if entry_date <= d < exit_date and d in closes_df.index]

        if mtm_dates:
            unrealized = pd.Series(0.0, index=mtm_dates)
            if valid_long and n_long:
                cl = closes_df.loc[mtm_dates, valid_long].ffill().fillna(ep_long)
                unrealized += ((cl / ep_long) - 1).mul(long_notional).sum(axis=1)
            if valid_short and n_short:
                cs = closes_df.loc[mtm_dates, valid_short].ffill().fillna(ep_short)
                unrealized -= ((cs / ep_short) - 1).mul(short_notional).sum(axis=1)
            starting = capital - period_net  # equity at start of this period
            for d in mtm_dates:
                equity_series[d] = float(starting + unrealized[d])

        # Mark end-of-period (the exit open captures realized P&L)
        equity_series[exit_date] = float(capital)

        logger.debug(
            f"{signal_date}: L={n_long} S={n_short} hold={hold_days}d "
            f"net={period_net:+.2f} equity={capital:.2f}"
        )

    if not equity_series:
        logger.warning("No trades executed for this date range.")
        return {}

    equity_curve = pd.Series(equity_series).sort_index()
    equity_curve.index = pd.to_datetime(equity_curve.index)

    trade_log      = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    period_ret_s   = pd.Series(period_rets)

    stats = {
        "start_date":    start_date,
        "end_date":      end_date,
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(capital, 2),
        "total_return":  round((capital / INITIAL_CAPITAL - 1) * 100, 2),
        "cagr":          round(cagr(equity_curve) * 100, 2),
        "sharpe":        round(sharpe(period_ret_s), 3),
        "max_drawdown":  round(max_drawdown(equity_curve) * 100, 2),
        "calmar":        round(calmar(equity_curve), 3),
        "win_rate":      round(win_rate(trade_log) * 100, 2) if not trade_log.empty else 0.0,
        "total_trades":  len(trade_log),
        "avg_hold_days": round(trade_log["hold_days"].mean(), 1) if not trade_log.empty else 0.0,
        "total_costs":   round(trade_log["cost"].sum(), 2) if not trade_log.empty else 0.0,
        "leverage":      leverage,
        "long_only":     long_only,
        "periods_run":   len(period_rets),
        "skipped_trades": skipped_trades,
    }

    return {
        "equity_curve":   equity_curve,
        "trade_log":      trade_log,
        "period_returns": period_ret_s,
        "summary_stats":  stats,
        "rankings":       rankings,
    }


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def _print_extremes(trade_log: pd.DataFrame, n: int = 5) -> None:
    if trade_log.empty:
        return
    cols = ["signal_date", "symbol", "direction", "entry_price",
            "exit_price", "return_pct", "net_pnl", "hold_days"]
    avail = [c for c in cols if c in trade_log.columns]

    best  = trade_log.nlargest(n,  "net_pnl")[avail]
    worst = trade_log.nsmallest(n, "net_pnl")[avail]

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)

    print(f"\n  Top {n} best trades:")
    print(best.to_string(index=False))
    print(f"\n  Top {n} worst trades:")
    print(worst.to_string(index=False))


def _print_trade_detail(
    trade_log: pd.DataFrame,
    symbol: str,
    signal_date: str,
    leverage: int,
) -> None:
    """Print full sizing audit for a single trade."""
    mask = (trade_log["symbol"] == symbol) & (trade_log["signal_date"] == signal_date)
    t = trade_log[mask]
    if t.empty:
        print(f"\n  No trade found for {symbol} on signal_date={signal_date}")
        return
    row = t.iloc[0]
    cap   = row.get("capital_at_entry", float("nan"))
    notl  = row["notional"]
    alloc = LONG_ALLOC if row["direction"] == 1 else SHORT_ALLOC
    n_pos = round(cap * alloc * leverage / notl) if notl else 0
    expected = cap * alloc * leverage / n_pos if n_pos else float("nan")

    raw_ret = row["exit_price"] / row["entry_price"] - 1

    print()
    print("=" * 62)
    print(f"  TRADE AUDIT: {symbol}  signal_date={signal_date}")
    print("=" * 62)
    print(f"  Entry date       : {row['entry_date']}")
    print(f"  Exit date        : {row['exit_date']}")
    print(f"  Direction        : {'LONG (+1)' if row['direction'] == 1 else 'SHORT (-1)'}")
    print(f"  Entry price      : {row['entry_price']:.6f}")
    print(f"  Exit price       : {row['exit_price']:.6f}")
    print(f"  Raw price return : {raw_ret * 100:+.4f}%")
    print(f"  Capital at entry : ${cap:>12,.2f}")
    print(f"  Basket symbols   : {n_pos}  (inferred from notional)")
    print(f"  Allocation frac  : {alloc:.2f} x {leverage}x leverage")
    print(f"  Notional (actual): ${notl:>12,.2f}  = cap * {alloc} * {leverage} / {n_pos}")
    print(f"  Notional (expect): ${expected:>12,.2f}  (should match actual)")
    print(f"  Gross P&L        : ${row['gross_pnl']:>+12,.2f}")
    print(f"  Cost             : ${row['cost']:>12,.2f}")
    print(f"  Net P&L          : ${row['net_pnl']:>+12,.2f}")
    print(f"  Hold days        : {row['hold_days']}")
    ok = abs(notl - expected) < 1.0
    print(f"  Sizing check     : {'PASS -- actual == expected' if ok else 'FAIL -- mismatch detected'}")
    print("=" * 62)


def _print_attribution(trade_log, rankings) -> None:
    q_stats, ics = factor_attribution(trade_log, rankings)
    if q_stats.empty:
        return

    print("\n  Quintile attribution:")
    print(q_stats.to_string())

    if not ics.empty:
        print("\n  Factor ICs (Spearman rank correlation with trade return):")
        for k, v in ics.items():
            bar = "#" * int(abs(v) * 30)
            sign = "+" if v >= 0 else "-"
            print(f"    {k:<16} {v:>+.4f}  {sign}{bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Backtest Engine")
    parser.add_argument("--start",     default="2024-08-01",
                        help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end",       default="2026-06-02",
                        help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--leverage",  type=int, default=3)
    parser.add_argument("--long-only", action="store_true", dest="long_only")
    args = parser.parse_args()

    result = run_backtest(args.start, args.end, args.leverage, args.long_only)

    if result:
        print_summary(result["summary_stats"])
        _print_trade_detail(
            result["trade_log"], "SIRENUSDT", "2026-03-16", args.leverage
        )
        _print_extremes(result["trade_log"])
        _print_attribution(result["trade_log"], result["rankings"])
