import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = 52) -> float:
    """Annualised Sharpe on weekly returns. periods=52 for weekly frequency."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Peak-to-trough drawdown as a fraction (negative number)."""
    if equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    return float(dd.min())


def cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate."""
    if len(equity_curve) < 2:
        return 0.0
    idx = pd.to_datetime(equity_curve.index)
    years = (idx[-1] - idx[0]).days / 365.25
    if years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1)


def calmar(equity_curve: pd.Series) -> float:
    """CAGR / abs(max_drawdown). Returns 0 if drawdown is zero."""
    mdd = max_drawdown(equity_curve)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(cagr(equity_curve) / abs(mdd))


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of individual trades with net_pnl > 0."""
    if trade_log.empty:
        return 0.0
    return float((trade_log["net_pnl"] > 0).mean())


def factor_attribution(
    trade_log: pd.DataFrame,
    weekly_rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      quintile_stats  -- avg return and total P&L grouped by quintile
      factor_ics      -- Spearman IC of each factor z-score vs trade return
    """
    if trade_log.empty or weekly_rankings.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    rank_cols = [
        "date", "symbol", "quintile",
        "momentum_z", "carry_z", "size_z", "liquidity_z", "oi_z", "composite_z",
    ]
    avail = [c for c in rank_cols if c in weekly_rankings.columns]
    merged = trade_log.merge(
        weekly_rankings[avail],
        left_on=["signal_date", "symbol"],
        right_on=["date", "symbol"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    quintile_stats = merged.groupby("quintile").agg(
        avg_return_pct=("return_pct", "mean"),
        total_net_pnl=("net_pnl", "sum"),
        n_trades=("net_pnl", "count"),
    )

    # Direction-adjusted return for IC: sign-flip shorts so long profit and
    # short profit both appear positive, aligned with composite factor rank.
    # return_for_ic = (exit_price - entry_price) / entry_price * direction
    merged["return_for_ic"] = (
        (merged["exit_price"] - merged["entry_price"])
        / merged["entry_price"]
        * merged["direction"]
    )

    factor_cols = ["momentum_z", "carry_z", "size_z", "liquidity_z", "oi_z", "composite_z"]
    ics = {}
    for fc in factor_cols:
        if fc in merged.columns:
            pair = merged[[fc, "return_for_ic"]].dropna()
            if len(pair) > 2:
                ics[fc] = round(
                    pair.corr(method="spearman").iloc[0, 1], 4
                )
    return quintile_stats, pd.Series(ics, name="spearman_IC")


def print_summary(stats: dict) -> None:
    print()
    print("=" * 54)
    print(f"  BACKTEST  {stats.get('start_date','')} -> {stats.get('end_date','')}")
    print("=" * 54)
    print(f"  Start capital  : ${stats.get('initial_capital', 0):>12,.2f}")
    print(f"  Final capital  : ${stats.get('final_capital', 0):>12,.2f}")
    print(f"  Total return   : {stats.get('total_return', 0):>+12.2f}%")
    print(f"  CAGR           : {stats.get('cagr', 0):>+12.2f}%")
    print(f"  Sharpe (wkly)  : {stats.get('sharpe', 0):>12.3f}")
    print(f"  Max drawdown   : {stats.get('max_drawdown', 0):>+12.2f}%")
    print(f"  Calmar ratio   : {stats.get('calmar', 0):>12.3f}")
    print(f"  Win rate       : {stats.get('win_rate', 0):>12.2f}%")
    print(f"  Total trades   : {stats.get('total_trades', 0):>12,d}")
    print(f"  Avg hold (d)   : {stats.get('avg_hold_days', 0):>12.1f}")
    print(f"  Total costs    : ${stats.get('total_costs', 0):>12,.2f}")
    print(f"  Leverage       : {stats.get('leverage', 1):>11}x")
    print(f"  Mode           : {'long-only' if stats.get('long_only') else 'long-short':>12}")
    print("=" * 54)
