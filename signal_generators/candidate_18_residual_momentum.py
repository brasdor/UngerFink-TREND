#!/usr/bin/env python3
"""
Candidate 18: Idiosyncratic / residual momentum.

Distinct from candidate 12's raw cross-sectional momentum (ranks trailing return
directly): this strips out each asset's BTC-beta component before ranking, targeting
the crash-reduction mechanism the residual-momentum literature (Blitz, Huij & Martens
2011) attributes to removing market-wide co-movement from the ranking signal -- an
asset that merely moved WITH BTC shouldn't rank as "momentum," only the excess/
idiosyncratic component should.

Simplification (stated explicitly, matching candidate 16's own precedent for flagging
simplifications): the textbook construction cumulates DAILY OLS residuals over the
formation window. This instead computes a single rolling beta (cov/var of daily returns
over the trailing `formation` days) and subtracts beta-scaled BTC cumulative return from
the asset's own cumulative return over the same window -- residual_score = cum_asset_ret
- beta * cum_btc_ret. This is a standard, much cheaper practitioner approximation of the
same idea (equivalent to a single-point OLS residual rather than a cumulated daily
residual series) and is vectorized across the whole panel via pandas rolling ops rather
than looping per-asset in Python, which would not be tractable at 290 symbols x a full
grid of formation windows.

Uses the same generate_universe_positions(panel, params) interface as candidate 12
(ranking requires simultaneous knowledge of the whole universe, not a per-symbol loop).
"""
from __future__ import annotations

import pandas as pd


def generate_universe_positions(panel: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    formation = params["formation"]
    quantile = params["quantile"]
    rebalance_n = params["rebalance_n"]
    side = params["side"]
    min_universe = 10

    if "BTCUSDT" not in panel:
        raise ValueError("candidate 18 requires BTCUSDT in the panel as the market benchmark")

    closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()}).sort_index()
    rets = closes.pct_change()
    btc_ret = rets["BTCUSDT"]

    roll_cov = rets.rolling(formation).cov(btc_ret)
    roll_var = btc_ret.rolling(formation).var()
    beta = roll_cov.div(roll_var, axis=0)

    cum_ret = closes.pct_change(formation)
    btc_cum_ret = closes["BTCUSDT"].pct_change(formation)
    residual = cum_ret.sub(beta.mul(btc_cum_ret, axis=0))

    all_dates = closes.index
    rebalance_dates = set(all_dates[formation::rebalance_n])

    membership = pd.DataFrame(0, index=all_dates, columns=closes.columns, dtype=int)
    current = pd.Series(0, index=closes.columns, dtype=int)
    for date in all_dates:
        if date in rebalance_dates:
            row = residual.loc[date].dropna()
            if len(row) >= min_universe:
                n_select = max(1, int(len(row) * quantile))
                ranked = row.sort_values(ascending=False)
                top = ranked.index[:n_select]
                bottom = ranked.index[-n_select:]
                new_assignment = pd.Series(0, index=closes.columns, dtype=int)
                new_assignment.loc[top] = 1
                if side == "long_short":
                    new_assignment.loc[bottom] = -1
                current = new_assignment
        membership.loc[date] = current

    return {sym: membership[sym].reindex(panel[sym].index).fillna(0).astype(int) for sym in panel}


PARAM_GRID = {
    "formation": [30, 60, 90],
    "quantile": [0.1, 0.2],
    "rebalance_n": [7, 14],
    "side": ["long_only", "long_short"],
}
