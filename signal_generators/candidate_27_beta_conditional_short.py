#!/usr/bin/env python3
"""
Candidate 27: BTC-beta-conditional short.

Mechanistically different from every candidate in the 21-26 batch: none of those try
to time any individual asset's own top -- this uses BTC's OWN confirmed downtrend as
the entry trigger (modules/btc_regime_gate.py, shared with candidate 28 for a valid
comparison) and trailing beta-to-BTC as the SELECTION criterion within that window,
shorting the highest-beta subset -- the assets structurally most likely to fall
hardest alongside BTC, not the ones showing their own idiosyncratic weakness.

At each rebalance: if BTC is NOT in a confirmed downtrend, positions go flat (no
selection is made, matching "only allow short entries when BTC is in a confirmed
downtrend" -- positions are only opened during confirmed-downtrend rebalances). If BTC
IS in a confirmed downtrend, rank all non-BTC symbols by trailing beta-to-BTC (rolling
cov/var regression over `beta_lookback` days) and short the top `quantile` fraction.

Uses the generate_universe_positions(panel, params) ranking interface (beta ranking
requires the whole panel simultaneously), same as candidates 12/18/19/20.
"""
from __future__ import annotations

import pandas as pd

from modules.btc_regime_gate import btc_confirmed_downtrend


def generate_universe_positions(panel: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    beta_lookback = params["beta_lookback"]
    quantile = params["quantile"]
    rebalance_n = params["rebalance_n"]
    trend_n = params["trend_n"]
    min_universe = 10

    if "BTCUSDT" not in panel:
        raise ValueError("candidate 27 requires BTCUSDT in the panel as the market benchmark")

    closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()}).sort_index()
    rets = closes.pct_change()
    btc_ret = rets["BTCUSDT"]

    roll_cov = rets.rolling(beta_lookback).cov(btc_ret)
    roll_var = btc_ret.rolling(beta_lookback).var()
    beta = roll_cov.div(roll_var, axis=0)
    beta = beta.drop(columns=["BTCUSDT"], errors="ignore")

    downtrend = btc_confirmed_downtrend(panel["BTCUSDT"], trend_n).reindex(closes.index).fillna(False)

    all_dates = closes.index
    rebalance_dates = set(all_dates[beta_lookback::rebalance_n])

    membership = pd.DataFrame(0, index=all_dates, columns=closes.columns, dtype=int)
    current = pd.Series(0, index=closes.columns, dtype=int)
    for date in all_dates:
        if date in rebalance_dates:
            if downtrend.loc[date]:
                row = beta.loc[date].dropna()
                if len(row) >= min_universe:
                    n_select = max(1, int(len(row) * quantile))
                    top_beta = row.sort_values(ascending=False).index[:n_select]
                    new_assignment = pd.Series(0, index=closes.columns, dtype=int)
                    new_assignment.loc[top_beta] = -1
                    current = new_assignment
                else:
                    current = pd.Series(0, index=closes.columns, dtype=int)
            else:
                current = pd.Series(0, index=closes.columns, dtype=int)
        membership.loc[date] = current

    return {sym: membership[sym].reindex(panel[sym].index).fillna(0).astype(int) for sym in panel}


PARAM_GRID = {
    "beta_lookback": [60, 90],
    "quantile": [0.1, 0.2],
    "rebalance_n": [7, 14],
    "trend_n": [50, 100],
}
