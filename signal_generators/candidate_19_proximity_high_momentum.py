#!/usr/bin/env python3
"""
Candidate 19: 52-week-high / proximity-to-high momentum (George & Hwang 2004).

Distinct formation mechanism from both candidate 12 (raw trailing return) and
candidate 18 (beta-stripped trailing return): ranks assets by how close their current
price sits to its own trailing high, not by return magnitude at all. Two assets with
identical trailing returns can have very different proximity-to-high if one dipped and
recovered while the other trended monotonically -- this is a genuinely different signal,
not a re-parameterization of return-based momentum.

`lookback` is the crypto-equivalent of the traditional 252-trading-day "52-week" window
-- grid spans 90/180/252 days since crypto cycles compress faster than traditional
equity cycles, so shorter proximity windows are plausible alternatives worth testing
alongside the literal 252-day anchor.

Uses the same generate_universe_positions(panel, params) interface as candidate 12/18.
"""
from __future__ import annotations

import pandas as pd


def generate_universe_positions(panel: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    lookback = params["lookback"]
    quantile = params["quantile"]
    rebalance_n = params["rebalance_n"]
    side = params["side"]
    min_universe = 10

    closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()}).sort_index()
    rolling_high = closes.rolling(lookback, min_periods=max(20, lookback // 4)).max()
    proximity = closes / rolling_high  # in (0, 1], 1.0 = currently AT the trailing high

    all_dates = closes.index
    rebalance_dates = set(all_dates[lookback::rebalance_n])

    membership = pd.DataFrame(0, index=all_dates, columns=closes.columns, dtype=int)
    current = pd.Series(0, index=closes.columns, dtype=int)
    for date in all_dates:
        if date in rebalance_dates:
            row = proximity.loc[date].dropna()
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
    "lookback": [90, 180, 252],
    "quantile": [0.1, 0.2],
    "rebalance_n": [7, 14],
    "side": ["long_only", "long_short"],
}
