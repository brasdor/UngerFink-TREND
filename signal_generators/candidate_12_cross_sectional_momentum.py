#!/usr/bin/env python3
"""
Candidate 12: Cross-sectional (rank-based) momentum.

Distinct from candidate 2's time-series/absolute momentum (which trades sign(own
trailing return), independently per asset): this ranks the ENTIRE universe by
trailing return at each rebalance date and goes long the top decile / short the
bottom decile (relative strength), regardless of whether those returns are
positive or negative in absolute terms. Genuinely different risk driver --
depends on the universe's dispersion/leadership rotation, not on any one asset's
own absolute trend.

This does NOT fit the standard per-symbol generate_signal(price_data, params)
interface used elsewhere, because ranking requires simultaneous knowledge of
every symbol's return at each date -- a single symbol's own price history isn't
enough. generate_universe_positions() below takes the whole panel at once and
returns a per-symbol position dict; the driver script feeds these into the
harness's lower-level trade-conversion primitives directly rather than through
run_t1_universe's per-symbol-independent loop.
"""
from __future__ import annotations

import pandas as pd


def generate_universe_positions(panel: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    """Returns {symbol: position_series} for every symbol in panel, each aligned to
    that symbol's own index. Rebalances every `rebalance_n` bars: ranks all symbols
    with valid trailing-`formation`-day returns, assigns the top `quantile` fraction
    long and (if side=='long_short') the bottom `quantile` fraction short, holding
    that assignment flat until the next rebalance."""
    formation = params["formation"]
    quantile = params["quantile"]
    rebalance_n = params["rebalance_n"]
    side = params["side"]
    min_universe = 10  # minimum symbols with valid data to attempt a meaningful rank

    closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()}).sort_index()
    trailing_ret = closes.pct_change(formation)

    all_dates = closes.index
    rebalance_dates = set(all_dates[formation::rebalance_n])

    membership = pd.DataFrame(0, index=all_dates, columns=closes.columns, dtype=int)
    current = pd.Series(0, index=closes.columns, dtype=int)
    for date in all_dates:
        if date in rebalance_dates:
            row = trailing_ret.loc[date].dropna()
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

    return {sym: membership[sym].reindex(panel[sym].index).fillna(0).astype(int)
            for sym in panel}


PARAM_GRID = {
    "formation": [30, 60, 90],
    "quantile": [0.1, 0.2],
    "rebalance_n": [7, 14],
    "side": ["long_only", "long_short"],
}
