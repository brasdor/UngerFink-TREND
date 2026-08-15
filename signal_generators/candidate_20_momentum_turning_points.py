#!/usr/bin/env python3
"""
Candidate 20: Momentum turning points -- dual-speed (slow/fast) time-series momentum
blended per asset into a Bull/Correction/Bear/Rebound classification, used to GATE which
assets are even eligible for the cross-sectional long/short ranking basket.

Mechanistically different from the regime filters already rejected earlier this session
(ADX/Hurst-based market-level filters retrofitted onto 02u/03u, which failed to
generalize): those bolt an EXTERNAL classifier onto an existing entry mechanism and can
interrupt/flatten open positions whenever the external state flips. This blends two
speeds of the SAME time-series-momentum signal, per-asset, and only affects the
formation/entry decision at each rebalance -- it cannot flip an OPEN position mid-hold,
since eligibility is recomputed only at rebalance dates exactly like candidate 12's own
membership assignment.

Per-asset state (slow_mom = pct_change(slow_n), fast_mom = pct_change(fast_n)):
    Bull:       slow > 0, fast > 0   (established uptrend continuing)
    Rebound:    slow <= 0, fast > 0  (downtrend but recent bounce)
    Correction: slow > 0, fast <= 0  (uptrend but recent pullback)
    Bear:       slow <= 0, fast <= 0 (established downtrend continuing)

Long basket is drawn ONLY from {Bull, Rebound} (i.e. anything with positive recent
(fast) momentum, regardless of the slow trend's sign); short basket (side=long_short)
ONLY from {Bear, Correction}. Within each eligible pool, ranking uses the combined
score = slow_mom + fast_mom (the same two signals used for classification, not a third
independent ranking variable) -- top quantile of the long-eligible pool goes long,
bottom quantile of the short-eligible pool goes short.

Uses the same generate_universe_positions(panel, params) interface as candidate 12/18/19.
"""
from __future__ import annotations

import pandas as pd


def generate_universe_positions(panel: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    slow_n = params["slow_n"]
    fast_n = params["fast_n"]
    quantile = params["quantile"]
    rebalance_n = params["rebalance_n"]
    side = params["side"]
    min_universe = 10

    closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()}).sort_index()
    slow_mom = closes.pct_change(slow_n)
    fast_mom = closes.pct_change(fast_n)
    score = slow_mom + fast_mom

    long_eligible = (fast_mom > 0)   # Bull (slow>0) or Rebound (slow<=0)
    short_eligible = (fast_mom <= 0)  # Correction (slow>0) or Bear (slow<=0)

    all_dates = closes.index
    rebalance_dates = set(all_dates[slow_n::rebalance_n])

    membership = pd.DataFrame(0, index=all_dates, columns=closes.columns, dtype=int)
    current = pd.Series(0, index=closes.columns, dtype=int)
    for date in all_dates:
        if date in rebalance_dates:
            score_row = score.loc[date]
            new_assignment = pd.Series(0, index=closes.columns, dtype=int)

            long_pool = score_row[long_eligible.loc[date]].dropna()
            if len(long_pool) >= min_universe:
                n_select = max(1, int(len(long_pool) * quantile))
                top = long_pool.sort_values(ascending=False).index[:n_select]
                new_assignment.loc[top] = 1

            if side == "long_short":
                short_pool = score_row[short_eligible.loc[date]].dropna()
                if len(short_pool) >= min_universe:
                    n_select_s = max(1, int(len(short_pool) * quantile))
                    bottom = short_pool.sort_values(ascending=True).index[:n_select_s]
                    new_assignment.loc[bottom] = -1

            current = new_assignment
        membership.loc[date] = current

    return {sym: membership[sym].reindex(panel[sym].index).fillna(0).astype(int) for sym in panel}


PARAM_GRID = {
    "slow_n": [180, 365],
    "fast_n": [20, 30],
    "quantile": [0.1, 0.2],
    "rebalance_n": [7, 14],
    "side": ["long_only", "long_short"],
}
