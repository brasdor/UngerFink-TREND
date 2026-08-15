#!/usr/bin/env python3
"""
Candidate 21: Funding-rate-extreme confirmation short.

Same Donchian breakdown trigger as candidate 7, but only allowed to fire when funding
is EXTREMELY positive at the moment of breakdown -- the empirical BTC test motivating
this batch found the real cost of a naive breakdown-short concentrated at unconfirmed
tops (shorting a breakdown that was really just a routine pullback in an ongoing
uptrend), not during genuine trend reversals. Extreme positive funding is a direct
market-structure signal of over-leveraged longs, i.e. real evidence a top may be
forming, distinct from candidate 22/24/25/26's price-structure-only confirmations.

price_data must include a `funding_rate` column (daily-resampled), merged the same
way as candidate 5 (see tools/run_candidate21_26_shorting_batch.py).

funding_threshold is a per-8h rate (e.g. 0.0005 = 0.05% per 8h); smoothed over 3 days
to avoid triggering on a single noisy funding reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    entry_n, exit_n = params["entry_n"], params["exit_n"]
    funding_threshold = params["funding_threshold"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]
    funding = price_data["funding_rate"]

    breakdown = close < low.rolling(entry_n).min().shift(1)
    funding_extreme = funding.rolling(3).mean() > funding_threshold
    confirmed_breakdown = breakdown & funding_extreme
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[confirmed_breakdown] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "entry_n": [20, 55, 100],
    "exit_n": [10, 20],
    "funding_threshold": [0.0003, 0.0006],
}
