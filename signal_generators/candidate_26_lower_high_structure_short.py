#!/usr/bin/env python3
"""
Candidate 26: Structural lower-high confirmation short.

Genuinely different top-confirmation MECHANISM from 22 (trend context), 24 (momentum),
and 25 (volume): classic Dow-theory swing structure. Requires a CONFIRMED lower high --
the most recent swing-high pivot sits below the previous swing-high pivot -- before a
subsequent breakdown is allowed to trigger a short, rather than trading a rolling-low
breakdown alone (candidate 7/23) with no structural context at all.

Swing-high pivot detection: bar i is a pivot high if its `high` is the max within a
centered window of `pivot_n` bars either side. This looks `pivot_n` bars into the
future to CONFIRM the pivot, so the signal is shifted forward by `pivot_n` bars --
the pivot is only used once those bars have actually occurred, no lookahead into
data not yet available at decision time. A forward loop over the (sparse) confirmed
pivot points tracks the two most recent distinct pivot prices to detect a lower high;
once detected, the signal stays "armed" for a following window, during which the
actual short entry fires on a `trigger_n`-day breakdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOWER_HIGH_ARM_WINDOW = 20  # bars a confirmed lower-high stays "active" waiting for the breakdown trigger


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    pivot_n = params["pivot_n"]
    trigger_n, exit_n = params["trigger_n"], params["exit_n"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]

    window = 2 * pivot_n + 1
    rolling_max = high.rolling(window, center=True).max()
    is_pivot_high = (high == rolling_max) & rolling_max.notna()
    # centered rolling looks pivot_n bars into the future -- shift forward so the pivot
    # is only "known" once those future bars have actually occurred
    confirmed_pivot = is_pivot_high.shift(pivot_n, fill_value=False)
    confirmed_price = high.shift(pivot_n)

    lower_high_signal = pd.Series(False, index=price_data.index)
    conf_idx = np.where(confirmed_pivot.to_numpy())[0]
    conf_prices = confirmed_price.to_numpy()
    last_pivot_price = None
    for i in conf_idx:
        price_i = conf_prices[i]
        if np.isnan(price_i):
            continue
        if last_pivot_price is not None and price_i < last_pivot_price:
            lower_high_signal.iloc[i] = True
        last_pivot_price = price_i

    armed = lower_high_signal.rolling(LOWER_HIGH_ARM_WINDOW, min_periods=1).max().astype(bool)
    breakdown_trigger = close < low.rolling(trigger_n).min().shift(1)
    entry = armed & breakdown_trigger
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[entry] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "pivot_n": [5, 10],
    "trigger_n": [10, 20],
    "exit_n": [10, 20],
}
