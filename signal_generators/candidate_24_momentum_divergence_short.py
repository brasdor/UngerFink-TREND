#!/usr/bin/env python3
"""
Candidate 24: Momentum/price divergence at highs -- top confirmation via momentum,
distinct from candidate 22 (trend-context confirmation) and candidate 25 (volume
confirmation).

Flags a bearish divergence when price makes a new `lookback`-day high while momentum
(rate-of-change over `mom_n` days) sits meaningfully below its own recent peak --
classic "price makes a higher high, momentum doesn't" top signal. A divergence flag
stays "armed" for a following window (10 bars); the actual short entry only fires on
a subsequent `trigger_n`-day breakdown WHILE armed -- divergence identifies the
top-confirmation condition, the breakdown trigger identifies the actual entry timing
(divergence alone is not tradeable, price can make new highs on weakening momentum
for a long time before actually reversing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIVERGENCE_ARM_WINDOW = 10  # bars a divergence flag stays "active" waiting for the breakdown trigger
NEAR_HIGH_TOLERANCE = 0.999  # price within 0.1% of the lookback high counts as "at the high"
MOMENTUM_DECAY_FRACTION = 0.7  # momentum must sit at/below 70% of its own recent peak to count as diverging


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    lookback, mom_n = params["lookback"], params["mom_n"]
    trigger_n, exit_n = params["trigger_n"], params["exit_n"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]

    momentum = close.pct_change(mom_n)
    price_at_high = close >= close.rolling(lookback).max() * NEAR_HIGH_TOLERANCE
    momentum_peak = momentum.rolling(lookback).max()
    divergence = price_at_high & (momentum < momentum_peak * MOMENTUM_DECAY_FRACTION)

    armed = divergence.rolling(DIVERGENCE_ARM_WINDOW, min_periods=1).max().astype(bool)
    breakdown_trigger = close < low.rolling(trigger_n).min().shift(1)
    entry = armed & breakdown_trigger
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[entry] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "lookback": [30, 60],
    "mom_n": [14],
    "trigger_n": [10, 20],
    "exit_n": [10, 20],
}
