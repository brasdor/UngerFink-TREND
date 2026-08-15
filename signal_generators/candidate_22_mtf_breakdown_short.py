#!/usr/bin/env python3
"""
Candidate 22: Multi-timeframe breakdown confirmation short.

Requires the breakdown to be confirmed on both a faster and a slower timeframe context
before entering -- NOT simply two Donchian breakdown lookbacks of different length
(low.rolling(slow_n).min() <= low.rolling(fast_n).min() always for slow_n > fast_n, so
requiring BOTH conditions on the same breakdown definition is trivially redundant --
whichever is harder to satisfy, i.e. the slow one, dominates and the fast condition adds
nothing). Instead: the FAST Donchian breakdown is the entry trigger (reactive), and the
SLOW-moving-average trend context (price already below its own slow_n-period mean) is
the confirming condition -- genuinely two independent pieces of information, standing in
for "higher timeframe" trend confirmation using only the 1D data this repo has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    fast_n, slow_n, exit_n = params["fast_n"], params["slow_n"], params["exit_n"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]

    breakdown_fast = close < low.rolling(fast_n).min().shift(1)
    slow_trend_down = close < close.rolling(slow_n).mean().shift(1)
    confirmed_breakdown = breakdown_fast & slow_trend_down
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[confirmed_breakdown] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "fast_n": [10, 20],
    "slow_n": [55, 100],
    "exit_n": [10, 20],
}
