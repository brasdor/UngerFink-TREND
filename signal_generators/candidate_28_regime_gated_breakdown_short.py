#!/usr/bin/env python3
"""
Candidate 28: Regime-gated breakdown short (comparison baseline for candidate 27).

Same BTC-confirmed-downtrend gate as candidate 27 (modules/btc_regime_gate.py,
trend_n=100 fixed here -- see run_candidates_27_28_shortgate.py for why: candidate 27
sweeps trend_n as part of its own grid, but this comparison needs ONE fixed gate
definition shared identically across both candidates' actual entries, and 100 is one
of candidate 27's own tested values), applied to candidate 7's plain Donchian
breakdown-short logic with NO beta-based selection -- isolates whether the regime gate
ALONE (without candidate 27's beta-ranking piece) fixes the bull-biased base-rate
problem that made candidate 7 (and the whole 21-26 batch) fail.

price_data must include a `btc_downtrend` boolean column, merged the same way
funding_rate was merged for candidate 21 (see the driver script) -- computed ONCE from
BTC's own price data and broadcast to every symbol by date alignment, since this is a
market-wide gate, not a per-symbol quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_N_FIXED = 100  # BTC EMA window for the shared downtrend gate


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    entry_n, exit_n = params["entry_n"], params["exit_n"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]
    btc_downtrend = price_data["btc_downtrend"]

    breakdown = close < low.rolling(entry_n).min().shift(1)
    confirmed_breakdown = breakdown & btc_downtrend
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[confirmed_breakdown] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "entry_n": [20, 55, 100],
    "exit_n": [10, 20],
}
