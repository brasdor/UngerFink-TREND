#!/usr/bin/env python3
"""
Candidate 23: Slower-only Donchian breakdown short.

Identical construction to candidate 7 (symmetric Donchian breakdown short) -- the
"slower-only" label refers entirely to the PARAM_GRID, not new logic. Candidate 7's
own extended grid (entry_n up to 124) kept climbing toward the edge with no interior
peak (edge_clustering flagged, no true optimum found within the tested range) --
before concluding the concept fails outright, this tests whether an even wider,
genuinely slower N/M resolves into an interior optimum or keeps climbing (confirming
failure) or reverses (revealing the true peak was just past the previously-tested edge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    entry_n, exit_n = params["entry_n"], params["exit_n"]
    high, low, close = price_data["high"], price_data["low"], price_data["close"]

    breakdown = close < low.rolling(entry_n).min().shift(1)
    breakout = close > high.rolling(exit_n).max().shift(1)

    raw = pd.Series(np.nan, index=price_data.index)
    raw[breakdown] = -1
    raw[breakout] = 0
    position = raw.ffill().fillna(0).astype(int)
    return position


PARAM_GRID = {
    "entry_n": [150, 200, 250, 300],
    "exit_n": [30, 50],
}
