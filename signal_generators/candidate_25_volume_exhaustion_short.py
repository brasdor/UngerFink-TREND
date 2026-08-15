#!/usr/bin/env python3
"""
Candidate 25: Volume exhaustion pattern at highs -- top confirmation via volume,
distinct from candidate 22 (trend) and candidate 24 (momentum).

`pattern` selects between two distinct exhaustion mechanisms, both tested explicitly
rather than picked a priori:
  - 'climax':    a blow-off top -- volume SPIKES far above its own recent average while
                 price sits at a new high (capitulation-style buying exhaustion).
  - 'declining': a "rally on fumes" -- volume TREND is falling while price still makes
                 new highs (participation drying up before the reversal).

Same arm-then-trigger structure as candidate 24: an exhaustion flag stays armed for a
following window; the actual short entry fires on a subsequent breakdown while armed.

Original T1 grid (16 combos) found the 'climax' pattern consistently and meaningfully
better than 'declining' -- best combo (lookback=30, trigger_n=20, exit_n=10) reached
avg_r=0.235R, t=2.49, zone_frac=0.71, a near-miss just under the 0.25R floor. PARAM_GRID
is preserved unchanged for the historical record; PARAM_GRID_REFINED (2026-08-15) tests
a tighter grid around that near-miss region with finer step sizes and a slightly LOWER
climax-volume-multiplier threshold (more trades at similar quality, rather than fewer
trades at a higher bar) -- climax_volume_mult is now a sweepable param (defaults to the
original CLIMAX_VOLUME_MULT=2.0 when omitted, so the original grid's results are exactly
reproducible unchanged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXHAUSTION_ARM_WINDOW = 10
NEAR_HIGH_TOLERANCE = 0.999
CLIMAX_VOLUME_MULT = 2.0        # default/original: volume must exceed 2x its own lookback average
VOLUME_TREND_WINDOW = 10        # window used to measure the 'declining' volume trend


def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
    lookback, pattern = params["lookback"], params["pattern"]
    trigger_n, exit_n = params["trigger_n"], params["exit_n"]
    climax_volume_mult = params.get("climax_volume_mult", CLIMAX_VOLUME_MULT)
    high, low, close, volume = (price_data["high"], price_data["low"],
                                 price_data["close"], price_data["volume"])

    price_at_high = close >= close.rolling(lookback).max() * NEAR_HIGH_TOLERANCE
    avg_volume = volume.rolling(lookback).mean()

    if pattern == "climax":
        volume_signal = volume > avg_volume * climax_volume_mult
    else:  # 'declining'
        volume_trend = volume.rolling(VOLUME_TREND_WINDOW).mean().diff(VOLUME_TREND_WINDOW)
        volume_signal = volume_trend < 0

    exhaustion = price_at_high & volume_signal
    armed = exhaustion.rolling(EXHAUSTION_ARM_WINDOW, min_periods=1).max().astype(bool)
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
    "pattern": ["climax", "declining"],
    "trigger_n": [10, 20],
    "exit_n": [10, 20],
}

# Refinement grid (2026-08-15): tighter around the near-miss (lookback=30, trigger_n=20,
# exit_n=10), finer steps, climax pattern only (declining was clearly and uniformly
# worse -- no reason to re-test it), climax_volume_mult swept slightly BELOW the
# original fixed 2.0 to test whether more (slightly less extreme) climax events at
# similar per-trade quality beats fewer, more extreme ones.
PARAM_GRID_REFINED = {
    "lookback": [25, 30, 35],
    "pattern": ["climax"],
    "trigger_n": [15, 20, 25],
    "exit_n": [10],
    "climax_volume_mult": [1.5, 1.75, 2.0],
}
