#!/usr/bin/env python3
"""
Shared BTC-confirmed-downtrend gate, built for candidates 27 (BTC-beta-conditional
short) and 28 (regime-gated breakdown short) -- both need the IDENTICAL gate
definition so their results are a valid apples-to-apples comparison of "regime gate
alone" vs "regime gate + beta selection", not confounded by two different downtrend
definitions.

Confirmed downtrend = BTC close below its own trailing EMA(trend_n) AND ADX/Hurst
regime state == 'trending' (modules/regime_classifier.py) -- combines a DIRECTIONAL
filter (below trend) with a STRENGTH/persistence filter (confirmed trending, not
choppy), so a routine dip below a moving average during a choppy period doesn't count
as a "confirmed" downtrend on its own. This reuses candidate 1's regime classifier
directly rather than reinventing regime detection -- the same ADX/Hurst infrastructure
already retrofitted (and rejected as an entry-timing filter) onto 02u/03u earlier this
session, now applied at the market (BTC) level as a short-side GATE rather than a
per-asset entry filter -- a different application, not assumed to behave the same way.
"""
from __future__ import annotations

import pandas as pd

from modules.regime_classifier import get_regime_state


def btc_confirmed_downtrend(btc_price_data: pd.DataFrame, trend_n: int) -> pd.Series:
    """Returns a boolean Series indexed to btc_price_data."""
    btc_close = btc_price_data["close"]
    btc_ema = btc_close.ewm(span=trend_n, adjust=False).mean()
    below_trend = btc_close < btc_ema
    regime = get_regime_state(btc_price_data, method="adx_hurst")
    return (below_trend & (regime == "trending")).fillna(False)
