"""
M4 Position Sizer
=================
ATR-based position sizing with regime and circuit-breaker multipliers.
"""
import pandas as pd


def atr(prices: pd.Series, period: int = 14) -> float:
    """
    Approximate ATR as the mean absolute daily price change over `period` days.
    Uses close-to-close differences as a proxy for True Range when only a
    single price series is available.
    Returns 0.0 if fewer than 2 valid data points exist.
    """
    changes = prices.dropna().diff().abs().dropna()
    if changes.empty:
        return 0.0
    tail = changes.tail(period)
    return float(tail.mean()) if not tail.empty else 0.0


def base_position_size(
    capital: float,
    risk_pct: float = 0.01,
    atr_value: float = 0.0,
    price: float = 0.0,
    leverage: float = 3.0,
    n_positions: int = 1,
    alloc_frac: float = 0.65,
) -> float:
    """
    Notional position size for one symbol.

    Formula: (capital * risk_pct) / (atr_value / price) * leverage
    Capped at: capital * alloc_frac / n_positions * leverage  (equal-weight ceiling)

    Parameters
    ----------
    capital      : current portfolio equity in USDT
    risk_pct     : fraction of capital to risk per trade (default 1%)
    atr_value    : ATR in price terms (output of atr())
    price        : current price of the symbol
    leverage     : exchange leverage multiplier
    n_positions  : number of symbols in this basket (for equal-weight cap)
    alloc_frac   : basket allocation fraction (0.65 for longs, 0.35 for shorts)
    """
    equal_weight_cap = capital * alloc_frac * leverage / max(n_positions, 1)

    if price <= 0 or atr_value <= 0:
        return equal_weight_cap

    norm_vol = atr_value / price       # daily volatility as fraction of price
    atr_size = (capital * risk_pct) / norm_vol * leverage

    return min(atr_size, equal_weight_cap)


def apply_risk_multiplier(
    base_size: float,
    regime_score: float,
    cb_multiplier: float,
) -> float:
    """final_size = base_size * regime_score * cb_multiplier"""
    return base_size * regime_score * cb_multiplier
