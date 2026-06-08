FEE_MAKER        = 0.0002   # 0.02% per side (limit orders)
SLIP_TIER1       = 0.0005   # 0.05% — symbols with avg_volume_30d >= $50M
SLIP_TIER2       = 0.0015   # 0.15% — symbols with avg_volume_30d < $50M
FUNDING_INTERVAL = 8        # hours between funding payments


def total_cost(
    symbol: str,
    avg_volume_30d: float,
    funding_rate_8h: float,
    hold_days: int,
) -> float:
    """
    Total round-trip cost as a fraction of position notional.

    Includes:
    - Maker fee (entry + exit)
    - Slippage (entry + exit), tiered by 30-day average volume
    - Funding payments accumulated during the hold period

    For long positions pass funding_rate_8h as-is (positive rate = cost).
    For short positions negate it (positive rate = income).
    """
    slip    = SLIP_TIER1 if avg_volume_30d >= 50_000_000 else SLIP_TIER2
    fee     = FEE_MAKER * 2
    funding = funding_rate_8h * (24 / FUNDING_INTERVAL) * hold_days
    return fee + (slip * 2) + funding
