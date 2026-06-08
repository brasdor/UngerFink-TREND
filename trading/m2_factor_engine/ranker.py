import numpy as np
import pandas as pd
from loguru import logger

# Maps raw factor column → z-score column name
_FACTOR_MAP: dict[str, str] = {
    "momentum":    "momentum_z",
    "carry":       "carry_z",
    "size":        "size_z",
    "liquidity":   "liquidity_z",
    "oi_momentum": "oi_z",
}

WINSOR_LOW, WINSOR_HIGH = 0.02, 0.98
MIN_AVG_VOLUME_30D = 10_000_000  # $10M USDT — micro-caps excluded

# Composite weights (must sum to 1.0)
WEIGHTS: dict[str, float] = {
    "momentum_z":  0.35,
    "carry_z":     0.25,
    "size_z":      0.20,
    "liquidity_z": 0.15,
    "oi_z":        0.05,
}


def rank_universe(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise each factor, combine into a composite score, and rank.

    Steps
    -----
    1. Drop non-ASCII symbol names (Binance test/regional tokens).
    2. Drop symbols below MIN_AVG_VOLUME_30D ($10M).
    3. Drop symbols where any raw factor is NaN.
    4. Winsorise each factor at 2nd/98th percentile.
    5. Z-score each factor across the surviving universe.
    6. Compute composite_z as weighted sum.
    7. Assign rank (1 = best) and quintile (5 = top 20 %).

    Returns
    -------
    DataFrame sorted by rank: symbol, momentum_z, carry_z, size_z,
    liquidity_z, oi_z, composite_z, quintile, rank
    """
    df = factors_df.copy()
    raw_cols = list(_FACTOR_MAP.keys())

    # 1. Drop non-ASCII symbol names (Binance test/regional tokens, e.g. Chinese-char pairs)
    non_ascii_mask = ~df["symbol"].apply(str.isascii)
    if non_ascii_mask.any():
        bad = df.loc[non_ascii_mask, "symbol"].tolist()
        logger.info(f"Dropped {len(bad)} non-ASCII symbol(s): {bad}")
        df = df[~non_ascii_mask].reset_index(drop=True)

    # 2. Minimum 30d average volume filter — drop micro-caps before ranking
    before_vol = len(df)
    df = df[df["avg_quote_volume_30d"] >= MIN_AVG_VOLUME_30D].reset_index(drop=True)
    vol_dropped = before_vol - len(df)
    if vol_dropped:
        logger.info(
            f"Volume filter (<${MIN_AVG_VOLUME_30D / 1e6:.0f}M): "
            f"dropped {vol_dropped} symbols ({len(df)} remain)"
        )

    # 3. Handle factors that are universally NaN for this date (e.g. carry/OI when
    #    historical funding data is unavailable).  Instead of dropping all symbols,
    #    set those z-scores to 0 and renormalize weights across available factors.
    available: list[tuple[str, str]] = []  # (raw_col, z_col) pairs with real data
    for raw, z_col in _FACTOR_MAP.items():
        nan_frac = df[raw].isna().mean() if len(df) > 0 else 1.0
        if nan_frac >= 0.99:
            df[z_col] = 0.0
            logger.debug(
                f"Factor '{raw}' is {nan_frac:.0%} NaN for this date"
                f" -- z-score set to 0, excluded from composite"
            )
        else:
            available.append((raw, z_col))

    # Drop symbols with NaN in any factor that IS available (partial data = real gap)
    avail_raw = [r for r, _ in available]
    if avail_raw:
        before = len(df)
        df = df.dropna(subset=avail_raw).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} symbols with NaN factors -- {len(df)} remain")

    if df.empty:
        logger.error("No symbols survived filters -- returning empty DataFrame")
        return df

    # Renormalize weights across available factors so they still sum to 1
    avail_z = [z for _, z in available]
    total_w = sum(WEIGHTS[z] for z in avail_z) if avail_z else 1.0
    adj_w = {z: WEIGHTS[z] / total_w for z in avail_z}

    # 4. Winsorise available factors at 2nd/98th percentile to mute extreme outliers
    for raw, _z in available:
        lo, hi = df[raw].quantile([WINSOR_LOW, WINSOR_HIGH])
        df[raw] = df[raw].clip(lo, hi)

    # 5. Z-score normalise available factors (guard against zero std)
    for raw, z_col in available:
        mu = df[raw].mean()
        sigma = df[raw].std()
        df[z_col] = (df[raw] - mu) / (sigma if sigma > 1e-12 else 1.0)

    # Composite score: available factors at renormalized weight; absent factors = 0
    df["composite_z"] = (
        sum(adj_w[z] * df[z] for z in avail_z) if avail_z else 0.0
    )

    # Rank: 1 = highest composite_z = best
    df["rank"] = df["composite_z"].rank(ascending=False, method="min").astype(int)

    # Quintile: Q5 = top 20 % (pct=1.0), Q1 = bottom 20 % (pct≈0)
    pct = df["composite_z"].rank(pct=True, ascending=True)
    df["quintile"] = np.ceil(pct * 5).clip(1, 5).astype(int)

    out_cols = [
        "symbol", "momentum_z", "carry_z", "size_z", "liquidity_z",
        "oi_z", "composite_z", "quintile", "rank",
    ]
    return df[out_cols].sort_values("rank").reset_index(drop=True)
