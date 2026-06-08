import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MIN_ROWS = 10
_MOM_WINDOW = 14
_OI_WINDOW = 14
_LOOKBACK_DAYS = 60   # 2× the longest factor window — buffer for weekends/gaps
_NAN_WARN_THRESHOLD = 0.20


def _date_to_ms(date_str: str) -> int:
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def _bulk_load(
    table: str,
    time_col: str,
    value_cols: list[str],
    symbols: list[str],
    from_ms: int,
    to_ms: int,
    conn,
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Load rows for all symbols in one SQL query (vectorised I/O)."""
    cols = ", ".join([sym_col, time_col] + value_cols)
    ph = ",".join("?" * len(symbols))
    sql = (
        f"SELECT {cols} FROM {table} "
        f"WHERE {sym_col} IN ({ph}) AND {time_col} >= ? AND {time_col} < ? "
        f"ORDER BY {time_col}"
    )
    return pd.read_sql_query(sql, conn, params=symbols + [from_ms, to_ms])


def compute_all_factors(date: str, universe: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Compute five factor scores for every symbol in universe as of `date`.
    All lookback windows use open_time < date_ms — zero future leakage.

    Returns
    -------
    DataFrame: symbol, momentum, carry, size, liquidity, oi_momentum
    """
    date_ms = _date_to_ms(date)
    from_ms = date_ms - _LOOKBACK_DAYS * 86_400_000
    from_30d = date_ms - 30 * 86_400_000

    symbols = universe["symbol"].tolist()

    # ── Bulk data load ────────────────────────────────────────────────────────
    ohlcv_raw = _bulk_load(
        "ohlcv", "open_time", ["close", "quote_volume"],
        symbols, from_ms, date_ms, conn,
    )
    funding_raw = _bulk_load(
        "funding_rates", "funding_time", ["funding_rate"],
        symbols, from_30d, date_ms, conn,
    )
    oi_raw = _bulk_load(
        "open_interest", "timestamp", ["open_interest"],
        symbols, from_ms, date_ms, conn,
    )

    # ── OHLCV pivots (pivot_table tolerates duplicate keys from data issues) ──
    close = (
        ohlcv_raw.pivot_table(
            index="open_time", columns="symbol", values="close", aggfunc="last"
        ).reindex(columns=symbols)
    )
    qvol = (
        ohlcv_raw.pivot_table(
            index="open_time", columns="symbol", values="quote_volume", aggfunc="last"
        ).reindex(columns=symbols)
    )

    counts = close.count()        # non-NaN rows per symbol (before ffill)
    close_ff = close.ffill()      # forward-fill intra-period gaps
    qvol_ff = qvol.ffill()

    # ── Factor 1: Momentum  close[t-1] / close[t-14] - 1 ─────────────────────
    # Data is already filtered to open_time < date, so iloc[-1] IS close[t-1].
    if len(close_ff) >= _MOM_WINDOW:
        base = close_ff.iloc[-_MOM_WINDOW].where(close_ff.iloc[-_MOM_WINDOW] > 0)
        mom = (close_ff.iloc[-1] / base - 1).reindex(symbols)
    else:
        mom = pd.Series(np.nan, index=symbols)
    mom = mom.where(counts >= _MIN_ROWS)

    # ── Factor 2: Carry  mean funding rate over last 30 days ─────────────────
    carry = (
        funding_raw.groupby("symbol")["funding_rate"]
        .mean()
        .reindex(symbols)
    )

    # ── Factor 3: Size  1 / 30d-avg-quote-volume (smaller coin = higher score)
    qvol_mean = qvol_ff.tail(30).mean()
    size = (1.0 / qvol_mean.where(qvol_mean > 0)).where(counts >= _MIN_ROWS)

    # ── Factor 4: Liquidity  −Amihud = −mean(|ret| / qvol) over 30 days ─────
    # Negative so that more-liquid symbols rank higher.
    daily_ret = close_ff.pct_change(fill_method=None)
    amihud = (daily_ret.abs() / qvol_ff.where(qvol_ff > 0)).tail(30).mean()
    liquidity = (-amihud).where(counts >= _MIN_ROWS)

    # ── Factor 5: OI Momentum  oi[t-1] / oi[t-14] - 1 ───────────────────────
    oi_piv = (
        oi_raw.pivot_table(
            index="timestamp", columns="symbol", values="open_interest", aggfunc="last"
        ).reindex(columns=symbols)
    )
    oi_counts = oi_piv.count()
    oi_ff = oi_piv.ffill()

    if len(oi_ff) >= _OI_WINDOW:
        oi_base = oi_ff.iloc[-_OI_WINDOW].where(oi_ff.iloc[-_OI_WINDOW] > 0)
        oi_mom = (oi_ff.iloc[-1] / oi_base - 1).reindex(symbols)
    else:
        oi_mom = pd.Series(np.nan, index=symbols)
    oi_mom = oi_mom.where(oi_counts >= _MIN_ROWS)

    # ── Assemble ──────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "symbol":               symbols,
        "momentum":             mom.reindex(symbols).values,
        "carry":                carry.reindex(symbols).values,
        "size":                 size.reindex(symbols).values,
        "liquidity":            liquidity.reindex(symbols).values,
        "oi_momentum":          oi_mom.reindex(symbols).values,
        # Pass-through for ranker volume filter — not z-scored
        "avg_quote_volume_30d": qvol_mean.reindex(symbols).values,
    })

    # Warn when a factor has too many NaNs across the universe
    nan_frac = out.drop(columns="symbol").isna().mean()
    for factor, frac in nan_frac.items():
        if frac > _NAN_WARN_THRESHOLD:
            n = int(frac * len(out))
            logger.warning(
                f"Factor '{factor}': {frac:.0%} of universe is NaN ({n}/{len(out)} symbols)"
            )

    return out
