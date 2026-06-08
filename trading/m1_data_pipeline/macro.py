import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m1_data_pipeline.database import upsert

_SERIES = ["M2SL", "DTWEXBGS", "FEDFUNDS", "T10YIE"]
_LOOKBACK_YEARS = 10


def fetch_macro() -> dict[str, int]:
    """
    Fetch FRED macro series, resample to weekly, forward-fill gaps, upsert to DB.
    Returns {series_id: rows_inserted}. Silently skips if FRED_API_KEY absent.
    """
    from loguru import logger

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — skipping macro fetch")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed — skipping macro fetch")
        return {}

    fred = Fred(api_key=api_key)
    start = str(date.today() - timedelta(days=365 * _LOOKBACK_YEARS))
    results = {}

    for series_id in _SERIES:
        try:
            raw: pd.Series = fred.get_series(series_id, observation_start=start)
            weekly = (
                raw.resample("W")
                .last()
                .ffill()
                .dropna()
            )
            rows = [
                {"series_id": series_id, "date": str(d.date()), "value": float(v)}
                for d, v in weekly.items()
            ]
            results[series_id] = upsert("macro", rows)
        except Exception as e:
            from loguru import logger as _log
            _log.warning(f"FRED {series_id} failed: {e}")
            results[series_id] = 0

    return results
