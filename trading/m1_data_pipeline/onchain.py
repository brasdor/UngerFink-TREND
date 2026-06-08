import os
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m1_data_pipeline.database import upsert

_BASE_URL = "https://api.glassnode.com/v1/metrics"
_METRICS = {
    "btc_price": ("market", "price_usd_close"),
    "btc_nupl": ("indicators", "nupl"),
    "btc_sopr": ("indicators", "sopr"),
}
_LOOKBACK_DAYS = 730


def fetch_onchain() -> dict[str, int]:
    """
    Fetch BTC on-chain metrics from Glassnode free tier.
    Silently skips if GLASSNODE_API_KEY absent.
    Returns {metric: rows_inserted}.
    """
    from loguru import logger

    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        logger.warning("GLASSNODE_API_KEY not set — skipping on-chain fetch")
        return {}

    since = int((datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)).timestamp())
    results = {}

    for metric_key, (category, endpoint) in _METRICS.items():
        url = f"{_BASE_URL}/{category}/{endpoint}"
        params = {
            "a": "BTC",
            "api_key": api_key,
            "s": since,
            "i": "24h",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = [
                {"metric": metric_key, "timestamp": int(pt["t"]) * 1000, "value": float(pt["v"])}
                for pt in data
                if pt.get("v") is not None
            ]
            results[metric_key] = upsert("onchain", rows)
        except Exception as e:
            logger.warning(f"Glassnode {metric_key} failed: {e}")
            results[metric_key] = 0

    return results
