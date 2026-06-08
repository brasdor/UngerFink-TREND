import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LOOKBACK_DAYS
from m1_data_pipeline.database import get_last_date, upsert

_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_LIMIT = 1500
_SLEEP_BETWEEN = 0.05
_SLEEP_429 = 0.5


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    rows = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "endTime": end_ms,
            "limit": _LIMIT,
        }
        while True:
            resp = requests.get(_KLINES_URL, params=params, timeout=10)
            if resp.status_code == 429:
                time.sleep(_SLEEP_429)
                continue
            resp.raise_for_status()
            break

        batch = resp.json()
        if not batch:
            break

        for k in batch:
            rows.append({
                "symbol": symbol,
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
            })

        # Advance past last candle returned
        current = int(batch[-1][0]) + 1
        if len(batch) < _LIMIT:
            break

    return rows


def fetch_ohlcv(symbols: list[str], incremental: bool = True) -> dict[str, int]:
    """
    Fetch daily OHLCV for each symbol and upsert to DB.
    Returns {symbol: rows_inserted}.
    """
    now = datetime.now(timezone.utc)
    yesterday_ms = _ms(now.replace(hour=0, minute=0, second=0, microsecond=0)
                       - timedelta(days=1))
    lookback_start = _ms(now - timedelta(days=LOOKBACK_DAYS))

    results = {}
    for symbol in symbols:
        last = get_last_date("ohlcv", symbol)

        if incremental and last is not None and last >= yesterday_ms:
            results[symbol] = 0
            continue

        start_ms = (last + 1) if (incremental and last) else lookback_start

        rows = _fetch_klines(symbol, start_ms, _ms(now))
        inserted = upsert("ohlcv", rows)
        results[symbol] = inserted
        time.sleep(_SLEEP_BETWEEN)

    return results
