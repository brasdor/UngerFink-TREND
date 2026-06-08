import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m1_data_pipeline.database import get_last_date, upsert

_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_LS_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
_SLEEP = 0.05
_SLEEP_429 = 0.5


def _get(url: str, params: dict) -> list:
    while True:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(_SLEEP_429)
            continue
        resp.raise_for_status()
        return resp.json()


def _fetch_funding(symbol: str, incremental: bool) -> int:
    last = get_last_date("funding_rates", symbol, date_col="funding_time")
    params = {"symbol": symbol, "limit": 1000}
    if incremental and last:
        params["startTime"] = last + 1

    rows = []
    while True:
        batch = _get(_FUNDING_URL, params)
        if not batch:
            break
        for r in batch:
            rows.append({
                "symbol": symbol,
                "funding_time": int(r["fundingTime"]),
                "funding_rate": float(r["fundingRate"]),
            })
        if len(batch) < 1000:
            break
        params["startTime"] = int(batch[-1]["fundingTime"]) + 1

    return upsert("funding_rates", rows)


def _fetch_oi(symbol: str, incremental: bool) -> int:
    last = get_last_date("open_interest", symbol, date_col="timestamp")
    params = {"symbol": symbol, "period": "1d", "limit": 500}
    if incremental and last:
        params["startTime"] = last + 1

    batch = _get(_OI_URL, params)
    rows = [
        {
            "symbol": symbol,
            "timestamp": int(r["timestamp"]),
            "open_interest": float(r["sumOpenInterest"]),
        }
        for r in batch
    ]
    return upsert("open_interest", rows)


def _fetch_ls(symbol: str, incremental: bool) -> int:
    last = get_last_date("ls_ratio", symbol, date_col="timestamp")
    params = {"symbol": symbol, "period": "1d", "limit": 500}
    if incremental and last:
        params["startTime"] = last + 1

    batch = _get(_LS_URL, params)
    rows = [
        {
            "symbol": symbol,
            "timestamp": int(r["timestamp"]),
            "long_ratio": float(r["longAccount"]),
            "short_ratio": float(r["shortAccount"]),
        }
        for r in batch
    ]
    return upsert("ls_ratio", rows)


def fetch_funding(symbols: list[str], incremental: bool = True) -> dict[str, dict]:
    """
    Fetch funding rates, open interest, and L/S ratio for all symbols.
    Returns {symbol: {funding: n, oi: n, ls: n}}.
    """
    results = {}
    for symbol in symbols:
        try:
            f = _fetch_funding(symbol, incremental)
            oi = _fetch_oi(symbol, incremental)
            ls = _fetch_ls(symbol, incremental)
            results[symbol] = {"funding": f, "oi": oi, "ls": ls}
        except Exception as e:
            results[symbol] = {"error": str(e)}
        time.sleep(_SLEEP)
    return results
