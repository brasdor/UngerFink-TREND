import os
import sys
import time
from datetime import date

import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EXCLUDED_SUFFIXES, EXCLUDED_TOKENS, MIN_VOLUME_USDT, UNIVERSE_SIZE
from m1_data_pipeline.database import upsert

_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"


def _is_excluded(base: str, symbol: str) -> bool:
    if base in EXCLUDED_TOKENS:
        return True
    for suffix in EXCLUDED_SUFFIXES:
        if symbol.endswith(suffix + "USDT"):
            return True
    return False


def fetch_universe(save: bool = True) -> pd.DataFrame:
    """
    Fetch USDT-margined perpetual universe from Binance, filter by volume,
    rank top UNIVERSE_SIZE, optionally persist snapshot, return DataFrame.
    """
    info_resp = requests.get(_EXCHANGE_INFO_URL, timeout=10)
    info_resp.raise_for_status()
    symbols_raw = info_resp.json()["symbols"]

    contracts = {}
    for s in symbols_raw:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and not _is_excluded(s["baseAsset"], s["symbol"])
        ):
            # Extract filter values
            min_qty = tick_size = None
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    min_qty = float(f["minQty"])
                if f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])

            contracts[s["symbol"]] = {
                "symbol": s["symbol"],
                "base_asset": s["baseAsset"],
                "price_precision": s["pricePrecision"],
                "qty_precision": s["quantityPrecision"],
                "min_qty": min_qty,
                "tick_size": tick_size,
            }

    ticker_resp = requests.get(_TICKER_URL, timeout=10)
    ticker_resp.raise_for_status()
    tickers = {t["symbol"]: float(t["quoteVolume"]) for t in ticker_resp.json()}

    rows = []
    for sym, meta in contracts.items():
        vol = tickers.get(sym, 0.0)
        if vol >= MIN_VOLUME_USDT:
            rows.append({**meta, "quote_volume": vol})

    df = (
        pd.DataFrame(rows)
        .sort_values("quote_volume", ascending=False)
        .head(UNIVERSE_SIZE)
        .reset_index(drop=True)
    )

    if save:
        today = str(date.today())
        db_rows = df.assign(date=today).to_dict("records")
        upsert("universe_history", db_rows)

    return df[["symbol", "base_asset", "quote_volume", "price_precision",
               "qty_precision", "min_qty", "tick_size"]]
