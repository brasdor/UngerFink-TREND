#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch 2H OHLCV data from Binance Spot for the MR universe
and save to data/raw_trend_t1/{SYMBOL}_2h.csv

Targets ~4 years of history (same depth as 1D data).
Paginates to get up to 17520 bars (4yr × 365d × 12 bars/day).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

try:
    import ccxt
except ImportError:
    raise SystemExit("Install ccxt: pip install ccxt")

RAW_DIR = Path(__file__).parent / "data" / "raw_trend_t1"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS_UNDERSCORE = [
    "AAVE_USDT", "ADA_USDT",  "ALT_USDT",  "APT_USDT",  "ARB_USDT",
    "ARKM_USDT", "ASTER_USDT","ATOM_USDT", "AVAX_USDT", "BCH_USDT",
    "BNB_USDT",  "BTC_USDT",  "CHZ_USDT",  "DASH_USDT", "DOGE_USDT",
    "DOT_USDT",  "EIGEN_USDT","ENA_USDT",  "ETH_USDT",  "FET_USDT",
    "FIL_USDT",  "GRT_USDT",  "HBAR_USDT", "ICP_USDT",  "INJ_USDT",
    "JTO_USDT",  "LINK_USDT", "LPT_USDT",  "LTC_USDT",  "MORPHO_USDT",
    "NEAR_USDT", "NIL_USDT",  "ONDO_USDT", "ORDI_USDT", "PENDLE_USDT",
    "PENGU_USDT","PEPE_USDT", "RENDER_USDT","SAGA_USDT","SEI_USDT",
    "SOL_USDT",  "SPK_USDT",  "SUI_USDT",  "TAO_USDT",  "TIA_USDT",
    "TON_USDT",  "TRX_USDT",  "UNI_USDT",  "WLD_USDT",  "XRP_USDT",
    "ZEC_USDT",  "ZEN_USDT",
]

TF          = "2h"
TARGET_BARS = 17520   # ~4 years at 2h
BATCH_SIZE  = 1000    # Binance max per request
MIN_BARS    = 200

# Fetch from this date forward (covers 2022 bear market — critical for crypto MR)
# 2021-01-01 UTC in milliseconds
SINCE_MS = 1609459200000


def underscore_to_ccxt(sym: str) -> str:
    """BTC_USDT -> BTC/USDT"""
    parts = sym.split("_")
    return f"{parts[0]}/{parts[1]}"


def fetch_symbol(exchange, ccxt_sym: str, out_path: Path) -> int:
    """Fetch bars from SINCE_MS forward, paginating. Returns bar count or -1 on fail."""
    all_rows: list = []
    since_ms: int = SINCE_MS

    # Paginate oldest → newest until we run out of data or hit TARGET_BARS
    for _ in range((TARGET_BARS // BATCH_SIZE) + 5):
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    ccxt_sym, timeframe=TF,
                    limit=BATCH_SIZE,
                    since=since_ms,
                )
                time.sleep(exchange.rateLimit / 1000.0)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [ERROR] {ccxt_sym}: {e}", flush=True)
                    return -1
                time.sleep(1.0 * (attempt + 1))

        if not batch:
            break

        all_rows.extend(batch)

        # Advance since_ms past the last returned candle
        since_ms = batch[-1][0] + 1

        if len(batch) < BATCH_SIZE:
            break  # reached current time — no more data

        if len(all_rows) >= TARGET_BARS:
            break

    if len(all_rows) < MIN_BARS:
        return 0

    # Keep most recent TARGET_BARS, drop last (forming) candle
    all_rows = all_rows[-TARGET_BARS:]
    all_rows = all_rows[:-1]

    df = pd.DataFrame(all_rows,
                      columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp","open","high","low","close","volume"]]
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    print("=" * 60, flush=True)
    print(f"  Fetching {TF} OHLCV from Binance Spot", flush=True)
    print(f"  Target : {TARGET_BARS} bars per symbol (~4yr)", flush=True)
    print(f"  Symbols: {len(SYMBOLS_UNDERSCORE)}", flush=True)
    print("=" * 60, flush=True)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })

    ok = 0
    skip = 0
    fail = 0

    for idx, sym_us in enumerate(SYMBOLS_UNDERSCORE, 1):
        out_path  = RAW_DIR / f"{sym_us}_{TF}.csv"
        ccxt_sym  = underscore_to_ccxt(sym_us)

        # Skip if already fetched
        if out_path.exists():
            try:
                existing = pd.read_csv(out_path)
                if len(existing) >= MIN_BARS:
                    print(f"  [{idx:2d}/{len(SYMBOLS_UNDERSCORE)}] {sym_us:20s} SKIP (already {len(existing)} bars)", flush=True)
                    skip += 1
                    continue
            except Exception:
                pass

        print(f"  [{idx:2d}/{len(SYMBOLS_UNDERSCORE)}] {sym_us:20s} fetching...", end=" ", flush=True)
        n = fetch_symbol(exchange, ccxt_sym, out_path)

        if n > 0:
            print(f"{n} bars saved", flush=True)
            ok += 1
        elif n == 0:
            print(f"TOO FEW BARS -- skipped", flush=True)
            fail += 1
        else:
            print(f"FETCH ERROR", flush=True)
            fail += 1

    print(f"\n  Done: {ok} fetched, {skip} skipped, {fail} failed", flush=True)
    print(f"  Files in {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
