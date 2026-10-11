#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refreshes data/exchange_status_cache.json -- one bulk exchangeInfo call per
venue (spot + futures), not one call per symbol, so this is cheap enough to
run daily wherever it CAN run.

Why this isn't wired into health_check.yml directly: fapi.binance.com
(futures) is geo-blocked from GitHub Actions' US runners -- confirmed
repeatedly elsewhere in this project (funding refresh, kline refresh). The
spot endpoint (api.binance.com) is NOT blocked and already used daily by
check_halted_symbols.py from CI. Until the futures side has a reachable
host (see docs/HEALTH_CONTRACT.md's funding-refresh recommendation -- the
same host solves both), this script is meant to be run locally, the same
way refresh_funding_local.py currently is, and its output committed.

tools/health/run_checks.py's check_exchange_status() only ever READS this
cache file -- never calls either exchangeInfo endpoint itself, so the
daily health run never depends on fapi being reachable.

Usage: python tools/health/refresh_exchange_status_cache.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "exchange_status_cache.json"

SPOT_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def _fetch(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    cache: dict = {"_comment": (
        "symbol -> {status, venue, delivery_date, fetched_at}. Refreshed by "
        "tools/health/refresh_exchange_status_cache.py, one bulk exchangeInfo "
        "call per venue. fapi.binance.com is geo-blocked from CI, so this is "
        "a local/manual-run cache, not a daily CI job -- see its docstring. "
        "tools/health/run_checks.py's check_exchange_status() only reads this."
    ), "fetched_at": now, "symbols": {}}

    try:
        spot = _fetch(SPOT_EXCHANGE_INFO)
        for s in spot.get("symbols", []):
            cache["symbols"][s["symbol"]] = {
                "status": s.get("status"), "venue": "spot", "delivery_date": None,
            }
        print(f"[EXCHANGE-STATUS] spot: {len(spot.get('symbols', []))} symbols")
    except Exception as exc:
        print(f"[EXCHANGE-STATUS] spot fetch failed: {exc}")

    try:
        fut = _fetch(FUTURES_EXCHANGE_INFO)
        for s in fut.get("symbols", []):
            dd = s.get("deliveryDate")
            dd_iso = None
            if dd and dd < 4_000_000_000_000:  # sentinel far-future value used for perpetuals with no delivery
                dd_iso = datetime.fromtimestamp(dd / 1000, tz=timezone.utc).date().isoformat()
            # futures symbol keys share the spot namespace (e.g. "BTCUSDT") --
            # prefix to keep them distinct in case a symbol exists on both
            # venues with different statuses.
            cache["symbols"][f"FUT:{s['symbol']}"] = {
                "status": s.get("status"), "venue": "futures", "delivery_date": dd_iso,
            }
        print(f"[EXCHANGE-STATUS] futures: {len(fut.get('symbols', []))} symbols")
    except Exception as exc:
        print(f"[EXCHANGE-STATUS] futures fetch failed (expected from a US-hosted "
              f"runner -- fapi.binance.com is geo-blocked): {exc}")

    if len(cache["symbols"]) == 0:
        print("[EXCHANGE-STATUS] both fetches failed -- not overwriting the existing cache")
        return 1

    CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    print(f"[EXCHANGE-STATUS] wrote {CACHE_PATH} ({len(cache['symbols'])} symbols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
