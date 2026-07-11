#!/usr/bin/env python3
"""
Daily futures data refresh for T9B CI — runs BEFORE regime_daily.py and engines.

Keeps three committed caches current:
  data/futures_universe/funding_rates/   (regime funding axis + S6/S7/S8 gates)
  data/futures_universe/ohlcv_1d/        (regime trend/vol axes + S8 signals)
  data/futures_universe/ohlcv_4h/        (S6/S7 signals)

NETWORK REALITY: GitHub Actions runners are US-based and fapi.binance.com
geo-blocks US IPs (HTTP 451) — verified 2026-07-11: engine live fetches
silently returned nothing on CI. Strategy:
  1. Probe fapi.binance.com once, loudly.
  2. If reachable  -> incremental fetch via fapi (klines + funding).
  3. If blocked    -> fall back to data.binance.vision daily kline dumps
                      (public CDN, not geo-blocked, ~1 day lag = exactly what
                      the engines need since they run for "yesterday").
                      Funding rates are NOT published daily on binance.vision
                      (monthly only) -> loud warning; funding then relies on
                      periodic local refresh pushed from a non-US machine.

Exit code is always 0 (continue-on-error safety) but failures are printed
loudly so they are visible in the step log.
"""
import io
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
FUND_DIR = ROOT / "data" / "futures_universe" / "funding_rates"
D1_DIR   = ROOT / "data" / "futures_universe" / "ohlcv_1d"
H4_DIR   = ROOT / "data" / "futures_universe" / "ohlcv_4h"

FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision/data/futures/um/daily/klines"

YESTERDAY = date.today() - timedelta(days=1)


def probe_fapi() -> bool:
    try:
        r = requests.get(f"{FAPI}/fapi/v1/ping", timeout=10)
        print(f"[PROBE] fapi.binance.com -> HTTP {r.status_code}")
        return r.status_code == 200
    except Exception as exc:
        print(f"[PROBE] fapi.binance.com unreachable: {exc}")
        return False


def symbols_from(dir_: Path, suffix: str) -> list:
    return sorted(f.stem.replace(suffix, "") for f in dir_.glob(f"*{suffix}.csv"))


# ---------------------------------------------------------------------------
# Path A — direct fapi (works from non-US IPs / if Binance unblocks runners)
# ---------------------------------------------------------------------------

def fapi_refresh_klines(interval: str, cache_dir: Path, suffix: str) -> int:
    updated = 0
    for sym in symbols_from(cache_dir, suffix):
        path = cache_dir / f"{sym}{suffix}.csv"
        try:
            df = pd.read_csv(path)
            last_ts = int(df["timestamp"].max())
        except Exception:
            continue
        try:
            r = requests.get(f"{FAPI}/fapi/v1/klines",
                             params={"symbol": sym, "interval": interval,
                                     "startTime": last_ts + 1, "limit": 1500},
                             timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue
        if not data:
            continue
        rows = [{"timestamp": d[0], "open": float(d[1]), "high": float(d[2]),
                 "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])}
                for d in data]
        nd = pd.DataFrame(rows)
        fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
        nd["date"] = pd.to_datetime(nd["timestamp"], unit="ms", utc=True).dt.strftime(fmt)
        comb = (pd.concat([df, nd], ignore_index=True)
                .drop_duplicates("timestamp").sort_values("timestamp"))
        comb.to_csv(path, index=False)
        updated += 1
        time.sleep(0.04)
    return updated


def fapi_refresh_funding() -> int:
    updated = 0
    for sym in symbols_from(FUND_DIR, "_funding"):
        path = FUND_DIR / f"{sym}_funding.csv"
        try:
            df = pd.read_csv(path)
            last_ts = int(df["funding_time"].max())
        except Exception:
            continue
        try:
            r = requests.get(f"{FAPI}/fapi/v1/fundingRate",
                             params={"symbol": sym, "startTime": last_ts + 1,
                                     "limit": 1000},
                             timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue
        if not data:
            continue
        nd = pd.DataFrame([{"funding_time": int(d["fundingTime"]),
                            "funding_rate": float(d["fundingRate"])} for d in data])
        comb = (pd.concat([df, nd], ignore_index=True)
                .drop_duplicates("funding_time").sort_values("funding_time"))
        comb.to_csv(path, index=False)
        updated += 1
        time.sleep(0.04)
    return updated


# ---------------------------------------------------------------------------
# Path B — data.binance.vision daily kline dumps (public CDN, no geo-block)
# ---------------------------------------------------------------------------

def vision_refresh_klines(interval: str, cache_dir: Path, suffix: str,
                          day: date) -> int:
    """Download {sym}-{interval}-{day}.zip for each symbol and merge."""
    updated = 0
    failed = 0
    day_str = day.strftime("%Y-%m-%d")
    for sym in symbols_from(cache_dir, suffix):
        path = cache_dir / f"{sym}{suffix}.csv"
        try:
            df = pd.read_csv(path)
            last_ts = int(df["timestamp"].max())
        except Exception:
            continue
        # Skip if cache already has bars for this day
        day_start_ms = int(pd.Timestamp(day_str, tz="UTC").value // 1_000_000)
        if last_ts >= day_start_ms + (0 if interval == "1d" else 20 * 3600 * 1000):
            continue
        url = f"{VISION}/{sym}/{interval}/{sym}-{interval}-{day_str}.zip"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                failed += 1
                continue
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
            # vision CSVs may or may not carry a header row
            if isinstance(raw.iloc[0, 0], str) and not str(raw.iloc[0, 0]).isdigit():
                raw = raw.iloc[1:].reset_index(drop=True)
        except Exception:
            failed += 1
            continue
        rows = pd.DataFrame({
            "timestamp": pd.to_numeric(raw[0]),
            "open":  pd.to_numeric(raw[1]), "high": pd.to_numeric(raw[2]),
            "low":   pd.to_numeric(raw[3]), "close": pd.to_numeric(raw[4]),
            "volume": pd.to_numeric(raw[5]),
        })
        fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
        rows["date"] = pd.to_datetime(rows["timestamp"], unit="ms", utc=True).dt.strftime(fmt)
        comb = (pd.concat([df, rows], ignore_index=True)
                .drop_duplicates("timestamp").sort_values("timestamp"))
        comb.to_csv(path, index=False)
        updated += 1
        time.sleep(0.02)
    if failed:
        print(f"  [{interval}] vision: {failed} symbols had no {day_str} dump "
              f"(delisted or not yet published)")
    return updated


def main() -> int:
    print(f"[REFRESH] futures data refresh for {YESTERDAY}")

    if probe_fapi():
        print("[MODE] fapi.binance.com reachable — direct incremental fetch")
        n1 = fapi_refresh_funding()
        print(f"[FUNDING] updated {n1} symbols via fapi")
        n2 = fapi_refresh_klines("1d", D1_DIR, "_1d")
        print(f"[1D] updated {n2} symbols via fapi")
        n3 = fapi_refresh_klines("4h", H4_DIR, "_4h")
        print(f"[4H] updated {n3} symbols via fapi")
    else:
        print("[MODE] fapi blocked — falling back to data.binance.vision dumps")
        n2 = vision_refresh_klines("1d", D1_DIR, "_1d", YESTERDAY)
        print(f"[1D] updated {n2} symbols via binance.vision")
        n3 = vision_refresh_klines("4h", H4_DIR, "_4h", YESTERDAY)
        print(f"[4H] updated {n3} symbols via binance.vision")
        print("[FUNDING] WARNING: funding rates cannot be refreshed from CI "
              "(fapi geo-blocked; binance.vision publishes funding monthly only).")
        print("[FUNDING] Funding CSVs will stale without a periodic local "
              "refresh pushed from a non-US machine. Check funding freshness:")
        try:
            btc = pd.read_csv(FUND_DIR / "BTCUSDT_funding.csv")
            last = pd.to_datetime(int(btc["funding_time"].max()), unit="ms")
            age_days = (pd.Timestamp.utcnow().tz_localize(None) - last).days
            print(f"[FUNDING] BTCUSDT funding last row: {last} ({age_days} days old)")
            if age_days > 7:
                print(f"[FUNDING] *** STALE > 7 DAYS — run local funding refresh NOW ***")
        except Exception as exc:
            print(f"[FUNDING] freshness check failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
