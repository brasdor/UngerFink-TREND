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
        except Exception as exc:
            print(f"  [{interval}] [WARN] {sym}: cache read failed ({exc}) -- skipping",
                  flush=True)
            continue
        try:
            r = requests.get(f"{FAPI}/fapi/v1/klines",
                             params={"symbol": sym, "interval": interval,
                                     "startTime": last_ts + 1, "limit": 1500},
                             timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"  [{interval}] [WARN] {sym}: fapi fetch failed ({exc}) -- "
                  f"will retry next run", flush=True)
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
                          upto_day: date) -> int:
    """
    Download and merge ALL missing {sym}-{interval}-{day}.zip dumps between
    each symbol's own cache last_ts and upto_day (inclusive) -- not just
    upto_day alone.

    Fix for the 2026-07 gap bug (confirmed this session): the old version
    only ever attempted the single most-recent day, so any day whose CDN
    dump wasn't ready at fetch time (transient publish delay, rate limit,
    a probe_fapi() flip mid-week) was silently and PERMANENTLY skipped --
    the next run's target moved on to a new "yesterday" and never revisited
    it. That produced exactly the sawtooth gap pattern found in
    BTCUSDT_1d.csv (07-11 -> 07-15 -> 07-17 -> 07-18 -> 07-19, nothing
    since), which in turn silently stalled bars_held-based exit tracking
    in S2/S3/S8.
    """
    updated = 0
    stale_symbols = []
    for sym in symbols_from(cache_dir, suffix):
        path = cache_dir / f"{sym}{suffix}.csv"
        try:
            df = pd.read_csv(path)
            last_ts = int(df["timestamp"].max())
        except Exception as exc:
            print(f"  [{interval}] [WARN] {sym}: cache read failed ({exc}) -- skipping",
                  flush=True)
            continue

        last_day = pd.Timestamp(last_ts, unit="ms", tz="UTC").date()
        missing_days = []
        d = last_day + timedelta(days=1)
        while d <= upto_day:
            missing_days.append(d)
            d += timedelta(days=1)
        if not missing_days:
            continue

        got_any = False
        still_missing = []
        for day in missing_days:
            day_str = day.strftime("%Y-%m-%d")
            url = f"{VISION}/{sym}/{interval}/{sym}-{interval}-{day_str}.zip"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code != 200:
                    still_missing.append(day_str)
                    continue
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
                if isinstance(raw.iloc[0, 0], str) and not str(raw.iloc[0, 0]).isdigit():
                    raw = raw.iloc[1:].reset_index(drop=True)
            except Exception as exc:
                still_missing.append(f"{day_str}({exc.__class__.__name__})")
                continue
            rows = pd.DataFrame({
                "timestamp": pd.to_numeric(raw[0]),
                "open":  pd.to_numeric(raw[1]), "high": pd.to_numeric(raw[2]),
                "low":   pd.to_numeric(raw[3]), "close": pd.to_numeric(raw[4]),
                "volume": pd.to_numeric(raw[5]),
            })
            fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
            rows["date"] = pd.to_datetime(rows["timestamp"], unit="ms", utc=True).dt.strftime(fmt)
            df = (pd.concat([df, rows], ignore_index=True)
                  .drop_duplicates("timestamp").sort_values("timestamp"))
            got_any = True
            time.sleep(0.02)

        if got_any:
            df.to_csv(path, index=False)
            updated += 1
        if still_missing:
            print(f"  [{interval}] [WARN] {sym}: dump(s) unavailable for "
                  f"{', '.join(still_missing)} (delisted or not yet published) "
                  f"-- will retry next run", flush=True)
            stale_symbols.append((sym, still_missing))

    if stale_symbols:
        max_gap = max(len(m) for _, m in stale_symbols)
        print(f"  [{interval}] [SUMMARY] {len(stale_symbols)} symbol(s) still have "
              f"unresolved gaps this run (worst case {max_gap} missing day(s) for a "
              f"single symbol)", flush=True)
    return updated


def check_cache_staleness(cache_dir: Path, suffix: str, label: str,
                          upto_day: date, max_gap_days: int = 3) -> None:
    """
    Loud, explicit staleness summary across the whole cache -- run at the end
    of every refresh so a growing gap (of the kind that silently stalled
    bars_held in S2/S3/S8) shows up in the CI log instead of drifting
    unnoticed. This does not fix gaps; it makes them visible.
    """
    worst_sym, worst_days = None, -1
    stale_count = 0
    for sym in symbols_from(cache_dir, suffix):
        try:
            df = pd.read_csv(cache_dir / f"{sym}{suffix}.csv")
            last_ts = int(df["timestamp"].max())
        except Exception:
            continue
        last_day = pd.Timestamp(last_ts, unit="ms", tz="UTC").date()
        gap_days = (upto_day - last_day).days
        if gap_days > max_gap_days:
            stale_count += 1
        if gap_days > worst_days:
            worst_sym, worst_days = sym, gap_days
    if stale_count:
        print(f"  [{label}] [STALENESS] {stale_count} symbol(s) more than "
              f"{max_gap_days} days behind {upto_day} -- worst: {worst_sym} "
              f"({worst_days} days stale). Position-aging logic downstream "
              f"depends on this cache; investigate before trusting bars_held.",
              flush=True)
    else:
        print(f"  [{label}] [STALENESS] OK -- all symbols within "
              f"{max_gap_days} days of {upto_day} (worst: {worst_sym}, "
              f"{worst_days} days)", flush=True)


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
        # Backfill ALL missing days per symbol up to yesterday, not just
        # yesterday alone -- see vision_refresh_klines docstring for why the
        # old single-day version silently produced permanent gaps.
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

    # Explicit staleness summary every run, regardless of path taken -- makes
    # a growing gap visible in the CI log instead of drifting unnoticed
    # (this is what let BTCUSDT_1d.csv and the WLD/DUSDT/HIGHUSDT gaps go
    # undetected for weeks before this session's audit).
    print()
    print("[STALENESS CHECK]")
    check_cache_staleness(D1_DIR, "_1d", "1D", YESTERDAY)
    check_cache_staleness(H4_DIR, "_4h", "4H", YESTERDAY)

    return 0


if __name__ == "__main__":
    sys.exit(main())
