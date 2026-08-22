#!/usr/bin/env python3
"""
Shared spot OHLCV refresh for T9B S1 (Donchian) and S3 (ConsecDownDays).

Consolidates three previously-independent, ~30-line inline fetchers (S1's,
S3's, and an orphaned dead copy that used to live in a spot variant of the
S2 engine) that each carried an independent copy of the same bug:

    hist["timestamp"] = (hist["time"].astype(np.int64) // 1_000_000).astype(int)

This assumes pd.to_datetime() always returns datetime64[ns]. Under
pandas>=2.0 the resolution actually returned depends on the environment: on
GitHub Actions' runner it resolved to datetime64[s], so .astype(np.int64)
produced epoch SECONDS instead of nanoseconds. Dividing that by 1_000_000
collapsed every row to ~0, which was then re-parsed downstream as
milliseconds -> 1970-01-01. That silently froze S1 and S3's OHLCV cache
from 2026-06-04 to 2026-08-22 while CI reported success every day (see
the root-cause investigation for the full chain).

Fix: never round-trip datetime -> int64 -> datetime. Timestamps are kept as
tz-aware pandas datetimes end to end; no resolution-dependent cast happens
anywhere in this module.

Data source: yfinance (Yahoo Finance). Binance's REST APIs return HTTP 451
from GitHub Actions' US-based runners; yfinance is geo-unrestricted. Unlike
the futures universe (data.binance.vision daily dumps), there is no public
per-day CDN mirror for spot klines, so yfinance remains the only viable
source here -- but symbols are batched into a single yf.download() call per
refresh instead of one HTTP request per symbol, and missing days are
backfilled per-symbol (last cached date + 1 -> upto_day) rather than always
re-pulling a fixed trailing window, so a transient gap doesn't silently
persist the way the old fixed-10-bar fetch did.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMMITTED_CACHE = ROOT / "data" / "universe" / "ohlcv_1d"

INITIAL_FETCH_DAYS = 1500   # ~4yr, used only when a symbol has no cache at all
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 5.0

_SESSION_OHLCV: Dict[str, pd.DataFrame] = {}


# =============================================================================
# PATHS / SYMBOL HELPERS
# =============================================================================

def safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def cache_path(symbol: str) -> Path:
    return COMMITTED_CACHE / f"{safe_sym(symbol)}_1d.csv"


def _yf_symbol(symbol: str) -> str:
    return f"{symbol.split('/')[0]}-USD"


# =============================================================================
# CACHE READ / WRITE  (no datetime<->int64 round trip anywhere)
# =============================================================================

def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  [WARN] cache read failed for {path.name}: {exc}")
        return pd.DataFrame()
    if "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    out = df[["time", "open", "high", "low", "close", "volume"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _merge(base: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        merged = live
    elif live.empty:
        merged = base
    else:
        merged = pd.concat([base, live], ignore_index=True)
    if merged.empty:
        return merged
    # Dedupe on the calendar day itself (a real datetime value, never a
    # truncated string) -- daily bars should never carry two rows for the
    # same UTC date.
    merged = merged.assign(_day=merged["time"].dt.normalize())
    merged = (
        merged.drop_duplicates(subset="_day", keep="last")
        .drop(columns="_day")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return merged


# =============================================================================
# LIVE FETCH
#
# Primary: data.binance.vision daily kline dumps (public CDN, same source
# already proven reliable for the futures universe in
# .github/scripts/refresh_futures_data.py -- not geo-blocked, not
# rate-limited the way Yahoo's yfinance endpoint is).
#
# Binance's dump format is not static: a 2021 BTCUSDT dump's open_time is a
# 13-digit millisecond value; an equivalent 2026 dump is a 16-digit
# MICROSECOND value (confirmed by direct fetch during this fix). Silently
# assuming one unit here would reproduce exactly the class of bug this
# module exists to eliminate, so the timestamp unit is detected from the
# value's magnitude rather than hardcoded.
#
# Fallback: yfinance, single-symbol calls, only for symbols/days the CDN
# can't serve (e.g. not listed on Binance Spot).
# =============================================================================

VISION_SPOT = "https://data.binance.vision/data/spot/daily/klines"


def _binance_spot_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def _epoch_col_to_datetime(raw_ts: pd.Series) -> pd.Series:
    """Magnitude-based epoch unit detection -- s / ms / us / ns, whichever the
    values actually are. Never assumes a fixed unit (see module docstring)."""
    raw_ts = pd.to_numeric(raw_ts, errors="coerce")
    finite = raw_ts.dropna()
    if finite.empty:
        return pd.to_datetime(raw_ts, utc=True, errors="coerce")
    magnitude = finite.abs().median()
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(raw_ts, unit=unit, utc=True, errors="coerce")


def _fetch_vision_day(bsym: str, day: date) -> Optional[pd.DataFrame]:
    import requests
    import zipfile
    import io

    url = f"{VISION_SPOT}/{bsym}/1d/{bsym}-1d-{day.strftime('%Y-%m-%d')}.zip"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        if raw.empty:
            return None
        # Some dumps carry a header row ("open_time,open,high,..."), some don't.
        first_val = raw.iloc[0, 0]
        if isinstance(first_val, str):
            try:
                float(first_val)
            except ValueError:
                raw = raw.iloc[1:].reset_index(drop=True)
        if raw.empty:
            return None
        row = raw.iloc[0]
        ts = _epoch_col_to_datetime(pd.Series([row[0]])).iloc[0]
        if pd.isna(ts):
            return None
        return pd.DataFrame([{
            "time": ts,
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "volume": float(row[5]),
        }])
    except Exception:
        return None


def _fetch_vision_range(symbol: str, start: date, end_inclusive: date) -> tuple[pd.DataFrame, List[date]]:
    """Fetch each missing day for one symbol from the Binance CDN, in
    parallel (each day is an independent static-file GET -- unlike
    yfinance, this CDN has shown no rate-limiting). Returns (fetched_rows,
    days_still_missing)."""
    import concurrent.futures

    bsym = _binance_spot_symbol(symbol)
    days = []
    d = start
    while d <= end_inclusive:
        days.append(d)
        d += timedelta(days=1)

    results: Dict[date, Optional[pd.DataFrame]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_vision_day, bsym, day): day for day in days}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()

    rows = [results[d] for d in days if results.get(d) is not None]
    missing = [d for d in days if results.get(d) is None]
    if rows:
        return pd.concat(rows, ignore_index=True), missing
    return pd.DataFrame(), missing


def _normalize_yf_download(hist: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single-ticker yfinance frame to time/open/high/low/close/volume."""
    hist = hist.reset_index()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0].lower() if c[1] == "" else c[0].lower() for c in hist.columns]
    else:
        hist.columns = [str(c).lower() for c in hist.columns]

    if "date" in hist.columns:
        hist = hist.rename(columns={"date": "time"})
    elif "datetime" in hist.columns:
        hist = hist.rename(columns={"datetime": "time"})

    hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in hist.columns:
            hist[col] = np.nan
        hist[col] = pd.to_numeric(hist[col], errors="coerce")

    hist = hist.dropna(subset=["time", "open", "high", "low", "close"])
    return hist[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _fetch_yfinance_range(symbol: str, start: date, end_inclusive: date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    yf_sym = _yf_symbol(symbol)
    end_exclusive = end_inclusive + timedelta(days=1)
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hist = yf.download(
                yf_sym,
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if hist is None or hist.empty:
                return pd.DataFrame()
            return _normalize_yf_download(hist)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC * attempt)
    print(f"    [yfinance] {symbol}: gave up after {MAX_RETRIES} attempts -- {last_exc}")
    return pd.DataFrame()


def _fetch_symbol_range(symbol: str, start: date, end_inclusive: date) -> pd.DataFrame:
    """CDN first (day by day); yfinance fallback only for whatever the CDN
    couldn't serve (delisted-from-spot symbols, transient publish gaps)."""
    vision_df, missing_days = _fetch_vision_range(symbol, start, end_inclusive)
    if not missing_days:
        return vision_df

    lo, hi = min(missing_days), max(missing_days)
    print(f"    [vision] {symbol}: {len(missing_days)} day(s) unavailable on CDN "
          f"({lo}..{hi}) -- trying yfinance fallback")
    yf_df = _fetch_yfinance_range(symbol, lo, hi)
    if yf_df.empty:
        return vision_df
    return _merge(vision_df, yf_df)


# =============================================================================
# PUBLIC API
# =============================================================================

def refresh_universe(symbols: List[str], upto_day: Optional[date] = None, label: str = "SPOT") -> int:
    """
    Backfill every symbol's committed cache up to upto_day (default:
    yesterday). Fetches each symbol's missing days from the Binance CDN
    (data.binance.vision), falling back to yfinance only for whatever the
    CDN can't serve. Writes updated caches to disk and prints a staleness
    summary at the end. Returns the number of symbols updated.
    """
    upto_day = upto_day or (date.today() - timedelta(days=1))
    print(f"[DATA] Refreshing {label} OHLCV cache for {len(symbols)} symbols (up to {upto_day})...")

    n_updated = 0
    for sym in symbols:
        base = _read_cache(cache_path(sym))
        if base.empty:
            start = upto_day - timedelta(days=INITIAL_FETCH_DAYS)
        else:
            last_day = base["time"].max().date()
            if last_day >= upto_day:
                continue
            start = last_day + timedelta(days=1)

        live = _fetch_symbol_range(sym, start, upto_day)
        if live.empty:
            print(f"    [WARN] {sym}: no live data returned for gap starting {start}")
            continue
        merged = _merge(base, live)
        if merged.empty:
            continue
        _write_cache(merged, cache_path(sym))
        n_updated += 1

    print(f"[DATA] Updated {n_updated}/{len(symbols)} symbol caches")
    print()
    check_staleness(symbols, upto_day, label=label)
    _SESSION_OHLCV.clear()
    return n_updated


def check_staleness(symbols: List[str], upto_day: date, max_gap_days: int = 3, label: str = "SPOT") -> List[tuple]:
    """
    Loud staleness summary across the whole cache -- the same pattern
    proven in .github/scripts/refresh_futures_data.py's
    check_cache_staleness(). Prints, does not raise; makes a growing gap
    visible in the CI log instead of drifting unnoticed the way the
    original spot cache freeze did for two months.
    """
    stale = []
    worst_sym, worst_days = None, -1
    for sym in symbols:
        path = cache_path(sym)
        if not path.exists():
            stale.append((sym, "missing"))
            continue
        try:
            last_row = pd.read_csv(path, usecols=["time"]).iloc[-1]["time"]
            last_day = pd.to_datetime(last_row, utc=True).date()
        except Exception:
            stale.append((sym, "unreadable"))
            continue
        gap_days = (upto_day - last_day).days
        if gap_days > worst_days:
            worst_sym, worst_days = sym, gap_days
        if gap_days > max_gap_days:
            stale.append((sym, gap_days))
    if stale:
        print(f"  [{label}] [STALENESS] {len(stale)}/{len(symbols)} symbol(s) more than "
              f"{max_gap_days} days behind {upto_day} -- worst: {worst_sym} ({worst_days} days). "
              f"Entry/exit signal detection requires a same-day bar for these; treat as inactive "
              f"until resolved: {', '.join(f'{s}({g})' for s, g in stale[:15])}"
              f"{' ...' if len(stale) > 15 else ''}", flush=True)
    else:
        print(f"  [{label}] [STALENESS] OK -- all {len(symbols)} symbols within "
              f"{max_gap_days} days of {upto_day} (worst: {worst_sym}, {worst_days} days)", flush=True)
    return stale


def load_ohlcv(symbol: str, up_to_date: Optional[date] = None) -> pd.DataFrame:
    """
    Read-only load from the committed cache for use inside an engine's daily
    loop. No network calls -- refresh_universe() must be called first to
    bring the cache current. Cached per symbol for the life of the process.
    """
    if symbol not in _SESSION_OHLCV:
        _SESSION_OHLCV[symbol] = _read_cache(cache_path(symbol))
    df = _SESSION_OHLCV[symbol]
    if up_to_date is not None:
        df = df[df["time"].dt.date <= up_to_date]
    return df.reset_index(drop=True)
