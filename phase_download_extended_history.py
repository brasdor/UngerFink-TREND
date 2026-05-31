#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE: EXTENDED HISTORICAL DATA COLLECTION
==========================================

Downloads all available 1D OHLCV data from Binance for the 24 UniverseV2
symbols, going back as far as possible (target: 2017-2018).

Uses paginated ccxt fetch_ohlcv with `since` parameter to walk backwards
from the earliest available bar on Binance.

Output:
  data/ohlcv_extended/{symbol}_1d_extended.csv   one file per symbol
  data/ohlcv_extended/coverage_report.csv        coverage summary

Coverage report columns:
  symbol, earliest_date, latest_date, total_bars,
  bars_pre_2021, bars_2021, bars_2022, bars_2023, bars_2024, bars_2025_plus,
  pre2021_flag     (WARN if bars_pre_2021 < 500)

Usage:
  python phase_download_extended_history.py
  python phase_download_extended_history.py --force-redownload
  python phase_download_extended_history.py --symbol BTC/USDT

Rate limits:
  Binance REST: 1200 request weight/min.  fetch_ohlcv = weight 1.
  We use 0.15s sleep between calls -- well within limits.
  Full run for 24 symbols takes ~2-5 minutes depending on history length.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT        = Path.cwd()
UNIVERSE_CSV = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUTPUT_DIR  = ROOT / "data" / "ohlcv_extended"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_CSV = OUTPUT_DIR / "coverage_report.csv"

TIMEFRAME    = "1d"
BATCH_LIMIT  = 1000          # bars per ccxt call (Binance max is 1000 for 1D)
SLEEP_SEC    = 0.15          # between paginated calls
MAX_RETRIES  = 3

# Earliest possible Binance data (approx launch 2017-07-01)
EARLIEST_TARGET = "2017-07-01"

PRE2021_WARN_THRESHOLD = 500  # flag if fewer than this many bars before 2021


# =============================================================================
# HELPERS
# =============================================================================

def safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def load_universe() -> List[str]:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_CSV}")
    df = pd.read_csv(UNIVERSE_CSV)
    syms = df["symbol"].dropna().tolist()
    print(f"[UNIVERSE] {len(syms)} symbols")
    return syms


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_full_history(
    exchange,
    symbol: str,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Download complete 1D history for a symbol using paginated calls.

    Strategy:
      1. Start at EARLIEST_TARGET timestamp.
      2. Fetch BATCH_LIMIT bars forward.
      3. Continue fetching from last bar until no new data is returned.
      4. Also fetch the most recent BATCH_LIMIT bars to ensure up-to-date.
      5. Merge, deduplicate, sort.

    Returns DataFrame or None on failure.
    """
    safe       = safe_sym(symbol)
    out_path   = OUTPUT_DIR / f"{safe}_1d_extended.csv"

    if out_path.exists() and not force:
        print(f"  {symbol:<20} -- already exists, skipping  (use --force-redownload)")
        return pd.read_csv(out_path)

    print(f"  {symbol:<20} -- downloading...", end="", flush=True)

    start_ts = int(
        pd.Timestamp(EARLIEST_TARGET, tz="UTC").timestamp() * 1000
    )

    all_rows: list = []
    current_since = start_ts
    consecutive_empty = 0

    # Forward walk from earliest target
    while True:
        rows = _fetch_with_retry(
            exchange, symbol, TIMEFRAME,
            since=current_since, limit=BATCH_LIMIT,
        )
        if not rows:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            time.sleep(0.5)
            continue

        consecutive_empty = 0
        all_rows.extend(rows)

        last_ts = rows[-1][0]
        if len(rows) < BATCH_LIMIT:
            # Reached the end of available history
            break

        # Advance by 1 bar to avoid overlap
        current_since = last_ts + 86_400_000   # +1 day in ms
        time.sleep(SLEEP_SEC)

    # Always fetch the most recent BATCH_LIMIT bars too (ensures no gap at end)
    recent = _fetch_with_retry(exchange, symbol, TIMEFRAME, since=None, limit=BATCH_LIMIT)
    if recent:
        all_rows.extend(recent)

    if not all_rows:
        print(f" [FAILED] no data returned")
        return None

    # Build DataFrame, deduplicate, sort
    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Convert timestamp to date string for readability (keep timestamp too)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date.astype(str)

    df.to_csv(out_path, index=False)
    n = len(df)
    earliest = df["date"].iloc[0] if n > 0 else "n/a"
    latest   = df["date"].iloc[-1] if n > 0 else "n/a"
    print(f" {n:>5} bars  {earliest} to {latest}")

    return df


def _fetch_with_retry(
    exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int],
    limit: int,
) -> list:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs: dict = {"limit": limit}
            if since is not None:
                kwargs["since"] = since
            result = exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
            time.sleep(SLEEP_SEC)
            return result or []
        except Exception as exc:
            wait = 1.0 * attempt
            print(f"\n    [WARN] attempt {attempt}/{MAX_RETRIES}: {exc}  (retry in {wait:.1f}s)")
            time.sleep(wait)
    return []


# =============================================================================
# COVERAGE REPORT
# =============================================================================

def build_coverage_report(symbols: List[str]) -> pd.DataFrame:
    """Read each downloaded file and compute coverage statistics."""
    rows = []

    for sym in symbols:
        safe     = safe_sym(sym)
        path     = OUTPUT_DIR / f"{safe}_1d_extended.csv"

        if not path.exists():
            rows.append({
                "symbol":        sym,
                "status":        "MISSING",
                "earliest_date": None,
                "latest_date":   None,
                "total_bars":    0,
                "bars_pre_2021": 0,
                "bars_2021":     0,
                "bars_2022":     0,
                "bars_2023":     0,
                "bars_2024":     0,
                "bars_2025_plus": 0,
                "pre2021_flag":  "NO_DATA",
            })
            continue

        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_numeric(df["timestamp"])
            df["year"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.year
            df["date_str"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date.astype(str)

            total   = len(df)
            pre2021 = int((df["year"] < 2021).sum())
            y2021   = int((df["year"] == 2021).sum())
            y2022   = int((df["year"] == 2022).sum())
            y2023   = int((df["year"] == 2023).sum())
            y2024   = int((df["year"] == 2024).sum())
            y2025p  = int((df["year"] >= 2025).sum())
            earliest = df["date_str"].iloc[0]  if total > 0 else None
            latest   = df["date_str"].iloc[-1] if total > 0 else None
            flag     = "WARN: <500 bars pre-2021" if pre2021 < PRE2021_WARN_THRESHOLD else "OK"

            rows.append({
                "symbol":         sym,
                "status":         "OK",
                "earliest_date":  earliest,
                "latest_date":    latest,
                "total_bars":     total,
                "bars_pre_2021":  pre2021,
                "bars_2021":      y2021,
                "bars_2022":      y2022,
                "bars_2023":      y2023,
                "bars_2024":      y2024,
                "bars_2025_plus": y2025p,
                "pre2021_flag":   flag,
            })
        except Exception as exc:
            rows.append({
                "symbol":        sym,
                "status":        f"ERROR: {exc}",
                "earliest_date": None,
                "latest_date":   None,
                "total_bars":    0,
                "bars_pre_2021": 0,
                "bars_2021":     0,
                "bars_2022":     0,
                "bars_2023":     0,
                "bars_2024":     0,
                "bars_2025_plus": 0,
                "pre2021_flag":  "ERROR",
            })

    return pd.DataFrame(rows)


def print_coverage_table(cov: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("COVERAGE REPORT -- 1D EXTENDED HISTORY")
    print("=" * 80)
    print(
        f"  {'Symbol':<20}  {'Earliest':<12}  {'Latest':<12}  "
        f"{'Total':>6}  {'Pre-2021':>9}  {'Flag'}"
    )
    print("  " + "-" * 76)
    for _, r in cov.iterrows():
        flag = r["pre2021_flag"]
        flag_mark = "  <-- WARN" if "WARN" in str(flag) else ""
        print(
            f"  {r['symbol']:<20}  "
            f"{str(r['earliest_date']):<12}  "
            f"{str(r['latest_date']):<12}  "
            f"{r['total_bars']:>6}  "
            f"{r['bars_pre_2021']:>9}{flag_mark}"
        )
    print()

    warn_syms = cov[cov["pre2021_flag"].str.contains("WARN", na=False)]["symbol"].tolist()
    if warn_syms:
        print(f"  [WARN] {len(warn_syms)} symbols with <{PRE2021_WARN_THRESHOLD} bars pre-2021:")
        for s in warn_syms:
            print(f"    - {s}")
        print()
        print("  These symbols were likely listed after 2020.")
        print("  Short-side research will need additional data collection for bear cycles.")
    else:
        print(f"  All {len(cov)} symbols have >= {PRE2021_WARN_THRESHOLD} bars pre-2021.")

    print()
    total_ok = int((cov["status"] == "OK").sum())
    print(f"  Downloaded: {total_ok}/{len(cov)} symbols OK")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Report:     {COVERAGE_CSV}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download extended 1D OHLCV history for UniverseV2 symbols",
    )
    p.add_argument(
        "--force-redownload", action="store_true",
        help="Re-download even if file already exists",
    )
    p.add_argument(
        "--symbol", type=str, default=None, metavar="SYM",
        help="Download a single symbol only (e.g. BTC/USDT)",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip download; just rebuild and print the coverage report",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("EXTENDED HISTORICAL DATA COLLECTION -- 1D OHLCV")
    print("=" * 70)
    print(f"Target: {EARLIEST_TARGET} to today")
    print(f"Output: {OUTPUT_DIR}")
    print()

    symbols = load_universe()

    if args.symbol:
        if args.symbol not in symbols:
            print(f"[WARN] {args.symbol} not in universe CSV; downloading anyway")
        symbols = [args.symbol]

    if not args.report_only:
        try:
            import ccxt  # type: ignore
        except ImportError:
            print("[ERROR] ccxt not installed.  Run:  pip install ccxt")
            return 1

        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        exchange.load_markets()
        print(f"[EXCHANGE] Connected to Binance Spot")
        print(f"[DOWNLOAD] {len(symbols)} symbols  batch={BATCH_LIMIT} bars  sleep={SLEEP_SEC}s")
        print()

        for sym in symbols:
            download_full_history(exchange, sym, force=args.force_redownload)

    # Build and save coverage report
    print()
    print("[REPORT] Building coverage report...")
    cov = build_coverage_report(symbols)
    cov.to_csv(COVERAGE_CSV, index=False)

    print_coverage_table(cov)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
