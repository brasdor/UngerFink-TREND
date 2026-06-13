#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Step A — Donchian Universe Expansion

Scans ALL available Binance Spot USDT pairs.
Downloads 1D OHLCV (target 2200 bars ≈ 6 years) for each.
Filters to:
  - ≥1500 1D bars
  - Excludes stablecoins: USDC, BUSD, TUSD, DAI, FDUSD
  - Excludes wrapped tokens: WBTC, WETH
  - Excludes leveraged tokens: base currency contains UP, DOWN, BULL, BEAR,
    or ends with leveraged ETF suffix (3L, 3S, 5L, 5S, 2L, 2S)

Output: data/universe/expanded_symbols.csv
        data/universe/ohlcv_1d/ (cached downloads)

STOP — prints expanded symbol list for human review before Step B.
"""

from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError:
    sys.exit("ccxt not installed: pip install ccxt")


# =============================================================================
# CONFIG
# =============================================================================

ROOT          = Path(__file__).resolve().parents[1]
EXISTING_CACHE = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"
NEW_CACHE      = ROOT / "data" / "universe" / "ohlcv_1d"
OUT_DIR        = ROOT / "data" / "universe"
NEW_CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BARS = 2200          # download target (6+ years)
MIN_BARS    = 1500          # inclusion threshold
BAR_MS      = 86_400_000   # 1 day in ms

STABLECOINS = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDS", "USDP",
               "USDN", "USDJ", "GUSD", "HUSD", "MUSD", "SUSD", "FRAX"}
WRAPPED     = {"WBTC", "WETH", "WBNB"}
LEV_SUBSTRINGS = ["UP", "DOWN", "BULL", "BEAR"]
LEV_SUFFIXES   = ["3L", "3S", "5L", "5S", "2L", "2S", "1L", "1S",
                  "10L", "10S", "4L", "4S", "3X", "5X"]


# =============================================================================
# EXCLUSION LOGIC
# =============================================================================

def exclusion_reason(symbol: str) -> Optional[str]:
    """Return exclusion reason string, or None if the symbol should be included."""
    base = symbol.split("/")[0]

    if base in STABLECOINS:
        return "stablecoin"

    if base in WRAPPED:
        return "wrapped"

    # Leveraged token by suffix
    for suf in LEV_SUFFIXES:
        if base.endswith(suf):
            return f"leveraged_suffix:{suf}"

    # Leveraged token by substring
    for sub in LEV_SUBSTRINGS:
        if sub in base:
            return f"leveraged_substr:{sub}"

    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        col = "timestamp" if "timestamp" in df.columns else "time"
        if pd.api.types.is_numeric_dtype(df[col]):
            df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    except Exception:
        return None


def _download(exchange, symbol: str) -> Optional[pd.DataFrame]:
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1_000)
    since  = now_ms - TARGET_BARS * BAR_MS
    limit  = 1_000
    rows   = []

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=limit)
        except Exception as exc:
            print(f"    [WARN] {symbol}: {exc}")
            time_mod.sleep(1.0)
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < limit or last_ts >= now_ms:
            break
        since = last_ts + 1
        time_mod.sleep(0.05)

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def get_ohlcv_1d(symbol: str, exchange) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")

    # Try existing cache first (any research dir)
    for cache_dir in [EXISTING_CACHE, NEW_CACHE]:
        for pattern in [f"{clean}_1d.csv", f"{clean}.csv"]:
            p = cache_dir / pattern
            if p.exists():
                df = _load_csv(p)
                if df is not None and len(df) >= MIN_BARS:
                    return df      # good enough, skip re-download
                # Too short — fall through to download

    # Download fresh
    df = _download(exchange, symbol)
    if df is not None and len(df) > 10:
        df.to_csv(NEW_CACHE / f"{clean}_1d.csv", index=False)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("Phase T17 Step A — Donchian Universe Expansion")
    print("=" * 70)

    exchange = ccxt.binance({"enableRateLimit": True})

    print("Loading Binance market list...", end=" ", flush=True)
    markets = exchange.load_markets()
    print("done.")

    all_usdt = sorted([
        s for s, m in markets.items()
        if s.endswith("/USDT") and m.get("spot", False) and m.get("active", False)
    ])
    print(f"Total active USDT spot pairs on Binance: {len(all_usdt)}")
    print()

    # Pre-filter: apply exclusion logic (no download needed)
    pre_excluded = {s: exclusion_reason(s) for s in all_usdt if exclusion_reason(s)}
    candidates   = [s for s in all_usdt if exclusion_reason(s) is None]
    print(f"Pre-excluded (stablecoin/wrapped/leveraged): {len(pre_excluded)}")
    print(f"Candidates for download: {len(candidates)}")
    print()

    # Download and count bars
    rows = []
    for i, sym in enumerate(candidates, 1):
        sym_safe = sym.encode("ascii", "replace").decode("ascii")
        print(f"  [{i:3d}/{len(candidates)}] {sym_safe:20s}", end=" ", flush=True)
        df = get_ohlcv_1d(sym, exchange)
        bars = len(df) if df is not None else 0
        first_date = str(df["time"].min().date()) if df is not None and bars > 0 else ""
        last_date  = str(df["time"].max().date()) if df is not None and bars > 0 else ""
        excl = "insufficient_bars" if bars < MIN_BARS else None
        included = excl is None
        print(f"bars={bars:4d}  {'INCLUDED' if included else f'SKIP({excl})'}")
        rows.append({
            "symbol":      sym,
            "base":        sym.split("/")[0],
            "bars":        bars,
            "first_date":  first_date,
            "last_date":   last_date,
            "excluded":    excl,
            "included":    included,
        })

    # Also record pre-excluded
    for sym, reason in pre_excluded.items():
        rows.append({
            "symbol":     sym,
            "base":       sym.split("/")[0],
            "bars":       0,
            "first_date": "",
            "last_date":  "",
            "excluded":   reason,
            "included":   False,
        })

    df_all = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    df_all.to_csv(OUT_DIR / "expanded_symbols.csv", index=False)

    included_df = df_all[df_all["included"]].copy()
    included_df = included_df.sort_values("symbol")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("STEP A RESULTS")
    print("=" * 70)
    print(f"Total Binance USDT pairs scanned : {len(all_usdt)}")
    print(f"Pre-excluded (stablecoin/wrapped): {sum(1 for r in pre_excluded.values() if 'stablecoin' in r or 'wrapped' in r)}")
    print(f"Pre-excluded (leveraged)         : {sum(1 for r in pre_excluded.values() if 'leveraged' in r)}")
    print(f"Insufficient bars (<{MIN_BARS})    : {sum(1 for r in rows if r['excluded'] == 'insufficient_bars')}")
    print(f"INCLUDED (≥{MIN_BARS} bars)       : {len(included_df)}")
    print(f"Current pipeline universe        : ~64 symbols")
    print(f"Net new symbols added            : {len(included_df) - 64}")
    print()
    print(f"{'Symbol':<22}  {'Bars':>5}  {'First':>12}  {'Last':>12}")
    print("-" * 60)
    for _, r in included_df.iterrows():
        print(f"  {r['symbol']:<20}  {int(r['bars']):>5}  {r['first_date']:>12}  {r['last_date']:>12}")

    print()
    print(f"Saved to: {OUT_DIR / 'expanded_symbols.csv'}")
    print()
    print("=" * 70)
    print("STEP A COMPLETE — awaiting review before Step B")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
