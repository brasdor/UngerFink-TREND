#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 -- Binance Futures Universe Download
Download all available 1D OHLCV data from Binance USD-M Futures perpetuals.
Uses ccxt binanceusdm -- works from local PC (no 451 restriction).

Output:
  data/futures_universe/ohlcv_1d/{SYMBOL}_1d.csv
  data/futures_universe/coverage_report.csv
  data/futures_universe/all_symbols.csv
  data/futures_universe/symbols_pre2021.csv
  data/futures_universe/symbols_pre2020.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT      = Path(__file__).resolve().parent
OUT_DIR   = ROOT / "data" / "futures_universe"
OHLCV_DIR = OUT_DIR / "ohlcv_1d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OHLCV_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 500

EXCLUDE_BASE = {
    'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDE', 'USDP',
    'USDT', 'UST', 'USDN',
    'LUNA', 'FTT', 'LUNC', 'LUNA2',
}
EXCLUDE_KEYWORDS = {'UP', 'DOWN', 'BULL', 'BEAR'}

TIMEFRAME = '1d'
SINCE_ISO = '2017-01-01T00:00:00Z'
REQUEST_PAUSE = 0.08   # seconds between paginated requests


def _strip_multiplier(base: str) -> str:
    """Remove numeric prefixes like 1000, 10000, 1000000."""
    for prefix in ('1000000', '100000', '10000', '1000', '100'):
        if base.startswith(prefix) and len(base) > len(prefix):
            return base[len(prefix):]
    return base


def _should_exclude(base: str) -> bool:
    nb = _strip_multiplier(base.upper())
    if nb in EXCLUDE_BASE:
        return True
    for kw in EXCLUDE_KEYWORDS:
        if kw in nb:
            return True
    return False


def _fetch_full_ohlcv(exchange: ccxt.Exchange, symbol: str) -> list:
    since  = exchange.parse8601(SINCE_ISO)
    result = []
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1500)
        except ccxt.RateLimitExceeded:
            time.sleep(2.0)
            continue
        except Exception as exc:
            print(f"    [fetch error] {exc}")
            break
        if not batch:
            break
        result.extend(batch)
        if len(batch) < 1500:
            break
        since = batch[-1][0] + 1          # advance cursor past last bar
        time.sleep(REQUEST_PAUSE)
    return result


def _save_ohlcv(sym_id: str, raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')
    df = (df.drop_duplicates('timestamp')
            .sort_values('timestamp')
            .reset_index(drop=True))
    df.to_csv(OHLCV_DIR / f"{sym_id}_1d.csv", index=False)
    return df


def p(*a):
    text = ' '.join(str(x) for x in a)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode(), flush=True)


def main() -> None:
    p("=" * 72)
    p("  PHASE 1 -- Binance Futures Universe Download")
    p("  Exchange : binanceusdm (USD-M perpetuals, local PC)")
    p("  Timeframe: 1D  |  Min bars:", MIN_BARS)
    p("=" * 72)

    p("\n  [1/4] Loading markets...")
    exchange = ccxt.binanceusdm({
        'timeout': 30_000,
        'enableRateLimit': True,
    })
    markets = exchange.load_markets()
    p(f"         Total markets on binanceusdm: {len(markets)}")

    # Filter: USDT-margined perpetual swaps, active, not excluded
    candidates: list[tuple[str, str, str]] = []   # (sym_id, ccxt_sym, base)
    for sym, mkt in markets.items():
        if (mkt.get('quote') == 'USDT'
                and mkt.get('settle') == 'USDT'
                and mkt.get('type') == 'swap'
                and mkt.get('active', True)):
            base = mkt.get('base', '')
            if not _should_exclude(base):
                candidates.append((mkt['id'], sym, base))

    p(f"         After exclusion filters: {len(candidates)} candidates")

    # Download
    p(f"\n  [2/4] Downloading OHLCV data ({len(candidates)} symbols)...")
    coverage  : list[dict] = []
    ok_symbols: list[str]  = []
    skipped = 0

    for i, (sym_id, ccxt_sym, base) in enumerate(sorted(candidates), 1):
        csv_path = OHLCV_DIR / f"{sym_id}_1d.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            first = df['date'].iloc[0]
            last  = df['date'].iloc[-1]
            n     = len(df)
            has_pre2021 = first <= '2020-12-31'
            has_pre2020 = first <= '2019-12-31'
            p(f"  [{i:03d}/{len(candidates)}] {sym_id:<20} CACHED {first} to {last}  ({n:>5} bars)")
            coverage.append({
                'symbol':        sym_id,
                'base':          base,
                'earliest_date': first,
                'latest_date':   last,
                'total_bars':    n,
                'has_pre2021':   has_pre2021,
                'has_pre2020':   has_pre2020,
            })
            ok_symbols.append(sym_id)
            continue

        raw = _fetch_full_ohlcv(exchange, ccxt_sym)

        if len(raw) < MIN_BARS:
            p(f"  [{i:03d}/{len(candidates)}] {sym_id:<20} SKIP  ({len(raw)} bars < {MIN_BARS})")
            skipped += 1
            continue

        df = _save_ohlcv(sym_id, raw)
        first = df['date'].iloc[0]
        last  = df['date'].iloc[-1]
        n     = len(df)

        has_pre2021 = first <= '2020-12-31'
        has_pre2020 = first <= '2019-12-31'

        p(f"  [{i:03d}/{len(candidates)}] {sym_id:<20} {first} to {last}  "
          f"({n:>5} bars){'  [pre2021]' if has_pre2021 else ''}"
          f"{'[pre2020]' if has_pre2020 else ''}")

        coverage.append({
            'symbol':        sym_id,
            'base':          base,
            'earliest_date': first,
            'latest_date':   last,
            'total_bars':    n,
            'has_pre2021':   has_pre2021,
            'has_pre2020':   has_pre2020,
        })
        ok_symbols.append(sym_id)

    # Save outputs
    p("\n  [3/4] Saving outputs...")
    cov_df = pd.DataFrame(coverage).sort_values('symbol')
    cov_df.to_csv(OUT_DIR / 'coverage_report.csv', index=False)
    p(f"         Coverage report: {OUT_DIR / 'coverage_report.csv'}")

    pd.DataFrame({'symbol': sorted(ok_symbols)}).to_csv(
        OUT_DIR / 'all_symbols.csv', index=False)

    pre2021 = [r['symbol'] for r in coverage if r['has_pre2021']]
    pd.DataFrame({'symbol': sorted(pre2021)}).to_csv(
        OUT_DIR / 'symbols_pre2021.csv', index=False)

    pre2020 = [r['symbol'] for r in coverage if r['has_pre2020']]
    pd.DataFrame({'symbol': sorted(pre2020)}).to_csv(
        OUT_DIR / 'symbols_pre2020.csv', index=False)

    p(f"         all_symbols.csv:      {len(ok_symbols)} symbols")
    p(f"         symbols_pre2021.csv:  {len(pre2021)} symbols")
    p(f"         symbols_pre2020.csv:  {len(pre2020)} symbols")

    # Summary
    p("\n  [4/4] Summary")
    p("=" * 72)
    p("  PHASE 1 COMPLETE")
    p("=" * 72)
    p(f"  Downloaded:         {len(ok_symbols):>5} symbols")
    p(f"  Skipped (<{MIN_BARS}bars): {skipped:>5} symbols")
    p(f"  With pre-2021 data: {len(pre2021):>5}")
    p(f"  With pre-2020 data: {len(pre2020):>5}")
    if coverage:
        earliest = min(r['earliest_date'] for r in coverage)
        latest   = max(r['latest_date']   for r in coverage)
        avg_bars = sum(r['total_bars'] for r in coverage) // len(coverage)
        p(f"  Dataset range:      {earliest} to {latest}")
        p(f"  Avg bars/symbol:    {avg_bars}")


if __name__ == "__main__":
    main()
