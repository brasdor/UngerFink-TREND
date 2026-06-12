#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- Funding Rate Extreme Short (Mean Reversion)

Hypothesis: Extreme positive funding rates signal crowded long positioning.
When the perpetual futures market is extremely long-biased, longs are paying
shorts large premiums every 8 hours -- signaling overextension. Mean reversion
downward is likely as the crowded trade unwinds.

This is unique to crypto perpetual futures and does NOT exist in TradFi.
The funding rate IS the signal -- no price action analysis needed.

Entry:   ANY 8h funding rate > entry_threshold on a symbol
         (aggregated to daily: max daily funding rate > threshold)
         Enter SHORT at open of NEXT 1D bar.
Exit:    When daily max funding rate drops below exit_threshold
         OR after max_hold_bars days -- whichever comes first.
Stop:    ATR * atr_mult ABOVE entry (hard safety stop only).

Structure:
  PHASE 1: Download historical funding rates from Binance Futures API.
           Saves to data/futures_universe/funding_rates/{SYMBOL}_funding.csv
           Skip if file exists (use --force to redownload).
  PHASE 2: Run T1 backtest grid on downloaded data aligned to 1D OHLCV.

Binance Futures funding schedule: every 8h at 00:00, 08:00, 16:00 UTC.
Earliest available data: ~September 2019 for BTC/ETH.

Parameter grid:
  entry_threshold: [0.0003, 0.0005, 0.0007, 0.001]  decimal (0.03%-0.10%/8h)
  exit_threshold:  [0.0001, 0.0002, 0.0003]          normalization (0.01%-0.03%/8h)
  max_hold_bars:   [5, 7, 10, 15]                    days (1D bars)
  atr_mult:        [2.0, 3.0]                  safety stop
  filter:          ["none", "ema200_below"]    regime filter

Stability zone: entry_threshold +/-1 step AND max_hold_bars +/-1 step, PASS if >=67%
§4.2 cost floor: 0.25R (Futures, higher than Spot 0.15R for funding costs)

PRIMARY GATE: 2022 must be POSITIVE
CONCENTRATION FLAG: 2025 > 40% of total R

Universe: symbols with pre-2021 data (58 symbols -- have 2022 bear cycle)
          + all symbols for wider signal coverage
Output:   data/research_fundingratemr_t1/
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    raise SystemExit("Install requests: pip install requests")


# =============================================================================
# CONFIG
# =============================================================================

ROOT          = Path(__file__).resolve().parent
OHLCV_1D      = ROOT / "data" / "futures_universe" / "ohlcv_1d"
FUNDING_DIR   = ROOT / "data" / "futures_universe" / "funding_rates"
OUT_DIR       = ROOT / "data" / "research_fundingratemr_t1"
FUNDING_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
API_LIMIT           = 1000          # max rows per API call
API_SLEEP_SEC       = 0.12          # ~8 calls/sec, well within 1200 weight/min
API_MAX_RETRIES     = 3
# Download from this date -- covers 2020-2022 bear cycles
DOWNLOAD_START_MS   = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)

# Parameter grid
ENTRY_THRESHOLDS = [0.0003, 0.0005, 0.0007, 0.001]  # decimal (0.03%-0.10%/8h)
EXIT_THRESHOLDS  = [0.0001, 0.0002, 0.0003]         # normalization (0.01%-0.03%/8h)
MAX_HOLD_BARS    = [5, 7, 10, 15]              # 1D bars
ATR_MULTS        = [2.0, 3.0]
FILTERS          = ["none", "ema200_below"]

ATR_N            = 14
EMA_N            = 200
ATR_PCT_WINDOW   = 252
ATR_PCT_MIN_BARS = 100

STABILITY_PASS_PCT = 67.0
COST_FLOOR_R       = 0.25
MIN_BARS           = 300
MIN_FUNDING_ROWS   = 100            # minimum funding rate rows to include a symbol

REPORT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
EPS          = 1e-12


def p(*a, **kw) -> None:
    kw.setdefault("flush", True)
    text = " ".join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), **kw)


# =============================================================================
# PHASE 1 -- DOWNLOAD FUNDING RATES
# =============================================================================

def download_funding_rates(symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
    """
    Download historical 8h funding rates for one symbol from Binance Futures.
    Paginates from DOWNLOAD_START_MS to now.
    Saves to FUNDING_DIR/{symbol}_funding.csv
    Returns DataFrame or None on failure.
    """
    out_path = FUNDING_DIR / f"{symbol}_funding.csv"
    if out_path.exists() and not force:
        try:
            df = pd.read_csv(out_path)
            if len(df) >= MIN_FUNDING_ROWS:
                return df
        except Exception:
            pass

    all_rows: List[dict] = []
    since_ms = DOWNLOAD_START_MS
    now_ms   = int(time.time() * 1000)

    for attempt in range(API_MAX_RETRIES):
        try:
            while since_ms < now_ms:
                params = {
                    "symbol":    symbol,
                    "startTime": since_ms,
                    "limit":     API_LIMIT,
                }
                resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=15)
                if resp.status_code == 400:
                    # Symbol not found on futures -- skip silently
                    return None
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list) or not data:
                    break
                for row in data:
                    all_rows.append({
                        "funding_time": int(row["fundingTime"]),
                        "funding_rate": float(row["fundingRate"]),
                    })
                last_ts = int(data[-1]["fundingTime"])
                if last_ts <= since_ms or len(data) < API_LIMIT:
                    break
                since_ms = last_ts + 1
                time.sleep(API_SLEEP_SEC)
            break
        except requests.exceptions.RequestException as e:
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows).drop_duplicates("funding_time").sort_values("funding_time")
    df.to_csv(out_path, index=False)
    return df


def load_or_download_funding(symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
    """Load from disk or download. Returns DataFrame with funding_time, funding_rate."""
    out_path = FUNDING_DIR / f"{symbol}_funding.csv"
    if out_path.exists() and not force:
        try:
            df = pd.read_csv(out_path)
            if len(df) >= MIN_FUNDING_ROWS:
                return df
        except Exception:
            pass
    return download_funding_rates(symbol, force=force)


def build_daily_funding(funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 8h funding rates to daily (UTC calendar day).
    Returns DataFrame indexed by date with:
      max_funding_rate, min_funding_rate, mean_funding_rate, n_readings
    """
    df = funding_df.copy()
    df["date"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.normalize()
    daily = df.groupby("date")["funding_rate"].agg(
        max_rate="max",
        min_rate="min",
        mean_rate="mean",
        n_readings="count",
    ).reset_index()
    daily = daily.rename(columns={"date": "date"})
    daily.index = pd.to_datetime(daily["date"])
    return daily


def run_download_phase(symbols: List[str], force: bool = False) -> Dict[str, pd.DataFrame]:
    """Download funding rates for all symbols. Returns dict of symbol -> daily_funding_df."""
    p(f"\nPHASE 1: Downloading historical funding rates ...")
    p(f"  Target: {len(symbols)} symbols, from {pd.Timestamp(DOWNLOAD_START_MS, unit='ms').date()}")
    p(f"  Saving to: {FUNDING_DIR}")
    p(f"  Force redownload: {force}")
    p()

    result: Dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols, 1):
        raw = load_or_download_funding(symbol, force=force)
        if raw is not None and len(raw) >= MIN_FUNDING_ROWS:
            daily = build_daily_funding(raw)
            result[symbol] = daily
            if i % 30 == 0 or i == len(symbols):
                p(f"  [{i}/{len(symbols)}]  downloaded={len(result)}")
        else:
            if i % 30 == 0:
                p(f"  [{i}/{len(symbols)}]  downloaded={len(result)}  (skipped: no data)")
        time.sleep(API_SLEEP_SEC)

    p(f"\n  Download complete: {len(result)} symbols with funding rate history")
    return result


# =============================================================================
# OHLCV LOADING
# =============================================================================

def _parse_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
        try:
            df.index = pd.to_datetime(ts.astype(float), unit="ms", utc=True)
        except (ValueError, TypeError):
            df.index = pd.to_datetime(ts, utc=True, errors="coerce")
    else:
        return None
    df = df.sort_index()
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].astype(float).dropna()
    return df if len(df) >= MIN_BARS else None


def load_1d(symbol: str) -> Optional[pd.DataFrame]:
    path = OHLCV_1D / f"{symbol}_1d.csv"
    if not path.exists():
        return None
    try:
        return _parse_ohlcv(pd.read_csv(path))
    except Exception:
        return None


def load_all_symbols() -> List[str]:
    return sorted(
        f.stem.replace("_1d", "").upper()
        for f in OHLCV_1D.glob("*_1d.csv")
    )


# =============================================================================
# INDICATORS
# =============================================================================

def precompute_symbol(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Compute ATR and EMA200 once per symbol (not once per combo)."""
    d = df.copy()

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"]    = tr.rolling(ATR_N).mean()
    d["ema200"] = d["close"].ewm(span=EMA_N, adjust=False).mean()

    warmup = max(EMA_N, ATR_N) + 5
    return d, warmup


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_symbol(
    symbol: str,
    d: pd.DataFrame,               # precomputed dataframe (from precompute_symbol)
    warmup: int,
    funding_dict: Dict,            # date -> max_rate (O(1) lookup)
    entry_thr: float,
    exit_thr: float,
    max_hold: int,
    atr_mult: float,
    filter_mode: str,
) -> List[dict]:
    """
    Backtest funding rate mean reversion short on one symbol.

    Signal logic:
      Day D: max_funding_rate > entry_thr  --> enter short at open of day D+1
      Exit:  max_funding_rate < exit_thr   OR  bars_held >= max_hold
      Stop:  high >= safety_stop --> stop out at safety_stop

    funding_dict: {normalized_date -> max_rate}  (O(1) lookup, built once per symbol)
    """
    # Extract numpy arrays for speed
    atr_arr    = d["atr"].values.astype(float)
    ema200_arr = d["ema200"].values.astype(float)
    high_arr   = d["high"].values.astype(float)
    open_arr   = d["open"].values.astype(float)
    close_arr  = d["close"].values.astype(float)
    timestamps = d.index
    # Pre-build date strings once for O(1) lookup (avoids tz/type mismatch)
    date_strs  = [str(pd.Timestamp(t).date()) for t in timestamps]

    pos    = None
    trades = []

    for i in range(warmup, len(d)):
        atr = atr_arr[i]
        if not math.isfinite(atr) or atr <= 0:
            continue

        close  = close_arr[i]
        high   = high_arr[i]
        open_  = open_arr[i]
        ema200 = ema200_arr[i]
        t      = timestamps[i]

        # O(1) funding rate lookup via date string key
        today_max_fr = funding_dict.get(date_strs[i], 0.0)

        # --- Manage open position ---
        if pos is not None:
            bars_held = i - pos["entry_bar"]

            # Safety stop
            if high >= pos["safety_stop"]:
                net_r = (pos["entry_price"] - pos["safety_stop"]) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, pos["safety_stop"], net_r, "safety_stop",
                    entry_thr, exit_thr, max_hold, atr_mult, filter_mode,
                ))
                pos = None
                continue

            # Check funding rate normalization OR time exit
            if today_max_fr < exit_thr or bars_held >= max_hold:
                reason = "fr_normalized" if today_max_fr < exit_thr else "time_exit"
                net_r  = (pos["entry_price"] - close) / max(pos["risk"], EPS)
                trades.append(_make_trade(
                    symbol, pos, t, close, net_r, reason,
                    entry_thr, exit_thr, max_hold, atr_mult, filter_mode,
                ))
                pos = None
                continue

        # --- Check yesterday's signal to enter today ---
        # Signal fires end-of-day D; enter at open of day D+1 (no lookahead)
        if pos is None and i > 0:
            prev_max_fr = funding_dict.get(date_strs[i - 1], 0.0)   # O(1) lookup

            if prev_max_fr > entry_thr:
                if filter_mode == "ema200_below" and close >= ema200:
                    pass  # regime filter blocks entry
                else:
                    safety_stop = open_ + atr * atr_mult
                    risk        = safety_stop - open_
                    if risk > EPS:
                        pos = {
                            "entry_bar":    i,
                            "entry_time":   t,
                            "entry_price":  open_,
                            "safety_stop":  safety_stop,
                            "risk":         risk,
                            "trigger_rate": prev_max_fr,
                        }

    return trades


def _make_trade(
    symbol: str,
    pos: dict,
    exit_time,
    exit_price: float,
    net_r: float,
    reason: str,
    entry_thr: float,
    exit_thr: float,
    max_hold: int,
    atr_mult: float,
    filter_mode: str,
) -> dict:
    return {
        "symbol":        symbol,
        "side":          "SHORT",
        "timeframe":     "1d",
        "entry_thr":     entry_thr,
        "exit_thr":      exit_thr,
        "max_hold":      max_hold,
        "atr_mult":      atr_mult,
        "filter_mode":   filter_mode,
        "trigger_rate":  pos.get("trigger_rate", 0.0),
        "entry_time":    pos["entry_time"],
        "exit_time":     exit_time,
        "entry_price":   pos["entry_price"],
        "exit_price":    float(exit_price),
        "safety_stop":   pos["safety_stop"],
        "initial_risk":  pos["risk"],
        "net_r":         float(net_r),
        "exit_reason":   reason,
        "entry_year":    pos["entry_time"].year,
    }


# =============================================================================
# STATISTICS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g =  r[r > 0].sum()
    l = -r[r < 0].sum()
    return float(g / l) if l > EPS else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(trades: List[dict]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0)
    r = np.array([t["net_r"] for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0)
    return dict(
        trades       = int(len(r)),
        total_r      = float(r.sum()),
        avg_r        = float(r.mean()),
        pf           = _pf(r),
        max_dd_r     = _dd(r),
        win_rate_pct = float((r > 0).mean() * 100),
    )


# =============================================================================
# YEAR-BY-YEAR
# =============================================================================

def year_by_year(trades: List[dict]) -> Dict[int, dict]:
    by_year: Dict[int, List[dict]] = {}
    for t in trades:
        yr = int(t.get("entry_year", 0))
        by_year.setdefault(yr, []).append(t)
    return {yr: _stats(by_year.get(yr, [])) for yr in REPORT_YEARS}


def _year_flags(yby: Dict[int, dict], total_r: float) -> List[str]:
    flags = []
    for yr in REPORT_YEARS:
        yr_r = yby[yr]["total_r"]
        threshold = 0.40 if yr == 2025 else 0.50
        if abs(total_r) > EPS and abs(yr_r) > threshold * abs(total_r):
            flags.append(
                f"[CONCENTRATED]  {yr} = {yr_r:+.1f}R = "
                f"{yr_r / total_r * 100:.0f}% of total R"
                + (" <-- 2025 RECENCY" if yr == 2025 else "")
            )
    y2022 = yby.get(2022, {})
    if y2022.get("trades", 0) > 0:
        if y2022["total_r"] < 0:
            flags.append(
                f"[!2022 NEGATIVE]  2022 = {y2022['total_r']:+.1f}R -- "
                f"short system MUST profit in 2022 bear year"
            )
    else:
        flags.append(
            "[!2022 NO TRADES]  -- funding rates may not be available for 2022"
        )
    return flags


# =============================================================================
# STABILITY -- 2D zone: entry_threshold +/-1 step AND max_hold_bars +/-1 step
# =============================================================================

def _adjacent(val, grid: list) -> list:
    try:
        idx = grid.index(val)
    except ValueError:
        return [val]
    result = [grid[idx]]
    if idx > 0:
        result.append(grid[idx - 1])
    if idx < len(grid) - 1:
        result.append(grid[idx + 1])
    return result


def stability_analysis(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (et_val, xt, am, fm), g in grid_df.groupby(
        ["entry_thr", "exit_thr", "atr_mult", "filter_mode"]
    ):
        lookup: Dict[Tuple, float] = {}
        for _, row in g.iterrows():
            lookup[(float(row["entry_thr"]), int(row["max_hold"]))] = float(row["avg_r"])

        for _, row in g.iterrows():
            et = float(row["entry_thr"])
            mh = int(row["max_hold"])

            zone_ets  = _adjacent(et, ENTRY_THRESHOLDS)
            zone_mhs  = _adjacent(mh, MAX_HOLD_BARS)
            zone_combos = [(ze, zm) for ze in zone_ets for zm in zone_mhs]
            zone_pass   = [1 for ze, zm in zone_combos
                           if lookup.get((ze, zm), -999) > 0]
            pct     = 100.0 * len(zone_pass) / max(len(zone_combos), 1)
            verdict = ("PASS" if pct >= STABILITY_PASS_PCT
                       else ("WARN" if pct >= 40 else "FAIL"))
            cf      = "PASS" if float(row["avg_r"]) > COST_FLOOR_R else "FAIL"

            rows.append(dict(
                entry_thr        = et,
                exit_thr         = xt,
                max_hold         = mh,
                atr_mult         = am,
                filter_mode      = fm,
                trades           = int(row["trades"]),
                total_r          = float(row["total_r"]),
                avg_r            = float(row["avg_r"]),
                pf               = float(row["pf"]),
                max_dd_r         = float(row["max_dd_r"]),
                win_rate_pct     = float(row["win_rate_pct"]),
                zone_size        = len(zone_combos),
                zone_pass        = len(zone_pass),
                zone_pct         = round(pct, 1),
                stability_verdict= verdict,
                sec42            = cf,
            ))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["filter_mode", "exit_thr", "atr_mult", "zone_pct", "avg_r"],
        ascending=[True, True, True, False, False]
    ).reset_index(drop=True)


# =============================================================================
# MAIN GRID RUN
# =============================================================================

def run_grid(
    symbols: List[str],
    funding_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[dict]]:
    """Run all parameter combos across all symbols with funding data."""
    combos = [
        (et, xt, mh, am, fm)
        for et in ENTRY_THRESHOLDS
        for xt in EXIT_THRESHOLDS
        for mh in MAX_HOLD_BARS
        for am in ATR_MULTS
        for fm in FILTERS
    ]

    combo_trades: Dict[Tuple, List[dict]] = {c: [] for c in combos}
    loaded = 0

    for sym_i, symbol in enumerate(symbols, 1):
        if symbol not in funding_data:
            continue
        ohlcv = load_1d(symbol)
        if ohlcv is None:
            continue
        daily_fund = funding_data[symbol]
        loaded += 1

        # Precompute indicators once per symbol
        d, warmup = precompute_symbol(ohlcv)

        # Build O(1) funding dict once per symbol -- use date STRING keys
        # to avoid timezone/type mismatch between pd.Timestamp and numpy datetime64
        funding_dict: Dict[str, float] = {}
        for idx_dt, row in daily_fund.iterrows():
            key = str(pd.Timestamp(idx_dt).date())   # "YYYY-MM-DD"
            funding_dict[key] = float(row["max_rate"])

        if sym_i % 30 == 0 or sym_i == len(symbols):
            p(f"  [{sym_i}/{len(symbols)}]  processed={loaded}")

        for (et, xt, mh, am, fm) in combos:
            trades = backtest_symbol(symbol, d, warmup, funding_dict, et, xt, mh, am, fm)
            combo_trades[(et, xt, mh, am, fm)].extend(trades)

    rows = []
    all_trades: List[dict] = []
    for (et, xt, mh, am, fm), trades in combo_trades.items():
        all_trades.extend(trades)
        s = _stats(trades)
        rows.append(dict(
            entry_thr  = et,
            exit_thr   = xt,
            max_hold   = mh,
            atr_mult   = am,
            filter_mode= fm,
            **s,
        ))

    grid_df = pd.DataFrame(rows)
    return grid_df, all_trades


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    grid_df: pd.DataFrame,
    sa: pd.DataFrame,
    all_trades_df: pd.DataFrame,
    n_symbols: int,
    funding_coverage: Dict[str, Tuple[str, str, int]],
) -> None:
    lines = [
        "PHASE T1 -- Funding Rate Extreme Short (Mean Reversion)",
        "=" * 80,
        "",
        "Hypothesis: Extreme positive funding rates signal crowded long positioning.",
        "  When longs pay extreme premiums to shorts every 8h, overextension is",
        "  confirmed. Mean reversion downward is likely as the crowded trade unwinds.",
        "  This signal is unique to crypto perpetuals -- no equivalent in TradFi.",
        "",
        "Entry:   max_daily_funding_rate > entry_threshold",
        "         Enter SHORT at open of next 1D bar.",
        "Exit:    max_daily_funding_rate < exit_threshold  OR  bars_held >= max_hold",
        "Stop:    ATR * atr_mult ABOVE entry (hard safety stop only).",
        "",
        f"Universe:       {n_symbols} symbols with both OHLCV and funding rate data",
        f"Funding source: {FUNDING_DIR}",
        "",
        "Parameter grid:",
        f"  entry_threshold: {ENTRY_THRESHOLDS}  (decimal; 0.0005 = 0.05%/8h)",
        f"  exit_threshold:  {EXIT_THRESHOLDS}   (decimal; 0.0001 = 0.01%/8h)",
        f"  max_hold_bars:   {MAX_HOLD_BARS}    (1D bars)",
        f"  atr_mult:        {ATR_MULTS}          (safety stop)",
        f"  filter_mode:     {FILTERS}",
        "",
        f"Stability zone: entry_threshold +/-1 step AND max_hold +/-1 step, PASS if >={STABILITY_PASS_PCT:.0f}%",
        f"Cost floor:     avg_r > {COST_FLOOR_R}R  (Futures §4.2)",
        "",
        "PRIMARY GATE:       2022 MUST BE POSITIVE",
        "CONCENTRATION FLAG: 2025 > 40% of total R",
        "",
    ]

    # Funding coverage summary
    lines += [
        "FUNDING RATE COVERAGE SUMMARY",
        "-" * 60,
        f"  {'Symbol':<20}  {'From':>10}  {'To':>10}  {'Rows':>6}",
    ]
    for sym, (fr, to, cnt) in sorted(funding_coverage.items())[:30]:
        lines.append(f"  {sym:<20}  {fr:>10}  {to:>10}  {cnt:>6}")
    if len(funding_coverage) > 30:
        lines.append(f"  ... and {len(funding_coverage) - 30} more symbols")
    lines.append("")

    # Results by filter_mode
    for fm in FILTERS:
        for am in ATR_MULTS:
            sub = sa[(sa["filter_mode"] == fm) & (sa["atr_mult"] == am)] if not sa.empty else pd.DataFrame()
            if sub.empty:
                continue

            pass_sa = sub[sub["stability_verdict"] == "PASS"]
            warn_sa = sub[sub["stability_verdict"] == "WARN"]
            top_pass = pass_sa.sort_values("avg_r", ascending=False).head(10)
            top_warn = warn_sa.sort_values("avg_r", ascending=False).head(5)

            lines += [
                "=" * 80,
                f"filter={fm}  ATR stop mult={am:.1f}  |  "
                f"PASS: {len(pass_sa)}  WARN: {len(warn_sa)}  "
                f"FAIL: {len(sub) - len(pass_sa) - len(warn_sa)}",
                "=" * 80,
            ]

            if not top_pass.empty:
                lines += ["TOP PASS COMBOS (sorted by avg_r)", "-" * 60]
                for _, row in top_pass.iterrows():
                    lines.append(
                        f"  [PASS] et={row['entry_thr']:.2f}%  xt={row['exit_thr']:.2f}%  "
                        f"mh={int(row['max_hold']):2d}  "
                        f"zone={int(row['zone_pass'])}/{int(row['zone_size'])} ({row['zone_pct']:.0f}%)  "
                        f"trades={int(row['trades'])}  "
                        f"avg_r={row['avg_r']:+.4f}  "
                        f"pf={row['pf']:.2f}  "
                        f"win%={row['win_rate_pct']:.1f}%  "
                        f"§4.2={row['sec42']}"
                    )
                lines.append("")

            if not top_warn.empty:
                lines += ["TOP WARN COMBOS", "-" * 60]
                for _, row in top_warn.iterrows():
                    lines.append(
                        f"  [WARN] et={row['entry_thr']:.2f}%  xt={row['exit_thr']:.2f}%  "
                        f"mh={int(row['max_hold']):2d}  "
                        f"zone={int(row['zone_pass'])}/{int(row['zone_size'])} ({row['zone_pct']:.0f}%)  "
                        f"trades={int(row['trades'])}  "
                        f"avg_r={row['avg_r']:+.4f}  "
                        f"§4.2={row['sec42']}"
                    )
                lines.append("")

    # --- Heatmap: entry_threshold x max_hold for best exit_thr/atr_mult ---
    if not grid_df.empty:
        best_xt = grid_df.groupby("exit_thr")["avg_r"].mean().idxmax()
        best_am = grid_df.groupby("atr_mult")["avg_r"].mean().idxmax()
        for fm in FILTERS:
            g = grid_df[
                (grid_df["filter_mode"] == fm) &
                (grid_df["exit_thr"]    == best_xt) &
                (grid_df["atr_mult"]    == best_am)
            ]
            if g.empty:
                continue
            lines += [
                f"avg_r HEATMAP  (filter={fm}  exit_thr={best_xt:.2f}%  ATR*{best_am:.1f})  * = passes §4.2",
                "-" * 65,
                f"  {'et\\mh':>8}  " + "  ".join(f"mh={mh:2d}" for mh in MAX_HOLD_BARS),
            ]
            for et in ENTRY_THRESHOLDS:
                row_vals = []
                for mh in MAX_HOLD_BARS:
                    sl = g[(g["entry_thr"] == et) & (g["max_hold"] == mh)]
                    if sl.empty:
                        row_vals.append("   ----")
                    else:
                        v  = float(sl.iloc[0]["avg_r"])
                        cf = "*" if v > COST_FLOOR_R else " "
                        sign = "+" if v >= 0 else ""
                        row_vals.append(f"{cf}{sign}{v:.3f}")
                lines.append(f"  et={et:.2f}%:  " + "  ".join(row_vals))
            lines.append("")

    # --- 2022 Check ---
    lines += [
        "=" * 80,
        "2022 BEAR MARKET CHECK  (PASS + §4.2 combos only)",
        "=" * 80,
    ]
    if sa.empty:
        lines.append("  No stability analysis data.")
    else:
        cf_pass = sa[(sa["stability_verdict"] == "PASS") & (sa["sec42"] == "PASS")]
        if cf_pass.empty:
            lines.append("  No combos pass both stability AND §4.2.")
            best_r = sa["avg_r"].max() if not sa.empty else 0.0
            lines.append(f"  Best avg_r across all combos: {best_r:+.4f}R")
        else:
            t2022 = all_trades_df[all_trades_df["entry_year"] == 2022] if not all_trades_df.empty else pd.DataFrame()
            for _, row in cf_pass.sort_values("avg_r", ascending=False).head(10).iterrows():
                sl = t2022[
                    (t2022["entry_thr"]   == row["entry_thr"]) &
                    (t2022["exit_thr"]    == row["exit_thr"]) &
                    (t2022["max_hold"]    == row["max_hold"]) &
                    (t2022["atr_mult"]    == row["atr_mult"]) &
                    (t2022["filter_mode"] == row["filter_mode"])
                ] if not t2022.empty else pd.DataFrame()
                s22    = _stats(sl.to_dict("records") if not sl.empty else [])
                status = "OK" if s22["total_r"] >= 0 else "FAIL -- 2022 NEGATIVE"
                lines.append(
                    f"  et={row['entry_thr']:.2f}%  xt={row['exit_thr']:.2f}%  "
                    f"mh={int(row['max_hold'])}  ATR*{row['atr_mult']:.1f}  "
                    f"filter={row['filter_mode']}  "
                    f"2022: t={s22['trades']}  avg_r={s22['avg_r']:+.4f}  "
                    f"total_r={s22['total_r']:+.1f}R  [{status}]"
                )
    lines.append("")

    # --- Year-by-year best combo ---
    lines += ["=" * 80, "YEAR-BY-YEAR  (best PASS combo)", "=" * 80]
    if sa.empty or all_trades_df.empty:
        lines.append("  No data.")
    else:
        best_rows = sa[sa["stability_verdict"] == "PASS"].sort_values(
            ["sec42", "avg_r"], ascending=[False, False]
        )
        if best_rows.empty:
            lines.append("  No PASS combo found.")
        else:
            best    = best_rows.iloc[0]
            bt = all_trades_df[
                (all_trades_df["entry_thr"]   == best["entry_thr"]) &
                (all_trades_df["exit_thr"]    == best["exit_thr"]) &
                (all_trades_df["max_hold"]    == best["max_hold"]) &
                (all_trades_df["atr_mult"]    == best["atr_mult"]) &
                (all_trades_df["filter_mode"] == best["filter_mode"])
            ].to_dict("records")
            yby   = year_by_year(bt)
            total = _stats(bt)
            flags = _year_flags(yby, total["total_r"])
            lines.append(
                f"  et={best['entry_thr']:.2f}%  xt={best['exit_thr']:.2f}%  "
                f"mh={int(best['max_hold'])}  ATR*{best['atr_mult']:.1f}  "
                f"filter={best['filter_mode']}  "
                f"total_r={total['total_r']:+.1f}R  avg_r={total['avg_r']:+.4f}R"
            )
            lines.append(f"  {'Year':>6}  {'Trades':>7}  {'TotalR':>8}  {'AvgR':>8}  {'WinPct':>7}  {'PF':>5}")
            lines.append(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}")
            for yr in REPORT_YEARS:
                s = yby[yr]
                if s["trades"] == 0:
                    continue
                lines.append(
                    f"  {yr:>6}  {s['trades']:>7}  {s['total_r']:>+8.2f}  "
                    f"{s['avg_r']:>+8.4f}  {s['win_rate_pct']:>7.1f}%  {s['pf']:>5.2f}"
                )
            lines.append("")
            for f_ in flags:
                lines.append(f"  {f_}")
            lines.append("")

    # --- Overall verdict ---
    lines += ["=" * 80, "OVERALL VERDICT", "=" * 80]
    if sa.empty:
        lines += [
            "  No backtest data -- check funding rate download.",
            "  VERDICT: INSUFFICIENT DATA",
        ]
    else:
        cf_pass = sa[(sa["stability_verdict"] == "PASS") & (sa["sec42"] == "PASS")]
        best_r  = sa["avg_r"].max()
        if cf_pass.empty:
            lines += [
                f"  NO combos pass both stability AND §4.2 cost floor.",
                f"  Best avg_r: {best_r:+.4f}R  (need >{COST_FLOOR_R}R)",
                "  VERDICT: HALT T1 -- insufficient edge.",
            ]
        else:
            best = cf_pass.sort_values("avg_r", ascending=False).iloc[0]
            lines += [
                f"  PASS combos (stability + §4.2): {len(cf_pass)}",
                f"  Best candidate:",
                f"    entry_threshold = {best['entry_thr']:.2f}%",
                f"    exit_threshold  = {best['exit_thr']:.2f}%",
                f"    max_hold_bars   = {int(best['max_hold'])}",
                f"    atr_mult        = {best['atr_mult']:.1f}",
                f"    filter_mode     = {best['filter_mode']}",
                f"    trades          = {int(best['trades'])}",
                f"    avg_r           = {best['avg_r']:+.4f}R",
                f"    pf              = {best['pf']:.2f}",
                f"    zone            = {int(best['zone_pass'])}/{int(best['zone_size'])} ({best['zone_pct']:.0f}%)",
                "",
                "  VERDICT: PROCEED TO T2 -- review 2022 gate above",
            ]

    report_path = OUT_DIR / "phase_t1_fundingratemr_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        p(line)


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase T1 -- Funding Rate Extreme Short"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force redownload of funding rate data even if files exist"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download phase (use only cached data)"
    )
    parser.add_argument(
        "--max-symbols", type=int, default=0,
        help="Limit to first N symbols (0 = all, for testing)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    p("=" * 80)
    p("Phase T1 -- Funding Rate Extreme Short (Mean Reversion)")
    p("=" * 80)
    p()

    # All symbols with 1D OHLCV
    symbols = load_all_symbols()
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    p(f"OHLCV universe: {len(symbols)} symbols")

    # Phase 1: Download / load funding rates
    if not args.skip_download:
        funding_raw = run_download_phase(symbols, force=args.force)
    else:
        p("Skipping download -- loading cached data only ...")
        funding_raw = {}
        for sym in symbols:
            path = FUNDING_DIR / f"{sym}_funding.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if len(df) >= MIN_FUNDING_ROWS:
                        funding_raw[sym] = build_daily_funding(df)
                except Exception:
                    pass
        p(f"  Loaded {len(funding_raw)} symbols from cache")

    if not funding_raw:
        p("ERROR: No funding rate data available. Run without --skip-download.")
        import sys; sys.exit(1)

    # Build coverage report
    funding_coverage: Dict[str, Tuple[str, str, int]] = {}
    for sym, daily_df in funding_raw.items():
        dates = pd.to_datetime(daily_df.index)
        funding_coverage[sym] = (
            str(dates.min().date()),
            str(dates.max().date()),
            len(daily_df),
        )
    coverage_df = pd.DataFrame(
        [{"symbol": s, "from": v[0], "to": v[1], "rows": v[2]}
         for s, v in funding_coverage.items()]
    )
    coverage_df.to_csv(OUT_DIR / "funding_coverage.csv", index=False)

    p(f"\nFunding data available for {len(funding_raw)} symbols")
    date_range = coverage_df["from"].min() + " to " + coverage_df["to"].max()
    p(f"Date range: {date_range}")

    # Phase 2: Backtest
    p("\nPHASE 2: Running T1 backtest grid ...")
    p(f"  Combos: {len(ENTRY_THRESHOLDS)} x {len(EXIT_THRESHOLDS)} x "
      f"{len(MAX_HOLD_BARS)} x {len(ATR_MULTS)} x {len(FILTERS)} = "
      f"{len(ENTRY_THRESHOLDS)*len(EXIT_THRESHOLDS)*len(MAX_HOLD_BARS)*len(ATR_MULTS)*len(FILTERS)}")
    p()

    grid_df, all_trades = run_grid(symbols, funding_raw)

    if not all_trades:
        p("WARNING: No trades generated. Check that funding rate dates align with OHLCV dates.")
        all_trades_df = pd.DataFrame()
    else:
        all_trades_df = pd.DataFrame(all_trades)
        p(f"Total trades across all combos: {len(all_trades_df)}")

    grid_df.to_csv(OUT_DIR / "phase_t1_fundingratemr_grid.csv", index=False)
    if not all_trades_df.empty:
        all_trades_df.to_csv(OUT_DIR / "phase_t1_fundingratemr_trades.csv", index=False)
    p(f"Saved grid CSV to {OUT_DIR}")

    p("\nRunning stability analysis ...")
    sa = stability_analysis(grid_df) if not grid_df.empty else pd.DataFrame()
    if not sa.empty:
        sa.to_csv(OUT_DIR / "phase_t1_fundingratemr_stability.csv", index=False)

    p("\nWriting report ...")
    write_report(grid_df, sa, all_trades_df, len(funding_raw), funding_coverage)

    p(f"\nReport saved to {OUT_DIR / 'phase_t1_fundingratemr_report.txt'}")
    p("\nDone. Do not proceed to T2 until you review the report.")


if __name__ == "__main__":
    main()
