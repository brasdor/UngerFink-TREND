#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- DONCHIAN UNIVERSEV2 EXITV2 PAPER ENGINE
=====================================================

Paper trading engine for the frozen DonchianLong_UniverseV2_ExitV2 config.

Frozen config (2026-05-30):
  Universe    : 24 symbols (filtered_symbols_v2_included_only.csv)
  Timeframe   : 1D
  Filter      : EMA200 price-above (close > EMA200)
  Entry       : Donchian N=20 upper band breakout (prev 20 highs)
  Stop        : ATR(14) x 2.0 below entry close
  Exit        : Donchian N=10 lower band (close) OR
                Chandelier (activates at +6R, trails at 5.0 x ATR)
  Max pos     : 8 concurrent (one per symbol)
  Risk/trade  : 0.25% of current closed equity
  Leverage    : 1.0x  (Binance Spot, LONG only)
  Side        : LONG only

OHLCV data strategy (works from GitHub Actions / US-based servers):
  1. Load committed historical cache: data/universe/ohlcv_1d/{sym}_1d.csv
     This file is tracked in git and checked out with the repo.
     It covers 2020-present for all 24 symbols.
  2. Append latest 10 bars via yfinance (Yahoo Finance).
     yfinance is geo-unrestricted and works from GitHub Actions US servers.
     Binance.com returns HTTP 451 from US IPs; ccxt/binanceus also fails
     on GitHub Actions -- yfinance is the only reliable live source.
  3. Save updated file back to data/universe/ohlcv_1d/ so the GitHub Actions
     commit step accumulates the cache one row per day.
  4. Use --no-download to skip live fetch (read committed cache only).

Usage:
  python phase_t9b_donchian_universev2_paper_engine.py
      --date 2026-05-31    run for a specific date (default: yesterday)
      --backfill           replay from freeze date (2026-05-30) through yesterday
      --no-download        use committed cache only, no live API calls
      --reset              wipe state.json and restart from $10,000
      --notify             print compact notification summary after run

Output files (data/t9b_paper/):
  state.json            persistent engine state (positions, equity)
  open_positions.csv    current open positions with R, stop, chandelier info
  signals_today.csv     all entry signals detected on last run date
  equity_curve.csv      one row per closed trade (cumulative equity)
  daily_log.csv         all events: entries, exits, skipped signals

PAPER ONLY -- NO REAL ORDERS.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN CONFIG  (2026-05-30)
# =============================================================================

SYSTEM_NAME            = "TREND_1D_DONCHIAN_EXITV2_T9B"
FREEZE_DATE            = date(2026, 5, 30)

DONCHIAN_ENTRY_N       = 20
DONCHIAN_EXIT_N        = 10
EMA_N                  = 200
ATR_N                  = 14

ATR_STOP_MULT          = 2.0
CHANDELIER_MULT        = 5.0
CHANDELIER_ACTIVATE_R  = 6.0   # R threshold before chandelier activates

RISK_PER_TRADE_PCT     = 0.0025   # 0.25%
MAX_OPEN_POSITIONS     = 8
INITIAL_CAPITAL        = 10_000.0
LEVERAGE               = 1.0
KILL_SWITCH_DD_PCT     = 35.0     # halt new entries if DD exceeds this from peak

LIMIT_BARS             = 1500     # bars to download per symbol (~4yr on 1D)
SLEEP_SEC              = 0.15
MAX_RETRIES            = 3

EPS = 1e-12

MIN_BARS_REQUIRED = EMA_N + DONCHIAN_ENTRY_N + 5   # minimum bars to compute indicators


# =============================================================================
# PATHS
# =============================================================================

ROOT         = Path.cwd()
DATA_DIR     = ROOT / "data" / "t9b_paper"
STATE_PATH   = DATA_DIR / "state.json"

UNIVERSE_CSV     = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
COMMITTED_CACHE  = ROOT / "data" / "universe" / "ohlcv_1d"   # tracked in git

OPEN_POS_CSV  = DATA_DIR / "open_positions.csv"
SIGNALS_CSV   = DATA_DIR / "signals_today.csv"
EQUITY_CSV    = DATA_DIR / "equity_curve.csv"
DAILY_LOG_CSV = DATA_DIR / "daily_log.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Per-run in-memory cache: symbol -> full parsed DataFrame.
# Populated on first access; persists for the whole script run.
# Avoids hitting the live API more than once per symbol per execution.
_SESSION_OHLCV: dict = {}

# Set to True when --no-download is passed; skips all live API calls.
_SKIP_LIVE_FETCH: bool = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    position_id:               str
    symbol:                    str
    entry_date:                str
    entry_price:               float
    initial_stop:              float
    current_stop:              float          # updates as chandelier ratchets
    initial_risk_per_unit:     float          # ATR x 2.0 at entry
    risk_amount_usdt:          float          # equity x 0.25% at entry
    qty:                       float          # shares = risk_amount / risk_per_unit
    highest_high_since_entry:  float
    max_favorable_r:           float
    chandelier_active:         bool
    chandelier_activated_date: Optional[str]
    atr_at_entry:              float
    donchian_entry_level:      float          # breakout level that triggered entry

    def current_r(self, price: float) -> float:
        return (price - self.entry_price) / max(self.initial_risk_per_unit, EPS)


# =============================================================================
# HELPERS
# =============================================================================

def safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _default_state()


def _default_state() -> dict:
    return {
        "system_name":           SYSTEM_NAME,
        "created_utc":           pd.Timestamp.utcnow().isoformat(),
        "last_run_date":         None,
        "paper_equity_usdt":     INITIAL_CAPITAL,
        "peak_equity_usdt":      INITIAL_CAPITAL,
        "drawdown_pct":          0.0,
        "kill_switch_triggered": False,
        "open_positions":        [],
        "closed_trade_count":    0,
    }


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


def positions_from_state(state: dict) -> Dict[str, Position]:
    """Deserialise positions from state JSON, tolerating missing optional fields."""
    fields = {f.name for f in dataclasses.fields(Position)}
    out: Dict[str, Position] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = Position(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping unloadable position {raw.get('position_id','?')}: {exc}")
    return out


def positions_to_state(state: dict, positions: Dict[str, Position]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=not path.exists())


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA200, ATR14, Donchian entry (N=20) and exit (N=10) bands.

    Donchian bands use shift(1) so a bar cannot trigger on its own high/low.
    Chandelier is computed live in position management, not here.
    """
    out = df.copy()

    # EMA200
    out["ema200"] = out["close"].ewm(span=EMA_N, adjust=False).mean()

    # ATR14 (Wilder smoothing = EWM with alpha=1/N)
    prev_c = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_c).abs(),
        (out["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / ATR_N, adjust=False).mean()

    # Donchian entry channel: max of PREVIOUS N=20 highs (no current bar)
    out["don_entry_high"] = out["high"].shift(1).rolling(DONCHIAN_ENTRY_N).max()

    # Donchian exit channel: min of PREVIOUS N=10 lows (no current bar)
    out["don_exit_low"] = out["low"].shift(1).rolling(DONCHIAN_EXIT_N).min()

    return out


# =============================================================================
# OHLCV LOADING
# =============================================================================

def load_ohlcv(symbol: str, up_to_date: Optional[date] = None) -> pd.DataFrame:
    """
    Load 1D OHLCV for a symbol.

    First call per symbol per session:
      1. Read committed historical cache (data/universe/ohlcv_1d/ -- in git)
      2. Fetch last 10 bars via yfinance (geo-unrestricted, works on GitHub Actions)
      3. Merge, deduplicate, save back to committed cache (one extra row per day)

    Subsequent calls for the same symbol reuse the in-memory session cache.

    up_to_date slicing: clips to that date for backfill mode (no lookahead).
    _SKIP_LIVE_FETCH=True (--no-download) reads committed cache only.
    """
    global _SESSION_OHLCV

    if symbol not in _SESSION_OHLCV:
        safe        = safe_sym(symbol)
        cache_path  = COMMITTED_CACHE / f"{safe}_1d.csv"

        # Load committed historical base
        raw_base: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                raw_base = pd.read_csv(cache_path)
            except Exception as exc:
                print(f"  [WARN] {symbol}: could not read committed cache -- {exc}")

        # Fetch latest bars (geo-unrestricted, skipped when --no-download)
        raw_live: Optional[pd.DataFrame] = None
        if not _SKIP_LIVE_FETCH:
            raw_live = _fetch_latest_bars(symbol, n_bars=10)

        # Parse each source individually (they may have different column formats:
        # committed cache uses ISO 'time' strings; live sources use ms 'timestamp').
        # Merging raw before parsing causes mixed-column overflow errors.
        df_base = _parse_ohlcv(raw_base) if raw_base is not None else pd.DataFrame()
        df_live = _parse_ohlcv(raw_live) if raw_live is not None else pd.DataFrame()

        if not df_base.empty and not df_live.empty:
            merged = (
                pd.concat([df_base, df_live], ignore_index=True)
                .drop_duplicates("time")
                .sort_values("time")
                .reset_index(drop=True)
            )
        elif not df_base.empty:
            merged = df_base
        elif not df_live.empty:
            merged = df_live
        else:
            merged = pd.DataFrame()

        # Save back to committed cache if live data was fetched
        if not merged.empty and raw_live is not None and cache_path.parent.exists():
            _save_committed_cache(merged, cache_path)

        _SESSION_OHLCV[symbol] = merged

    df = _SESSION_OHLCV[symbol]

    if up_to_date is not None:
        df = df[df["time"].dt.date <= up_to_date].copy()

    return df.reset_index(drop=True)


def _save_committed_cache(df: pd.DataFrame, path: Path) -> None:
    """Save a parsed DataFrame to the committed cache in ISO-date string format."""
    out = df[["time", "open", "high", "low", "close", "volume"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(path, index=False)


def _parse_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw OHLCV DataFrame to standard columns with UTC timestamps."""
    df = raw.copy()

    # Detect and parse timestamp column (handles ms-integer or ISO-string formats)
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"),
            unit="ms", utc=True, errors="coerce",
        )
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates("time").sort_values("time")
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live data fetchers (geo-unrestricted)
# ---------------------------------------------------------------------------

def _fetch_latest_bars(symbol: str, n_bars: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch the latest n_bars 1D candles via yfinance (Yahoo Finance).

    yfinance is geo-unrestricted and works from GitHub Actions US servers.
    Binance.com returns HTTP 451 from US IPs; binanceus (ccxt) also fails
    on GitHub Actions. yfinance is the only reliable source here.
    """
    return _try_yfinance(symbol, n_bars)


def _try_yfinance(symbol: str, n_bars: int) -> Optional[pd.DataFrame]:
    """
    Fetch from Yahoo Finance via yfinance.
    Converts BTC/USDT -> BTC-USD. Prices in USD ~ USDT for signal purposes.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return None

    base   = symbol.split("/")[0]
    yf_sym = f"{base}-USD"

    try:
        # period="30d" gives more than enough bars; we take the last n_bars
        hist = yf.download(
            yf_sym,
            period="30d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if hist is None or hist.empty:
            return None

        hist = hist.tail(n_bars).reset_index()

        # Flatten multi-level columns if present (yfinance >= 0.2 can return them)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() if c[1] == "" else c[0].lower()
                            for c in hist.columns]
        else:
            hist.columns = [str(c).lower() for c in hist.columns]

        # Rename date -> time
        if "date" in hist.columns:
            hist.rename(columns={"date": "time"}, inplace=True)
        elif "datetime" in hist.columns:
            hist.rename(columns={"datetime": "time"}, inplace=True)

        hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")
        hist["timestamp"] = (hist["time"].astype(np.int64) // 1_000_000).astype(int)

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in hist.columns:
                hist[col] = np.nan

        return hist[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    except Exception as exc:
        print(f"    [yfinance] {symbol}: {exc}")
        return None


def refresh_ohlcv_cache(symbols: List[str]) -> None:
    """
    Pre-fetch latest bars for all symbols into the committed cache.
    Skips symbols whose committed cache already has data through yesterday.
    No-op when _SKIP_LIVE_FETCH is True (--no-download mode).
    """
    if _SKIP_LIVE_FETCH:
        print("[DATA] --no-download set; using committed cache only")
        return

    yesterday = date.today() - timedelta(days=1)
    n_updated = 0

    print(f"[DATA] Updating committed OHLCV cache for {len(symbols)} symbols...")

    for sym in symbols:
        safe       = safe_sym(sym)
        cache_path = COMMITTED_CACHE / f"{safe}_1d.csv"

        # Skip if cache already covers yesterday
        if cache_path.exists():
            try:
                last_row  = pd.read_csv(cache_path, usecols=["time"]).iloc[-1]["time"]
                last_date = pd.to_datetime(last_row, utc=True).date()
                if last_date >= yesterday:
                    continue
            except Exception:
                pass

        # Clear session cache so load_ohlcv re-fetches
        _SESSION_OHLCV.pop(sym, None)
        df = load_ohlcv(sym)   # triggers live fetch + saves to committed cache
        if not df.empty:
            n_updated += 1
        time.sleep(SLEEP_SEC)

    print(f"[DATA] Updated {n_updated}/{len(symbols)} symbol caches")


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_entry_signal(
    symbol: str,
    df: pd.DataFrame,
    run_date: date,
) -> Optional[dict]:
    """
    Evaluate the closed 1D bar for run_date.

    Entry conditions (both required):
      1. close > don_entry_high  (close breaks prior-20-bar Donchian upper band)
      2. close > ema200          (price above EMA200 -- bull regime filter)

    Returns signal dict or None.
    """
    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    dfi = add_indicators(df)

    day_rows = dfi[dfi["time"].dt.date == run_date]
    if day_rows.empty:
        return None

    row = day_rows.iloc[-1]
    close     = float(row["close"])
    ema200    = float(row["ema200"])
    don_high  = float(row["don_entry_high"])
    don_low   = float(row["don_exit_low"])
    atr       = float(row["atr"])

    if not all(np.isfinite([close, ema200, don_high, atr])):
        return None
    if atr <= EPS or don_high <= EPS:
        return None

    if close > don_high and close > ema200:
        return {
            "symbol":              symbol,
            "signal_date":         str(run_date),
            "close":               round(close, 8),
            "ema200":              round(ema200, 8),
            "don_entry_high":      round(don_high, 8),
            "don_exit_low":        round(don_low, 8) if np.isfinite(don_low) else None,
            "atr":                 round(atr, 8),
            "initial_stop":        round(close - ATR_STOP_MULT * atr, 8),
            "initial_risk_unit":   round(ATR_STOP_MULT * atr, 8),
            "signal":              "BUY",
        }
    return None


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def update_and_check_exit(
    pos: Position,
    df: pd.DataFrame,
    run_date: date,
) -> tuple:
    """
    Process the closed 1D bar for run_date against an open position.

    Exit logic (evaluated in priority order):
      1. Chandelier stop (low <= stop that was set at END OF YESTERDAY)
      2. Donchian N=10 lower band (close < prior-10-bar min low)

    Updates pos in-place for non-exiting bars (highest_high, chandelier state).

    Returns: (exit_record dict | None, updated Position)
    """
    if df.empty or len(df) < ATR_N + DONCHIAN_EXIT_N + 5:
        return None, pos

    dfi = add_indicators(df)

    day_rows = dfi[dfi["time"].dt.date == run_date]
    if day_rows.empty:
        return None, pos

    row      = day_rows.iloc[-1]
    bar_high = float(row["high"])
    bar_low  = float(row["low"])
    bar_close = float(row["close"])
    atr      = float(row["atr"]) if np.isfinite(row["atr"]) else pos.atr_at_entry
    don_exit = float(row["don_exit_low"])

    if atr <= EPS:
        atr = pos.atr_at_entry

    # ---- Stop level in effect at the START of this bar (set by yesterday) ----
    stop_at_bar_open = pos.current_stop

    # ---- Exit checks against yesterday's stop levels -------------------------

    # 1. Chandelier: bar low <= stop that was active at bar open
    if pos.chandelier_active and bar_low <= stop_at_bar_open:
        gross_r  = (stop_at_bar_open - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id":            pos.position_id,
            "symbol":                 pos.symbol,
            "entry_date":             pos.entry_date,
            "exit_date":              str(run_date),
            "entry_price":            pos.entry_price,
            "exit_price":             stop_at_bar_open,
            "initial_stop":           pos.initial_stop,
            "final_stop":             stop_at_bar_open,
            "qty":                    pos.qty,
            "gross_r":                round(gross_r, 6),
            "pnl_usdt":               round(pnl_usdt, 4),
            "max_favorable_r":        pos.max_favorable_r,
            "exit_reason":            "chandelier",
            "chandelier_active":      True,
            "chandelier_activated":   pos.chandelier_activated_date,
        }, pos

    # 2. Donchian N=10 close exit
    if np.isfinite(don_exit) and bar_close < don_exit:
        gross_r  = (bar_close - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id":            pos.position_id,
            "symbol":                 pos.symbol,
            "entry_date":             pos.entry_date,
            "exit_date":              str(run_date),
            "entry_price":            pos.entry_price,
            "exit_price":             bar_close,
            "initial_stop":           pos.initial_stop,
            "final_stop":             pos.current_stop,
            "qty":                    pos.qty,
            "gross_r":                round(gross_r, 6),
            "pnl_usdt":               round(pnl_usdt, 4),
            "max_favorable_r":        pos.max_favorable_r,
            "exit_reason":            "donchian_exit",
            "chandelier_active":      pos.chandelier_active,
            "chandelier_activated":   pos.chandelier_activated_date,
        }, pos

    # ---- No exit: update position state for TOMORROW -------------------------

    # Update highest high
    pos.highest_high_since_entry = max(pos.highest_high_since_entry, bar_high)

    # Update MFE (in R)
    mfe_r = (pos.highest_high_since_entry - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
    pos.max_favorable_r = max(pos.max_favorable_r, mfe_r)

    # Ratchet chandelier stop if MFE has reached activation threshold
    if pos.max_favorable_r >= CHANDELIER_ACTIVATE_R:
        new_ch_stop = pos.highest_high_since_entry - CHANDELIER_MULT * atr
        if not pos.chandelier_active:
            pos.chandelier_active         = True
            pos.chandelier_activated_date = str(run_date)
        # Chandelier only ratchets up -- never down
        pos.current_stop = max(pos.current_stop, new_ch_stop)

    return None, pos


# =============================================================================
# DAILY RUN
# =============================================================================

def run_one_day(
    run_date: date,
    symbols: List[str],
    state: dict,
) -> dict:
    """
    Run one calendar day of paper trading.
    Exits are processed before new entries (same-bar sequencing).
    Returns result dict consumed by output writers and console.
    """
    equity    = float(state["paper_equity_usdt"])
    peak      = float(state["peak_equity_usdt"])
    kill_sw   = bool(state.get("kill_switch_triggered", False))
    positions = positions_from_state(state)

    log_events:   List[dict] = []
    signals_seen: List[dict] = []
    exits_today:  List[dict] = []

    def _log(event: str, symbol: str, detail: str, **extra):
        log_events.append({
            "run_date": str(run_date),
            "event":    event,
            "symbol":   symbol,
            "detail":   detail,
            **extra,
        })

    # ------------------------------------------------------------------
    # STEP 1: Process exits on all open positions
    # ------------------------------------------------------------------
    closed_ids: List[str] = []

    for pid, pos in list(positions.items()):
        df = load_ohlcv(pos.symbol, up_to_date=run_date)
        exit_rec, pos = update_and_check_exit(pos, df, run_date)

        if exit_rec is not None:
            equity += exit_rec["pnl_usdt"]
            peak    = max(peak, equity)
            dd_pct  = (equity - peak) / max(peak, EPS) * 100.0

            exit_rec["equity_after"] = round(equity, 4)
            exit_rec["dd_pct"]       = round(dd_pct, 4)
            exits_today.append(exit_rec)
            closed_ids.append(pid)

            _log(
                "EXIT", pos.symbol,
                f"reason={exit_rec['exit_reason']}  "
                f"R={exit_rec['gross_r']:.3f}  "
                f"pnl=${exit_rec['pnl_usdt']:.2f}  "
                f"equity=${equity:,.2f}",
                gross_r=exit_rec["gross_r"],
                pnl_usdt=exit_rec["pnl_usdt"],
                equity=round(equity, 2),
                dd_pct=round(dd_pct, 2),
            )

            if dd_pct <= -KILL_SWITCH_DD_PCT:
                kill_sw = True
                _log("KILL_SWITCH", "",
                     f"DD={dd_pct:.2f}% breached -{KILL_SWITCH_DD_PCT}% threshold",
                     equity=round(equity, 2))
        else:
            # Update chandelier / highest_high state for non-exiting positions
            positions[pid] = pos

    for pid in closed_ids:
        positions.pop(pid, None)

    # ------------------------------------------------------------------
    # STEP 2: Scan for new entry signals
    # ------------------------------------------------------------------
    if not kill_sw:
        for sym in symbols:
            # Skip if already holding this symbol
            if any(p.symbol == sym for p in positions.values()):
                continue

            df  = load_ohlcv(sym, up_to_date=run_date)
            sig = check_entry_signal(sym, df, run_date)

            if sig is None:
                continue

            signals_seen.append(sig)

            if len(positions) >= MAX_OPEN_POSITIONS:
                _log(
                    "SIGNAL_SKIPPED", sym,
                    f"portfolio cap {MAX_OPEN_POSITIONS} reached  "
                    f"(close={sig['close']:.6g})",
                    close=sig["close"],
                )
                continue

            # Accept entry
            risk_amount    = equity * RISK_PER_TRADE_PCT
            risk_per_unit  = sig["initial_risk_unit"]
            qty            = risk_amount / max(risk_per_unit, EPS)
            entry_price    = sig["close"]
            initial_stop   = sig["initial_stop"]

            pid = f"{safe_sym(sym)}_{run_date.strftime('%Y%m%d')}"
            pos = Position(
                position_id               = pid,
                symbol                    = sym,
                entry_date                = str(run_date),
                entry_price               = entry_price,
                initial_stop              = initial_stop,
                current_stop              = initial_stop,
                initial_risk_per_unit     = risk_per_unit,
                risk_amount_usdt          = risk_amount,
                qty                       = qty,
                highest_high_since_entry  = entry_price,
                max_favorable_r           = 0.0,
                chandelier_active         = False,
                chandelier_activated_date = None,
                atr_at_entry              = sig["atr"],
                donchian_entry_level      = sig["don_entry_high"],
            )
            positions[pid] = pos

            _log(
                "ENTRY", sym,
                f"price={entry_price:.6g}  "
                f"stop={initial_stop:.6g}  "
                f"risk=${risk_amount:.2f}  "
                f"qty={qty:.8g}  "
                f"equity=${equity:,.2f}",
                entry_price=round(entry_price, 8),
                initial_stop=round(initial_stop, 8),
                risk_amount=round(risk_amount, 4),
                equity=round(equity, 2),
            )

    # ------------------------------------------------------------------
    # STEP 3: Commit state
    # ------------------------------------------------------------------
    dd_now = (equity - peak) / max(peak, EPS) * 100.0

    state["last_run_date"]         = str(run_date)
    state["paper_equity_usdt"]     = equity
    state["peak_equity_usdt"]      = peak
    state["drawdown_pct"]          = round(dd_now, 4)
    state["kill_switch_triggered"] = kill_sw
    state["closed_trade_count"]    = state.get("closed_trade_count", 0) + len(exits_today)

    positions_to_state(state, positions)

    return {
        "run_date":      str(run_date),
        "equity":        equity,
        "peak":          peak,
        "dd_pct":        dd_now,
        "open_count":    len(positions),
        "new_signals":   len(signals_seen),
        "new_entries":   sum(1 for e in log_events if e["event"] == "ENTRY"),
        "exits":         len(exits_today),
        "kill_switch":   kill_sw,
        "log_events":    log_events,
        "signals_today": signals_seen,
        "exits_today":   exits_today,
        "positions":     positions,
    }


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_open_positions(positions: Dict[str, Position], run_date: date) -> None:
    if not positions:
        pd.DataFrame(columns=[
            "position_id", "symbol", "entry_date", "as_of_date",
            "entry_price", "initial_stop", "current_stop",
            "max_favorable_r", "chandelier_active",
        ]).to_csv(OPEN_POS_CSV, index=False)
        return

    rows = []
    for pos in positions.values():
        rows.append({
            "position_id":            pos.position_id,
            "symbol":                 pos.symbol,
            "entry_date":             pos.entry_date,
            "as_of_date":             str(run_date),
            "entry_price":            pos.entry_price,
            "initial_stop":           pos.initial_stop,
            "current_stop":           pos.current_stop,
            "initial_risk_per_unit":  pos.initial_risk_per_unit,
            "risk_amount_usdt":       round(pos.risk_amount_usdt, 4),
            "qty":                    pos.qty,
            "atr_at_entry":           pos.atr_at_entry,
            "donchian_entry_level":   pos.donchian_entry_level,
            "highest_high":           pos.highest_high_since_entry,
            "max_favorable_r":        round(pos.max_favorable_r, 4),
            "chandelier_active":      pos.chandelier_active,
            "chandelier_activated":   pos.chandelier_activated_date,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    if not signals:
        pd.DataFrame(columns=[
            "symbol", "signal_date", "close", "ema200",
            "don_entry_high", "atr", "initial_stop", "signal",
        ]).to_csv(SIGNALS_CSV, index=False)
        return
    pd.DataFrame(signals).to_csv(SIGNALS_CSV, index=False)


def append_equity_curve(exits: List[dict]) -> None:
    if not exits:
        return
    rows = [{
        "exit_date":      ex["exit_date"],
        "symbol":         ex["symbol"],
        "entry_date":     ex["entry_date"],
        "gross_r":        ex["gross_r"],
        "pnl_usdt":       ex["pnl_usdt"],
        "equity":         ex["equity_after"],
        "dd_pct":         ex["dd_pct"],
        "exit_reason":    ex["exit_reason"],
        "max_favorable_r": ex["max_favorable_r"],
    } for ex in exits]
    append_csv(EQUITY_CSV, rows)


def append_daily_log(events: List[dict]) -> None:
    if events:
        append_csv(DAILY_LOG_CSV, events)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_banner() -> None:
    print("=" * 70)
    print("T9B -- DONCHIAN UNIVERSEV2 EXITV2 PAPER ENGINE")
    print("=" * 70)
    print(f"Frozen config (2026-05-30):")
    print(f"  1D / EMA200-price / Donchian N=20 entry / N=10 exit")
    print(f"  Chandelier ACT={CHANDELIER_ACTIVATE_R}R trail={CHANDELIER_MULT}xATR")
    print(f"  Max {MAX_OPEN_POSITIONS} positions / {RISK_PER_TRADE_PCT*100:.2f}% risk / Leverage {LEVERAGE}x")
    print(f"  24 symbols / Binance Spot / LONG only")
    print(f"Output: {DATA_DIR}")
    print()


def print_day_summary(result: dict, verbose: bool = True) -> None:
    d   = result["run_date"]
    eq  = result["equity"]
    dd  = result["dd_pct"]
    op  = result["open_count"]
    si  = result["new_signals"]
    en  = result["new_entries"]
    ex  = result["exits"]
    ks  = " [KILL-SWITCH]" if result["kill_switch"] else ""
    print(
        f"  {d}  equity=${eq:>10,.2f}  DD={dd:>+6.2f}%  "
        f"open={op}  signals={si}  entries={en}  exits={ex}{ks}"
    )
    if verbose:
        for e in result["log_events"]:
            if e["event"] in ("ENTRY", "EXIT", "KILL_SWITCH", "SIGNAL_SKIPPED"):
                sym = e["symbol"] if e["symbol"] else ""
                print(f"    [{e['event']:<14}] {sym:<16} {e['detail']}")


def print_final_summary(state: dict) -> None:
    eq  = float(state["paper_equity_usdt"])
    ret = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    dd  = float(state["drawdown_pct"])
    op  = len(state.get("open_positions", []))
    cl  = state.get("closed_trade_count", 0)
    print()
    print("-" * 70)
    print("FINAL STATE")
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Total return   : {ret:>+.2f}%")
    print(f"  Max drawdown   : {dd:>+.2f}%")
    print(f"  Open positions : {op}")
    print(f"  Closed trades  : {cl}")
    print(f"  Kill-switch    : {'TRIGGERED' if state.get('kill_switch_triggered') else 'not triggered'}")
    print()
    print(f"  open_positions.csv : {OPEN_POS_CSV}")
    print(f"  signals_today.csv  : {SIGNALS_CSV}")
    print(f"  equity_curve.csv   : {EQUITY_CSV}")
    print(f"  daily_log.csv      : {DAILY_LOG_CSV}")
    print()
    print("[PAPER ONLY] No real orders were placed or submitted.")


# =============================================================================
# UNIVERSE
# =============================================================================

def load_universe() -> List[str]:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_CSV}")
    df = pd.read_csv(UNIVERSE_CSV)
    if "symbol" not in df.columns:
        raise ValueError(f"No 'symbol' column in {UNIVERSE_CSV}")
    syms = df["symbol"].dropna().tolist()
    print(f"[UNIVERSE] {len(syms)} symbols from {UNIVERSE_CSV.name}")
    return syms


# =============================================================================
# NOTIFY OUTPUT  (compact, phone-readable, GitHub Actions log friendly)
# =============================================================================

def _print_notify(state: dict, run_date: date) -> None:
    """
    Prints a compact, phone-readable summary after the daily run.
    Used by GitHub Actions to confirm the run completed and surface key info.
    All output is ASCII.
    """
    eq        = float(state.get("paper_equity_usdt", INITIAL_CAPITAL))
    peak      = float(state.get("peak_equity_usdt", INITIAL_CAPITAL))
    dd        = float(state.get("drawdown_pct", 0.0))
    ret_pct   = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    n_open    = len(state.get("open_positions", []))
    n_closed  = int(state.get("closed_trade_count", 0))
    kill_sw   = state.get("kill_switch_triggered", False)

    days_running  = (run_date - FREEZE_DATE).days + 1
    review_date   = FREEZE_DATE + timedelta(days=90)
    days_to_review = (review_date - run_date).days

    print()
    print("=" * 50)
    print(f"T9B DAILY UPDATE -- {run_date}")
    print("=" * 50)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)")
    print(f"Peak equity   : ${peak:>10,.2f}")
    print(f"Max drawdown  : {dd:>+.2f}%")
    print(f"Open positions: {n_open}")
    print(f"Closed trades : {n_closed}")
    print(f"Days running  : {days_running}")
    print(f"Days to review: {days_to_review}  (3-month live review: {review_date})")
    if kill_sw:
        print("KILL-SWITCH   : TRIGGERED -- no new entries")
    print()

    # Open positions
    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"  {'Symbol':<16} {'Entry':<12} {'Stop':<12} {'MaxR':>6}  {'Chandelier'}")
        print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*6}  {'-'*10}")
        for p in positions:
            sym   = str(p.get("symbol", "?"))
            edate = str(p.get("entry_date", "?"))
            stop  = p.get("current_stop", 0)
            maxr  = float(p.get("max_favorable_r", 0))
            ch    = "yes" if p.get("chandelier_active") else "no"
            print(f"  {sym:<16} {edate:<12} {stop:<12.6g} {maxr:>+6.2f}R  ch={ch}")
    else:
        print("OPEN POSITIONS: none")
    print()

    # Signals today
    signals_path = SIGNALS_CSV
    if signals_path.exists():
        try:
            sig_df = pd.read_csv(signals_path)
            if not sig_df.empty and len(sig_df) > 0:
                print(f"SIGNALS TODAY ({len(sig_df)})")
                for _, row in sig_df.iterrows():
                    sym   = str(row.get("symbol", "?"))
                    close = row.get("close", "?")
                    stop  = row.get("initial_stop", "?")
                    print(f"  {sym:<16}  close={close:<12.6g}  stop={stop:<12.6g}")
            else:
                print("SIGNALS TODAY: none")
        except Exception:
            print("SIGNALS TODAY: (could not read)")
    else:
        print("SIGNALS TODAY: none")

    print()
    if kill_sw:
        print("STATUS: KILL-SWITCH ACTIVE")
    elif dd < -20.0:
        print("STATUS: WARN -- drawdown > 20%")
    else:
        print("STATUS: OK")
    print("=" * 50)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T9B Donchian UniverseV2 ExitV2 Paper Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py --date 2026-05-31\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py --backfill\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py --backfill --no-download\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py --notify\n"
            "  python phase_t9b_donchian_universev2_paper_engine.py --reset\n"
        ),
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Run for a specific date (default: yesterday -- most recent closed 1D bar)",
    )
    grp.add_argument(
        "--backfill", action="store_true",
        help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip Binance OHLCV cache refresh (use cached/existing files only)",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Delete state.json and restart paper equity from $10,000",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-event detail (show only daily summary lines)",
    )
    p.add_argument(
        "--notify", action="store_true",
        help="Print a compact notification-style summary after the run (for CI/automation)",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    print_banner()

    # Reset
    if args.reset:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        for f in [EQUITY_CSV, DAILY_LOG_CSV]:
            if f.exists():
                f.unlink()
        print("[RESET] State and log files cleared. Starting from $10,000.")
        print()

    # Universe
    symbols = load_universe()

    # Determine run dates.
    # Default to yesterday -- the most recently closed 1D bar.
    # The daily GitHub Actions workflow fires at 08:00 UTC, 8 hours after yesterday's candle closed.
    today     = date.today()
    yesterday = today - timedelta(days=1)

    if args.backfill:
        run_dates: List[date] = []
        d = FREEZE_DATE
        while d <= yesterday:
            run_dates.append(d)
            d += timedelta(days=1)
        print(f"[BACKFILL] {len(run_dates)} days: {FREEZE_DATE} to {yesterday}")
    elif args.date:
        run_dates = [date.fromisoformat(args.date)]
        print(f"[MODE] Single date: {args.date}")
    else:
        run_dates = [yesterday]
        print(f"[MODE] Yesterday (default): {yesterday}")

    # Propagate --no-download flag to the global fetch-control
    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    # Refresh committed cache (appends latest bars via yfinance)
    refresh_ohlcv_cache(symbols)

    print()

    # Load state
    state = load_state()
    n_open   = len(state.get("open_positions", []))
    n_closed = state.get("closed_trade_count", 0)
    print(f"[STATE] equity=${state['paper_equity_usdt']:,.2f}  "
          f"open={n_open}  closed={n_closed}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}")
    print()

    # Daily loop
    for run_date in run_dates:
        result = run_one_day(run_date, symbols, state)
        save_state(state)

        write_open_positions(result["positions"], run_date)
        write_signals_today(result["signals_today"], run_date)
        append_equity_curve(result["exits_today"])
        append_daily_log(result["log_events"])

        print_day_summary(result, verbose=verbose)

        if result["kill_switch"]:
            print()
            print("[HALT] Kill-switch triggered. Stopping run.")
            print("       No new entries will be accepted until state is reviewed.")
            break

    print_final_summary(state)

    if args.notify:
        _print_notify(state, run_dates[-1] if run_dates else yesterday)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
