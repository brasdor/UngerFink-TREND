#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- CONSECDOWNDAYSMR 1D PAPER ENGINE
===============================================

Paper trading engine for the frozen ConsecDownDaysMR 1D config.

Frozen config (2026-06-02):
  Universe    : 52-symbol crypto universe
  Timeframe   : 1D
  Filter      : ema200_price_above  (price must be above EMA200 -- mandatory)
  Entry       : 5 consecutive down closes (close[i] < close[i-1])
                AND close > EMA(200) on signal bar -- enter LONG at signal close
  Safety stop : ATR(14) x 2.0 below entry close  (safety net only)
  Exit        : Fixed time exit after 20 bars  (Variant E)
  Max pos     : uncapped  (system rarely fires >5 simultaneous)
  Risk/trade  : 0.25% of current equity
  Leverage    : 1.0x  (Binance Spot, LONG only)
  Character   : BULL MARKET SPECIALIST (pairs with RSI MR for all-weather coverage)

OHLCV data strategy (geo-unrestricted, works from GitHub Actions):
  1. Load committed historical cache: data/universe/ohlcv_1d/{sym}_1d.csv
     Tracked in git, same cache shared with Donchian and RSI MR T9B engines.
  2. Append latest bars via yfinance (geo-unrestricted, works on GitHub Actions).
     Binance.com returns HTTP 451 from US IPs; ccxt/binanceus also fails.
  3. Commit updated files back via git in the workflow.
  4. --no-download: skip live fetch, use committed cache only.

  Binance.com (global) is NOT used -- HTTP 451 from US-based servers.

Usage:
  python phase_t9b_consecdowndays_paper_engine.py
      --date 2026-06-02    run for a specific date
      --backfill           replay from freeze date (2026-06-02) to yesterday
      --no-download        use committed cache only
      --reset              wipe state and restart from $10,000
      --notify             compact notification summary (for CI)

Output files (data/t9b_consecdowndays_paper/):
  state.json            persistent engine state
  open_positions.csv    current open positions with bars_held, stop, consec_at_entry
  signals_today.csv     consec-down signals detected today
  equity_curve.csv      one row per closed trade
  daily_log.csv         all events: entries, exits, skipped signals

PAPER ONLY -- NO REAL ORDERS.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import t9b_shared
from signal_arbitrator import SignalArbitrator


# =============================================================================
# FROZEN CONFIG  (2026-06-02)
# =============================================================================

SYSTEM_NAME         = "ConsecDownDaysMR_1D_T9B"
FREEZE_DATE         = date(2026, 6, 2)

CONSEC_N            = 5       # entry requires 5 consecutive down closes
EMA_N               = 200     # EMA200 filter -- mandatory
ATR_N               = 14
ATR_STOP_MULT       = 2.0     # safety stop only -- primary exit is always time
TIME_EXIT_BARS      = 20      # exit after exactly 20 daily bars

RISK_PER_TRADE_PCT  = 0.0025  # 0.25% of current equity
MIN_ORDER_SIZE_USDT = 15.0    # Fix 2: minimum position notional
MAX_OPEN_POSITIONS  = None    # uncapped (system rarely fires >5 simultaneous)
INITIAL_CAPITAL     = 10_000.0
LEVERAGE            = 1.0
KILL_SWITCH_DD_PCT  = 35.0    # halt new entries if DD from peak exceeds this

LIMIT_BARS          = 2000    # bars to load per symbol
SLEEP_SEC           = 0.15
MAX_RETRIES         = 3
MIN_BARS_REQUIRED   = EMA_N + CONSEC_N + 10   # ~215 bars needed

EPS = 1e-12


# =============================================================================
# PATHS
# =============================================================================

ROOT            = Path.cwd()
DATA_DIR        = ROOT / "data" / "t9b_consecdowndays_paper"
STATE_PATH      = DATA_DIR / "state.json"
OPEN_POS_CSV    = DATA_DIR / "open_positions.csv"
SIGNALS_CSV     = DATA_DIR / "signals_today.csv"
EQUITY_CSV      = DATA_DIR / "equity_curve.csv"
DAILY_LOG_CSV   = DATA_DIR / "daily_log.csv"

# Shared committed OHLCV cache (also used by Donchian + RSI MR T9B, tracked in git)
COMMITTED_CACHE = ROOT / "data" / "universe" / "ohlcv_1d"

DATA_DIR.mkdir(parents=True, exist_ok=True)
COMMITTED_CACHE.mkdir(parents=True, exist_ok=True)

_SESSION_OHLCV:   dict = {}
_SKIP_LIVE_FETCH: bool = False


# =============================================================================
# UNIVERSE  (52-symbol original crypto universe used throughout T1-T8)
# =============================================================================

CD_UNIVERSE = [
    "AAVE/USDT", "ADA/USDT",  "ALT/USDT",  "APT/USDT",  "ARB/USDT",
    "ARKM/USDT","ASTER/USDT","ATOM/USDT", "AVAX/USDT", "BCH/USDT",
    "BNB/USDT", "BTC/USDT",  "CHZ/USDT",  "DASH/USDT", "DOGE/USDT",
    "DOT/USDT", "EIGEN/USDT","ENA/USDT",  "ETH/USDT",  "FET/USDT",
    "FIL/USDT", "GRT/USDT",  "HBAR/USDT", "ICP/USDT",  "INJ/USDT",
    "JTO/USDT", "LINK/USDT", "LPT/USDT",  "LTC/USDT",  "MORPHO/USDT",
    "NEAR/USDT","NIL/USDT",  "ONDO/USDT", "ORDI/USDT", "PENDLE/USDT",
    "PENGU/USDT","PEPE/USDT","RENDER/USDT","SAGA/USDT","SEI/USDT",
    "SOL/USDT", "SPK/USDT",  "SUI/USDT",  "TAO/USDT",  "TIA/USDT",
    "TON/USDT", "TRX/USDT",  "UNI/USDT",  "WLD/USDT",  "XRP/USDT",
    "ZEC/USDT", "ZEN/USDT",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CDMRPosition:
    position_id:           str
    symbol:                str
    entry_date:            str    # ISO date string
    entry_price:           float
    stop_loss:             float  # ATR x 2.0 below entry (safety net)
    initial_risk_per_unit: float  # ATR x 2.0 at entry
    risk_amount_usdt:      float  # equity x 0.25% at entry
    qty:                   float
    bars_held:             int    # incremented each daily bar processed
    atr_at_entry:          float
    consec_at_entry:       int    # consecutive down closes at signal bar

    def current_r(self, price: float) -> float:
        return (price - self.entry_price) / max(self.initial_risk_per_unit, EPS)


# =============================================================================
# STATE HELPERS
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


def positions_from_state(state: dict) -> Dict[str, CDMRPosition]:
    fields = {f.name for f in dataclasses.fields(CDMRPosition)}
    out: Dict[str, CDMRPosition] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = CDMRPosition(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping position {raw.get('position_id','?')}: {exc}")
    return out


def positions_to_state(state: dict, positions: Dict[str, CDMRPosition]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def _reconcile_state(state: dict, run_date: date) -> None:
    """Fix 3: Verify open positions consistency on engine startup."""
    positions = state.get("open_positions", [])
    issues: list[str] = []
    for pos in positions:
        pid  = pos.get("position_id", "?")
        sym  = pos.get("symbol", "")
        ep   = pos.get("entry_price")
        qty  = pos.get("qty")
        stop = pos.get("stop_loss")
        bh   = pos.get("bars_held")
        ed   = pos.get("entry_date")
        if not sym:
            issues.append(f"{pid}: missing symbol")
        try:
            if float(ep) <= 0:
                issues.append(f"{sym}: entry_price={ep} <= 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: entry_price not numeric ({ep!r})")
        try:
            if float(qty) <= 0:
                issues.append(f"{sym}: qty={qty} <= 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: qty not numeric ({qty!r})")
        try:
            if stop is not None and ep is not None and float(stop) >= float(ep):
                issues.append(f"{sym}: stop={stop} >= entry_price={ep} (invalid for LONG)")
        except (TypeError, ValueError):
            pass
        try:
            if int(bh) < 0:
                issues.append(f"{sym}: bars_held={bh} < 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: bars_held not numeric ({bh!r})")
        try:
            date.fromisoformat(str(ed))
        except (ValueError, TypeError):
            issues.append(f"{sym}: invalid entry_date={ed!r}")

    if issues:
        print(f"[RECONCILE] {len(issues)} issue(s) in open positions:")
        for iss in issues:
            print(f"  [WARN] {iss}")
        rows = [{"run_date": str(run_date), "event": "RECONCILE_WARN",
                 "symbol": "", "detail": iss} for iss in issues]
        append_csv(DAILY_LOG_CSV, rows)
    else:
        print(f"[RECONCILE] OK  ({len(positions)} position(s) verified)")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


# =============================================================================
# OHLCV LOADING  (identical strategy to Donchian + RSI MR T9B)
# =============================================================================

def load_ohlcv(symbol: str, up_to_date: Optional[date] = None) -> pd.DataFrame:
    global _SESSION_OHLCV

    if symbol not in _SESSION_OHLCV:
        safe       = safe_sym(symbol)
        cache_path = COMMITTED_CACHE / f"{safe}_1d.csv"

        raw_base: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                raw_base = pd.read_csv(cache_path)
            except Exception as exc:
                print(f"  [WARN] {symbol}: cache read error -- {exc}")

        raw_live: Optional[pd.DataFrame] = None
        if not _SKIP_LIVE_FETCH:
            raw_live = _fetch_latest_bars(symbol, n_bars=10)

        df_base = _parse_ohlcv(raw_base) if raw_base is not None else pd.DataFrame()
        df_live = _parse_ohlcv(raw_live) if raw_live is not None else pd.DataFrame()

        if not df_base.empty and not df_live.empty:
            merged = (pd.concat([df_base, df_live], ignore_index=True)
                      .drop_duplicates("time").sort_values("time").reset_index(drop=True))
        elif not df_base.empty:
            merged = df_base
        elif not df_live.empty:
            merged = df_live
        else:
            merged = pd.DataFrame()

        if not merged.empty and raw_live is not None and cache_path.parent.exists():
            _save_committed_cache(merged, cache_path)

        if len(merged) > LIMIT_BARS:
            merged = merged.iloc[-LIMIT_BARS:].reset_index(drop=True)

        _SESSION_OHLCV[symbol] = merged

    df = _SESSION_OHLCV[symbol]
    if up_to_date is not None:
        df = df[df["time"].dt.date <= up_to_date].copy()
    return df.reset_index(drop=True)


def _save_committed_cache(df: pd.DataFrame, path: Path) -> None:
    out = df[["time", "open", "high", "low", "close", "volume"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(path, index=False)


def _parse_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"),
            unit="ms", utc=True, errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates("time").sort_values("time")
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _fetch_latest_bars(symbol: str, n_bars: int = 10) -> Optional[pd.DataFrame]:
    """yfinance only -- geo-unrestricted, works from GitHub Actions US servers."""
    return _try_yfinance(symbol, n_bars)


def _try_yfinance(symbol: str, n_bars: int) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        return None
    base   = symbol.split("/")[0]
    yf_sym = f"{base}-USD"
    try:
        hist = yf.download(yf_sym, period="30d", interval="1d",
                           progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        hist = hist.tail(n_bars).reset_index()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() for c in hist.columns]
        else:
            hist.columns = [str(c).lower() for c in hist.columns]
        if "date" in hist.columns:
            hist.rename(columns={"date": "time"}, inplace=True)
        elif "datetime" in hist.columns:
            hist.rename(columns={"datetime": "time"}, inplace=True)
        hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")
        hist["timestamp"] = (hist["time"].astype(np.int64) // 1_000_000).astype(int)
        for col in ["open","high","low","close","volume"]:
            if col not in hist.columns:
                hist[col] = np.nan
        return hist[["timestamp","open","high","low","close","volume"]].copy()
    except Exception as e:
        print(f"    [yfinance] {symbol}: {e}")
        return None


def refresh_ohlcv_cache(symbols: List[str]) -> None:
    if _SKIP_LIVE_FETCH:
        print("[DATA] --no-download: using committed cache only")
        return
    yesterday = date.today() - timedelta(days=1)
    n_updated = 0
    print(f"[DATA] Refreshing OHLCV cache for {len(symbols)} symbols...")
    for sym in symbols:
        safe       = safe_sym(sym)
        cache_path = COMMITTED_CACHE / f"{safe}_1d.csv"
        if cache_path.exists():
            try:
                last_row  = pd.read_csv(cache_path, usecols=["time"]).iloc[-1]["time"]
                last_date = pd.to_datetime(last_row, utc=True).date()
                if last_date >= yesterday:
                    continue
            except Exception:
                pass
        _SESSION_OHLCV.pop(sym, None)
        df = load_ohlcv(sym)
        if not df.empty:
            n_updated += 1
        time.sleep(SLEEP_SEC)
    print(f"[DATA] Updated {n_updated}/{len(symbols)} caches")


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out  = df.copy()
    cls  = out["close"]
    hgh  = out["high"]
    lw   = out["low"]

    # ATR(14) -- simple rolling
    prev_c = cls.shift(1)
    tr = pd.concat([hgh - lw, (hgh - prev_c).abs(), (lw - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_N).mean()

    # EMA(200)
    out["ema200"] = cls.ewm(span=EMA_N, adjust=False).mean()

    # Consecutive down closes: count of successive bars where close < prev_close
    close_arr = cls.values
    consec    = [0] * len(close_arr)
    cnt       = 0
    for i in range(1, len(close_arr)):
        if close_arr[i] < close_arr[i - 1]:
            cnt += 1
        else:
            cnt = 0
        consec[i] = cnt
    out["consec_down"] = consec

    return out


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_entry_signal(symbol: str, df: pd.DataFrame, run_date: date) -> Optional[dict]:
    """
    Entry conditions (both required):
      1. consec_down >= CONSEC_N  (5 consecutive closes, each < prior close)
      2. close > EMA200           (bull market filter -- mandatory)

    Enter at close of signal bar.
    """
    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    dfi      = add_indicators(df)
    day_rows = dfi[dfi["time"].dt.date == run_date]
    if day_rows.empty:
        return None

    row       = day_rows.iloc[-1]
    close     = float(row["close"])
    atr       = float(row["atr"])
    ema200    = float(row["ema200"])
    consec    = int(row["consec_down"])

    if not all(np.isfinite([close, atr, ema200])) or atr <= EPS:
        return None

    # Both conditions required
    if consec >= CONSEC_N and close > ema200:
        stop = close - ATR_STOP_MULT * atr
        return {
            "symbol":      symbol,
            "signal_date": str(run_date),
            "close":       round(close, 8),
            "consec":      consec,
            "atr":         round(atr, 8),
            "ema200":      round(ema200, 8),
            "stop_loss":   round(stop, 8),
            "risk_unit":   round(ATR_STOP_MULT * atr, 8),
            "signal":      "BUY_CD",
        }
    return None


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def update_and_check_exit(
    pos: CDMRPosition,
    df: pd.DataFrame,
    run_date: date,
) -> tuple:
    """
    Exit logic (priority order):
      1. Safety stop : bar low <= pos.stop_loss  -> exit at stop_loss price
      2. Time exit   : pos.bars_held >= TIME_EXIT_BARS -> exit at bar close

    Non-exiting bar: increment bars_held.
    """
    if df.empty:
        return None, pos

    day_rows = df[df["time"].dt.date == run_date]
    if day_rows.empty:
        return None, pos

    row       = day_rows.iloc[-1]
    bar_low   = float(row["low"])
    bar_close = float(row["close"])

    # 1. Safety stop
    if bar_low <= pos.stop_loss:
        gross_r  = (pos.stop_loss - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id":   pos.position_id,
            "symbol":        pos.symbol,
            "entry_date":    pos.entry_date,
            "exit_date":     str(run_date),
            "entry_price":   pos.entry_price,
            "exit_price":    pos.stop_loss,
            "stop_loss":     pos.stop_loss,
            "bars_held":     pos.bars_held,
            "gross_r":       round(gross_r, 6),
            "pnl_usdt":      round(pnl_usdt, 4),
            "exit_reason":   "safety_stop",
        }, pos

    # 2. Time exit (primary -- Variant E)
    if pos.bars_held >= TIME_EXIT_BARS:
        gross_r  = (bar_close - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id":   pos.position_id,
            "symbol":        pos.symbol,
            "entry_date":    pos.entry_date,
            "exit_date":     str(run_date),
            "entry_price":   pos.entry_price,
            "exit_price":    bar_close,
            "stop_loss":     pos.stop_loss,
            "bars_held":     pos.bars_held,
            "gross_r":       round(gross_r, 6),
            "pnl_usdt":      round(pnl_usdt, 4),
            "exit_reason":   "time_exit_20bars",
        }, pos

    # No exit this bar
    pos.bars_held += 1
    return None, pos


# =============================================================================
# DAILY RUN
# =============================================================================

def run_one_day(run_date: date, symbols: List[str], state: dict) -> dict:
    equity    = float(state["paper_equity_usdt"])
    peak      = float(state["peak_equity_usdt"])
    kill_sw   = bool(state.get("kill_switch_triggered", False))
    positions = positions_from_state(state)

    log_events:   List[dict] = []
    signals_seen: List[dict] = []
    exits_today:  List[dict] = []

    def _log(event: str, symbol: str, detail: str, **extra):
        log_events.append({"run_date": str(run_date), "event": event,
                           "symbol": symbol, "detail": detail, **extra})

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
            _log("EXIT", pos.symbol,
                 f"reason={exit_rec['exit_reason']}  bars={exit_rec['bars_held']}  "
                 f"R={exit_rec['gross_r']:.3f}  pnl=${exit_rec['pnl_usdt']:.2f}  "
                 f"equity=${equity:,.2f}",
                 gross_r=exit_rec["gross_r"], pnl_usdt=exit_rec["pnl_usdt"],
                 equity=round(equity, 2), dd_pct=round(dd_pct, 2))
            if dd_pct <= -KILL_SWITCH_DD_PCT:
                kill_sw = True
                _log("KILL_SWITCH", "",
                     f"DD={dd_pct:.2f}% breached -{KILL_SWITCH_DD_PCT}% threshold",
                     equity=round(equity, 2))
        else:
            positions[pid] = pos

    for pid in closed_ids:
        positions.pop(pid, None)

    # ------------------------------------------------------------------
    # STEP 2: Scan for new entry signals (uncapped)
    # ------------------------------------------------------------------
    if not kill_sw:
        cross_syms = t9b_shared.get_cross_system_symbols('consecdown')  # Fix 1
        arbitrator = SignalArbitrator('consecdown', run_date)
        for sym in symbols:
            # One position per symbol at a time
            if any(p.symbol == sym for p in positions.values()):
                continue

            # Fix 1: cross-system duplicate check
            if t9b_shared.normalize_sym(sym) in cross_syms:
                _log("SIGNAL_SKIPPED", sym, "duplicate_cross_system")
                continue

            df  = load_ohlcv(sym, up_to_date=run_date)
            sig = check_entry_signal(sym, df, run_date)
            if sig is None:
                continue

            signals_seen.append(sig)

            # Cap check (None = uncapped)
            if MAX_OPEN_POSITIONS is not None and len(positions) >= MAX_OPEN_POSITIONS:
                _log("SIGNAL_SKIPPED", sym,
                     f"cap {MAX_OPEN_POSITIONS} reached  "
                     f"(consec={sig['consec']}  close={sig['close']:.6g})",
                     close=sig["close"])
                continue

            risk_amount   = equity * RISK_PER_TRADE_PCT
            risk_per_unit = sig["risk_unit"]
            qty           = risk_amount / max(risk_per_unit, EPS)

            # Fix 2: minimum order size validation
            if qty * sig["close"] < MIN_ORDER_SIZE_USDT:
                _log("SIGNAL_SKIPPED", sym,
                     f"position_too_small  size=${qty * sig['close']:.2f} < ${MIN_ORDER_SIZE_USDT}",
                     close=sig["close"])
                continue

            # Signal Arbitration Manager (Rules 1-6)
            decision, arb_reason = arbitrator.check_signal(sym, "LONG", risk_amount)
            if decision == "REJECTED":
                _log("SIGNAL_SKIPPED", sym, f"arbitrator: {arb_reason}",
                     arbitrator_reject_reason=arb_reason)
                continue

            pid           = f"{safe_sym(sym)}_{run_date.strftime('%Y%m%d')}"

            pos = CDMRPosition(
                position_id           = pid,
                symbol                = sym,
                entry_date            = str(run_date),
                entry_price           = sig["close"],
                stop_loss             = sig["stop_loss"],
                initial_risk_per_unit = risk_per_unit,
                risk_amount_usdt      = risk_amount,
                qty                   = qty,
                bars_held             = 0,
                atr_at_entry          = sig["atr"],
                consec_at_entry       = sig["consec"],
            )
            positions[pid] = pos

            _log("ENTRY", sym,
                 f"consec={sig['consec']}  price={sig['close']:.6g}  "
                 f"ema200={sig['ema200']:.6g}  stop={sig['stop_loss']:.6g}  "
                 f"risk=${risk_amount:.2f}  bars_to_exit={TIME_EXIT_BARS}  "
                 f"equity=${equity:,.2f}",
                 entry_price=round(sig["close"], 8),
                 stop=round(sig["stop_loss"], 8),
                 risk_amount=round(risk_amount, 4),
                 equity=round(equity, 2))

    # ------------------------------------------------------------------
    # STEP 3: Commit state
    # ------------------------------------------------------------------
    dd_now = (equity - peak) / max(peak, EPS) * 100.0
    state.update({
        "last_run_date":         str(run_date),
        "paper_equity_usdt":     equity,
        "peak_equity_usdt":      peak,
        "drawdown_pct":          round(dd_now, 4),
        "kill_switch_triggered": kill_sw,
        "closed_trade_count":    state.get("closed_trade_count", 0) + len(exits_today),
    })
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

def write_open_positions(positions: Dict[str, CDMRPosition], run_date: date) -> None:
    if not positions:
        pd.DataFrame(columns=[
            "position_id","symbol","entry_date","as_of_date",
            "entry_price","stop_loss","bars_held","bars_remaining",
            "consec_at_entry","atr_at_entry","risk_amount_usdt","qty",
        ]).to_csv(OPEN_POS_CSV, index=False)
        return
    rows = []
    for pos in positions.values():
        rows.append({
            "position_id":      pos.position_id,
            "symbol":           pos.symbol,
            "entry_date":       pos.entry_date,
            "as_of_date":       str(run_date),
            "entry_price":      pos.entry_price,
            "stop_loss":        pos.stop_loss,
            "bars_held":        pos.bars_held,
            "bars_remaining":   max(0, TIME_EXIT_BARS - pos.bars_held),
            "consec_at_entry":  pos.consec_at_entry,
            "atr_at_entry":     pos.atr_at_entry,
            "risk_amount_usdt": round(pos.risk_amount_usdt, 4),
            "qty":              pos.qty,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    if not signals:
        pd.DataFrame(columns=[
            "symbol","signal_date","close","consec","atr","ema200","stop_loss","signal"
        ]).to_csv(SIGNALS_CSV, index=False)
        return
    pd.DataFrame(signals).to_csv(SIGNALS_CSV, index=False)


def append_equity_curve(exits: List[dict]) -> None:
    if not exits:
        return
    rows = [{
        "exit_date":   ex["exit_date"],
        "symbol":      ex["symbol"],
        "entry_date":  ex["entry_date"],
        "bars_held":   ex["bars_held"],
        "gross_r":     ex["gross_r"],
        "pnl_usdt":    ex["pnl_usdt"],
        "equity":      ex["equity_after"],
        "dd_pct":      ex["dd_pct"],
        "exit_reason": ex["exit_reason"],
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
    print("T9B -- CONSECDOWNDAYSMR 1D PAPER ENGINE")
    print("=" * 70)
    print(f"Frozen config (2026-06-02):")
    print(f"  1D / {CONSEC_N}-consec-down + close>EMA{EMA_N} / Time exit {TIME_EXIT_BARS} bars / ATR stop x{ATR_STOP_MULT}")
    print(f"  Uncapped positions / {RISK_PER_TRADE_PCT*100:.2f}% risk / EMA200 filter")
    print(f"  {len(CD_UNIVERSE)} symbols / Binance Spot / LONG only")
    print(f"  Character: BULL MARKET SPECIALIST (pairs with RSI MR for all-weather)")
    print(f"  Output: {DATA_DIR}")
    print()


def print_day_summary(result: dict, verbose: bool = True) -> None:
    d, eq, dd = result["run_date"], result["equity"], result["dd_pct"]
    op, si, en, ex = (result["open_count"], result["new_signals"],
                      result["new_entries"], result["exits"])
    ks = " [KILL-SWITCH]" if result["kill_switch"] else ""
    print(f"  {d}  equity=${eq:>10,.2f}  DD={dd:>+6.2f}%  "
          f"open={op}  signals={si}  entries={en}  exits={ex}{ks}")
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
    print("FINAL STATE -- ConsecDownDaysMR 1D Paper")
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Total return   : {ret:>+.2f}%")
    print(f"  Current DD     : {dd:>+.2f}%")
    print(f"  Open positions : {op}  (uncapped)")
    print(f"  Closed trades  : {cl}")
    print(f"  Kill-switch    : {'TRIGGERED' if state.get('kill_switch_triggered') else 'not triggered'}")
    print()
    print(f"  open_positions.csv : {OPEN_POS_CSV}")
    print(f"  signals_today.csv  : {SIGNALS_CSV}")
    print(f"  equity_curve.csv   : {EQUITY_CSV}")
    print(f"  daily_log.csv      : {DAILY_LOG_CSV}")
    print()
    print("[PAPER ONLY] No real orders were placed.")


def _print_notify(state: dict, run_date: date) -> None:
    eq           = float(state.get("paper_equity_usdt", INITIAL_CAPITAL))
    peak         = float(state.get("peak_equity_usdt", INITIAL_CAPITAL))
    dd           = float(state.get("drawdown_pct", 0.0))
    ret_pct      = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    n_open       = len(state.get("open_positions", []))
    n_closed     = int(state.get("closed_trade_count", 0))
    kill_sw      = state.get("kill_switch_triggered", False)
    days_running = (run_date - FREEZE_DATE).days + 1
    review_date  = FREEZE_DATE + timedelta(days=90)
    days_to_rev  = (review_date - run_date).days

    print()
    print("=" * 50)
    print(f"T9B CD-MR DAILY UPDATE -- {run_date}")
    print("=" * 50)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)")
    print(f"Peak equity   : ${peak:>10,.2f}")
    print(f"Max drawdown  : {dd:>+.2f}%")
    print(f"Open positions: {n_open}  (uncapped)")
    print(f"Closed trades : {n_closed}")
    print(f"Days running  : {days_running}")
    print(f"Days to review: {days_to_rev}  (3-month: {review_date})")
    if kill_sw:
        print("KILL-SWITCH   : TRIGGERED -- no new entries")
    print()

    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"  {'Symbol':<16} {'Entry':<12} {'Stop':>12} {'Held':>5} {'Rem':>5} {'CD':>4}")
        print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*5} {'-'*5} {'-'*4}")
        for p in sorted(positions, key=lambda x: x.get("bars_held", 0), reverse=True):
            sym  = str(p.get("symbol", "?"))
            edate = str(p.get("entry_date", "?"))
            stop  = p.get("stop_loss", 0)
            held  = int(p.get("bars_held", 0))
            rem   = max(0, TIME_EXIT_BARS - held)
            cdat  = int(p.get("consec_at_entry", 0))
            print(f"  {sym:<16} {edate:<12} {stop:>12.6g} {held:>5} {rem:>5} {cdat:>4}")
    else:
        print("OPEN POSITIONS: none")
    print()

    if SIGNALS_CSV.exists():
        try:
            sig_df = pd.read_csv(SIGNALS_CSV)
            if not sig_df.empty:
                print(f"CONSEC-DOWN SIGNALS TODAY ({len(sig_df)})")
                for _, row in sig_df.iterrows():
                    sym   = str(row.get("symbol", "?"))
                    cdown = row.get("consec", "?")
                    cls   = row.get("close", "?")
                    stop  = row.get("stop_loss", "?")
                    ema   = row.get("ema200", "?")
                    print(f"  {sym:<16}  consec={cdown}  close={cls:<12.6g}  "
                          f"ema200={ema:<12.6g}  stop={stop:<12.6g}")
            else:
                print("CONSEC-DOWN SIGNALS TODAY: none")
        except Exception:
            print("CONSEC-DOWN SIGNALS TODAY: (could not read)")
    else:
        print("CONSEC-DOWN SIGNALS TODAY: none")

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
    ap = argparse.ArgumentParser(
        description="T9B ConsecDownDaysMR 1D Paper Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase_t9b_consecdowndays_paper_engine.py\n"
            "  python phase_t9b_consecdowndays_paper_engine.py --date 2026-06-02\n"
            "  python phase_t9b_consecdowndays_paper_engine.py --backfill\n"
            "  python phase_t9b_consecdowndays_paper_engine.py --notify\n"
            "  python phase_t9b_consecdowndays_paper_engine.py --reset\n"
        ),
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD")
    grp.add_argument("--backfill", action="store_true",
                     help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip live fetch; use committed cache only")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe state.json and restart from $10,000")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-event detail")
    ap.add_argument("--notify", action="store_true",
                    help="Print compact notification summary (for CI)")
    return ap.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args    = parse_args()
    verbose = not args.quiet

    print_banner()

    if args.reset:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        for f in [EQUITY_CSV, DAILY_LOG_CSV]:
            if f.exists():
                f.unlink()
        print("[RESET] State and logs cleared. Starting from $10,000.")
        print()

    symbols = CD_UNIVERSE

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
        print(f"[MODE] Yesterday: {yesterday}")

    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    refresh_ohlcv_cache(symbols)
    print()

    state    = load_state()
    n_open   = len(state.get("open_positions", []))
    n_closed = state.get("closed_trade_count", 0)
    print(f"[STATE] equity=${state['paper_equity_usdt']:,.2f}  "
          f"open={n_open}  closed={n_closed}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}")
    print()

    # Fix 3: state reconciliation on startup
    _reconcile_state(state, run_dates[0])
    print()

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
            print("[HALT] Kill-switch triggered. No new entries until reviewed.")
            break

    print_final_summary(state)

    if args.notify:
        _print_notify(state, run_dates[-1] if run_dates else yesterday)

    return 0


if __name__ == "__main__":
    t9b_shared.run_engine("consecdown", main)
