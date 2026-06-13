#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T9A — BINANCE PAPER SIM ENGINE
====================================

Purpose
-------
First paper-live engine for the Trend Following System.

This engine:
- downloads Binance spot OHLCV data through ccxt
- uses CLOSED 6H candles only
- scans up to 70 USDT spot assets
- applies frozen T8 logic:
    * Donchian breakout
    * EMA slope trend structure
    * ATR initial stop
    * wide Chandelier trailing after +4R
    * no early breakeven
    * max 5 open positions
    * low portfolio heat
- opens PAPER positions only
- updates exits on closed candles only
- persists state to JSON
- logs signals, open positions, closed trades, equity, skipped signals

Important
---------
NO REAL ORDERS.
NO API KEYS REQUIRED.
NO LIVE TRADING.

Recommended loop:
    run every 15-30 minutes.
The engine itself only acts when a new 6H closed candle is available.

Dependencies:
    pip install ccxt pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json
import math
import time

import ccxt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG — FROZEN T8 / T9A
# ============================================================

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data" / "paper_trend_t9a"
CACHE_DIR = DATA_DIR / "ohlcv_cache"
STATE_PATH = DATA_DIR / "trend_t9a_state.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_NAME = "TREND_6H_DONCHIAN_WIDE_T9A"

TIMEFRAME = "6h"
LIMIT_BARS = 250
UNIVERSE_SIZE = 70

# Core trend parameters kept simple and frozen from research direction.
DONCHIAN_ENTRY_N = 20
EMA_SLOPE_N = 50
EMA_SLOPE_LOOKBACK = 10
ATR_N = 14
INITIAL_STOP_ATR_MULT = 2.0

# Wide exit engineering from T3B.
CH_ACTIVATE_R = 4.0
CH_ATR_MULT = 4.0
CH_LOOKBACK = 22

# Portfolio layer from T6/T8.
INITIAL_CAPITAL_USDT = 10_000.0
RISK_PER_TRADE_PCT = 0.0025      # 0.25%
MAX_PORTFOLIO_HEAT_PCT = 0.015   # 1.5%
MAX_OPEN_POSITIONS = 5
MAX_POSITIONS_PER_SYMBOL = 1

# Paper notional proxy.
LEVERAGE_PROXY = 3.0
MAX_NOTIONAL_PER_TRADE_PCT = 0.35
MAX_MARGIN_USAGE_PCT = 0.85

# Risk control.
KILL_SWITCH_DD_PCT = 35.0
EQUITY_FLOOR_PCT = 50.0

# Data / exchange.
EXCHANGE_ID = "binance"
QUOTE = "USDT"
SLEEP_BETWEEN_CALLS_SEC = 0.05
MAX_RETRIES = 3

# Universe exclusions.
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD", "USDD", "DAI",
    "USD1", "RLUSD", "EUR", "AEUR", "WBTC", "BTTC"
}
EXCLUDE_CONTAINS = ["UP/", "DOWN/", "BULL/", "BEAR/", "3L/", "3S/"]

# Output files.
SIGNALS_CSV = DATA_DIR / "signals_trend_t9a.csv"
OPEN_POSITIONS_CSV = DATA_DIR / "open_positions_trend_t9a.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_trend_t9a.csv"
EQUITY_CSV = DATA_DIR / "equity_trend_t9a.csv"
SKIPPED_CSV = DATA_DIR / "skipped_signals_trend_t9a.csv"
HEALTH_JSON = DATA_DIR / "system_health_trend_t9a.json"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    position_id: str
    symbol: str
    side: str
    timeframe: str
    entry_time: str
    entry_price: float
    initial_stop: float
    current_stop: float
    initial_risk_per_unit: float
    risk_amount_usdt: float
    notional_usdt: float
    margin_reserved_usdt: float
    qty: float
    highest_high_since_entry: float
    lowest_low_since_entry: float
    max_favorable_r: float
    chandelier_active: bool
    entry_bar_time: str
    last_update_time: str
    status: str = "OPEN"


# ============================================================
# UTILS
# ============================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def write_df(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    tmp.replace(path)


def profit_factor(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema50"] = out["close"].ewm(span=EMA_SLOPE_N, adjust=False).mean()
    out["ema50_slope"] = out["ema50"] - out["ema50"].shift(EMA_SLOPE_LOOKBACK)

    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = out["tr"].rolling(ATR_N).mean()

    # Entry levels must use previous bars only, never current bar.
    out["donchian_high_entry"] = out["high"].shift(1).rolling(DONCHIAN_ENTRY_N).max()
    out["donchian_low_entry"] = out["low"].shift(1).rolling(DONCHIAN_ENTRY_N).min()

    out["ch_high"] = out["high"].rolling(CH_LOOKBACK).max()
    out["ch_low"] = out["low"].rolling(CH_LOOKBACK).min()

    return out


def prepare_closed_candles(raw: pd.DataFrame) -> pd.DataFrame:
    """
    ccxt may return the currently forming candle.
    We drop the last candle unless its close time is safely in the past.
    For 6h, candle open timestamp + 6h must be <= now.
    """
    if raw.empty:
        return raw

    df = raw.copy()
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    now = pd.Timestamp.utcnow()
    timeframe_delta = pd.Timedelta(hours=6)

    df["close_time"] = df["time"] + timeframe_delta
    closed = df[df["close_time"] <= now].copy()
    return closed.reset_index(drop=True)


# ============================================================
# EXCHANGE / DATA
# ============================================================

def make_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        }
    })
    return exchange


def get_universe(exchange) -> List[str]:
    markets = exchange.load_markets()

    rows = []
    for symbol, market in markets.items():
        if not market.get("spot", False):
            continue
        if not market.get("active", True):
            continue
        if market.get("quote") != QUOTE:
            continue
        if ":" in symbol:
            continue

        base = market.get("base")
        if base in EXCLUDE_BASES:
            continue
        if any(x in symbol for x in EXCLUDE_CONTAINS):
            continue

        # Prefer liquid markets. ccxt may not always expose volume here.
        rows.append(symbol)

    # We do not have reliable live volume from load_markets, so keep exchange order.
    # This is acceptable for paper sim; for production, universe should be frozen externally.
    rows = sorted(set(rows))

    # Prefer major symbols first if present.
    preferred = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "TRX/USDT", "DOT/USDT", "BCH/USDT",
        "LTC/USDT", "UNI/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT",
    ]
    ordered = [s for s in preferred if s in rows] + [s for s in rows if s not in preferred]
    return ordered[:UNIVERSE_SIZE]


def fetch_ohlcv_with_cache(exchange, symbol: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{safe_symbol_name(symbol)}_{TIMEFRAME}.csv"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT_BARS)
            raw = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            raw.to_csv(cache_path, index=False)
            return raw
        except Exception as e:
            last_error = e
            time.sleep(0.5 * attempt)

    # Fallback to cache if available.
    if cache_path.exists():
        return pd.read_csv(cache_path)

    raise RuntimeError(f"Could not fetch {symbol}: {last_error}")


# ============================================================
# STATE
# ============================================================

def default_state() -> dict:
    return {
        "system_name": SYSTEM_NAME,
        "created_utc": utc_now_str(),
        "last_run_utc": None,
        "last_closed_bar_time": None,
        "closed_equity_usdt": INITIAL_CAPITAL_USDT,
        "peak_equity_usdt": INITIAL_CAPITAL_USDT,
        "drawdown_pct": 0.0,
        "kill_switch_triggered": False,
        "open_positions": [],
        "closed_trade_count": 0,
        "last_error": None,
    }


def load_positions(state: dict) -> Dict[str, Position]:
    positions = {}
    for p in state.get("open_positions", []):
        pos = Position(**p)
        positions[pos.position_id] = pos
    return positions


def store_positions(state: dict, positions: Dict[str, Position]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def compute_closed_equity_from_trades() -> Tuple[float, int]:
    trades = read_csv_or_empty(CLOSED_TRADES_CSV)
    if trades.empty or "pnl_usdt" not in trades.columns:
        return INITIAL_CAPITAL_USDT, 0
    equity = INITIAL_CAPITAL_USDT + float(pd.to_numeric(trades["pnl_usdt"], errors="coerce").fillna(0).sum())
    return equity, len(trades)


def reserved_margin(positions: Dict[str, Position]) -> float:
    return sum(p.margin_reserved_usdt for p in positions.values())


def open_risk_amount(positions: Dict[str, Position]) -> float:
    return sum(p.risk_amount_usdt for p in positions.values())


# ============================================================
# SIGNALS
# ============================================================

def generate_signal(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    if len(df) < max(DONCHIAN_ENTRY_N, EMA_SLOPE_N, ATR_N) + EMA_SLOPE_LOOKBACK + 5:
        return None

    dfi = add_indicators(df)
    row = dfi.iloc[-1]

    if not np.isfinite(row["atr"]) or row["atr"] <= 0:
        return None

    close = float(row["close"])
    atr = float(row["atr"])
    bar_time = str(row["time"])

    long_breakout = close > float(row["donchian_high_entry"]) and float(row["ema50_slope"]) > 0
    short_breakout = close < float(row["donchian_low_entry"]) and float(row["ema50_slope"]) < 0

    if long_breakout:
        initial_stop = close - INITIAL_STOP_ATR_MULT * atr
        return {
            "timestamp_utc": utc_now_str(),
            "symbol": symbol,
            "timeframe": TIMEFRAME,
            "bar_time": bar_time,
            "side": "LONG",
            "entry_price": close,
            "initial_stop": initial_stop,
            "atr": atr,
            "ema50": float(row["ema50"]),
            "ema50_slope": float(row["ema50_slope"]),
            "donchian_high_entry": float(row["donchian_high_entry"]),
            "donchian_low_entry": float(row["donchian_low_entry"]),
            "signal": "BUY",
        }

    if short_breakout:
        initial_stop = close + INITIAL_STOP_ATR_MULT * atr
        return {
            "timestamp_utc": utc_now_str(),
            "symbol": symbol,
            "timeframe": TIMEFRAME,
            "bar_time": bar_time,
            "side": "SHORT",
            "entry_price": close,
            "initial_stop": initial_stop,
            "atr": atr,
            "ema50": float(row["ema50"]),
            "ema50_slope": float(row["ema50_slope"]),
            "donchian_high_entry": float(row["donchian_high_entry"]),
            "donchian_low_entry": float(row["donchian_low_entry"]),
            "signal": "SELL",
        }

    return None


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def update_position_with_closed_bar(pos: Position, df: pd.DataFrame) -> Tuple[Optional[dict], Position]:
    """
    Manage exit on the latest closed candle only.
    Stop hit is approximated with closed candle high/low range.
    """
    if df.empty:
        return None, pos

    dfi = add_indicators(df)
    row = dfi.iloc[-1]

    bar_time = str(row["time"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    atr = float(row["atr"]) if np.isfinite(row["atr"]) else np.nan

    pos.highest_high_since_entry = max(pos.highest_high_since_entry, high)
    pos.lowest_low_since_entry = min(pos.lowest_low_since_entry, low)

    if pos.side == "LONG":
        favorable_r = (high - pos.entry_price) / max(pos.initial_risk_per_unit, 1e-12)
        pos.max_favorable_r = max(pos.max_favorable_r, favorable_r)

        if pos.max_favorable_r >= CH_ACTIVATE_R and np.isfinite(atr):
            pos.chandelier_active = True
            ch_stop = pos.highest_high_since_entry - CH_ATR_MULT * atr
            pos.current_stop = max(pos.current_stop, ch_stop)

        exit_hit = low <= pos.current_stop
        exit_price = pos.current_stop if exit_hit else None

        if exit_hit:
            gross_r = (exit_price - pos.entry_price) / max(pos.initial_risk_per_unit, 1e-12)
            pnl = pos.risk_amount_usdt * gross_r
            return {
                "timestamp_utc": utc_now_str(),
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "timeframe": pos.timeframe,
                "entry_time": pos.entry_time,
                "exit_time": bar_time,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "initial_stop": pos.initial_stop,
                "final_stop": pos.current_stop,
                "gross_r": gross_r,
                "net_r": gross_r,
                "pnl_usdt": pnl,
                "risk_amount_usdt": pos.risk_amount_usdt,
                "notional_usdt": pos.notional_usdt,
                "qty": pos.qty,
                "exit_reason": "STOP_OR_TRAILING",
                "max_favorable_r": pos.max_favorable_r,
                "chandelier_active": pos.chandelier_active,
            }, pos

    else:  # SHORT
        favorable_r = (pos.entry_price - low) / max(pos.initial_risk_per_unit, 1e-12)
        pos.max_favorable_r = max(pos.max_favorable_r, favorable_r)

        if pos.max_favorable_r >= CH_ACTIVATE_R and np.isfinite(atr):
            pos.chandelier_active = True
            ch_stop = pos.lowest_low_since_entry + CH_ATR_MULT * atr
            pos.current_stop = min(pos.current_stop, ch_stop)

        exit_hit = high >= pos.current_stop
        exit_price = pos.current_stop if exit_hit else None

        if exit_hit:
            gross_r = (pos.entry_price - exit_price) / max(pos.initial_risk_per_unit, 1e-12)
            pnl = pos.risk_amount_usdt * gross_r
            return {
                "timestamp_utc": utc_now_str(),
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "timeframe": pos.timeframe,
                "entry_time": pos.entry_time,
                "exit_time": bar_time,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "initial_stop": pos.initial_stop,
                "final_stop": pos.current_stop,
                "gross_r": gross_r,
                "net_r": gross_r,
                "pnl_usdt": pnl,
                "risk_amount_usdt": pos.risk_amount_usdt,
                "notional_usdt": pos.notional_usdt,
                "qty": pos.qty,
                "exit_reason": "STOP_OR_TRAILING",
                "max_favorable_r": pos.max_favorable_r,
                "chandelier_active": pos.chandelier_active,
            }, pos

    pos.last_update_time = bar_time
    return None, pos


def try_open_position(signal: dict, state: dict, positions: Dict[str, Position]) -> Tuple[Optional[Position], Optional[dict]]:
    equity = float(state["closed_equity_usdt"])
    peak = float(state.get("peak_equity_usdt", equity))
    dd_pct = float(state.get("drawdown_pct", 0.0))

    symbol = signal["symbol"]
    side = signal["side"]
    entry_price = float(signal["entry_price"])
    initial_stop = float(signal["initial_stop"])
    risk_per_unit = abs(entry_price - initial_stop)

    open_same_symbol = any(p.symbol == symbol for p in positions.values())
    if open_same_symbol and MAX_POSITIONS_PER_SYMBOL <= 1:
        return None, {"reason": "SYMBOL_ALREADY_OPEN"}

    if state.get("kill_switch_triggered", False):
        return None, {"reason": "KILL_SWITCH"}

    if dd_pct <= -KILL_SWITCH_DD_PCT:
        return None, {"reason": "DD_KILL_SWITCH"}

    if equity <= INITIAL_CAPITAL_USDT * EQUITY_FLOOR_PCT / 100.0:
        return None, {"reason": "EQUITY_FLOOR"}

    if len(positions) >= MAX_OPEN_POSITIONS:
        return None, {"reason": "MAX_OPEN_POSITIONS"}

    if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
        return None, {"reason": "INVALID_STOP_DISTANCE"}

    risk_amount = equity * RISK_PER_TRADE_PCT
    current_heat = open_risk_amount(positions) / max(equity, 1e-12)
    if current_heat + RISK_PER_TRADE_PCT > MAX_PORTFOLIO_HEAT_PCT + 1e-12:
        return None, {"reason": "MAX_PORTFOLIO_HEAT"}

    qty = risk_amount / risk_per_unit
    notional = qty * entry_price
    max_notional = equity * MAX_NOTIONAL_PER_TRADE_PCT
    if notional > max_notional:
        notional = max_notional
        qty = notional / entry_price
        risk_amount = qty * risk_per_unit

    margin = notional / LEVERAGE_PROXY
    reserved = reserved_margin(positions)
    if reserved + margin > equity * MAX_MARGIN_USAGE_PCT:
        return None, {"reason": "MAX_MARGIN_USAGE"}

    position_id = f"{safe_symbol_name(symbol)}_{side}_{signal['bar_time']}"

    pos = Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        timeframe=TIMEFRAME,
        entry_time=signal["bar_time"],
        entry_price=entry_price,
        initial_stop=initial_stop,
        current_stop=initial_stop,
        initial_risk_per_unit=risk_per_unit,
        risk_amount_usdt=risk_amount,
        notional_usdt=notional,
        margin_reserved_usdt=margin,
        qty=qty,
        highest_high_since_entry=entry_price,
        lowest_low_since_entry=entry_price,
        max_favorable_r=0.0,
        chandelier_active=False,
        entry_bar_time=signal["bar_time"],
        last_update_time=signal["bar_time"],
        status="OPEN",
    )

    return pos, None


# ============================================================
# MAIN ENGINE
# ============================================================

def main() -> int:
    print("=" * 80)
    print("PHASE T9A — BINANCE PAPER SIM ENGINE")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Timeframe:    {TIMEFRAME}")
    print("NO REAL ORDERS — PAPER ONLY")
    print("")

    state = load_json(STATE_PATH, default_state())

    # Recompute closed equity from trade log for robustness.
    closed_equity, closed_count = compute_closed_equity_from_trades()
    state["closed_equity_usdt"] = closed_equity
    state["closed_trade_count"] = closed_count
    state["peak_equity_usdt"] = max(float(state.get("peak_equity_usdt", INITIAL_CAPITAL_USDT)), closed_equity)
    state["drawdown_pct"] = (closed_equity - state["peak_equity_usdt"]) / max(state["peak_equity_usdt"], 1e-12) * 100.0

    if state["drawdown_pct"] <= -KILL_SWITCH_DD_PCT:
        state["kill_switch_triggered"] = True

    positions = load_positions(state)

    exchange = make_exchange()

    universe = get_universe(exchange)
    print(f"Universe size: {len(universe)}")
    print(f"Open positions at start: {len(positions)}")
    print(f"Closed equity: {closed_equity:.2f} USDT | DD: {state['drawdown_pct']:.2f}%")
    print("")

    signal_rows = []
    skipped_rows = []
    closed_rows = []
    errors = []

    # Fetch data for all needed symbols: universe + open position symbols.
    symbols_to_scan = list(dict.fromkeys(universe + [p.symbol for p in positions.values()]))

    latest_bar_times = []

    data_by_symbol: Dict[str, pd.DataFrame] = {}

    for idx, symbol in enumerate(symbols_to_scan, 1):
        try:
            raw = fetch_ohlcv_with_cache(exchange, symbol)
            closed = prepare_closed_candles(raw)
            if not closed.empty:
                latest_bar_times.append(str(closed.iloc[-1]["time"]))
            data_by_symbol[symbol] = closed
            print(f"[DATA] {idx:03d}/{len(symbols_to_scan)} {symbol}: {len(closed)} closed bars")
        except Exception as e:
            msg = f"{symbol}: {e}"
            errors.append(msg)
            print(f"[ERR] {msg}")

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # --------------------------------------------------------
    # 1) Manage exits first
    # --------------------------------------------------------
    for pid, pos in list(positions.items()):
        df = data_by_symbol.get(pos.symbol)
        if df is None or df.empty:
            continue

        closed_trade, updated_pos = update_position_with_closed_bar(pos, df)

        if closed_trade is not None:
            closed_rows.append(closed_trade)
            positions.pop(pid, None)

            # Update closed equity immediately.
            state["closed_equity_usdt"] = float(state["closed_equity_usdt"]) + float(closed_trade["pnl_usdt"])
            state["closed_trade_count"] = int(state.get("closed_trade_count", 0)) + 1
            state["peak_equity_usdt"] = max(float(state["peak_equity_usdt"]), float(state["closed_equity_usdt"]))
            state["drawdown_pct"] = (
                float(state["closed_equity_usdt"]) - float(state["peak_equity_usdt"])
            ) / max(float(state["peak_equity_usdt"]), 1e-12) * 100.0

            if state["drawdown_pct"] <= -KILL_SWITCH_DD_PCT:
                state["kill_switch_triggered"] = True
        else:
            positions[pid] = updated_pos

    append_csv(CLOSED_TRADES_CSV, closed_rows)

    # --------------------------------------------------------
    # 2) Generate entry signals
    # --------------------------------------------------------
    for symbol in universe:
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            continue

        sig = generate_signal(symbol, df)
        if sig is None:
            continue

        signal_rows.append(sig)

        pos, skip = try_open_position(sig, state, positions)

        if pos is not None:
            positions[pos.position_id] = pos
        else:
            row = sig.copy()
            row["skip_reason"] = skip["reason"] if skip else "UNKNOWN"
            row["closed_equity_usdt"] = state["closed_equity_usdt"]
            row["open_positions"] = len(positions)
            row["portfolio_heat_pct"] = open_risk_amount(positions) / max(float(state["closed_equity_usdt"]), 1e-12) * 100.0
            skipped_rows.append(row)

    append_csv(SIGNALS_CSV, signal_rows)
    append_csv(SKIPPED_CSV, skipped_rows)

    # --------------------------------------------------------
    # 3) Save open positions snapshot
    # --------------------------------------------------------
    open_df = pd.DataFrame([asdict(p) for p in positions.values()])
    write_df(OPEN_POSITIONS_CSV, open_df)

    # --------------------------------------------------------
    # 4) Save equity snapshot
    # --------------------------------------------------------
    equity_row = {
        "timestamp_utc": utc_now_str(),
        "closed_equity_usdt": state["closed_equity_usdt"],
        "peak_equity_usdt": state["peak_equity_usdt"],
        "drawdown_pct": state["drawdown_pct"],
        "closed_trade_count": state["closed_trade_count"],
        "open_positions": len(positions),
        "open_risk_amount_usdt": open_risk_amount(positions),
        "portfolio_heat_pct": open_risk_amount(positions) / max(float(state["closed_equity_usdt"]), 1e-12) * 100.0,
        "reserved_margin_usdt": reserved_margin(positions),
        "kill_switch_triggered": state["kill_switch_triggered"],
    }
    append_csv(EQUITY_CSV, [equity_row])

    # --------------------------------------------------------
    # 5) Save state and health
    # --------------------------------------------------------
    state["last_run_utc"] = utc_now_str()
    state["last_closed_bar_time"] = max(latest_bar_times) if latest_bar_times else state.get("last_closed_bar_time")
    state["last_error"] = "; ".join(errors[-5:]) if errors else None
    store_positions(state, positions)
    save_json(STATE_PATH, state)

    health = {
        "system_name": SYSTEM_NAME,
        "timestamp_utc": utc_now_str(),
        "status": "KILL_SWITCH" if state["kill_switch_triggered"] else "OK",
        "timeframe": TIMEFRAME,
        "universe_size": len(universe),
        "open_positions": len(positions),
        "closed_equity_usdt": state["closed_equity_usdt"],
        "drawdown_pct": state["drawdown_pct"],
        "portfolio_heat_pct": equity_row["portfolio_heat_pct"],
        "reserved_margin_usdt": equity_row["reserved_margin_usdt"],
        "signals_this_run": len(signal_rows),
        "new_closed_trades_this_run": len(closed_rows),
        "skipped_this_run": len(skipped_rows),
        "errors_this_run": errors,
        "last_closed_bar_time": state["last_closed_bar_time"],
        "paper_only": True,
    }
    save_json(HEALTH_JSON, health)

    print("")
    print("=" * 80)
    print("T9A RUN SUMMARY")
    print("=" * 80)
    print(f"Signals this run:       {len(signal_rows)}")
    print(f"Skipped this run:       {len(skipped_rows)}")
    print(f"Closed trades this run: {len(closed_rows)}")
    print(f"Open positions:         {len(positions)}")
    print(f"Closed equity:          {state['closed_equity_usdt']:.2f} USDT")
    print(f"Drawdown:               {state['drawdown_pct']:.2f}%")
    print(f"Portfolio heat:         {equity_row['portfolio_heat_pct']:.2f}%")
    print(f"Kill-switch:            {state['kill_switch_triggered']}")
    print("")
    print("[OK] signals_trend_t9a.csv")
    print("[OK] open_positions_trend_t9a.csv")
    print("[OK] closed_trades_trend_t9a.csv")
    print("[OK] equity_trend_t9a.csv")
    print("[OK] skipped_signals_trend_t9a.csv")
    print("[OK] trend_t9a_state.json")
    print("[OK] system_health_trend_t9a.json")
    print("")
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
