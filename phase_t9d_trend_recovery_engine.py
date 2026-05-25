#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T9D — TREND RECOVERY & RECONCILIATION ENGINE
==================================================

Recovery engine for Trend T9A / T9A V2 paper-live.

Use after:
- PC off
- loop stopped
- crash
- internet down

It replays CLOSED 6H candles from each open position's last_update_time,
updates trailing/Chandelier state, closes positions that would have hit
stop/trailing, and syncs state/equity/dashboard files.

NO REAL ORDERS. PAPER ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json
import time

import ccxt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG — must match T9A V2
# ============================================================

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data" / "paper_trend_t9a"
CACHE_DIR = DATA_DIR / "ohlcv_cache"

STATE_PATH = DATA_DIR / "trend_t9a_state.json"

OPEN_POSITIONS_CSV = DATA_DIR / "open_positions_trend_t9a.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_trend_t9a.csv"
EQUITY_CSV = DATA_DIR / "equity_trend_t9a.csv"
HEALTH_JSON = DATA_DIR / "system_health_trend_t9a.json"

RECOVERY_EVENTS_CSV = DATA_DIR / "phase_t9d_recovery_events.csv"
RECOVERY_REPORT_CSV = DATA_DIR / "phase_t9d_recovery_report.csv"

TIMEFRAME = "6h"
TIMEFRAME_HOURS = 6
LIMIT_BARS = 500

ATR_N = 14
CH_ACTIVATE_R = 4.0
CH_ATR_MULT = 4.0

INITIAL_CAPITAL_USDT = 10_000.0
KILL_SWITCH_DD_PCT = 35.0

EXCHANGE_ID = "binance"
SLEEP_BETWEEN_CALLS_SEC = 0.05
MAX_RETRIES = 3


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
    chandelier_activation_time: Optional[str]
    chandelier_activation_price: Optional[float]
    entry_bar_time: str
    last_update_time: str
    status: str = "OPEN"


# ============================================================
# BASIC HELPERS
# ============================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    tmp.replace(path)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def default_state() -> dict:
    return {
        "system_name": "TREND_6H_DONCHIAN_WIDE_T9A",
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


# ============================================================
# POSITION LOAD / SAVE
# ============================================================

def _clean_optional(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _position_from_dict(p: dict) -> Optional[Position]:
    p = dict(p)

    p.setdefault("chandelier_activation_time", None)
    p.setdefault("chandelier_activation_price", None)
    p.setdefault("status", "OPEN")

    for k in list(p.keys()):
        p[k] = _clean_optional(p[k])

    # Required fields for backward compatibility fallback.
    p.setdefault("highest_high_since_entry", p.get("entry_price"))
    p.setdefault("lowest_low_since_entry", p.get("entry_price"))
    p.setdefault("max_favorable_r", 0.0)
    p.setdefault("chandelier_active", False)
    p.setdefault("last_update_time", p.get("entry_time"))
    p.setdefault("entry_bar_time", p.get("entry_time"))

    try:
        return Position(**p)
    except Exception as e:
        print(f"[WARN] Could not parse position: {e}")
        return None


def load_positions_from_state(state: dict) -> Dict[str, Position]:
    positions = {}
    for p in state.get("open_positions", []):
        pos = _position_from_dict(p)
        if pos is not None:
            positions[pos.position_id] = pos
    return positions


def load_positions_from_csv() -> Dict[str, Position]:
    df = read_csv_or_empty(OPEN_POSITIONS_CSV)
    positions = {}
    if df.empty:
        return positions

    for _, r in df.iterrows():
        pos = _position_from_dict(r.to_dict())
        if pos is not None:
            positions[pos.position_id] = pos
    return positions


def store_positions(state: dict, positions: Dict[str, Position]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def compute_closed_equity_from_trades() -> Tuple[float, int]:
    trades = read_csv_or_empty(CLOSED_TRADES_CSV)
    if trades.empty or "pnl_usdt" not in trades.columns:
        return INITIAL_CAPITAL_USDT, 0

    pnl = pd.to_numeric(trades["pnl_usdt"], errors="coerce").fillna(0.0).sum()
    return INITIAL_CAPITAL_USDT + float(pnl), len(trades)


def current_reserved_margin(positions: Dict[str, Position]) -> float:
    return sum(float(p.margin_reserved_usdt) for p in positions.values())


def current_open_risk_amount(positions: Dict[str, Position]) -> float:
    return sum(float(p.risk_amount_usdt) for p in positions.values())


# ============================================================
# CANDLES / INDICATORS
# ============================================================

def make_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    return exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def fetch_ohlcv_with_cache(exchange, symbol: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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

    if cache_path.exists():
        print(f"[CACHE] Using cached candles for {symbol}")
        return pd.read_csv(cache_path)

    raise RuntimeError(f"Could not fetch {symbol}: {last_error}")


def prepare_closed_candles(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()

    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError("No timestamp/time column in candles.")

    df = df.sort_values("time").reset_index(drop=True)
    now = pd.Timestamp.utcnow()
    df["close_time"] = df["time"] + pd.Timedelta(hours=TIMEFRAME_HOURS)
    df = df[df["close_time"] <= now].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    return df.reset_index(drop=True)


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = out["tr"].rolling(ATR_N).mean()
    return out


def filter_replay_bars(df: pd.DataFrame, pos: Position) -> pd.DataFrame:
    start_raw = pos.last_update_time or pos.entry_time
    start = pd.to_datetime(start_raw, utc=True, errors="coerce")
    if pd.isna(start):
        start = pd.to_datetime(pos.entry_time, utc=True, errors="coerce")

    dfi = df.copy()
    dfi["time"] = pd.to_datetime(dfi["time"], utc=True, errors="coerce")
    dfi = dfi[dfi["time"] > start].copy()
    return dfi.sort_values("time").reset_index(drop=True)


# ============================================================
# RECOVERY REPLAY
# ============================================================

def replay_position(pos: Position, replay_bars: pd.DataFrame) -> Tuple[Optional[dict], Position, List[dict]]:
    events = []

    if replay_bars.empty:
        return None, pos, events

    dfi = add_atr(replay_bars)

    for _, row in dfi.iterrows():
        bar_time = str(row["time"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        atr = float(row["atr"]) if np.isfinite(row["atr"]) else np.nan

        old_stop = float(pos.current_stop)
        old_mfe = float(pos.max_favorable_r)
        old_trail = bool(pos.chandelier_active)

        pos.highest_high_since_entry = max(float(pos.highest_high_since_entry), high)
        pos.lowest_low_since_entry = min(float(pos.lowest_low_since_entry), low)

        side = pos.side.upper()

        if side == "LONG":
            favorable_r = (high - float(pos.entry_price)) / max(float(pos.initial_risk_per_unit), 1e-12)
            pos.max_favorable_r = max(float(pos.max_favorable_r), favorable_r)

            if pos.max_favorable_r >= CH_ACTIVATE_R and np.isfinite(atr):
                was_active = bool(pos.chandelier_active)
                pos.chandelier_active = True
                ch_stop = float(pos.highest_high_since_entry) - CH_ATR_MULT * atr
                pos.current_stop = max(float(pos.current_stop), ch_stop)

                if not was_active:
                    pos.chandelier_activation_time = bar_time
                    pos.chandelier_activation_price = close
                    events.append({
                        "timestamp_utc": utc_now_str(),
                        "event_type": "TRAILING_ACTIVATED_RECOVERY",
                        "position_id": pos.position_id,
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "bar_time": bar_time,
                        "price": close,
                        "mfe_r": pos.max_favorable_r,
                        "old_stop": old_stop,
                        "new_stop": pos.current_stop,
                    })

            exit_hit = low <= float(pos.current_stop)

            if exit_hit:
                exit_price = float(pos.current_stop)
                gross_r = (exit_price - float(pos.entry_price)) / max(float(pos.initial_risk_per_unit), 1e-12)
                pnl = float(pos.risk_amount_usdt) * gross_r

                closed = build_closed_row(pos, bar_time, exit_price, gross_r, pnl)
                events.append(build_closed_event(pos, bar_time, exit_price, gross_r, pnl, old_stop))
                pos.status = "CLOSED_RECOVERY"
                pos.last_update_time = bar_time
                return closed, pos, events

        elif side == "SHORT":
            favorable_r = (float(pos.entry_price) - low) / max(float(pos.initial_risk_per_unit), 1e-12)
            pos.max_favorable_r = max(float(pos.max_favorable_r), favorable_r)

            if pos.max_favorable_r >= CH_ACTIVATE_R and np.isfinite(atr):
                was_active = bool(pos.chandelier_active)
                pos.chandelier_active = True
                ch_stop = float(pos.lowest_low_since_entry) + CH_ATR_MULT * atr
                pos.current_stop = min(float(pos.current_stop), ch_stop)

                if not was_active:
                    pos.chandelier_activation_time = bar_time
                    pos.chandelier_activation_price = close
                    events.append({
                        "timestamp_utc": utc_now_str(),
                        "event_type": "TRAILING_ACTIVATED_RECOVERY",
                        "position_id": pos.position_id,
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "bar_time": bar_time,
                        "price": close,
                        "mfe_r": pos.max_favorable_r,
                        "old_stop": old_stop,
                        "new_stop": pos.current_stop,
                    })

            exit_hit = high >= float(pos.current_stop)

            if exit_hit:
                exit_price = float(pos.current_stop)
                gross_r = (float(pos.entry_price) - exit_price) / max(float(pos.initial_risk_per_unit), 1e-12)
                pnl = float(pos.risk_amount_usdt) * gross_r

                closed = build_closed_row(pos, bar_time, exit_price, gross_r, pnl)
                events.append(build_closed_event(pos, bar_time, exit_price, gross_r, pnl, old_stop))
                pos.status = "CLOSED_RECOVERY"
                pos.last_update_time = bar_time
                return closed, pos, events

        if abs(float(pos.current_stop) - old_stop) > 1e-12 or abs(float(pos.max_favorable_r) - old_mfe) > 1e-12:
            events.append({
                "timestamp_utc": utc_now_str(),
                "event_type": "POSITION_UPDATED_RECOVERY",
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "bar_time": bar_time,
                "close": close,
                "old_stop": old_stop,
                "new_stop": pos.current_stop,
                "old_mfe_r": old_mfe,
                "new_mfe_r": pos.max_favorable_r,
                "chandelier_active_before": old_trail,
                "chandelier_active_after": pos.chandelier_active,
            })

        pos.last_update_time = bar_time

    return None, pos, events


def build_closed_row(pos: Position, bar_time: str, exit_price: float, gross_r: float, pnl: float) -> dict:
    return {
        "timestamp_utc": utc_now_str(),
        "recovery": True,
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
        "exit_reason": "RECOVERY_STOP_OR_TRAILING",
        "max_favorable_r": pos.max_favorable_r,
        "chandelier_active": pos.chandelier_active,
        "chandelier_activation_time": pos.chandelier_activation_time,
        "chandelier_activation_price": pos.chandelier_activation_price,
    }


def build_closed_event(pos: Position, bar_time: str, exit_price: float, gross_r: float, pnl: float, old_stop: float) -> dict:
    return {
        "timestamp_utc": utc_now_str(),
        "event_type": "POSITION_CLOSED_RECOVERY",
        "position_id": pos.position_id,
        "symbol": pos.symbol,
        "side": pos.side,
        "bar_time": bar_time,
        "exit_price": exit_price,
        "realized_r": gross_r,
        "pnl_usdt": pnl,
        "old_stop": old_stop,
        "final_stop": pos.current_stop,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("PHASE T9D — TREND RECOVERY & RECONCILIATION ENGINE")
    print("=" * 80)
    print(f"Data dir: {DATA_DIR}")
    print("NO REAL ORDERS — PAPER RECOVERY ONLY")
    print("")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    state = load_json(STATE_PATH, default_state())
    positions = load_positions_from_state(state)

    if not positions:
        csv_positions = load_positions_from_csv()
        if csv_positions:
            print("[INFO] State had no open positions; loaded positions from open_positions CSV.")
            positions = csv_positions

    print(f"Open positions before recovery: {len(positions)}")

    closed_equity, closed_count = compute_closed_equity_from_trades()
    state["closed_equity_usdt"] = closed_equity
    state["closed_trade_count"] = closed_count
    state["peak_equity_usdt"] = max(float(state.get("peak_equity_usdt", INITIAL_CAPITAL_USDT)), closed_equity)
    state["drawdown_pct"] = (
        closed_equity - float(state["peak_equity_usdt"])
    ) / max(float(state["peak_equity_usdt"]), 1e-12) * 100.0

    exchange = make_exchange()

    updated_positions = {}
    closed_rows = []
    all_events = []
    errors = []

    for idx, (pid, pos) in enumerate(list(positions.items()), 1):
        print(f"[REPLAY] {idx}/{len(positions)} {pos.symbol} {pos.side} from {pos.last_update_time}")

        try:
            raw = fetch_ohlcv_with_cache(exchange, pos.symbol)
            candles = prepare_closed_candles(raw)
            replay_bars = filter_replay_bars(candles, pos)

            if replay_bars.empty:
                print("  -> no new closed bars")
                updated_positions[pid] = pos
                continue

            print(f"  -> replay bars: {len(replay_bars)} ({replay_bars['time'].iloc[0]} -> {replay_bars['time'].iloc[-1]})")

            closed_trade, updated_pos, events = replay_position(pos, replay_bars)
            all_events.extend(events)

            if closed_trade:
                closed_rows.append(closed_trade)

                state["closed_equity_usdt"] = float(state["closed_equity_usdt"]) + float(closed_trade["pnl_usdt"])
                state["closed_trade_count"] = int(state.get("closed_trade_count", 0)) + 1
                state["peak_equity_usdt"] = max(float(state["peak_equity_usdt"]), float(state["closed_equity_usdt"]))
                state["drawdown_pct"] = (
                    float(state["closed_equity_usdt"]) - float(state["peak_equity_usdt"])
                ) / max(float(state["peak_equity_usdt"]), 1e-12) * 100.0

                print(f"  -> CLOSED recovery: R={closed_trade['net_r']:.3f} | PnL={closed_trade['pnl_usdt']:.2f}")
            else:
                updated_positions[pid] = updated_pos
                print(f"  -> still open | stop={updated_pos.current_stop:.8f} | MFE={updated_pos.max_favorable_r:.2f}R | trail={updated_pos.chandelier_active}")

        except Exception as e:
            msg = f"{pos.symbol} {pos.side}: {e}"
            errors.append(msg)
            updated_positions[pid] = pos
            print(f"  -> ERROR: {msg}")

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    append_csv(CLOSED_TRADES_CSV, closed_rows)
    append_csv(RECOVERY_EVENTS_CSV, all_events)

    open_df = pd.DataFrame([asdict(p) for p in updated_positions.values()])
    write_df(OPEN_POSITIONS_CSV, open_df)

    store_positions(state, updated_positions)
    state["last_recovery_utc"] = utc_now_str()
    state["last_error"] = "; ".join(errors[-5:]) if errors else None

    if state["drawdown_pct"] <= -KILL_SWITCH_DD_PCT:
        state["kill_switch_triggered"] = True

    save_json(STATE_PATH, state)

    equity_row = {
        "timestamp_utc": utc_now_str(),
        "event": "T9D_RECOVERY",
        "closed_equity_usdt": state["closed_equity_usdt"],
        "peak_equity_usdt": state["peak_equity_usdt"],
        "drawdown_pct": state["drawdown_pct"],
        "closed_trade_count": state["closed_trade_count"],
        "open_positions": len(updated_positions),
        "open_risk_amount_usdt": current_open_risk_amount(updated_positions),
        "portfolio_heat_pct": current_open_risk_amount(updated_positions) / max(float(state["closed_equity_usdt"]), 1e-12) * 100.0,
        "reserved_margin_usdt": current_reserved_margin(updated_positions),
        "kill_switch_triggered": state.get("kill_switch_triggered", False),
    }
    append_csv(EQUITY_CSV, [equity_row])

    health = {
        "system_name": state.get("system_name", "TREND_6H_DONCHIAN_WIDE_T9A"),
        "timestamp_utc": utc_now_str(),
        "status": "KILL_SWITCH" if state.get("kill_switch_triggered") else "OK",
        "recovery_engine": "T9D",
        "open_positions_before": len(positions),
        "open_positions_after": len(updated_positions),
        "closed_recovered": len(closed_rows),
        "events_logged": len(all_events),
        "closed_equity_usdt": state["closed_equity_usdt"],
        "drawdown_pct": state["drawdown_pct"],
        "portfolio_heat_pct": equity_row["portfolio_heat_pct"],
        "reserved_margin_usdt": equity_row["reserved_margin_usdt"],
        "errors": errors,
        "paper_only": True,
    }
    save_json(HEALTH_JSON, health)

    report = [{
        "timestamp_utc": utc_now_str(),
        "open_positions_before": len(positions),
        "open_positions_after": len(updated_positions),
        "closed_recovered": len(closed_rows),
        "events_logged": len(all_events),
        "errors": len(errors),
        "closed_equity_usdt": state["closed_equity_usdt"],
        "drawdown_pct": state["drawdown_pct"],
        "portfolio_heat_pct": equity_row["portfolio_heat_pct"],
        "reserved_margin_usdt": equity_row["reserved_margin_usdt"],
        "kill_switch_triggered": state.get("kill_switch_triggered", False),
    }]
    append_csv(RECOVERY_REPORT_CSV, report)

    print("")
    print("=" * 80)
    print("T9D RECOVERY SUMMARY")
    print("=" * 80)
    print(f"Open before:       {len(positions)}")
    print(f"Closed recovered:  {len(closed_rows)}")
    print(f"Open after:        {len(updated_positions)}")
    print(f"Events logged:     {len(all_events)}")
    print(f"Closed equity:     {state['closed_equity_usdt']:.2f} USDT")
    print(f"Drawdown:          {state['drawdown_pct']:.2f}%")
    print(f"Portfolio heat:    {equity_row['portfolio_heat_pct']:.2f}%")
    print(f"Kill-switch:       {state.get('kill_switch_triggered', False)}")
    print(f"Errors:            {len(errors)}")
    print("")
    print("[OK] phase_t9d_recovery_events.csv")
    print("[OK] phase_t9d_recovery_report.csv")
    print("[OK] open_positions_trend_t9a.csv")
    print("[OK] closed_trades_trend_t9a.csv")
    print("[OK] equity_trend_t9a.csv")
    print("[OK] trend_t9a_state.json")
    print("[OK] system_health_trend_t9a.json")
    print("")
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
