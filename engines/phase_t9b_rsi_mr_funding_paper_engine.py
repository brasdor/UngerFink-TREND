#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- RSI MR FUTURES FUNDING GATE 1D PAPER ENGINE (SYSTEM 8)
====================================================================

Paper trading engine for the frozen RSI_MR_Futures_FundingGate config.
System 8 in portfolio. Bear-market long MR specialist on Binance Futures 1D.

Frozen config (2026-06-22):
  Universe    : 290 Binance Futures USDT symbols (auto-discovered from caches)
  Timeframe   : 1D (Binance Futures USD-M perpetuals)
  Entry       : RSI(10) < 20 on daily close
                AND funding_rate < 0 at signal bar  (exact -- no threshold drift)
  Safety stop : ATR(14) x 2.0 below entry  (safety net only)
  Exit        : Fixed time exit after 25 daily bars
  Filter      : none
  Cap         : uncapped
  Risk/trade  : 0.25% of current equity, $150 ceiling
  Leverage    : 1.0x  (Binance Futures, LONG only)
  Validated   : avg_r +0.613R / t=5.28 / win 66.9% / PF 3.46 / CAGR +10.3%
  Best year   : 2022 (+$3,208, 140 trades) -- bear squeeze specialist

Data (committed caches, same as S6/S7):
  data/futures_universe/ohlcv_1d/{SYM}_1d.csv       (timestamp ms + date)
  data/futures_universe/funding_rates/{SYM}_funding.csv (funding_time ms, funding_rate)
  Live fetch via fapi.binance.com (may be geo-blocked from GitHub runners --
  falls back to committed cache silently, same behavior as S6/S7).

Usage:
  python engines/phase_t9b_rsi_mr_funding_paper_engine.py
      --date 2026-07-10   run for a specific date
      --backfill          replay from freeze date (2026-06-19) to yesterday
      --no-download       use committed caches only
      --reset             wipe state and restart
      --notify            compact notification summary (for CI)

Output files (data/t9b_rsi_mr_funding_paper/):
  state.json / open_positions.csv / signals_today.csv
  equity_curve.csv / daily_log.csv

PAPER ONLY -- NO REAL ORDERS.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import t9b_shared
from signal_arbitrator import SignalArbitrator


# =============================================================================
# FROZEN CONFIG  (2026-06-22 -- data/research_rsi_mr_futures_fundinggate_t5t6t7t8/)
# =============================================================================

SYSTEM_NAME         = "RSI_MR_Futures_FundingGate_1D_T9B"
FREEZE_DATE         = date(2026, 6, 19)   # T9B start per pipeline

RSI_N               = 10
OVERSOLD_THR        = 20      # enter when RSI(10) < 20 on close
ATR_N               = 14
ATR_STOP_MULT       = 2.0     # safety stop only -- primary exit is always time
TIME_EXIT_BARS      = 25      # exit after exactly 25 daily bars
# Funding gate: funding_rate < 0 required at signal bar (longs RECEIVE payment).
# Exactly zero-threshold per frozen config -- do not drift.

RISK_PER_TRADE_PCT  = 0.0025  # 0.25% of current equity
CAPITAL_CEILING     = 150.0   # USD ceiling on risk per trade
MIN_ORDER_SIZE_USDT = 15.0
MAX_OPEN_POSITIONS  = None    # uncapped
INITIAL_CAPITAL     = 10_000.0  # per frozen T8 config (S2/S8 pool split settled at deployment)
LEVERAGE            = 1.0
KILL_SWITCH_DD_PCT  = 35.0

LIMIT_BARS          = 2000
MIN_BARS_REQUIRED   = 60      # RSI(10)+ATR(14) warmup + margin

EPS = 1e-12


# =============================================================================
# PATHS
# =============================================================================

ROOT            = Path.cwd()
DATA_DIR        = ROOT / "data" / "t9b_rsi_mr_funding_paper"
STATE_PATH      = DATA_DIR / "state.json"
OPEN_POS_CSV    = DATA_DIR / "open_positions.csv"
SIGNALS_CSV     = DATA_DIR / "signals_today.csv"
EQUITY_CSV      = DATA_DIR / "equity_curve.csv"
DAILY_LOG_CSV   = DATA_DIR / "daily_log.csv"

CACHE_1D        = ROOT / "data" / "futures_universe" / "ohlcv_1d"
CACHE_FUNDING   = ROOT / "data" / "futures_universe" / "funding_rates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_OHLCV:   dict = {}
_SESSION_FUNDING: dict = {}
_SKIP_LIVE_FETCH: bool = False


# =============================================================================
# UNIVERSE  (auto-discovered: symbols with both 1D OHLCV and funding data)
# =============================================================================

def _discover_universe() -> List[str]:
    d1_syms = {f.stem.replace("_1d", "") for f in CACHE_1D.glob("*_1d.csv")}
    fr_syms = {f.stem.replace("_funding", "") for f in CACHE_FUNDING.glob("*_funding.csv")}
    return sorted(d1_syms & fr_syms)

S8_UNIVERSE: List[str] = _discover_universe()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class S8Position:
    position_id:           str
    symbol:                str
    entry_date:            str
    entry_price:           float
    stop_loss:             float   # ATR x 2.0 below entry (safety net)
    initial_risk_per_unit: float
    risk_amount_usdt:      float
    qty:                   float
    bars_held:             int
    atr_at_entry:          float
    rsi_at_entry:          float
    funding_at_entry:      float

    def current_r(self, price: float) -> float:
        return (price - self.entry_price) / max(self.initial_risk_per_unit, EPS)


# =============================================================================
# STATE HELPERS
# =============================================================================

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


def positions_from_state(state: dict) -> Dict[str, S8Position]:
    fields = {f.name for f in dataclasses.fields(S8Position)}
    out: Dict[str, S8Position] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = S8Position(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping position {raw.get('position_id','?')}: {exc}", flush=True)
    return out


def positions_to_state(state: dict, positions: Dict[str, S8Position]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def _reconcile_state(state: dict, run_date: date) -> None:
    positions = state.get("open_positions", [])
    issues: list[str] = []
    for pos in positions:
        pid  = pos.get("position_id", "?")
        sym  = pos.get("symbol", "")
        ep   = pos.get("entry_price")
        qty  = pos.get("qty")
        stop = pos.get("stop_loss")
        bh   = pos.get("bars_held")
        if not sym:
            issues.append(f"{pid}: missing symbol")
        try:
            if float(ep) <= 0:
                issues.append(f"{sym}: entry_price={ep} <= 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: entry_price not numeric")
        try:
            if float(qty) <= 0:
                issues.append(f"{sym}: qty={qty} <= 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: qty not numeric")
        try:
            if stop is not None and ep is not None and float(stop) >= float(ep):
                issues.append(f"{sym}: stop={stop} >= entry={ep} (invalid for LONG)")
        except (TypeError, ValueError):
            pass
        try:
            if int(bh) < 0:
                issues.append(f"{sym}: bars_held={bh} < 0")
        except (TypeError, ValueError):
            issues.append(f"{sym}: bars_held not numeric")
    if issues:
        print(f"[RECONCILE] {len(issues)} issue(s):", flush=True)
        for iss in issues:
            print(f"  [WARN] {iss}", flush=True)
        rows = [{"run_date": str(run_date), "event": "RECONCILE_WARN",
                 "symbol": "", "detail": iss} for iss in issues]
        append_csv(DAILY_LOG_CSV, rows)
    else:
        print(f"[RECONCILE] OK  ({len(positions)} position(s) verified)", flush=True)


# =============================================================================
# OHLCV LOADING  (committed 1D futures cache + optional live fetch)
# =============================================================================

def load_1d_ohlcv(symbol: str, up_to_date: Optional[date] = None) -> pd.DataFrame:
    global _SESSION_OHLCV
    if symbol not in _SESSION_OHLCV:
        cache_path = CACHE_1D / f"{symbol}_1d.csv"
        df_base: pd.DataFrame = pd.DataFrame()
        if cache_path.exists():
            try:
                df_base = _parse_ohlcv_1d(pd.read_csv(cache_path))
            except Exception as exc:
                print(f"  [WARN] {symbol}: 1D cache read error -- {exc}", flush=True)

        df_live: pd.DataFrame = pd.DataFrame()
        if not _SKIP_LIVE_FETCH:
            raw = _fetch_binance_1d(symbol, limit=15)
            if raw is not None:
                df_live = _parse_ohlcv_1d(raw)

        if not df_base.empty and not df_live.empty:
            merged = (pd.concat([df_base, df_live], ignore_index=True)
                      .drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True))
        elif not df_base.empty:
            merged = df_base
        else:
            merged = df_live

        if not merged.empty and not df_live.empty and cache_path.parent.exists():
            out = merged.copy()
            out["date"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
            out = out.rename(columns={"ts_ms": "timestamp"})
            out[["timestamp", "open", "high", "low", "close", "volume", "date"]].to_csv(
                cache_path, index=False)

        if len(merged) > LIMIT_BARS:
            merged = merged.iloc[-LIMIT_BARS:].reset_index(drop=True)

        _SESSION_OHLCV[symbol] = merged

    df = _SESSION_OHLCV[symbol]
    if up_to_date is not None:
        cutoff_ms = int(pd.Timestamp(str(up_to_date + timedelta(days=1)), tz="UTC").value // 1_000_000)
        df = df[df["ts_ms"] < cutoff_ms].copy()
    return df.reset_index(drop=True)


def _parse_ohlcv_1d(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "timestamp" in df.columns:
        df["ts_ms"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    elif "ts_ms" in df.columns:
        df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce").astype("Int64")
    else:
        return pd.DataFrame()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    df = df.dropna(subset=["ts_ms", "open", "high", "low", "close"])
    df["ts_ms"] = df["ts_ms"].astype(int)
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms")
    return df[["ts_ms", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _fetch_binance_1d(symbol: str, limit: int = 15) -> Optional[pd.DataFrame]:
    try:
        import requests
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": "1d", "limit": limit},
                         timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return pd.DataFrame([{
            "timestamp": d[0], "open": float(d[1]), "high": float(d[2]),
            "low": float(d[3]), "close": float(d[4]), "volume": float(d[5]),
        } for d in data])
    except Exception:
        return None  # geo-block / network -- fall back to committed cache


# =============================================================================
# FUNDING RATES  (committed cache; gate = latest rate at/before signal bar close)
# =============================================================================

def load_funding(symbol: str) -> pd.DataFrame:
    global _SESSION_FUNDING
    if symbol not in _SESSION_FUNDING:
        path = CACHE_FUNDING / f"{symbol}_funding.csv"
        df = pd.DataFrame()
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["funding_time"] = pd.to_numeric(df["funding_time"], errors="coerce")
                df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
                df = (df.dropna().drop_duplicates("funding_time")
                        .sort_values("funding_time").reset_index(drop=True))
            except Exception as exc:
                print(f"  [WARN] {symbol}: funding read error -- {exc}", flush=True)
        _SESSION_FUNDING[symbol] = df
    return _SESSION_FUNDING[symbol]


def funding_at(symbol: str, bar_close_ms: int) -> Optional[float]:
    """Latest funding rate with funding_time <= bar close. None if no data."""
    df = load_funding(symbol)
    if df.empty:
        return None
    sub = df[df["funding_time"] <= bar_close_ms]
    if sub.empty:
        return None
    return float(sub["funding_rate"].iloc[-1])


# =============================================================================
# INDICATORS  -- ported VERBATIM from step4_rsi_mr_futures_funding_t1.py
# (SMA-seeded Wilder recursion; verified 257/257 exact trade match vs frozen)
# =============================================================================

def _compute_rsi(close: np.ndarray, n: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_g = np.full(len(close), np.nan)
    avg_l = np.full(len(close), np.nan)
    if len(close) < n + 1:
        return np.full(len(close), np.nan)
    avg_g[n] = gain[1:n+1].mean()
    avg_l[n] = loss[1:n+1].mean()
    alpha = 1.0 / n
    for i in range(n+1, len(close)):
        avg_g[i] = avg_g[i-1] * (1 - alpha) + gain[i] * alpha
        avg_l[i] = avg_l[i-1] * (1 - alpha) + loss[i] * alpha
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:n] = np.nan
    return rsi


def _compute_atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, n: int = 14) -> np.ndarray:
    nb = len(cl)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = np.nanmean(tr[1:n+1])
        for i in range(n+1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i-1]):
                atr[i] = (atr[i-1]*(n-1)+tr[i])/n
    return atr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = _compute_rsi(out["close"].values.astype(float), RSI_N)
    out["atr"] = _compute_atr(out["high"].values.astype(float),
                              out["low"].values.astype(float),
                              out["close"].values.astype(float), ATR_N)
    return out


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_entry_signal(symbol: str, df: pd.DataFrame, run_date: date) -> Optional[dict]:
    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    dfi = add_indicators(df)
    day_ms = int(pd.Timestamp(str(run_date), tz="UTC").value // 1_000_000)
    day_rows = dfi[dfi["ts_ms"] == day_ms]
    if day_rows.empty:
        return None

    row   = day_rows.iloc[-1]
    close = float(row["close"])
    rsi   = float(row["rsi"])
    atr   = float(row["atr"])

    if not all(np.isfinite([close, rsi, atr])) or atr <= EPS:
        return None

    if rsi >= OVERSOLD_THR:
        return None

    # Funding gate: rate < 0, referenced at signal bar OPEN timestamp --
    # identical to research script get_rate(fp, ts_arr[i]).
    # Verified 257/257 exact trade match vs frozen research trades.
    fr = funding_at(symbol, day_ms)
    if fr is None or fr >= 0:
        return None

    stop = close - ATR_STOP_MULT * atr
    return {
        "symbol":      symbol,
        "signal_date": str(run_date),
        "close":       round(close, 8),
        "rsi":         round(rsi, 4),
        "atr":         round(atr, 8),
        "funding":     round(fr, 8),
        "stop_loss":   round(stop, 8),
        "risk_unit":   round(ATR_STOP_MULT * atr, 8),
        "signal":      "BUY_MR_FUNDING",
    }


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def update_and_check_exit(pos: S8Position, df: pd.DataFrame, run_date: date) -> tuple:
    if df.empty:
        return None, pos
    day_ms = int(pd.Timestamp(str(run_date), tz="UTC").value // 1_000_000)
    day_rows = df[df["ts_ms"] == day_ms]
    if day_rows.empty:
        return None, pos

    row       = day_rows.iloc[-1]
    bar_low   = float(row["low"])
    bar_close = float(row["close"])

    # Research convention: bars_held = i - e_bar at the CURRENT bar.
    # Increment before checks so time exit fires exactly 25 bars after signal.
    pos.bars_held += 1

    if bar_low <= pos.stop_loss:
        gross_r  = (pos.stop_loss - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id": pos.position_id, "symbol": pos.symbol,
            "entry_date": pos.entry_date, "exit_date": str(run_date),
            "entry_price": pos.entry_price, "exit_price": pos.stop_loss,
            "bars_held": pos.bars_held, "gross_r": round(gross_r, 6),
            "pnl_usdt": round(pnl_usdt, 4), "exit_reason": "safety_stop",
        }, pos

    if pos.bars_held >= TIME_EXIT_BARS:
        gross_r  = (bar_close - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id": pos.position_id, "symbol": pos.symbol,
            "entry_date": pos.entry_date, "exit_date": str(run_date),
            "entry_price": pos.entry_price, "exit_price": bar_close,
            "bars_held": pos.bars_held, "gross_r": round(gross_r, 6),
            "pnl_usdt": round(pnl_usdt, 4), "exit_reason": "time_exit_25bars",
        }, pos

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

    # STEP 1: exits
    closed_ids: List[str] = []
    for pid, pos in list(positions.items()):
        df = load_1d_ohlcv(pos.symbol, up_to_date=run_date)
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
                 gross_r=exit_rec["gross_r"], pnl_usdt=exit_rec["pnl_usdt"])
            if dd_pct <= -KILL_SWITCH_DD_PCT:
                kill_sw = True
                _log("KILL_SWITCH", "",
                     f"DD={dd_pct:.2f}% breached -{KILL_SWITCH_DD_PCT}% threshold")
        else:
            positions[pid] = pos
    for pid in closed_ids:
        positions.pop(pid, None)

    # STEP 2: entries
    if not kill_sw:
        cross_syms = t9b_shared.get_cross_system_symbols("rsi_mr_funding")
        arbitrator = SignalArbitrator("rsi_mr_funding", run_date)
        for sym in symbols:
            if any(p.symbol == sym for p in positions.values()):
                continue
            if t9b_shared.normalize_sym(sym) in cross_syms:
                _log("SIGNAL_SKIPPED", sym, "duplicate_cross_system")
                continue

            df  = load_1d_ohlcv(sym, up_to_date=run_date)
            sig = check_entry_signal(sym, df, run_date)
            if sig is None:
                continue
            signals_seen.append(sig)

            rw, rv = t9b_shared.get_regime_weight("rsi_mr_funding")
            regime_scale  = rw * rv * 7
            risk_amount   = min(equity * RISK_PER_TRADE_PCT * regime_scale, CAPITAL_CEILING)
            risk_per_unit = sig["risk_unit"]
            qty           = risk_amount / max(risk_per_unit, EPS)

            if qty * sig["close"] < MIN_ORDER_SIZE_USDT:
                _log("SIGNAL_SKIPPED", sym,
                     f"position_too_small  size=${qty * sig['close']:.2f}")
                continue

            try:
                decision, arb_reason = arbitrator.check_signal(sym, "LONG", risk_amount)
            except Exception as exc:
                decision, arb_reason = "ACCEPTED", f"arbitrator_error:{exc}"
            if decision == "REJECTED":
                _log("SIGNAL_SKIPPED", sym, f"arbitrator: {arb_reason}")
                continue

            pid = f"{sym}_{run_date.strftime('%Y%m%d')}"
            pos = S8Position(
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
                rsi_at_entry          = sig["rsi"],
                funding_at_entry      = sig["funding"],
            )
            positions[pid] = pos
            _log("ENTRY", sym,
                 f"RSI={sig['rsi']:.1f}  funding={sig['funding']:.6f}  "
                 f"price={sig['close']:.6g}  stop={sig['stop_loss']:.6g}  "
                 f"risk=${risk_amount:.2f}  equity=${equity:,.2f}")

    # STEP 3: commit state
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
        "run_date": str(run_date), "equity": equity, "dd_pct": dd_now,
        "open_count": len(positions), "new_signals": len(signals_seen),
        "new_entries": sum(1 for e in log_events if e["event"] == "ENTRY"),
        "exits": len(exits_today), "kill_switch": kill_sw,
        "log_events": log_events, "signals_today": signals_seen,
        "exits_today": exits_today, "positions": positions,
    }


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_open_positions(positions: Dict[str, S8Position], run_date: date) -> None:
    cols = ["position_id","symbol","entry_date","as_of_date","entry_price","stop_loss",
            "bars_held","bars_remaining","rsi_at_entry","funding_at_entry",
            "atr_at_entry","risk_amount_usdt","qty"]
    if not positions:
        pd.DataFrame(columns=cols).to_csv(OPEN_POS_CSV, index=False)
        return
    rows = []
    for pos in positions.values():
        rows.append({
            "position_id": pos.position_id, "symbol": pos.symbol,
            "entry_date": pos.entry_date, "as_of_date": str(run_date),
            "entry_price": pos.entry_price, "stop_loss": pos.stop_loss,
            "bars_held": pos.bars_held,
            "bars_remaining": max(0, TIME_EXIT_BARS - pos.bars_held),
            "rsi_at_entry": pos.rsi_at_entry, "funding_at_entry": pos.funding_at_entry,
            "atr_at_entry": pos.atr_at_entry,
            "risk_amount_usdt": round(pos.risk_amount_usdt, 4), "qty": pos.qty,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    cols = ["symbol","signal_date","close","rsi","atr","funding","stop_loss","signal"]
    if not signals:
        pd.DataFrame(columns=cols).to_csv(SIGNALS_CSV, index=False)
        return
    pd.DataFrame(signals).to_csv(SIGNALS_CSV, index=False)


def append_equity_curve(exits: List[dict]) -> None:
    if not exits:
        return
    rows = [{
        "exit_date": ex["exit_date"], "symbol": ex["symbol"],
        "entry_date": ex["entry_date"], "bars_held": ex["bars_held"],
        "gross_r": ex["gross_r"], "pnl_usdt": ex["pnl_usdt"],
        "equity": ex["equity_after"], "dd_pct": ex["dd_pct"],
        "exit_reason": ex["exit_reason"],
    } for ex in exits]
    append_csv(EQUITY_CSV, rows)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_banner() -> None:
    print("=" * 70)
    print("T9B -- RSI MR FUTURES FUNDING GATE 1D PAPER ENGINE (SYSTEM 8)")
    print("=" * 70)
    print("Frozen config (2026-06-22):")
    print(f"  1D / RSI({RSI_N}) < {OVERSOLD_THR} AND funding < 0 / "
          f"time exit {TIME_EXIT_BARS} bars / ATR stop x{ATR_STOP_MULT}")
    print(f"  Uncapped / {RISK_PER_TRADE_PCT*100:.2f}% risk / ${CAPITAL_CEILING:.0f} ceiling / LONG only")
    print(f"  Universe: {len(S8_UNIVERSE)} symbols / Binance Futures 1D")
    print(f"  Validated: avg_r +0.613R / t=5.28 / PF 3.46 / 2022 best year")
    print(f"  Output: {DATA_DIR}")
    print()


def print_day_summary(result: dict, verbose: bool = True) -> None:
    d, eq, dd = result["run_date"], result["equity"], result["dd_pct"]
    ks = " [KILL-SWITCH]" if result["kill_switch"] else ""
    print(f"  {d}  equity=${eq:>10,.2f}  DD={dd:>+6.2f}%  "
          f"open={result['open_count']}  signals={result['new_signals']}  "
          f"entries={result['new_entries']}  exits={result['exits']}{ks}")
    if verbose:
        for e in result["log_events"]:
            if e["event"] in ("ENTRY", "EXIT", "KILL_SWITCH", "SIGNAL_SKIPPED"):
                print(f"    [{e['event']:<14}] {e['symbol']:<16} {e['detail']}")


def print_final_summary(state: dict) -> None:
    eq  = float(state["paper_equity_usdt"])
    ret = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    print()
    print("-" * 70)
    print("FINAL STATE -- RSI MR Futures Funding Gate 1D Paper (System 8)")
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Total return   : {ret:>+.2f}%")
    print(f"  Current DD     : {float(state['drawdown_pct']):>+.2f}%")
    print(f"  Open positions : {len(state.get('open_positions', []))}  (uncapped)")
    print(f"  Closed trades  : {state.get('closed_trade_count', 0)}")
    print(f"  Kill-switch    : {'TRIGGERED' if state.get('kill_switch_triggered') else 'not triggered'}")
    print()
    print("[PAPER ONLY] No real orders were placed.")


def _print_notify(state: dict, run_date: date) -> None:
    eq           = float(state.get("paper_equity_usdt", INITIAL_CAPITAL))
    ret_pct      = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    n_open       = len(state.get("open_positions", []))
    kill_sw      = state.get("kill_switch_triggered", False)
    days_running = (run_date - FREEZE_DATE).days + 1
    review_date  = FREEZE_DATE + timedelta(days=90)

    print()
    print("=" * 55)
    print(f"T9B RSI MR FUNDING (SYSTEM 8) DAILY UPDATE -- {run_date}")
    print("=" * 55)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)")
    print(f"Peak equity   : ${float(state.get('peak_equity_usdt', INITIAL_CAPITAL)):>10,.2f}")
    print(f"Max drawdown  : {float(state.get('drawdown_pct', 0.0)):>+.2f}%")
    print(f"Open positions: {n_open}  (uncapped)")
    print(f"Closed trades : {state.get('closed_trade_count', 0)}")
    print(f"Days running  : {days_running}")
    print(f"Days to review: {(review_date - run_date).days}  (3-month: {review_date})")
    print()
    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN POSITIONS ({len(positions)})")
        for p in sorted(positions, key=lambda x: x.get("bars_held", 0), reverse=True):
            rem = max(0, TIME_EXIT_BARS - int(p.get("bars_held", 0)))
            print(f"  {p.get('symbol','?'):<16} entry={p.get('entry_date','?')}  "
                  f"held={p.get('bars_held',0)}  rem={rem}")
    else:
        print("OPEN POSITIONS: none")
    print()
    if kill_sw:
        print("STATUS: KILL-SWITCH ACTIVE")
    else:
        print("STATUS: OK")
    print("=" * 55)


# =============================================================================
# CLI + MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T9B RSI MR Futures Funding Gate Paper Engine (S8)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD")
    grp.add_argument("--backfill", action="store_true",
                     help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--notify", action="store_true")
    return ap.parse_args()


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
        print(f"[RESET] State cleared. Starting from ${INITIAL_CAPITAL:,.0f}.")
        print()

    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    symbols = S8_UNIVERSE
    print(f"[UNIVERSE] {len(symbols)} symbols with 1D OHLCV + funding data")

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
    print()

    state = load_state()
    print(f"[STATE] equity=${float(state['paper_equity_usdt']):,.2f}  "
          f"open={len(state.get('open_positions', []))}  "
          f"closed={state.get('closed_trade_count', 0)}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}", flush=True)
    print()
    _reconcile_state(state, run_dates[0] if run_dates else yesterday)
    print()

    for run_date in run_dates:
        result = run_one_day(run_date, symbols, state)
        save_state(state)
        write_open_positions(result["positions"], run_date)
        write_signals_today(result["signals_today"], run_date)
        append_equity_curve(result["exits_today"])
        append_csv(DAILY_LOG_CSV, result["log_events"])
        print_day_summary(result, verbose=verbose)
        if result["kill_switch"]:
            print("\n[HALT] Kill-switch triggered. No new entries until reviewed.", flush=True)
            break

    print_final_summary(state)

    if args.notify:
        _print_notify(state, run_dates[-1] if run_dates else yesterday)

    return 0


if __name__ == "__main__":
    t9b_shared.run_engine("rsi_mr_funding", main)
