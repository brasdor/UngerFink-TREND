#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- MACROSS SHORT FUNDING GATE 4H PAPER ENGINE (SYSTEM 8)
==================================================================

Paper trading engine for the frozen MACrossShort_FundingGate config.
System 8 in portfolio. Short-side trend specialist on Binance Futures 4H.

Frozen config (2026-06-12):
  Universe    : 101+ Binance Futures USDT symbols (auto-discovered from funding cache)
  Timeframe   : 4H (Binance Futures USD-M perpetuals)
  Entry       : EMA(20) crosses BELOW EMA(30) on bar close
                AND close < EMA200
                AND funding_rate >= 0.01% at signal bar
  Stop        : ATR(14) x 2.0 ABOVE entry price (hard stop for short)
  Exit        : Fixed time exit after 35 x 4H bars (~5.8 calendar days)
  Risk/trade  : 0.25% of current equity, $150 ceiling
  Leverage    : 1.0x  (Binance Futures, SHORT only)
  Cap         : Uncapped
  Validated   : avg_r +0.3145R / t=11.3 / CAGR +46.4% / maxDD -20.96%
  CAVEAT      : 2026 partial -31.6R -- monitor; win rate 47.2% (structural)

Entry is at open of the bar AFTER the signal fires.
Exit checks stop (bar high >= stop_loss) then time (bars_held >= 35).

Usage:
  python phase_t9b_macross_short_paper_engine.py
      --date 2026-06-12    run for a specific date (processes 4H bars up to midnight UTC)
      --backfill           replay from freeze date (2026-06-12) to yesterday
      --no-download        use committed caches only
      --reset              wipe state and restart from $10,000
      --notify             compact notification summary (for CI)

Output files (data/t9b_macross_paper/):
  state.json            persistent engine state
  open_positions.csv    current open short positions
  pending_entries.csv   signals awaiting next-bar entry
  signals_today.csv     signals detected in this run
  equity_curve.csv      one row per closed trade
  daily_log.csv         all events: entries, exits, skipped signals

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

import numpy as np
import pandas as pd

import t9b_shared
from signal_arbitrator import SignalArbitrator


# =============================================================================
# FROZEN CONFIG  (2026-06-12)
# =============================================================================

SYSTEM_NAME         = "MACrossShort_4H_T9B"
FREEZE_DATE         = date(2026, 6, 12)

FAST_EMA            = 20
SLOW_EMA            = 30
EMA_N               = 200     # close must be BELOW EMA200
ATR_N               = 14
ATR_STOP_MULT       = 2.0     # stop ABOVE entry for short
HOLD_BARS           = 35      # time exit after 35 x 4H bars
FUNDING_THRESHOLD   = 0.0001  # 0.01% per 8h period -- discovery constraint

RISK_PER_TRADE_PCT  = 0.0025  # 0.25%
CAPITAL_CEILING     = 150.0   # USD ceiling on risk per trade
MIN_ORDER_USDT      = 15.0
MAX_OPEN_POSITIONS  = None    # uncapped
INITIAL_CAPITAL     = 10_000.0
LEVERAGE            = 1.0
KILL_SWITCH_DD_PCT  = 35.0

LIMIT_BARS          = 3000
MIN_BARS_REQUIRED   = EMA_N + 100   # EMA200 warmup + margin

EPS                 = 1e-12


# =============================================================================
# PATHS
# =============================================================================

ROOT              = Path.cwd()
DATA_DIR          = ROOT / "data" / "t9b_macross_paper"
STATE_PATH        = DATA_DIR / "state.json"
OPEN_POS_CSV      = DATA_DIR / "open_positions.csv"
PENDING_CSV       = DATA_DIR / "pending_entries.csv"
SIGNALS_CSV       = DATA_DIR / "signals_today.csv"
EQUITY_CSV        = DATA_DIR / "equity_curve.csv"
DAILY_LOG_CSV     = DATA_DIR / "daily_log.csv"

CACHE_4H          = ROOT / "data" / "futures_universe" / "ohlcv_4h"
CACHE_FUNDING     = ROOT / "data" / "futures_universe" / "funding_rates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_OHLCV:   dict = {}
_SESSION_FUNDING: dict = {}
_SKIP_LIVE_FETCH: bool = False


# =============================================================================
# UNIVERSE
# =============================================================================

def _discover_universe() -> List[str]:
    h4_syms = {f.stem.replace("_4h", "") for f in CACHE_4H.glob("*_4h.csv")}
    fr_syms = {f.stem.replace("_funding", "") for f in CACHE_FUNDING.glob("*_funding.csv")}
    return sorted(h4_syms & fr_syms)

MC_UNIVERSE: List[str] = _discover_universe()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MCShortPosition:
    position_id:           str
    symbol:                str
    entry_ts_ms:           int
    entry_date:            str
    entry_price:           float
    stop_loss:             float   # ABOVE entry for short
    initial_risk_per_unit: float
    risk_amount_usdt:      float
    qty:                   float
    bars_held:             int
    atr_at_entry:          float
    funding_at_entry:      float
    fast_ema_at_entry:     float
    slow_ema_at_entry:     float

    def current_r(self, price: float) -> float:
        return (self.entry_price - price) / max(self.initial_risk_per_unit, EPS)


# =============================================================================
# STATE HELPERS
# =============================================================================

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _default_state()


def _default_state() -> dict:
    # Seed last_processed_ts_ms to midnight UTC of the day before freeze so the
    # first run starts paper trading at the freeze date instead of replaying
    # the entire cached history (System 7 state was seeded the same way).
    seed_ts_ms = int(pd.Timestamp(str(FREEZE_DATE - timedelta(days=1)),
                                  tz="UTC").value // 1_000_000)
    return {
        "system_name":           SYSTEM_NAME,
        "created_utc":           pd.Timestamp.utcnow().isoformat(),
        "last_run_date":         None,
        "last_processed_ts_ms":  seed_ts_ms,
        "paper_equity_usdt":     INITIAL_CAPITAL,
        "peak_equity_usdt":      INITIAL_CAPITAL,
        "drawdown_pct":          0.0,
        "kill_switch_triggered": False,
        "open_positions":        [],
        "pending_entries":       [],
        "closed_trade_count":    0,
    }


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


def positions_from_state(state: dict) -> Dict[str, MCShortPosition]:
    fields = {f.name for f in dataclasses.fields(MCShortPosition)}
    out: Dict[str, MCShortPosition] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = MCShortPosition(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping position {raw.get('position_id','?')}: {exc}", flush=True)
    return out


def positions_to_state(state: dict, positions: Dict[str, MCShortPosition]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def _reconcile_state(state: dict) -> None:
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
            if stop is not None and ep is not None and float(stop) <= float(ep):
                issues.append(f"{sym}: stop={stop} <= entry={ep} (invalid for SHORT)")
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
    else:
        print(f"[RECONCILE] OK  ({len(positions)} position(s) verified)", flush=True)


# =============================================================================
# OHLCV LOADING  (4H Binance Futures cache + live API update)
# =============================================================================

def load_4h_ohlcv(symbol: str, cutoff_ts_ms: Optional[int] = None) -> pd.DataFrame:
    global _SESSION_OHLCV
    if symbol not in _SESSION_OHLCV:
        cache_path = CACHE_4H / f"{symbol}_4h.csv"
        df_base: pd.DataFrame = pd.DataFrame()
        if cache_path.exists():
            try:
                df_base = _parse_ohlcv_4h(pd.read_csv(cache_path))
            except Exception as exc:
                print(f"  [WARN] {symbol}: 4H cache read error -- {exc}", flush=True)

        df_live: pd.DataFrame = pd.DataFrame()
        if not _SKIP_LIVE_FETCH:
            df_live = _fetch_binance_4h(symbol, limit=30) or pd.DataFrame()

        if not df_base.empty and not df_live.empty:
            merged = (pd.concat([df_base, df_live], ignore_index=True)
                      .drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True))
        elif not df_base.empty:
            merged = df_base
        elif not df_live.empty:
            merged = df_live
        else:
            merged = pd.DataFrame()

        if not merged.empty and not df_live.empty and cache_path.parent.exists():
            merged[["ts_ms", "open", "high", "low", "close", "volume"]].to_csv(
                cache_path, index=False)

        if len(merged) > LIMIT_BARS:
            merged = merged.iloc[-LIMIT_BARS:].reset_index(drop=True)

        _SESSION_OHLCV[symbol] = merged

    df = _SESSION_OHLCV[symbol]
    if cutoff_ts_ms is not None:
        df = df[df["ts_ms"] < cutoff_ts_ms].copy()
    return df.reset_index(drop=True)


def _parse_ohlcv_4h(raw: pd.DataFrame) -> pd.DataFrame:
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


def _fetch_binance_4h(symbol: str, limit: int = 30) -> Optional[pd.DataFrame]:
    try:
        import requests
        url = "https://fapi.binance.com/fapi/v1/klines"
        r = requests.get(url, params={"symbol": symbol, "interval": "4h", "limit": limit},
                         timeout=10)
        r.raise_for_status()
        rows = []
        for k in r.json():
            rows.append({"ts_ms": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
        return pd.DataFrame(rows) if rows else None
    except Exception as exc:
        print(f"    [binance4h] {symbol}: {exc}", flush=True)
        return None


# =============================================================================
# FUNDING RATE  (cached CSV + live API)
# =============================================================================

def load_funding_rates(symbol: str) -> pd.DataFrame:
    global _SESSION_FUNDING
    if symbol not in _SESSION_FUNDING:
        path = CACHE_FUNDING / f"{symbol}_funding.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "funding_time" not in df.columns or "funding_rate" not in df.columns:
                    _SESSION_FUNDING[symbol] = pd.DataFrame()
                else:
                    df["funding_time"] = pd.to_numeric(df["funding_time"], errors="coerce").astype("Int64")
                    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
                    df = df.dropna(subset=["funding_time", "funding_rate"])
                    df = df.sort_values("funding_time").reset_index(drop=True)
                    df["funding_time"] = df["funding_time"].astype(int)
                    _SESSION_FUNDING[symbol] = df
            except Exception as exc:
                print(f"  [WARN] {symbol}: funding cache read error -- {exc}", flush=True)
                _SESSION_FUNDING[symbol] = pd.DataFrame()
        else:
            _SESSION_FUNDING[symbol] = pd.DataFrame()
    return _SESSION_FUNDING[symbol]


_CURRENT_FUNDING: dict = {}


def _fetch_all_funding_rates(symbols: List[str]) -> None:
    global _CURRENT_FUNDING
    if _SKIP_LIVE_FETCH:
        return
    try:
        import requests
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
        r.raise_for_status()
        for item in r.json():
            sym = item.get("symbol", "")
            try:
                _CURRENT_FUNDING[sym] = float(item.get("lastFundingRate", "0"))
            except (TypeError, ValueError):
                pass
        fetched = sum(1 for s in symbols if s in _CURRENT_FUNDING)
        print(f"[FUNDING] Fetched current rates for {fetched}/{len(symbols)} symbols", flush=True)
    except Exception as exc:
        print(f"[FUNDING] Live fetch failed: {exc}  (using cached rates only)", flush=True)


def get_funding_at_ts(symbol: str, ts_ms: int) -> float:
    df = load_funding_rates(symbol)
    if not df.empty:
        prior = df[df["funding_time"] <= ts_ms]
        if not prior.empty:
            return float(prior.iloc[-1]["funding_rate"])
    return _CURRENT_FUNDING.get(symbol, 0.0)


# =============================================================================
# INDICATORS  (EMA20/EMA30 cross-down, EMA200, ATR14)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return df
    out = df.copy()
    cls = out["close"]
    hgh = out["high"]
    lw  = out["low"]

    prev_c = cls.shift(1)
    tr = pd.concat([hgh - lw, (hgh - prev_c).abs(), (lw - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_N).mean()

    out["ema_fast"] = cls.ewm(span=FAST_EMA, adjust=False).mean()
    out["ema_slow"] = cls.ewm(span=SLOW_EMA, adjust=False).mean()
    out["ema200"]   = cls.ewm(span=EMA_N, adjust=False).mean()
    out["ema_fast_prev"] = out["ema_fast"].shift(1)
    out["ema_slow_prev"] = out["ema_slow"].shift(1)
    return out


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_signal(row: pd.Series, funding_rate: float) -> Optional[dict]:
    """
    Signal conditions on a single 4H bar (checked at bar close):
      1. ema_fast crosses BELOW ema_slow on this bar
         (prev bar: fast >= slow; this bar: fast < slow)
      2. close < EMA200
      3. funding_rate >= 0.01%

    Entry: at OPEN of the NEXT 4H bar after signal.
    Stop : entry + ATR x 2.0 (computed at fill from signal-bar ATR)
    """
    ema_f      = row.get("ema_fast", np.nan)
    ema_s      = row.get("ema_slow", np.nan)
    ema_f_prev = row.get("ema_fast_prev", np.nan)
    ema_s_prev = row.get("ema_slow_prev", np.nan)
    close      = float(row["close"])
    ema200     = row.get("ema200", np.nan)
    atr        = row.get("atr", np.nan)

    if not (np.isfinite(ema_f) and np.isfinite(ema_s) and
            np.isfinite(ema_f_prev) and np.isfinite(ema_s_prev)):
        return None
    if not (ema_f < ema_s and ema_f_prev >= ema_s_prev):
        return None  # no cross-down on this bar
    if not np.isfinite(ema200) or close >= ema200:
        return None  # must be below EMA200
    if not np.isfinite(atr) or atr <= EPS:
        return None
    if funding_rate < FUNDING_THRESHOLD:
        return None  # funding gate

    return {
        "signal_close": round(close, 8),
        "ema_fast":     round(float(ema_f), 8),
        "ema_slow":     round(float(ema_s), 8),
        "ema200":       round(float(ema200), 8),
        "atr":          round(float(atr), 8),
        "stop_loss":    round(close + ATR_STOP_MULT * float(atr), 8),
        "risk_unit":    round(ATR_STOP_MULT * float(atr), 8),
        "funding_rate": float(funding_rate),
    }


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def check_exit(pos: MCShortPosition, bar: pd.Series) -> Optional[dict]:
    """
    Exit priority:
      1. Stop hit  : bar HIGH >= stop_loss -> exit at stop_loss (loss = -1R)
      2. Time exit : bars_held >= 35       -> exit at bar close
    """
    bar_high  = float(bar["high"])
    bar_close = float(bar["close"])
    ru        = max(pos.initial_risk_per_unit, EPS)

    if bar_high >= pos.stop_loss:
        gross_r  = (pos.entry_price - pos.stop_loss) / ru
        return {
            "exit_reason": "stop_hit",
            "exit_price":  round(pos.stop_loss, 8),
            "gross_r":     round(gross_r, 6),
            "pnl_usdt":    round(pos.risk_amount_usdt * gross_r, 4),
        }

    if pos.bars_held >= HOLD_BARS:
        gross_r = (pos.entry_price - bar_close) / ru
        return {
            "exit_reason": "time_exit_35bars",
            "exit_price":  round(bar_close, 8),
            "gross_r":     round(gross_r, 6),
            "pnl_usdt":    round(pos.risk_amount_usdt * gross_r, 4),
        }

    return None


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def run_daily(run_date: date, symbols: List[str], state: dict) -> dict:
    equity    = float(state["paper_equity_usdt"])
    peak      = float(state["peak_equity_usdt"])
    kill_sw   = bool(state.get("kill_switch_triggered", False))
    positions = positions_from_state(state)
    pending   = list(state.get("pending_entries", []))

    last_ts_ms   = int(state.get("last_processed_ts_ms", 0))
    cutoff_ts_ms = int(pd.Timestamp(str(run_date), tz="UTC").value // 1_000_000)

    log_events:  List[dict] = []
    signals_run: List[dict] = []
    exits_run:   List[dict] = []
    max_ts_seen: int = last_ts_ms
    cross_skip_logged: set = set()

    def _log(event, symbol, detail, **extra):
        log_events.append({"run_date": str(run_date), "bar_ts_ms": max_ts_seen,
                           "event": event, "symbol": symbol, "detail": detail, **extra})

    cross_syms = t9b_shared.get_cross_system_symbols("macross")
    arbitrator = SignalArbitrator("macross", run_date)

    for sym in symbols:
        df_raw = load_4h_ohlcv(sym, cutoff_ts_ms=cutoff_ts_ms)
        if df_raw.empty or len(df_raw) < MIN_BARS_REQUIRED:
            continue

        df = add_indicators(df_raw)
        new_df = df[df["ts_ms"] > last_ts_ms].copy().reset_index(drop=True)
        if new_df.empty:
            continue

        for i in range(len(new_df)):
            bar    = new_df.iloc[i]
            bar_ts = int(bar["ts_ms"])
            max_ts_seen = max(max_ts_seen, bar_ts)

            # ---- 1. Fill pending entries at this bar's open ----
            filled_pids = []
            for pe in pending:
                if pe["symbol"] != sym:
                    continue
                if pe["signal_ts_ms"] >= bar_ts:
                    continue

                entry_price = float(bar["open"])
                risk_unit   = ATR_STOP_MULT * float(pe["atr"])
                stop_loss   = entry_price + risk_unit
                risk_amount = min(equity * RISK_PER_TRADE_PCT, CAPITAL_CEILING)
                qty         = risk_amount / max(risk_unit, EPS)

                if qty * entry_price < MIN_ORDER_USDT:
                    _log("SIGNAL_SKIPPED", sym,
                         f"position_too_small  size=${qty*entry_price:.2f} < ${MIN_ORDER_USDT}",
                         entry_price=entry_price)
                    filled_pids.append(id(pe))
                    continue
                if kill_sw:
                    _log("SIGNAL_SKIPPED", sym, "kill_switch_active")
                    filled_pids.append(id(pe))
                    continue

                # Signal Arbitration Manager (Rules 1-6)
                decision, arb_reason = arbitrator.check_signal(sym, "SHORT", risk_amount)
                if decision == "REJECTED":
                    _log("SIGNAL_SKIPPED", sym, f"arbitrator: {arb_reason}",
                         arbitrator_reject_reason=arb_reason)
                    filled_pids.append(id(pe))
                    continue

                pid = f"{sym}_{bar_ts}"
                pos = MCShortPosition(
                    position_id           = pid,
                    symbol                = sym,
                    entry_ts_ms           = bar_ts,
                    entry_date            = str(pd.Timestamp(bar_ts, unit="ms", tz="UTC").date()),
                    entry_price           = round(entry_price, 8),
                    stop_loss             = round(stop_loss, 8),
                    initial_risk_per_unit = round(risk_unit, 8),
                    risk_amount_usdt      = round(risk_amount, 4),
                    qty                   = round(qty, 8),
                    bars_held             = 0,
                    atr_at_entry          = pe["atr"],
                    funding_at_entry      = pe["funding_rate"],
                    fast_ema_at_entry     = pe["ema_fast"],
                    slow_ema_at_entry     = pe["ema_slow"],
                )
                positions[pid] = pos
                filled_pids.append(id(pe))
                _log("ENTRY", sym,
                     f"price={entry_price:.6g}  stop={stop_loss:.6g}  "
                     f"risk=${risk_amount:.2f}  funding={pe['funding_rate']*100:.4f}%  "
                     f"equity=${equity:,.2f}",
                     entry_price=round(entry_price, 8), stop=round(stop_loss, 8),
                     risk_amount=round(risk_amount, 4), equity=round(equity, 2))

            pending = [pe for pe in pending if id(pe) not in filled_pids]

            # ---- 2. Process exits ----
            closed_pids: List[str] = []
            for pid, pos in list(positions.items()):
                if pos.symbol != sym:
                    continue
                exit_rec = check_exit(pos, bar)
                if exit_rec is not None:
                    equity += exit_rec["pnl_usdt"]
                    peak    = max(peak, equity)
                    dd_pct  = (equity - peak) / max(peak, EPS) * 100.0
                    exit_rec.update({
                        "position_id":  pid,
                        "symbol":       sym,
                        "entry_date":   pos.entry_date,
                        "exit_date":    str(pd.Timestamp(bar_ts, unit="ms", tz="UTC").date()),
                        "entry_price":  pos.entry_price,
                        "stop_loss":    pos.stop_loss,
                        "bars_held":    pos.bars_held,
                        "equity_after": round(equity, 4),
                        "dd_pct":       round(dd_pct, 4),
                    })
                    exits_run.append(exit_rec)
                    closed_pids.append(pid)
                    _log("EXIT", sym,
                         f"reason={exit_rec['exit_reason']}  bars={pos.bars_held}  "
                         f"R={exit_rec['gross_r']:.3f}  pnl=${exit_rec['pnl_usdt']:.2f}  "
                         f"equity=${equity:,.2f}",
                         gross_r=exit_rec["gross_r"], pnl_usdt=exit_rec["pnl_usdt"],
                         equity=round(equity, 2), dd_pct=round(dd_pct, 4))
                    if dd_pct <= -KILL_SWITCH_DD_PCT:
                        kill_sw = True
                        _log("KILL_SWITCH", "", f"DD={dd_pct:.2f}% breached -{KILL_SWITCH_DD_PCT}%",
                             equity=round(equity, 2))
                else:
                    pos.bars_held += 1

            for pid in closed_pids:
                positions.pop(pid, None)

            # ---- 3. New entry signals ----
            if kill_sw:
                continue
            if any(p.symbol == sym for p in positions.values()):
                continue
            if any(pe["symbol"] == sym for pe in pending):
                continue
            if t9b_shared.normalize_sym(sym) in cross_syms:
                if sym not in cross_skip_logged:
                    _log("SIGNAL_SKIPPED", sym, "duplicate_cross_system")
                    cross_skip_logged.add(sym)
                continue

            funding = get_funding_at_ts(sym, bar_ts)
            sig = check_signal(bar, funding)
            if sig is not None:
                pe = {"symbol": sym, "signal_ts_ms": bar_ts, **sig}
                pending.append(pe)
                signals_run.append({**pe, "run_date": str(run_date)})
                _log("SIGNAL", sym,
                     f"cross_down ema{FAST_EMA}<ema{SLOW_EMA}  close={sig['signal_close']:.6g}  "
                     f"ema200={sig['ema200']:.6g}  funding={sig['funding_rate']*100:.4f}%  "
                     f"stop={sig['stop_loss']:.6g}",
                     close=sig["signal_close"], funding=sig["funding_rate"])

    dd_now = (equity - peak) / max(peak, EPS) * 100.0
    state.update({
        "last_run_date":         str(run_date),
        "last_processed_ts_ms":  max_ts_seen,
        "paper_equity_usdt":     equity,
        "peak_equity_usdt":      peak,
        "drawdown_pct":          round(dd_now, 4),
        "kill_switch_triggered": kill_sw,
        "closed_trade_count":    state.get("closed_trade_count", 0) + len(exits_run),
        "pending_entries":       pending,
    })
    positions_to_state(state, positions)

    return {
        "run_date":      str(run_date),
        "equity":        equity,
        "peak":          peak,
        "dd_pct":        dd_now,
        "open_count":    len(positions),
        "pending_count": len(pending),
        "new_signals":   len(signals_run),
        "new_entries":   sum(1 for e in log_events if e["event"] == "ENTRY"),
        "exits":         len(exits_run),
        "kill_switch":   kill_sw,
        "log_events":    log_events,
        "signals_run":   signals_run,
        "exits_run":     exits_run,
        "positions":     positions,
        "pending":       pending,
    }


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_open_positions(positions: Dict[str, MCShortPosition], run_date: date) -> None:
    cols = ["position_id", "symbol", "entry_date", "as_of_date", "entry_price",
            "stop_loss", "bars_held", "bars_remaining", "atr_at_entry",
            "funding_at_entry", "fast_ema_at_entry", "slow_ema_at_entry",
            "risk_amount_usdt", "qty"]
    if not positions:
        pd.DataFrame(columns=cols).to_csv(OPEN_POS_CSV, index=False)
        return
    rows = []
    for pos in positions.values():
        rows.append({
            "position_id":       pos.position_id,
            "symbol":            pos.symbol,
            "entry_date":        pos.entry_date,
            "as_of_date":        str(run_date),
            "entry_price":       pos.entry_price,
            "stop_loss":         pos.stop_loss,
            "bars_held":         pos.bars_held,
            "bars_remaining":    max(0, HOLD_BARS - pos.bars_held),
            "atr_at_entry":      pos.atr_at_entry,
            "funding_at_entry":  round(pos.funding_at_entry * 100, 6),
            "fast_ema_at_entry": pos.fast_ema_at_entry,
            "slow_ema_at_entry": pos.slow_ema_at_entry,
            "risk_amount_usdt":  round(pos.risk_amount_usdt, 4),
            "qty":               pos.qty,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_pending_entries(pending: List[dict], run_date: date) -> None:
    cols = ["symbol", "signal_ts_ms", "signal_close", "stop_loss",
            "ema_fast", "ema_slow", "atr", "funding_rate"]
    if not pending:
        pd.DataFrame(columns=cols).to_csv(PENDING_CSV, index=False)
        return
    pd.DataFrame([{c: pe.get(c, 0) for c in cols} for pe in pending]).to_csv(
        PENDING_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    cols = ["symbol", "run_date", "signal_ts_ms", "signal_close", "ema_fast",
            "ema_slow", "ema200", "atr", "funding_rate", "stop_loss"]
    if not signals:
        pd.DataFrame(columns=cols).to_csv(SIGNALS_CSV, index=False)
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
        "entry_price": ex["entry_price"],
        "exit_price":  ex["exit_price"],
        "gross_r":     ex["gross_r"],
        "pnl_usdt":    ex["pnl_usdt"],
        "equity":      ex["equity_after"],
        "dd_pct":      ex["dd_pct"],
        "exit_reason": ex["exit_reason"],
    } for ex in exits]
    append_csv(EQUITY_CSV, rows)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_banner() -> None:
    print("=" * 70, flush=True)
    print("T9B -- MACROSS SHORT FUNDING GATE 4H PAPER ENGINE (SYSTEM 8)", flush=True)
    print("=" * 70, flush=True)
    print(f"Frozen config (2026-06-12):", flush=True)
    print(f"  4H / EMA{FAST_EMA} crosses below EMA{SLOW_EMA} / EMA{EMA_N} below /"
          f" funding>={FUNDING_THRESHOLD*100:.2f}% /"
          f" stop ATRx{ATR_STOP_MULT} above / time exit {HOLD_BARS} bars", flush=True)
    print(f"  Uncapped / {RISK_PER_TRADE_PCT*100:.2f}% risk / $150 ceiling / SHORT only", flush=True)
    print(f"  Universe: {len(MC_UNIVERSE)} symbols / Binance Futures 4H", flush=True)
    print(f"  Validated: avg_r +0.3145R / t=11.3 / CAGR +46.4% / maxDD -20.96%", flush=True)
    print(f"  Output: {DATA_DIR}", flush=True)
    print(flush=True)


def print_day_summary(result: dict, verbose: bool = True) -> None:
    d, eq, dd = result["run_date"], result["equity"], result["dd_pct"]
    op, pe, si, en, ex = (result["open_count"], result["pending_count"],
                          result["new_signals"], result["new_entries"], result["exits"])
    ks = " [KILL-SWITCH]" if result["kill_switch"] else ""
    print(f"  {d}  equity=${eq:>10,.2f}  DD={dd:>+6.2f}%  "
          f"open={op}  pending={pe}  signals={si}  entries={en}  exits={ex}{ks}", flush=True)
    if verbose:
        for e in result["log_events"]:
            if e["event"] in ("ENTRY", "EXIT", "KILL_SWITCH", "SIGNAL", "SIGNAL_SKIPPED"):
                sym = e["symbol"] if e["symbol"] else ""
                print(f"    [{e['event']:<16}] {sym:<18} {e['detail']}", flush=True)


def print_final_summary(state: dict) -> None:
    eq  = float(state["paper_equity_usdt"])
    ret = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    dd  = float(state["drawdown_pct"])
    print(flush=True)
    print("-" * 70, flush=True)
    print("FINAL STATE -- MACross Short Funding Gate 4H Paper (System 8)", flush=True)
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})", flush=True)
    print(f"  Total return   : {ret:>+.2f}%", flush=True)
    print(f"  Current DD     : {dd:>+.2f}%", flush=True)
    print(f"  Open positions : {len(state.get('open_positions', []))}  (uncapped)", flush=True)
    print(f"  Pending entries: {len(state.get('pending_entries', []))}", flush=True)
    print(f"  Closed trades  : {state.get('closed_trade_count', 0)}", flush=True)
    print(f"  Kill-switch    : {'TRIGGERED' if state.get('kill_switch_triggered') else 'not triggered'}", flush=True)
    print(flush=True)
    print("[PAPER ONLY] No real orders were placed.", flush=True)


def _print_notify(state: dict, run_date: date) -> None:
    eq           = float(state.get("paper_equity_usdt", INITIAL_CAPITAL))
    peak         = float(state.get("peak_equity_usdt", INITIAL_CAPITAL))
    dd           = float(state.get("drawdown_pct", 0.0))
    ret_pct      = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    n_open       = len(state.get("open_positions", []))
    n_pending    = len(state.get("pending_entries", []))
    n_closed     = int(state.get("closed_trade_count", 0))
    kill_sw      = state.get("kill_switch_triggered", False)
    days_running = (run_date - FREEZE_DATE).days + 1
    review_date  = FREEZE_DATE + timedelta(days=90)
    days_to_rev  = (review_date - run_date).days

    print(flush=True)
    print("=" * 55, flush=True)
    print(f"T9B MACROSS SHORT (SYSTEM 8) DAILY UPDATE -- {run_date}", flush=True)
    print("=" * 55, flush=True)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)", flush=True)
    print(f"Peak equity   : ${peak:>10,.2f}", flush=True)
    print(f"Max drawdown  : {dd:>+.2f}%", flush=True)
    print(f"Open positions: {n_open}  (uncapped)", flush=True)
    print(f"Pending fills : {n_pending}", flush=True)
    print(f"Closed trades : {n_closed}", flush=True)
    print(f"Days running  : {days_running}", flush=True)
    print(f"Days to review: {days_to_rev}  (3-month: {review_date})", flush=True)
    if kill_sw:
        print("KILL-SWITCH   : TRIGGERED -- no new entries", flush=True)
    print(flush=True)

    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN SHORT POSITIONS ({len(positions)})", flush=True)
        print(f"  {'Symbol':<18} {'Entry':<12} {'Stop':>12} {'Held':>5} {'Rem':>5} {'Funding%':>9}", flush=True)
        for p in sorted(positions, key=lambda x: x.get("bars_held", 0), reverse=True):
            sym   = str(p.get("symbol", "?"))
            edate = str(p.get("entry_date", "?"))
            stop  = p.get("stop_loss", 0)
            held  = int(p.get("bars_held", 0))
            rem   = max(0, HOLD_BARS - held)
            fr    = float(p.get("funding_at_entry", 0)) * 100
            print(f"  {sym:<18} {edate:<12} {stop:>12.6g} {held:>5} {rem:>5} {fr:>8.4f}%", flush=True)
    else:
        print("OPEN SHORT POSITIONS: none", flush=True)
    print(flush=True)

    pending = state.get("pending_entries", [])
    if pending:
        print(f"PENDING ENTRIES ({len(pending)})", flush=True)
        for pe in pending:
            sym   = str(pe.get("symbol", "?"))
            close = pe.get("signal_close", 0)
            stop  = pe.get("stop_loss", 0)
            fr    = float(pe.get("funding_rate", 0)) * 100
            print(f"  {sym:<18}  signal_close={close:<12.6g}  stop={stop:<12.6g}  funding={fr:.4f}%", flush=True)
    else:
        print("PENDING ENTRIES: none", flush=True)
    print(flush=True)

    if kill_sw:
        print("STATUS: KILL-SWITCH ACTIVE", flush=True)
    elif dd < -20.0:
        print("STATUS: WARN -- drawdown > 20%", flush=True)
    else:
        print("STATUS: OK", flush=True)
    print("=" * 55, flush=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="T9B MACross Short Funding Gate 4H Paper Engine (System 8)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD",
                     help="Process 4H bars up to midnight UTC of this date")
    grp.add_argument("--backfill", action="store_true",
                     help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip live fetch; use committed caches only")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe state.json and restart from $10,000")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-event detail")
    ap.add_argument("--notify", action="store_true",
                    help="Print compact notification summary (for CI)")
    return ap.parse_args()


def main() -> int:
    args    = parse_args()
    verbose = not args.quiet

    print_banner()

    if args.reset:
        for f in [STATE_PATH, EQUITY_CSV, DAILY_LOG_CSV, PENDING_CSV]:
            if f.exists():
                f.unlink()
        print("[RESET] State and logs cleared. Starting from $10,000.", flush=True)
        print(flush=True)

    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    symbols = MC_UNIVERSE
    print(f"[UNIVERSE] {len(symbols)} symbols with 4H OHLCV + funding data", flush=True)

    today     = date.today()
    yesterday = today - timedelta(days=1)

    if args.backfill:
        run_dates = []
        d = FREEZE_DATE
        while d <= yesterday:
            run_dates.append(d)
            d += timedelta(days=1)
        print(f"[BACKFILL] {len(run_dates)} days: {FREEZE_DATE} to {yesterday}", flush=True)
    elif args.date:
        run_dates = [date.fromisoformat(args.date)]
        print(f"[MODE] Single date: {args.date}", flush=True)
    else:
        run_dates = [yesterday]
        print(f"[MODE] Yesterday: {yesterday}", flush=True)

    print(flush=True)
    _fetch_all_funding_rates(symbols)
    print(flush=True)

    state = load_state()
    print(f"[STATE] equity=${state['paper_equity_usdt']:,.2f}  "
          f"open={len(state.get('open_positions', []))}  "
          f"pending={len(state.get('pending_entries', []))}  "
          f"closed={state.get('closed_trade_count', 0)}  "
          f"last_ts_ms={state.get('last_processed_ts_ms', 0)}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}", flush=True)
    print(flush=True)

    _reconcile_state(state)
    print(flush=True)

    for run_date in run_dates:
        result = run_daily(run_date, symbols, state)
        save_state(state)
        write_open_positions(result["positions"], run_date)
        write_pending_entries(result["pending"], run_date)
        write_signals_today(result["signals_run"], run_date)
        append_equity_curve(result["exits_run"])
        append_csv(DAILY_LOG_CSV, result["log_events"])
        print_day_summary(result, verbose=verbose)
        if result["kill_switch"]:
            print(flush=True)
            print("[HALT] Kill-switch triggered. No new entries until reviewed.", flush=True)
            break

    print_final_summary(state)

    if args.notify:
        _print_notify(state, run_dates[-1] if run_dates else yesterday)

    return 0


if __name__ == "__main__":
    t9b_shared.run_engine("macross", main)
