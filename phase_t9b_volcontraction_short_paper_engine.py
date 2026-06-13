#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- VOLCONTRACTIONSHORT FUNDING GATE 4H PAPER ENGINE
==============================================================

Paper trading engine for the frozen VolContractionShort_FundingGate config.
System 7 in portfolio. Short-side specialist on Binance Futures 4H.

Frozen config (2026-06-11):
  Universe    : 101+ Binance Futures USDT symbols (auto-discovered from funding cache)
  Timeframe   : 4H (Binance Futures USD-M perpetuals)
  Entry       : ATR(14) < 20th pct for >= 15 consecutive bars
                AND close breaks below compression-period lowest low
                AND close < EMA200
                AND funding_rate >= 0.01% at signal bar
  Stop        : ATR(14) x 2.0 ABOVE entry price (hard stop for short)
  Exit        : Fixed time exit after 30 x 4H bars (~5 calendar days)
  Risk/trade  : 0.25% of current equity, $150 ceiling
  Leverage    : 1.0x  (Binance Futures, SHORT only)
  Cap         : Uncapped
  Character   : SHORT-SIDE SPECIALIST (bear regime specialist, all-weather)

Data strategy:
  1. Load committed 4H cache: data/futures_universe/ohlcv_4h/{SYM}_4h.csv
  2. Append latest bars from Binance Futures public API (fapi.binance.com)
     Note: fapi.binance.com public endpoints work globally.
  3. Funding rate: fetch latest from fapi premiumIndex, fallback to cached CSV.
  4. --no-download: skip live fetch, use committed caches only.
  5. Process all new 4H bars since last run (replay approach for multi-bar days).

Entry is at open of the bar AFTER the signal fires.
Exit checks stop (bar high >= stop_loss) then time (bars_held >= 30).

Usage:
  python phase_t9b_volcontraction_short_paper_engine.py
      --date 2026-06-11    run for a specific date (processes 4H bars up to midnight UTC)
      --backfill           replay from freeze date (2026-06-11) to yesterday
      --no-download        use committed caches only
      --reset              wipe state and restart from $10,000
      --notify             compact notification summary (for CI)

Output files (data/t9b_volcontraction_paper/):
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
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent / "engines"))

import numpy as np
import pandas as pd

import t9b_shared
from signal_arbitrator import SignalArbitrator


# =============================================================================
# FROZEN CONFIG  (2026-06-11)
# =============================================================================

SYSTEM_NAME         = "VolContractionShort_4H_T9B"
FREEZE_DATE         = date(2026, 6, 11)

COMPRESSION_BARS    = 15      # cb: minimum consecutive compressed 4H bars
ATR_PERCENTILE      = 20      # ap: compressed = ATR(14) < 20th pct of trailing 252 bars
ATR_N               = 14
ATR_STOP_MULT       = 2.0     # stop ABOVE entry for short
HOLD_BARS           = 30      # time exit after 30 x 4H bars
EMA_N               = 200     # close must be BELOW EMA200
FUNDING_THRESHOLD   = 0.0001  # 0.01% per 8h period -- discovery constraint

RISK_PER_TRADE_PCT  = 0.0025  # 0.25%
CAPITAL_CEILING     = 150.0   # USD ceiling on risk per trade
MIN_ORDER_USDT      = 15.0    # minimum notional
MAX_OPEN_POSITIONS  = None    # uncapped
INITIAL_CAPITAL     = 10_000.0
LEVERAGE            = 1.0
KILL_SWITCH_DD_PCT  = 35.0

LIMIT_BARS          = 3000    # bars to load per symbol (500 days of 4H)
SLEEP_SEC           = 0.10
MIN_BARS_REQUIRED   = EMA_N + 252 + COMPRESSION_BARS + 10  # ~477 bars

FOUR_HOURS_MS       = 4 * 3600 * 1_000
EPS                 = 1e-12


# =============================================================================
# PATHS
# =============================================================================

ROOT              = Path.cwd()
DATA_DIR          = ROOT / "data" / "t9b_volcontraction_paper"
STATE_PATH        = DATA_DIR / "state.json"
OPEN_POS_CSV      = DATA_DIR / "open_positions.csv"
PENDING_CSV       = DATA_DIR / "pending_entries.csv"
SIGNALS_CSV       = DATA_DIR / "signals_today.csv"
EQUITY_CSV        = DATA_DIR / "equity_curve.csv"
DAILY_LOG_CSV     = DATA_DIR / "daily_log.csv"

CACHE_4H          = ROOT / "data" / "futures_universe" / "ohlcv_4h"
CACHE_FUNDING     = ROOT / "data" / "futures_universe" / "funding_rates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_OHLCV:    dict = {}
_SESSION_FUNDING:  dict = {}
_SKIP_LIVE_FETCH:  bool = False


# =============================================================================
# UNIVERSE  (auto-discovered: symbols with both 4H OHLCV and funding cache)
# =============================================================================

def _discover_universe() -> List[str]:
    h4_syms = {f.stem.replace("_4h", "") for f in CACHE_4H.glob("*_4h.csv")}
    fr_syms = {f.stem.replace("_funding", "") for f in CACHE_FUNDING.glob("*_funding.csv")}
    return sorted(h4_syms & fr_syms)

VC_UNIVERSE: List[str] = _discover_universe()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VCShortPosition:
    position_id:           str
    symbol:                str
    entry_ts_ms:           int
    entry_date:            str
    entry_price:           float
    stop_loss:             float   # ABOVE entry for short
    initial_risk_per_unit: float   # stop_loss - entry_price (positive)
    risk_amount_usdt:      float
    qty:                   float
    bars_held:             int
    atr_at_entry:          float
    funding_at_entry:      float
    compression_low:       float
    consec_at_entry:       int

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
    return {
        "system_name":             SYSTEM_NAME,
        "created_utc":             pd.Timestamp.utcnow().isoformat(),
        "last_run_date":           None,
        "last_processed_ts_ms":    0,
        "paper_equity_usdt":       INITIAL_CAPITAL,
        "peak_equity_usdt":        INITIAL_CAPITAL,
        "drawdown_pct":            0.0,
        "kill_switch_triggered":   False,
        "open_positions":          [],
        "pending_entries":         [],
        "closed_trade_count":      0,
    }


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


def positions_from_state(state: dict) -> Dict[str, VCShortPosition]:
    fields = {f.name for f in dataclasses.fields(VCShortPosition)}
    out: Dict[str, VCShortPosition] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = VCShortPosition(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping position {raw.get('position_id','?')}: {exc}")
    return out


def positions_to_state(state: dict, positions: Dict[str, VCShortPosition]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


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
        print(f"[RECONCILE] {len(issues)} issue(s):")
        for iss in issues:
            print(f"  [WARN] {iss}")
    else:
        print(f"[RECONCILE] OK  ({len(positions)} position(s) verified)")


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
                print(f"  [WARN] {symbol}: 4H cache read error -- {exc}")

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
            _save_4h_cache(merged, cache_path)

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


def _save_4h_cache(df: pd.DataFrame, path: Path) -> None:
    out = df[["ts_ms", "open", "high", "low", "close", "volume"]].copy()
    out.to_csv(path, index=False)


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
        print(f"    [binance4h] {symbol}: {exc}")
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
                print(f"  [WARN] {symbol}: funding cache read error -- {exc}")
                _SESSION_FUNDING[symbol] = pd.DataFrame()
        else:
            _SESSION_FUNDING[symbol] = pd.DataFrame()
    return _SESSION_FUNDING[symbol]


_CURRENT_FUNDING: dict = {}


def _fetch_all_funding_rates(symbols: List[str]) -> None:
    """Fetch current funding rates from Binance Futures premiumIndex in one call."""
    global _CURRENT_FUNDING
    if _SKIP_LIVE_FETCH:
        return
    try:
        import requests
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
        r.raise_for_status()
        for item in r.json():
            sym = item.get("symbol", "")
            rate_str = item.get("lastFundingRate", "0")
            try:
                _CURRENT_FUNDING[sym] = float(rate_str)
            except (TypeError, ValueError):
                pass
        fetched = sum(1 for s in symbols if s in _CURRENT_FUNDING)
        print(f"[FUNDING] Fetched current rates for {fetched}/{len(symbols)} symbols")
    except Exception as exc:
        print(f"[FUNDING] Live fetch failed: {exc}  (using cached rates only)")


def get_funding_at_ts(symbol: str, ts_ms: int) -> float:
    """Return funding rate applicable at ts_ms -- from cache, fallback to current API rate."""
    df = load_funding_rates(symbol)
    if not df.empty:
        prior = df[df["funding_time"] <= ts_ms]
        if not prior.empty:
            return float(prior.iloc[-1]["funding_rate"])
    return _CURRENT_FUNDING.get(symbol, 0.0)


# =============================================================================
# INDICATORS  (4H: ATR, ATR percentile, EMA200, compression streak + low)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    out = df.copy()
    cls = out["close"]
    hgh = out["high"]
    lw  = out["low"]
    n   = len(out)

    # ATR(14)
    prev_c = cls.shift(1)
    tr = pd.concat([hgh - lw, (hgh - prev_c).abs(), (lw - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean()

    # ATR 20th percentile over trailing 252 bars (min 50 bars)
    atr_pct20 = atr.rolling(252, min_periods=50).quantile(ATR_PERCENTILE / 100.0)

    # EMA(200)
    ema200 = cls.ewm(span=EMA_N, adjust=False).mean()

    # Consecutive compressed bars and compression lowest low
    atr_vals  = atr.values
    pct_vals  = atr_pct20.values
    low_vals  = lw.values
    consec    = np.zeros(n)
    comp_low  = np.full(n, np.nan)
    cnt       = 0
    low_track = np.inf

    for i in range(1, n):
        a = atr_vals[i]
        p = pct_vals[i]
        if np.isfinite(a) and np.isfinite(p) and a < p:
            cnt += 1
            low_track = min(low_track, float(low_vals[i]))
        else:
            cnt       = 0
            low_track = np.inf
        consec[i]   = cnt
        comp_low[i] = low_track if cnt > 0 else np.nan

    out["atr"]       = atr_vals
    out["atr_pct20"] = pct_vals
    out["ema200"]    = ema200.values
    out["consec"]    = consec
    out["comp_low"]  = comp_low
    return out


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_signal(row: pd.Series, funding_rate: float) -> Optional[dict]:
    """
    Signal conditions on a single 4H bar (checked at bar close):
      1. consec_comp >= 15 consecutive compressed bars
      2. close < compression_low  (breaks below compression period's lowest low)
      3. close < EMA200
      4. funding_rate >= 0.01%

    Entry: at OPEN of the NEXT 4H bar after signal.
    Stop : close + ATR x 2.0  (ABOVE entry for short)
    """
    consec   = row.get("consec", 0)
    comp_low = row.get("comp_low", np.nan)
    close    = float(row["close"])
    ema200   = row.get("ema200", np.nan)
    atr      = row.get("atr", np.nan)

    if consec < COMPRESSION_BARS:
        return None
    if not np.isfinite(comp_low) or close >= comp_low:
        return None  # not breaking below compression low
    if not np.isfinite(ema200) or close >= ema200:
        return None  # must be below EMA200
    if not np.isfinite(atr) or atr <= EPS:
        return None
    if funding_rate < FUNDING_THRESHOLD:
        return None  # funding gate

    stop     = close + ATR_STOP_MULT * atr
    risk_unit = ATR_STOP_MULT * atr

    return {
        "consec":      int(consec),
        "comp_low":    float(comp_low),
        "atr":         round(float(atr), 8),
        "ema200":      round(float(ema200), 8),
        "signal_close": round(close, 8),
        "stop_loss":   round(stop, 8),
        "risk_unit":   round(risk_unit, 8),
        "funding_rate": float(funding_rate),
    }


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def check_exit(pos: VCShortPosition, bar: pd.Series) -> Optional[dict]:
    """
    Exit priority:
      1. Stop hit  : bar HIGH >= stop_loss -> exit at stop_loss (loss = -1R)
      2. Time exit : bars_held >= 30       -> exit at bar close

    For SHORT: profit = entry_price - exit_price
    """
    bar_high  = float(bar["high"])
    bar_close = float(bar["close"])
    ru        = max(pos.initial_risk_per_unit, EPS)

    # 1. Stop hit
    if bar_high >= pos.stop_loss:
        gross_r  = (pos.entry_price - pos.stop_loss) / ru  # = -1.0
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "exit_reason": "stop_hit",
            "exit_price":  round(pos.stop_loss, 8),
            "gross_r":     round(gross_r, 6),
            "pnl_usdt":    round(pnl_usdt, 4),
        }

    # 2. Time exit
    if pos.bars_held >= HOLD_BARS:
        gross_r  = (pos.entry_price - bar_close) / ru
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "exit_reason": "time_exit_30bars",
            "exit_price":  round(bar_close, 8),
            "gross_r":     round(gross_r, 6),
            "pnl_usdt":    round(pnl_usdt, 4),
        }

    return None


# =============================================================================
# MAIN PROCESSING LOOP  (processes all new 4H bars since last run)
# =============================================================================

def run_daily(run_date: date, symbols: List[str], state: dict) -> dict:
    """
    Process all 4H bars from state['last_processed_ts_ms']+1 up to
    midnight UTC of run_date (exclusive). This naturally handles
    multiple 4H bars per day and backfill scenarios.
    """
    equity   = float(state["paper_equity_usdt"])
    peak     = float(state["peak_equity_usdt"])
    kill_sw  = bool(state.get("kill_switch_triggered", False))
    positions = positions_from_state(state)
    pending   = list(state.get("pending_entries", []))

    last_ts_ms  = int(state.get("last_processed_ts_ms", 0))
    cutoff_ts_ms = int(pd.Timestamp(str(run_date), tz="UTC").value // 1_000_000)

    log_events:    List[dict] = []
    signals_run:   List[dict] = []
    exits_run:     List[dict] = []
    max_ts_seen:   int = last_ts_ms
    cross_skip_logged: set = set()  # suppress repeat cross-system logs per run

    def _log(event, symbol, detail, **extra):
        log_events.append({"run_date": str(run_date), "bar_ts_ms": max_ts_seen,
                           "event": event, "symbol": symbol, "detail": detail, **extra})

    cross_syms = t9b_shared.get_cross_system_symbols("volcontraction")
    arbitrator = SignalArbitrator("volcontraction", run_date)

    for sym in symbols:
        df_raw = load_4h_ohlcv(sym, cutoff_ts_ms=cutoff_ts_ms)
        if df_raw.empty or len(df_raw) < MIN_BARS_REQUIRED:
            continue

        df = add_indicators(df_raw)

        # New bars: after last_ts_ms and before cutoff
        new_df = df[df["ts_ms"] > last_ts_ms].copy().reset_index(drop=True)
        if new_df.empty:
            continue

        for i in range(len(new_df)):
            bar     = new_df.iloc[i]
            bar_ts  = int(bar["ts_ms"])
            max_ts_seen = max(max_ts_seen, bar_ts)

            # ---- 1. Fill any pending entry for this symbol at this bar's open ----
            filled_pids = []
            for pe in pending:
                if pe["symbol"] != sym:
                    continue
                if pe["signal_ts_ms"] >= bar_ts:
                    continue  # same bar or future -- not ready yet

                entry_price = float(bar["open"])
                risk_unit   = pe["risk_unit"]
                stop_loss   = entry_price + ATR_STOP_MULT * float(pe["atr"])

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
                pos = VCShortPosition(
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
                    compression_low       = pe["comp_low"],
                    consec_at_entry       = pe["consec"],
                )
                positions[pid] = pos
                filled_pids.append(id(pe))
                _log("ENTRY", sym,
                     f"price={entry_price:.6g}  stop={stop_loss:.6g}  "
                     f"risk=${risk_amount:.2f}  funding={pe['funding_rate']*100:.4f}%  "
                     f"consec={pe['consec']}  comp_low={pe['comp_low']:.6g}  "
                     f"equity=${equity:,.2f}",
                     entry_price=round(entry_price, 8), stop=round(stop_loss, 8),
                     risk_amount=round(risk_amount, 4), equity=round(equity, 2))

            pending = [pe for pe in pending if id(pe) not in filled_pids]

            # ---- 2. Process exits for open positions on this symbol ----
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
                        "position_id": pid,
                        "symbol":      sym,
                        "entry_date":  pos.entry_date,
                        "exit_date":   str(pd.Timestamp(bar_ts, unit="ms", tz="UTC").date()),
                        "entry_price": pos.entry_price,
                        "stop_loss":   pos.stop_loss,
                        "bars_held":   pos.bars_held,
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

            # ---- 3. Check for new entry signals ----
            if kill_sw:
                continue
            if any(p.symbol == sym for p in positions.values()):
                continue  # already in this symbol
            if any(pe["symbol"] == sym for pe in pending):
                continue  # already a pending entry for this symbol
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
                     f"consec={sig['consec']}  close={sig['signal_close']:.6g}  "
                     f"comp_low={sig['comp_low']:.6g}  ema200={sig['ema200']:.6g}  "
                     f"funding={sig['funding_rate']*100:.4f}%  stop={sig['stop_loss']:.6g}",
                     close=sig["signal_close"], funding=sig["funding_rate"])

    # Commit state
    dd_now = (equity - peak) / max(peak, EPS) * 100.0
    state.update({
        "last_run_date":           str(run_date),
        "last_processed_ts_ms":    max_ts_seen,
        "paper_equity_usdt":       equity,
        "peak_equity_usdt":        peak,
        "drawdown_pct":            round(dd_now, 4),
        "kill_switch_triggered":   kill_sw,
        "closed_trade_count":      state.get("closed_trade_count", 0) + len(exits_run),
        "pending_entries":         pending,
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

def write_open_positions(positions: Dict[str, VCShortPosition], run_date: date) -> None:
    if not positions:
        pd.DataFrame(columns=[
            "position_id", "symbol", "entry_date", "as_of_date",
            "entry_price", "stop_loss", "bars_held", "bars_remaining",
            "atr_at_entry", "funding_at_entry", "compression_low",
            "consec_at_entry", "risk_amount_usdt", "qty",
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
            "bars_remaining":   max(0, HOLD_BARS - pos.bars_held),
            "atr_at_entry":     pos.atr_at_entry,
            "funding_at_entry": round(pos.funding_at_entry * 100, 6),
            "compression_low":  pos.compression_low,
            "consec_at_entry":  pos.consec_at_entry,
            "risk_amount_usdt": round(pos.risk_amount_usdt, 4),
            "qty":              pos.qty,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_pending_entries(pending: List[dict], run_date: date) -> None:
    if not pending:
        pd.DataFrame(columns=[
            "symbol", "signal_ts_ms", "signal_close", "stop_loss",
            "comp_low", "consec", "atr", "funding_rate",
        ]).to_csv(PENDING_CSV, index=False)
        return
    rows = []
    for pe in pending:
        rows.append({
            "symbol":       pe.get("symbol", ""),
            "signal_ts_ms": pe.get("signal_ts_ms", 0),
            "signal_close": pe.get("signal_close", 0),
            "stop_loss":    pe.get("stop_loss", 0),
            "comp_low":     pe.get("comp_low", 0),
            "consec":       pe.get("consec", 0),
            "atr":          pe.get("atr", 0),
            "funding_rate": pe.get("funding_rate", 0),
        })
    pd.DataFrame(rows).to_csv(PENDING_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    if not signals:
        pd.DataFrame(columns=[
            "symbol", "run_date", "signal_ts_ms", "signal_close",
            "comp_low", "consec", "atr", "ema200", "funding_rate", "stop_loss",
        ]).to_csv(SIGNALS_CSV, index=False)
        return
    pd.DataFrame(signals).to_csv(SIGNALS_CSV, index=False)


def append_equity_curve(exits: List[dict]) -> None:
    if not exits:
        return
    rows = [{
        "exit_date":    ex["exit_date"],
        "symbol":       ex["symbol"],
        "entry_date":   ex["entry_date"],
        "bars_held":    ex["bars_held"],
        "entry_price":  ex["entry_price"],
        "exit_price":   ex["exit_price"],
        "gross_r":      ex["gross_r"],
        "pnl_usdt":     ex["pnl_usdt"],
        "equity":       ex["equity_after"],
        "dd_pct":       ex["dd_pct"],
        "exit_reason":  ex["exit_reason"],
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
    print("T9B -- VOLCONTRACTIONSHORT FUNDING GATE 4H PAPER ENGINE")
    print("=" * 70)
    print(f"Frozen config (2026-06-11):")
    print(f"  4H / ATR compression cb={COMPRESSION_BARS} ap={ATR_PERCENTILE} /"
          f" EMA{EMA_N} below / funding>={FUNDING_THRESHOLD*100:.2f}% /"
          f" stop ATRx{ATR_STOP_MULT} above / time exit {HOLD_BARS} bars")
    print(f"  Uncapped / {RISK_PER_TRADE_PCT*100:.2f}% risk / $150 ceiling / SHORT only")
    print(f"  Universe: {len(VC_UNIVERSE)} symbols / Binance Futures 4H")
    print(f"  Character: SHORT-SIDE SPECIALIST -- bear regime strongest (+0.375R 2022)")
    print(f"  Output: {DATA_DIR}")
    print()


def print_day_summary(result: dict, verbose: bool = True) -> None:
    d, eq, dd = result["run_date"], result["equity"], result["dd_pct"]
    op, pe, si, en, ex = (result["open_count"], result["pending_count"],
                          result["new_signals"], result["new_entries"], result["exits"])
    ks = " [KILL-SWITCH]" if result["kill_switch"] else ""
    print(f"  {d}  equity=${eq:>10,.2f}  DD={dd:>+6.2f}%  "
          f"open={op}  pending={pe}  signals={si}  entries={en}  exits={ex}{ks}")
    if verbose:
        for e in result["log_events"]:
            if e["event"] in ("ENTRY", "EXIT", "KILL_SWITCH", "SIGNAL", "SIGNAL_SKIPPED"):
                sym = e["symbol"] if e["symbol"] else ""
                print(f"    [{e['event']:<16}] {sym:<18} {e['detail']}")


def print_final_summary(state: dict) -> None:
    eq  = float(state["paper_equity_usdt"])
    ret = (eq / INITIAL_CAPITAL - 1.0) * 100.0
    dd  = float(state["drawdown_pct"])
    op  = len(state.get("open_positions", []))
    pe  = len(state.get("pending_entries", []))
    cl  = state.get("closed_trade_count", 0)
    print()
    print("-" * 70)
    print("FINAL STATE -- VolContractionShort Funding Gate 4H Paper")
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Total return   : {ret:>+.2f}%")
    print(f"  Current DD     : {dd:>+.2f}%")
    print(f"  Open positions : {op}  (uncapped)")
    print(f"  Pending entries: {pe}  (signals awaiting next bar fill)")
    print(f"  Closed trades  : {cl}")
    print(f"  Kill-switch    : {'TRIGGERED' if state.get('kill_switch_triggered') else 'not triggered'}")
    print()
    print(f"  open_positions.csv : {OPEN_POS_CSV}")
    print(f"  pending_entries.csv: {PENDING_CSV}")
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
    n_pending    = len(state.get("pending_entries", []))
    n_closed     = int(state.get("closed_trade_count", 0))
    kill_sw      = state.get("kill_switch_triggered", False)
    days_running = (run_date - FREEZE_DATE).days + 1
    review_date  = FREEZE_DATE + timedelta(days=90)
    days_to_rev  = (review_date - run_date).days

    print()
    print("=" * 55)
    print(f"T9B VOL-CONTRACTION SHORT DAILY UPDATE -- {run_date}")
    print("=" * 55)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)")
    print(f"Peak equity   : ${peak:>10,.2f}")
    print(f"Max drawdown  : {dd:>+.2f}%")
    print(f"Open positions: {n_open}  (uncapped)")
    print(f"Pending fills : {n_pending}")
    print(f"Closed trades : {n_closed}")
    print(f"Days running  : {days_running}")
    print(f"Days to review: {days_to_rev}  (3-month: {review_date})")
    if kill_sw:
        print("KILL-SWITCH   : TRIGGERED -- no new entries")
    print()

    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN SHORT POSITIONS ({len(positions)})")
        print(f"  {'Symbol':<18} {'Entry':<12} {'Stop':>12} {'Held':>5} {'Rem':>5} {'Funding%':>9}")
        print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*5} {'-'*5} {'-'*9}")
        for p in sorted(positions, key=lambda x: x.get("bars_held", 0), reverse=True):
            sym   = str(p.get("symbol", "?"))
            edate = str(p.get("entry_date", "?"))
            stop  = p.get("stop_loss", 0)
            held  = int(p.get("bars_held", 0))
            rem   = max(0, HOLD_BARS - held)
            fr    = float(p.get("funding_at_entry", 0)) * 100
            print(f"  {sym:<18} {edate:<12} {stop:>12.6g} {held:>5} {rem:>5} {fr:>8.4f}%")
    else:
        print("OPEN SHORT POSITIONS: none")
    print()

    pending = state.get("pending_entries", [])
    if pending:
        print(f"PENDING ENTRIES (fills at next bar open) ({len(pending)})")
        for pe in pending:
            sym   = str(pe.get("symbol", "?"))
            close = pe.get("signal_close", 0)
            stop  = pe.get("stop_loss", 0)
            fr    = float(pe.get("funding_rate", 0)) * 100
            print(f"  {sym:<18}  signal_close={close:<12.6g}  stop={stop:<12.6g}  funding={fr:.4f}%")
    else:
        print("PENDING ENTRIES: none")
    print()

    if SIGNALS_CSV.exists():
        try:
            sig_df = pd.read_csv(SIGNALS_CSV)
            if not sig_df.empty:
                print(f"SIGNALS THIS RUN ({len(sig_df)})")
                for _, row in sig_df.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    cons = row.get("consec", "?")
                    cls  = row.get("signal_close", "?")
                    stop = row.get("stop_loss", "?")
                    fr   = float(row.get("funding_rate", 0)) * 100
                    print(f"  {sym:<18}  consec={cons}  close={cls:<12.6g}  "
                          f"stop={stop:<12.6g}  funding={fr:.4f}%")
            else:
                print("SIGNALS THIS RUN: none")
        except Exception:
            print("SIGNALS THIS RUN: (could not read)")
    else:
        print("SIGNALS THIS RUN: none")

    print()
    if kill_sw:
        print("STATUS: KILL-SWITCH ACTIVE")
    elif dd < -20.0:
        print("STATUS: WARN -- drawdown > 20%")
    else:
        print("STATUS: OK")
    print("=" * 55)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="T9B VolContractionShort Funding Gate 4H Paper Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase_t9b_volcontraction_short_paper_engine.py\n"
            "  python phase_t9b_volcontraction_short_paper_engine.py --date 2026-06-11\n"
            "  python phase_t9b_volcontraction_short_paper_engine.py --backfill\n"
            "  python phase_t9b_volcontraction_short_paper_engine.py --notify\n"
            "  python phase_t9b_volcontraction_short_paper_engine.py --reset\n"
        ),
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD",
                     help="Process 4H bars up to midnight UTC of this date")
    grp.add_argument("--backfill", action="store_true",
                     help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip live fetch; use committed caches only")
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
        for f in [STATE_PATH, EQUITY_CSV, DAILY_LOG_CSV, PENDING_CSV]:
            if f.exists():
                f.unlink()
        print("[RESET] State and logs cleared. Starting from $10,000.")
        print()

    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    symbols = VC_UNIVERSE
    print(f"[UNIVERSE] {len(symbols)} symbols with 4H OHLCV + funding data")

    today     = date.today()
    yesterday = today - timedelta(days=1)

    if args.backfill:
        run_dates = []
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

    # Fetch current funding rates once at the start (batch, much faster)
    _fetch_all_funding_rates(symbols)
    print()

    state    = load_state()
    n_open   = len(state.get("open_positions", []))
    n_pend   = len(state.get("pending_entries", []))
    n_closed = state.get("closed_trade_count", 0)
    last_ts  = state.get("last_processed_ts_ms", 0)
    print(f"[STATE] equity=${state['paper_equity_usdt']:,.2f}  "
          f"open={n_open}  pending={n_pend}  closed={n_closed}  "
          f"last_ts_ms={last_ts}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}")
    print()

    _reconcile_state(state)
    print()

    for run_date in run_dates:
        result = run_daily(run_date, symbols, state)
        save_state(state)
        write_open_positions(result["positions"], run_date)
        write_pending_entries(result["pending"], run_date)
        write_signals_today(result["signals_run"], run_date)
        append_equity_curve(result["exits_run"])
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
    t9b_shared.run_engine("volcontraction", main)
