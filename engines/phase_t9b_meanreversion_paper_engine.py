#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T9B -- MEANREVERSIONRSI 1D PAPER ENGINE (FUTURES CONFIG)
===============================================================

Paper trading engine for the frozen MeanReversionRSI 1D config,
migrated to Binance Futures USD-M per Step 18 (PASS, 2026-06-19).

Frozen signal config (unchanged from Spot T8 freeze 2026-06-01):
  Entry       : RSI(14) < 25 on daily close -- enter LONG (signal-bar close)
  Safety stop : ATR(14) x 3.0 below entry close  (safety net only)
  Exit        : Fixed time exit after 20 bars
  Filter      : none  (EMA200 confirmed net-negative across all TFs)
  Max pos     : 10 concurrent
  Risk/trade  : 0.25% of current equity
  Leverage    : 1.0x  LONG only

Futures migration (2026-07-12, Step 18):
  Venue       : Binance Futures USD-M  (was Binance Spot)
  Universe    : 290-symbol Futures universe, auto-discovered from
                data/futures_universe/ohlcv_1d/  (was 52-symbol Spot list)
  Cost floor  : 0.25R (Futures long)  -- Step 18 net avg_r +0.3162R clears it
  Funding     : NO funding gate (S2 enters regardless of funding sign --
                the funding gate variant is System 8, a separate engine).
                Funding cost is implicitly favourable: S2's oversold signal
                selects negative-funding environments (finding #21).
  Kill-switch : -35% DD from peak, unchanged from Spot engine.
  Signal math : SMA-seeded Wilder RSI/ATR, ported verbatim from
                step18_s2_futures_costcheck.py (finding #22 discipline).

Data strategy (same as System 8 engine):
  1. Committed cache: data/futures_universe/ohlcv_1d/{SYM}_1d.csv
     (kept current by CI step 3a: fapi probe -> binance.vision fallback)
  2. Optional live top-up via fapi.binance.com (falls back silently to
     cache if geo-blocked -- HTTP 451 from US GitHub runners).
  3. --no-download: committed cache only.

State continuity: same state dir (data/t9b_mr_paper/), same T9B clock
(FREEZE_DATE 2026-06-01). Open Spot positions are migrated on first run:
symbols normalized BTC/USDT -> BTCUSDT; positions whose symbol has no
Futures data are closed at entry price (0 P&L, not counted in metrics).

Usage:
  python phase_t9b_meanreversion_paper_engine.py
      --date 2026-07-12    run for a specific date
      --backfill           replay from freeze date through yesterday
      --no-download        use committed cache only
      --reset              wipe state and restart
      --notify             compact notification summary (for CI)

Output files (data/t9b_mr_paper/):
  state.json            persistent engine state
  open_positions.csv    current open positions with bars_held, stop
  signals_today.csv     RSI signals detected today
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
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import t9b_shared
from signal_arbitrator import SignalArbitrator


# =============================================================================
# FROZEN CONFIG  (signal params frozen 2026-06-01; Futures venue 2026-07-12)
# =============================================================================

SYSTEM_NAME         = "MR_RSI14_1D_T9B"
FREEZE_DATE         = date(2026, 6, 1)    # T9B clock start (unchanged)
MIGRATION_DATE      = date(2026, 7, 12)   # Spot -> Futures venue migration

RSI_N               = 14
OVERSOLD_THR        = 25      # enter when RSI < 25 on close
ATR_N               = 14
ATR_STOP_MULT       = 3.0     # safety stop only -- primary exit is always time
TIME_EXIT_BARS      = 20      # exit after exactly 20 daily bars

RISK_PER_TRADE_PCT  = 0.0025  # 0.25% of current equity
MIN_ORDER_SIZE_USDT = 15.0    # Fix 2: minimum position notional
MAX_OPEN_POSITIONS  = 10
INITIAL_CAPITAL     = 12_000.0  # Scheme C: $12k MR pool (40% S2 within pool)
LEVERAGE            = 1.0
KILL_SWITCH_DD_PCT  = 35.0    # halt new entries if DD from peak exceeds this
COST_FLOOR_R        = 0.25    # Futures long cost floor (research gate, doc only)

LIMIT_BARS          = 2000
MIN_BARS_REQUIRED   = 50      # research convention (step18: len(df) >= 50)

EPS = 1e-12


# =============================================================================
# PATHS
# =============================================================================

ROOT            = Path.cwd()
DATA_DIR        = ROOT / "data" / "t9b_mr_paper"
STATE_PATH      = DATA_DIR / "state.json"
OPEN_POS_CSV    = DATA_DIR / "open_positions.csv"
SIGNALS_CSV     = DATA_DIR / "signals_today.csv"
# NOT "equity_curve.csv" -- owned by .github/scripts/mark_to_market.py
# (different schema). Same collision/fix as the Donchian/ConsecDown engines
# -- confirmed live here: 11 real trade-exit rows (2026-08-04 to 08-20)
# were silently interleaved into the MTM file before this fix.
EQUITY_CSV      = DATA_DIR / "engine_equity_curve.csv"
DAILY_LOG_CSV   = DATA_DIR / "daily_log.csv"

CACHE_1D        = ROOT / "data" / "futures_universe" / "ohlcv_1d"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_OHLCV:   dict = {}
_SKIP_LIVE_FETCH: bool = False


# =============================================================================
# UNIVERSE  (auto-discovered: all symbols with committed 1D Futures OHLCV)
# =============================================================================

def _discover_universe() -> List[str]:
    return sorted(f.stem.replace("_1d", "") for f in CACHE_1D.glob("*_1d.csv"))

MR_UNIVERSE: List[str] = _discover_universe()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MRPosition:
    position_id:           str
    symbol:                str
    entry_date:            str    # ISO date string
    entry_price:           float
    stop_loss:             float  # ATR x 3.0 below entry (safety net)
    initial_risk_per_unit: float  # ATR x 3.0 at entry
    risk_amount_usdt:      float  # equity x 0.25% at entry
    qty:                   float
    bars_held:             int    # incremented each daily bar processed
    atr_at_entry:          float
    rsi_at_entry:          float

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


def positions_from_state(state: dict) -> Dict[str, MRPosition]:
    fields = {f.name for f in dataclasses.fields(MRPosition)}
    out: Dict[str, MRPosition] = {}
    for raw in state.get("open_positions", []):
        kwargs = {k: v for k, v in raw.items() if k in fields}
        try:
            pos = MRPosition(**kwargs)
            out[pos.position_id] = pos
        except Exception as exc:
            print(f"[WARN] Skipping position {raw.get('position_id','?')}: {exc}")
    return out


def positions_to_state(state: dict, positions: Dict[str, MRPosition]) -> None:
    state["open_positions"] = [asdict(p) for p in positions.values()]


def append_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def _reconcile_state(state: dict, run_date: date) -> None:
    """Fix 3: verify open positions on startup.

    Migration handling (2026-07-12): positions opened by the Spot engine use
    'BTC/USDT' symbol format -- normalize to Futures 'BTCUSDT'. Positions
    whose normalized symbol has no Futures 1D data are closed at entry price
    (0 P&L, logged, not counted in metrics). Positions predating FREEZE_DATE
    are invalid seeds and closed the same way.
    """
    positions = state.get("open_positions", [])

    # ── Spot -> Futures symbol normalization (one-time migration) ────────
    migrated = 0
    for pos in positions:
        sym = str(pos.get("symbol", ""))
        if "/" in sym:
            pos["symbol"] = sym.replace("/", "").replace(":", "")
            migrated += 1
    if migrated:
        print(f"[MIGRATE] Normalized {migrated} Spot-format symbol(s) to Futures format")

    # ── Invalid-seed / no-data purge ──────────────────────────────────────
    valid  = []
    purged = []
    for pos in positions:
        sym = str(pos.get("symbol", ""))
        try:
            edate = date.fromisoformat(str(pos.get("entry_date", "")))
        except (ValueError, TypeError):
            valid.append(pos)
            continue
        if edate < FREEZE_DATE:
            purged.append((pos, f"entry_date={edate} predates engine start {FREEZE_DATE}"))
        elif not (CACHE_1D / f"{sym}_1d.csv").exists():
            purged.append((pos, "no Futures 1D data after Spot->Futures migration"))
        else:
            valid.append(pos)

    if purged:
        rows = []
        for pos, why in purged:
            sym = pos.get("symbol", "?")
            ep  = pos.get("entry_price", 0)
            print(f"[PURGED] {sym}: {why} -> closed at entry_price={ep} "
                  f"(0 P&L, not counted in metrics)", flush=True)
            rows.append({
                "run_date": str(run_date),
                "event":    "INVALID_SEED_CLOSED",
                "symbol":   sym,
                "detail":   f"{why}; closed at entry_price={ep}; not in metrics",
            })
        state["open_positions"] = valid
        append_csv(DAILY_LOG_CSV, rows)
        positions = valid

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


# =============================================================================
# OHLCV LOADING  (committed 1D Futures cache + optional live fetch --
#                 identical strategy to System 8 engine)
# =============================================================================

def load_ohlcv(symbol: str, up_to_date: Optional[date] = None) -> pd.DataFrame:
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
# INDICATORS  -- ported VERBATIM from step18_s2_futures_costcheck.py
# (SMA-seeded Wilder recursion; matches the Step 18 research trades)
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

    # NO funding gate -- S2 enters regardless of funding sign (System 8
    # is the funding-gated variant and runs as a separate engine).
    stop = close - ATR_STOP_MULT * atr
    return {
        "symbol":      symbol,
        "signal_date": str(run_date),
        "close":       round(close, 8),
        "rsi":         round(rsi, 4),
        "atr":         round(atr, 8),
        "stop_loss":   round(stop, 8),
        "risk_unit":   round(ATR_STOP_MULT * atr, 8),
        "signal":      "BUY_MR",
    }


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def update_and_check_exit(
    pos: MRPosition,
    df: pd.DataFrame,
    run_date: date,
) -> tuple:
    """
    Process the bar for run_date.

    Exit logic (priority order):
      1. Safety stop: bar low <= pos.stop_loss  -> exit at stop_loss price
      2. Time exit  : bars_held >= TIME_EXIT_BARS -> exit at close

    Research convention (step18): bars_held = i - e_bar at the CURRENT bar,
    i.e. calendar days elapsed since entry (1D bars = 1 calendar day by
    design). bars_held is RECOMPUTED from entry_date every call rather than
    incremented -- this is the fix for the 2026-07 data-gap bug: the old
    `pos.bars_held += 1` only ran when df had an exact-date row, so a missing
    OHLCV date silently froze the counter while last_run_date advanced
    normally (positions held 46+ calendar days with bars_held stuck at 8-9).
    Recomputing from entry_date self-heals any accumulated drift and can
    never silently stall.
    """
    entry_dt = date.fromisoformat(str(pos.entry_date))
    calendar_days_elapsed = (run_date - entry_dt).days
    prev_bars_held = pos.bars_held
    pos.bars_held = calendar_days_elapsed
    if pos.bars_held - prev_bars_held > 1:
        print(f"  [DATA_GAP] {pos.symbol}: bars_held corrected {prev_bars_held} -> "
              f"{pos.bars_held} (calendar days since entry {pos.entry_date}); "
              f"OHLCV cache was missing {pos.bars_held - prev_bars_held - 1} "
              f"intervening date(s).", flush=True)

    day_ms = int(pd.Timestamp(str(run_date), tz="UTC").value // 1_000_000)
    day_rows = df[df["ts_ms"] == day_ms] if not df.empty else df

    if day_rows.empty:
        if pos.bars_held < TIME_EXIT_BARS:
            print(f"  [DATA_GAP] {pos.symbol}: no OHLCV row for {run_date} "
                  f"(cache/live both missing this date). bars_held={pos.bars_held} "
                  f"via calendar-day count; safety stop cannot be checked today.",
                  flush=True)
            return None, pos
        if df.empty:
            print(f"  [DATA_GAP][CRITICAL] {pos.symbol}: time exit overdue "
                  f"(bars_held={pos.bars_held} >= {TIME_EXIT_BARS}) but NO cached "
                  f"price available at all -- cannot force exit. Manual "
                  f"intervention required.", flush=True)
            return None, pos
        proxy_row   = df.iloc[-1]
        proxy_close = float(proxy_row["close"])
        proxy_date  = pd.to_datetime(int(proxy_row["ts_ms"]), unit="ms", utc=True).date()
        print(f"  [DATA_GAP][FORCED_EXIT] {pos.symbol}: time exit overdue "
              f"(bars_held={pos.bars_held}) with no price for {run_date} -- forcing "
              f"exit at last available close {proxy_close} (as of {proxy_date}). "
              f"True {run_date} price unavailable; treat this fill as approximate.",
              flush=True)
        gross_r  = (proxy_close - pos.entry_price) / max(pos.initial_risk_per_unit, EPS)
        pnl_usdt = pos.risk_amount_usdt * gross_r
        return {
            "position_id":   pos.position_id,
            "symbol":        pos.symbol,
            "entry_date":    pos.entry_date,
            "exit_date":     str(run_date),
            "entry_price":   pos.entry_price,
            "exit_price":    proxy_close,
            "stop_loss":     pos.stop_loss,
            "bars_held":     pos.bars_held,
            "gross_r":       round(gross_r, 6),
            "pnl_usdt":      round(pnl_usdt, 4),
            "exit_reason":   "time_exit_20bars_data_gap_proxy",
        }, pos

    row       = day_rows.iloc[-1]
    bar_low   = float(row["low"])
    bar_close = float(row["close"])

    # 1. Safety stop hit
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

    # 2. Time exit (primary exit -- Variant E)
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
    # STEP 2: Scan for new entry signals
    # ------------------------------------------------------------------
    if not kill_sw:
        cross_syms = t9b_shared.get_cross_system_symbols('rsi_mr')  # Fix 1
        arbitrator = SignalArbitrator('rsi_mr', run_date)
        for sym in symbols:
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

            if len(positions) >= MAX_OPEN_POSITIONS:
                _log("SIGNAL_SKIPPED", sym,
                     f"cap {MAX_OPEN_POSITIONS} reached  (RSI={sig['rsi']:.1f}  close={sig['close']:.6g})",
                     close=sig["close"])
                continue

            rw, rv = t9b_shared.get_regime_weight("rsi_mr")
            regime_scale = rw * rv * 7
            risk_amount   = equity * RISK_PER_TRADE_PCT * regime_scale
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

            pid = f"{sym}_{run_date.strftime('%Y%m%d')}"

            pos = MRPosition(
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
            )
            positions[pid] = pos

            _log("ENTRY", sym,
                 f"RSI={sig['rsi']:.1f}  price={sig['close']:.6g}  "
                 f"stop={sig['stop_loss']:.6g}  risk=${risk_amount:.2f}  "
                 f"bars_to_exit={TIME_EXIT_BARS}  equity=${equity:,.2f}",
                 entry_price=round(sig["close"], 8), stop=round(sig["stop_loss"], 8),
                 risk_amount=round(risk_amount, 4), equity=round(equity, 2))

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

def write_open_positions(positions: Dict[str, MRPosition], run_date: date) -> None:
    if not positions:
        pd.DataFrame(columns=[
            "position_id","symbol","entry_date","as_of_date",
            "entry_price","stop_loss","bars_held","bars_remaining",
            "rsi_at_entry","atr_at_entry","risk_amount_usdt",
        ]).to_csv(OPEN_POS_CSV, index=False)
        return
    rows = []
    for pos in positions.values():
        rows.append({
            "position_id":     pos.position_id,
            "symbol":          pos.symbol,
            "entry_date":      pos.entry_date,
            "as_of_date":      str(run_date),
            "entry_price":     pos.entry_price,
            "stop_loss":       pos.stop_loss,
            "bars_held":       pos.bars_held,
            "bars_remaining":  max(0, TIME_EXIT_BARS - pos.bars_held),
            "rsi_at_entry":    pos.rsi_at_entry,
            "atr_at_entry":    pos.atr_at_entry,
            "risk_amount_usdt": round(pos.risk_amount_usdt, 4),
            "qty":             pos.qty,
        })
    pd.DataFrame(rows).to_csv(OPEN_POS_CSV, index=False)


def write_signals_today(signals: List[dict], run_date: date) -> None:
    if not signals:
        pd.DataFrame(columns=["symbol","signal_date","close","rsi","atr","stop_loss","signal"]
                     ).to_csv(SIGNALS_CSV, index=False)
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
    print("T9B -- MEANREVERSIONRSI 1D PAPER ENGINE (FUTURES CONFIG)")
    print("=" * 70)
    print(f"Frozen signal config (2026-06-01), Futures venue (Step 18, 2026-07-12):")
    print(f"  1D / RSI({RSI_N}) < {OVERSOLD_THR} entry / Time exit {TIME_EXIT_BARS} bars / ATR stop x{ATR_STOP_MULT}")
    print(f"  Max {MAX_OPEN_POSITIONS} positions / {RISK_PER_TRADE_PCT*100:.2f}% risk / No filter / No funding gate")
    print(f"  {len(MR_UNIVERSE)} symbols / Binance Futures USD-M / LONG only")
    print(f"  Cost floor {COST_FLOOR_R}R (Futures) -- Step 18 net avg_r +0.3162R")
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
            if e["event"] in ("ENTRY","EXIT","KILL_SWITCH","SIGNAL_SKIPPED"):
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
    print("FINAL STATE -- MeanReversionRSI 1D Paper (Futures)")
    print(f"  Paper equity   : ${eq:>10,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Total return   : {ret:>+.2f}%")
    print(f"  Current DD     : {dd:>+.2f}%")
    print(f"  Open positions : {op}")
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
    print(f"T9B MR (FUTURES) DAILY UPDATE -- {run_date}")
    print("=" * 50)
    print(f"Paper equity  : ${eq:>10,.2f}  ({ret_pct:+.2f}%)")
    print(f"Peak equity   : ${peak:>10,.2f}")
    print(f"Max drawdown  : {dd:>+.2f}%")
    print(f"Open positions: {n_open}  (cap={MAX_OPEN_POSITIONS})")
    print(f"Closed trades : {n_closed}")
    print(f"Days running  : {days_running}")
    print(f"Days to review: {days_to_rev}  (3-month: {review_date})")
    if kill_sw:
        print("KILL-SWITCH   : TRIGGERED -- no new entries")
    print()

    positions = state.get("open_positions", [])
    if positions:
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"  {'Symbol':<16} {'Entry':<12} {'Stop':>12} {'Held':>5} {'Rem':>5}")
        print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*5} {'-'*5}")
        for p in sorted(positions, key=lambda x: x.get("bars_held", 0), reverse=True):
            sym  = str(p.get("symbol","?"))
            edate = str(p.get("entry_date","?"))
            stop  = p.get("stop_loss", 0)
            held  = int(p.get("bars_held", 0))
            rem   = max(0, TIME_EXIT_BARS - held)
            print(f"  {sym:<16} {edate:<12} {stop:>12.6g} {held:>5} {rem:>5}")
    else:
        print("OPEN POSITIONS: none")
    print()

    if SIGNALS_CSV.exists():
        try:
            sig_df = pd.read_csv(SIGNALS_CSV)
            if not sig_df.empty:
                print(f"RSI SIGNALS TODAY ({len(sig_df)})")
                for _, row in sig_df.iterrows():
                    sym  = str(row.get("symbol","?"))
                    rsi  = row.get("rsi","?")
                    cls  = row.get("close","?")
                    stop = row.get("stop_loss","?")
                    print(f"  {sym:<16}  RSI={rsi:<6.1f}  close={cls:<12.6g}  stop={stop:<12.6g}")
            else:
                print("RSI SIGNALS TODAY: none")
        except Exception:
            print("RSI SIGNALS TODAY: (could not read)")
    else:
        print("RSI SIGNALS TODAY: none")

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
        description="T9B MeanReversionRSI 1D Paper Engine (Futures config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase_t9b_meanreversion_paper_engine.py\n"
            "  python phase_t9b_meanreversion_paper_engine.py --date 2026-07-12\n"
            "  python phase_t9b_meanreversion_paper_engine.py --backfill\n"
            "  python phase_t9b_meanreversion_paper_engine.py --notify\n"
            "  python phase_t9b_meanreversion_paper_engine.py --reset\n"
        ),
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD")
    grp.add_argument("--backfill", action="store_true",
                     help=f"Replay from freeze date ({FREEZE_DATE}) through yesterday")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip live fetch; use committed cache only")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe state.json and restart")
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
        print(f"[RESET] State and logs cleared. Starting from ${INITIAL_CAPITAL:,.0f}.")
        print()

    global _SKIP_LIVE_FETCH
    _SKIP_LIVE_FETCH = args.no_download

    symbols = MR_UNIVERSE
    print(f"[UNIVERSE] {len(symbols)} Futures symbols with committed 1D OHLCV")

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

    state    = load_state()
    n_open   = len(state.get("open_positions", []))
    n_closed = state.get("closed_trade_count", 0)
    print(f"[STATE] equity=${state['paper_equity_usdt']:,.2f}  "
          f"open={n_open}  closed={n_closed}  "
          f"kill_sw={'YES' if state.get('kill_switch_triggered') else 'no'}")
    print()

    # Fix 3: state reconciliation (incl. Spot->Futures symbol migration)
    _reconcile_state(state, run_dates[0] if run_dates else yesterday)
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
    t9b_shared.run_engine("rsi_mr", main)
