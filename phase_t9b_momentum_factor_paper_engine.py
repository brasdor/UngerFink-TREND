#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9B MOMENTUM FACTOR -- PAPER TRADING ENGINE
============================================

Frozen config (2026-06-07):
  Universe  : 290 Binance Futures symbols (data/futures_universe/ohlcv_1d/)
  Signal    : 20-day return ranking
  Long      : top 20 by return, price above EMA200
  Short     : bottom 10 by return, no filter
  Rebalance : biweekly (every 2 Fridays, ~14 days)
  Allocation: 50% equity long / 50% equity short, equal-weight per basket
  Cost      : 0.15% round-trip on turnover fraction
  Leverage  : 1.0x (futures, not leveraged)
  Capital   : $10,000 starting equity

Data source:
  CSVs from data/futures_universe/ohlcv_1d/ (from Phase 1 download).
  Delta update via ccxt binanceusdm on each run (local PC only, not GitHub Actions).
  Use --no-download to skip live fetch.

Usage:
  python phase_t9b_momentum_factor_paper_engine.py
  python phase_t9b_momentum_factor_paper_engine.py --no-download
  python phase_t9b_momentum_factor_paper_engine.py --reset
  python phase_t9b_momentum_factor_paper_engine.py --date 2026-06-20
  python phase_t9b_momentum_factor_paper_engine.py --notify

Output (data/t9b_momentum_paper/):
  state.json          persistent state (positions, equity, last rebal date)
  daily_log.csv       all events (REBAL, MTM, SKIP)
  open_positions.csv  current basket with MTM
  signals_today.csv   new basket on rebalance days only
  equity_curve.csv    one row per rebalance (equity history)

PAPER ONLY -- NO REAL ORDERS.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import t9b_shared

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
OHLCV_DIR  = ROOT / "data" / "futures_universe" / "ohlcv_1d"
SYM_FILE   = ROOT / "data" / "futures_universe" / "all_symbols.csv"
OUT_DIR    = ROOT / "data" / "t9b_momentum_paper"
STATE_FILE = OUT_DIR / "state.json"
LOG_FILE   = OUT_DIR / "daily_log.csv"
POS_FILE   = OUT_DIR / "open_positions.csv"
SIG_FILE   = OUT_DIR / "signals_today.csv"
EQ_FILE    = OUT_DIR / "equity_curve.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Frozen config ─────────────────────────────────────────────────────────
SYSTEM_NAME  = "MomentumFactor_Lb20_L20_S10_Biweekly_T9B"
LOOKBACK     = 20
LONG_K       = 20
SHORT_K      = 10
EMA_SPAN     = 200
LONG_ALLOC   = 0.50   # fraction of equity to long basket
SHORT_ALLOC  = 0.50   # fraction of equity to short basket
REBAL_DAYS   = 14     # minimum days between rebalances
COST_RT      = 0.0015 # 0.15% round-trip on turnover fraction
STARTING_EQ  = 10_000.0
EPS          = 1e-10
MIN_ORDER_SIZE_USDT  = 15.0        # Fix 2: minimum per-position notional
MIN_SHORT_VOL_USDT   = 1_000_000   # Fix 5: min avg daily USDT volume for shorts
FUNDING_WARN_RATE    = 0.0005      # Fix 4: warn if funding > 0.05% per 8h


# ── Utilities ─────────────────────────────────────────────────────────────

def p(*a, **kw):
    kw.setdefault('flush', True)
    text = ' '.join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode(), **kw)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T9B Momentum Factor Paper Engine")
    ap.add_argument('--date',        default=None, help='Run date YYYY-MM-DD (default: today)')
    ap.add_argument('--no-download', action='store_true', help='Skip live ccxt data update')
    ap.add_argument('--reset',       action='store_true', help='Wipe state and restart from $10k')
    ap.add_argument('--notify',      action='store_true', help='Print compact summary at end')
    return ap.parse_args()


# ── State ─────────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "system_name":        SYSTEM_NAME,
        "created_utc":        datetime.utcnow().isoformat(),
        "last_run_date":      None,
        "last_rebal_date":    None,
        "paper_equity_usdt":  STARTING_EQ,
        "peak_equity_usdt":   STARTING_EQ,
        "drawdown_pct":       0.0,
        "total_rebal_count":  0,
        "long_positions":     [],
        "short_positions":    [],
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    return _empty_state()


def save_state(state: dict) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, default=str)


# ── Data update (delta via ccxt) ──────────────────────────────────────────

def update_ohlcv_delta(symbols: list[str], full_update: bool = False) -> int:
    """
    Fetch new candles since last date for each symbol and append to CSV.
    full_update=True fetches all symbols; False fetches only those with CSV.
    Returns count of symbols updated.
    """
    try:
        import ccxt
    except ImportError:
        p("  [WARN] ccxt not installed -- skipping data update")
        return 0

    p(f"  Fetching delta candles for {len(symbols)} symbols via ccxt binanceusdm...")
    exchange = ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 30_000})

    try:
        markets = exchange.load_markets()
    except Exception as exc:
        p(f"  [WARN] load_markets failed: {exc} -- using cached data")
        return 0

    # Build sym_id -> ccxt_symbol mapping
    id_to_ccxt: dict[str, str] = {}
    for ccxt_sym, mkt in markets.items():
        if (mkt.get('type') == 'swap'
                and mkt.get('quote') == 'USDT'
                and mkt.get('settle') == 'USDT'):
            id_to_ccxt[mkt['id']] = ccxt_sym

    updated = errors = skipped = 0
    for sym_id in symbols:
        csv_path = OHLCV_DIR / f"{sym_id}_1d.csv"
        if not csv_path.exists():
            skipped += 1
            continue

        ccxt_sym = id_to_ccxt.get(sym_id)
        if not ccxt_sym:
            skipped += 1
            continue

        try:
            df_old  = pd.read_csv(csv_path)
            last_ts = int(df_old['timestamp'].max())
            since   = last_ts + 86_400_000  # one day after last bar (ms)

            batch = exchange.fetch_ohlcv(ccxt_sym, '1d', since=since, limit=10)
            if not batch:
                updated += 1
                continue

            df_new = pd.DataFrame(
                batch, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['date'] = (pd.to_datetime(df_new['timestamp'], unit='ms', utc=True)
                              .dt.strftime('%Y-%m-%d'))

            df_combined = (pd.concat([df_old, df_new])
                           .drop_duplicates('timestamp')
                           .sort_values('timestamp')
                           .reset_index(drop=True))
            df_combined.to_csv(csv_path, index=False)
            updated += 1

        except Exception:
            errors += 1

        time.sleep(0.05)

    p(f"  Delta update: {updated} OK  |  {errors} errors  |  {skipped} skipped")
    return updated


# ── Universe loading ───────────────────────────────────────────────────────

def load_close_matrix() -> pd.DataFrame:
    """Load close prices for all symbols from CSVs. Returns (date x symbol) DataFrame."""
    if not SYM_FILE.exists():
        raise FileNotFoundError(f"Symbol list not found: {SYM_FILE}")

    syms = pd.read_csv(SYM_FILE)['symbol'].tolist()
    frames: dict[str, pd.Series] = {}
    for sym in syms:
        path = OHLCV_DIR / f"{sym}_1d.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates('date').set_index('date').sort_index()
            if len(df) >= LOOKBACK + 5:
                frames[sym] = df['close'].astype(float)
        except Exception:
            pass

    return pd.DataFrame(frames).sort_index()


def load_vol_usdt_matrix() -> pd.DataFrame:
    """Load avg daily USDT volume (close * volume) matrix for Fix 5 liquidity filter."""
    if not SYM_FILE.exists():
        return pd.DataFrame()
    syms = pd.read_csv(SYM_FILE)['symbol'].tolist()
    frames: dict[str, pd.Series] = {}
    for sym in syms:
        path = OHLCV_DIR / f"{sym}_1d.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=['date', 'close', 'volume'])
            df['date']    = pd.to_datetime(df['date'])
            df            = df.drop_duplicates('date').set_index('date').sort_index()
            close         = pd.to_numeric(df['close'],  errors='coerce')
            vol           = pd.to_numeric(df['volume'], errors='coerce')
            frames[sym]   = (close * vol).astype(float)
        except Exception:
            pass
    return pd.DataFrame(frames).sort_index()


def _reconcile_state_momentum(state: dict, run_date) -> None:
    """Fix 3: Verify open positions consistency on engine startup."""
    issues: list[str] = []
    for side, key in [('LONG', 'long_positions'), ('SHORT', 'short_positions')]:
        for pos in state.get(key, []):
            sym  = pos.get('symbol', '')
            ep   = pos.get('entry_price')
            qty  = pos.get('quantity')
            notl = pos.get('notional')
            ed   = pos.get('entry_date')
            if not sym:
                issues.append(f"{side}: missing symbol")
            try:
                if float(ep) <= 0:
                    issues.append(f"{side}/{sym}: entry_price={ep} <= 0")
            except (TypeError, ValueError):
                issues.append(f"{side}/{sym}: entry_price not numeric ({ep!r})")
            try:
                if float(qty) <= 0:
                    issues.append(f"{side}/{sym}: quantity={qty} <= 0")
            except (TypeError, ValueError):
                issues.append(f"{side}/{sym}: quantity not numeric ({qty!r})")
            try:
                if float(notl) <= 0:
                    issues.append(f"{side}/{sym}: notional={notl} <= 0")
            except (TypeError, ValueError):
                issues.append(f"{side}/{sym}: notional not numeric ({notl!r})")
            try:
                from datetime import date as _date
                _date.fromisoformat(str(ed))
            except (ValueError, TypeError):
                issues.append(f"{side}/{sym}: invalid entry_date={ed!r}")

    n_long  = len(state.get('long_positions', []))
    n_short = len(state.get('short_positions', []))
    if issues:
        p(f"[RECONCILE] {len(issues)} issue(s):")
        for iss in issues:
            p(f"  [WARN] {iss}")
        _log_event(run_date, 'RECONCILE_WARN',
                   f"{len(issues)} issue(s): " + " | ".join(issues),
                   float(state.get('paper_equity_usdt', STARTING_EQ)))
    else:
        p(f"[RECONCILE] OK  ({n_long} longs, {n_short} shorts verified)")


def check_funding_rates(short_positions: list[dict]) -> None:
    """Fix 4: Fetch and warn on high funding rates for short positions."""
    if not short_positions:
        return
    try:
        import ccxt
    except ImportError:
        p("  [SKIP] ccxt not installed -- funding rate check skipped")
        return
    p("  Checking funding rates for short positions...")
    try:
        exchange = ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 20_000})
        markets  = exchange.load_markets()
    except Exception as exc:
        p(f"  [WARN] Cannot connect for funding rate check: {exc}")
        return

    id_to_ccxt: dict[str, str] = {}
    for ccxt_sym, mkt in markets.items():
        if (mkt.get('type') == 'swap' and mkt.get('quote') == 'USDT'
                and mkt.get('settle') == 'USDT'):
            id_to_ccxt[mkt['id']] = ccxt_sym

    warned = 0
    for pos in short_positions:
        sym_id   = pos['symbol']
        ccxt_sym = id_to_ccxt.get(sym_id)
        if not ccxt_sym:
            p(f"  [SKIP] {sym_id}: not found in markets")
            continue
        try:
            info = exchange.fetch_funding_rate(ccxt_sym)
            rate = float(info.get('fundingRate', 0.0) or 0.0)
            rate_pct = rate * 100
            if rate > FUNDING_WARN_RATE:
                ann_pct = rate * 3 * 365 * 100
                p(f"  [WARN] HIGH FUNDING: {sym_id}  {rate_pct:.4f}%/8h  "
                  f"(ann ~{ann_pct:.0f}%)  -- FLAG FOR MANUAL REVIEW")
                warned += 1
            else:
                p(f"  [INFO] {sym_id}  funding={rate_pct:.4f}%/8h  OK")
            time.sleep(0.1)
        except Exception as exc:
            p(f"  [WARN] {sym_id}: funding fetch failed ({exc})")

    if warned:
        p(f"  [SUMMARY] {warned}/{len(short_positions)} shorts have HIGH funding rate")
    else:
        p(f"  [SUMMARY] All {len(short_positions)} shorts funding OK")


def get_latest_price(close: pd.DataFrame, sym: str, as_of: Date) -> float | None:
    """Return latest available close price for sym on or before as_of."""
    if sym not in close.columns:
        return None
    series = close[sym].dropna()
    series = series[series.index <= pd.Timestamp(as_of)]
    return float(series.iloc[-1]) if not series.empty else None


# ── Signal computation ─────────────────────────────────────────────────────

def compute_basket(close: pd.DataFrame, as_of: Date,
                   vol_usdt: pd.DataFrame | None = None) -> tuple[list[str], list[str], dict]:
    """
    Compute long and short baskets.
    Returns (long_syms, short_syms, metadata_dict).
    Applies Fix 5 (liquidity filter on shorts) and Fix 1 (cross-system dedup).
    """
    ts = pd.Timestamp(as_of)
    avail = close.index[close.index <= ts]
    if len(avail) < LOOKBACK + 5:
        return [], [], {}

    t0   = avail[-1]                          # most recent available bar
    t_lb = avail[-LOOKBACK - 1]               # LOOKBACK bars ago

    close_t0  = close.loc[t0]
    close_tlb = close.loc[t_lb]

    ret = (close_t0 / close_tlb - 1).replace([np.inf, -np.inf], np.nan).dropna()

    # EMA200 at t0
    ema200_series = close.ewm(span=EMA_SPAN, min_periods=20, adjust=False).mean().loc[t0]

    # Long universe: price > EMA200
    long_eligible = [
        s for s in ret.index
        if (pd.notna(close_t0.get(s))
            and pd.notna(ema200_series.get(s))
            and float(close_t0.get(s, 0)) > float(ema200_series.get(s, 0)))
    ]
    sig_long  = ret.reindex(long_eligible).dropna()
    sig_short = ret.copy()

    # Fix 5: liquidity filter on short universe
    excluded_illiquid = 0
    if vol_usdt is not None and not vol_usdt.empty:
        avail_vol = vol_usdt.index[vol_usdt.index <= ts]
        if len(avail_vol) >= LOOKBACK:
            avg_vol = vol_usdt.loc[avail_vol[-LOOKBACK:]].mean()
            liquid_syms = set(avg_vol[avg_vol >= MIN_SHORT_VOL_USDT].index)
            n_before = len(sig_short)
            sig_short = sig_short.reindex(
                [s for s in sig_short.index if s in liquid_syms]).dropna()
            excluded_illiquid = n_before - len(sig_short)

    meta = {
        'signal_date':       str(t0.date()),
        'lookback_from':     str(t_lb.date()),
        'n_universe':        len(ret),
        'n_long_eligible':   len(long_eligible),
        'n_illiquid_excluded': excluded_illiquid,
    }

    if len(sig_short) < SHORT_K:
        meta['skip_reason'] = f"Only {len(sig_short)} liquid shorts available (need {SHORT_K})"
        return [], [], meta

    actual_long_k = min(len(sig_long), LONG_K)
    longs  = list(sig_long.nlargest(actual_long_k).index) if actual_long_k > 0 else []

    # Fix 1: cross-system duplicate filter
    cross_syms = t9b_shared.get_cross_system_symbols('momentum')
    n_before_l = len(longs)
    longs  = [s for s in longs  if t9b_shared.normalize_sym(s) not in cross_syms]
    # Refill shorts from full ranked list (skip cross-system duplicates)
    shorts_candidates = [s for s in sig_short.nsmallest(len(sig_short)).index
                         if t9b_shared.normalize_sym(s) not in cross_syms]
    shorts = shorts_candidates[:SHORT_K]

    if n_before_l != len(longs):
        meta['cross_dedup_longs'] = n_before_l - len(longs)
    if len(shorts) < SHORT_K:
        meta['cross_dedup_shorts'] = f"Only {len(shorts)}/{SHORT_K} after cross-system dedup"

    if not shorts:
        return [], [], meta

    meta['actual_long_k'] = len(longs)
    meta['long_min_ret']  = (round(float(sig_long.nlargest(len(longs)).iloc[-1]) * 100, 2)
                             if longs else None)
    meta['short_max_ret'] = round(float(sig_short.reindex(shorts).iloc[-1]) * 100, 2) if shorts else None
    if len(longs) < LONG_K:
        meta['long_note'] = (f"Only {len(longs)}/{LONG_K} symbols above EMA200 "
                             f"-- bear market condition, {len(longs)} longs opened")
    if excluded_illiquid:
        p(f"  [INFO] Fix5: {excluded_illiquid} illiquid symbols excluded from short universe")

    return longs, shorts, meta


# ── Rebalance logic ───────────────────────────────────────────────────────

def is_rebal_day(today: Date, state: dict) -> bool:
    """
    True if today should trigger a rebalance.
    Rule: first run always rebalances. Subsequent: Friday + >= REBAL_DAYS since last.
    """
    if state['last_rebal_date'] is None:
        return True  # first run: rebalance immediately
    last = Date.fromisoformat(state['last_rebal_date'])
    days_since = (today - last).days
    if days_since < REBAL_DAYS:
        return False
    return today.weekday() == 4  # 4 = Friday


def do_rebalance(state: dict, close: pd.DataFrame, today: Date,
                 vol_usdt: pd.DataFrame | None = None) -> list[dict]:
    """
    Execute basket rebalance. Updates state in-place.
    Returns list of signal rows for signals_today.csv.
    """
    equity = float(state['paper_equity_usdt'])

    # Current baskets
    old_longs  = {pos['symbol'] for pos in state['long_positions']}
    old_shorts = {pos['symbol'] for pos in state['short_positions']}

    # Compute unrealized P&L from current positions (to realize into equity)
    realized_pnl = 0.0
    for pos in state['long_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        realized_pnl += (price - pos['entry_price']) * pos['quantity']
    for pos in state['short_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        realized_pnl += (pos['entry_price'] - price) * pos['quantity']
    equity += realized_pnl

    # New baskets (Fix 5 liquidity filter + Fix 1 cross-system dedup applied inside)
    longs, shorts, meta = compute_basket(close, today, vol_usdt=vol_usdt)

    if not shorts:
        p("  [WARN] Cannot fill short basket -- rebalance skipped")
        _log_event(today, 'SKIP', 'Cannot fill short basket', equity)
        return []
    if 'long_note' in meta:
        p(f"  [INFO] {meta['long_note']}")

    new_longs  = set(longs)
    new_shorts = set(shorts)

    # Turnover
    opened_l = new_longs  - old_longs
    closed_l = old_longs  - new_longs
    held_l   = old_longs  & new_longs
    opened_s = new_shorts - old_shorts
    closed_s = old_shorts - new_shorts
    held_s   = old_shorts & new_shorts

    turn_l = (len(opened_l | closed_l) / max(len(old_longs | new_longs), 1)
              if old_longs else 1.0)
    turn_s = (len(opened_s | closed_s) / max(len(old_shorts | new_shorts), 1)
              if old_shorts else 1.0)
    cost_frac = ((turn_l + turn_s) / 2.0) * COST_RT
    equity_after_cost = equity * (1.0 - cost_frac)

    # Build new position list
    # If no longs (EMA200 filter in bear market), 50% stays as cash
    long_alloc  = equity_after_cost * LONG_ALLOC  / len(longs)  if longs  else 0.0
    short_alloc = equity_after_cost * SHORT_ALLOC / max(len(shorts), 1)

    new_long_pos  = []
    new_short_pos = []
    signal_rows   = []

    for rank, sym in enumerate(longs, 1):
        price = get_latest_price(close, sym, today)
        if not price or price <= 0:
            continue
        qty = long_alloc / price
        # Fix 2: minimum order size validation
        if long_alloc < MIN_ORDER_SIZE_USDT:
            _log_event(today, 'SKIP',
                       f"position_too_small: {sym} LONG alloc=${long_alloc:.2f} < ${MIN_ORDER_SIZE_USDT}",
                       equity_after_cost)
            continue
        new_long_pos.append({
            'symbol':      sym,
            'side':        'LONG',
            'entry_date':  str(today),
            'entry_price': round(price, 8),
            'quantity':    round(qty, 8),
            'notional':    round(qty * price, 4),
        })
        action = 'HOLD' if sym in held_l else 'OPEN'
        signal_rows.append({
            'symbol': sym, 'side': 'LONG', 'action': action,
            'rank': rank, 'entry_price': round(price, 8),
            'quantity': round(qty, 8), 'notional': round(qty * price, 4),
            'period_return_pct': round(
                (close.loc[close.index[close.index <= pd.Timestamp(today)][-1], sym]
                 / close.iloc[-LOOKBACK - 1].get(sym, np.nan) - 1) * 100, 2)
            if sym in close.columns else np.nan,
        })

    for rank, sym in enumerate(shorts, 1):
        price = get_latest_price(close, sym, today)
        if not price or price <= 0:
            continue
        qty = short_alloc / price
        # Fix 2: minimum order size validation
        if short_alloc < MIN_ORDER_SIZE_USDT:
            _log_event(today, 'SKIP',
                       f"position_too_small: {sym} SHORT alloc=${short_alloc:.2f} < ${MIN_ORDER_SIZE_USDT}",
                       equity_after_cost)
            continue
        new_short_pos.append({
            'symbol':      sym,
            'side':        'SHORT',
            'entry_date':  str(today),
            'entry_price': round(price, 8),
            'quantity':    round(qty, 8),
            'notional':    round(qty * price, 4),
        })
        action = 'HOLD' if sym in held_s else 'OPEN'
        signal_rows.append({
            'symbol': sym, 'side': 'SHORT', 'action': action,
            'rank': rank, 'entry_price': round(price, 8),
            'quantity': round(qty, 8), 'notional': round(qty * price, 4),
            'period_return_pct': np.nan,
        })

    for sym in closed_l:
        signal_rows.append({'symbol': sym, 'side': 'LONG', 'action': 'CLOSE',
                            'rank': 0, 'entry_price': np.nan,
                            'quantity': np.nan, 'notional': np.nan,
                            'period_return_pct': np.nan})
    for sym in closed_s:
        signal_rows.append({'symbol': sym, 'side': 'SHORT', 'action': 'CLOSE',
                            'rank': 0, 'entry_price': np.nan,
                            'quantity': np.nan, 'notional': np.nan,
                            'period_return_pct': np.nan})

    # Update state
    state['long_positions']    = new_long_pos
    state['short_positions']   = new_short_pos
    state['last_rebal_date']   = str(today)
    state['paper_equity_usdt'] = round(equity_after_cost, 4)
    state['peak_equity_usdt']  = round(
        max(state['peak_equity_usdt'], equity_after_cost), 4)
    state['drawdown_pct']      = round(
        (equity_after_cost - state['peak_equity_usdt'])
        / max(state['peak_equity_usdt'], EPS) * 100, 4)
    state['total_rebal_count'] = state.get('total_rebal_count', 0) + 1

    # Log
    period_ret = (equity - float(state.get('_last_rebal_equity', STARTING_EQ))) \
                 / max(float(state.get('_last_rebal_equity', STARTING_EQ)), EPS) * 100
    state['_last_rebal_equity'] = equity_after_cost

    detail = (f"REBAL #{state['total_rebal_count']}  "
              f"L={len(new_long_pos)} ({len(opened_l)} new {len(held_l)} held {len(closed_l)} closed)  "
              f"S={len(new_short_pos)} ({len(opened_s)} new {len(held_s)} held {len(closed_s)} closed)  "
              f"pnl=${realized_pnl:+.2f}  cost={cost_frac*100:.3f}%  "
              f"eq=${equity_after_cost:,.2f}")
    _log_event(today, 'REBAL', detail, equity_after_cost)

    # Equity curve
    _append_equity_curve(today, 'REBAL', equity_after_cost, realized_pnl, cost_frac)

    p(f"  Rebalance #{state['total_rebal_count']}:")
    p(f"    Period P&L:    ${realized_pnl:>+,.2f}  "
      f"({(realized_pnl / max(state.get('_last_rebal_equity', STARTING_EQ), EPS)) * 100:>+.2f}%)")
    p(f"    Cost:          -{cost_frac*100:.3f}%  "
      f"(Lturn={turn_l:.0%}  Sturn={turn_s:.0%})")
    p(f"    New equity:    ${equity_after_cost:>,.2f}")
    p(f"    Long basket:   {len(new_long_pos):>2} positions  ({len(opened_l)} new, {len(held_l)} held, {len(closed_l)} closed)")
    p(f"    Short basket:  {len(new_short_pos):>2} positions  ({len(opened_s)} new, {len(held_s)} held, {len(closed_s)} closed)")
    p(f"    Signal date:   {meta.get('signal_date', '?')}  "
      f"(universe={meta.get('n_universe', 0)}, long_eligible={meta.get('n_long_eligible', 0)})")

    return signal_rows


# ── Daily MTM (non-rebalance days) ────────────────────────────────────────

def compute_unrealized_pnl(state: dict, close: pd.DataFrame, today: Date) -> float:
    """Compute total unrealized P&L across all open positions."""
    pnl = 0.0
    for pos in state['long_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        pnl  += (price - pos['entry_price']) * pos['quantity']
    for pos in state['short_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        pnl  += (pos['entry_price'] - price) * pos['quantity']
    return pnl


def do_daily_mtm(state: dict, close: pd.DataFrame, today: Date) -> None:
    """Compute and log daily MTM (no position changes)."""
    pnl  = compute_unrealized_pnl(state, close, today)
    base = float(state['paper_equity_usdt'])
    pct  = pnl / max(base, EPS) * 100
    mtm_equity = base + pnl

    detail = (f"MTM  unrealized_pnl=${pnl:+,.2f}  ({pct:>+.2f}%)  "
              f"mtm_equity=${mtm_equity:,.2f}  "
              f"days_since_rebal={(today - Date.fromisoformat(state['last_rebal_date'])).days}"
              if state['last_rebal_date'] else "MTM  no positions")
    _log_event(today, 'MTM', detail, mtm_equity)


# ── Output writers ─────────────────────────────────────────────────────────

def _log_event(run_date: Date, event: str, detail: str, equity: float) -> None:
    exists = LOG_FILE.exists()
    with open(LOG_FILE, 'a', encoding='utf-8', newline='') as f:
        if not exists:
            f.write('run_date,event,detail,equity\n')
        detail_clean = detail.replace('"', "'").replace('\n', ' ')
        f.write(f'{run_date},{event},"{detail_clean}",{equity:.4f}\n')


def _append_equity_curve(run_date: Date, event: str,
                         equity: float, pnl: float, cost_frac: float) -> None:
    exists = EQ_FILE.exists()
    with open(EQ_FILE, 'a', encoding='utf-8', newline='') as f:
        if not exists:
            f.write('date,event,equity,pnl_usdt,cost_pct\n')
        f.write(f'{run_date},{event},{equity:.4f},{pnl:.4f},{cost_frac*100:.4f}\n')


def write_open_positions(state: dict, close: pd.DataFrame, today: Date) -> None:
    rows = []
    for pos in state['long_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        upnl  = (price - pos['entry_price']) * pos['quantity']
        upct  = (price / max(pos['entry_price'], EPS) - 1) * 100
        days  = (today - Date.fromisoformat(pos['entry_date'])).days
        rows.append({
            'symbol':          pos['symbol'],
            'side':            'LONG',
            'entry_date':      pos['entry_date'],
            'entry_price':     pos['entry_price'],
            'current_price':   round(price, 8),
            'quantity':        pos['quantity'],
            'notional_usdt':   round(pos['quantity'] * price, 4),
            'unrealized_pnl':  round(upnl, 4),
            'unrealized_pct':  round(upct, 3),
            'days_held':       days,
        })
    for pos in state['short_positions']:
        price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
        upnl  = (pos['entry_price'] - price) * pos['quantity']
        upct  = (pos['entry_price'] / max(price, EPS) - 1) * 100
        days  = (today - Date.fromisoformat(pos['entry_date'])).days
        rows.append({
            'symbol':          pos['symbol'],
            'side':            'SHORT',
            'entry_date':      pos['entry_date'],
            'entry_price':     pos['entry_price'],
            'current_price':   round(price, 8),
            'quantity':        pos['quantity'],
            'notional_usdt':   round(pos['quantity'] * price, 4),
            'unrealized_pnl':  round(upnl, 4),
            'unrealized_pct':  round(upct, 3),
            'days_held':       days,
        })
    pd.DataFrame(rows).to_csv(POS_FILE, index=False)


def write_signals(signal_rows: list[dict]) -> None:
    if not signal_rows:
        pd.DataFrame().to_csv(SIG_FILE, index=False)
        return
    df = pd.DataFrame(signal_rows)
    df = df.sort_values(['side', 'action', 'rank'])
    df.to_csv(SIG_FILE, index=False)


def print_notify(state: dict, close: pd.DataFrame, today: Date, rebalanced: bool) -> None:
    pnl  = compute_unrealized_pnl(state, close, today)
    base = float(state['paper_equity_usdt'])
    p()
    p("=" * 60)
    p(f"  T9B MOMENTUM FACTOR  |  {today}  |  PAPER")
    p("=" * 60)
    p(f"  Equity (realized):   ${base:>10,.2f}")
    p(f"  Unrealized P&L:      ${pnl:>+10,.2f}")
    p(f"  MTM equity:          ${base + pnl:>10,.2f}")
    p(f"  Peak equity:         ${state['peak_equity_usdt']:>10,.2f}")
    p(f"  Drawdown:            {state['drawdown_pct']:>+9.2f}%")
    p(f"  Long positions:      {len(state['long_positions']):>4}")
    p(f"  Short positions:     {len(state['short_positions']):>4}")
    p(f"  Rebalanced today:    {'YES' if rebalanced else 'no'}")
    if state['last_rebal_date']:
        last = Date.fromisoformat(state['last_rebal_date'])
        nxt  = last + timedelta(days=REBAL_DAYS)
        # advance to next Friday on or after nxt
        while nxt.weekday() != 4:
            nxt += timedelta(days=1)
        p(f"  Last rebal:          {state['last_rebal_date']}")
        p(f"  Next rebal target:   {nxt}")
    p("=" * 60)

    if state['long_positions']:
        p()
        p("  LONG BASKET:")
        p(f"  {'Symbol':<16}  {'Entry':>10}  {'Now':>10}  {'Qty':>12}  {'P&L':>10}  {'Pct':>7}")
        p("  " + "-" * 68)
        for pos in sorted(state['long_positions'], key=lambda x: x['symbol']):
            price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
            upnl  = (price - pos['entry_price']) * pos['quantity']
            upct  = (price / max(pos['entry_price'], EPS) - 1) * 100
            p(f"  {pos['symbol']:<16}  {pos['entry_price']:>10.6f}  {price:>10.6f}  "
              f"{pos['quantity']:>12.6f}  ${upnl:>+8.2f}  {upct:>+6.2f}%")

    if state['short_positions']:
        p()
        p("  SHORT BASKET:")
        p(f"  {'Symbol':<16}  {'Entry':>10}  {'Now':>10}  {'Qty':>12}  {'P&L':>10}  {'Pct':>7}")
        p("  " + "-" * 68)
        for pos in sorted(state['short_positions'], key=lambda x: x['symbol']):
            price = get_latest_price(close, pos['symbol'], today) or pos['entry_price']
            upnl  = (pos['entry_price'] - price) * pos['quantity']
            upct  = (pos['entry_price'] / max(price, EPS) - 1) * 100
            p(f"  {pos['symbol']:<16}  {pos['entry_price']:>10.6f}  {price:>10.6f}  "
              f"{pos['quantity']:>12.6f}  ${upnl:>+8.2f}  {upct:>+6.2f}%")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    today = (Date.fromisoformat(args.date) if args.date
             else Date.today())

    p("=" * 60)
    p(f"  T9B MOMENTUM FACTOR PAPER ENGINE  |  {today}")
    p("=" * 60)

    # State
    if args.reset:
        p("  --reset: wiping state")
        state = _empty_state()
    else:
        state = load_state()

    if state['last_run_date'] == str(today):
        p("  Already ran today -- exiting (use --reset or --date to override)")
        return

    # Data update
    all_syms = (pd.read_csv(SYM_FILE)['symbol'].tolist()
                if SYM_FILE.exists() else [])

    if not args.no_download and all_syms:
        # On rebalance days update all; otherwise only current basket
        rebal_check = is_rebal_day(today, state)
        if rebal_check:
            update_syms = all_syms
        else:
            basket_syms = (
                [pos['symbol'] for pos in state['long_positions']]
                + [pos['symbol'] for pos in state['short_positions']]
            )
            update_syms = basket_syms if basket_syms else all_syms[:50]
        update_ohlcv_delta(update_syms)
    else:
        p("  --no-download: using cached CSVs")

    # Fix 3: state reconciliation on startup
    _reconcile_state_momentum(state, today)

    # Load close matrix
    p("  Loading close matrix...")
    close = load_close_matrix()
    if close.empty:
        p("  ERROR: no close data loaded -- aborting")
        return
    p(f"  Loaded: {close.shape[1]} symbols  |  "
      f"{close.index[0].date()} to {close.index[-1].date()}")

    # Rebalance or MTM
    rebalanced   = False
    signal_rows  = []

    if is_rebal_day(today, state):
        p(f"\n  REBALANCE DAY")
        # Fix 5: load volume matrix for liquidity filter
        vol_usdt = load_vol_usdt_matrix()
        signal_rows = do_rebalance(state, close, today, vol_usdt=vol_usdt)
        rebalanced  = True
        # Fix 4: funding rate monitor for short positions after rebalance
        if state.get('short_positions'):
            p()
            check_funding_rates(state['short_positions'])
    else:
        days_since = 0
        if state['last_rebal_date']:
            days_since = (today - Date.fromisoformat(state['last_rebal_date'])).days
        next_friday = today + timedelta(days=1)
        while next_friday.weekday() != 4:
            next_friday += timedelta(days=1)
        p(f"\n  MTM day (days since last rebal: {days_since})")
        do_daily_mtm(state, close, today)

    # Write outputs
    write_open_positions(state, close, today)
    if rebalanced:
        write_signals(signal_rows)

    # Update state
    state['last_run_date'] = str(today)
    save_state(state)

    if args.notify:
        print_notify(state, close, today, rebalanced)
    else:
        pnl  = compute_unrealized_pnl(state, close, today)
        base = float(state['paper_equity_usdt'])
        p(f"\n  Done.  Equity=${base:,.2f}  Unrealized P&L=${pnl:+,.2f}  "
          f"MTM=${base+pnl:,.2f}  "
          f"L={len(state['long_positions'])}  S={len(state['short_positions'])}")

    p()


if __name__ == '__main__':
    main()
