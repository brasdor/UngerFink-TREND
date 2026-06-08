"""
M4 Risk Engine
==============
Single entry point called by M5 before every rebalancing.
Orchestrates regime filter, circuit breakers, and ATR-based position sizing.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn
from m4_risk.regime_filter import regime_score
from m4_risk.circuit_breaker import (
    CBState, check_circuit_breakers, get_position_multiplier,
)
from m4_risk.position_sizer import atr, base_position_size, apply_risk_multiplier

_LONG_Q    = 5
_SHORT_Q   = 1
_LEVERAGE  = 3.0
_LONG_ALLOC  = 0.65
_SHORT_ALLOC = 0.35
_RISK_PCT    = 0.01   # 1% of capital per ATR unit


# ─── DB setup ─────────────────────────────────────────────────────────────────

def init_risk_db(conn) -> None:
    """Create risk_state table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_state (
            timestamp           INT  NOT NULL,
            cb_state            TEXT NOT NULL,
            regime_score        REAL,
            position_multiplier REAL,
            notes               TEXT,
            PRIMARY KEY (timestamp)
        )
    """)


def _log_risk_state(conn, state: dict) -> None:
    ts = int(time.time() * 1000)
    regime = state["regime"]
    notes_payload = json.dumps({
        "cb_triggers":     state.get("cb_triggers", []),
        "regime_notes":    regime.get("notes", []),
        "whale_subscores": regime.get("whale_subscores", {}),
    }, separators=(",", ":"))
    conn.execute(
        "INSERT OR REPLACE INTO risk_state "
        "(timestamp, cb_state, regime_score, position_multiplier, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            ts,
            state["cb_state"].value,
            regime["combined"],
            state["position_multiplier"],
            notes_payload[:4000],
        ),
    )


# ─── Telegram alert ───────────────────────────────────────────────────────────

def _send_telegram(message: str) -> None:
    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        logger.debug("Telegram alert sent")
    except Exception as exc:
        logger.warning(f"Telegram alert failed -- skipping ({exc})")


# ─── Main entry point ─────────────────────────────────────────────────────────

def get_risk_state(conn, portfolio_state: dict) -> dict:
    """
    Called by M5 before every rebalancing.

    portfolio_state keys
    --------------------
    daily_pnl_pct            float  today's P&L as % of equity
    peak_to_trough_pct       float  drawdown from equity peak (negative)
    btc_24h_move             float  BTC 24-hour price move (signed %)
    consecutive_losing_weeks int    consecutive weeks with negative return
    lowest_volume_coin_pct   float  worst held coin's volume vs 30d avg (0-1)

    Returns
    -------
    dict with keys:
      cb_state            CBState
      cb_triggers         list[str]
      regime              dict  (full regime_score output)
      position_multiplier float  (cb_mult x regime.combined)
      can_trade           bool   (False when STOPPED)
      summary             str    (one-line status)
    """
    init_risk_db(conn)

    # Regime
    regime = regime_score(conn)

    # Circuit breakers
    cb_state, cb_triggers = check_circuit_breakers(portfolio_state)
    cb_mult = get_position_multiplier(cb_state)

    # Combined multiplier
    pos_mult = cb_mult * regime["combined"]
    can_trade = cb_state != CBState.STOPPED

    summary = (
        f"cb={cb_state.value} regime={regime['combined']:.3f} "
        f"pos_mult={pos_mult:.3f} can_trade={can_trade}"
    )

    state = {
        "cb_state":            cb_state,
        "cb_triggers":         cb_triggers,
        "regime":              regime,
        "position_multiplier": round(pos_mult, 4),
        "can_trade":           can_trade,
        "summary":             summary,
    }

    _log_risk_state(conn, state)
    logger.info(f"risk_state: {summary}")

    # Alert if degraded
    if cb_state != CBState.OK or regime["combined"] < 0.5:
        alert_lines = [f"[M4 RISK ALERT] {summary}"]
        if cb_triggers:
            alert_lines.append("Circuit breakers: " + "; ".join(cb_triggers))
        if regime["combined"] < 0.5:
            alert_lines.append(f"Low regime score: {regime['combined']:.3f}")
        _send_telegram("\n".join(alert_lines))

    return state


# ─── Position sizing ───────────────────────────────────────────────────────────

def compute_position_sizes(
    rankings: pd.DataFrame,
    capital: float,
    risk_state: dict,
    conn,
) -> pd.DataFrame:
    """
    Compute ATR-scaled, risk-multiplied position sizes for Q5 (long) and
    Q1 (short) baskets.

    Parameters
    ----------
    rankings    : weekly_rankings DataFrame (must include symbol, quintile columns)
    capital     : current portfolio equity in USDT
    risk_state  : output of get_risk_state()
    conn        : active SQLite connection

    Returns
    -------
    DataFrame with columns: symbol, direction, notional_size, collateral, leverage
    """
    q5_syms = rankings[rankings["quintile"] == _LONG_Q]["symbol"].tolist()
    q1_syms = rankings[rankings["quintile"] == _SHORT_Q]["symbol"].tolist()

    n_long  = len(q5_syms)
    n_short = len(q1_syms)

    if not q5_syms and not q1_syms:
        logger.warning("compute_position_sizes: no Q5 or Q1 symbols in rankings")
        return pd.DataFrame()

    all_syms = q5_syms + q1_syms

    # Fetch last 20 days of closes (need 15+ for ATR) using the most recent
    # available data, not a wall-clock window (handles weekend / holiday gaps).
    max_ts = conn.execute("SELECT MAX(open_time) FROM ohlcv").fetchone()[0] or 0
    start_ts = max_ts - 25 * 86_400_000   # 25-day window for safety

    ph = ",".join("?" * len(all_syms))
    raw = pd.read_sql_query(
        f"SELECT symbol, open_time, close FROM ohlcv "
        f"WHERE symbol IN ({ph}) AND open_time >= ? "
        f"ORDER BY symbol, open_time",
        conn,
        params=all_syms + [start_ts],
    )

    pos_mult = risk_state["position_multiplier"]
    regime   = risk_state["regime"]["combined"]
    # cb_mult is embedded in pos_mult (pos_mult = cb_mult * regime)
    # For apply_risk_multiplier: pass regime and cb_mult separately
    cb_mult  = pos_mult / regime if regime > 0 else 0.0

    rows: list[dict] = []

    def _size_symbol(sym, alloc_frac, n_pos, direction):
        sym_closes = raw[raw["symbol"] == sym]["close"]
        if sym_closes.empty:
            price   = 0.0
            atr_val = 0.0
        else:
            price   = float(sym_closes.iloc[-1])
            atr_val = atr(sym_closes, period=14)

        base   = base_position_size(
            capital     = capital,
            risk_pct    = _RISK_PCT,
            atr_value   = atr_val,
            price       = price,
            leverage    = _LEVERAGE,
            n_positions = n_pos,
            alloc_frac  = alloc_frac,
        )
        final      = apply_risk_multiplier(base, regime, cb_mult)
        collateral = final / _LEVERAGE
        return {
            "symbol":        sym,
            "direction":     direction,
            "notional_size": round(final, 2),
            "collateral":    round(collateral, 2),
            "leverage":      int(_LEVERAGE),
            "price":         round(price, 6),
            "atr":           round(atr_val, 6),
            "base_size":     round(base, 2),
        }

    for sym in q5_syms:
        rows.append(_size_symbol(sym, _LONG_ALLOC, n_long, 1))

    for sym in q1_syms:
        # Short max notional per spec: long_notional_per_sym * 0.538 (= 35/65)
        rows.append(_size_symbol(sym, _SHORT_ALLOC, n_short, -1))

    return pd.DataFrame(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _run_test() -> None:
    log_path = os.path.join(_ROOT, "data", "pipeline.log")
    logger.add(log_path, rotation="10 MB", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}")

    # Zero portfolio state — no live positions yet
    portfolio_state = {
        "daily_pnl_pct":            0.0,
        "peak_to_trough_pct":       0.0,
        "btc_24h_move":             0.0,
        "consecutive_losing_weeks": 0,
        "lowest_volume_coin_pct":   1.0,
    }

    with get_conn() as conn:
        init_risk_db(conn)
        state = get_risk_state(conn, portfolio_state)

        # Load latest weekly rankings
        rankings = pd.read_sql_query(
            "SELECT * FROM weekly_rankings "
            "WHERE date = (SELECT MAX(date) FROM weekly_rankings)",
            conn,
        )

        sizes = compute_position_sizes(
            rankings    = rankings,
            capital     = 10_000.0,
            risk_state  = state,
            conn        = conn,
        )

    # ── Print risk state ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  M4 RISK STATE")
    print("=" * 60)
    print(f"  cb_state            : {state['cb_state'].value}")
    print(f"  cb_triggers         : {state['cb_triggers'] or 'none'}")
    print(f"  regime.macro        : {state['regime']['macro']:.4f}")
    print(f"  regime.whale        : {state['regime']['whale']:.4f}")
    subs = state['regime'].get('whale_subscores', {})
    print(f"    .funding_proxy    : {subs.get('funding_proxy', 'n/a')}")
    print(f"    .ls_proxy         : {subs.get('ls_proxy', 'n/a')}")
    print(f"    .oi_proxy         : {subs.get('oi_proxy', 'n/a')}")
    print(f"  regime.combined     : {state['regime']['combined']:.4f}")
    print(f"  position_multiplier : {state['position_multiplier']:.4f}")
    print(f"  can_trade           : {state['can_trade']}")
    print(f"  timestamp           : {state['regime']['timestamp']}")
    print()
    print("  Regime notes:")
    for note in state["regime"]["notes"]:
        print(f"    - {note}")
    print("=" * 60)

    # ── Print position sizing table ───────────────────────────────────────────
    if sizes.empty:
        print("\n  No positions computed.")
        return

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", 60)

    print()
    print("=" * 90)
    print(f"  POSITION SIZES  (capital $10,000 | {len(sizes)} positions | "
          f"pos_mult={state['position_multiplier']:.4f})")
    print("=" * 90)

    display_cols = ["symbol", "direction", "collateral", "notional_size", "leverage", "price", "atr"]
    avail = [c for c in display_cols if c in sizes.columns]
    longs  = sizes[sizes["direction"] ==  1][avail]
    shorts = sizes[sizes["direction"] == -1][avail]

    print(f"\n  LONG basket (Q5) -- {len(longs)} symbols:")
    if not longs.empty:
        print(longs.to_string(index=False))

    print(f"\n  SHORT basket (Q1) -- {len(shorts)} symbols:")
    if not shorts.empty:
        print(shorts.to_string(index=False))

    total_notional   = float(sizes["notional_size"].sum())
    total_collateral = float(sizes["collateral"].sum())
    print()
    print(f"  Total notional   : ${total_notional:>10,.2f}")
    print(f"  Total collateral : ${total_collateral:>10,.2f}  ({total_collateral / 10_000 * 100:.1f}% of capital)")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 Risk Engine")
    parser.add_argument("--test", action="store_true",
                        help="Run diagnostic test with zero portfolio state")
    args = parser.parse_args()

    if args.test:
        _run_test()
    else:
        parser.print_help()
