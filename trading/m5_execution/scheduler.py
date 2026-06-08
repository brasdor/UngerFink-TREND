"""
M5 Execution Scheduler
======================
Orchestrates the full weekly rebalancing cycle.  Calls M2 (factor ranking),
M4 (risk state + position sizing), and the paper/live executor in sequence.

Usage
-----
  # Run one rebalancing now (no waiting for Monday)
  python -m m5_execution.scheduler --run-now --mode paper --capital 10000

  # Start the weekly scheduler (runs every Monday at 00:05 UTC)
  python -m m5_execution.scheduler --start --mode paper --capital 10000
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn
from m4_risk.circuit_breaker import CBState
from m5_execution.paper_mode import PaperPortfolio
from m5_execution.engine import BinanceExecutor
from m5_execution.reconciler import reconcile


# ── DB schema ─────────────────────────────────────────────────────────────────

def init_m5_db(conn) -> None:
    """Create M5 tables if they do not already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS position_state (
            symbol      TEXT PRIMARY KEY,
            direction   INT,
            entry_price REAL,
            notional    REAL,
            collateral  REAL,
            entry_time  INT,
            status      TEXT DEFAULT 'open'
        );

        CREATE TABLE IF NOT EXISTS execution_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INT,
            symbol    TEXT,
            action    TEXT,
            price     REAL,
            notional  REAL,
            order_id  TEXT,
            mode      TEXT,
            notes     TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            direction   INT,
            entry_price REAL,
            exit_price  REAL,
            notional    REAL,
            collateral  REAL,
            entry_time  INT,
            exit_time   INT,
            pnl         REAL,
            status      TEXT
        );
    """)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _send_telegram(message: str) -> None:
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
    except Exception as exc:
        logger.warning(f"Telegram send failed -- skipping ({exc})")


def _log_execution(
    conn,
    symbol: str,
    action: str,
    price: float,
    notional: float,
    order_id: str | None,
    mode: str,
    notes: str = "",
) -> None:
    conn.execute(
        "INSERT INTO execution_log "
        "(timestamp, symbol, action, price, notional, order_id, mode, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (int(time.time() * 1000), symbol, action, price, notional, order_id, mode, notes),
    )


def _fetch_latest_prices(conn, symbols: list[str]) -> dict[str, float]:
    """Return {symbol: latest_close} from ohlcv for the given symbol list."""
    if not symbols:
        return {}
    ph = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT o.symbol, o.close FROM ohlcv o "
        f"INNER JOIN ("
        f"  SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
        f"  WHERE symbol IN ({ph}) GROUP BY symbol"
        f") m ON o.symbol=m.symbol AND o.open_time=m.max_t",
        symbols,
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def _build_portfolio_state(conn, portfolio: PaperPortfolio, capital: float) -> dict:
    """Build M4 portfolio_state dict from current paper portfolio."""
    pnl = portfolio.get_pnl_summary()

    daily_pnl_pct = pnl["daily_pnl"] / capital if capital > 0 else 0.0
    total_pnl_pct = pnl["total_pnl"] / capital if capital > 0 else 0.0
    # peak_to_trough: if we're below start, treat it as drawdown
    peak_to_trough_pct = min(0.0, total_pnl_pct)

    # BTC 24h price change from ohlcv
    btc_rows = conn.execute(
        "SELECT close FROM ohlcv WHERE symbol='BTCUSDT' ORDER BY open_time DESC LIMIT 2"
    ).fetchall()
    btc_24h_move = 0.0
    if len(btc_rows) >= 2:
        btc_24h_move = (btc_rows[0][0] - btc_rows[1][0]) / btc_rows[1][0]

    # Consecutive losing weeks — approximate from weekly execution_log groups
    # (defaults to 0 on a fresh portfolio; improves with history)
    consec_loss = 0
    weekly_pnl = conn.execute(
        "SELECT SUM(pnl) FROM paper_trades WHERE status='closed' "
        "GROUP BY CAST(exit_time / 604800000 AS INT) ORDER BY 1 DESC LIMIT 5"
    ).fetchall()
    for row in weekly_pnl:
        if row[0] is not None and row[0] < 0:
            consec_loss += 1
        else:
            break

    # Lowest volume coin: compare latest day vol vs 30d avg for held symbols
    lowest_vol_pct = 1.0
    positions = portfolio.get_positions()
    if not positions.empty:
        syms = positions["symbol"].tolist()
        ph = ",".join("?" * len(syms))
        vol_rows = conn.execute(
            f"SELECT symbol, AVG(quote_volume) FROM ohlcv "
            f"WHERE symbol IN ({ph}) "
            f"GROUP BY symbol",
            syms,
        ).fetchall()
        avg_vol = {r[0]: float(r[1]) for r in vol_rows}

        latest_rows = conn.execute(
            f"SELECT o.symbol, o.quote_volume FROM ohlcv o "
            f"INNER JOIN ("
            f"  SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
            f"  WHERE symbol IN ({ph}) GROUP BY symbol"
            f") m ON o.symbol=m.symbol AND o.open_time=m.max_t",
            syms,
        ).fetchall()
        for sym, latest_v in latest_rows:
            avg = avg_vol.get(sym, 1.0)
            if avg > 0:
                ratio = latest_v / avg
                lowest_vol_pct = min(lowest_vol_pct, ratio)

    return {
        "daily_pnl_pct":            round(daily_pnl_pct, 6),
        "peak_to_trough_pct":       round(peak_to_trough_pct, 6),
        "btc_24h_move":             round(btc_24h_move, 6),
        "consecutive_losing_weeks": consec_loss,
        "lowest_volume_coin_pct":   round(lowest_vol_pct, 4),
    }


def _print_mode_banner(mode: str) -> None:
    line = "=" * 54
    label = f"TRADING MODE: {mode.upper()}"
    pad = (54 - len(label) - 4) // 2
    logger.info(line)
    logger.info(f"==  {' ' * pad}{label}{' ' * pad}  ==")
    logger.info(line)


# ── Paper rebalancing ─────────────────────────────────────────────────────────

def run_paper_rebalancing(capital: float, conn) -> None:
    """
    Full paper-mode rebalancing sequence (11 steps).
    Modifies paper_trades, position_state, execution_log in the supplied conn.
    """
    mode_str = "paper"
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    portfolio = PaperPortfolio(conn)

    # Step 1 — Reconcile ───────────────────────────────────────────────────────
    result = reconcile(conn, portfolio)
    if not result.ok:
        logger.error(f"Reconciliation FAILED -- aborting rebalancing: {result.mismatches}")
        _send_telegram("[M5] Rebalancing ABORTED: reconciliation failed\n" +
                       "\n".join(result.mismatches))
        return
    if result.mismatches:
        logger.warning(f"Reconcile warnings: {result.mismatches}")

    # Step 2 — Risk state from M4 ─────────────────────────────────────────────
    from m4_risk.risk_engine import get_risk_state, compute_position_sizes, init_risk_db
    init_risk_db(conn)
    port_state = _build_portfolio_state(conn, portfolio, capital)
    risk_state = get_risk_state(conn, port_state)
    logger.info(
        f"Risk state: cb={risk_state['cb_state'].value} "
        f"regime={risk_state['regime']['combined']:.3f} "
        f"pos_mult={risk_state['position_multiplier']:.3f}"
    )

    # Step 3 — Circuit breaker STOPPED ────────────────────────────────────────
    if risk_state["cb_state"] == CBState.STOPPED:
        msg = (
            f"[M5] Rebalancing ABORTED: circuit breaker STOPPED\n"
            f"Triggers: {'; '.join(risk_state['cb_triggers'])}"
        )
        logger.error(msg)
        _send_telegram(msg)
        return

    # Step 4 — Position sizes from M4 ─────────────────────────────────────────
    rankings = pd.read_sql_query(
        "SELECT * FROM weekly_rankings "
        "WHERE date=(SELECT MAX(date) FROM weekly_rankings)",
        conn,
    )
    if rankings.empty:
        logger.error("No weekly_rankings available -- aborting")
        return

    # Current equity = initial capital + realised + unrealised
    portfolio.mark_to_market()
    pnl_summary = portfolio.get_pnl_summary()
    current_equity = capital + pnl_summary["total_pnl"]

    sizes = compute_position_sizes(rankings, current_equity, risk_state, conn)
    if sizes.empty:
        logger.warning("compute_position_sizes returned empty -- skipping rebalancing")
        return

    # Step 5 — Current open positions ─────────────────────────────────────────
    current_df = portfolio.get_positions()
    current_syms: set[str] = (
        set(current_df["symbol"].tolist()) if not current_df.empty else set()
    )

    # Step 6 — Diff: close / keep / open ──────────────────────────────────────
    new_syms: set[str] = set(sizes["symbol"].tolist())
    to_close = current_syms - new_syms
    to_open  = new_syms - current_syms
    to_keep  = current_syms & new_syms

    logger.info(
        f"Diff: close={len(to_close)} open={len(to_open)} keep={len(to_keep)}"
    )

    # Fetch latest prices for all involved symbols
    all_involved = list(to_close | to_open | to_keep)
    prices = _fetch_latest_prices(conn, all_involved)

    # Step 7 — Close positions no longer in basket ────────────────────────────
    closed_ok = 0
    closed_skip = 0
    for sym in sorted(to_close):
        price = prices.get(sym, 0.0)
        if price <= 0:
            logger.warning(f"No price for {sym} -- skipping close")
            closed_skip += 1
            continue
        try:
            # Determine direction before closing (for log action label)
            dir_row = conn.execute(
                "SELECT direction FROM position_state WHERE symbol=?", (sym,)
            ).fetchone()
            direction = dir_row[0] if dir_row else 0
            action = "close_long" if direction == 1 else "close_short"

            pnl = portfolio.close_position(sym, price)
            _log_execution(conn, sym, action, price, 0.0, None, mode_str,
                           f"pnl={pnl:+.2f}")
            logger.info(f"Closed {sym} @ {price:.6f}  pnl={pnl:+.2f}")
            closed_ok += 1
        except Exception as exc:
            logger.error(f"Failed to close {sym}: {exc}")
            closed_skip += 1

    # Step 8 — Open new positions ─────────────────────────────────────────────
    opened_ok = 0
    opened_skip = 0
    for _, row in sizes[sizes["symbol"].isin(to_open)].iterrows():
        sym      = str(row["symbol"])
        direction = int(row["direction"])
        notional  = float(row["notional_size"])
        price     = prices.get(sym, 0.0)

        if price <= 0 or notional <= 0:
            logger.warning(f"Skipping {sym}: price={price} notional={notional}")
            opened_skip += 1
            continue
        try:
            action = "open_long" if direction == 1 else "open_short"
            portfolio.open_position(sym, direction, notional, price)
            _log_execution(conn, sym, action, price, notional, None, mode_str,
                           f"dir={direction}")
            logger.info(
                f"Opened {'LONG' if direction==1 else 'SHORT'} {sym} "
                f"@ {price:.6f}  notional={notional:.2f}"
            )
            opened_ok += 1
        except Exception as exc:
            logger.error(f"Failed to open {sym}: {exc}")
            opened_skip += 1

    # Step 9 — Summary log ────────────────────────────────────────────────────
    final_pnl = portfolio.get_pnl_summary()
    logger.info(
        f"Rebalancing complete: "
        f"opened={opened_ok} closed={closed_ok} kept={len(to_keep)} "
        f"skip_open={opened_skip} skip_close={closed_skip}"
    )

    # Step 10 — Telegram summary ──────────────────────────────────────────────
    tg_msg = (
        f"[M5 PAPER] Rebalancing {today_str}\n"
        f"cb={risk_state['cb_state'].value}  "
        f"regime={risk_state['regime']['combined']:.3f}  "
        f"pos_mult={risk_state['position_multiplier']:.3f}\n"
        f"opened={opened_ok}  closed={closed_ok}  kept={len(to_keep)}\n"
        f"total_pnl={final_pnl['total_pnl']:+.2f}  "
        f"equity={current_equity + final_pnl['total_pnl'] - pnl_summary['total_pnl']:.2f}"
    )
    if opened_skip or closed_skip:
        tg_msg += f"\nSkipped: open={opened_skip} close={closed_skip}"
    _send_telegram(tg_msg)


# ── Live rebalancing (Phase 1+) ───────────────────────────────────────────────

def run_rebalancing(capital: float, conn) -> None:
    """
    Full live-mode rebalancing.  Identical flow to run_paper_rebalancing but
    uses BinanceExecutor for order placement.
    Not used in Phase 0 — switch TRADING_MODE=live in .env to activate.
    """
    executor = BinanceExecutor(conn)

    result = reconcile(conn, executor)
    if not result.ok:
        logger.error(f"Reconciliation FAILED -- aborting: {result.mismatches}")
        _send_telegram("[M5] LIVE rebalancing ABORTED: reconciliation failed\n" +
                       "\n".join(result.mismatches))
        return

    from m4_risk.risk_engine import get_risk_state, compute_position_sizes, init_risk_db
    init_risk_db(conn)
    risk_state = get_risk_state(conn, {
        "daily_pnl_pct": 0.0, "peak_to_trough_pct": 0.0,
        "btc_24h_move": 0.0, "consecutive_losing_weeks": 0,
        "lowest_volume_coin_pct": 1.0,
    })

    if risk_state["cb_state"] == CBState.STOPPED:
        logger.error("CB STOPPED -- aborting live rebalancing")
        _send_telegram("[M5] LIVE rebalancing ABORTED: CB STOPPED")
        return

    rankings = pd.read_sql_query(
        "SELECT * FROM weekly_rankings WHERE date=(SELECT MAX(date) FROM weekly_rankings)",
        conn,
    )
    sizes = compute_position_sizes(rankings, capital, risk_state, conn)
    if sizes.empty:
        return

    current_df = executor.get_open_positions()
    current_syms = set(current_df["symbol"].tolist()) if not current_df.empty else set()
    new_syms = set(sizes["symbol"].tolist())
    to_close = current_syms - new_syms
    to_open  = new_syms - current_syms

    prices = _fetch_latest_prices(conn, list(to_close | to_open))

    for sym in sorted(to_close):
        price = prices.get(sym, 0.0)
        if price <= 0:
            continue
        try:
            dir_row = conn.execute(
                "SELECT direction FROM position_state WHERE symbol=?", (sym,)
            ).fetchone()
            side = "sell" if (dir_row and dir_row[0] == 1) else "buy"
            order_id = executor.place_limit_order(sym, side, 0, price)
            _log_execution(conn, sym, f"close_{side}", price, 0, order_id, "live")
        except Exception as exc:
            logger.error(f"Live close {sym}: {exc}")

    for _, row in sizes[sizes["symbol"].isin(to_open)].iterrows():
        sym      = str(row["symbol"])
        direction = int(row["direction"])
        notional  = float(row["notional_size"])
        price     = prices.get(sym, 0.0)
        if price <= 0 or notional <= 0:
            continue
        try:
            side = "buy" if direction == 1 else "sell"
            order_id = executor.place_limit_order(sym, side, notional, price)
            _log_execution(conn, sym, f"open_{side}", price, notional, order_id, "live")
        except Exception as exc:
            logger.error(f"Live open {sym}: {exc}")


# ── CLI output helpers ────────────────────────────────────────────────────────

def _print_paper_summary(conn) -> None:
    """Print paper_trades table summary after rebalancing."""
    pf = PaperPortfolio(conn)
    pnl = pf.get_pnl_summary()

    print()
    print("=" * 60)
    print("  PAPER_TRADES SUMMARY")
    print("=" * 60)
    print(f"  Open positions    : {pnl['open_positions']}")
    print(f"  Total notional    : ${pnl['total_notional']:>10,.2f}")
    print(f"  Total collateral  : ${pnl['total_collateral']:>10,.2f}")
    print(f"  Unrealised P&L    : ${pnl['unrealised_pnl']:>+10,.2f}")
    print(f"  Realised P&L      : ${pnl['realised_pnl']:>+10,.2f}")
    print(f"  Total P&L         : ${pnl['total_pnl']:>+10,.2f}")
    print("=" * 60)

    positions = pf.get_positions()
    if not positions.empty:
        print()
        print("  Open positions:")
        cols = ["symbol", "direction", "entry_price", "notional", "collateral", "pnl"]
        avail = [c for c in cols if c in positions.columns]
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.width", 120)
        print(positions[avail].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def _job(capital: float, mode: str) -> None:
    """Single rebalancing run — opens its own DB connection."""
    # Run M2 ranking BEFORE opening the scheduler connection.
    # M2 opens, writes to, and closes its own SQLite connection internally.
    # Running it here avoids the "database is locked" write conflict.
    from m2_factor_engine.engine import run_weekly_ranking
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        run_weekly_ranking(_today)
        logger.info(f"M2 ranking complete for {_today}")
    except Exception as exc:
        logger.warning(f"M2 run_weekly_ranking failed ({exc}) -- will use most recent rankings")

    with get_conn() as conn:
        init_m5_db(conn)
        if mode == "paper":
            run_paper_rebalancing(capital, conn)
        else:
            run_rebalancing(capital, conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 Execution Scheduler")
    parser.add_argument("--run-now", action="store_true",
                        help="Run one rebalancing immediately")
    parser.add_argument("--start", action="store_true",
                        help="Start weekly scheduler (every Monday 00:05 UTC)")
    parser.add_argument("--mode", default=os.getenv("TRADING_MODE", "paper"),
                        choices=["paper", "live"],
                        help="Trading mode (default: value of TRADING_MODE in .env)")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Capital in USDT (default: 10000)")
    args = parser.parse_args()

    # ── Mode banner ───────────────────────────────────────────────────────────
    _print_mode_banner(args.mode)

    if args.mode == "live":
        logger.warning(
            "LIVE mode selected.  Real orders will be placed on Binance.  "
            "Ensure BINANCE_API_KEY and BINANCE_SECRET_KEY are set correctly."
        )

    # ── Add file log ──────────────────────────────────────────────────────────
    log_path = os.path.join(_ROOT, "data", "pipeline.log")
    logger.add(log_path, rotation="10 MB", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}")

    if args.run_now:
        logger.info(f"Running immediate rebalancing: mode={args.mode} capital={args.capital}")
        _job(args.capital, args.mode)

        # Print summary from DB (read-only, new connection)
        with get_conn() as conn:
            init_m5_db(conn)
            _print_paper_summary(conn)

    elif args.start:
        import schedule

        logger.info(
            f"Starting weekly scheduler: mode={args.mode} capital={args.capital}  "
            f"(runs every Monday at 00:05 UTC)"
        )

        def _weekly():
            _job(args.capital, args.mode)

        schedule.every().monday.at("00:05").do(_weekly)

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

    else:
        parser.print_help()
