"""
M7 Squeeze Scanner — daily orchestrator
========================================
Invoked by run_daily.bat Step 4.

Flow each run:
  1. Init DB schema (idempotent)
  2. Check exits on all open squeeze trades
  3. Run compute_squeeze_scores()
  4. Log all scored symbols to squeeze_scan_log
  5. Open paper trade for highest-scoring HIGH_CONVICTION candidate
     if open positions < MAX_POSITIONS
  6. Send Telegram alert for HIGH_CONVICTION candidates and exits
  7. Print scan summary to console

CLI:
  python -m m7_squeeze.scanner
"""
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
from m7_squeeze.scorer import compute_squeeze_scores
from m7_squeeze.executor import SqueezeExecutor


# ── DB schema ──────────────────────────────────────────────────────────────────

def init_m7_db(conn) -> None:
    """Create M7 tables and add notes column to paper_trades if absent."""
    # squeeze_scan_log — one row per symbol per daily scan
    conn.execute("""
        CREATE TABLE IF NOT EXISTS squeeze_scan_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date       TEXT,
            symbol          TEXT,
            total_score     REAL,
            funding_score   REAL,
            ls_score        REAL,
            oi_score        REAL,
            volume_score    REAL,
            threshold       TEXT,
            avg_funding_rate REAL,
            long_ratio      REAL,
            oi_change_pct   REAL,
            caution_flag    INT DEFAULT 0
        )
    """)

    # Add notes column to paper_trades (no-op if already present)
    try:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN notes TEXT DEFAULT NULL")
        logger.debug("Added notes column to paper_trades")
    except Exception:
        pass   # column already exists


# ── Telegram helper ────────────────────────────────────────────────────────────

def _telegram(text: str) -> None:
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=5,
        )
    except Exception as exc:
        logger.warning(f"Telegram send failed: {exc}")


# ── price helper ───────────────────────────────────────────────────────────────

def _latest_price(symbol: str, conn) -> float:
    """Return most recent close from ohlcv, or 0.0 if unavailable."""
    row = conn.execute(
        "SELECT close FROM ohlcv WHERE symbol=? ORDER BY open_time DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return float(row[0]) if row and row[0] else 0.0


# ── log scan results ───────────────────────────────────────────────────────────

def _log_scan(scores: pd.DataFrame, scan_date: str, conn) -> None:
    """Insert all scored rows into squeeze_scan_log."""
    if scores.empty:
        return
    for _, row in scores.iterrows():
        conn.execute(
            "INSERT INTO squeeze_scan_log "
            "(scan_date, symbol, total_score, funding_score, ls_score, oi_score, "
            " volume_score, threshold, avg_funding_rate, long_ratio, oi_change_pct, caution_flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                scan_date,
                row.get("symbol"),
                row.get("total_score"),
                row.get("funding_score"),
                row.get("ls_score"),
                row.get("oi_score"),
                row.get("volume_score"),
                row.get("threshold"),
                row.get("avg_funding_rate"),
                row.get("long_ratio"),
                row.get("oi_change_pct"),
                int(row.get("caution_flag", 0)),
            ),
        )


# ── main daily scan ────────────────────────────────────────────────────────────

def run_daily_scan(conn) -> None:
    """Called daily by run_daily.bat Step 4."""
    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    executor  = SqueezeExecutor()

    print()
    print("=" * 68)
    print(f"  M7 SQUEEZE SCANNER  |  {scan_date}  UTC")
    print("=" * 68)

    # Step 1 — check exits on existing squeeze trades ─────────────────────────
    closed = executor.check_exits(conn)
    if closed:
        print(f"\n  Exits processed: {len(closed)}")
        for t in closed:
            sign = "+" if t["pnl"] >= 0 else ""
            print(
                f"    CLOSED {t['symbol']:<18} "
                f"pnl={sign}{t['pnl']:.2f}  ret={sign}{t['return_pct']:.1f}%  "
                f"reason={t['reason']}"
            )
            _telegram(
                f"[M7 SQUEEZE EXIT] {t['symbol']}\n"
                f"P&L: {sign}{t['pnl']:.2f} USDT ({sign}{t['return_pct']:.1f}%)\n"
                f"Reason: {t['reason']}"
            )
    else:
        print("\n  Exits processed: 0")

    # Step 2 — score all universe symbols ─────────────────────────────────────
    scores = compute_squeeze_scores(conn)

    # Step 3 — log all candidates to squeeze_scan_log ─────────────────────────
    if not scores.empty:
        _log_scan(scores, scan_date, conn)

    # Step 4 — print scan table (symbols with score >= 40) ────────────────────
    above_40 = scores[scores["total_score"] >= 40] if not scores.empty else pd.DataFrame()
    print(f"\n  Symbols scored >= 40:  {len(above_40)}")

    if not above_40.empty:
        hdr = (
            f"\n  {'Symbol':<18} {'Score':>6}  {'Fund':>5}  {'L/S':>5}  "
            f"{'OI':>5}  {'Vol':>5}  {'Threshold':<15}  {'Caution'}"
        )
        sep = "  " + "-" * 78
        print(hdr)
        print(sep)
        for _, r in above_40.iterrows():
            caution_str = " [CAUTION]" if int(r.get("caution_flag", 0)) else ""
            print(
                f"  {r['symbol']:<18} {r['total_score']:>6.0f}  "
                f"{r['funding_score']:>5.0f}  {r['ls_score']:>5.0f}  "
                f"{r['oi_score']:>5.0f}  {r['volume_score']:>5.0f}  "
                f"{r['threshold']:<15}{caution_str}"
            )
    else:
        print("  (no symbols above threshold)")

    # HIGH_CONVICTION candidates
    hc = scores[scores["threshold"] == "HIGH_CONVICTION"] if not scores.empty else pd.DataFrame()
    print(f"\n  HIGH_CONVICTION candidates (score >= 85):  {len(hc)}")

    # Step 5 — open new squeeze trade if budget allows ────────────────────────
    n_open = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status='open' AND notes='squeeze_trade'"
    ).fetchone()[0]

    opened_trades = []
    if not hc.empty and n_open < executor.MAX_POSITIONS:
        slots = executor.MAX_POSITIONS - n_open
        for _, candidate in hc.iterrows():
            if slots <= 0:
                break
            sym   = candidate["symbol"]
            score = float(candidate["total_score"])
            price = _latest_price(sym, conn)
            if price <= 0:
                logger.warning(f"M7: no price for {sym} — skipping trade open")
                continue
            trade_id = executor.open_squeeze_trade(sym, score, price, conn)
            if trade_id is not None:
                opened_trades.append({
                    "symbol":     sym,
                    "score":      score,
                    "price":      price,
                    "trade_id":   trade_id,
                    "funding":    candidate.get("avg_funding_rate"),
                    "ls_ratio":   candidate.get("long_ratio"),
                })
                slots -= 1

                _telegram(
                    f"[M7 SQUEEZE ENTRY] {sym}\n"
                    f"Score: {score:.0f}  Price: {price:.6f}\n"
                    f"Funding: {candidate.get('avg_funding_rate', float('nan')):.6f}  "
                    f"L/S ratio: {candidate.get('long_ratio', float('nan')):.3f}"
                )

    if opened_trades:
        print(f"\n  New trades opened: {len(opened_trades)}")
        for t in opened_trades:
            print(
                f"    OPENED {t['symbol']:<18} "
                f"score={t['score']:.0f}  price={t['price']:.6f}  id={t['trade_id']}"
            )
    else:
        print(f"\n  New trades opened: 0")

    # Step 6 — summary ─────────────────────────────────────────────────────────
    summary = executor.get_squeeze_summary(conn)
    print()
    print("  Squeeze portfolio summary:")
    print(f"    Open positions   : {summary['open_count']}/{executor.MAX_POSITIONS}")
    print(f"    Open notional    : ${summary['open_notional']:>10,.2f}")
    print(f"    Open collateral  : ${summary['open_collateral']:>10,.2f}")
    print(f"    Realised P&L     : ${summary['realised_pnl']:>+10,.2f}")
    print(f"    Closed trades    : {summary['closed_count']}")
    print(f"    Win rate         : {summary['win_rate']:.1f}%")

    # Open positions detail
    op = summary["open_positions"]
    if not op.empty:
        print()
        print("  Open squeeze positions:")
        syms = op["symbol"].tolist()
        live_prices = {
            r[0]: float(r[1])
            for r in conn.execute(
                f"SELECT o.symbol, o.close FROM ohlcv o "
                f"INNER JOIN (SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
                f"WHERE symbol IN ({','.join('?'*len(syms))}) GROUP BY symbol) m "
                f"ON o.symbol=m.symbol AND o.open_time=m.max_t",
                syms,
            ).fetchall()
        }
        hdr2 = f"    {'Symbol':<18} {'Entry':>10} {'Now':>10} {'P&L':>9} {'Ret%':>7}"
        print(hdr2)
        print("    " + "-" * 60)
        for _, pos in op.iterrows():
            sym = pos["symbol"]
            ep  = float(pos["entry_price"])
            cp  = live_prices.get(sym, ep)
            ntnl = float(pos["notional"])
            coll = float(pos["collateral"])
            pnl  = (cp - ep) / ep * ntnl
            ret  = pnl / coll * 100 if coll > 0 else 0.0
            print(
                f"    {sym:<18} {ep:>10.5f} {cp:>10.5f} "
                f"{pnl:>+9.2f} {ret:>+6.2f}%"
            )

    # Exclude-from-candidates confirmation
    factor_open = conn.execute(
        "SELECT COUNT(*) FROM paper_trades "
        "WHERE status='open' AND (notes IS NULL OR notes != 'squeeze_trade')"
    ).fetchone()[0]
    print(f"\n  Factor trades excluded from candidates: {factor_open}")

    # Scan log row count
    log_count = conn.execute(
        "SELECT COUNT(*) FROM squeeze_scan_log WHERE scan_date=?", (scan_date,)
    ).fetchone()[0]
    print(f"  squeeze_scan_log rows inserted today:   {log_count}")
    print("=" * 68)
    print()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_path = os.path.join(_ROOT, "data", "pipeline.log")
    logger.add(
        log_path,
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )

    with get_conn() as conn:
        # Ensure M5 tables exist (paper_trades may not be present on fresh DB)
        from m5_execution.scheduler import init_m5_db
        init_m5_db(conn)

        # Ensure M4 risk_state table exists (needed by scorer CB check)
        from m4_risk.risk_engine import init_risk_db
        init_risk_db(conn)

        # Ensure M7 schema
        init_m7_db(conn)

        run_daily_scan(conn)
