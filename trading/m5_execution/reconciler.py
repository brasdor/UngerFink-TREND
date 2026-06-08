"""
M5 Reconciler
=============
Run on every startup before any trading.  Compares position_state (SQLite)
against actual open positions (paper_trades or Binance live account) and
resolves discrepancies.

Rule
----
- Symbol in exchange but NOT in SQLite  → CRITICAL; reconcile() returns ok=False
  (do not trade until investigated)
- Symbol in SQLite but NOT in exchange  → WARNING; stale row marked 'closed'
  (exchange is ground truth)
"""
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class ReconcileResult:
    ok: bool
    mismatches: list[str] = field(default_factory=list)


def reconcile(conn, executor) -> ReconcileResult:
    """
    Parameters
    ----------
    conn     : active SQLite connection
    executor : PaperPortfolio or BinanceExecutor — must implement get_open_positions()

    Returns
    -------
    ReconcileResult.ok = True  → safe to trade
    ReconcileResult.ok = False → unknown positions found; abort rebalancing
    """
    # ── Fetch actual open positions ───────────────────────────────────────────
    try:
        actual_df = executor.get_open_positions()
    except Exception as exc:
        logger.error(f"reconcile: cannot fetch positions: {exc}")
        return ReconcileResult(ok=False, mismatches=[f"fetch error: {exc}"])

    actual_syms: set[str] = (
        set(actual_df["symbol"].tolist()) if not actual_df.empty else set()
    )

    # ── Fetch position_state ──────────────────────────────────────────────────
    try:
        db_df = pd.read_sql_query(
            "SELECT symbol FROM position_state WHERE status='open'", conn
        )
    except Exception as exc:
        logger.error(f"reconcile: cannot read position_state: {exc}")
        return ReconcileResult(ok=False, mismatches=[f"db read error: {exc}"])

    db_syms: set[str] = (
        set(db_df["symbol"].tolist()) if not db_df.empty else set()
    )

    mismatches: list[str] = []

    # ── Phantom positions: on exchange but unknown to SQLite ──────────────────
    phantom = actual_syms - db_syms
    for sym in sorted(phantom):
        msg = f"CRITICAL: {sym} open on exchange but missing from position_state"
        logger.critical(msg)
        mismatches.append(msg)

    # ── Stale records: in SQLite but not on exchange ──────────────────────────
    stale = db_syms - actual_syms
    for sym in sorted(stale):
        msg = f"WARNING: {sym} in position_state but not on exchange -- marking closed"
        logger.warning(msg)
        mismatches.append(msg)
        conn.execute(
            "UPDATE position_state SET status='closed' WHERE symbol=?", (sym,)
        )
        # Mirror to paper_trades (no-op for live mode)
        conn.execute(
            "UPDATE paper_trades SET status='closed' WHERE symbol=? AND status='open'",
            (sym,),
        )

    ok = len(phantom) == 0

    if ok and not stale:
        logger.info(f"reconcile: OK -- {len(actual_syms)} position(s) match")
    elif ok:
        logger.info(
            f"reconcile: cleaned {len(stale)} stale record(s); "
            f"{len(actual_syms)} actual position(s) confirmed"
        )
    else:
        logger.error(
            f"reconcile: FAILED -- {len(phantom)} unknown position(s) on exchange; "
            "abort rebalancing until investigated"
        )

    return ReconcileResult(ok=ok, mismatches=mismatches)
