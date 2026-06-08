"""
M5 Paper Mode
=============
Simulated portfolio backed by SQLite.  Mirrors the BinanceExecutor interface
so the scheduler can switch modes by swapping the executor object.
"""
import time

import pandas as pd
from loguru import logger

_LEVERAGE = 3.0


class PaperPortfolio:
    """
    Manages simulated positions in the paper_trades and position_state tables.
    All methods assume they are called inside an active get_conn() context so
    the caller controls commit/rollback.
    """

    def __init__(self, conn):
        self._conn = conn

    # ── position management ────────────────────────────────────────────────────

    def open_position(
        self, symbol: str, direction: int, notional: float, price: float
    ) -> int:
        """
        Record a new simulated entry.
        Returns the paper_trades.id of the new row.
        direction: +1 = long, -1 = short.
        """
        collateral = notional / _LEVERAGE
        ts = int(time.time() * 1000)
        cur = self._conn.execute(
            "INSERT INTO paper_trades "
            "(symbol, direction, entry_price, exit_price, notional, collateral, "
            " entry_time, exit_time, pnl, status) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, 0.0, 'open')",
            (symbol, direction, price, notional, collateral, ts),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO position_state "
            "(symbol, direction, entry_price, notional, collateral, entry_time, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'open')",
            (symbol, direction, price, notional, collateral, ts),
        )
        logger.debug(
            f"paper: opened {'LONG' if direction == 1 else 'SHORT'} "
            f"{symbol} @ {price:.6f}  notional={notional:.2f}"
        )
        return cur.lastrowid

    def close_position(self, symbol: str, price: float) -> float:
        """
        Record exit for the most-recently-opened open position for symbol.
        Returns realised P&L.
        """
        row = self._conn.execute(
            "SELECT id, direction, entry_price, notional FROM paper_trades "
            "WHERE symbol=? AND status='open' ORDER BY entry_time DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if row is None:
            logger.warning(f"paper: no open position to close for {symbol}")
            return 0.0
        trade_id, direction, entry_price, notional = row
        pnl = (price - entry_price) / entry_price * notional * direction
        ts = int(time.time() * 1000)
        self._conn.execute(
            "UPDATE paper_trades SET exit_price=?, exit_time=?, pnl=?, status='closed' WHERE id=?",
            (price, ts, round(pnl, 6), trade_id),
        )
        self._conn.execute("DELETE FROM position_state WHERE symbol=?", (symbol,))
        logger.debug(f"paper: closed {symbol} @ {price:.6f}  pnl={pnl:+.2f}")
        return pnl

    def get_positions(self) -> pd.DataFrame:
        """Return all currently open simulated positions."""
        return pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE status='open' ORDER BY entry_time",
            self._conn,
        )

    def get_open_positions(self) -> pd.DataFrame:
        """Alias matching BinanceExecutor interface for reconciler duck-typing."""
        return self.get_positions()

    # ── analytics ─────────────────────────────────────────────────────────────

    def get_pnl_summary(self) -> dict:
        """
        Returns dict:
          realised_pnl    — closed trades only
          unrealised_pnl  — open trades (last mark_to_market value stored in pnl)
          total_pnl       — realised + unrealised
          daily_pnl       — P&L from positions opened or closed in last 24h
          open_positions  — count
          total_notional  — sum of open position notionals
          total_collateral — sum of open position collaterals
        """
        realised = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM paper_trades WHERE status='closed'"
        ).fetchone()[0]
        unrealised = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM paper_trades WHERE status='open'"
        ).fetchone()[0]
        today_start = int(time.time() * 1000) - 86_400_000
        daily = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM paper_trades "
            "WHERE exit_time >= ? OR (status='open' AND entry_time >= ?)",
            (today_start, today_start),
        ).fetchone()[0]
        n_open, tot_notional, tot_collateral = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(notional), 0.0), COALESCE(SUM(collateral), 0.0) "
            "FROM paper_trades WHERE status='open'"
        ).fetchone()
        return {
            "realised_pnl":    round(realised, 2),
            "unrealised_pnl":  round(unrealised, 2),
            "total_pnl":       round(realised + unrealised, 2),
            "daily_pnl":       round(daily, 2),
            "open_positions":  n_open,
            "total_notional":  round(tot_notional, 2),
            "total_collateral": round(tot_collateral, 2),
        }

    def mark_to_market(self, current_prices: dict | None = None) -> dict:
        """
        Recompute unrealised P&L for every open position and persist to pnl column.
        If current_prices is None, fetches the latest close from ohlcv.
        Returns {symbol: unrealised_pnl}.
        """
        positions = self.get_positions()
        if positions.empty:
            return {}

        if current_prices is None:
            syms = positions["symbol"].tolist()
            ph = ",".join("?" * len(syms))
            rows = self._conn.execute(
                f"SELECT o.symbol, o.close "
                f"FROM ohlcv o "
                f"INNER JOIN ("
                f"  SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
                f"  WHERE symbol IN ({ph}) GROUP BY symbol"
                f") m ON o.symbol=m.symbol AND o.open_time=m.max_t",
                syms,
            ).fetchall()
            current_prices = {r[0]: float(r[1]) for r in rows}

        result: dict = {}
        for _, pos in positions.iterrows():
            sym = pos["symbol"]
            px = current_prices.get(sym)
            if not px or px <= 0:
                continue
            pnl = (
                (px - pos["entry_price"]) / pos["entry_price"]
                * pos["notional"]
                * pos["direction"]
            )
            self._conn.execute(
                "UPDATE paper_trades SET pnl=? WHERE id=?",
                (round(pnl, 6), int(pos["id"])),
            )
            result[sym] = round(pnl, 4)

        return result
