"""
M7 Squeeze Executor
===================
Paper-trade entry and exit for squeeze positions.

Writes directly to paper_trades (with notes='squeeze_trade') rather than
delegating to PaperPortfolio so it can tag and filter by trade type.
Requires the notes column added by init_m7_db().

Capital budget: $2,000 split across MAX_POSITIONS=3 at 3× leverage.
  collateral per trade = 2000 / 3 ≈ $666.67
  notional per trade   = collateral × 3 ≈ $2,001
"""
import os
import sys
import time

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_NOTES_TAG  = "squeeze_trade"
_LEVERAGE   = 3.0


class SqueezeExecutor:
    CAPITAL_BUDGET   = 2_000.0
    MAX_POSITIONS    = 3
    MAX_HOLD_DAYS    = 5
    STOP_LOSS_PCT    = 0.15   # 15% loss → exit
    TAKE_PROFIT_PCT  = 0.30   # 30% gain → exit
    SCORE_DECAY_EXIT = 40     # re-score below this → exit

    # ── entry ──────────────────────────────────────────────────────────────────

    def open_squeeze_trade(
        self,
        symbol: str,
        score: float,
        entry_price: float,
        conn,
    ) -> int | None:
        """
        Open a LONG paper trade tagged as a squeeze trade.
        Returns paper_trades.id or None if budget is exhausted / price is invalid.
        """
        if entry_price <= 0:
            logger.warning(f"squeeze open {symbol}: invalid price {entry_price}")
            return None

        # Count currently open squeeze positions
        n_open = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status='open' AND notes=?",
            (_NOTES_TAG,),
        ).fetchone()[0]

        if n_open >= self.MAX_POSITIONS:
            logger.info(
                f"squeeze open {symbol}: budget exhausted "
                f"({n_open}/{self.MAX_POSITIONS} positions open)"
            )
            return None

        collateral = round(self.CAPITAL_BUDGET / self.MAX_POSITIONS, 2)
        notional   = round(collateral * _LEVERAGE, 2)
        ts         = int(time.time() * 1000)

        cur = conn.execute(
            "INSERT INTO paper_trades "
            "(symbol, direction, entry_price, exit_price, notional, collateral, "
            " entry_time, exit_time, pnl, status, notes) "
            "VALUES (?, 1, ?, NULL, ?, ?, ?, NULL, 0.0, 'open', ?)",
            (symbol, entry_price, notional, collateral, ts, _NOTES_TAG),
        )
        trade_id = cur.lastrowid
        logger.info(
            f"[M7] Opened squeeze LONG {symbol} @ {entry_price:.6f}  "
            f"score={score:.0f}  collateral={collateral:.2f}  notional={notional:.2f}  "
            f"id={trade_id}"
        )
        return trade_id

    # ── exit ───────────────────────────────────────────────────────────────────

    def check_exits(self, conn) -> list[dict]:
        """
        Evaluate all open squeeze trades for exit conditions (in priority order):
          1. Stop loss:   price < entry × (1 - STOP_LOSS_PCT)
          2. Take profit: price > entry × (1 + TAKE_PROFIT_PCT)
          3. Max hold:    entry_time older than MAX_HOLD_DAYS days
          4. Score decay: rescore < SCORE_DECAY_EXIT

        Returns list of closed-trade dicts for caller to alert on.
        """
        try:
            rows = conn.execute(
                "SELECT id, symbol, entry_price, entry_time, notional, collateral "
                "FROM paper_trades WHERE status='open' AND notes=?",
                (_NOTES_TAG,),
            ).fetchall()
        except Exception as exc:
            logger.warning(f"check_exits: could not read squeeze trades: {exc}")
            return []

        if not rows:
            return []

        # Fetch latest prices for all held symbols
        symbols = [r[1] for r in rows]
        ph = ",".join("?" * len(symbols))
        price_rows = conn.execute(
            f"SELECT o.symbol, o.close FROM ohlcv o "
            f"INNER JOIN ("
            f"  SELECT symbol, MAX(open_time) AS max_t FROM ohlcv "
            f"  WHERE symbol IN ({ph}) GROUP BY symbol"
            f") m ON o.symbol=m.symbol AND o.open_time=m.max_t",
            symbols,
        ).fetchall()
        prices = {r[0]: float(r[1]) for r in price_rows}

        now_ms       = int(time.time() * 1000)
        max_age_ms   = self.MAX_HOLD_DAYS * 86_400_000
        closed_trades = []

        # Lazy import to avoid circular dependency at module level
        from m7_squeeze.scorer import rescore_symbol

        for trade_id, symbol, entry_price, entry_time, notional, collateral in rows:
            current_price = prices.get(symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"check_exits {symbol}: no current price — skipping")
                continue

            pnl = (current_price - entry_price) / entry_price * notional   # direction=1 (long)

            reason = None

            # 1. Stop loss
            if current_price < entry_price * (1 - self.STOP_LOSS_PCT):
                reason = f"stop_loss (price={current_price:.6f} entry={entry_price:.6f})"

            # 2. Take profit
            elif current_price > entry_price * (1 + self.TAKE_PROFIT_PCT):
                reason = f"take_profit (price={current_price:.6f} entry={entry_price:.6f})"

            # 3. Max hold
            elif (now_ms - entry_time) > max_age_ms:
                age_days = (now_ms - entry_time) / 86_400_000
                reason = f"max_hold ({age_days:.1f}d > {self.MAX_HOLD_DAYS}d)"

            # 4. Score decay (most expensive check — done last)
            else:
                current_score = rescore_symbol(symbol, conn)
                if current_score < self.SCORE_DECAY_EXIT:
                    reason = f"score_decay (score={current_score:.0f} < {self.SCORE_DECAY_EXIT})"

            if reason:
                ts = int(time.time() * 1000)
                conn.execute(
                    "UPDATE paper_trades "
                    "SET exit_price=?, exit_time=?, pnl=?, status='closed' WHERE id=?",
                    (round(current_price, 8), ts, round(pnl, 6), trade_id),
                )
                ret_pct = pnl / collateral * 100 if collateral > 0 else 0.0
                logger.info(
                    f"[M7] Closed squeeze {symbol} @ {current_price:.6f}  "
                    f"pnl={pnl:+.2f}  ret={ret_pct:+.1f}%  reason={reason}"
                )
                closed_trades.append({
                    "symbol":        symbol,
                    "trade_id":      trade_id,
                    "entry_price":   entry_price,
                    "exit_price":    current_price,
                    "pnl":           round(pnl, 2),
                    "return_pct":    round(ret_pct, 2),
                    "reason":        reason,
                })

        return closed_trades

    # ── summary ────────────────────────────────────────────────────────────────

    def get_squeeze_summary(self, conn) -> dict:
        """Open squeeze positions, realised P&L, and win rate."""
        try:
            n_open, open_notional, open_collateral = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(notional),0), COALESCE(SUM(collateral),0) "
                "FROM paper_trades WHERE status='open' AND notes=?",
                (_NOTES_TAG,),
            ).fetchone()

            closed = conn.execute(
                "SELECT pnl FROM paper_trades WHERE status='closed' AND notes=?",
                (_NOTES_TAG,),
            ).fetchall()
            pnls         = [float(r[0]) for r in closed if r[0] is not None]
            realised_pnl = round(sum(pnls), 2)
            n_closed     = len(pnls)
            n_wins       = sum(1 for p in pnls if p > 0)
            win_rate     = round(n_wins / n_closed * 100, 1) if n_closed else 0.0

            open_positions = pd.read_sql_query(
                "SELECT id, symbol, entry_price, notional, collateral, pnl, entry_time "
                "FROM paper_trades WHERE status='open' AND notes=?",
                conn, params=[_NOTES_TAG],
            )

            return {
                "open_count":       n_open,
                "open_notional":    round(float(open_notional), 2),
                "open_collateral":  round(float(open_collateral), 2),
                "realised_pnl":     realised_pnl,
                "closed_count":     n_closed,
                "win_rate":         win_rate,
                "open_positions":   open_positions,
            }
        except Exception as exc:
            logger.warning(f"get_squeeze_summary failed: {exc}")
            return {
                "open_count": 0, "open_notional": 0.0, "open_collateral": 0.0,
                "realised_pnl": 0.0, "closed_count": 0, "win_rate": 0.0,
                "open_positions": pd.DataFrame(),
            }
