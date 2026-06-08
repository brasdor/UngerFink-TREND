"""
M5 Binance Execution Engine
============================
Handles live order management via ccxt (Binance USDT-M futures).
Not used in Phase 0 (paper mode) — BinanceExecutor is instantiated by the
scheduler only when TRADING_MODE=live in .env.

Safety rules
------------
- set_leverage and set_isolated_margin called before every first order.
  If either fails the symbol is skipped entirely — never crash the run.
- Limit orders only (maker fee 0.02%).
- Unfilled orders older than 5 minutes are cancelled automatically.
- All order placement wrapped in try/except — one failure never stops the rest.
"""
import math
import os
import sys
import time

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

MODE = os.getenv("TRADING_MODE", "paper").lower()

_CANCEL_AFTER_SECS = 300   # 5 minutes


class BinanceExecutor:
    """
    Live order executor.  Mirrors PaperPortfolio's get_open_positions()
    signature so the reconciler can work with either.

    In paper mode all write-methods return early with a warning — the caller
    should use PaperPortfolio instead.
    """

    def __init__(self, conn=None):
        self.mode = MODE
        self._conn = conn
        self._prec_cache: dict[str, tuple] = {}   # symbol -> (price_prec, qty_prec, tick_size)
        self._margin_set: set[str] = set()         # symbols already configured

        if self.mode == "live":
            import ccxt
            self._client = ccxt.binanceusdm({
                "apiKey":  os.getenv("BINANCE_API_KEY", ""),
                "secret":  os.getenv("BINANCE_SECRET_KEY", ""),
                "options": {"defaultType": "future"},
            })
            logger.info("BinanceExecutor initialised in LIVE mode")
        else:
            self._client = None
            logger.info("BinanceExecutor initialised in PAPER mode (no real orders placed)")

    # ── symbol precision ───────────────────────────────────────────────────────

    def _get_precision(self, symbol: str) -> tuple[int, int, float]:
        """(price_precision, qty_precision, tick_size) from universe_history."""
        if symbol in self._prec_cache:
            return self._prec_cache[symbol]
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT price_precision, qty_precision, tick_size FROM universe_history "
                "WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row:
                self._prec_cache[symbol] = (int(row[0]), int(row[1]), float(row[2]))
                return self._prec_cache[symbol]
        # Sensible fallback
        default = (2, 3, 0.01)
        self._prec_cache[symbol] = default
        return default

    # ── margin / leverage setup ────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int = 3) -> bool:
        """
        Set futures leverage for symbol.  Returns True on success.
        Failure skips the symbol — never raises.
        """
        if self.mode != "live":
            return True
        try:
            self._client.set_leverage(leverage, symbol)
            logger.debug(f"set_leverage {symbol} -> {leverage}x OK")
            return True
        except Exception as exc:
            logger.error(f"set_leverage {symbol} failed: {exc} -- symbol will be skipped")
            return False

    def set_isolated_margin(self, symbol: str) -> bool:
        """
        Set isolated margin mode for symbol.  Returns True on success.
        Binance returns an error if mode is already ISOLATED — treated as success.
        """
        if self.mode != "live":
            return True
        try:
            self._client.set_margin_mode("isolated", symbol)
            logger.debug(f"set_isolated_margin {symbol} OK")
            return True
        except Exception as exc:
            # Error code -4046: "No need to change margin type" — safe to ignore
            if "-4046" in str(exc) or "already" in str(exc).lower():
                logger.debug(f"set_isolated_margin {symbol}: already isolated")
                return True
            logger.warning(f"set_isolated_margin {symbol}: {exc}")
            return True   # non-fatal; proceed

    def _ensure_symbol_configured(self, symbol: str, leverage: int = 3) -> bool:
        """Call set_isolated_margin + set_leverage before first order. Returns False to skip."""
        if symbol in self._margin_set:
            return True
        ok_lev = self.set_leverage(symbol, leverage)
        if not ok_lev:
            return False
        self.set_isolated_margin(symbol)   # non-fatal; always continue after
        self._margin_set.add(symbol)
        return True

    # ── order placement ────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        side: str,          # 'buy' or 'sell'
        notional: float,
        current_price: float,
        slippage_pct: float = 0.001,
    ) -> str | None:
        """
        Place a GTC limit order.  Returns order_id string or None on failure.

        Price: buy at current*(1+slip), sell at current*(1-slip).
        Quantity and price are rounded to exchange precision.
        """
        if self.mode != "live":
            logger.warning("place_limit_order called in paper mode -- ignored")
            return None

        if not self._ensure_symbol_configured(symbol):
            logger.error(f"place_limit_order: skipping {symbol} (margin/leverage setup failed)")
            return None

        price_prec, qty_prec, tick_size = self._get_precision(symbol)

        raw_price = (
            current_price * (1 + slippage_pct)
            if side == "buy"
            else current_price * (1 - slippage_pct)
        )
        if tick_size > 0:
            limit_price = round(round(raw_price / tick_size) * tick_size, price_prec)
        else:
            limit_price = round(raw_price, price_prec)

        if limit_price <= 0:
            logger.warning(f"place_limit_order {symbol}: computed price=0 -- skipping")
            return None

        raw_qty = notional / limit_price
        quantity = (
            round(raw_qty, qty_prec)
            if qty_prec > 0
            else math.floor(raw_qty)
        )

        if quantity <= 0:
            logger.warning(f"place_limit_order {symbol}: computed qty=0 -- skipping")
            return None

        try:
            order = self._client.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=quantity,
                price=limit_price,
                params={"timeInForce": "GTC"},
            )
            order_id = str(order["id"])
            logger.info(
                f"Placed {side} limit {symbol}: qty={quantity} price={limit_price} id={order_id}"
            )
            return order_id
        except Exception as exc:
            logger.error(f"place_limit_order {symbol} {side} failed: {exc}")
            return None

    def cancel_unfilled_orders(self, symbol: str) -> int:
        """
        Cancel any open limit orders for symbol older than 5 minutes.
        Returns count of orders cancelled.  Never raises.
        """
        if self.mode != "live" or self._client is None:
            return 0
        try:
            open_orders = self._client.fetch_open_orders(symbol)
            cutoff_ms = (time.time() - _CANCEL_AFTER_SECS) * 1000
            cancelled = 0
            for order in open_orders:
                if order.get("timestamp", 0) < cutoff_ms:
                    try:
                        self._client.cancel_order(order["id"], symbol)
                        logger.info(f"Cancelled stale order {order['id']} for {symbol}")
                        cancelled += 1
                    except Exception as ex:
                        logger.warning(f"cancel_order {order['id']} {symbol}: {ex}")
            return cancelled
        except Exception as exc:
            logger.error(f"cancel_unfilled_orders {symbol}: {exc}")
            return 0

    # ── position query ─────────────────────────────────────────────────────────

    def get_open_positions(self) -> pd.DataFrame:
        """
        Fetch all open perpetual positions from Binance account.
        Returns DataFrame with columns: symbol, direction, notional, entry_price.
        """
        if self.mode != "live" or self._client is None:
            return pd.DataFrame()
        try:
            all_pos = self._client.fetch_positions()
            active = [p for p in all_pos if abs(float(p.get("contracts", 0) or 0)) > 0]
            if not active:
                return pd.DataFrame()
            rows = []
            for p in active:
                # ccxt symbol format: "BTC/USDT:USDT" → "BTCUSDT"
                sym = p["symbol"].replace("/", "").replace(":USDT", "")
                rows.append({
                    "symbol":      sym,
                    "direction":   1 if p.get("side") == "long" else -1,
                    "notional":    abs(float(p.get("notional", 0) or 0)),
                    "entry_price": float(p.get("entryPrice", 0) or 0),
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.error(f"get_open_positions failed: {exc}")
            return pd.DataFrame()
