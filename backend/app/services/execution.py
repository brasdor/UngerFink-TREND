"""Binance spot execution via ccxt.

Synchronous ccxt calls are wrapped here; async routers call these through
``asyncio.to_thread`` so the event loop is never blocked. This module knows
nothing about the database — it only talks to the exchange and does sizing math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ccxt

from app.config import Settings


class ExecutionError(Exception):
    """Raised for any execution-layer problem (config, sizing, exchange)."""


@dataclass
class SizedOrder:
    qty: float                 # rounded to the symbol's lot size
    notional_usdt: float       # qty * reference price
    reference_price: float     # entry/limit price used for sizing


def _build_client(settings: Settings) -> "ccxt.binance":
    if not settings.exchange_api_key or not settings.exchange_secret:
        raise ExecutionError(
            "Exchange API key/secret are not configured. Add them to backend/.env "
            "(use TESTNET keys first)."
        )
    client = ccxt.binance(
        {
            "apiKey": settings.exchange_api_key,
            "secret": settings.exchange_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if settings.exchange_testnet:
        client.set_sandbox_mode(True)  # route everything to Binance testnet (fake money)
    return client


class ExchangeClient:
    """Thin, reusable wrapper around a ccxt Binance spot client."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = _build_client(settings)
        self._markets_loaded = False

    # -- read-only -----------------------------------------------------------
    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            self._client.load_markets()
            self._markets_loaded = True

    def fetch_balance(self) -> dict:
        """Return free balances keyed by asset (non-zero only)."""
        bal = self._client.fetch_balance()
        free = bal.get("free", {})
        return {asset: amt for asset, amt in free.items() if amt}

    def market(self, symbol: str) -> dict:
        self._ensure_markets()
        if symbol not in self._client.markets:
            raise ExecutionError(f"Unknown market: {symbol}")
        return self._client.markets[symbol]

    # -- sizing --------------------------------------------------------------
    def size_long(
        self,
        symbol: str,
        equity_usdt: float,
        risk_pct: float,
        entry: float,
        stop: float,
    ) -> SizedOrder:
        """Size a long entry so the loss to the stop equals ``risk_pct`` of equity.

        Rounds quantity to the symbol's lot size and validates Binance's minimum
        quantity and minimum notional. Raises ExecutionError if it can't be met.
        """
        if entry <= 0 or stop <= 0 or entry <= stop:
            raise ExecutionError(f"Invalid entry/stop for long: entry={entry} stop={stop}")

        per_unit_risk = entry - stop
        risk_usdt = equity_usdt * risk_pct
        raw_qty = risk_usdt / per_unit_risk

        market = self.market(symbol)
        qty_str = self._client.amount_to_precision(symbol, raw_qty)
        qty = float(qty_str)

        limits = market.get("limits", {})
        min_qty = (limits.get("amount") or {}).get("min")
        min_notional = (limits.get("cost") or {}).get("min")

        if min_qty is not None and qty < min_qty:
            raise ExecutionError(
                f"Computed qty {qty} below exchange minimum {min_qty} for {symbol}"
            )

        notional = qty * entry
        if min_notional is not None and notional < min_notional:
            raise ExecutionError(
                f"Order notional {notional:.2f} USDT below exchange minimum "
                f"{min_notional} for {symbol}"
            )

        return SizedOrder(qty=qty, notional_usdt=notional, reference_price=entry)

    # -- placement -----------------------------------------------------------
    def place_long(
        self,
        symbol: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> dict:
        """Place a BUY order. order_type is 'market' or 'limit'.

        Returns the raw ccxt order dict. The caller persists/reconciles it.
        """
        self._ensure_markets()
        if order_type == "market":
            return self._client.create_order(symbol, "market", "buy", qty)
        if order_type == "limit":
            if limit_price is None:
                raise ExecutionError("limit order requires a limit_price")
            price_str = self._client.price_to_precision(symbol, limit_price)
            return self._client.create_order(symbol, "limit", "buy", qty, float(price_str))
        raise ExecutionError(f"Unsupported order_type: {order_type}")

    def fetch_order(self, symbol: str, exchange_order_id: str) -> dict:
        return self._client.fetch_order(exchange_order_id, symbol)
