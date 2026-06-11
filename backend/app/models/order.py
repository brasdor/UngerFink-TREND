from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Order(Base, TimestampMixin):
    """A live (or testnet) order placed on the exchange.

    Records what we *intended* to do and what *actually* filled, so the two can
    be reconciled. Every order carries a unique idempotency key so the same
    signal can never be placed twice.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # What this order is for
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY / SELL
    signal_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Guards against double-placement (one signal -> one order)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    # What we intended
    order_type: Mapped[str] = mapped_column(String(8), nullable=False)  # market / limit
    requested_qty: Mapped[float] = mapped_column(Float, nullable=False)
    requested_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # limit px
    requested_notional_usdt: Mapped[float] = mapped_column(Float, nullable=False)
    intended_stop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    equity_at_request: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Execution context / safety
    testnet: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # What actually happened
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending / submitted / filled / partially_filled / rejected / canceled / error
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    filled_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fill_notional_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fee: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fee_currency: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Reconciliation (intended vs actual)
    reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reconcile_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_orders_strategy_symbol", "strategy", "symbol"),
        Index("ix_orders_status", "status"),
    )
