from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Trade(Base, TimestampMixin):
    """Closed trades from all strategies (paper and research)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # LONG / SHORT
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)

    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)

    initial_stop: Mapped[float] = mapped_column(Float, nullable=False)
    initial_risk: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_r: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    mae_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mfe_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    exit_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chandelier_activated: Mapped[Optional[bool]] = mapped_column(nullable=True)
    chandelier_activation_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    position_size_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    equity_at_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="paper"
    )  # paper / research / backtest
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_trades_entry_time", "entry_time"),
        Index("ix_trades_strategy_symbol", "strategy", "symbol"),
    )
