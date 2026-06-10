from datetime import datetime

from sqlalchemy import DateTime, Float, BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Candle(Base):
    """OHLCV candle data (TimescaleDB hypertable candidate)."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_candles_symbol_tf_time", "symbol", "timeframe", "timestamp", unique=True),
    )
