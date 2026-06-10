from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EquitySnapshot(Base):
    """Time-series equity snapshots (TimescaleDB hypertable candidate)."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    equity_usdt: Mapped[float] = mapped_column(Float, nullable=False)
    open_pnl_usdt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    closed_pnl_usdt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)

    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    portfolio_heat_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
