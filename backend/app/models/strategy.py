from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Strategy(Base, TimestampMixin):
    """Frozen strategy configurations."""

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # trend / mean_reversion

    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="binance_spot")

    frozen_config: Mapped[dict] = mapped_column(JSON, nullable=False)

    risk_per_trade: Mapped[float] = mapped_column(Float, nullable=False, default=0.0025)
    max_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    max_heat_pct: Mapped[float] = mapped_column(Float, nullable=False, default=1.5)
    kill_switch_dd_pct: Mapped[float] = mapped_column(Float, nullable=False, default=35.0)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_paper_live: Mapped[bool] = mapped_column(Boolean, default=False)
    paper_start_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    universe_symbols: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
