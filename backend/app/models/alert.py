from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Alert(Base, TimestampMixin):
    """Alert rule definitions."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    alert_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # drawdown / equity_high / heat / trailing / custom

    condition: Mapped[dict] = mapped_column(JSON, nullable=False)
    strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_channels: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )  # ["browser", "discord", "telegram"]

    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_triggered: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AlertHistory(Base):
    """Alert trigger history."""

    __tablename__ = "alert_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
