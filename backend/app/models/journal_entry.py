from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class JournalEntry(Base, TimestampMixin):
    """Trade journal annotations."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    title: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mood: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    lesson_learned: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
