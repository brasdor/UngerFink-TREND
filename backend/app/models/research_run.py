from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ResearchRun(Base, TimestampMixin):
    """Metadata for research phase runs."""

    __tablename__ = "research_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    phase: Mapped[str] = mapped_column(String(8), nullable=False)  # T1, T2, ... T18
    phase_name: Mapped[str] = mapped_column(String(128), nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # pending / running / passed / failed / warn

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    gate_result: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # PASS / FAIL / WARN
    gate_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    output_dir: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    summary_stats: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    total_trades: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
