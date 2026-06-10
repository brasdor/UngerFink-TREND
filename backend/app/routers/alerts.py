from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.alert import Alert, AlertHistory

router = APIRouter()


@router.get("/")
async def list_alerts(db: AsyncSession = Depends(get_db)):
    """List all alert rules."""
    result = await db.execute(select(Alert).order_by(Alert.name))
    alerts = result.scalars().all()
    return {"alerts": [
        {
            "id": a.id,
            "name": a.name,
            "alert_type": a.alert_type,
            "condition": a.condition,
            "strategy": a.strategy,
            "is_enabled": a.is_enabled,
            "notify_channels": a.notify_channels,
            "last_triggered": a.last_triggered.isoformat() if a.last_triggered else None,
        }
        for a in alerts
    ]}


@router.get("/history")
async def alert_history(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get recent alert trigger history."""
    result = await db.execute(
        select(AlertHistory).order_by(desc(AlertHistory.triggered_at)).limit(limit)
    )
    history = result.scalars().all()
    return {"history": [
        {
            "id": h.id,
            "alert_id": h.alert_id,
            "triggered_at": h.triggered_at.isoformat(),
            "message": h.message,
            "acknowledged": h.acknowledged,
        }
        for h in history
    ]}
