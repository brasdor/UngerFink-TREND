from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.signal import Signal

router = APIRouter()


@router.get("/")
async def list_signals(
    strategy: Optional[str] = None,
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List signals with filtering."""
    stmt = select(Signal).order_by(desc(Signal.signal_time))

    if strategy:
        stmt = stmt.where(Signal.strategy == strategy)
    if status:
        stmt = stmt.where(Signal.status == status)
    if symbol:
        stmt = stmt.where(Signal.symbol == symbol)

    result = await db.execute(stmt.offset(offset).limit(limit))
    signals = result.scalars().all()

    return {
        "signals": [
            {
                "id": s.id,
                "strategy": s.strategy,
                "symbol": s.symbol,
                "side": s.side,
                "timeframe": s.timeframe,
                "signal_time": s.signal_time.isoformat(),
                "signal_price": s.signal_price,
                "status": s.status,
                "skip_reason": s.skip_reason,
            }
            for s in signals
        ]
    }


@router.get("/pending")
async def get_pending_signals(
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get only pending (unacted) signals."""
    stmt = select(Signal).where(Signal.status == "pending").order_by(desc(Signal.signal_time))
    if strategy:
        stmt = stmt.where(Signal.strategy == strategy)
    result = await db.execute(stmt)
    signals = result.scalars().all()
    return {"count": len(signals), "signals": [
        {"id": s.id, "symbol": s.symbol, "side": s.side, "signal_time": s.signal_time.isoformat(), "signal_price": s.signal_price}
        for s in signals
    ]}
