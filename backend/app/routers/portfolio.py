from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.position import Position
from app.models.equity_snapshot import EquitySnapshot

router = APIRouter()


@router.get("/positions")
async def get_open_positions(
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get all currently open positions."""
    stmt = select(Position).where(Position.is_active == True)
    if strategy:
        stmt = stmt.where(Position.strategy == strategy)
    result = await db.execute(stmt)
    positions = result.scalars().all()
    return {"positions": [_pos_to_dict(p) for p in positions]}


@router.get("/summary")
async def get_portfolio_summary(
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Portfolio summary: equity, heat, drawdown, open positions count."""
    # Latest equity snapshot
    eq_stmt = select(EquitySnapshot).order_by(EquitySnapshot.timestamp.desc()).limit(1)
    if strategy:
        eq_stmt = eq_stmt.where(EquitySnapshot.strategy == strategy)
    eq_result = await db.execute(eq_stmt)
    equity = eq_result.scalar_one_or_none()

    # Open positions count
    pos_stmt = select(func.count()).select_from(Position).where(Position.is_active == True)
    if strategy:
        pos_stmt = pos_stmt.where(Position.strategy == strategy)
    pos_count = (await db.execute(pos_stmt)).scalar()

    if equity:
        return {
            "equity_usdt": equity.equity_usdt,
            "open_pnl_usdt": equity.open_pnl_usdt,
            "drawdown_pct": equity.drawdown_pct,
            "portfolio_heat_pct": equity.portfolio_heat_pct,
            "peak_equity": equity.peak_equity,
            "open_positions": pos_count,
            "last_update": equity.timestamp.isoformat(),
        }
    return {
        "equity_usdt": 10000.0,
        "open_pnl_usdt": 0.0,
        "drawdown_pct": 0.0,
        "portfolio_heat_pct": 0.0,
        "peak_equity": 10000.0,
        "open_positions": pos_count or 0,
        "last_update": None,
    }


@router.get("/equity-curve")
async def get_equity_curve(
    strategy: Optional[str] = None,
    days: int = Query(default=90, ge=1, le=730),
    db: AsyncSession = Depends(get_db),
):
    """Equity curve data for charting."""
    stmt = select(EquitySnapshot).order_by(EquitySnapshot.timestamp.asc())
    if strategy:
        stmt = stmt.where(EquitySnapshot.strategy == strategy)
    # Limit to last N days would use a date filter in production
    result = await db.execute(stmt.limit(5000))
    snapshots = result.scalars().all()
    return {
        "data": [
            {
                "timestamp": s.timestamp.isoformat(),
                "equity": s.equity_usdt,
                "drawdown_pct": s.drawdown_pct,
                "heat_pct": s.portfolio_heat_pct,
            }
            for s in snapshots
        ]
    }


def _pos_to_dict(p: Position) -> dict:
    return {
        "id": p.id,
        "strategy": p.strategy,
        "symbol": p.symbol,
        "side": p.side,
        "timeframe": p.timeframe,
        "entry_time": p.entry_time.isoformat(),
        "entry_price": p.entry_price,
        "initial_stop": p.initial_stop,
        "current_stop": p.current_stop,
        "current_price": p.current_price,
        "current_r": p.current_r,
        "mfe_r": p.mfe_r,
        "chandelier_active": p.chandelier_active,
        "position_size_usdt": p.position_size_usdt,
    }
