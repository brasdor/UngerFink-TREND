from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.trade import Trade

router = APIRouter()


@router.get("/")
async def list_trades(
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    min_r: Optional[float] = None,
    max_r: Optional[float] = None,
    source: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List closed trades with filtering."""
    stmt = select(Trade).order_by(desc(Trade.exit_time))

    if strategy:
        stmt = stmt.where(Trade.strategy == strategy)
    if symbol:
        stmt = stmt.where(Trade.symbol == symbol)
    if side:
        stmt = stmt.where(Trade.side == side.upper())
    if min_r is not None:
        stmt = stmt.where(Trade.pnl_r >= min_r)
    if max_r is not None:
        stmt = stmt.where(Trade.pnl_r <= max_r)
    if source:
        stmt = stmt.where(Trade.source == source)

    # Count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar()

    # Paginate
    result = await db.execute(stmt.offset(offset).limit(limit))
    trades = result.scalars().all()

    return {
        "total": total,
        "trades": [_trade_to_dict(t) for t in trades],
    }


@router.get("/stats")
async def trade_stats(
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Aggregate trade statistics."""
    stmt = select(Trade)
    if strategy:
        stmt = stmt.where(Trade.strategy == strategy)
    result = await db.execute(stmt)
    trades = result.scalars().all()

    if not trades:
        return {"total_trades": 0}

    pnl_rs = [t.pnl_r for t in trades]
    wins = [r for r in pnl_rs if r > 0]
    losses = [r for r in pnl_rs if r <= 0]

    return {
        "total_trades": len(trades),
        "total_r": round(sum(pnl_rs), 2),
        "avg_r": round(sum(pnl_rs) / len(pnl_rs), 3),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_win_r": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss_r": round(sum(losses) / len(losses), 2) if losses else 0,
        "best_trade_r": round(max(pnl_rs), 2),
        "worst_trade_r": round(min(pnl_rs), 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else None,
    }


@router.get("/{trade_id}")
async def get_trade(trade_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single trade by ID."""
    result = await db.execute(select(Trade).where(Trade.id == trade_id))
    trade = result.scalar_one_or_none()
    if not trade:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Trade not found")
    return _trade_to_dict(trade)


def _trade_to_dict(t: Trade) -> dict:
    return {
        "id": t.id,
        "strategy": t.strategy,
        "symbol": t.symbol,
        "side": t.side,
        "timeframe": t.timeframe,
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat(),
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "initial_stop": t.initial_stop,
        "pnl_r": t.pnl_r,
        "pnl_usdt": t.pnl_usdt,
        "mae_r": t.mae_r,
        "mfe_r": t.mfe_r,
        "exit_reason": t.exit_reason,
        "chandelier_activated": t.chandelier_activated,
        "source": t.source,
        "notes": t.notes,
    }
