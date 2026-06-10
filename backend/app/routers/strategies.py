from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.strategy import Strategy

router = APIRouter()


@router.get("/")
async def list_strategies(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """List all strategies."""
    stmt = select(Strategy)
    if active_only:
        stmt = stmt.where(Strategy.is_active == True)
    result = await db.execute(stmt.order_by(Strategy.name))
    strategies = result.scalars().all()
    return {"strategies": [_strat_to_dict(s) for s in strategies]}


@router.get("/{strategy_name}")
async def get_strategy(strategy_name: str, db: AsyncSession = Depends(get_db)):
    """Get strategy details by name."""
    result = await db.execute(select(Strategy).where(Strategy.name == strategy_name))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return _strat_to_dict(strategy, full=True)


@router.patch("/{strategy_name}/toggle")
async def toggle_strategy(strategy_name: str, db: AsyncSession = Depends(get_db)):
    """Enable/disable a strategy."""
    result = await db.execute(select(Strategy).where(Strategy.name == strategy_name))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.is_active = not strategy.is_active
    await db.flush()
    return {"name": strategy.name, "is_active": strategy.is_active}


def _strat_to_dict(s: Strategy, full: bool = False) -> dict:
    d = {
        "name": s.name,
        "display_name": s.display_name,
        "strategy_type": s.strategy_type,
        "timeframe": s.timeframe,
        "is_active": s.is_active,
        "is_paper_live": s.is_paper_live,
        "max_positions": s.max_positions,
        "risk_per_trade": s.risk_per_trade,
    }
    if full:
        d.update({
            "frozen_config": s.frozen_config,
            "exchange": s.exchange,
            "max_heat_pct": s.max_heat_pct,
            "kill_switch_dd_pct": s.kill_switch_dd_pct,
            "paper_start_date": s.paper_start_date.isoformat() if s.paper_start_date else None,
            "description": s.description,
            "universe_symbols": s.universe_symbols,
        })
    return d
