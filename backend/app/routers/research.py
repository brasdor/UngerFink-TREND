from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.research_run import ResearchRun

router = APIRouter()


@router.get("/")
async def list_research_runs(
    strategy: Optional[str] = None,
    phase: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all research phase runs."""
    stmt = select(ResearchRun).order_by(ResearchRun.strategy, ResearchRun.phase)
    if strategy:
        stmt = stmt.where(ResearchRun.strategy == strategy)
    if phase:
        stmt = stmt.where(ResearchRun.phase == phase.upper())

    result = await db.execute(stmt)
    runs = result.scalars().all()
    return {"runs": [_run_to_dict(r) for r in runs]}


@router.get("/{run_id}")
async def get_research_run(run_id: int, db: AsyncSession = Depends(get_db)):
    """Get details of a specific research run."""
    from fastapi import HTTPException
    result = await db.execute(select(ResearchRun).where(ResearchRun.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Research run not found")
    return _run_to_dict(run, full=True)


def _run_to_dict(r: ResearchRun, full: bool = False) -> dict:
    d = {
        "id": r.id,
        "strategy": r.strategy,
        "phase": r.phase,
        "phase_name": r.phase_name,
        "status": r.status,
        "gate_result": r.gate_result,
        "total_trades": r.total_trades,
        "total_r": r.total_r,
        "win_rate": r.win_rate,
        "avg_r": r.avg_r,
    }
    if full:
        d.update({
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "gate_details": r.gate_details,
            "output_dir": r.output_dir,
            "summary_stats": r.summary_stats,
            "notes": r.notes,
        })
    return d
