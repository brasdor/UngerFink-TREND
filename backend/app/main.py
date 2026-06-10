from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import portfolio, trades, signals, strategies, research, alerts, websocket

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: create tables if using SQLite (dev mode)
    from app.database import engine
    from app.models.base import Base
    from app.models import (
        Trade, Position, EquitySnapshot, Candle,
        Strategy, Signal, Alert, AlertHistory, JournalEntry, ResearchRun,
    )
    if "sqlite" in str(engine.url):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="UngerFink-TREND trading system API",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(portfolio.router, prefix=f"{settings.api_prefix}/portfolio", tags=["Portfolio"])
app.include_router(trades.router, prefix=f"{settings.api_prefix}/trades", tags=["Trades"])
app.include_router(signals.router, prefix=f"{settings.api_prefix}/signals", tags=["Signals"])
app.include_router(strategies.router, prefix=f"{settings.api_prefix}/strategies", tags=["Strategies"])
app.include_router(research.router, prefix=f"{settings.api_prefix}/research", tags=["Research"])
app.include_router(alerts.router, prefix=f"{settings.api_prefix}/alerts", tags=["Alerts"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
