"""Live (testnet-first) order execution endpoints.

Safety chain on every placement:
  live-switch gate -> idempotency -> sizing -> per-order cap ->
  daily-loss kill-switch -> place -> reconcile.

Real money is gated three ways that this code cannot self-grant: the API keys
(user adds to .env), `exchange_testnet` (defaults True), and
`live_trading_enabled` (defaults False).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_db
from app.models.order import Order
from app.models.trade import Trade
from app.services.execution import ExchangeClient, ExecutionError

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PlaceOrderRequest(BaseModel):
    strategy: str
    symbol: str = Field(..., examples=["FET/USDT"])
    side: str = "BUY"                       # spot long entries only for now
    order_type: str = "market"             # market | limit
    entry: float                           # reference/intended entry price
    stop: float                            # protective stop (for sizing)
    limit_price: Optional[float] = None    # required for limit orders
    equity_usdt: Optional[float] = None    # if omitted, free USDT balance is used
    risk_pct: Optional[float] = None       # if omitted, settings.default_risk_pct
    signal_id: Optional[int] = None
    idempotency_key: Optional[str] = None  # generated if omitted
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _order_to_dict(o: Order) -> dict:
    return {
        "id": o.id,
        "strategy": o.strategy,
        "symbol": o.symbol,
        "side": o.side,
        "order_type": o.order_type,
        "status": o.status,
        "testnet": o.testnet,
        "dry_run": o.dry_run,
        "requested_qty": o.requested_qty,
        "requested_price": o.requested_price,
        "requested_notional_usdt": o.requested_notional_usdt,
        "intended_stop": o.intended_stop,
        "filled_qty": o.filled_qty,
        "avg_fill_price": o.avg_fill_price,
        "fill_notional_usdt": o.fill_notional_usdt,
        "exchange_order_id": o.exchange_order_id,
        "reconciled": o.reconciled,
        "reconcile_note": o.reconcile_note,
        "error_message": o.error_message,
        "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
        "filled_at": o.filled_at.isoformat() if o.filled_at else None,
    }


async def _today_live_realized_pnl(db: AsyncSession) -> float:
    """Sum of realized PnL (USDT) from live trades closed today (UTC)."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = select(func.coalesce(func.sum(Trade.pnl_usdt), 0.0)).where(
        Trade.source == "live", Trade.exit_time >= start
    )
    return float((await db.execute(stmt)).scalar() or 0.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/config")
async def execution_config(settings: Settings = Depends(get_settings)):
    """Current safety posture — drives the UI's warnings/toggles."""
    return {
        "testnet": settings.exchange_testnet,
        "live_trading_enabled": settings.live_trading_enabled,
        "keys_configured": bool(settings.exchange_api_key and settings.exchange_secret),
        "default_risk_pct": settings.default_risk_pct,
        "max_order_usdt": settings.max_order_usdt,
        "daily_loss_limit_usdt": settings.daily_loss_limit_usdt,
    }


@router.get("/balance")
async def get_balance(settings: Settings = Depends(get_settings)):
    """Read-only connectivity check — proves the keys work without placing orders."""
    try:
        client = ExchangeClient(settings)
        balance = await asyncio.to_thread(client.fetch_balance)
        return {"testnet": settings.exchange_testnet, "free": balance}
    except ExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # exchange/network error
        raise HTTPException(status_code=502, detail=f"Exchange error: {e}")


@router.get("/orders")
async def list_orders(limit: int = 50, db: AsyncSession = Depends(get_db)):
    stmt = select(Order).order_by(Order.id.desc()).limit(limit)
    orders = (await db.execute(stmt)).scalars().all()
    return {"orders": [_order_to_dict(o) for o in orders]}


@router.post("/place")
async def place_order(
    body: PlaceOrderRequest,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
):
    # --- 0. only spot long entries for now --------------------------------
    if body.side.upper() != "BUY":
        raise HTTPException(status_code=400, detail="Only BUY (long entry) is supported yet.")

    # --- 1. live-switch gate ----------------------------------------------
    if not body.dry_run and not settings.live_trading_enabled:
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled. Set LIVE_TRADING_ENABLED=true or send dry_run=true.",
        )

    # --- 2. idempotency: never place the same key twice -------------------
    idem = body.idempotency_key or f"{body.strategy}:{body.symbol}:{uuid.uuid4().hex[:12]}"
    existing = (
        await db.execute(select(Order).where(Order.idempotency_key == idem))
    ).scalar_one_or_none()
    if existing is not None:
        return {"idempotent": True, "order": _order_to_dict(existing)}

    risk_pct = body.risk_pct if body.risk_pct is not None else settings.default_risk_pct

    # --- 3. sizing ---------------------------------------------------------
    try:
        client = ExchangeClient(settings)
        equity = body.equity_usdt
        if equity is None:
            balance = await asyncio.to_thread(client.fetch_balance)
            equity = float(balance.get("USDT", 0.0))
        if equity <= 0:
            raise ExecutionError("No equity available to size the order.")
        sized = await asyncio.to_thread(
            client.size_long, body.symbol, equity, risk_pct, body.entry, body.stop
        )
    except ExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Exchange error during sizing: {e}")

    # --- 4. per-order notional cap ----------------------------------------
    if sized.notional_usdt > settings.max_order_usdt:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Order notional {sized.notional_usdt:.2f} USDT exceeds cap "
                f"{settings.max_order_usdt} USDT."
            ),
        )

    # --- 5. daily-loss kill-switch ----------------------------------------
    if not body.dry_run:
        realized = await _today_live_realized_pnl(db)
        if realized <= -abs(settings.daily_loss_limit_usdt):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Daily loss limit hit (realized {realized:.2f} USDT). "
                    "No new live orders today."
                ),
            )

    # --- 6. persist the intended order (pending) --------------------------
    order = Order(
        strategy=body.strategy,
        symbol=body.symbol,
        side="BUY",
        signal_id=body.signal_id,
        idempotency_key=idem,
        order_type=body.order_type,
        requested_qty=sized.qty,
        requested_price=body.limit_price if body.order_type == "limit" else None,
        requested_notional_usdt=sized.notional_usdt,
        intended_stop=body.stop,
        risk_pct=risk_pct,
        equity_at_request=equity,
        testnet=settings.exchange_testnet,
        dry_run=body.dry_run,
        status="pending",
    )
    db.add(order)
    await db.flush()  # get order.id without ending the transaction

    # --- 7. dry-run short-circuit (no real order) -------------------------
    if body.dry_run:
        order.status = "filled"
        order.filled_qty = sized.qty
        order.avg_fill_price = sized.reference_price
        order.fill_notional_usdt = sized.notional_usdt
        order.filled_at = datetime.now(timezone.utc)
        order.reconciled = True
        order.reconcile_note = "dry run — simulated fill, no real order placed"
        await db.commit()
        return {"dry_run": True, "order": _order_to_dict(order)}

    # --- 8. place for real (testnet or live) ------------------------------
    try:
        order.submitted_at = datetime.now(timezone.utc)
        order.status = "submitted"
        result = await asyncio.to_thread(
            client.place_long, body.symbol, sized.qty, body.order_type, body.limit_price
        )
    except Exception as e:
        order.status = "error"
        order.error_message = str(e)
        await db.commit()
        raise HTTPException(status_code=502, detail=f"Order placement failed: {e}")

    # --- 9. record + reconcile fill ---------------------------------------
    order.exchange_order_id = str(result.get("id")) if result.get("id") else None
    filled = float(result.get("filled") or 0.0)
    avg = result.get("average") or result.get("price")
    order.filled_qty = filled
    order.avg_fill_price = float(avg) if avg else None
    if order.avg_fill_price and filled:
        order.fill_notional_usdt = filled * order.avg_fill_price
    fee = result.get("fee") or {}
    order.fee = fee.get("cost")
    order.fee_currency = fee.get("currency")

    ccxt_status = (result.get("status") or "").lower()
    if ccxt_status == "closed" or (filled and filled >= sized.qty * 0.999):
        order.status = "filled"
        order.filled_at = datetime.now(timezone.utc)
    elif filled > 0:
        order.status = "partially_filled"
    else:
        order.status = "submitted"  # e.g. resting limit order

    # reconciliation: did we get roughly what we asked for?
    if order.fill_notional_usdt:
        drift = abs(order.fill_notional_usdt - sized.notional_usdt) / sized.notional_usdt
        order.reconciled = drift <= 0.05
        order.reconcile_note = (
            "ok" if order.reconciled
            else f"fill notional drifted {drift:.1%} from requested"
        )
    else:
        order.reconciled = False
        order.reconcile_note = f"no fill yet (status={order.status})"

    await db.commit()
    return {"order": _order_to_dict(order)}
