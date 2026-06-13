#!/usr/bin/env python3
"""
Data Migration Script — Import existing CSV data into PostgreSQL
================================================================

Imports:
- paper_trend_t9a: closed trades, equity snapshots, signals, positions
- t9b_paper: Donchian universe V2 paper trades
- t9b_mr_paper: Mean Reversion paper trades
- t9b_consecdowndays_paper: ConsecDownDays paper trades
- Research run metadata (scanned from data/ directories)

Usage:
    # Ensure PostgreSQL is running (docker-compose up db)
    python migrate_data.py

    # Dry-run (print what would be imported):
    python migrate_data.py --dry-run

Requirements:
    pip install pandas sqlalchemy asyncpg asyncio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.models import Base
from app.models.trade import Trade
from app.models.position import Position
from app.models.equity_snapshot import EquitySnapshot
from app.models.signal import Signal
from app.models.strategy import Strategy
from app.models.research_run import ResearchRun

settings = get_settings()

DATA_DIR = Path(settings.data_dir)
PROJECT_ROOT = Path(settings.project_root)


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

STRATEGIES = [
    {
        "name": "trend_6h_donchian",
        "display_name": "Trend Following 6H Donchian",
        "strategy_type": "trend",
        "timeframe": "6h",
        "exchange": "binance_spot",
        "risk_per_trade": 0.0025,
        "max_positions": 5,
        "max_heat_pct": 1.5,
        "kill_switch_dd_pct": 35.0,
        "is_active": True,
        "is_paper_live": True,
        "frozen_config": {
            "entry": "donchian_20_breakout",
            "filter": "ema50_slope_10bar",
            "initial_stop": "atr14_x2",
            "chandelier_activation": "+4R_MFE",
            "chandelier_atr_mult": 4.0,
            "chandelier_lookback": 22,
            "universe_size": 70,
        },
        "description": "Frozen T8 config. Donchian 20 breakout with EMA50 slope filter, ATR×2 stop, Chandelier +4R trailing.",
    },
    {
        "name": "donchian_1d_universev2",
        "display_name": "Donchian 1D Universe V2",
        "strategy_type": "trend",
        "timeframe": "1d",
        "exchange": "binance_spot",
        "risk_per_trade": 0.0025,
        "max_positions": 3,
        "max_heat_pct": 0.75,
        "kill_switch_dd_pct": 35.0,
        "is_active": True,
        "is_paper_live": True,
        "frozen_config": {
            "entry": "donchian_20_breakout",
            "filter": "ema200_price",
            "initial_stop": "atr14_x2",
            "chandelier_activation": "+4R_MFE",
            "chandelier_atr_mult": 3.0,
        },
        "description": "T9B Donchian 1D paper engine. EMA200 price filter.",
    },
    {
        "name": "meanreversion_rsi_1d",
        "display_name": "Mean Reversion RSI 1D",
        "strategy_type": "mean_reversion",
        "timeframe": "1d",
        "exchange": "binance_spot",
        "risk_per_trade": 0.0025,
        "max_positions": 3,
        "max_heat_pct": 0.75,
        "kill_switch_dd_pct": 35.0,
        "is_active": True,
        "is_paper_live": True,
        "frozen_config": {
            "entry": "rsi_oversold",
            "filter": "ema200_above",
        },
        "description": "T9B Mean Reversion RSI paper engine.",
    },
    {
        "name": "consecdowndays_mr_1d",
        "display_name": "Consecutive Down Days MR 1D",
        "strategy_type": "mean_reversion",
        "timeframe": "1d",
        "exchange": "binance_spot",
        "risk_per_trade": 0.0025,
        "max_positions": 3,
        "max_heat_pct": 0.75,
        "kill_switch_dd_pct": 35.0,
        "is_active": True,
        "is_paper_live": True,
        "frozen_config": {
            "entry": "consecutive_down_days",
            "min_down_days": 3,
        },
        "description": "T9B Consecutive Down Days mean reversion paper engine.",
    },
]


# ============================================================
# IMPORT FUNCTIONS
# ============================================================


def parse_datetime(val) -> Optional[datetime]:
    """Safely parse datetime from various formats."""
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        dt = pd.to_datetime(val, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


async def import_strategies(session: AsyncSession, dry_run: bool):
    """Import strategy definitions."""
    print("\n📋 Importing strategies...")
    for s in STRATEGIES:
        if dry_run:
            print(f"  [DRY] Would insert strategy: {s['name']}")
            continue
        strategy = Strategy(**s)
        session.add(strategy)
    if not dry_run:
        await session.flush()
    print(f"  ✓ {len(STRATEGIES)} strategies")


async def import_t9a_trades(session: AsyncSession, dry_run: bool):
    """Import closed trades from paper_trend_t9a."""
    csv_path = DATA_DIR / "paper_trend_t9a" / "closed_trades_trend_t9a.csv"
    if not csv_path.exists():
        print(f"  ⚠ Skipping T9A trades — file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n📈 Importing T9A closed trades ({len(df)} rows)...")

    count = 0
    for _, row in df.iterrows():
        trade = Trade(
            strategy="trend_6h_donchian",
            symbol=str(row.get("symbol", "")),
            side=str(row.get("side", "LONG")),
            timeframe=str(row.get("timeframe", "6h")),
            entry_time=parse_datetime(row.get("entry_time")),
            exit_time=parse_datetime(row.get("exit_time")),
            entry_price=float(row.get("entry_price", 0)),
            exit_price=float(row.get("exit_price", 0)),
            initial_stop=float(row.get("initial_stop", 0)),
            initial_risk=abs(float(row.get("entry_price", 0)) - float(row.get("initial_stop", 0))),
            pnl_r=float(row.get("net_r", row.get("gross_r", 0))),
            pnl_usdt=float(row.get("pnl_usdt", 0)) if pd.notna(row.get("pnl_usdt")) else None,
            mfe_r=float(row.get("max_favorable_r", 0)) if pd.notna(row.get("max_favorable_r")) else None,
            exit_reason=str(row.get("exit_reason", "")) if pd.notna(row.get("exit_reason")) else None,
            chandelier_activated=bool(row.get("chandelier_active", False)),
            chandelier_activation_time=parse_datetime(row.get("chandelier_activation_time")),
            position_size_usdt=float(row.get("notional_usdt", 0)) if pd.notna(row.get("notional_usdt")) else None,
            source="paper",
        )
        if dry_run:
            count += 1
            continue
        session.add(trade)
        count += 1

    if not dry_run:
        await session.flush()
    print(f"  ✓ {count} trades imported")


async def import_t9a_equity(session: AsyncSession, dry_run: bool):
    """Import equity snapshots from paper_trend_t9a."""
    csv_path = DATA_DIR / "paper_trend_t9a" / "equity_trend_t9a.csv"
    if not csv_path.exists():
        print(f"  ⚠ Skipping T9A equity — file not found")
        return

    df = pd.read_csv(csv_path)
    print(f"\n💰 Importing T9A equity snapshots ({len(df)} rows)...")

    count = 0
    for _, row in df.iterrows():
        snapshot = EquitySnapshot(
            strategy="trend_6h_donchian",
            timestamp=parse_datetime(row.get("timestamp_utc")),
            equity_usdt=float(row.get("closed_equity_usdt", 10000)),
            open_pnl_usdt=0.0,
            closed_pnl_usdt=float(row.get("closed_equity_usdt", 10000)) - 10000,
            drawdown_pct=float(row.get("drawdown_pct", 0)),
            peak_equity=float(row.get("peak_equity_usdt", 10000)),
            open_positions=int(row.get("open_positions", 0)),
            portfolio_heat_pct=float(row.get("portfolio_heat_pct", 0)),
        )
        if dry_run:
            count += 1
            continue
        session.add(snapshot)
        count += 1

    if not dry_run:
        await session.flush()
    print(f"  ✓ {count} equity snapshots imported")


async def import_t9a_signals(session: AsyncSession, dry_run: bool):
    """Import signals from paper_trend_t9a."""
    csv_path = DATA_DIR / "paper_trend_t9a" / "signals_trend_t9a.csv"
    if not csv_path.exists():
        print(f"  ⚠ Skipping T9A signals — file not found")
        return

    df = pd.read_csv(csv_path)
    print(f"\n🔔 Importing T9A signals ({len(df)} rows)...")

    count = 0
    for _, row in df.iterrows():
        signal = Signal(
            strategy="trend_6h_donchian",
            symbol=str(row.get("symbol", "")),
            side=str(row.get("side", "LONG")),
            timeframe=str(row.get("timeframe", "6h")),
            signal_time=parse_datetime(row.get("timestamp_utc", row.get("bar_time"))),
            signal_price=float(row.get("entry_price", 0)),
            status="taken",
        )
        if dry_run:
            count += 1
            continue
        session.add(signal)
        count += 1

    # Also import skipped signals
    skipped_path = DATA_DIR / "paper_trend_t9a" / "skipped_signals_trend_t9a.csv"
    if skipped_path.exists():
        df_skip = pd.read_csv(skipped_path)
        for _, row in df_skip.iterrows():
            signal = Signal(
                strategy="trend_6h_donchian",
                symbol=str(row.get("symbol", "")),
                side=str(row.get("side", "LONG")),
                timeframe=str(row.get("timeframe", "6h")),
                signal_time=parse_datetime(row.get("timestamp_utc", row.get("bar_time"))),
                signal_price=float(row.get("entry_price", 0)) if pd.notna(row.get("entry_price")) else 0,
                status="skipped",
                skip_reason=str(row.get("reason", "")) if pd.notna(row.get("reason")) else None,
            )
            if not dry_run:
                session.add(signal)
            count += 1

    if not dry_run:
        await session.flush()
    print(f"  ✓ {count} signals imported")


async def import_t9a_positions(session: AsyncSession, dry_run: bool):
    """Import current open positions from paper_trend_t9a."""
    csv_path = DATA_DIR / "paper_trend_t9a" / "open_positions_trend_t9a.csv"
    if not csv_path.exists():
        print(f"  ⚠ Skipping T9A positions — file not found")
        return

    df = pd.read_csv(csv_path)
    print(f"\n📊 Importing T9A open positions ({len(df)} rows)...")

    count = 0
    for _, row in df.iterrows():
        entry_price = float(row.get("entry_price", 0))
        initial_stop = float(row.get("initial_stop", 0))
        position = Position(
            strategy="trend_6h_donchian",
            symbol=str(row.get("symbol", "")),
            side=str(row.get("side", "LONG")),
            timeframe=str(row.get("timeframe", "6h")),
            entry_time=parse_datetime(row.get("entry_time")),
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=float(row.get("current_stop", initial_stop)),
            initial_risk=abs(entry_price - initial_stop),
            current_price=float(row.get("current_price")) if pd.notna(row.get("current_price")) else None,
            current_r=float(row.get("current_r")) if pd.notna(row.get("current_r")) else None,
            mfe_r=float(row.get("max_favorable_r", 0)) if pd.notna(row.get("max_favorable_r")) else 0,
            chandelier_active=bool(row.get("chandelier_active", False)),
            is_active=True,
        )
        if dry_run:
            count += 1
            continue
        session.add(position)
        count += 1

    if not dry_run:
        await session.flush()
    print(f"  ✓ {count} open positions imported")


async def import_research_metadata(session: AsyncSession, dry_run: bool):
    """Scan data/ directories and create research run entries."""
    print("\n🔬 Scanning research directories...")

    research_dirs = sorted(DATA_DIR.glob("research_*"))
    count = 0

    for rd in research_dirs:
        # Parse strategy and phase from directory name
        # Format: research_{strategy}_{phase} or research_{strategy}_{extra}_{phase}
        name = rd.name.replace("research_", "")

        # Known strategies
        strategy_map = {
            "trend": "trend_6h_donchian",
            "dualma": "dualma",
            "meanreversionrsi": "meanreversion_rsi_1d",
            "consecdowndays_mr": "consecdowndays_mr_1d",
            "keltnerlong": "keltnerlong",
            "linregbreakout": "linregbreakout",
            "bollingerbandmr": "bollingerbandmr",
            "rsimrshort": "rsimrshort",
            "donchianlong_universev2": "donchian_1d_universev2",
        }

        strategy = None
        phase = None

        for key, strat_name in strategy_map.items():
            if name.startswith(key):
                strategy = strat_name
                remainder = name[len(key):].lstrip("_")
                # Extract phase (t1, t2, etc.)
                if remainder:
                    phase_part = remainder.upper()
                    if phase_part.startswith("T"):
                        phase = phase_part.split("_")[0]
                break

        if not strategy or not phase:
            continue

        run = ResearchRun(
            strategy=strategy,
            phase=phase,
            phase_name=f"Phase {phase}",
            status="completed",
            output_dir=str(rd),
        )

        # Try to read summary stats from CSVs in the directory
        summary_csvs = list(rd.glob("*summary*"))
        if summary_csvs:
            try:
                sdf = pd.read_csv(summary_csvs[0])
                if "total_R" in sdf.columns:
                    run.total_r = float(sdf["total_R"].sum())
                if "win_rate" in sdf.columns:
                    run.win_rate = float(sdf["win_rate"].mean())
                if "avg_r" in sdf.columns:
                    run.avg_r = float(sdf["avg_r"].mean())
                if "n_trades" in sdf.columns:
                    run.total_trades = int(sdf["n_trades"].sum())
            except Exception:
                pass

        if dry_run:
            print(f"  [DRY] {strategy} / {phase} — {rd.name}")
            count += 1
            continue

        session.add(run)
        count += 1

    if not dry_run:
        await session.flush()
    print(f"  ✓ {count} research runs cataloged")


# ============================================================
# MAIN
# ============================================================


async def create_tables(engine):
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables created")


async def main(dry_run: bool = False):
    print("=" * 60)
    print("UngerFink-TREND — Data Migration")
    print("=" * 60)
    print(f"Database: {settings.database_url}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Dry run:  {dry_run}")
    print("=" * 60)

    engine = create_async_engine(settings.database_url, echo=False)

    if not dry_run:
        await create_tables(engine)

    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_factory() as session:
        try:
            await import_strategies(session, dry_run)
            await import_t9a_trades(session, dry_run)
            await import_t9a_equity(session, dry_run)
            await import_t9a_signals(session, dry_run)
            await import_t9a_positions(session, dry_run)
            await import_research_metadata(session, dry_run)

            if not dry_run:
                await session.commit()
                print("\n✅ Migration complete! All data committed.")
            else:
                print("\n🔍 Dry run complete — no data written.")

        except Exception as e:
            if not dry_run:
                await session.rollback()
            print(f"\n❌ Migration failed: {e}")
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate CSV data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be imported without writing")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
