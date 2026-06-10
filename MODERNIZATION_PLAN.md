# UngerFink-TREND — Full Web App Modernization Plan

## Problem Statement

The UngerFink-TREND project is a sophisticated systematic trading research pipeline (T1→T18) with paper-live simulation engines, but the UI layer is fragmented across 3 Streamlit scripts and batch files. There's no unified interface to monitor all strategies, explore research results, track trade journal entries, or receive real-time alerts.

## Proposed Approach

Build a **full-stack web application** with:
- **Backend:** Python FastAPI wrapping existing research engines + new API layer
- **Frontend:** React/Next.js with real-time WebSocket updates
- **Database:** PostgreSQL (with TimescaleDB for time-series data)
- **Deployment:** Docker Compose (self-hosted) + cloud option (Vercel + Railway/Supabase)

The existing Python engines (T1–T18, T9A/B paper engines) remain untouched — the API layer adapts around them.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NEXT.JS FRONTEND                         │
│                                                              │
│  Dashboard │ Research Explorer │ Trade Journal │ Alerts      │
│  Portfolio │ Strategy Monitor  │ Backtest Viz  │ Settings    │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────┴──────────────────────────────────┐
│                     FASTAPI BACKEND                           │
│                                                              │
│  /api/portfolio     — live positions, equity, heat           │
│  /api/trades        — closed trades, journal entries         │
│  /api/signals       — pending/skipped signals                │
│  /api/research      — browse research phase results          │
│  /api/strategies    — strategy configs, status, health       │
│  /api/backtest      — run/view backtests on demand           │
│  /api/alerts        — alert rules, notification history      │
│  /ws/live           — WebSocket for real-time updates        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  SERVICES LAYER                                      │    │
│  │  - DataIngestionService (ccxt, candle polling)       │    │
│  │  - SignalEngine (wraps T9A/T9B logic)                │    │
│  │  - PortfolioService (positions, risk, heat)          │    │
│  │  - ResearchService (reads research CSVs)             │    │
│  │  - AlertService (rules engine, notifications)        │    │
│  │  - BacktestService (on-demand replay)                │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                     POSTGRESQL + TIMESCALEDB                  │
│                                                              │
│  trades (hypertable)     — all closed trades with R, MAE/MFE│
│  positions               — current open positions            │
│  equity_snapshots        — time-series equity curve          │
│  signals                 — all signals generated             │
│  candles (hypertable)    — OHLCV cache                       │
│  strategies              — frozen configs per strategy       │
│  research_runs           — metadata for each phase run       │
│  alerts                  — alert rules and history           │
│  journal_entries         — manual trade notes/annotations    │
└─────────────────────────────────────────────────────────────┘
```

---

## Todos

### Phase 1: Foundation & Infrastructure

- **pg-schema** — Design and implement PostgreSQL/TimescaleDB schema (trades, positions, equity, candles, strategies, alerts, journal)
- **fastapi-scaffold** — Scaffold FastAPI project with proper structure (routers, services, models, config, CORS, auth)
- **data-migration** — Write migration scripts to import existing CSV data (paper_trend_t9a, t9b_*) into PostgreSQL
- **nextjs-scaffold** — Scaffold Next.js 14 app with App Router, Tailwind CSS, shadcn/ui components, dark mode
- **docker-compose** — Docker Compose setup (PostgreSQL + TimescaleDB, FastAPI, Next.js, Redis for pub/sub)

### Phase 2: Core Backend API

- **portfolio-api** — Portfolio endpoints: GET open positions, equity curve, portfolio heat, drawdown stats
- **trades-api** — Trades endpoints: list/filter/search closed trades, trade detail with MAE/MFE chart
- **signals-api** — Signals endpoints: pending signals, skipped signals with reasons, signal history
- **strategies-api** — Strategy CRUD: list strategies, view frozen config, health status, enable/disable
- **data-ingestion** — Background candle ingestion service (replace PS1 loop with async Python scheduler)
- **websocket-live** — WebSocket server for real-time position updates, new signals, price ticks

### Phase 3: Frontend — Dashboard & Portfolio

- **dashboard-page** — Main dashboard: portfolio summary cards (equity, open P&L, heat %, drawdown), mini equity chart
- **positions-page** — Open positions table with candlestick mini-charts, entry/stop/trailing markers, live P&L
- **equity-page** — Full equity curve with drawdown overlay, underwater chart, rolling Sharpe
- **trades-page** — Trades table with advanced filtering (by strategy, symbol, date range, R-multiple range)
- **trade-detail** — Individual trade view: candlestick chart with entry/exit markers, stop progression, R-curve

### Phase 4: Frontend — Research Explorer

- **research-browser** — Browse all research phases (T1–T18) with results summary cards
- **research-detail** — Phase detail: parameter grids, heatmaps, equity curves, gate check results
- **strategy-comparison** — Side-by-side strategy comparison (Donchian vs DualMA vs MeanRevRSI etc.)
- **backtest-runner** — On-demand backtest UI: select strategy/params/universe/timeframe, run, view results

### Phase 5: Frontend — Trade Journal & Alerts

- **journal-page** — Trade journal: annotate trades with notes, screenshots, lessons learned, tags
- **alerts-config** — Alert rule builder: drawdown threshold, equity new high, position hit trailing, heat warning
- **notifications** — Notification system: browser push, optional Discord/Telegram webhook integration
- **kill-switch-ui** — Manual kill-switch panel with confirmation, auto-halt status display

### Phase 6: Real-Time Engine

- **scheduler-service** — Replace PowerShell loops with APScheduler/Celery: run T9A every 15min, T9B daily
- **candle-watcher** — Efficient candle close detection (poll Binance, emit events on new closed candles)
- **event-bus** — Internal event bus (Redis pub/sub): new_candle → signal_check → position_update → UI push
- **health-monitor** — System health dashboard: last run times, error counts, API connectivity, data freshness

### Phase 7: Deployment

- **docker-prod** — Production Docker Compose with Traefik reverse proxy, SSL, health checks
- **cloud-deploy** — Cloud deployment guide: Vercel (frontend) + Railway (FastAPI + PostgreSQL) or Supabase
- **backup-strategy** — Automated PostgreSQL backups, trade data export to CSV/JSON
- **monitoring** — Application monitoring: uptime checks, error alerting, resource usage

### Phase 8: Polish & Advanced Features

- **mobile-responsive** — Fully responsive design for mobile portfolio monitoring
- **performance-analytics** — Advanced analytics: Monte Carlo equity projections, rolling stats, correlation analysis
- **multi-account** — Support multiple paper accounts / strategy variants running in parallel
- **api-docs** — Auto-generated API documentation (FastAPI OpenAPI + Swagger UI)

---

## Key Design Decisions

1. **Existing engines stay untouched** — FastAPI wraps them, reads their output files, and optionally calls them via subprocess
2. **Incremental migration** — Phase 1 imports existing CSV data; engines can still write CSVs as before while the new DB layer catches up
3. **Real-time is additive** — The 15-min polling loop still works; WebSocket just pushes state changes to connected clients
4. **No auth initially** — Single-user local deployment; auth can be added later for cloud deployment
5. **TimescaleDB for time-series** — Equity snapshots and OHLCV candles benefit from hypertable compression and time-based queries

---

## Deployment Options

### Option A: Self-Hosted (Docker Compose)

```yaml
# docker-compose.yml
services:
  db:
    image: timescale/timescaledb:latest-pg16
    volumes: [pgdata:/var/lib/postgresql/data]
    ports: ["5432:5432"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  api:
    build: ./backend
    depends_on: [db, redis]
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]  # mount existing data dir
  
  web:
    build: ./frontend
    depends_on: [api]
    ports: ["3000:3000"]
  
  scheduler:
    build: ./backend
    command: python -m app.scheduler
    depends_on: [db, redis, api]
    volumes: ["./data:/app/data"]
```

### Option B: Cloud Deployment

| Component | Service | Notes |
|-----------|---------|-------|
| Frontend | Vercel | Free tier, auto-deploy from GitHub |
| Backend API | Railway | $5/mo, auto-scale, GitHub deploy |
| Database | Railway PostgreSQL or Supabase | TimescaleDB on Railway |
| Redis | Railway or Upstash | For pub/sub + caching |
| Scheduler | Railway worker | Separate process for candle polling |

---

## File Structure (New)

```
P:\MCH\UngerFink-TREND\
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── database.py          # SQLAlchemy + async engine
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── routers/             # API route handlers
│   │   │   ├── portfolio.py
│   │   │   ├── trades.py
│   │   │   ├── signals.py
│   │   │   ├── strategies.py
│   │   │   ├── research.py
│   │   │   ├── alerts.py
│   │   │   └── websocket.py
│   │   ├── services/            # Business logic
│   │   │   ├── data_ingestion.py
│   │   │   ├── signal_engine.py
│   │   │   ├── portfolio.py
│   │   │   ├── research.py
│   │   │   ├── alerts.py
│   │   │   └── backtest.py
│   │   ├── scheduler.py         # APScheduler jobs
│   │   └── events.py            # Redis pub/sub event bus
│   ├── migrations/              # Alembic migrations
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js App Router pages
│   │   │   ├── page.tsx         # Dashboard
│   │   │   ├── positions/
│   │   │   ├── trades/
│   │   │   ├── research/
│   │   │   ├── journal/
│   │   │   ├── alerts/
│   │   │   └── settings/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── charts/          # TradingView lightweight-charts wrappers
│   │   │   ├── tables/          # Data tables with sorting/filtering
│   │   │   └── ui/              # shadcn/ui components
│   │   ├── hooks/               # Custom React hooks (useWebSocket, usePortfolio)
│   │   ├── lib/                 # API client, utils
│   │   └── stores/              # Zustand state management
│   ├── Dockerfile
│   ├── package.json
│   └── tailwind.config.ts
├── docker-compose.yml
├── docker-compose.prod.yml
│
│  # Existing files (unchanged)
├── phase_t1_*.py ... phase_t18_*.py
├── phase_t9a_*.py, phase_t9b_*.py
├── data/
├── ARCHITECTURE.md
└── requirements.txt
```

---

## Tech Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | Next.js 14 + React 18 | App Router, SSR, great DX |
| UI Components | shadcn/ui + Tailwind CSS | Professional look, dark mode, accessible |
| Charts | TradingView Lightweight Charts + Recharts | Professional trading charts + analytics |
| State | Zustand + React Query (TanStack) | Simple state + smart server-state caching |
| Backend | FastAPI (Python 3.11+) | Async, fast, auto-docs, same language as engines |
| ORM | SQLAlchemy 2.0 (async) | Mature, TimescaleDB support |
| Migrations | Alembic | Standard for SQLAlchemy |
| Database | PostgreSQL 16 + TimescaleDB | Time-series optimized, production-grade |
| Cache/PubSub | Redis 7 | Real-time event distribution |
| Scheduler | APScheduler | Replace PS1/BAT loops, in-process |
| Containerization | Docker Compose | Reproducible dev + prod environments |
| Auth (later) | NextAuth.js + FastAPI JWT | When needed for cloud deployment |

---

## Notes & Considerations

- **Data continuity:** The migration script must preserve all existing paper-trading history (equity curves, trades, signals) — this is irreplaceable observation data.
- **Backward compatibility:** Keep the existing CSV-writing behavior in T9A/T9B engines as a fallback. The new system reads from DB but engines can still write CSVs.
- **Research phases are offline:** T1–T18 scripts are run manually or via pipeline_agent. The web app just browses their output — it doesn't re-run them (except the optional backtest runner in Phase 4).
- **Kill-switch integrity:** The kill-switch logic must remain in the engine itself (not just UI). The UI provides visibility and manual override, but automated halts stay in Python.
- **No real money:** This system is paper-only. If real execution is ever added, it would be a separate, heavily-audited module.
