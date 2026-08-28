# UngerFink_TREND — Architecture & Project Overview

> **Purpose:** Trend-following (and mean-reversion) research, paper-live simulation, and semi/auto execution system for Binance Spot USDT pairs. Built in the "Unger-style" discipline: no-curve-fitting, offline research first, frozen config, paper observation before any real capital.

> **⚠️ Status note (2026-06):** This document's §1–§13 describe the original **T9A 6H trend** research pipeline and remain accurate as history. The system has since grown: scripts were reorganized into folders, a **T9B daily multi-strategy paper system** runs in the cloud, a **FastAPI + Next.js web app** was added, plus **Telegram alerts** and **testnet-first live execution** (manual + automatic). All of that is documented in **§14 (Current Production Layers)** — read that for the present-day picture. Real money remains off; everything live runs on Binance **testnet**.

> **Repository layout (post-reorg):** runnable scripts now live in folders, not the repo root:
> `research/` (T1–T18 backtest pipeline) · `engines/` (live T9 paper engines, `t9b_shared`, summaries, `pipeline_agent`) · `dashboards/` (Streamlit monitors) · `tools/` (data fetch/migrate). Always run them **from the repo root**, e.g. `python engines/phase_t9b_donchian_universev2_paper_engine.py`. The `backend/` and `frontend/` folders hold the web app.

---

## Table of Contents

1. [Project Mission](#1-project-mission)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Research Pipeline — Phase by Phase](#3-research-pipeline--phase-by-phase)
4. [Core Trading Logic](#4-core-trading-logic)
5. [Portfolio & Risk Model](#5-portfolio--risk-model)
6. [Paper-Live Engine (T9A)](#6-paper-live-engine-t9a)
7. [Recovery Engine (T9D)](#7-recovery-engine-t9d)
8. [Monitoring Dashboards](#8-monitoring-dashboards)
9. [Automation](#9-automation)
10. [Data Directory Structure](#10-data-directory-structure)
11. [Dependencies](#11-dependencies)
12. [Key Design Decisions & Invariants](#12-key-design-decisions--invariants)
13. [Research Status & Warnings](#13-research-status--warnings)
14. [Current Production Layers (2026 update)](#14-current-production-layers-2026-update)

---

## 1. Project Mission

Build a systematic, rules-based trend-following system for Binance Spot USDT pairs and bring it to paper-live observation before considering real capital deployment.

The methodology follows a strict discipline:
- **No live optimisation** — parameters are frozen after research and never tuned mid-observation.
- **Closed candles only** — no intrabar execution assumptions.
- **R-multiple accounting** — all P&L measured in initial risk units (R), not percentage or USDT.
- **Kill-switches** — automatic halt triggers at defined drawdown levels.
- **Research pipeline** — each phase builds on, but does not modify, the previous phase's output.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  RESEARCH PIPELINE                       │
│                                                          │
│  T1 ──► T2 ──► T3 ──► T3B ──► T4 ──► T5 ──► T6 ──► T7 │
│                                 │                        │
│                      T10 ──► T11 ──► T12 ──► T12B       │
│                       │                                  │
│                      T13 ──► T14                         │
└────────────────────────────────┬────────────────────────┘
                                 │ frozen config (T8)
                                 ▼
┌─────────────────────────────────────────────────────────┐
│                  PAPER-LIVE LAYER                        │
│                                                          │
│  run_trend_t9a_loop.ps1  (every 15 min)                 │
│        │                                                 │
│        ▼                                                 │
│  phase_t9a_binance_paper_sim_engine_V2.py               │
│        │  ← Binance OHLCV via ccxt (no API keys)        │
│        │  → data/paper_trend_t9a/*.csv + state.json     │
│        │                                                 │
│  phase_t9d_trend_recovery_engine.py  (on-demand)        │
│        │  replays missed candles after downtime          │
└────────────────────────────────┬────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────┐
│               MONITORING DASHBOARDS                      │
│                                                          │
│  dashboard_trend_t9c_V3.py  (Streamlit)                 │
│  dashboard_trend_candles_entry_trailing_4h.py           │
│  dashboard_trend_t9_RECOVERY_SYNC_FIXED.py              │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Research Pipeline — Phase by Phase

### T1 — Clean Trend Concept Discovery (`phase_t1_trend_discovery.py`)

**Goal:** Discover a raw, unoptimised trend edge from scratch.

- Fetches top 50 liquid USDT spot symbols from Binance by quote volume.
- Tests 2H and 4H timeframes.
- Exhaustive parameter grid:
  - Donchian entry windows: [20, 40, 55]
  - Donchian exit windows: [10, 20]
  - ATR multipliers: [2.0, 3.0]
  - EMA lengths: [50]
  - EMA slope lookbacks: [5, 10]
- Side modes: LONG, SHORT, BOTH tested separately.
- Outputs: `data/research_trend_t1/phase_t1_trend_discovery_trades.csv` + summary.

**Key output:** Identifies the concept `donchian_ema_clean` as the foundational edge.

---

### T2 — Core Trend Engine (`phase_t2_core_trend_engine.py`)

**Goal:** Freeze the best T1 concept into a clean, readable engine.

- Expands universe to 70 assets.
- Frozen default: Donchian 20/10 · ATR×2 · EMA50 · slope lookback 10 · LONG only.
- Adds MAE/MFE tracking (Max Adverse/Favorable Excursion in R) per trade.
- Adds per-asset summaries and equity curve output.
- Output: 4 CSVs (trades, equity, summary, asset\_summary).

---

### T3 — Exit Engineering (`phase_t3_exit_engineering.py`)

**Goal:** Add Unger-style exit engineering on top of the T2 core.

- Breakeven trigger at **+1R** MFE — stop moves to entry.
- Chandelier trailing activated at **+2R** MFE — stop = N-bar high/low ± ATR×3.
- Adds stop timeline CSV for exit-path analysis.
- Conclusion: early breakeven proved too aggressive; led to T3B.

---

### T3B — Wide Exit Engineering (`phase_t3b_wide_exit_engineering.py`)

**Goal:** Revise T3 with wider trailing. The key pivot in the research.

- **No breakeven** (removed).
- Chandelier activates later at **+4R** MFE.
- Wider ATR multiplier for the trailing stop.
- Expands timeframes to include **6H** in addition to 2H and 4H.
- Output is the canonical trade list used by T4, T5, T6.

---

### T4 — Trend Robustness Engine (`phase_t4_trend_robustness_engine.py`)

**Goal:** Try to break the system statistically.

Takes T3B trade list as input — no signal changes. Tests:
- Baseline stats by timeframe and side.
- **Block bootstrap Monte Carlo** (2,000 runs, block sizes 1/3/5/10/20 bars, seed=42).
- Extra execution cost stress: additional 0R to 0.20R per trade.
- Rolling calendar windows (180-day window, 90-day step).
- Period splits: first half / second half / last 100 trades.
- Remove top N assets (1, 3, 5, 10).
- Remove top N months (1, 2, 3).
- Asset concentration diagnostics.

---

### T5 — Portfolio Replay Engine (`phase_t5_trend_portfolio_replay_engine.py`)

**Goal:** Apply realistic portfolio constraints to the T3B trade list.

Takes T3B trades and enforces:
- Max open positions (tested: 3, 5, 8).
- One position per asset at a time.
- Same-side exposure caps.
- Signal queue when entries arrive simultaneously.
- Portfolio heat tracking (sum of concurrent open risks).
- Logs skipped signals separately.

Tests multiple portfolio variants including pure-LONG, pure-SHORT, and mixed for 6H.

---

### T6 — Capital & Execution Engine V2 (`phase_t6_capital_execution_engine_V2.py`)

**Goal:** Translate R-multiple results into realistic USDT capital simulation.

- Starting capital: **$10,000 USDT**.
- Fixed fractional risk: **0.25% per trade**.
- Max portfolio heat: **1.5%** (sum of open initial risks / equity).
- Leverage proxy: 3× for notional calculation.
- Max notional per trade: 35% of equity.
- Kill-switch: stop new entries at **-35% closed-equity drawdown**.
- Equity floor: stop if equity falls below **50%** of initial capital.
- Variant-by-variant evaluation (inherits T5 variant names).

---

### T7 — Strict Robustness Engine (`phase_t7_strict_robustness_engine.py`)

**Goal:** Final attempt to break the best T6 capital variants.

Tests on best variants (`T5_6H_ALL_max5`, `T5_6H_ALL_max3`, `T5_6H_SHORT_max5`):
- Extra cost levels: 0%, 5%, 10%, 15% additional R per trade.
- Remove top 1/3/5 assets.
- Remove top 1/2 months.
- Long-only vs short-only splits.
- Recent-trade degradation analysis.
- Rolling window of 30 trades.

---

### T8 — Paper Live Blueprint (`phase_t8_paper_live_blueprint.py`)

**Goal:** Freeze and document the production configuration. Gate to paper-live.

Generates JSON artifacts:
- `phase_t8_frozen_config.json` — immutable parameters for T9.
- `phase_t8_live_state.json` — initial runtime state template.
- `phase_t8_system_health.json` — system health checklist.
- `phase_t8_paper_live_checklist.txt` — operator checklist.
- `phase_t8_closed_equity_template.csv` — equity tracking template.

**Frozen configuration:**

| Parameter | Value |
|-----------|-------|
| Exchange | Binance Spot |
| Timeframe | 6H |
| Universe | 70 assets |
| Entry | Donchian 20 breakout |
| Trend filter | EMA50 slope > 0 (10-bar lookback) |
| Initial stop | 2 × ATR14 |
| Chandelier activation | +4R MFE |
| Chandelier ATR mult | 4× |
| Chandelier lookback | 22 bars |
| Max open positions | 5 |
| Risk per trade | 0.25% |
| Max portfolio heat | 1.5% |
| Kill-switch DD | 35% |
| Equity floor | 50% |
| Leverage proxy | 3× |

---

### T10 — Trailing Robustness Study (`phase_t10_trailing_robustness_study.py`)

**Goal:** Offline sensitivity analysis of Chandelier parameters.

Tests all combinations of:
- Activation levels: +2R, +3R, +4R, +5R
- ATR multipliers: 3×, 4×, 5×

Uses 70 assets × 6H × 1,000 bars. Results feed T11, T12, T13, T14.

---

### T11 — Portfolio Heat & Capacity Study (`phase_t11_portfolio_heat_capacity_study.py`)

**Goal:** Find optimal portfolio capacity parameters.

Tests T10 variants (`ACT4_ATR3`, `ACT3_ATR3`, `ACT5_ATR3`) across a grid:
- Max open: 5, 8, 10, 12
- Risk per trade: 0.25%, 0.35%, 0.50%
- Max heat: 1.5%, 2.0%, 3.0%

---

### T12 — Cluster & Correlation Exposure Engine (`phase_t12_cluster_correlation_exposure_engine.py`)

**Goal:** Detect crowding and correlation risk across the portfolio.

Classifies all 70 assets into clusters:

| Cluster | Examples |
|---------|---------|
| BTC_BETA | BTC, ETH, BNB, SOL, XRP |
| L1_L2 | ETH, SOL, ADA, AVAX, DOT, NEAR |
| MEME | DOGE, SHIB, PEPE, FLOKI, BONK |
| AI | FET, RNDR, TAO, WLD, GRT |
| DEFI | UNI, AAVE, CRV, CAKE, LDO |
| GAMING | AXS, SAND, MANA, GALA, IMX |
| EXCHANGE | BNB, OKB, CRO |
| RWA | ONDO, OM, PENDLE |
| PRIVACY | ZEC, DASH, XMR |
| ALT_OTHER | everything else |

Also groups into beta buckets: `HIGH_CRYPTO_BETA` vs `OTHER_BETA`.

Tests max-per-cluster and max-per-beta constraints.

---

### T12B — Simple Portfolio Filter Validation (`phase_t12b_simple_portfolio_filter_validation.py`)

**Goal:** Validate practical portfolio filters after T12 revealed crowding.

Tests 6 filter scenarios combining same-side cap, per-cluster cap, and beta cap.
No entry/exit changes. Pure portfolio constraint validation.

---

### T13 — Timeframe Archetype Validation (`phase_t13_timeframe_archetype_validation.py`)

**Goal:** Compare the frozen trend archetype across multiple timeframes.

Tests identical frozen logic (Donchian 20, EMA50, ATR×2, Chandelier ACT4 ATR3) on:
- 4H (1,500 bars)
- 6H (1,200 bars)
- 8H (1,000 bars)
- 1D (800 bars)

---

### T14 — Realistic 6H Portfolio Replay (`phase_t14_realistic_6h_portfolio_replay.py`)

**Goal:** Final offline portfolio replay using validated 6H trades.

Preferentially uses T13 6H trades, falls back to T10 `ACT4_ATR3`.

Tests 3 same-side cap scenarios:
- `BASE_MAX5_HEAT15` — no same-side cap
- `SAME_SIDE_CAP4` — max 4 per side
- `SAME_SIDE_CAP3` — max 3 per side

---

## 4. Core Trading Logic

The same signal logic is frozen across all research phases and in production (T9A).

### Entry Signal

```
LONG entry (closed candle):
  close  > Donchian High(20)  [shifted: based on prior 20 candles, not including current]
  EMA50 slope > 0             [EMA50[now] - EMA50[10 bars ago] > 0]

SHORT entry (research only — not executable on Binance Spot):
  close  < Donchian Low(20)
  EMA50 slope < 0
```

> Shift by 1 bar on Donchian levels is critical to avoid lookahead bias.

### Stop Loss

```
LONG:  initial_stop = entry_price - (ATR14 × 2.0)
SHORT: initial_stop = entry_price + (ATR14 × 2.0)
```

### Trailing Stop (Chandelier)

Activates only after **+4R MFE** (Max Favorable Excursion):

```
LONG:  chandelier_stop = max(22-bar high, shifted) - ATR14 × 4.0
SHORT: chandelier_stop = min(22-bar low, shifted) + ATR14 × 4.0

Stop only moves in the favorable direction (ratchets, never retreats).
```

### Exit Conditions

| Condition | Exit Price |
|-----------|-----------|
| Low (LONG) or High (SHORT) touches stop | Stop price |
| End of data (research only) | Last close |

### R-Multiple Calculation

```
LONG:  pnl_R = (exit_price - entry_price) / initial_risk
SHORT: pnl_R = (entry_price - exit_price) / initial_risk
```

---

## 5. Portfolio & Risk Model

```
Position size (units) = (equity × RISK_PCT) / initial_risk_per_unit

Where:
  equity        = current closed-trade equity in USDT
  RISK_PCT      = 0.0025 (0.25%)
  initial_risk  = entry_price - initial_stop (LONG)
```

**Portfolio constraints enforced at entry time:**

| Constraint | Value |
|------------|-------|
| Max open positions | 5 |
| Max positions per symbol | 1 |
| Max portfolio heat | 1.5% of equity |
| Max notional per trade | 35% of equity |
| Max total margin usage | 85% of equity |
| Leverage proxy | 3× |

**Kill-switch conditions (halt new entries):**

| Trigger | Level |
|---------|-------|
| Closed-equity drawdown | −35% |
| Equity floor | 50% of initial capital |

---

## 6. Paper-Live Engine (T9A)

**File:** `phase_t9a_binance_paper_sim_engine_V2.py`

### How It Works

1. **Called by** `run_trend_t9a_loop.ps1` every 15 minutes.
2. **Downloads** closed 6H OHLCV candles from Binance via `ccxt` (no API keys needed — public endpoint).
3. **Detects** new closed candles since last run by comparing timestamps.
4. **Only acts** when a new closed candle is confirmed — no intrabar logic.
5. **Manages** open positions: updates Chandelier trailing stop, checks exit conditions.
6. **Scans** all 70 symbols for new entry signals.
7. **Persists** full state to `data/paper_trend_t9a/trend_t9a_state.json`.
8. **Logs** to CSV files (signals, open positions, closed trades, equity, skipped signals).

### State Files

| File | Content |
|------|---------|
| `trend_t9a_state.json` | Full system state: open positions, equity, kill-switch flag |
| `open_positions_trend_t9a.csv` | Current open paper positions |
| `closed_trades_trend_t9a.csv` | All closed paper trades |
| `equity_trend_t9a.csv` | Closed-trade equity curve |
| `signals_trend_t9a.csv` | All entry signals detected |
| `skipped_signals_trend_t9a.csv` | Signals skipped due to portfolio constraints |
| `system_health_trend_t9a.json` | Runtime health metrics |
| `ohlcv_cache/` | Local OHLCV cache per symbol (avoids redundant API calls) |

### Universe Selection

On each run, the engine ranks all active USDT spot symbols by 24H quote volume and selects the top 70. Excluded:

- Stablecoins and fiat proxies (USDC, BUSD, TUSD, PAXG, EUR, etc.)
- Leveraged tokens (UP, DOWN, BULL, BEAR, 3L, 3S suffix patterns)
- Perpetual futures (`":" in symbol`)
- WBTC, BTTC

---

## 7. Recovery Engine (T9D)

**File:** `phase_t9d_trend_recovery_engine.py`

Used after crashes, internet outages, or planned downtime.

Reads open positions from `trend_t9a_state.json` and replays all closed 6H candles from each position's `last_update_time` forward. Updates trailing stop state, closes any positions that would have triggered, and syncs all dashboard CSV files.

Output:
- `phase_t9d_recovery_events.csv` — each candle processed per position.
- `phase_t9d_recovery_report.csv` — summary of what was recovered.

---

## 8. Monitoring Dashboards

All dashboards are **read-only**. No orders are placed.

### `dashboard_trend_t9c_V3.py` (Primary)

The main Streamlit dashboard for T9A paper-live monitoring.

**Sections:**
- System health JSON display (kill-switch status, equity, drawdown).
- **Open positions** — table + individual Plotly candlestick chart per position showing:
  - Last N candles (6H, 140 bars default)
  - Entry candle marker
  - Initial stop (horizontal line)
  - Current trailing stop (horizontal line)
  - Live price (horizontal line)
  - Chandelier activation marker (if activated)
- Closed trades table with P&L statistics.
- Equity curve chart (closed-trade equity only).
- Signals log.
- Skipped signals log.

Run: `streamlit run dashboard_trend_t9c_V3.py`

### `dashboard_trend_candles_entry_trailing_4h.py`

Earlier version targeting 4H candles. Looks for data in multiple fallback directories (`data/live_trend`, `data/paper_live_trend`, etc.). Compatible with older simulation CSVs.

### `dashboard_trend_t9_RECOVERY_SYNC.py` / `dashboard_trend_t9_RECOVERY_SYNC_FIXED.py`

Recovery-specific dashboards for reviewing state reconciliation.

---

## 9. Automation

**File:** `run_trend_t9a_loop.ps1`

```powershell
while ($true) {
    python engines/phase_t9a_binance_paper_sim_engine_V2.py
    Start-Sleep -Seconds 900   # 15 minutes
}
```

The engine is idempotent — if no new closed 6H candle has appeared since the last run, it exits immediately without modifying state.

---

## 10. Data Directory Structure

```
data/
├── raw_trend_t1/                  # Raw OHLCV cache from T1 (2H/4H)
├── raw_trend_t2/                  # Raw OHLCV cache from T2 (2H/4H)
├── research_trend_t1/             # T1 trades + summary CSVs
├── research_trend_t2/             # T2 trades + equity + summary CSVs
├── research_trend_t3/             # T3 trades + stop timeline CSVs
├── research_trend_t3b/            # T3B trades (canonical input for T4/T5/T6)
├── research_trend_t4/             # T4 robustness reports + Monte Carlo
├── research_trend_t5/             # T5 portfolio replay (multiple variants)
├── research_trend_t6/             # T6 capital simulation (trade log + equity)
├── research_trend_t7/             # T7 stress tests + master summary
├── research_trend_t8/             # T8 frozen config JSON + health template
├── research_trend_t10/            # T10 trailing sensitivity results
├── research_trend_t11/            # T11 portfolio heat/capacity grid
├── research_trend_t12/            # T12 cluster/correlation analysis
├── research_trend_t12b/           # T12B portfolio filter validation
├── research_trend_t13/            # T13 timeframe archetype comparison
├── research_trend_t14/            # T14 realistic 6H portfolio replay
└── paper_trend_t9a/               # LIVE paper-sim state and logs
    ├── trend_t9a_state.json
    ├── open_positions_trend_t9a.csv
    ├── closed_trades_trend_t9a.csv
    ├── equity_trend_t9a.csv
    ├── signals_trend_t9a.csv
    ├── skipped_signals_trend_t9a.csv
    ├── system_health_trend_t9a.json
    └── ohlcv_cache/               # Per-symbol 6H OHLCV cache
```

---

## 11. Dependencies

```
pip install ccxt pandas numpy streamlit plotly
```

| Library | Usage |
|---------|-------|
| `ccxt` | Binance OHLCV data (public endpoint, no API keys) |
| `pandas` | DataFrame manipulation, CSV I/O |
| `numpy` | Numerical computation, ATR, rolling stats |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive candlestick charts in dashboard |

The research/engine layer installs from the repo-root `requirements.txt`. The web app has its own `backend/requirements.txt` adding `fastapi`, `uvicorn`, `sqlalchemy[asyncio]`, `aiosqlite`/`asyncpg`, `alembic`, `redis`, `httpx`, and uses `ccxt` for **order placement** (not just data) — see §14. The frontend is `Next.js` + `React` + `Tailwind` (`frontend/package.json`).

**Python version:** 3.8+ for research/engines; the web app targets 3.11.

---

## 12. Key Design Decisions & Invariants

| Decision | Rationale |
|----------|-----------|
| Closed candles only — always drop last forming candle | Prevents lookahead bias; any live execution is on next open |
| Donchian levels shifted by 1 bar (`shift(1)`) | Entry cannot be based on the candle that is simultaneously closing |
| R-multiples instead of USDT P&L | Normalises across assets of different price scales |
| ATR-based sizing | Volatility-adjusted position sizing; risk is equal across all trades |
| No breakeven move (T3B lesson) | Early breakeven was statistically detrimental — it cut winners too early |
| Chandelier activates late (+4R) | Lets winners run; Unger-style "let the trend breathe" |
| Chandelier stop only ratchets | Never retreats against the trade — only tightens |
| Fixed fractional risk (0.25%) | Compound growth, position size adjusts with equity |
| Portfolio heat cap (1.5%) | Limits simultaneous catastrophic drawdown from correlated positions |
| Short research-only / long preferred live | Binance Spot does not support direct shorting |
| State persisted to JSON | Survives engine restarts and system crashes |
| T9D recovery engine exists | Explicit design for downtime resilience |
| Research/paper layer uses no API keys | All market data from the public Binance OHLCV endpoint |
| Execution layer keys are testnet + gated | Order placement (§14) needs keys, but defaults to Binance **testnet** with a `LIVE_TRADING_ENABLED` master switch off; real keys are never committed (`backend/.env`, GitHub secrets) |

---

## 13. Research Status & Warnings

As of the T8 freeze (current paper-live phase):

**Confirmed strengths:**
- Cost robustness survived: edge persists under realistic extra costs.
- Low drawdown under capital constraints.
- 6H timeframe structure appears robust.
- Portfolio heat discipline preserved.
- Long-only structure particularly strong.

**Active warnings (from T8 system health):**
- Dependency on best assets still present — removing top 5 assets meaningfully impacts results.
- Recent last-20-trade window showed some degradation.
- Sample size still limited — paper-live observation of at least 100–200 more trades required before considering real deployment.
- Short-side edge is theoretically present but not executable on Binance Spot.

**Rule:**
> No new filters. No optimisation. **No real capital yet.**
> Paper observation (and **testnet** execution dry-running, §14) is the only authorised activity at this stage. The documented real-money review date is **2026-08-28**.

---

## 14. Current Production Layers (2026 update)

§1–§13 describe the original T9A 6H trend pipeline. This section documents what the repository actually runs today. Everything live is on Binance **testnet** (fake money); no real capital is deployed.

### 14.1 Repository layout (after the script reorg)

The ~106 formerly-root scripts now live in folders (run all **from the repo root**):

| Folder | Contents |
|--------|----------|
| `research/` | T1–T18 backtest pipeline + studies (`phase_t*`) |
| `engines/` | Live T9 paper engines (`phase_t9a/b/d_*`), `t9b_shared.py`, summaries (`t9b_combined_summary.py`), `pipeline_agent.py` |
| `dashboards/` | Streamlit monitors (`dashboard_trend_*`) |
| `tools/` | `fetch_2h_data.py`, `phase_download_extended_history.py`, `migrate_data.py`, `telegram_status_bot.py` |
| `backend/` | FastAPI web app (API + DB) |
| `frontend/` | Next.js web UI |

Scripts that derive the repo root from their own location use `Path(__file__).resolve().parents[1]`; the live engines use `Path.cwd()`. CI / `.bat` / `.ps1` / `pipeline_agent` invocation paths were all updated accordingly.

### 14.2 T9B — daily multi-strategy paper system (the live system)

Runs **unattended in GitHub Actions** (`.github/workflows/t9b_daily.yml`, daily at 08:00 UTC), not on a local PC. Three frozen strategies, each in its own engine and data dir:

| Strategy | Engine | Data dir |
|----------|--------|----------|
| Donchian breakout (UniverseV2, ExitV2) | `engines/phase_t9b_donchian_universev2_paper_engine.py` | `data/t9b_paper/` |
| MeanReversionRSI 1D | `engines/phase_t9b_meanreversion_paper_engine.py` | `data/t9b_mr_paper/` |
| ConsecDownDaysMR 1D | `engines/phase_t9b_consecdowndays_paper_engine.py` | `data/t9b_consecdowndays_paper/` |

Each run: downloads candles (public ccxt) → updates paper positions → writes `signals_today.csv`, `open_positions.csv`, `equity_curve.csv`, `daily_log.csv`, `state.json` → commits them back → opens a **GitHub Issue** per signal (`.github/scripts/create_signal_issues.py`) → sends a **Telegram** summary. Cross-strategy symbol dedup via `engines/t9b_shared.py`.

### 14.3 Web application (`backend/` + `frontend/`)

A separate layer over the file-based pipeline:
- **Backend** — FastAPI + async SQLAlchemy. Models: `Trade`, `Position`, `Signal`, `Candle`, `Strategy`, `Alert`, `EquitySnapshot`, `JournalEntry`, `ResearchRun`, **`Order`**. Routers under `/api/*`: portfolio, trades, signals, strategies, research, alerts, **execution**, websocket. SQLite for local dev (auto-creates tables); Postgres/TimescaleDB via `docker compose`. `tools/migrate_data.py` loads CSV data into the DB.
- **Frontend** — Next.js dashboard (Positions, Trades, Equity, Signals, Research, **Trade Desk**, …) reading the API.
- Run locally with `docker compose up` (API :8000, web :3000).

### 14.4 Telegram

- **Daily signal summary** — `.github/scripts/send_telegram.py`, a step in the daily workflow (after engines), via bot `@ungertrend_bot`. Uses GitHub secrets `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`. Sends even when there are no signals ("no new signals"). Failures never break the workflow.
- **Recipients** — `TELEGRAM_CHAT_ID` accepts a **comma-separated list** of chat ids; `send_telegram.py` and `check_workflow_failures.py` fan out to all of them, and one failed delivery never silences the rest. It held a single id until 2026-08-28, which is why only one operator ever received the push messages. Note this is independent of the Worker's `ALLOWED_CHAT` allowlist, which governs who may *send commands* — a person must be on both lists to both receive alerts and use `/status`. A chat missing from `ALLOWED_CHAT` is ignored silently, which is indistinguishable from a dead bot. See `cloudflare/README.md` → "Adding a second person".
- **`/status` command** — `tools/telegram_status_bot.py`, a local long-polling listener (run via `run_status_bot.bat`). Answers `/status` with open positions per strategy, last run, auto-exec ledger, and testnet balance. Read-only; only responds while running (serverless polling would exceed free Actions minutes). Backend `app/services/telegram.py` is the reusable async notifier.
- **Always-on commands** — `cloudflare/status_worker.js`, a free Cloudflare Worker on a Telegram webhook, is the always-on replacement for the local poller. It reads committed CSVs / `data/status_snapshot.json` through the GitHub Contents API, so it holds no exchange keys and can never place an order. Commands: `/status`, `/positions`, `/pnl`, `/price [SYM…]`, `/24h [SYM…]`, `/systems`, and one per system (`/s1`…`/s8`, `/donchian`, `/rsimr`, `/macross`, `/c12`, `/c19`, plus aliases). `/price` and `/24h` with no argument cover **every coin currently held** across all 9 systems; they read Binance's public `/ticker/24hr` (spot, then futures for futures-only symbols) and need no new secrets. Replies over Telegram's 4096-char limit are split automatically. Full command table in `cloudflare/README.md`.

### 14.5 Live execution (testnet-first)

Two paths, both **spot, long-only**, both defaulting to **testnet** with real money gated off:

**Manual ("Trade Desk")** — `backend/app/routers/execution.py` + `backend/app/services/execution.py` (ccxt) + frontend `/execution` page. `POST /api/execution/place` runs the safety chain: live-switch gate → idempotency → risk-based sizing (lot/min-notional rounding) → per-order cap → daily-loss kill-switch → place → reconcile. Every order is an `Order` row (intended vs actual fill). A **dry-run** toggle simulates without sending.

**Automatic** — `.github/scripts/auto_execute.py`, a daily-workflow step. After the engines run it mirrors the paper engine: market **BUY** for each accepted entry (event=ENTRY), market **SELL** for any position we hold that the engine has closed (detected by the symbol leaving `open_positions.csv`). Idempotent via an append-only BUY/SELL ledger `data/auto_orders/placed.csv` (net position = BUYs − SELLs). Telegram-confirms each fill.

**Safety posture (all deliberate, currently set to testnet):**

| Gate | Default | Where |
|------|---------|-------|
| `EXCHANGE_TESTNET` | `true` (fake money) | `.env` / repo var |
| `LIVE_TRADING_ENABLED` (manual) | `false` | `.env` |
| `AUTO_EXECUTE_ENABLED` (auto) | off unless set | repo var |
| `MAX_ORDER_USDT` cap | 100 | config / env |
| `daily_loss_limit_usdt` | 200 | config |
| API keys | testnet, never committed | `backend/.env`, GitHub secrets |

**Going to real money** is a deliberate two-switch change (real keys **and** `EXCHANGE_TESTNET=false`) the operator makes consciously — not before the 2026-08-28 review. See `LIVE_EXECUTION_AND_TELEGRAM_PLAN.md` for the full plan.
