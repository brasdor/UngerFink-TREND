# Live Execution + Telegram Alerts — Implementation Plan

Status: **planned, not yet built.** Locked decisions baked in. Real money stays
gated behind a master switch that defaults OFF and behind keys only the user can add.

---

## Locked decisions
| Decision | Choice |
|---|---|
| Market | **Binance Spot only** (long-only USDT) — matches the 3 live paper engines |
| Autonomy | **Button-click only** (human approves each order); full-auto is a later phase |
| Exchange library | **ccxt** (testnet via `set_sandbox_mode(True)`) |

---

# Part 1 — Semi-Automatic Live Trade Execution

## Goal
A **"Place trade"** button in the web dashboard that places the real Binance spot
order on click. Human approves, machine executes. Testnet-first, hard safety rails.

## Architecture (extends the existing FastAPI app)
| Piece | What it is |
|---|---|
| `backend/app/services/execution.py` | Thin **ccxt** wrapper: connect, fetch balance, place/cancel order, poll fill |
| `backend/app/services/telegram.py`  | Shared notifier (reused by Part 2) |
| `backend/app/routers/execution.py`  | `POST /execution/place`, `GET /execution/status/{id}`, `GET /execution/balance` |
| `backend/app/models/order.py` (new) | Live order record: intended vs actual fill, status, idempotency key |
| `backend/app/config.py` additions   | `exchange_api_key`, `exchange_secret`, `testnet`, `live_trading_enabled`, `max_order_usdt`, `daily_loss_limit_usdt` |
| Frontend button + confirm dialog    | On a signal/position card → dialog (size, entry, stop) → calls the endpoint |

## Build order (each step verified before the next)
- **Phase 0 — Connectivity:** ccxt + Binance **testnet** keys; fetch balance only, no orders.
- **Phase 1 — Execution service:** place market & limit orders on testnet; size from frozen
  risk %; round to Binance lot-size / min-notional.
- **Phase 2 — Endpoints + integrity:** routes, `Order` record, **idempotency** (no double-place),
  **reconciliation** (intended vs actual fill).
- **Phase 3 — Frontend:** button + confirmation dialog + visible **dry-run toggle**.
- **Phase 4 — Safety rails:** `LIVE_TRADING_ENABLED` master switch (default **OFF**),
  per-order max-size cap, daily-loss kill-switch.
- **Phase 5 — Full testnet validation:** place → fill → record → stop, per entry type. **← review checkpoint.**
- **Phase 6 — Go-live (user-gated):** swap testnet→real keys, tiny size, one symbol, scale gradually.

## Authorization / secrets (user's part)
- Binance **testnet** keys first → later real keys with **trade permission only,
  withdrawals disabled, IP-whitelisted**.
- Keys live in `backend/.env` (gitignored). **Never** in the repo.

## Non-negotiable safety rails
Master `LIVE_TRADING_ENABLED` defaults OFF · testnet flag routes to fake money ·
per-order notional cap · daily realized-loss kill-switch · idempotency per signal ·
UI confirmation dialog · fill reconciliation with mismatch alerts.

---

# Part 2 — Telegram Signal & Fill Alerts

## Goal
Push a phone notification the moment a signal fires (and later, when a live order
is placed/filled), without needing a PC or to open GitHub.

## Two integration points
1. **Cloud (daily engines, no PC):** a step in `.github/workflows/t9b_daily.yml`, after the
   engines run, reads each engine's `signals_today.csv` + combined summary and sends a
   Telegram message.
2. **Web app (execution):** `app/services/telegram.py` sends a message when a live order is
   placed or filled (shared with Part 1).

## Pieces
| Piece | What it is |
|---|---|
| `.github/scripts/send_telegram.py` | ~40 lines: read signals CSVs, format message, POST to Telegram Bot API |
| Workflow step in `t9b_daily.yml`   | Runs after the combined summary; `continue-on-error` so a Telegram outage never fails the run |
| `backend/app/services/telegram.py` | Reusable `send_message()` for the web app / execution |

## Setup (user's part)
1. Create a bot via **@BotFather** → copy the **bot token**.
2. Get your **chat ID** (message the bot, or use `@userinfobot`).
3. Add as **GitHub repo secrets**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   (and to `backend/.env` for the web-app side).

## Robustness
One message/day → no rate-limit concern. All sends wrapped so a Telegram failure
never breaks the trading workflow (same graceful pattern as `create_signal_issues.py`).

## Natural future synergy
Telegram **inline buttons** (Place / Skip) could become the mobile version of the
"approve & place" button — letting you approve a trade from your phone. Out of scope
for the first build, but the two features are designed to converge there.

---

## Example Telegram message
```
🟢 UngerFink T9B — 2026-06-11

Donchian:  FET/USDT  ENTRY @ 1.234  stop 1.101  (0.25% risk)
RSI-MR:    no signals
ConsecDn:  ADA/USDT  EXIT  @ 0.612  (+1.8R)

Open: 5 positions · Equity: 10,640 USDT (+6.4%)
```
