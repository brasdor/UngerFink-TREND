# Always-on `/status` — Cloudflare Worker setup

This makes `/status` answer 24/7 with **no PC and no cost** — a free Cloudflare
Worker receives Telegram messages and reads the latest data committed in your
GitHub repo. ~10 minutes, all in the browser (no command line).

> The serverless `/status` shows all 9 systems (S1-S8 + both T9 candidates):
> equity, open positions, today's realized/unrealized P&L, kill-switch state,
> the current regime and each system's capital weight under it, and an alert
> summary (missed runs, stale OHLCV/funding data). `/positions` and `/pnl`
> remain the narrower per-position views. None of these show the exchange
> balance — API keys must never live in a public worker. Use the local bot or
> the Trade Desk for balance.

### Commands

| Command | Shows |
|---------|-------|
| `/status` | all 9 systems, regime, weights, alerts — one view |
| `/positions` | every open position with P&L |
| `/pnl` | total P&L summary |
| `/price` | USD price + 24h move of **every coin currently held** |
| `/price BTC ETH` | USD price of the coins you name |
| `/24h` | 24h move of every coin held, **biggest mover first**, with day low/high |
| `/24h SOL` | 24h move of the coins you name |
| `/systems` | list the per-system commands |
| `/s1` … `/s8`, `/donchian`, `/rsimr`, `/consecdown`, `/momentum`, `/volcontraction`, `/macross`, `/c12`, `/c19` | one system in detail: equity, weight, kill-switch, open positions with P&L, today's signals |

Symbols are forgiving: `btc`, `BTC`, `BTC_USDT` and `BTCUSDT` all resolve to
`BTCUSDT`. Prices come from Binance `/ticker/24hr` — spot first, then futures
for the futures-only symbols in the book (`BTCDOMUSDT`, `1000PEPEUSDT`, …).
A symbol neither venue knows shows `n/a` rather than failing the command.

`/price` and `/24h` need **no new secrets** — Binance's ticker endpoint is
public and unauthenticated. Replies longer than Telegram's 4096-character
limit are split across messages automatically.

---

## Step 1 — Create a GitHub read token (so the worker can read your private repo)

1. GitHub → **Settings → Developer settings → Fine-grained tokens → Generate new token**.
2. **Repository access** → Only select repositories → `UngerFink-TREND`.
3. **Permissions** → Repository permissions → **Contents: Read-only**.
4. Generate, copy the token (starts with `github_pat_…`). Keep it for Step 3.

## Step 2 — Create the Worker

1. Sign up free at **dash.cloudflare.com** → **Workers & Pages** → **Create** → **Create Worker**.
2. Name it e.g. `ungerfink-status` → **Deploy** (the placeholder), then **Edit code**.
3. Delete the sample, paste the entire contents of **`cloudflare/status_worker.js`**, and **Deploy**.
4. Copy the Worker URL (e.g. `https://ungerfink-status.YOURNAME.workers.dev`).

## Step 3 — Add the settings (Worker → Settings → Variables and Secrets)

Add these. Mark the three secrets as **Encrypt**:

| Name | Value | Type |
|------|-------|------|
| `BOT_TOKEN` | your Telegram bot token | secret |
| `GH_TOKEN` | the token from Step 1 | secret |
| `WEBHOOK_SECRET` | any random string you make up | secret |
| `GH_REPO` | `brasdor/UngerFink-TREND` | text |
| `GH_BRANCH` | `master` | text |
| `ALLOWED_CHAT` | `7804609823` (your chat id; or `*` for anyone) | text |

**Deploy** again so the variables take effect.

## Step 4 — Point Telegram at the Worker

Open this URL in a browser (replace both placeholders):

```
https://api.telegram.org/bot<BOT_TOKEN>/setWebhook?url=<WORKER_URL>&secret_token=<WEBHOOK_SECRET>
```

You should see `{"ok":true,"result":true,...}`.
*(Or send me the Worker URL + the secret and I'll set the webhook for you.)*

## Step 5 — Test

Send **`/status`** to @ungertrend_bot. You get an instant reply — and it keeps
working forever, no PC required.

---

## Notes

- **Webhook vs local poller:** setting a webhook **disables** `getUpdates`, so do
  **not** also run `tools/telegram_status_bot.py` afterwards — the Worker replaces it.
  (The daily `send_telegram.py` only *sends*, so it's unaffected.)
- **To undo:** `https://api.telegram.org/bot<BOT_TOKEN>/deleteWebhook` re-enables the
  local poller.
- **Multiple users:** set `ALLOWED_CHAT` to a comma-separated list of chat ids, or
  `*` for anyone who messages the bot (remember `/status` reveals positions).

## Adding a second person (e.g. a co-operator)

There are **two separate channels**, and being added to one does not add you to
the other. This is why a second person can be unable to see anything while the
first sees everything:

| Channel | Direction | Controlled by | Where to change it |
|---------|-----------|---------------|--------------------|
| Daily summary + failure alerts | bot **pushes** to you | `TELEGRAM_CHAT_ID` secret | GitHub → Settings → Secrets → Actions |
| `/status`, `/price`, … | you **pull** by sending a command | `ALLOWED_CHAT` variable | Cloudflare Worker → Settings → Variables |

To add someone to both:

1. **Get their chat id.** They must message @ungertrend_bot at least once
   (a bot cannot message a user first). Then open
   `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates` and read
   `message.chat.id` for their message. A chat id cannot be derived from a
   username — this step is unavoidable.
2. **Push:** set the `TELEGRAM_CHAT_ID` secret to a comma-separated list,
   e.g. `7804609823,123456789`. `send_telegram.py` and
   `check_workflow_failures.py` fan out to every id, and one bad id no longer
   silences the others.
3. **Pull:** add the same id to the Worker's `ALLOWED_CHAT` variable, then
   redeploy the Worker. A chat that is not listed is **silently ignored** —
   no error is sent, which looks exactly like a dead bot.

**Alternative — a group chat.** Put both people in one Telegram group, add the
bot, and use the group's chat id (negative, e.g. `-1001234567890`) as the single
value for both settings. One id to maintain, and new people are added by
inviting them to the group rather than by editing configuration.
