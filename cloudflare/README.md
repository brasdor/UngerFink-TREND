# Always-on `/status` — Cloudflare Worker setup

This makes `/status` answer 24/7 with **no PC and no cost** — a free Cloudflare
Worker receives Telegram messages and reads the latest data committed in your
GitHub repo. ~10 minutes, all in the browser (no command line).

> The serverless `/status` shows open positions, last run, and the auto-exec
> ledger. It does **not** show the exchange balance — API keys must never live in
> a public worker. Use the local bot or the Trade Desk for balance.

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
