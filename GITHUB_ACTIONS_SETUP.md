# T9B Paper Engine — GitHub Actions Setup Guide

This guide sets up automated daily paper trading that runs in the cloud
with no local PC required. Results accumulate in the repository and are
readable on any browser, including mobile.

**Time required: ~15 minutes**

---

## What it does

Every day at 08:00 UTC (after yesterday's 1D candle closed):
1. Downloads latest Binance OHLCV data (public API, no key needed)
2. Checks all 24 UniverseV2 symbols for entry/exit signals
3. Updates paper positions and equity
4. Commits results back to the repository
5. Opens a GitHub Issue if any new signals fired

You can check the status at any time by visiting your repository on GitHub.

---

## Step 1 — Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in (create a free account if needed)
2. Click the **+** icon (top right) → **New repository**
3. Settings:
   - **Repository name**: `UngerFink-TREND` (or any name you prefer)
   - **Visibility**: Private (recommended — your research stays private)
   - **Initialize**: Leave all checkboxes **unchecked** (we push existing code)
4. Click **Create repository**
5. GitHub shows a page with the remote URL. Copy it — it looks like:
   ```
   https://github.com/YOUR-USERNAME/UngerFink-TREND.git
   ```

---

## Step 2 — Push the Project to GitHub

Open a terminal (PowerShell or Git Bash) in `C:\Users\Jean\UngerFink-TREND`:

```powershell
# One-time setup: tell Git who you are (if not done already)
git config --global user.email "your@email.com"
git config --global user.name "Your Name"

# Connect to the remote repository
git remote add origin https://github.com/YOUR-USERNAME/UngerFink-TREND.git

# Check what will be pushed (large data dirs are in .gitignore)
git status

# Stage all files not excluded by .gitignore
git add .

# Create the first commit
git commit -m "Initial commit: T9B paper engine + research pipeline"

# Push to GitHub
git push -u origin master
```

**If the push asks for credentials:**
- Use your GitHub username
- For the password, use a **Personal Access Token** (PAT), not your account password:
  - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token → scope: **repo** (full control of private repos)
  - Copy the token and use it as the password

---

## Step 3 — Verify the Repository on GitHub

1. Visit `https://github.com/YOUR-USERNAME/UngerFink-TREND`
2. Confirm you can see:
   - `phase_t9b_donchian_universev2_paper_engine.py`
   - `requirements.txt`
   - `.github/workflows/t9b_daily.yml`
   - `data/t9b_paper/state.json`
   - `data/t9b_paper/open_positions.csv`

If these files are present, the push worked.

---

## Step 4 — Enable GitHub Actions

GitHub Actions should be enabled automatically for your repository.
To confirm:

1. In your repository, click the **Actions** tab (between Pull requests and Projects)
2. You should see **T9B Daily Paper Trading** listed as a workflow
3. If you see a yellow banner saying "Workflows aren't being run on this fork" — click
   **I understand my workflows, go ahead and enable them**

No secrets are needed. The `GITHUB_TOKEN` is automatically provided by GitHub Actions for:
- Committing output files back to the repository
- Creating issues for new signals

---

## Step 5 — Trigger a Manual Test Run

Before waiting for the scheduled run at 08:00 UTC, test it manually:

1. Go to the **Actions** tab in your repository
2. Click **T9B Daily Paper Trading** in the left sidebar
3. Click **Run workflow** (blue button, top right of the workflow list)
4. In the popup:
   - Leave **run_date** blank to use yesterday's date
   - Or enter a specific date: `2026-05-31`
5. Click **Run workflow**
6. A new run appears in the list — click it to watch the live log

The run takes approximately 2–3 minutes (most time is downloading OHLCV data).

---

## Step 6 — Checking Results from Your Phone

### View the latest open positions:
1. Visit `github.com/YOUR-USERNAME/UngerFink-TREND`
2. Navigate to `data/t9b_paper/open_positions.csv`
3. GitHub renders CSVs as a table — readable on mobile

### View the daily log (all events):
Navigate to `data/t9b_paper/daily_log.csv`

### View the equity curve:
Navigate to `data/t9b_paper/equity_curve.csv`

### View today's signals:
Navigate to `data/t9b_paper/signals_today.csv`

### View Issues (signals that fired):
1. Click the **Issues** tab in your repository
2. All signal issues are listed there with entry details
3. You will also receive GitHub email notifications for new issues
   (check Settings → Notifications to configure)

### View the Actions log (detailed run output):
1. Click **Actions** tab
2. Click the most recent run
3. Click **t9b-daily** job to expand the steps
4. Click any step to see the output, including the `--notify` summary

---

## Step 7 — Enable Email Notifications for Signals

1. Go to `github.com/settings/notifications`
2. Under **Participating, @mentions and custom routing**:
   - Enable **Email** for Issues
3. Now you receive an email whenever a new signal issue is created

This means: if FET/USDT breaks out, you get an email with the entry details
while you are on holiday, without needing to check the website.

---

## Workflow Schedule

The workflow runs at **08:00 UTC** every day:

| UTC    | CET/CEST (Central Europe) | Description                        |
|--------|---------------------------|------------------------------------|
| 08:00  | 09:00 / 10:00             | Workflow starts                    |
| ~08:03 | ~09:03 / ~10:03           | Results committed, issues created  |

To change the schedule, edit `.github/workflows/t9b_daily.yml`, line:
```yaml
- cron: '0 8 * * *'
```
Use [crontab.guru](https://crontab.guru) to generate a different schedule.

---

## Managing Long Holidays

While you are away:
- The workflow runs automatically every day with no action required
- Positions accumulate in `data/t9b_paper/state.json`
- Each run commits the updated CSVs — full audit trail in git history
- GitHub Issues alert you to new signals via email
- Check `data/t9b_paper/open_positions.csv` on GitHub anytime for a position snapshot

No API keys. No local PC. No intervention needed.

---

## Common Issues and Fixes

### Run fails: "nothing to commit"
This is not an error — it means no positions changed today (no entries/exits).
The workflow still succeeds.

### Run fails: "Permission denied" on git push
Go to repository **Settings → Actions → General → Workflow permissions**
and select **Read and write permissions**. Save.

### Run fails: OHLCV download error for one symbol
Binance occasionally has brief API outages. The engine retries 3 times and
falls back to cached data. The run continues for all other symbols.
Missing one day on one symbol is acceptable.

### "gh: command not found"
The `gh` CLI is pre-installed on GitHub Actions ubuntu-latest. If this error
appears, it likely means the runner image changed. The `create_signal_issues.py`
script handles this gracefully (skips issue creation without failing the run).

### Run stops showing signals after many months
The Donchian N=20 breakout requires price above the previous 20-bar high AND
above EMA200. In a sideways or bear market, signals legitimately drop off.
This is expected behavior — the EMA200 filter is doing its job.

---

## After 3 Months of Paper Trading

The 3-month review date is **2026-08-28**.

Before considering live capital:
1. Review `data/t9b_paper/equity_curve.csv` — is the curve directionally upward?
2. Check `data/t9b_paper/daily_log.csv` — any unexpected behavior?
3. Review the risk sizing theoretical results in
   `data/research_donchian_riskV2_comparison.txt` — Variant A (0.50%) is the
   first step if paper results are satisfactory
4. Implement risk sizing changes one step at a time: 0.25% → 0.50% → 0.75%
5. Each step requires 1 month of paper observation before the next

See Section 17.5 of `TREND_FOLLOWING_RESEARCH_PIPELINE_4.md` for full rules.

---

## Repository Structure (what is and is not committed)

Files **committed** to GitHub (tracked):
```
phase_t9b_donchian_universev2_paper_engine.py
t9b_daily_summary.py
requirements.txt
.github/workflows/t9b_daily.yml
.github/scripts/create_signal_issues.py
data/t9b_paper/state.json          <- updated daily by Actions
data/t9b_paper/open_positions.csv  <- updated daily
data/t9b_paper/signals_today.csv   <- updated daily
data/t9b_paper/equity_curve.csv    <- grows as trades close
data/t9b_paper/daily_log.csv       <- grows daily
data/universe/
data/research_*/                   <- research outputs
```

Files **not committed** (in .gitignore — regenerated from Binance):
```
data/t9b_paper/ohlcv_cache/        <- redownloaded on each run
data/ohlcv_extended/               <- large historical data (~50MB)
data/research_donchian_riskV2_C/ohlcv_1d/
```

---

*T9B paper engine configuration frozen: 2026-05-30*
*Review date: 2026-08-28 (90 days)*
*No real trades until 3-month paper confirmation.*
