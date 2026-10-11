# Funding refresh host — setup for both options

fapi.binance.com is geo-blocked from GitHub's US-hosted Actions runners
(confirmed repeatedly: klines, exchange status, funding itself). Something
that isn't a GitHub-hosted runner has to fetch funding daily. Two ways to do
that; pick one. Both need a non-US host — confirm first:

```
curl -s -o /dev/null -w '%{http_code}\n' https://fapi.binance.com/fapi/v1/ping
```

Expect `200`. If you get `451` or a timeout, this host is also blocked and
neither option below will work from it.

## Shared hardening, either option

This executes repository code on a machine you own — treat it accordingly,
not as "just a cron job":

- **Private repo.** A public repo + a host that can push to it is a bigger
  attack surface than it needs to be; for the self-hosted-runner option
  specifically, GitHub's own docs call out public-repo self-hosted runners
  as a known compromise vector (anyone who can open a PR can get code run on
  your machine). Keep this repo private regardless of which option you pick.
- **Dedicated non-root user.** Whichever option, run it as a user created
  for this one purpose, not your own login or root. No other secrets on
  that host should be reachable from that user's environment.
- **Narrowest credential that works.** The runner option needs no new
  credential (it pushes via the job's own `GITHUB_TOKEN`). The cron option
  needs a deploy key — generate one *for this repo only*, grant it **write**
  access (Settings → Deploy keys → Add deploy key → check "Allow write
  access"), and don't reuse a key that has access to anything else.
- **No fork/PR triggers, ever.** Neither option in this repo listens to
  `pull_request` / `pull_request_target`. If you ever add a trigger to
  `funding_refresh_runner.yml`, it must stay schedule/workflow_dispatch only.

## Option A — self-hosted GitHub Actions runner

Already written: `.github/workflows/funding_refresh_runner.yml`. Reuses the
exact script (`tools/refresh_funding_local.py`) and the same retry-with-
autostash push pattern every other workflow in this repo already uses — no
new git/push logic to get wrong.

**Setup**, on the host:

1. GitHub → this repo → **Settings → Actions → Runners → New self-hosted
   runner**. Follow the exact download/config commands shown there for your
   OS.
2. When prompted for labels, add `funding-refresh` (keep the default
   `self-hosted` label too — the workflow targets both:
   `runs-on: [self-hosted, funding-refresh]`). **Don't** add this label to
   any runner that also serves other workflows — this one should do nothing
   else.
3. Install as a service under the dedicated user:
   ```
   sudo ./svc.sh install <that-user>
   sudo ./svc.sh start
   ```
4. `python3 -m pip install -r requirements.txt` once, so the first real run
   isn't also the first dependency install.
5. Trigger it once by hand (Actions tab → "Funding Refresh (self-hosted)" →
   Run workflow) and confirm it commits `data/futures_universe/funding_rates/`.

**Tradeoffs**: fully visible in the Actions UI (logs, history, re-run
button), no separate credential to manage, but it IS a GitHub Actions
runner on your box — the attack surface is "GitHub Actions' own runner
software plus whatever this workflow does," not nothing.

## Option B — host cron + deploy key

No file in this repo does this yet — this is the setup, not a deployment.

**Setup**, on the host:

1. Generate a key pair *for this purpose only*:
   ```
   ssh-keygen -t ed25519 -C "funding-refresh-deploy-key" -f ~/.ssh/ungerfink_funding_deploy -N ""
   ```
2. GitHub → repo → **Settings → Deploy keys → Add deploy key**. Paste the
   **public** key (`ungerfink_funding_deploy.pub`). Check **Allow write
   access**. This key can push to this repo and nothing else — that's the
   point of a deploy key over a personal token.
3. On the host, as the dedicated user, clone via SSH and point git at this
   key specifically (don't rely on a default `~/.ssh/config` that might
   apply to other keys/repos too):
   ```
   mkdir -p ~/ungerfink-funding && cd ~/ungerfink-funding
   GIT_SSH_COMMAND="ssh -i ~/.ssh/ungerfink_funding_deploy -o IdentitiesOnly=yes" \
     git clone git@github.com:brasdor/UngerFink-TREND.git .
   git config core.sshCommand "ssh -i ~/.ssh/ungerfink_funding_deploy -o IdentitiesOnly=yes"
   python3 -m pip install -r requirements.txt
   ```
4. Crontab (as that user, `crontab -e`) — once daily, after the main daily
   workflows (08:00 UTC) and the health check (~11:30 UTC):
   ```
   0 13 * * * cd ~/ungerfink-funding && git pull --ff-only && python3 tools/refresh_funding_local.py >> ~/funding_refresh.log 2>&1
   ```
   `--ff-only` deliberately: if the local clone has somehow diverged, the
   pull fails loudly instead of the cron job inventing a merge.
5. `tools/refresh_funding_local.py` commits and pushes itself (same retry
   logic as the workflow option, just invoked from cron instead of Actions).
6. Confirm the log file is actually being written and rotated (a cron job
   that silently stops is the exact failure
   `check_funding_refresh_job_alive` in the health layer is watching for —
   see below — but don't rely on that alone; check `~/funding_refresh.log`
   yourself the first few days).

**Tradeoffs**: smaller attack surface (no Actions runner software, no
GitHub-side registration to keep current), but no logs in the Actions UI —
failures only show up as "no commit happened" (the health check) or in the
host's own log file, and the deploy key has to be rotated/revoked manually
if the host is ever decommissioned (a self-hosted runner's token is revoked
by just deleting the runner registration in GitHub's UI).

## Dead-runner / dead-cron detection

Either option's only visible trace *in this repo* is a commit touching
`data/futures_universe/funding_rates/`. `tools/health/run_checks.py`'s
`check_funding_refresh_job_alive()` flags CRITICAL if that directory hasn't
been committed to in over 26 hours — independent of
`check_funding_staleness()`, which looks at whether the *data* is current,
not whether the *job* is still alive. The two can disagree in either
direction (funding only settles every 8h, so data can look fresh for a day
or two after the job already died; a job that's merely slow but alive isn't
the same failure as one that's dead).

This check runs as part of the existing daily health_check.yml — no
separate wiring needed, and it fires regardless of which option above you
end up running.
