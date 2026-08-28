#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- alert on step failures that continue-on-error hid.

Every engine step in t9b_daily.yml and t9_candidates_daily.yml runs with
continue-on-error: true so one broken engine doesn't stop the others --
but that also makes the job report "success" even when a step genuinely
crashed. This is the exact silent-failure shape behind today's incident:
an early exception with no output written leaves nothing for the commit
step to pick up, and nothing else in the run ever complains.

Reads step outcomes passed in via STEP_OUTCOMES ("Label=outcome,..."),
sends one consolidated Telegram alert if any step outcome isn't
success/skipped. No-ops (exit 0, logs to stdout) if everything is clean,
if Telegram secrets aren't configured, or on any send error -- this
script must never fail the workflow it's reporting on.

Env:
  WORKFLOW_LABEL     human-readable name for the alert header
  STEP_OUTCOMES      "Label1=outcome1,Label2=outcome2,..."
  RUN_URL            optional link back to the Actions run
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone


def _send(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
         "disable_web_page_preview": "true"}
    ).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[TELEGRAM] send failed: {exc}")
        return False


def main() -> int:
    workflow_label = os.environ.get("WORKFLOW_LABEL", "Workflow")
    raw = os.environ.get("STEP_OUTCOMES", "")
    run_url = os.environ.get("RUN_URL", "")

    failed: list[tuple[str, str]] = []
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, outcome = pair.split("=", 1)
        name, outcome = name.strip(), outcome.strip()
        if outcome and outcome not in ("success", "skipped"):
            failed.append((name, outcome))

    if not failed:
        print(f"[ALERT] {workflow_label}: all steps OK, no alert sent")
        return 0

    print(f"[ALERT] {workflow_label}: {len(failed)} step(s) not OK: {failed}")

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    # Comma-separated so every operator gets the alert, not just the one whose
    # chat id happens to be in the secret. A single id keeps working unchanged.
    chat_ids = [c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
    if not token or not chat_ids:
        print("[ALERT] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set -- skipping send")
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"⚠️ <b>{workflow_label} -- step failure(s)</b>  {today}"]
    for name, outcome in failed:
        lines.append(f"  ❌ <code>{name}</code>: {outcome}")
    if run_url:
        lines.append(f"\n{run_url}")
    lines.append("\n(continue-on-error kept the job green -- this step failed underneath it.)")

    text = "\n".join(lines)
    delivered = 0
    for chat_id in chat_ids:
        if _send(token, chat_id, text):
            delivered += 1
        else:
            print(f"[ALERT] delivery to {chat_id} failed")
    print(f"[ALERT] delivered={delivered}/{len(chat_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
