#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- sends one line, every single day, confirming the
pipeline is alive: how many systems are current, whether the data caches
are fresh, and today's regime. Unlike check_missed_runs.py /
check_workflow_failures.py, which only speak up when something is wrong,
this one speaks every day on purpose -- so a day with NO digest at all
(not a clean one, none) is itself the signal that something broke
upstream of Telegram delivery, independent of whatever it would have
reported.

Reads data/status_snapshot.json, written earlier in the same job by
build_status_snapshot.py -- no recomputation, so this can never drift
from what /status itself reports.

Fails closed like check_workflow_failures.py / send_telegram.py: a
missing secret or a failed send exits non-zero so the job goes red and
GitHub's own failure email becomes a second channel.

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_PATH = ROOT / "data" / "status_snapshot.json"


def _safe_print(text: str) -> None:
    """Some consoles (Windows cp1252) can't encode the emoji in this digest.
    Same fix as check_missed_runs.py -- never let a diagnostic print crash
    the run over that."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _send(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
         "disable_web_page_preview": "true"}
    ).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except Exception as exc:
        return False, str(exc)


def build_digest_text(today: str) -> str:
    if not SNAPSHOT_PATH.exists():
        _safe_print(f"[DIGEST] {SNAPSHOT_PATH} missing -- cannot build digest")
        return f"⚠️ <b>Daily digest {today}</b> -- status_snapshot.json missing, cannot report"

    try:
        snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[DIGEST] could not read {SNAPSHOT_PATH}: {exc}")
        return f"⚠️ <b>Daily digest {today}</b> -- status_snapshot.json unreadable, cannot report"

    systems = snap.get("systems", {})
    alerts = snap.get("alerts", {})
    regime = snap.get("regime", {})

    n_total = len(systems)
    missed = alerts.get("missed_runs", [])
    # ohlcv_issues today; funding_issues once Phase 2 wires the funding
    # staleness check into compute_heartbeat() -- read generically so this
    # digest picks it up automatically without a second edit.
    issues = list(alerts.get("ohlcv_issues", [])) + list(alerts.get("funding_issues", []))
    n_missed = len({m["system"] for m in missed})
    n_current = n_total - n_missed

    trend = regime.get("trend", "?")
    funding = regime.get("funding", "?")

    clear = not missed and not issues
    marker = "✅" if clear else "⚠️"
    bits = [
        f"{n_current}/{n_total} systems current",
        "data OK" if not issues else f"{len(issues)} data issue(s)",
        f"regime {trend}/{funding}",
    ]
    return f"{marker} <b>Daily digest {today}</b> -- " + " · ".join(bits)


def main() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = build_digest_text(today)
    _safe_print(f"[DIGEST] {text}")

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_ids = [c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
    if not token or not chat_ids:
        print("[DIGEST] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set -- failing closed "
              "(no proof-of-life message could be sent today)")
        return 1

    delivered = 0
    last_err = ""
    for chat_id in chat_ids:
        ok, err = _send(token, chat_id, text)
        if ok:
            delivered += 1
        else:
            last_err = err
            print(f"[DIGEST] delivery to {chat_id} failed: {err}")

    print(f"[DIGEST] delivered={delivered}/{len(chat_ids)}")
    if delivered < len(chat_ids):
        print(f"[DIGEST] delivery incomplete -- failing closed (last error: {last_err})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
