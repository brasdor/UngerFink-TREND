#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- push a Telegram summary of today's T9B signals.

Reads each engine's signals_today.csv (+ state.json for the run date, daily_log
for accepted/skipped classification) and sends one compact message to a Telegram
chat. Mirrors the system registry in create_signal_issues.py.

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (from @BotFather / your chat id)

Never fails the workflow: any error is logged and the script exits 0.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEMS = [
    {"name": "Donchian",   "label": "DonchianLong",    "data_dir": ROOT / "data" / "t9b_paper"},
    {"name": "RSI-MR",     "label": "MeanReversionRSI", "data_dir": ROOT / "data" / "t9b_mr_paper"},
    {"name": "ConsecDown", "label": "ConsecDownDaysMR", "data_dir": ROOT / "data" / "t9b_consecdowndays_paper"},
]


def _fmt(val, decimals: int = 6) -> str:
    try:
        return f"{float(val):.{decimals}g}"
    except (TypeError, ValueError):
        return str(val)


def _classify(data_dir: Path, run_date: str) -> tuple[set, set]:
    """Return (accepted_symbols, skipped_symbols) from today's daily_log."""
    accepted: set = set()
    skipped: set = set()
    daily_log = data_dir / "daily_log.csv"
    if daily_log.exists() and run_date:
        try:
            log = pd.read_csv(daily_log)
            today = log[log["run_date"] == run_date]
            accepted = set(today[today["event"] == "ENTRY"]["symbol"].dropna().tolist())
            skipped = set(today[today["event"] == "SIGNAL_SKIPPED"]["symbol"].dropna().tolist())
        except Exception:
            pass
    return accepted, skipped


def _system_lines(system: dict) -> tuple[str, int]:
    """Build the message block for one engine. Returns (text, signal_count)."""
    data_dir = system["data_dir"]
    signals_csv = data_dir / "signals_today.csv"
    if not signals_csv.exists():
        return f"<b>{system['name']}</b>: (no file)", 0

    try:
        df = pd.read_csv(signals_csv)
    except Exception:
        return f"<b>{system['name']}</b>: (unreadable)", 0

    if df.empty:
        return f"<b>{system['name']}</b>: no signals", 0

    run_date = ""
    state_json = data_dir / "state.json"
    if state_json.exists():
        try:
            run_date = json.loads(state_json.read_text(encoding="utf-8")).get("last_run_date", "")
        except Exception:
            pass

    accepted, skipped = _classify(data_dir, run_date)

    rows = []
    for _, row in df.iterrows():
        sym = str(row.get("symbol", "?"))
        close = _fmt(row.get("close", "?"))
        if sym in accepted:
            tag = "\U0001F7E2 ENTRY"      # green circle
        elif sym in skipped:
            tag = "⚪ SKIP"           # white circle
        else:
            tag = "\U0001F50D detected"   # magnifier
        rows.append(f"  {tag}  <code>{sym}</code> @ {close}")

    header = f"<b>{system['name']}</b> ({len(df)} signal{'s' if len(df) != 1 else ''})"
    return header + "\n" + "\n".join(rows), len(df)


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
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[TELEGRAM] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set -- skipping")
        return 0

    blocks = []
    total = 0
    for system in SYSTEMS:
        text, count = _system_lines(system)
        blocks.append(text)
        total += count

    today = ""
    try:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        pass

    title = f"\U0001F4C8 <b>UngerFink T9B</b> -- {today}"
    if total == 0:
        title += "  (no new signals)"
    message = title + "\n\n" + "\n\n".join(blocks)

    ok = _send(token, chat_id, message)
    print(f"[TELEGRAM] sent={ok}  total_signals={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
