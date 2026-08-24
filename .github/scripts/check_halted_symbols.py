#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- monthly re-verification of the halted-symbol
suppression list (data/halted_symbols_suppression.json) that keeps
check_missed_runs.py's daily OHLCV staleness check from alerting forever
on a symbol that's genuinely halted exchange-side (confirmed for
TON_USDT/ARDR_USDT on 2026-08-23 via Binance's live exchangeInfo).

Runs monthly (see .github/workflows/monthly_halted_symbol_check.yml), not
on trust: re-queries the same live endpoint used to originally confirm
each symbol's BREAK status. A symbol whose status is no longer BREAK is
removed from the suppression file in this same run -- the very next daily
heartbeat then evaluates it like any other symbol and alerts normally if
it's stale for a real reason. Nothing about "resuming alerts" needs to be
implemented separately; it falls out of the suppression file no longer
listing the symbol.

A symbol that can't be re-verified this run (network error, unexpected
API response) is left suppressed rather than dropped -- a transient API
hiccup should never silently turn back on an alert for a symbol that's
almost certainly still halted -- but it's logged loudly, since a
suppression list that can go stale without anyone noticing is exactly the
class of problem this whole alerting system exists to prevent.

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (optional -- only used to note a
  symbol resuming trading, a rare and actionable event; silence otherwise)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
SUPPRESSION_FILE = ROOT / "data" / "halted_symbols_suppression.json"

EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"


def _binance_symbol(local_key: str) -> str:
    """'TON_USDT' -> 'TONUSDT' -- matches how check_ohlcv_staleness() names
    the local cache file (symbol.replace('/','_')), reversed for the API."""
    return local_key.replace("_", "")


def query_status(binance_symbol: str) -> str | None:
    """Returns the live exchange status string (e.g. 'TRADING', 'BREAK'),
    or None if the query itself failed (network error, symbol not found,
    unexpected shape)."""
    try:
        r = requests.get(EXCHANGE_INFO_URL, params={"symbol": binance_symbol}, timeout=15)
        if r.status_code != 200:
            print(f"    [WARN] {binance_symbol}: HTTP {r.status_code}")
            return None
        data = r.json()
        symbols = data.get("symbols", [])
        if not symbols:
            print(f"    [WARN] {binance_symbol}: no symbol entry in response")
            return None
        return symbols[0].get("status")
    except Exception as exc:
        print(f"    [WARN] {binance_symbol}: query failed -- {exc}")
        return None


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
    if not SUPPRESSION_FILE.exists():
        print(f"[HALTED-CHECK] {SUPPRESSION_FILE} does not exist -- nothing to re-verify")
        return 0

    suppression = json.loads(SUPPRESSION_FILE.read_text(encoding="utf-8"))
    today = date.today().isoformat()

    resumed: list[tuple[str, str]] = []
    still_halted: list[str] = []
    unverified: list[str] = []

    print(f"[HALTED-CHECK] re-verifying {len([k for k in suppression if not k.startswith('_')])} "
          f"suppressed symbol(s) via live exchangeInfo...")

    for key in list(suppression.keys()):
        if key.startswith("_"):
            continue  # e.g. "_comment"
        entry = suppression[key]
        bsym = _binance_symbol(key)
        status = query_status(bsym)

        if status is None:
            unverified.append(key)
            print(f"  {key} ({bsym}): could not re-verify this run -- leaving suppressed")
            continue

        if status == "BREAK":
            entry["last_verified"] = today
            entry["last_verified_status"] = status
            still_halted.append(key)
            print(f"  {key} ({bsym}): still BREAK -- remains suppressed")
        else:
            resumed.append((key, status))
            del suppression[key]
            print(f"  {key} ({bsym}): status is now '{status}' -- RESUMED, removing from suppression")

    if resumed:
        SUPPRESSION_FILE.write_text(json.dumps(suppression, indent=2) + "\n", encoding="utf-8")
        print(f"[HALTED-CHECK] wrote updated suppression file -- {len(resumed)} symbol(s) removed")
    else:
        # still refresh last_verified timestamps for symbols that remain halted
        SUPPRESSION_FILE.write_text(json.dumps(suppression, indent=2) + "\n", encoding="utf-8")
        print("[HALTED-CHECK] no symbols resumed -- suppression list unchanged (verification timestamps refreshed)")

    if unverified:
        print(f"[HALTED-CHECK] WARNING: {len(unverified)} symbol(s) could not be re-verified "
              f"this run: {unverified} -- will retry next month")

    if not resumed:
        print("[HALTED-CHECK] done, no alert needed")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"ℹ️ <b>Halted symbol(s) resumed trading</b>  {now_str}"]
    for key, status in resumed:
        lines.append(f"  <code>{key}</code>: now '{status}' -- removed from staleness suppression")
    lines.append("\n(daily staleness alerting now applies to these again, same as any other symbol.)")
    message = "\n".join(lines)

    if not token or not chat_id:
        print(f"[HALTED-CHECK] {len(resumed)} symbol(s) resumed but TELEGRAM_* not set")
        return 0

    ok = _send(token, chat_id, message)
    print(f"[HALTED-CHECK] sent={ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
