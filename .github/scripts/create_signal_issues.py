#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- create issues for T9B signals fired today.

Reads:
  data/t9b_paper/signals_today.csv   -- all detected signals
  data/t9b_paper/daily_log.csv       -- to classify accepted vs skipped
  data/t9b_paper/state.json          -- to get last_run_date

Creates one GitHub issue per detected signal using the 'gh' CLI tool
(pre-installed on GitHub Actions ubuntu-latest runners).
Requires GH_TOKEN or GITHUB_TOKEN environment variable.

Exit codes:
  0 -- success (including no signals today)
  1 -- critical error (should not happen in normal operation)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent.parent
SIGNALS_CSV  = ROOT / "data" / "t9b_paper" / "signals_today.csv"
DAILY_LOG    = ROOT / "data" / "t9b_paper" / "daily_log.csv"
STATE_JSON   = ROOT / "data" / "t9b_paper" / "state.json"


def main() -> int:
    # ── load signals ─────────────────────────────────────────────────────────
    if not SIGNALS_CSV.exists():
        print("[ISSUES] signals_today.csv not found -- no issues to create")
        return 0

    try:
        sig_df = pd.read_csv(SIGNALS_CSV)
    except Exception as exc:
        print(f"[ISSUES] Could not read signals_today.csv: {exc}")
        return 0

    if sig_df.empty or len(sig_df) == 0:
        print("[ISSUES] No signals today -- no issues to create")
        return 0

    # ── get run date from state ───────────────────────────────────────────────
    run_date = ""
    if STATE_JSON.exists():
        try:
            state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            run_date = state.get("last_run_date", "")
        except Exception:
            pass

    # ── classify signals as accepted/skipped from daily_log ──────────────────
    accepted_symbols: set = set()
    skipped_symbols:  set = set()

    if DAILY_LOG.exists() and run_date:
        try:
            log = pd.read_csv(DAILY_LOG)
            today_log = log[log["run_date"] == run_date]
            accepted_symbols = set(
                today_log[today_log["event"] == "ENTRY"]["symbol"].dropna().tolist()
            )
            skipped_symbols = set(
                today_log[today_log["event"] == "SIGNAL_SKIPPED"]["symbol"].dropna().tolist()
            )
        except Exception as exc:
            print(f"[ISSUES] Could not read daily_log.csv: {exc}")

    # ── check gh CLI is available ─────────────────────────────────────────────
    gh_check = subprocess.run(["gh", "--version"], capture_output=True)
    if gh_check.returncode != 0:
        print("[ISSUES] 'gh' CLI not available -- skipping issue creation")
        return 0

    # ── create one issue per signal ───────────────────────────────────────────
    created = 0
    for _, row in sig_df.iterrows():
        sym      = str(row.get("symbol", "?"))
        sig_date = str(row.get("signal_date", run_date))
        close    = row.get("close", "?")
        don_high = row.get("don_entry_high", "?")
        stop     = row.get("initial_stop", "?")
        atr      = row.get("atr", "?")
        ema200   = row.get("ema200", "?")

        if sym in accepted_symbols:
            status_line = "ACCEPTED -- paper entry placed in state.json"
        elif sym in skipped_symbols:
            status_line = "SKIPPED  -- portfolio cap (max 8) was reached; not entered"
        else:
            status_line = "DETECTED"

        title = f"T9B Signal: {sym} {sig_date}"

        body = (
            f"## T9B Paper Signal\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Symbol** | `{sym}` |\n"
            f"| **Date** | {sig_date} |\n"
            f"| **Status** | {status_line} |\n"
            f"| **Entry price (close)** | {close} |\n"
            f"| **Donchian breakout level** | {don_high} |\n"
            f"| **Initial stop (ATR x2)** | {stop} |\n"
            f"| **ATR at signal** | {atr} |\n"
            f"| **EMA200** | {ema200} |\n\n"
            f"---\n\n"
            f"### Config\n"
            f"```\n"
            f"System  : DonchianLong UniverseV2 ExitV2 -- T9B Paper Engine\n"
            f"Entry   : Donchian N=20 breakout above EMA200\n"
            f"Stop    : ATR(14) x 2.0 below entry close\n"
            f"Exit    : Donchian N=10 lower band OR Chandelier (ACT=6R trail=5.0xATR)\n"
            f"Risk    : 0.25% of paper equity per trade\n"
            f"Max pos : 8 concurrent\n"
            f"```\n\n"
            f"_This is an automated paper trading signal. No real trade was placed._\n"
            f"_Do not act on this signal until T9B has run for at least 3 months._"
        )

        # Try with label first; fall back without label if it does not exist
        for cmd in [
            ["gh", "issue", "create", "--title", title, "--body", body, "--label", "paper-signal"],
            ["gh", "issue", "create", "--title", title, "--body", body],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                url = result.stdout.strip()
                print(f"[ISSUES] Created: {title}")
                print(f"         {url}")
                created += 1
                break
            else:
                if "--label" in cmd:
                    continue   # retry without label
                print(f"[ISSUES] Failed to create issue for {sym}: {result.stderr.strip()}")

    print(f"[ISSUES] Done -- {created}/{len(sig_df)} issues created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
