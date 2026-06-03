#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- create issues for T9B signals fired today.

Handles all three T9B paper engines:
  1. DonchianLong_UniverseV2_ExitV2  (data/t9b_paper/)
  2. MeanReversionRSI 1D             (data/t9b_mr_paper/)
  3. ConsecDownDaysMR 1D             (data/t9b_consecdowndays_paper/)

For each system, reads signals_today.csv + daily_log.csv + state.json,
creates one GitHub issue per detected signal using the 'gh' CLI tool.
Requires GH_TOKEN or GITHUB_TOKEN environment variable.

Exit codes:
  0 -- success (including no signals today)
  1 -- critical error
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# System registry -- one entry per T9B engine
SYSTEMS = [
    {
        "name":      "DonchianLong",
        "label":     "DonchianLong_UniverseV2_ExitV2",
        "data_dir":  ROOT / "data" / "t9b_paper",
        "config_md": (
            "System  : DonchianLong UniverseV2 ExitV2 -- T9B Paper Engine\n"
            "Entry   : Donchian N=20 breakout above EMA200\n"
            "Stop    : ATR(14) x 2.0 below entry close\n"
            "Exit    : Donchian N=10 OR Chandelier (ACT=6R trail=5.0xATR)\n"
            "Risk    : 0.25% of paper equity per trade\n"
            "Max pos : 8 concurrent"
        ),
        "issue_label": "paper-signal",
    },
    {
        "name":      "RSI_MR",
        "label":     "MeanReversionRSI 1D",
        "data_dir":  ROOT / "data" / "t9b_mr_paper",
        "config_md": (
            "System  : MeanReversionRSI 1D -- T9B Paper Engine\n"
            "Entry   : RSI(14) < 25 (no EMA filter)\n"
            "Stop    : ATR(14) x 3.0 below entry close (safety net only)\n"
            "Exit    : Fixed time exit after 20 bars\n"
            "Risk    : 0.25% of paper equity per trade\n"
            "Max pos : 10 concurrent"
        ),
        "issue_label": "paper-signal",
    },
    {
        "name":      "ConsecDownMR",
        "label":     "ConsecDownDaysMR 1D",
        "data_dir":  ROOT / "data" / "t9b_consecdowndays_paper",
        "config_md": (
            "System  : ConsecDownDaysMR 1D -- T9B Paper Engine\n"
            "Entry   : 5 consecutive down closes AND close > EMA(200)\n"
            "Stop    : ATR(14) x 2.0 below entry close (safety net only)\n"
            "Exit    : Fixed time exit after 20 bars\n"
            "Risk    : 0.25% of paper equity per trade\n"
            "Max pos : uncapped"
        ),
        "issue_label": "paper-signal",
    },
]


def _fmt(val, decimals: int = 6) -> str:
    try:
        return f"{float(val):.{decimals}g}"
    except (TypeError, ValueError):
        return str(val)


def _build_issue_body(system: dict, row: pd.Series,
                      run_date: str, status_line: str) -> str:
    sym  = str(row.get("symbol", "?"))
    date = str(row.get("signal_date", run_date))
    cls  = _fmt(row.get("close", "?"))

    # System-specific signal fields
    extra_rows = ""
    if system["name"] == "DonchianLong":
        extra_rows = (
            f"| **Donchian breakout level** | {_fmt(row.get('don_entry_high', '?'))} |\n"
            f"| **Initial stop (ATR x2)** | {_fmt(row.get('initial_stop', '?'))} |\n"
            f"| **ATR at signal** | {_fmt(row.get('atr', '?'))} |\n"
            f"| **EMA200** | {_fmt(row.get('ema200', '?'))} |\n"
        )
    elif system["name"] == "RSI_MR":
        extra_rows = (
            f"| **RSI(14) at signal** | {_fmt(row.get('rsi', '?'), 4)} |\n"
            f"| **Safety stop** | {_fmt(row.get('stop_loss', '?'))} |\n"
            f"| **ATR at signal** | {_fmt(row.get('atr', '?'))} |\n"
        )
    elif system["name"] == "ConsecDownMR":
        extra_rows = (
            f"| **Consec down closes** | {_fmt(row.get('consec', '?'), 0)} |\n"
            f"| **EMA200 at signal** | {_fmt(row.get('ema200', '?'))} |\n"
            f"| **Safety stop** | {_fmt(row.get('stop_loss', '?'))} |\n"
            f"| **ATR at signal** | {_fmt(row.get('atr', '?'))} |\n"
        )

    return (
        f"## T9B Paper Signal -- {system['label']}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Symbol** | `{sym}` |\n"
        f"| **Date** | {date} |\n"
        f"| **Status** | {status_line} |\n"
        f"| **Entry price (close)** | {cls} |\n"
        f"{extra_rows}"
        f"\n---\n\n"
        f"### Config\n"
        f"```\n"
        f"{system['config_md']}\n"
        f"```\n\n"
        f"_Automated paper trading signal. No real trade was placed._\n"
        f"_Do not act on this signal until T9B has run for at least 3 months._"
    )


def process_system(system: dict, gh_ok: bool) -> int:
    data_dir    = system["data_dir"]
    signals_csv = data_dir / "signals_today.csv"
    daily_log   = data_dir / "daily_log.csv"
    state_json  = data_dir / "state.json"

    if not signals_csv.exists():
        print(f"[{system['name']}] signals_today.csv not found -- skipping")
        return 0

    try:
        sig_df = pd.read_csv(signals_csv)
    except Exception as exc:
        print(f"[{system['name']}] Could not read signals: {exc}")
        return 0

    if sig_df.empty or len(sig_df) == 0:
        print(f"[{system['name']}] No signals today")
        return 0

    # Get run date
    run_date = ""
    if state_json.exists():
        try:
            state    = json.loads(state_json.read_text(encoding="utf-8"))
            run_date = state.get("last_run_date", "")
        except Exception:
            pass

    # Classify signals
    accepted_symbols: set = set()
    skipped_symbols:  set = set()
    if daily_log.exists() and run_date:
        try:
            log       = pd.read_csv(daily_log)
            today_log = log[log["run_date"] == run_date]
            accepted_symbols = set(
                today_log[today_log["event"] == "ENTRY"]["symbol"].dropna().tolist()
            )
            skipped_symbols = set(
                today_log[today_log["event"] == "SIGNAL_SKIPPED"]["symbol"].dropna().tolist()
            )
        except Exception as exc:
            print(f"[{system['name']}] Could not read daily_log: {exc}")

    print(f"[{system['name']}] {len(sig_df)} signal(s) today  "
          f"(accepted={len(accepted_symbols)}  skipped={len(skipped_symbols)})")

    if not gh_ok:
        for _, row in sig_df.iterrows():
            sym = str(row.get("symbol", "?"))
            print(f"  [SKIP-NO-GH] {sym} -- gh CLI not available")
        return 0

    created = 0
    for _, row in sig_df.iterrows():
        sym = str(row.get("symbol", "?"))
        sig_date = str(row.get("signal_date", run_date))

        if sym in accepted_symbols:
            status_line = "ACCEPTED -- paper entry placed in state.json"
        elif sym in skipped_symbols:
            status_line = "SKIPPED  -- portfolio cap reached; not entered"
        else:
            status_line = "DETECTED"

        title = f"T9B {system['name']}: {sym} {sig_date}"
        body  = _build_issue_body(system, row, run_date, status_line)

        for cmd in [
            ["gh", "issue", "create", "--title", title, "--body", body,
             "--label", system["issue_label"]],
            ["gh", "issue", "create", "--title", title, "--body", body],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                url = result.stdout.strip()
                print(f"  [CREATED] {title}")
                print(f"            {url}")
                created += 1
                break
            else:
                if "--label" in cmd:
                    continue
                print(f"  [FAILED] {sym}: {result.stderr.strip()}")

    print(f"[{system['name']}] Done -- {created}/{len(sig_df)} issues created")
    return created


def main() -> int:
    gh_check = subprocess.run(["gh", "--version"], capture_output=True)
    gh_ok    = gh_check.returncode == 0
    if not gh_ok:
        print("[ISSUES] 'gh' CLI not available -- will log signals but not create issues")

    total_created = 0
    for sys_cfg in SYSTEMS:
        total_created += process_system(sys_cfg, gh_ok)

    print(f"\n[ISSUES] Total issues created across all systems: {total_created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
