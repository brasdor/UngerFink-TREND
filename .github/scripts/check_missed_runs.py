#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- detect a scheduled run that never happened, AND
detect the failure mode that started this whole investigation: a run that
"succeeds" every day while silently processing frozen/corrupted data.

Three distinct failure modes, three checks, one consolidated alert:

1. Missed run: the workflow never fired, or the engine died before
   writing any output. continue-on-error + "only commit when there's a
   diff" leave zero trace inside that run for check_workflow_failures.py
   to see. Detected here by comparing each system's actual last commit on
   its state.json (git history, not a field inside the file a broken
   pipeline could leave stale) against the expected daily cadence.

2. Step failure that continue-on-error hid: covered by the separate
   check_workflow_failures.py, run from inside t9b_daily.yml /
   t9_candidates_daily.yml themselves.

3. THE ORIGINAL INCIDENT'S actual shape, and the reason (1) and (2) alone
   are not sufficient: the spot OHLCV freeze exited 0 and committed
   state.json daily throughout (confirmed: 2026-06-04 through at least
   2026-06-07 all committed normally) while quietly processing corrupted
   1970-01-01 rows. Neither a missed-run check nor a step-outcome check
   would ever have caught it -- the pipeline never stopped succeeding.
   The only thing that actually would have caught it is exactly what
   spot_data_refresh.check_staleness() and refresh_futures_data.py's
   check_cache_staleness() already compute and print to the CI log: how
   stale is the most-stale symbol's last real bar. That printed warning
   never reached anyone, because nothing read the CI log. This script
   reuses that same "last row's date vs. today" check directly and routes
   it to Telegram instead of a log line no one was watching.

Runs from its own workflow (heartbeat_check.yml), independently of and
later than the workflows it's watching -- a check embedded inside
t9b_daily.yml can't detect t9b_daily.yml not firing, and today's incident
already proved a check that only looks at "did the step exit non-zero"
isn't enough either.

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "engines"))

MAX_STALE_HOURS = 36.0  # daily cron at 08:00 UTC + generous buffer for retries/delays
MAX_OHLCV_STALE_DAYS = 3  # same threshold already used in spot_data_refresh.py /
                          # refresh_futures_data.py's own staleness prints


def _spot_universe() -> set[str] | None:
    """S1's + S3's actual trading universe (66 symbols), not every file that
    happens to sit in data/universe/ohlcv_1d/ -- that directory also holds
    unrelated/historical symbol files (e.g. 0G_USDT, from an old exploratory
    universe list) nothing currently trades, which would otherwise show up
    as false-positive staleness noise. Falls back to "check every file" if
    either source can't be read, since an overly-broad check is safer than
    silently checking nothing.
    """
    try:
        s1 = pd.read_csv(ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv")
        syms = set(s1["symbol"].dropna().tolist())
        import phase_t9b_consecdowndays_paper_engine as s3_engine  # noqa: E402
        syms |= set(s3_engine.CD_UNIVERSE)
        return {s.replace("/", "_").replace(":", "_") for s in syms}
    except Exception as exc:
        print(f"  [WARN] could not build spot universe allowlist ({exc}); checking every file")
        return None


SUPPRESSION_FILE = ROOT / "data" / "halted_symbols_suppression.json"


def _suppressed_symbols() -> set[str]:
    """Confirmed-exchange-halted symbols (see check_halted_symbols.py) --
    excluded from staleness alerting so a permanent halt doesn't page
    forever. Missing/unreadable file just means nothing is suppressed."""
    if not SUPPRESSION_FILE.exists():
        return set()
    try:
        raw = json.loads(SUPPRESSION_FILE.read_text(encoding="utf-8"))
        return {k for k in raw if not k.startswith("_")}
    except Exception as exc:
        print(f"  [WARN] could not read {SUPPRESSION_FILE} ({exc}); suppressing nothing")
        return set()


OHLCV_DIRS = [
    ("spot universe (S1/S3)", ROOT / "data" / "universe" / "ohlcv_1d", _spot_universe),
    ("futures universe (S2/S5-S8/candidates)", ROOT / "data" / "futures_universe" / "ohlcv_1d", None),
]

SYSTEMS = [
    ("S1 Donchian",       "data/t9b_paper/state.json"),
    ("S2 RSI-MR",         "data/t9b_mr_paper/state.json"),
    ("S3 ConsecDown",     "data/t9b_consecdowndays_paper/state.json"),
    ("S5 Momentum",       "data/t9b_momentum_paper/state.json"),
    ("S6 VolContraction", "data/t9b_volcontraction_paper/state.json"),
    ("S7 MACross",        "data/t9b_macross_paper/state.json"),
    ("S8 RSI-MR-Funding", "data/t9b_rsi_mr_funding_paper/state.json"),
    ("Candidate 12",      "data/t9_candidate12_paper/state.json"),
    ("Candidate 19",      "data/t9_candidate19_paper/state.json"),
]


def last_commit_time(rel_path: str):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", rel_path],
            cwd=ROOT, capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        if not out:
            return None
        return datetime.fromisoformat(out)
    except Exception as exc:
        print(f"  [WARN] git log failed for {rel_path}: {exc}")
        return None


def _safe_print(text: str) -> None:
    """Some consoles (Windows cp1252) can't encode the emoji used in Telegram
    messages. Never let a diagnostic print crash the run over that."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


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


def check_ohlcv_staleness(label: str, cache_dir: Path, today,
                          allowlist: set[str] | None = None,
                          suppress: set[str] | None = None) -> tuple[str, int] | None:
    """
    Worst-case staleness across every symbol file in cache_dir (or, if
    allowlist is given, just the symbols actually traded -- avoids flagging
    an unrelated/historical file nothing currently trades). Same "last
    row's date vs. today" check already proven in spot_data_refresh.py /
    refresh_futures_data.py, just routed to Telegram instead of a CI log
    line. This is the check that would actually have caught the original
    spot-OHLCV freeze -- a missed-run or step-outcome check would not have,
    since that pipeline kept exiting 0 and committing daily throughout.

    suppress excludes confirmed-exchange-halted symbols (see
    data/halted_symbols_suppression.json) so a real, permanent exchange
    halt doesn't fire this alert every single day forever. Re-verified
    monthly, not trusted indefinitely -- see check_halted_symbols.py.
    """
    if not cache_dir.exists():
        return (f"{label}: cache dir missing ({cache_dir})", 9999)

    worst_sym, worst_days = None, -1
    n_checked = 0
    n_suppressed = 0
    for f in cache_dir.glob("*_1d.csv"):
        sym = f.stem.replace("_1d", "")
        if allowlist is not None and sym not in allowlist:
            continue
        if suppress is not None and sym in suppress:
            n_suppressed += 1
            continue
        try:
            last_row = pd.read_csv(f, usecols=["time"]).iloc[-1]["time"]
            last_date = pd.to_datetime(last_row, utc=True).date()
        except Exception:
            try:
                # futures cache uses "timestamp" (ms) instead of "time"
                last_row = pd.read_csv(f, usecols=["timestamp"]).iloc[-1]["timestamp"]
                last_date = pd.to_datetime(int(last_row), unit="ms", utc=True).date()
            except Exception:
                continue
        n_checked += 1
        gap_days = (today - last_date).days
        if gap_days > worst_days:
            worst_sym, worst_days = sym, gap_days

    suppressed_note = f", {n_suppressed} suppressed" if n_suppressed else ""
    if n_checked == 0 or worst_days <= MAX_OHLCV_STALE_DAYS:
        print(f"  {label}: OK ({n_checked} symbols checked{suppressed_note}, worst: {worst_sym}, {worst_days}d)")
        return None
    print(f"  {label}: STALE -- worst: {worst_sym} ({worst_days}d){suppressed_note}  *** STALE ***")
    return (f"{label}: {worst_sym} is {worst_days}d behind (checked {n_checked} symbols{suppressed_note})", worst_days)


def main() -> int:
    now = datetime.now(timezone.utc)
    stale: list[tuple[str, str]] = []

    print("[HEARTBEAT] checking last real commit per system...")
    for name, rel_path in SYSTEMS:
        path = ROOT / rel_path
        if not path.exists():
            stale.append((name, f"{rel_path} does not exist"))
            print(f"  {name}: MISSING FILE")
            continue
        ts = last_commit_time(rel_path)
        if ts is None:
            stale.append((name, "no commit history found for state.json"))
            print(f"  {name}: NO COMMIT HISTORY")
            continue
        age_hours = (now - ts).total_seconds() / 3600.0
        flag = "" if age_hours <= MAX_STALE_HOURS else "  *** STALE ***"
        print(f"  {name}: last commit {ts.isoformat()}  ({age_hours:.1f}h ago){flag}")
        if age_hours > MAX_STALE_HOURS:
            stale.append((name, f"{age_hours:.1f}h since last real update (threshold {MAX_STALE_HOURS:.0f}h)"))

    print("[HEARTBEAT] checking OHLCV cache staleness (the check that would have "
          "caught the original incident -- a pipeline that keeps exiting 0 while "
          "silently processing frozen data)...")
    suppressed = _suppressed_symbols()
    if suppressed:
        print(f"  (suppressing confirmed-halted symbols: {sorted(suppressed)})")

    data_issues: list[str] = []
    for label, cache_dir, allowlist_fn in OHLCV_DIRS:
        allowlist = allowlist_fn() if allowlist_fn is not None else None
        result = check_ohlcv_staleness(label, cache_dir, now.date(), allowlist=allowlist, suppress=suppressed)
        if result is not None:
            data_issues.append(result[0])

    if not stale and not data_issues:
        print("[HEARTBEAT] all systems current, no alert sent")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    today = now.strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🚨 <b>Daily heartbeat -- problem(s) found</b>  {today}"]

    if stale:
        lines.append("\n<b>Missed run(s):</b>")
        for name, detail in stale:
            lines.append(f"  ⚠️ <code>{name}</code>: {detail}")
        lines.append("(state.json wasn't committed on schedule -- the daily workflow "
                      "either didn't fire or the engine crashed before writing output.)")

    if data_issues:
        lines.append("\n<b>OHLCV cache staleness:</b>")
        for issue in data_issues:
            lines.append(f"  ⚠️ {issue}")
        lines.append(f"(more than {MAX_OHLCV_STALE_DAYS}d behind -- the same check that "
                      "would have caught the 2026-06 spot-OHLCV freeze on day 1 instead "
                      "of after ~2 months.)")

    message = "\n".join(lines)

    if not token or not chat_id:
        _safe_print(f"[HEARTBEAT] problem(s) found but TELEGRAM_* not set:\n{message}")
        return 0

    ok = _send(token, chat_id, message)
    print(f"[HEARTBEAT] sent={ok}  missed_runs={len(stale)}  data_issues={len(data_issues)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
