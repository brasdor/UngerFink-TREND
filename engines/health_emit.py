#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The one place an engine or a shared/consumer module calls into instead of
swallowing an exception, silently falling back to a default, or dropping a
row that wouldn't parse. See docs/HEALTH_CONTRACT.md for the full contract.

emit()/resolve() ONLY append to data/health/health_events.jsonl. Neither
one touches data/health/health_report.json -- that file is owned
end-to-end by tools/health/run_checks.py, which derives it by replaying
every event. Keeping emit() this dumb is deliberate: an engine crashing
mid-write can corrupt at most one appended line, never the aggregate
report, and two engines calling emit() around the same time just append
two lines (no read-modify-write race on a shared JSON file).

Both functions are defensive about their own failure: a disk-full or a bad
argument here must never be the reason an engine crashes. Neither raises.

Usage, at a site that used to do this:

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()

do this instead:

    try:
        df = pd.read_csv(path)
        resolve("S2", "daily_log_unparseable")
    except Exception as exc:
        emit("S2", "daily_log_unparseable", "WARNING",
             f"{path} did not parse: {exc}",
             evidence={"path": str(path), "error": str(exc)},
             suggested_action="Check for a ragged row or duplicate header; "
                               "see docs/HEALTH_CONTRACT.md.")
        df = pd.DataFrame()
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HEALTH_DIR = ROOT / "data" / "health"
EVENTS_PATH = HEALTH_DIR / "health_events.jsonl"

# Must match tools/health/run_checks.py's SCHEMA_VERSION -- both write
# events to the same file and must agree on the schema they claim.
SCHEMA_VERSION = "1.0"

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(system: str, check: str, key: str = "") -> str:
    """Same scheme the runner uses -- an engine emit() and a registry check
    for the same real-world condition must produce the same id to converge
    on one finding. See docs/HEALTH_CONTRACT.md's "Stable ids" section."""
    return f"{system}:{check}:{key}" if key else f"{system}:{check}"


def _append(event: dict) -> None:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str)
    with _LOCK:
        with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def emit(system: str, check: str, severity: str, message: str, *,
         key: str = "", evidence: dict[str, Any] | None = None,
         suggested_action: str = "", run_date: str | None = None) -> str:
    """Append an 'open' event. Returns the finding id. Never raises."""
    try:
        today = run_date or _now_iso()[:10]
        finding_id = make_id(system, check, key)
        _append({
            "schema_version": SCHEMA_VERSION,
            "event": "open",
            "id": finding_id,
            "severity": severity,
            "system": system,
            "check": check,
            "message": message,
            "evidence": evidence or {},
            "suggested_action": suggested_action,
            "run_date": today,
            "ts": _now_iso(),
        })
        try:
            print(f"[HEALTH] {severity} {finding_id}: {message}")
        except UnicodeEncodeError:
            print(f"[HEALTH] {severity} {finding_id}: "
                  f"{message.encode('ascii', errors='replace').decode('ascii')}")
        return finding_id
    except Exception as exc:  # emit() must never crash its caller
        try:
            print(f"[HEALTH] emit() itself failed ({exc}) while reporting "
                  f"{system}/{check}: {message}")
        except Exception:
            pass
        return make_id(system, check, key)


def resolve(system: str, check: str, *, key: str = "", run_date: str | None = None) -> str:
    """Append a 'resolve' event for the id emit() would have used. Call
    this on the success path right after a previously-guarded operation
    works, so a finding that recovers on its own closes itself instead of
    staying open forever (engine-emitted findings are not auto-resolved by
    the runner -- see docs/HEALTH_CONTRACT.md). A no-op, not an error, if
    the id was never open. Never raises."""
    try:
        today = run_date or _now_iso()[:10]
        finding_id = make_id(system, check, key)
        _append({
            "schema_version": SCHEMA_VERSION,
            "event": "resolve",
            "id": finding_id,
            "run_date": today,
            "ts": _now_iso(),
        })
        return finding_id
    except Exception as exc:
        try:
            print(f"[HEALTH] resolve() itself failed ({exc}) for {system}/{check}")
        except Exception:
            pass
        return make_id(system, check, key)
