#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health check runner. One function per check, registered at the bottom.
Writes data/health/health_report.json (current open findings + a per-check
pass/fail summary) and appends to data/health/health_events.jsonl. See
docs/HEALTH_CONTRACT.md for the full contract both files follow.

Every check is called through _run_one(), which catches ANY exception the
check raises and turns it into a CRITICAL "<name>_crashed" finding instead
of letting the runner itself die -- a check that can't run is itself a
health problem, not a reason to report nothing.

This script has no Telegram code and no knowledge of how findings get
delivered to a person. It only writes the two files above. A separate
consumer (not in this repo) is responsible for delivery.

Exit code: non-zero iff any CRITICAL finding is open after this run.

Env:
  STEP_OUTCOMES   optional, same format check_workflow_failures.py reads
                  ("Label=outcome,Label=outcome,...") -- when present
                  (i.e. this runner is invoked as a step inside one of the
                  daily workflows), check_step_failure() flags any step
                  that isn't success/skipped. Absent when run standalone
                  or from health_check.yml, where it's simply skipped.
  WORKFLOW_LABEL  optional, used only to label check_step_failure findings.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

import check_missed_runs as heartbeat  # noqa: E402 -- reuse OHLCV staleness / suppression

HEALTH_DIR = ROOT / "data" / "health"
REPORT_PATH = HEALTH_DIR / "health_report.json"
EVENTS_PATH = HEALTH_DIR / "health_events.jsonl"

TODAY = date.today()
TODAY_STR = TODAY.isoformat()

# ---------------------------------------------------------------------------
# System registry -- caps are read from each engine's own constants/config
# where one exists (see commit message / report for how these were sourced);
# `None` means "no cap configured for this system", not "cap is zero".
# ---------------------------------------------------------------------------
SYSTEMS = [
    # id, label, data_dir, equity_file, max_open, leverage_max, config_path
    dict(id="S1", label="Donchian",         data_dir=ROOT / "data" / "t9b_paper",
         equity_file="equity_curve.csv", max_open=8,  leverage_max=1.0, config_path=None),
    dict(id="S2", label="RSI-MR",           data_dir=ROOT / "data" / "t9b_mr_paper",
         equity_file="equity_curve.csv", max_open=10, leverage_max=1.0, config_path=None),
    dict(id="S3", label="ConsecDown",       data_dir=ROOT / "data" / "t9b_consecdowndays_paper",
         equity_file="engine_equity_curve.csv", max_open=None, leverage_max=1.0, config_path=None),
    dict(id="S5", label="Momentum",         data_dir=ROOT / "data" / "t9b_momentum_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=1.0, config_path=None),
    dict(id="S6", label="VolContraction",   data_dir=ROOT / "data" / "t9b_volcontraction_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=1.0, config_path=None),
    dict(id="S7", label="MACross",          data_dir=ROOT / "data" / "t9b_macross_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=1.0, config_path=None),
    dict(id="S8", label="RSI-MR-Funding",   data_dir=ROOT / "data" / "t9b_rsi_mr_funding_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=1.0, config_path=None),
    dict(id="Candidate12", label="Candidate 12", data_dir=ROOT / "data" / "t9_candidate12_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=None,
         config_path=ROOT / "data" / "research_candidate12_cross_sectional_momentum_t8" / "phase_t8_frozen_config.json"),
    dict(id="Candidate19", label="Candidate 19", data_dir=ROOT / "data" / "t9_candidate19_paper",
         equity_file="equity_curve.csv", max_open=None, leverage_max=None,
         config_path=ROOT / "data" / "research_candidate19_t8" / "phase_t8_frozen_config.json"),
]

FUNDING_DIR = ROOT / "data" / "futures_universe" / "funding_rates"
MAX_FUNDING_STALE_DAYS = 3        # same threshold already used for OHLCV staleness
KILL_SWITCH_DD_PCT = 35.0         # matches KILL_SWITCH_DD_PCT in every engine's own source
KILL_SWITCH_WARN_BAND_PCT = 10.0  # warn once within this many points of the kill-switch level
FROZEN_MIN_RUN_LENGTH = 5         # N identical consecutive values -> "frozen", not "quiet market"
STOPOUT_WINDOW = 20                # trailing N closed trades
STOPOUT_WARN_FRACTION = 0.60       # S7's actual rate over its worst week was ~0.73


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Finding construction + low-level CSV helpers
# ---------------------------------------------------------------------------

def finding(system: str, check: str, severity: str, message: str, *,
            key: str = "", evidence: dict[str, Any] | None = None,
            suggested_action: str = "") -> dict:
    return {
        "system": system,
        "check": check,
        "severity": severity,
        "message": message,
        "key": key,
        "evidence": evidence or {},
        "suggested_action": suggested_action,
    }


def _finding_id(f: dict) -> str:
    return f"{f['system']}:{f['check']}:{f['key']}" if f["key"] else f"{f['system']}:{f['check']}"


def read_rows(path: Path) -> list[list[str]]:
    """Every row as a raw list of strings, via csv.reader -- tolerant of
    ragged rows (unlike pd.read_csv's default C parser), which several of
    these log files legitimately have (different event types log different
    column counts). Returns [] if the file doesn't exist or can't be read
    at all."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return list(csv.reader(fh))
    except Exception:
        return []


def _col(header: list[str], name: str) -> int | None:
    try:
        return header.index(name)
    except ValueError:
        return None


def event_rows(data_dir: Path, filename: str = "daily_log.csv") -> list[dict]:
    """Parse an event-log CSV (daily_log.csv) row by row using its own
    header for column positions -- works even for the systems whose file
    pd.read_csv's default parser can't handle (confirmed: S2, S3, S6, S8,
    as of 2026-10-10 -- ragged rows, not corruption: SIGNAL_SKIPPED rows
    are narrower than ENTRY/EXIT rows by design). Returns a dict per row
    with whatever named columns that row actually has; short rows simply
    lack the later keys rather than raising."""
    rows = read_rows(data_dir / filename)
    if len(rows) < 2:
        return []
    header = rows[0]
    out = []
    for r in rows[1:]:
        if r == header:
            continue  # a literal duplicate header row -- not a data row
        d = {header[i]: r[i] for i in range(min(len(header), len(r)))}
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# DATA checks
# ---------------------------------------------------------------------------

def check_ohlcv_staleness_spot() -> list[dict]:
    allowlist = heartbeat._spot_universe()
    suppress = heartbeat._suppressed_symbols()
    result = heartbeat.check_ohlcv_staleness(
        "spot universe (S1/S3)", ROOT / "data" / "universe" / "ohlcv_1d",
        TODAY, allowlist=allowlist, suppress=suppress)
    if result is None:
        return []
    msg, worst_days = result
    return [finding("data/universe", "ohlcv_staleness_spot", "CRITICAL", msg,
                     evidence={"worst_days": worst_days},
                     suggested_action="Run engines/spot_data_refresh.py; "
                                      "check for a halted symbol needing suppression.")]


def check_ohlcv_staleness_futures() -> list[dict]:
    suppress = heartbeat._suppressed_symbols()
    result = heartbeat.check_ohlcv_staleness(
        "futures universe (S2/S5-S8/candidates)", ROOT / "data" / "futures_universe" / "ohlcv_1d",
        TODAY, allowlist=None, suppress=suppress)
    if result is None:
        return []
    msg, worst_days = result
    return [finding("data/futures_universe", "ohlcv_staleness_futures", "CRITICAL", msg,
                     evidence={"worst_days": worst_days},
                     suggested_action="Run .github/scripts/refresh_futures_data.py; "
                                      "check for a halted symbol needing suppression.")]


def check_funding_staleness() -> list[dict]:
    if not FUNDING_DIR.exists():
        return [finding("data/futures_universe/funding_rates", "funding_staleness", "CRITICAL",
                         f"funding cache dir missing ({FUNDING_DIR})",
                         suggested_action="Check refresh_futures_data.py / the funding refresh job.")]
    ages: dict[str, int] = {}
    for f in FUNDING_DIR.glob("*_funding.csv"):
        sym = f.stem.replace("_funding", "")
        try:
            df = pd.read_csv(f, usecols=["funding_time"])
            last_ms = int(df["funding_time"].iloc[-1])
            last_date = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).date()
        except Exception:
            continue
        ages[sym] = (TODAY - last_date).days
    if not ages or max(ages.values()) <= MAX_FUNDING_STALE_DAYS:
        return []
    worst_sym = max(ages, key=ages.get)
    worst_days = ages[worst_sym]
    sorted_ages = sorted(ages.values())
    median_days = sorted_ages[len(sorted_ages) // 2]
    n_bulk = sum(1 for v in ages.values() if v == median_days)
    return [finding(
        "data/futures_universe/funding_rates", "funding_staleness", "CRITICAL",
        f"funding_rates: {n_bulk}/{len(ages)} symbols are {median_days}d behind "
        f"(a mass freeze, not a few stragglers); worst is {worst_sym} at {worst_days}d",
        evidence={"worst_symbol": worst_sym, "worst_days": worst_days,
                  "median_days": median_days, "n_at_median": n_bulk, "n_checked": len(ages)},
        suggested_action="Funding refresh needs a reachable (non-US) host -- "
                          "GH runners are geo-blocked from fapi.binance.com. "
                          "Feeds S6/S7 gates, S8's filter, and the regime funding axis.",
    )]


def check_frozen_values() -> list[dict]:
    """A column that reports the exact same value for FROZEN_MIN_RUN_LENGTH+
    consecutive rows isn't necessarily broken (a genuinely quiet funding
    market can do this for a day or two) but regime_history.csv's
    funding_avg sitting at the identical float for a week straight (as
    happened 2026-09-13..19, traced to the frozen funding cache feeding it)
    is the exact shape this check exists to catch."""
    path = ROOT / "data" / "regime_history.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    out = []
    for col in ("funding_avg", "breadth", "vol_mult"):
        if col not in df.columns or len(df) < FROZEN_MIN_RUN_LENGTH:
            continue
        vals = df[col].tail(30).tolist()
        run_len = 1
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] == vals[i - 1]:
                run_len += 1
            else:
                break
        if run_len >= FROZEN_MIN_RUN_LENGTH:
            out.append(finding(
                "data/regime_history.csv", "frozen_value_detection", "WARNING",
                f"regime_history.{col} has been exactly {vals[-1]} for the last "
                f"{run_len} rows -- check the upstream feed, not just this file",
                key=col, evidence={"column": col, "value": vals[-1], "run_length": run_len},
                suggested_action="Check whether the upstream source (e.g. funding cache) is stale.",
            ))
    return out


_TS_1970_RE = re.compile(r"1970-01-01|^0*,|,0*,")


def check_timestamps_1970() -> list[dict]:
    """Residual 1970-01-01 rows from the original OHLCV-freeze bug (the
    int64 cast on a datetime64[s] column, fixed going forward in
    spot_data_refresh.py) that were never purged from files written before
    the fix. Scoped to the spot universe, where the bug lived."""
    out = []
    spot_dir = ROOT / "data" / "universe" / "ohlcv_1d"
    if not spot_dir.exists():
        return out
    hit_files = []
    for f in spot_dir.glob("*_1d.csv"):
        try:
            with open(f, encoding="utf-8") as fh:
                next(fh, None)  # header
                for line in fh:
                    if line.startswith("1970-01-01"):
                        hit_files.append(f.stem)
                        break
        except Exception:
            continue
    if hit_files:
        out.append(finding(
            "data/universe/ohlcv_1d", "timestamps_1970", "WARNING",
            f"{len(hit_files)} spot OHLCV file(s) still carry a residual 1970-01-01 row "
            f"from the pre-fix era (first few: {sorted(hit_files)[:5]})",
            evidence={"n_files": len(hit_files), "symbols": sorted(hit_files)},
            suggested_action="One-time cleanup: drop any row with a 1970-01-01 timestamp "
                              "from these files. Does not affect current trading (sits "
                              "before all real history) but corrupts anything computing "
                              "min(date) or full-history stats.",
        ))
    return out


def check_symbol_gaps() -> list[dict]:
    """A symbol's OHLCV file missing one or more calendar days inside its
    own covered range (as opposed to being stale at the end, which
    check_ohlcv_staleness_* already covers). Scoped to the spot universe
    actually traded by S1/S3 (same allowlist check_ohlcv_staleness_spot
    uses) -- the full data/universe/ohlcv_1d/ directory also holds long-
    delisted symbols (e.g. FTT, frozen since the exchange collapse) whose
    "gaps" are just history nobody will ever trade, not a live problem.
    Futures isn't scoped the same way (no comparable allowlist exists for
    it yet). Ignores the first 7 days of a file's own history (new
    listings can have a short real gap right at IPO)."""
    out = []
    spot_allowlist = heartbeat._spot_universe()
    for label, cache_dir, date_col, allowlist in (
        ("spot universe", ROOT / "data" / "universe" / "ohlcv_1d", "time", spot_allowlist),
        ("futures universe", ROOT / "data" / "futures_universe" / "ohlcv_1d", "date", None),
    ):
        if not cache_dir.exists():
            continue
        worst_sym, worst_gaps = None, 0
        for f in cache_dir.glob("*_1d.csv"):
            sym = f.stem.replace("_1d", "")
            if allowlist is not None and sym not in allowlist:
                continue
            try:
                df = pd.read_csv(f, usecols=[date_col])
                dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.date.dropna()
                dates = sorted(set(dates))
            except Exception:
                continue
            if len(dates) < 10:
                continue
            dates = dates[7:]  # skip near-IPO noise
            if len(dates) < 2:
                continue
            full = set(pd.date_range(dates[0], dates[-1], freq="D").date)
            gaps = len(full - set(dates))
            if gaps > worst_gaps:
                worst_sym, worst_gaps = f.stem, gaps
        if worst_gaps > 2:  # a couple of missing days can be a legit exchange hiccup
            out.append(finding(
                label.replace(" ", "_"), "symbol_gaps", "WARNING",
                f"{label}: {worst_sym} has {worst_gaps} missing calendar day(s) inside "
                f"its own covered history",
                key=str(worst_sym), evidence={"symbol": worst_sym, "n_gaps": worst_gaps},
                suggested_action="Re-backfill this symbol from the CDN; check for an "
                                  "exchange halt around the gap date(s).",
            ))
    return out


# ---------------------------------------------------------------------------
# RUNS checks
# ---------------------------------------------------------------------------

def check_heartbeat_36h() -> list[dict]:
    out = []
    for name, rel_path in heartbeat.SYSTEMS:
        ts = heartbeat.last_commit_time(rel_path)
        if ts is None:
            out.append(finding(name, "heartbeat_36h", "CRITICAL",
                                f"{rel_path}: no commit history found",
                                suggested_action="Confirm the daily workflow for this "
                                                  "system is enabled and fired."))
            continue
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        if age_hours > heartbeat.MAX_STALE_HOURS:
            out.append(finding(name, "heartbeat_36h", "CRITICAL",
                                f"{age_hours:.1f}h since last real state.json commit "
                                f"(threshold {heartbeat.MAX_STALE_HOURS:.0f}h)",
                                evidence={"age_hours": round(age_hours, 1)},
                                suggested_action="Check the daily workflow's Actions log."))
    return out


def check_missed_run_single_day() -> list[dict]:
    """The 36h heartbeat can't see a single missed day (36h < 2 daily
    cycles). This compares each system's actual last_run_date inside its
    own state.json to what it should be by now.

    Every engine defaults to processing "yesterday" (run day D -> run_date
    D-1), so the floor this check uses depends on whether today's cron has
    already fired when this runs:

    - HEALTH_SAME_DAY_CHECK=1 (set by health_check.yml, scheduled hours
      after the 08:00 UTC daily crons, and by the in-workflow invocations
      in t9b_daily.yml / t9_candidates_daily.yml, which run after their own
      commit step on the same calendar day): floor is TODAY-1 -- today's
      run_date should already be committed.
    - unset (manual/ad-hoc runs, where "has today's cron fired yet" is
      unknown): floor is TODAY-2, the value that's correct regardless of
      time of day. This can't catch a same-day single miss, only an
      established one, but a tighter floor here would false-positive on
      every manual run before the daily cron's usual time -- as it did
      when first tested on 2026-10-10.

    Either way, a miss caught THIS way is a real-time catch. Once a second
    day's run has quietly advanced last_run_date past the skipped date (the
    engine processes "yesterday" relative to whenever it next runs, not
    relative to its own last state), the gap leaves no trace here --
    confirmed against the unexplained 2026-09-29 candidates miss, which by
    the next day looked identical to a normal one-day lag. Catching that
    case after the fact needs the catch-up mechanism proposed separately,
    not this check.
    """
    out = []
    floor_offset = 1 if os.environ.get("HEALTH_SAME_DAY_CHECK") else 2
    expected = (TODAY - timedelta(days=floor_offset)).isoformat()
    for sysconf in SYSTEMS:
        state_path = sysconf["data_dir"] / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        last_run = state.get("last_run_date")
        if not last_run:
            continue
        try:
            gap_days = (date.fromisoformat(expected) - date.fromisoformat(last_run)).days
        except Exception:
            continue
        if gap_days == 1:
            out.append(finding(
                sysconf["id"], "missed_run_single_day", "WARNING",
                f"last_run_date={last_run}, expected {expected} -- one day appears "
                f"to have been skipped (not yet 36h stale, so heartbeat_36h won't catch it)",
                evidence={"last_run_date": last_run, "expected": expected},
                suggested_action="Check this workflow's Actions log for the missed date; "
                                  "the engine will NOT automatically backfill it (see "
                                  "catch-up proposal in docs/HEALTH_CONTRACT.md follow-ups).",
            ))
        elif gap_days > 1:
            out.append(finding(
                sysconf["id"], "missed_run_single_day", "CRITICAL",
                f"last_run_date={last_run}, expected {expected} -- {gap_days} day(s) behind",
                evidence={"last_run_date": last_run, "expected": expected, "gap_days": gap_days},
                suggested_action="Multiple days missed -- check the workflow's recent "
                                  "Actions runs, not just today's.",
            ))
    return out


def check_step_failure() -> list[dict]:
    """Only meaningful when invoked as a step inside a daily workflow with
    STEP_OUTCOMES set (same env var check_workflow_failures.py reads).
    Standalone / health_check.yml runs simply have nothing to check here."""
    raw = os.environ.get("STEP_OUTCOMES", "")
    if not raw:
        return []
    label = os.environ.get("WORKFLOW_LABEL", "workflow")
    out = []
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, outcome = (p.strip() for p in pair.split("=", 1))
        if outcome and outcome not in ("success", "skipped"):
            out.append(finding(
                label, "step_failure", "CRITICAL", f"step '{name}' outcome={outcome}",
                key=name, evidence={"step": name, "outcome": outcome},
                suggested_action="See this run's Actions log for the step's own error.",
            ))
    return out


# ---------------------------------------------------------------------------
# FILES checks
# ---------------------------------------------------------------------------

# Files expected to have ONE consistent column count on every row.
_FIXED_SCHEMA_FILES = ["equity_curve.csv", "engine_equity_curve.csv",
                        "open_positions.csv", "mtm_positions.csv", "signals_today.csv"]

_KNOWN_EQUITY_COLUMNS = ({"paper_equity"}, {"equity"})  # recognized equity_curve.csv schemas


def check_csv_integrity() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        data_dir = sysconf["data_dir"]
        if not data_dir.exists():
            continue

        for fname in _FIXED_SCHEMA_FILES:
            rows = read_rows(data_dir / fname)
            if len(rows) < 2:
                continue
            header = rows[0]
            dupe_header_at = [i for i, r in enumerate(rows[1:], start=1) if r == header]
            if dupe_header_at:
                out.append(finding(
                    sysconf["id"], "csv_duplicate_header", "WARNING",
                    f"{fname}: header row repeated at line(s) {dupe_header_at[:5]}",
                    key=fname, evidence={"file": fname, "rows": dupe_header_at[:10]},
                    suggested_action=f"De-duplicate {fname}; a repeated header usually "
                                      f"means an append without a header-exists check.",
                ))
            ragged = [i for i, r in enumerate(rows[1:], start=1)
                      if len(r) != len(header) and r != header]
            if ragged:
                out.append(finding(
                    sysconf["id"], "csv_ragged_rows", "WARNING",
                    f"{fname}: {len(ragged)} row(s) don't match the {len(header)}-column "
                    f"header (first bad line: {ragged[0] + 1})",
                    key=fname, evidence={"file": fname, "n_bad": len(ragged),
                                          "first_bad_line": ragged[0] + 1},
                    suggested_action=f"{fname} is expected to be fixed-schema -- "
                                      f"inspect line {ragged[0] + 1} directly.",
                ))

            if fname in ("equity_curve.csv", "engine_equity_curve.csv"):
                cols = set(header)
                if not any(known <= cols for known in _KNOWN_EQUITY_COLUMNS):
                    out.append(finding(
                        sysconf["id"], "equity_curve_unknown_schema", "WARNING",
                        f"{fname}: columns {header} match neither recognized equity_curve "
                        f"schema ({_KNOWN_EQUITY_COLUMNS}) -- equity_reconciliation will "
                        f"skip this system",
                        key=fname, evidence={"file": fname, "columns": header},
                        suggested_action="Confirm whether this is a deliberate alternate "
                                          "schema (e.g. S5's REBAL-event format) or drift.",
                    ))

        # daily_log.csv: a heterogeneous event log by design -- only flag a
        # literal duplicate header (real corruption) or total unparseability
        # via pandas' default reader, since several consumer scripts still
        # call plain pd.read_csv() on it and silently lose data when it fails.
        dl_rows = read_rows(data_dir / "daily_log.csv")
        if len(dl_rows) >= 2:
            header = dl_rows[0]
            dupe_header_at = [i for i, r in enumerate(dl_rows[1:], start=1) if r == header]
            if dupe_header_at:
                out.append(finding(
                    sysconf["id"], "csv_duplicate_header", "WARNING",
                    f"daily_log.csv: header row repeated at line(s) {dupe_header_at[:5]}",
                    key="daily_log.csv",
                    evidence={"file": "daily_log.csv", "rows": dupe_header_at[:10]},
                    suggested_action="De-duplicate daily_log.csv.",
                ))
            try:
                pd.read_csv(data_dir / "daily_log.csv")
            except Exception as exc:
                out.append(finding(
                    sysconf["id"], "csv_not_pandas_parseable", "WARNING",
                    f"daily_log.csv: plain pd.read_csv() fails ({exc}) -- any consumer "
                    f"using it without on_bad_lines='skip' silently gets nothing",
                    key="daily_log.csv", evidence={"file": "daily_log.csv", "error": str(exc)},
                    suggested_action="Confirm every reader of this file passes "
                                      "on_bad_lines='skip' (auto_execute.py already does; "
                                      "send_telegram.py / create_signal_issues.py now do too).",
                ))
    return out


def check_json_integrity() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        path = sysconf["data_dir"] / "state.json"
        if not path.exists():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            out.append(finding(sysconf["id"], "json_not_parseable", "CRITICAL",
                                f"state.json does not parse: {exc}",
                                suggested_action="This system's state is unreadable -- "
                                                  "treat as a halted system until fixed."))
            continue
        for required in ("last_run_date", "paper_equity_usdt"):
            if required not in state:
                out.append(finding(sysconf["id"], "json_missing_field", "WARNING",
                                    f"state.json is missing expected field '{required}'",
                                    key=required, evidence={"field": required},
                                    suggested_action="Confirm this system's engine still "
                                                      "writes this field on every save."))
    return out


def check_regime_history_integrity() -> list[dict]:
    path = ROOT / "data" / "regime_history.csv"
    if not path.exists():
        return [finding("data/regime_history.csv", "regime_history_missing", "CRITICAL",
                         "regime_history.csv does not exist")]
    try:
        df = pd.read_csv(path)
        dates = pd.to_datetime(df["date"]).dt.date
    except Exception as exc:
        return [finding("data/regime_history.csv", "regime_history_unparseable", "CRITICAL",
                         f"regime_history.csv does not parse: {exc}")]
    out = []
    dupes = sorted({str(d) for d in dates[dates.duplicated(keep=False)]})
    if dupes:
        out.append(finding(
            "data/regime_history.csv", "regime_history_duplicate_dates", "WARNING",
            f"{len(dupes)} duplicated date(s): {dupes[:10]}",
            evidence={"dates": dupes},
            suggested_action="get_regime_weight() already resolves duplicates via "
                              "keep='last', but a duplicate row means something re-ran "
                              "a historical date out of order -- find out what.",
        ))
    if len(dates) >= 2:
        full_range = set(pd.date_range(dates.min(), dates.max(), freq="D").date)
        missing = sorted(str(d) for d in (full_range - set(dates)))
        if missing:
            out.append(finding(
                "data/regime_history.csv", "regime_history_missing_dates", "WARNING",
                f"{len(missing)} date(s) missing inside the file's own covered range: "
                f"{missing[:10]}",
                evidence={"dates": missing},
                suggested_action="Any T9B/candidate run on a missing date fell back to "
                                  "equal weight via get_regime_weight()'s own fallback -- "
                                  "rebuild these rows via the date-sliced re-simulation.",
            ))
    return out


# ---------------------------------------------------------------------------
# INVARIANTS checks
# ---------------------------------------------------------------------------

def check_drawdown_vs_kill_switch() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        state_path = sysconf["data_dir"] / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dd = state.get("drawdown_pct")
        if dd is None:
            continue
        dd = abs(float(dd))
        killed = bool(state.get("kill_switch_triggered", False))
        if dd >= KILL_SWITCH_DD_PCT and not killed:
            out.append(finding(
                sysconf["id"], "drawdown_vs_kill_switch", "CRITICAL",
                f"drawdown {dd:.2f}% >= kill-switch level {KILL_SWITCH_DD_PCT}% but "
                f"kill_switch_triggered is not set",
                evidence={"drawdown_pct": dd, "threshold": KILL_SWITCH_DD_PCT},
                suggested_action="The engine's own kill-switch should have fired on its "
                                  "next run -- if it hasn't by tomorrow, investigate why.",
            ))
        elif dd >= KILL_SWITCH_DD_PCT - KILL_SWITCH_WARN_BAND_PCT:
            out.append(finding(
                sysconf["id"], "drawdown_approaching_kill_switch", "WARNING",
                f"drawdown {dd:.2f}% is within {KILL_SWITCH_DD_PCT - dd:.1f} points of "
                f"the {KILL_SWITCH_DD_PCT}% kill-switch level",
                evidence={"drawdown_pct": dd, "threshold": KILL_SWITCH_DD_PCT},
                suggested_action="No action required yet -- worth watching.",
            ))
    return out


def check_no_kill_switch_configured() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        state_path = sysconf["data_dir"] / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "kill_switch_triggered" not in state:
            out.append(finding(
                sysconf["id"], "no_kill_switch_configured", "WARNING",
                "state.json has no kill_switch_triggered field -- this system has no "
                "safety net, regardless of what its drawdown currently is",
                suggested_action="Confirm whether the engine code checks "
                                  "KILL_SWITCH_DD_PCT but never persists the field, or "
                                  "never checks it at all; propose a threshold consistent "
                                  "with the other systems (35% DD) before applying.",
            ))
    return out


def check_position_cap() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        max_open = sysconf["max_open"]
        if sysconf["config_path"] is not None and sysconf["config_path"].exists():
            try:
                cfg = json.loads(sysconf["config_path"].read_text(encoding="utf-8"))
                max_open = cfg.get("max_open_positions", max_open)
            except Exception:
                pass
        if max_open is None:
            continue
        state_path = sysconf["data_dir"] / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        op = state.get("open_positions")
        if op is None:
            op = (state.get("long_positions") or []) + (state.get("short_positions") or [])
        n_open = len(op)
        if n_open > max_open:
            out.append(finding(
                sysconf["id"], "position_cap_breach", "CRITICAL",
                f"{n_open} open positions exceeds the configured cap of {max_open}",
                evidence={"n_open": n_open, "max_open": max_open},
                suggested_action="Check whether a recent code change relaxed the cap "
                                  "intentionally, or positions are being opened past it.",
            ))
    return out


def check_leverage_cap() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        leverage_max = sysconf["leverage_max"]
        if sysconf["config_path"] is not None and sysconf["config_path"].exists():
            try:
                cfg = json.loads(sysconf["config_path"].read_text(encoding="utf-8"))
                leverage_max = cfg.get("leverage_max", leverage_max)
            except Exception:
                pass
        if leverage_max is None:
            continue
        state_path = sysconf["data_dir"] / "state.json"
        pos_path = sysconf["data_dir"] / "open_positions.csv"
        if not state_path.exists() or not pos_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            equity = float(state.get("paper_equity_usdt", 0) or 0)
            positions = pd.read_csv(pos_path)
        except Exception:
            continue
        if equity <= 0 or positions.empty:
            continue
        if "notional_usdt" in positions.columns:
            total_notional = pd.to_numeric(positions["notional_usdt"], errors="coerce").fillna(0).sum()
        elif {"qty", "entry_price"} <= set(positions.columns):
            # No notional_usdt column (e.g. S2) -- derive it. This is also
            # what caught S2's COSUSDT position actually sizing itself to
            # ~$115k notional on a $10k account (qty=236.9M at entry_price
            # =$0.000485, risk_amount=$35.53 but stop_loss only 0.0000002
            # away from entry -- the sizing formula divides by that near-
            # zero stop distance and the result never got capped).
            qty = pd.to_numeric(positions["qty"], errors="coerce").fillna(0)
            price = pd.to_numeric(positions["entry_price"], errors="coerce").fillna(0)
            total_notional = float((qty * price).sum())
        else:
            continue
        implied_leverage = total_notional / equity
        if implied_leverage > leverage_max * 1.05:  # 5% slack for intraday mark noise
            out.append(finding(
                sysconf["id"], "leverage_cap_breach", "CRITICAL",
                f"implied leverage {implied_leverage:.2f}x exceeds configured cap "
                f"{leverage_max:.2f}x (notional ${total_notional:,.0f} / equity ${equity:,.0f})",
                evidence={"implied_leverage": round(implied_leverage, 3),
                          "leverage_max": leverage_max, "total_notional": total_notional,
                          "equity": equity},
                suggested_action="Check position sizing -- a leverage breach is a direct "
                                  "liquidation-risk increase versus the T6 sweep this cap "
                                  "was set from.",
            ))
    return out


def check_equity_reconciliation() -> list[dict]:
    """state.json's paper_equity_usdt vs. the last row of this system's own
    equity_curve.csv -- these are written by different scripts (the engine
    vs. mark_to_market.py) and have drifted silently before (S2's wrong
    OHLCV directory meant mark_to_market computed $0 unrealized for a month
    while state.json kept moving). Skips systems whose equity_curve.csv
    doesn't match a recognized schema (flagged separately, see
    check_csv_integrity's equity_curve_unknown_schema)."""
    out = []
    for sysconf in SYSTEMS:
        state_path = sysconf["data_dir"] / "state.json"
        eq_path = sysconf["data_dir"] / sysconf["equity_file"]
        if not state_path.exists() or not eq_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            eq_df = pd.read_csv(eq_path)
        except Exception:
            continue
        if eq_df.empty:
            continue
        eq_col = "paper_equity" if "paper_equity" in eq_df.columns else (
            "equity" if "equity" in eq_df.columns else None)
        if eq_col is None:
            continue
        state_equity = state.get("paper_equity_usdt")
        if state_equity is None:
            continue
        curve_equity = float(pd.to_numeric(eq_df[eq_col], errors="coerce").dropna().iloc[-1])
        diff = abs(float(state_equity) - curve_equity)
        tolerance = max(1.0, 0.005 * abs(curve_equity))  # $1 floor or 0.5%, whichever's bigger
        if diff > tolerance:
            out.append(finding(
                sysconf["id"], "equity_reconciliation", "WARNING",
                f"state.json paper_equity_usdt=${state_equity:,.2f} vs. "
                f"{sysconf['equity_file']}'s last {eq_col}=${curve_equity:,.2f} "
                f"(diff ${diff:,.2f})",
                evidence={"state_equity": state_equity, "curve_equity": curve_equity,
                          "diff": diff},
                suggested_action=f"Check whether {sysconf['equity_file']} and state.json "
                                  f"are being written by the same run, or one is stale.",
            ))
    return out


# ---------------------------------------------------------------------------
# BEHAVIOR checks
# ---------------------------------------------------------------------------

_REASON_RE = re.compile(r"reason=([\w]+)")


def _exit_reasons(data_dir: Path, window_days: int | None = None) -> list[tuple[str, str]]:
    """[(run_date, reason), ...] for every EXIT event in daily_log.csv,
    using event_rows() so this works for the systems whose file plain
    pd.read_csv can't parse."""
    rows = event_rows(data_dir, "daily_log.csv")
    out = []
    for r in rows:
        if r.get("event") != "EXIT":
            continue
        detail = r.get("detail", "")
        m = _REASON_RE.search(detail or "")
        reason = m.group(1) if m else "unknown"
        out.append((r.get("run_date", ""), reason))
    if window_days is not None:
        cutoff = (TODAY - timedelta(days=window_days)).isoformat()
        out = [(d, r) for d, r in out if d >= cutoff]
    return out


def check_stopout_rate_elevated() -> list[dict]:
    out = []
    for sysconf in SYSTEMS:
        if not (sysconf["data_dir"] / "daily_log.csv").exists():
            continue
        exits = _exit_reasons(sysconf["data_dir"])
        if len(exits) < 5:
            continue
        window = exits[-STOPOUT_WINDOW:]
        n_stop = sum(1 for _, reason in window if "stop" in reason.lower())
        frac = n_stop / len(window)
        if frac >= STOPOUT_WARN_FRACTION:
            out.append(finding(
                sysconf["id"], "stopout_rate_elevated", "WARNING",
                f"{n_stop}/{len(window)} of the last {len(window)} exits were stop-outs "
                f"({frac:.0%}) -- the S7 rally-whipsaw pattern",
                evidence={"n_stop": n_stop, "n_window": len(window), "fraction": round(frac, 3)},
                suggested_action="Shadow-replay this system's recent entries against "
                                  "current (non-stale) funding/regime data before changing "
                                  "any stop rule.",
            ))
    return out


def check_no_entries_exits_despite_signals(window_days: int = 7) -> list[dict]:
    """A system logging SIGNAL_SKIPPED (i.e. it's seeing candidates) but
    zero ENTRY/EXIT for window_days straight, while it still has open
    capacity, suggests something is silently suppressing it -- the exact
    shape of the frozen-OHLCV incident, generalized."""
    out = []
    cutoff = (TODAY - timedelta(days=window_days)).isoformat()
    for sysconf in SYSTEMS:
        if not (sysconf["data_dir"] / "daily_log.csv").exists():
            continue
        rows = event_rows(sysconf["data_dir"])
        recent = [r for r in rows if (r.get("run_date") or "") >= cutoff]
        if not recent:
            continue
        n_signals = sum(1 for r in recent if r.get("event") in ("SIGNAL_SKIPPED", "DETECTED"))
        n_actions = sum(1 for r in recent if r.get("event") in ("ENTRY", "EXIT", "REBAL"))
        if n_signals >= 3 and n_actions == 0:
            out.append(finding(
                sysconf["id"], "no_entries_exits_despite_signals", "WARNING",
                f"{n_signals} signal(s) seen in the last {window_days}d but zero "
                f"ENTRY/EXIT/REBAL events logged",
                evidence={"n_signals": n_signals, "window_days": window_days},
                suggested_action="Check the arbitrator / cross-system suppression and the "
                                  "regime weight for this system -- confirm it isn't being "
                                  "silently zeroed out.",
            ))
    return out


# ---------------------------------------------------------------------------
# Suppression-list meta-check
# ---------------------------------------------------------------------------

def check_suppression_list_stale() -> list[dict]:
    path = ROOT / "data" / "halted_symbols_suppression.json"
    if not path.exists():
        return []
    try:
        suppression = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for key, entry in suppression.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        last_verified = entry.get("last_verified")
        if not last_verified:
            out.append(finding("data/halted_symbols_suppression.json", "suppression_never_verified",
                                "WARNING", f"{key}: no last_verified timestamp on record",
                                key=key, suggested_action="Run the monthly re-check."))
            continue
        try:
            age_days = (TODAY - date.fromisoformat(last_verified)).days
        except Exception:
            continue
        if age_days > 40:  # monthly check + buffer
            out.append(finding(
                "data/halted_symbols_suppression.json", "suppression_list_stale", "WARNING",
                f"{key}: last re-verified {age_days}d ago (monthly check appears to have "
                f"stopped running)",
                key=key, evidence={"symbol": key, "age_days": age_days},
                suggested_action="Check monthly_halted_symbol_check.yml's recent runs.",
            ))
    return out


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

REGISTRY: list[tuple[str, Callable[[], list[dict]]]] = [
    ("ohlcv_staleness_spot", check_ohlcv_staleness_spot),
    ("ohlcv_staleness_futures", check_ohlcv_staleness_futures),
    ("funding_staleness", check_funding_staleness),
    ("frozen_value_detection", check_frozen_values),
    ("timestamps_1970", check_timestamps_1970),
    ("symbol_gaps", check_symbol_gaps),
    ("heartbeat_36h", check_heartbeat_36h),
    ("missed_run_single_day", check_missed_run_single_day),
    ("step_failure", check_step_failure),
    ("csv_integrity", check_csv_integrity),
    ("json_integrity", check_json_integrity),
    ("regime_history_integrity", check_regime_history_integrity),
    ("drawdown_vs_kill_switch", check_drawdown_vs_kill_switch),
    ("no_kill_switch_configured", check_no_kill_switch_configured),
    ("position_cap", check_position_cap),
    ("leverage_cap", check_leverage_cap),
    ("equity_reconciliation", check_equity_reconciliation),
    ("stopout_rate_elevated", check_stopout_rate_elevated),
    ("no_entries_exits_despite_signals", check_no_entries_exits_despite_signals),
    ("suppression_list_stale", check_suppression_list_stale),
]

# check names this runner can authoritatively resolve, because it has a
# function that re-derives the condition fresh every run. Any other id in
# health_events.jsonl (engine-emitted via health_emit.emit()) is left alone.
REGISTRY_CHECK_NAMES = {name for name, _ in REGISTRY} | {f"{name}_crashed" for name, _ in REGISTRY}


def _run_one(name: str, fn: Callable[[], list[dict]]) -> tuple[list[dict], str]:
    try:
        results = fn()
        return results, "pass" if not results else "fail"
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        _safe_print(f"[HEALTH] check '{name}' crashed: {exc}\n{tb}")
        crashed = [finding(
            "runner", f"{name}_crashed", "CRITICAL",
            f"check '{name}' raised {type(exc).__name__}: {exc}",
            evidence={"traceback": tb},
            suggested_action=f"Fix or temporarily disable the '{name}' check itself -- "
                              f"it did not produce a real finding this run.",
        )]
        return crashed, "fail"


def _load_events() -> list[dict]:
    if not EVENTS_PATH.exists():
        return []
    events = []
    with open(EVENTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _append_events(events: list[dict]) -> None:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, default=str) + "\n")


def _reconcile(prior_events: list[dict], this_run_findings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (new_events_to_append, current_open_findings_for_report)."""
    now = datetime.now(timezone.utc).isoformat()

    # Replay history: per id, track the running "open streak" start and
    # whether it's currently open.
    by_id: dict[str, dict] = {}
    for e in prior_events:
        fid = e.get("id")
        if not fid:
            continue
        st = by_id.setdefault(fid, {"first_seen": None, "last_seen": None, "open": False,
                                     "latest": None})
        if e.get("event") == "open":
            if st["first_seen"] is None:
                st["first_seen"] = e.get("run_date")
            st["last_seen"] = e.get("run_date")
            st["open"] = True
            st["latest"] = e
        elif e.get("event") == "resolve":
            st["open"] = False
            st["first_seen"] = None

    prior_open_ids = {fid for fid, st in by_id.items() if st["open"]}

    this_run_ids = set()
    new_events = []
    for f in this_run_findings:
        fid = _finding_id(f)
        this_run_ids.add(fid)
        event = {
            "event": "open", "id": fid, "severity": f["severity"], "system": f["system"],
            "check": f["check"], "message": f["message"], "evidence": f["evidence"],
            "suggested_action": f["suggested_action"], "run_date": TODAY_STR, "ts": now,
        }
        new_events.append(event)
        st = by_id.setdefault(fid, {"first_seen": None, "last_seen": None, "open": False, "latest": None})
        if st["first_seen"] is None:
            st["first_seen"] = TODAY_STR
        st["last_seen"] = TODAY_STR
        st["open"] = True
        st["latest"] = event

    # Auto-resolve: ids this runner owns, were open before, not re-opened now.
    for fid in prior_open_ids - this_run_ids:
        check_name = by_id[fid]["latest"].get("check") if by_id[fid]["latest"] else None
        if check_name in REGISTRY_CHECK_NAMES:
            new_events.append({"event": "resolve", "id": fid, "run_date": TODAY_STR, "ts": now})
            by_id[fid]["open"] = False
            by_id[fid]["first_seen"] = None

    current_open = []
    for fid, st in by_id.items():
        if not st["open"] or st["latest"] is None:
            continue
        latest = st["latest"]
        current_open.append({
            "id": fid,
            "severity": latest["severity"],
            "system": latest["system"],
            "check": latest["check"],
            "message": latest["message"],
            "first_seen": st["first_seen"],
            "last_seen": st["last_seen"],
            "status": "open",
            "evidence": latest.get("evidence", {}),
            "suggested_action": latest.get("suggested_action", ""),
            "run_date": latest["run_date"],
        })

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    current_open.sort(key=lambda f: (severity_order.get(f["severity"], 3), f["id"]))
    return new_events, current_open


def main() -> int:
    prior_events = _load_events()

    this_run_findings: list[dict] = []
    checks_summary: dict[str, dict] = {}

    for name, fn in REGISTRY:
        results, status = _run_one(name, fn)
        this_run_findings.extend(results)
        checks_summary[name] = {"status": status, "n_findings": len(results)}

    new_events, current_open = _reconcile(prior_events, this_run_findings)
    _append_events(new_events)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks_summary,
        "findings": current_open,
    }
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    n_critical = sum(1 for f in current_open if f["severity"] == "CRITICAL")
    n_warning = sum(1 for f in current_open if f["severity"] == "WARNING")
    _safe_print(f"[HEALTH] {len(current_open)} open finding(s): "
                f"{n_critical} CRITICAL, {n_warning} WARNING")
    for f in current_open:
        _safe_print(f"  [{f['severity']}] {f['id']}: {f['message']}")

    return 1 if n_critical else 0


if __name__ == "__main__":
    sys.exit(main())
