"""
M5 Missed Rebalancing Recovery
===============================
Detects and recovers missed Monday rebalancings.
Called daily at 06:00 via run_daily.bat, AFTER the data pipeline has run.

Safety rules enforced here:
- M1 incremental MUST succeed before any M2 ranking or rebalancing.
- Hard cap: recover at most 4 missed Mondays per call.
- If more than 4 are missed: send critical alert, stop -- wait for manual review.
- If rebalancing fails mid-way: log partial state, send alert, mark date as failed,
  do NOT retry automatically.
- Naturally idempotent: after recovery the execution_log is updated, so a second
  call on the same day finds no new missed dates and exits silently.
- Dates that previously failed (and were not manually cleared) are skipped on
  re-run to prevent the system from retrying a broken state.
"""
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn
from m5_execution.scheduler import init_m5_db, run_paper_rebalancing

_MAX_CATCHUP          = 4
_DEFAULT_CAPITAL      = 10_000.0
_FAILURE_LOCK_PATH    = os.path.join(_ROOT, "data", "recovery_failures.json")


# ── Failure lock ──────────────────────────────────────────────────────────────

def _load_failed_dates() -> set:
    """Dates that previously failed recovery and need manual review."""
    if not os.path.exists(_FAILURE_LOCK_PATH):
        return set()
    try:
        with open(_FAILURE_LOCK_PATH) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_failed_dates(dates: set) -> None:
    try:
        with open(_FAILURE_LOCK_PATH, "w") as f:
            json.dump(sorted(dates), f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not save recovery_failures.json: {exc}")


# ── Core helpers ──────────────────────────────────────────────────────────────

def _last_rebalancing_date(conn) -> date | None:
    """Return the UTC date of the most recent rebalancing, or None if never run."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM execution_log "
        "WHERE action IN ('open_long','open_short','close_long','close_short')"
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).date()


def _missed_mondays(last_reb: date, today: date) -> list:
    """
    Return all Mondays strictly after last_reb and on-or-before today.
    If last_reb was itself a Monday, the next expected Monday is 7 days later.
    """
    days_ahead = (7 - last_reb.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7   # last_reb is Monday -- next Monday is 7 days out
    first = last_reb + timedelta(days=days_ahead)
    result = []
    d = first
    while d <= today:
        result.append(d)
        d += timedelta(days=7)
    return result


def _run_m1_incremental() -> bool:
    """Run M1 incremental pipeline. Returns True on success."""
    logger.info("Running M1 incremental to ensure data is current...")
    rc = subprocess.run(
        [sys.executable, "-m", "m1_data_pipeline.pipeline", "--mode", "incremental"],
        cwd=_ROOT,
    ).returncode
    if rc != 0:
        logger.error(f"M1 incremental FAILED (exit {rc}) -- aborting catch-up")
        return False
    logger.info("M1 incremental complete")
    return True


# ── Main recovery function ────────────────────────────────────────────────────

def check_and_recover(capital: float = _DEFAULT_CAPITAL) -> list:
    """
    Detect and recover missed Monday rebalancings.
    Returns list of recovered date strings (empty if nothing missed or needed).
    Safe to call multiple times per day -- naturally idempotent via execution_log.
    """
    today = datetime.now(timezone.utc).date()

    with get_conn() as conn:
        init_m5_db(conn)
        last_reb = _last_rebalancing_date(conn)

    if last_reb is None:
        logger.info(
            "missed_rebalancing: no execution_log records -- "
            "fresh system, skipping catch-up"
        )
        return []

    missed = _missed_mondays(last_reb, today)

    if not missed:
        logger.info(
            f"missed_rebalancing: last rebalancing {last_reb} -- "
            f"no missed Mondays as of {today}"
        )
        return []

    # Hard cap check
    if len(missed) > _MAX_CATCHUP:
        msg = (
            f"missed_rebalancing: {len(missed)} missed Mondays detected "
            f"({', '.join(str(d) for d in missed)}) -- "
            f"exceeds safety cap of {_MAX_CATCHUP}. Manual review required."
        )
        logger.critical(msg)
        try:
            from m6_monitor.alerts import send_alert
            send_alert(msg, level="critical")
        except Exception:
            pass
        return []

    # Filter out previously failed dates
    failed_dates = _load_failed_dates()
    to_recover = [d for d in missed if d.isoformat() not in failed_dates]
    skipped    = [d for d in missed if d.isoformat() in failed_dates]

    for d in skipped:
        logger.warning(
            f"missed_rebalancing: {d} previously failed -- "
            f"skipping (delete data/recovery_failures.json to retry)"
        )

    if not to_recover:
        return []

    logger.info(
        f"missed_rebalancing: {len(to_recover)} missed Monday(s) to recover: "
        + ", ".join(str(d) for d in to_recover)
    )

    # M1 runs ONCE before any ranking -- stale data guard
    if not _run_m1_incremental():
        msg = (
            f"Missed rebalancing catch-up ABORTED: M1 incremental failed. "
            f"Missed dates: {', '.join(str(d) for d in to_recover)}"
        )
        try:
            from m6_monitor.alerts import send_alert
            send_alert(msg, level="critical")
        except Exception:
            pass
        return []

    recovered = []

    for missed_monday in to_recover:
        missed_str = missed_monday.isoformat()
        logger.info(f"Missed rebalancing detected for {missed_str} -- recovering now")

        try:
            from m6_monitor.alerts import send_alert
            send_alert(
                f"Missed rebalancing {missed_str} -- running catch-up now",
                level="warning",
            )
        except Exception:
            pass

        # M2 ranking for the missed Monday (outside the DB connection)
        try:
            from m2_factor_engine.engine import run_weekly_ranking
            run_weekly_ranking(missed_str)
            logger.info(f"M2 ranking complete for {missed_str}")
        except Exception as exc:
            logger.warning(
                f"M2 ranking failed for {missed_str} ({exc}) -- "
                f"proceeding with most recent rankings"
            )

        # Full paper rebalancing
        try:
            with get_conn() as conn:
                init_m5_db(conn)
                run_paper_rebalancing(capital, conn)
        except Exception as exc:
            msg = (
                f"Catch-up rebalancing for {missed_str} FAILED: {exc}. "
                f"Portfolio may be in partial state. Manual review required."
            )
            logger.error(msg)
            failed_dates.add(missed_str)
            _save_failed_dates(failed_dates)
            try:
                from m6_monitor.alerts import send_alert
                send_alert(msg, level="critical")
            except Exception:
                pass
            # Do not retry -- break and wait for manual review
            break

        recovered.append(missed_str)
        logger.info(f"Catch-up rebalancing complete for {missed_str}")
        try:
            from m6_monitor.alerts import send_alert
            send_alert(
                f"Catch-up complete for {missed_str}",
                level="info",
            )
        except Exception:
            pass

    return recovered


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M5 Missed Rebalancing Recovery")
    parser.add_argument(
        "--capital", type=float, default=_DEFAULT_CAPITAL,
        help=f"Initial capital USDT (default: {_DEFAULT_CAPITAL})"
    )
    args = parser.parse_args()

    recovered = check_and_recover(capital=args.capital)
    if recovered:
        print(f"Recovered {len(recovered)} missed rebalancing(s): {', '.join(recovered)}")
    else:
        print("No missed rebalancings detected.")
