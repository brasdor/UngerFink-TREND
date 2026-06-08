"""
M6 Heartbeat
============
Server health monitor.  Checks five conditions every 5 minutes in --loop mode
and sends a Telegram alert if any check fails.
Never crashes on a locked SQLite database -- all reads use a retry wrapper.

CLI:
  python -m m6_monitor.heartbeat --once   # single check, print result, exit
  python -m m6_monitor.heartbeat --loop   # continuous (every 5 minutes)
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import get_conn

_DB_PATH            = os.path.join(_ROOT, "data", "trading.db")
_HEARTBEAT_INTERVAL = 300   # 5 minutes


# ── SQLite retry helper ───────────────────────────────────────────────────────

def _safe_query(conn, sql: str, params: tuple = (), retries: int = 3, delay: int = 3):
    """Execute a query with retry on locked database. Returns fetchone() result or None."""
    for attempt in range(retries):
        try:
            return conn.execute(sql, params).fetchone()
        except Exception as exc:
            if attempt < retries - 1:
                logger.debug(f"DB retry {attempt + 1}/{retries}: {exc}")
                time.sleep(delay)
            else:
                logger.warning(f"DB query failed after {retries} attempts: {exc}")
    return None


# ── Health checks ─────────────────────────────────────────────────────────────

def check_health(conn) -> dict:
    """
    Run all health checks.
    Returns {ok: bool, checks: list[{name, status, detail}]}.
    status values: 'ok' | 'warn' | 'fail'
    """
    now_ms = int(time.time() * 1000)
    checks = []

    # 1. Last pipeline run within 25 hours
    try:
        row = _safe_query(conn, "SELECT MAX(run_time) FROM pipeline_runs")
        if row and row[0]:
            age_h = (now_ms - int(row[0])) / 3_600_000
            if age_h <= 25:
                checks.append({"name": "pipeline_run", "status": "ok",
                                "detail": f"Last run {age_h:.1f}h ago"})
            else:
                checks.append({"name": "pipeline_run", "status": "fail",
                                "detail": f"Last run {age_h:.1f}h ago (limit: 25h)"})
        else:
            checks.append({"name": "pipeline_run", "status": "warn",
                            "detail": "No pipeline_runs records found"})
    except Exception as exc:
        checks.append({"name": "pipeline_run", "status": "warn",
                        "detail": f"Could not check: {exc}"})

    # 2. Last rebalancing within 8 days
    try:
        row = _safe_query(
            conn,
            "SELECT MAX(timestamp) FROM execution_log WHERE action LIKE 'open%'",
        )
        if row and row[0]:
            age_d = (now_ms - int(row[0])) / 86_400_000
            if age_d <= 8:
                checks.append({"name": "rebalancing", "status": "ok",
                                "detail": f"Last rebalancing {age_d:.1f}d ago"})
            else:
                checks.append({"name": "rebalancing", "status": "fail",
                                "detail": f"Last rebalancing {age_d:.1f}d ago (limit: 8d)"})
        else:
            checks.append({"name": "rebalancing", "status": "warn",
                            "detail": "No rebalancing records found"})
    except Exception as exc:
        checks.append({"name": "rebalancing", "status": "warn",
                        "detail": f"Could not check: {exc}"})

    # 3. Database file size (warn > 2 GB, fail > 5 GB)
    try:
        size_gb = os.path.getsize(_DB_PATH) / 1_073_741_824
        if size_gb < 2.0:
            checks.append({"name": "db_size", "status": "ok",
                            "detail": f"{size_gb:.2f} GB"})
        elif size_gb < 5.0:
            checks.append({"name": "db_size", "status": "warn",
                            "detail": f"{size_gb:.2f} GB (approaching limit)"})
        else:
            checks.append({"name": "db_size", "status": "fail",
                            "detail": f"{size_gb:.2f} GB (exceeds 5 GB limit)"})
    except Exception as exc:
        checks.append({"name": "db_size", "status": "warn",
                        "detail": f"Could not check: {exc}"})

    # 4. Disk space: at least 1 GB free
    try:
        free_gb = shutil.disk_usage(_ROOT).free / 1_073_741_824
        if free_gb >= 1.0:
            checks.append({"name": "disk_space", "status": "ok",
                            "detail": f"{free_gb:.1f} GB free"})
        else:
            checks.append({"name": "disk_space", "status": "fail",
                            "detail": f"{free_gb:.2f} GB free (minimum: 1 GB)"})
    except Exception as exc:
        checks.append({"name": "disk_space", "status": "warn",
                        "detail": f"Could not check: {exc}"})

    # 5. All open positions have valid entry prices
    try:
        bad_row = _safe_query(
            conn,
            "SELECT COUNT(*) FROM paper_trades "
            "WHERE status='open' AND (entry_price IS NULL OR entry_price <= 0)",
        )
        tot_row = _safe_query(
            conn,
            "SELECT COUNT(*) FROM paper_trades WHERE status='open'",
        )
        bad   = int(bad_row[0])  if bad_row  else 0
        total = int(tot_row[0])  if tot_row  else 0
        if bad == 0:
            checks.append({"name": "position_prices", "status": "ok",
                            "detail": f"{total} open positions, all prices valid"})
        else:
            checks.append({"name": "position_prices", "status": "fail",
                            "detail": f"{bad}/{total} open positions have invalid entry price"})
    except Exception as exc:
        checks.append({"name": "position_prices", "status": "warn",
                        "detail": f"Could not check: {exc}"})

    overall_ok = all(c["status"] != "fail" for c in checks)
    return {"ok": overall_ok, "checks": checks}


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_report(result: dict) -> str:
    header = f"Heartbeat: {'OK' if result['ok'] else 'FAILED'}"
    lines  = [header]
    for c in result["checks"]:
        tag = "OK  " if c["status"] == "ok" else ("WARN" if c["status"] == "warn" else "FAIL")
        lines.append(f"  [{tag}] {c['name']}: {c['detail']}")
    return "\n".join(lines)


# ── Single run ────────────────────────────────────────────────────────────────

def run_once() -> tuple:
    """
    Open a fresh DB connection, run all checks, send alert if any fail.
    Returns (ok: bool, result: dict).
    """
    try:
        with get_conn() as conn:
            result = check_health(conn)
    except Exception as exc:
        msg = f"Cannot open database: {exc}"
        logger.error(f"Heartbeat: {msg}")
        try:
            from m6_monitor.alerts import send_heartbeat_alert
            send_heartbeat_alert(msg)
        except Exception:
            pass
        return False, {"ok": False, "checks": [{"name": "db_open", "status": "fail", "detail": msg}]}

    report_str = _format_report(result)
    if result["ok"]:
        logger.info(report_str.replace("\n", " | "))
    else:
        logger.warning(report_str)
        try:
            from m6_monitor.alerts import send_heartbeat_alert
            send_heartbeat_alert(report_str)
        except Exception as exc:
            logger.warning(f"Could not send heartbeat alert: {exc}")

    return result["ok"], result


# ── Continuous loop ───────────────────────────────────────────────────────────

def run_heartbeat_loop() -> None:
    """Run health checks every 5 minutes, alerting on any failure."""
    log_path = os.path.join(_ROOT, "data", "heartbeat.log")
    logger.add(log_path, rotation="10 MB", level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.info("Heartbeat loop started (interval: 5 min)")
    while True:
        run_once()
        time.sleep(_HEARTBEAT_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M6 Heartbeat Monitor")
    parser.add_argument("--once", action="store_true",
                        help="Run one check, print result, and exit")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously every 5 minutes")
    args = parser.parse_args()

    if args.once:
        ok, result = run_once()
        print(_format_report(result))
        sys.exit(0 if ok else 1)
    elif args.loop:
        run_heartbeat_loop()
    else:
        parser.print_help()
