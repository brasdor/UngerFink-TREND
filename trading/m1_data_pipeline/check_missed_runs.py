"""
M1 Check Missed Runs
====================
Detects data gaps by querying pipeline_runs.run_time (epoch-ms, UTC).
If the last successful pipeline run is more than 25 hours old, runs an
incremental catch-up so the next pipeline step has fresh data.

Called from run_daily.bat BEFORE the main pipeline run.
Exit 0 = data current or catch-up succeeded.
Exit 1 = catch-up failed.
"""
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from m1_data_pipeline.database import get_conn

_STALE_THRESHOLD_H = 25


def check_and_catchup() -> int:
    now_ms = int(time.time() * 1000)

    # Query the pipeline_runs table for the last completed run
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(run_time) FROM pipeline_runs"
            ).fetchone()
    except Exception as exc:
        print(f"check_missed_runs: cannot read pipeline_runs: {exc} -- skipping check",
              flush=True)
        return 0   # do not block the main pipeline run

    if not row or not row[0]:
        print("check_missed_runs: no pipeline_runs records -- fresh system, no catch-up needed",
              flush=True)
        return 0

    age_h   = (now_ms - int(row[0])) / 3_600_000
    age_d   = age_h / 24

    if age_h <= _STALE_THRESHOLD_H:
        print(f"check_missed_runs: last run {age_h:.1f}h ago -- OK", flush=True)
        return 0

    print(
        f"check_missed_runs: last run {age_d:.1f} day(s) ago "
        f"({age_h:.0f}h -- threshold {_STALE_THRESHOLD_H}h) -- running catch-up",
        flush=True,
    )

    result = subprocess.run(
        [sys.executable, "-m", "m1_data_pipeline.pipeline", "--mode", "incremental"],
        cwd=_ROOT,
    )

    if result.returncode != 0:
        print(
            f"check_missed_runs: catch-up pipeline FAILED (exit {result.returncode})",
            flush=True,
        )
        return 1

    print("check_missed_runs: catch-up pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(check_and_catchup())
