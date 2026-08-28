#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh futures funding rates from a machine that can reach Binance, then push.

WHY THIS EXISTS
---------------
fapi.binance.com returns HTTP 451 to GitHub's US-hosted runners, so the daily
workflow cannot fetch funding. Klines survive that block because
.github/scripts/refresh_futures_data.py falls back to data.binance.vision's
DAILY kline dumps -- but binance.vision publishes funding only MONTHLY, so
there is no equivalent fallback and funding just stales.

refresh_futures_data.py already says the remedy is "a periodic local refresh
pushed from a non-US machine". Nothing automated it, so it happened only when
somebody remembered: as of 2026-08-28 the 289 funding files had a median age
of 5 days and a worst case of 78 days, while funding settles every 8 hours.

Stale funding is not cosmetic. It feeds the regime funding axis (which sets
every system's capital weight) and the S6 / S7 / S8 funding gates, so a stale
file makes those systems trade yesterday's carry.

WHAT IT DOES
------------
Reuses fapi_refresh_funding() from .github/scripts/refresh_futures_data.py --
the same incremental fetch the workflow would run if it were not blocked, not
a second implementation that can drift from it. Then commits and pushes only
the funding directory.

USAGE
-----
    python tools/refresh_funding_local.py              # refresh, commit, push
    python tools/refresh_funding_local.py --check      # report ages, change nothing
    python tools/refresh_funding_local.py --no-push    # refresh + commit, no push

Run it from the repository root. run_funding_refresh.bat wraps it for Task
Scheduler; see setup_task_scheduler.ps1.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data" / "futures_universe" / "funding_rates"
REFRESH_SCRIPT = ROOT / ".github" / "scripts" / "refresh_futures_data.py"

# Funding settles every 8h. A file older than this is stale enough that the
# gates reading it are acting on materially old carry.
STALE_DAYS = 2


def load_refresh_module():
    """Import refresh_futures_data.py by path (it lives outside any package)."""
    spec = importlib.util.spec_from_file_location("refresh_futures_data", REFRESH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def funding_ages() -> list[tuple[float, str]]:
    """(age_in_days, filename) for every funding file, oldest first."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    ages: list[tuple[float, str]] = []
    for path in sorted(FUND_DIR.glob("*_funding.csv")):
        try:
            frame = pd.read_csv(path)
            last = pd.to_datetime(int(frame["funding_time"].max()), unit="ms")
            ages.append(((now - last).total_seconds() / 86400.0, path.name))
        except Exception as exc:
            print(f"[FUNDING] could not read {path.name}: {exc}")
    ages.sort(reverse=True)
    return ages


def report(ages: list[tuple[float, str]], header: str) -> None:
    if not ages:
        print(f"[FUNDING] {header}: no funding files found")
        return
    values = [age for age, _ in ages]
    median = sorted(values)[len(values) // 2]
    stale = [(a, n) for a, n in ages if a > STALE_DAYS]
    print(f"[FUNDING] {header}: {len(ages)} files  "
          f"median {median:.1f}d  worst {values[0]:.1f}d  "
          f"stale(>{STALE_DAYS}d) {len(stale)}")
    for age, name in ages[:5]:
        print(f"           {age:6.1f}d  {name}")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def commit_and_push(push: bool) -> int:
    add = git("add", "--force", str(FUND_DIR.relative_to(ROOT).as_posix()))
    if add.returncode != 0:
        print(f"[GIT] add failed: {add.stderr.strip()}")
        return 1

    if git("diff", "--cached", "--quiet").returncode == 0:
        print("[GIT] funding already up to date -- nothing to commit")
        return 0

    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    message = f"Funding rates: local refresh {today} [skip ci]"
    commit = git("commit", "-m", message)
    if commit.returncode != 0:
        print(f"[GIT] commit failed: {commit.stderr.strip()}")
        return 1
    print(f"[GIT] committed: {message}")

    if not push:
        print("[GIT] --no-push given; commit left local")
        return 0

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    # Same race as the two daily workflows: another push may land first.
    for attempt in range(1, 6):
        pushed = git("push", "origin", branch)
        if pushed.returncode == 0:
            print(f"[GIT] pushed to {branch}")
            return 0
        print(f"[GIT] push attempt {attempt} rejected -- rebasing and retrying")
        rebase = git("-c", "rebase.autoStash=true", "pull", "--rebase", "origin", branch)
        if rebase.returncode != 0:
            print(f"[GIT] rebase failed: {rebase.stderr.strip()}")
    print("[GIT] ERROR: could not push after 5 attempts")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report funding ages and exit without changing anything")
    parser.add_argument("--no-push", action="store_true",
                        help="refresh and commit but do not push")
    args = parser.parse_args()

    if not FUND_DIR.exists():
        print(f"[FUNDING] {FUND_DIR} does not exist -- run from the repo root")
        return 1

    report(funding_ages(), "before")
    if args.check:
        return 0

    refresh = load_refresh_module()
    if not refresh.probe_fapi():
        print("[FUNDING] fapi.binance.com is NOT reachable from this machine "
              "(geo-blocked, or offline). Funding cannot be refreshed here -- "
              "run this from a machine that can reach Binance.")
        return 1

    print("[FUNDING] fapi reachable -- refreshing")
    updated = refresh.fapi_refresh_funding()
    print(f"[FUNDING] updated {updated} symbols")

    report(funding_ages(), "after")
    return commit_and_push(push=not args.no_push)


if __name__ == "__main__":
    sys.exit(main())
