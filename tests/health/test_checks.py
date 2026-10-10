#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixture-based tests for tools/health/run_checks.py. Every test builds its
fixture in a temp directory and monkeypatches the runner's module-level
path constants (ROOT, SYSTEMS, FUNDING_DIR) to point at it -- nothing here
ever touches a live file under data/. stdlib unittest only, no pytest
dependency.

Run: python -m unittest discover -s tests/health -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "health"))
sys.path.insert(0, str(ROOT / "engines"))

import run_checks  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TempRootTestCase(unittest.TestCase):
    """Base class: gives each test a throwaway directory and a clean way
    to monkeypatch run_checks' module globals, auto-restored in tearDown."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="health_test_"))
        self._patched: dict[str, object] = {}

    def tearDown(self):
        for name, value in self._patched.items():
            setattr(run_checks, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch(self, name: str, value) -> None:
        if name not in self._patched:
            self._patched[name] = getattr(run_checks, name)
        setattr(run_checks, name, value)

    def write_csv(self, path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def one_system(self, sys_id: str = "TEST1", **overrides) -> dict:
        base = dict(id=sys_id, label=sys_id, data_dir=self.tmp / sys_id,
                    equity_file="equity_curve.csv", max_open=None,
                    leverage_max=1.0, config_path=None)
        base.update(overrides)
        return base


class TestFundingStalenessRealGap(TempRootTestCase):
    """The real FET_USDT file as committed at the point the original
    OHLCV freeze was discovered (git commit 49544e4b, 2026-06-16) -- its
    last real bar is 2026-06-13. The incident narrative this project was
    built on describes rediscovering this as "~21 days stale" around
    2026-07-04; reproduced here against today=2026-07-15 the true gap is
    (2026-07-15 - 2026-06-13) = 32 days. Either way this asserts the one
    thing that actually matters: the staleness check flags real frozen
    production data as CRITICAL, using the exact file that was frozen, not
    a synthetic stand-in."""

    def test_fet_flagged_stale(self):
        cache_dir = self.tmp / "ohlcv_1d"
        cache_dir.mkdir(parents=True)
        shutil.copy(FIXTURES / "FET_USDT_1d_as_of_2026-07-15.csv",
                    cache_dir / "FET_USDT_1d.csv")
        result = run_checks.heartbeat.check_ohlcv_staleness(
            "test", cache_dir, date(2026, 7, 15), allowlist=None, suppress=None)
        self.assertIsNotNone(result, "real frozen FET data was not flagged")
        msg, worst_days = result
        self.assertGreaterEqual(worst_days, run_checks.heartbeat.MAX_OHLCV_STALE_DAYS)
        self.assertIn("FET_USDT", msg)


class TestFundingStalenessFrozenFile(TempRootTestCase):
    def test_frozen_funding_file_flagged_critical(self):
        funding_dir = self.tmp / "funding_rates"
        funding_dir.mkdir(parents=True)
        import datetime as _dt
        old_date = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=50)
        frozen_ms = int(old_date.timestamp() * 1000)
        self.write_csv(funding_dir / "BTCUSDT_funding.csv",
                        ["funding_time,funding_rate", f"{frozen_ms},0.0001"])
        self.patch("FUNDING_DIR", funding_dir)
        findings = run_checks.check_funding_staleness()
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["severity"], "CRITICAL")
        self.assertEqual(findings[0]["check"], "funding_staleness")
        self.assertIn("BTCUSDT", findings[0]["message"])


class Test1970Rows(TempRootTestCase):
    def test_1970_row_flagged(self):
        spot_dir = self.tmp / "data" / "universe" / "ohlcv_1d"
        spot_dir.mkdir(parents=True)
        self.write_csv(spot_dir / "ZZZ_USDT_1d.csv", [
            "time,open,high,low,close,volume",
            "1970-01-01 00:00:01+00:00,1.0,1.0,1.0,1.0,1.0",
            "2026-01-01 00:00:00+00:00,1.0,1.0,1.0,1.0,1.0",
        ])
        self.patch("ROOT", self.tmp)
        findings = run_checks.check_timestamps_1970()
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "timestamps_1970")
        self.assertIn("ZZZ_USDT", str(findings[0]["evidence"]["symbols"]))

    def test_clean_file_not_flagged(self):
        spot_dir = self.tmp / "data" / "universe" / "ohlcv_1d"
        spot_dir.mkdir(parents=True)
        self.write_csv(spot_dir / "ZZZ_USDT_1d.csv", [
            "time,open,high,low,close,volume",
            "2026-01-01 00:00:00+00:00,1.0,1.0,1.0,1.0,1.0",
        ])
        self.patch("ROOT", self.tmp)
        self.assertEqual(run_checks.check_timestamps_1970(), [])


class TestDuplicateHeader(TempRootTestCase):
    def test_duplicate_header_flagged(self):
        sysconf = self.one_system()
        header = "date,paper_equity,unrealized_pnl,total_value,open_positions,total_cost,total_market_value"
        self.write_csv(sysconf["data_dir"] / "equity_curve.csv", [
            header,
            "2026-01-01,10000,0,10000,0,0,0",
            header,  # duplicated header row -- the real corruption signature
            "2026-01-02,10000,0,10000,0,0,0",
        ])
        self.patch("SYSTEMS", [sysconf])
        findings = run_checks.check_csv_integrity()
        dup = [f for f in findings if f["check"] == "csv_duplicate_header"]
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]["system"], "TEST1")


class TestSkippedRunDate(TempRootTestCase):
    """missed_run_single_day: last_run_date sitting two days behind the
    safe (ad-hoc) floor must be flagged; sitting at exactly the floor
    (normal, pre-cron state) must NOT be -- this is the false-positive
    this check produced on first live test (2026-10-10) before the floor
    was corrected from TODAY-1 to TODAY-2 for unset HEALTH_SAME_DAY_CHECK."""

    def test_two_days_behind_is_flagged(self):
        today = date.today()
        last_run = (today - timedelta(days=4)).isoformat()
        sysconf = self.one_system(id="S_TEST")
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"last_run_date": last_run, "paper_equity_usdt": 10000})
        self.patch("SYSTEMS", [sysconf])
        import os
        os.environ.pop("HEALTH_SAME_DAY_CHECK", None)
        findings = run_checks.check_missed_run_single_day()
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "missed_run_single_day")

    def test_at_safe_floor_is_not_flagged(self):
        today = date.today()
        last_run = (today - timedelta(days=2)).isoformat()  # the safe, ad-hoc floor
        sysconf = self.one_system(id="S_TEST")
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"last_run_date": last_run, "paper_equity_usdt": 10000})
        self.patch("SYSTEMS", [sysconf])
        import os
        os.environ.pop("HEALTH_SAME_DAY_CHECK", None)
        findings = run_checks.check_missed_run_single_day()
        self.assertEqual(findings, [], "normal pre-cron state must not false-positive")

    def test_same_day_mode_catches_tighter_floor(self):
        today = date.today()
        last_run = (today - timedelta(days=2)).isoformat()
        sysconf = self.one_system(id="S_TEST")
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"last_run_date": last_run, "paper_equity_usdt": 10000})
        self.patch("SYSTEMS", [sysconf])
        import os
        os.environ["HEALTH_SAME_DAY_CHECK"] = "1"
        try:
            findings = run_checks.check_missed_run_single_day()
        finally:
            del os.environ["HEALTH_SAME_DAY_CHECK"]
        self.assertEqual(len(findings), 1)


class TestCrashingCheck(TempRootTestCase):
    def test_crash_yields_crashed_finding(self):
        def boom():
            raise ValueError("synthetic failure for the test")

        findings, status = run_checks._run_one("synthetic", boom)
        self.assertEqual(status, "fail")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "synthetic_crashed")
        self.assertEqual(findings[0]["severity"], "CRITICAL")
        self.assertEqual(findings[0]["system"], "runner")
        self.assertIn("synthetic failure", findings[0]["message"])

    def test_clean_check_not_marked_crashed(self):
        findings, status = run_checks._run_one("ok", lambda: [])
        self.assertEqual(status, "pass")
        self.assertEqual(findings, [])


class TestNoKillSwitchConfigured(TempRootTestCase):
    def test_missing_field_flagged(self):
        sysconf = self.one_system(id="S5_TEST")
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"last_run_date": "2026-01-01", "paper_equity_usdt": 10000})
        self.patch("SYSTEMS", [sysconf])
        findings = run_checks.check_no_kill_switch_configured()
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "no_kill_switch_configured")

    def test_present_field_not_flagged(self):
        sysconf = self.one_system(id="S1_TEST")
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"last_run_date": "2026-01-01", "paper_equity_usdt": 10000,
                          "kill_switch_triggered": False})
        self.patch("SYSTEMS", [sysconf])
        self.assertEqual(run_checks.check_no_kill_switch_configured(), [])


class TestLeverageCapBreach(TempRootTestCase):
    """Reproduces the real live finding (S2, 2026-10-10): a position sized
    via qty * entry_price alone, with no notional_usdt column, that blows
    past the configured 1x cap."""

    def test_no_notional_column_still_computed(self):
        sysconf = self.one_system(id="S2_TEST", leverage_max=1.0)
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"paper_equity_usdt": 10000})
        self.write_csv(sysconf["data_dir"] / "open_positions.csv", [
            "symbol,entry_price,qty",
            "COSUSDT,0.000485,236873819",
        ])
        self.patch("SYSTEMS", [sysconf])
        findings = run_checks.check_leverage_cap()
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "leverage_cap_breach")
        self.assertGreater(findings[0]["evidence"]["implied_leverage"], 1.0)

    def test_within_cap_not_flagged(self):
        sysconf = self.one_system(id="S1_TEST", leverage_max=1.0)
        self.write_json(sysconf["data_dir"] / "state.json",
                         {"paper_equity_usdt": 10000})
        self.write_csv(sysconf["data_dir"] / "open_positions.csv", [
            "symbol,entry_price,qty",
            "BTCUSDT,50000,0.1",  # $5,000 notional on $10,000 equity -- 0.5x
        ])
        self.patch("SYSTEMS", [sysconf])
        self.assertEqual(run_checks.check_leverage_cap(), [])


class TestReconciliation(TempRootTestCase):
    """Open -> resolve lifecycle through run_checks' own reconciliation
    logic, independent of any real check -- synthetic findings in, verify
    first_seen/last_seen and auto-resolve behave per docs/HEALTH_CONTRACT.md."""

    def test_open_then_resolve_on_disappearance(self):
        f = run_checks.finding("T", "synthetic_check", "WARNING", "still here")
        events1, open1 = run_checks._reconcile([], [f])
        self.assertEqual(len(open1), 1)
        self.assertEqual(open1[0]["first_seen"], open1[0]["last_seen"])

        # Second run: finding no longer reproduced, and its check name IS
        # in the registry's owned set (synthetic_check isn't really
        # registered, so patch REGISTRY_CHECK_NAMES to include it for this
        # test -- the mechanism under test is "owned check + gone => resolve").
        run_checks.REGISTRY_CHECK_NAMES.add("synthetic_check")
        try:
            events2, open2 = run_checks._reconcile(events1, [])
        finally:
            run_checks.REGISTRY_CHECK_NAMES.discard("synthetic_check")
        self.assertEqual(open2, [])
        self.assertTrue(any(e["event"] == "resolve" for e in events2))

    def test_engine_emitted_finding_not_auto_resolved(self):
        """A check name NOT in REGISTRY_CHECK_NAMES (i.e. emitted by an
        engine's health_emit.emit(), not a registry check) must stay open
        even when this run's registry checks don't reproduce it -- the
        runner has no logic re-deriving that specific condition."""
        f = run_checks.finding("S1", "some_engine_emitted_thing", "WARNING", "x")
        events1, open1 = run_checks._reconcile([], [f])
        events2, open2 = run_checks._reconcile(events1, [])  # nothing re-opens it
        self.assertEqual(len(open2), 1, "engine-emitted finding was wrongly auto-resolved")
        self.assertFalse(any(e["event"] == "resolve" for e in events2))


if __name__ == "__main__":
    unittest.main()
