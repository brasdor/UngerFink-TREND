#!/usr/bin/env python3
"""Research-only snapshot staleness is a WARNING; funding / exchange-cache staleness stays CRITICAL."""
import json, sys, tempfile, unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "health"))
sys.path.insert(0, str(ROOT / "engines"))
import run_checks as rc  # noqa: E402


class TestSnapshotSeverity(unittest.TestCase):
    def _run(self, hours_old):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fwd = td / "data" / "riskoff" / "forward"
            fwd.mkdir(parents=True)
            old = (datetime.now(timezone.utc) - timedelta(hours=hours_old)).isoformat()
            (fwd / "snapshot_heartbeat.json").write_text(json.dumps(
                {k: {"last_success_utc": old, "detail": ""} for k in ("stablecoins", "epu", "gpr", "dominance")}), encoding="utf-8")
            # OI log with a stale last bar
            ms = int((datetime.now(timezone.utc) - timedelta(hours=hours_old)).timestamp() * 1000)
            (fwd / "oi_btcusdt_5m.csv").write_text(f"bar_time_ms\n{ms}\n", encoding="utf-8")
            with mock.patch.object(rc, "ROOT", td):
                return rc.check_forward_snapshots_fresh()

    def test_stale_snapshots_and_oi_log_are_warning_not_critical(self):
        out = self._run(70)
        checks = {f["check"] for f in out}
        self.assertEqual(checks, {"snapshot_stale", "oi_log_stale"})
        self.assertEqual(len([f for f in out if f["check"] == "snapshot_stale"]), 4)
        self.assertTrue(all(f["severity"] == "WARNING" for f in out), [f["severity"] for f in out])

    def test_fresh_snapshots_raise_nothing(self):
        self.assertEqual(self._run(2), [])

    def test_missing_snapshots_are_warning(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(rc, "ROOT", Path(td)):
                out = rc.check_forward_snapshots_fresh()
        self.assertTrue(out and all(f["severity"] == "WARNING" for f in out))

    def test_funding_and_exchange_cache_checks_remain_critical(self):
        src = (ROOT / "tools" / "health" / "run_checks.py").read_text(encoding="utf-8")
        fa = src[src.index("def check_funding_staleness"):src.index("def check_frozen_values")]
        self.assertIn('"CRITICAL"', fa)
        fj = src[src.index("def check_funding_refresh_job_alive"):src.index("# Registry + runner")]
        self.assertIn('"CRITICAL"', fj)
        ex = src[src.index("def check_exchange_status"):src.index("# RUNS checks")]
        self.assertIn('"WARNING" if mode == "stale" else "CRITICAL"', ex)


if __name__ == "__main__":
    unittest.main()
