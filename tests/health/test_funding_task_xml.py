#!/usr/bin/env python3
"""The funding-refresh scheduled task template: local-time daily trigger (no UTC offset, so it follows DST),
a logon trigger, wake-to-run and run-missed-on-wake."""
import re, unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


class TestFundingTaskXml(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(ROOT / "tools" / "windows" / "funding_refresh_task.xml").getroot()

    def test_daily_trigger_is_0730_local_wall_clock(self):
        sb = self.root.find(".//t:CalendarTrigger/t:StartBoundary", NS).text
        self.assertRegex(sb, r"^\d{4}-\d{2}-\d{2}T07:30:00$")      # no Z / +hh:mm => stored as local time
        self.assertEqual(self.root.find(".//t:ScheduleByDay/t:DaysInterval", NS).text, "1")

    def test_logon_trigger_wake_and_catch_up(self):
        self.assertIsNotNone(self.root.find(".//t:LogonTrigger", NS))
        self.assertEqual(self.root.find(".//t:Settings/t:WakeToRun", NS).text, "true")
        self.assertEqual(self.root.find(".//t:Settings/t:StartWhenAvailable", NS).text, "true")

    def test_runs_push_mode_job(self):
        args = self.root.find(".//t:Exec/t:Arguments", NS).text
        self.assertIn("run_funding_refresh.bat", args)
        self.assertNotIn("--no-push", (ROOT / "run_funding_refresh.bat").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
