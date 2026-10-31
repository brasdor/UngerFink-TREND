#!/usr/bin/env python3
"""Findings close when the registry check that owns their slug passes again (slug != check name included)."""
import ast, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "health"))
sys.path.insert(0, str(ROOT / "engines"))
import run_checks as rc  # noqa: E402


def _open(fid, slug, system="x"):
    return {"event": "open", "id": fid, "severity": "CRITICAL", "system": system, "check": slug, "message": "m",
            "evidence": {}, "suggested_action": "", "run_date": "2026-10-11"}


def _resolved_ids(events):
    return {e["id"] for e in events if e["event"] == "resolve"}


class TestAutoResolve(unittest.TestCase):
    def test_every_slug_emitted_by_the_registry_is_mapped(self):
        tree = ast.parse((ROOT / "tools" / "health" / "run_checks.py").read_text(encoding="utf-8"))
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, fn in rc.REGISTRY:
            for c in ast.walk(fns[fn.__name__]):
                if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "finding" and len(c.args) > 1 \
                        and isinstance(c.args[1], ast.Constant):
                    self.assertEqual(rc.owning_check(c.args[1].value), name, c.args[1].value)

    def test_non_registry_slug_resolves_when_owning_check_no_longer_raises_it(self):
        prior = [_open("data/exchange_status_cache.json:exchange_status_not_trading:FUT:ACXUSDT",
                       "exchange_status_not_trading"),
                 _open("S2:leverage_cap_breach", "leverage_cap_breach", "S2"),
                 _open("funding_refresh_job:funding_refresh_job_dead", "funding_refresh_job_dead", "funding_refresh_job")]
        new, cur = rc._reconcile(prior, [])
        self.assertEqual(_resolved_ids(new), {p["id"] for p in prior})
        self.assertEqual(cur, [])

    def test_still_firing_finding_stays_open(self):
        prior = [_open("S2:leverage_cap_breach", "leverage_cap_breach", "S2")]
        now = [rc.finding("S2", "leverage_cap_breach", "CRITICAL", "still", key="")]
        new, cur = rc._reconcile(prior, now)
        self.assertEqual(_resolved_ids(new), set())
        self.assertEqual(len(cur), 1)

    def test_crashed_owner_does_not_resolve(self):
        prior = [_open("S2:leverage_cap_breach", "leverage_cap_breach", "S2")]
        crashed = [rc.finding("runner", "leverage_cap_crashed", "CRITICAL", "boom")]
        new, cur = rc._reconcile(prior, crashed)
        self.assertEqual(_resolved_ids(new), set())
        self.assertTrue(any(f["id"] == "S2:leverage_cap_breach" for f in cur))

    def test_engine_emitted_findings_are_left_alone(self):
        prior = [_open("S2:flat_window_rsi_invalidated:ACXUSDT", "flat_window_rsi_invalidated", "S2"),
                 _open("S2:tradability_state_fix_pending:X", "tradability_state_fix_pending", "S2")]
        new, cur = rc._reconcile(prior, [])
        self.assertEqual(_resolved_ids(new), set())
        self.assertEqual(len(cur), 2)

    def test_crashed_finding_resolves_once_check_runs_again(self):
        prior = [_open("runner:leverage_cap_crashed", "leverage_cap_crashed", "runner")]
        new, _ = rc._reconcile(prior, [])
        self.assertEqual(_resolved_ids(new), {"runner:leverage_cap_crashed"})


if __name__ == "__main__":
    unittest.main()
