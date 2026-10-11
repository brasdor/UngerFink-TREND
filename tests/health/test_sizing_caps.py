#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fails if any sizing path lacks the stop-distance floor + notional cap
that fixed the S2 leverage bug (2026-10-10/11: qty = risk_amount /
risk_per_unit had no ceiling, risk_per_unit had no floor -- confirmed to
blow out to ~39x on a SETTLING futures contract's frozen price).

This is a source-presence test, not a numerical one -- the numerical
proof (old vs new formula produce identical qty for every historical
entry except the 8 confirmed S2 breaches, and the P&L-driving fields
never change) is in the commit history and the tools/health/run_checks.py
leverage_cap check that runs against live state daily. This test's job is
narrower and more durable: make it impossible to add a 7th engine (or
quietly revert the fix in one of the existing six) without this suite
noticing, by asserting the known markers of the fix are present in every
file that computes a position size from a risk amount and a stop
distance.

Run: python -m unittest discover -s tests/health -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Every T9B paper engine with its own risk_amount / stop_distance sizing
# formula (the shape that was vulnerable), plus the one real-order path
# (auto_execute.py, disabled by default) and the two candidates (a
# differently-shaped but equivalent cap -- see its own markers below).
RISK_PER_UNIT_SIZING_FILES = [
    ROOT / "engines" / "phase_t9b_donchian_universev2_paper_engine.py",      # S1
    ROOT / "engines" / "phase_t9b_meanreversion_paper_engine.py",            # S2
    ROOT / "engines" / "phase_t9b_consecdowndays_paper_engine.py",           # S3
    ROOT / "phase_t9b_volcontraction_short_paper_engine.py",                 # S6
    ROOT / "phase_t9b_macross_short_paper_engine.py",                        # S7
    ROOT / "engines" / "phase_t9b_rsi_mr_funding_paper_engine.py",           # S8
    ROOT / ".github" / "scripts" / "auto_execute.py",                       # real orders
]

CANDIDATE_FILES = [
    ROOT / "engines" / "phase_t9_momentum_candidates_paper_engine.py",       # C12 + C19
]


class TestEveryRiskPerUnitSizingPathHasTheCap(unittest.TestCase):
    def test_floor_and_cap_markers_present(self):
        missing = []
        for path in RISK_PER_UNIT_SIZING_FILES:
            if not path.exists():
                missing.append(f"{path}: FILE NOT FOUND")
                continue
            src = path.read_text(encoding="utf-8")
            if "MIN_RISK_PER_UNIT_PCT" not in src:
                missing.append(f"{path}: no stop-distance floor (MIN_RISK_PER_UNIT_PCT)")
            if "leverage_cap_bound" not in src:
                missing.append(f"{path}: no notional-cap emit (leverage_cap_bound)")
        self.assertEqual(missing, [], "sizing path(s) missing the floor/cap: " + "; ".join(missing))

    def test_floor_never_feeds_the_stored_pnl_denominator(self):
        """The floor must stay out of whatever gets stored as
        initial_risk_per_unit (every exit branch's gross_r/pnl_usdt reads
        that field) -- this is the exact mistake caught in review on
        2026-10-11 (S2 initially stored the FLOORED value). A sizing-only
        fix must assign risk_per_unit (or risk_unit) unchanged, with the
        floored copy under a *_for_sizing / *_sized name."""
        for path in RISK_PER_UNIT_SIZING_FILES:
            if not path.exists() or path.name == "auto_execute.py":
                continue  # no Position/pnl_usdt concept in the real-order path
            src = path.read_text(encoding="utf-8")
            self.assertTrue(
                "_for_sizing" in src or "_sized" in src,
                f"{path}: no separate *_for_sizing/*_sized variable -- the floored "
                f"value may be leaking into the stored risk-per-unit field used for P&L",
            )

    def test_candidates_have_a_position_pct_cap_with_observability(self):
        for path in CANDIDATE_FILES:
            self.assertTrue(path.exists(), f"{path}: FILE NOT FOUND")
            src = path.read_text(encoding="utf-8")
            self.assertIn("max_position_pct", src,
                          f"{path}: no max_position_pct cap on sized notional")
            self.assertIn("position_pct_cap_bound", src,
                          f"{path}: cap exists but doesn't emit when it binds")


if __name__ == "__main__":
    unittest.main()
