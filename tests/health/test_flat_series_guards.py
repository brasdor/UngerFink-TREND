#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat-series tests for every engine, per the 2026-10-11 follow-up: for a
symbol with zero price range over its full indicator lookback, no engine
should produce a signal (or, for the ranking-based systems, should rank
that symbol into a basket) -- confirmed root cause across the T9B/
candidate family is "a legitimate-looking but meaningless number falls
out of a formula fed constant input," not a crash or an obvious error.

Each test builds a wholly synthetic flat OHLCV series (never real/live
data) and asserts no entry/no ranking. stdlib unittest only.

Run: python -m unittest discover -s tests/health -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))


def _flat_dates(n: int, end: date) -> list:
    return [end - timedelta(days=n - 1 - i) for i in range(n)]


class TestRSIFreezeGuard(unittest.TestCase):
    """The confirmed real bug: RSI locks at its pre-flatline value forever
    once delta=0 every bar, instead of reverting to a neutral reading."""

    def test_s2_rsi_compute_goes_nan_on_flat_tail(self):
        import phase_t9b_meanreversion_paper_engine as s2
        close = np.concatenate([np.linspace(1.0, 0.5, 20), np.full(60, 0.5)])
        rsi = s2._compute_rsi(close, 14)
        self.assertTrue(np.isnan(rsi[-1]), "RSI must go NaN once the trailing window is flat")
        self.assertTrue(np.isnan(rsi[-10:]).all())

    def test_s8_rsi_compute_goes_nan_on_flat_tail(self):
        import phase_t9b_rsi_mr_funding_paper_engine as s8
        close = np.concatenate([np.linspace(1.0, 0.5, 20), np.full(60, 0.5)])
        rsi = s8._compute_rsi(close, 14)
        self.assertTrue(np.isnan(rsi[-1]))

    def test_s2_no_entry_signal_on_pure_flat_series(self):
        import phase_t9b_meanreversion_paper_engine as s2
        n = 60
        dates = _flat_dates(n, date(2026, 10, 10))
        close = np.full(n, 0.5)
        df = pd.DataFrame({
            "ts_ms": [int(pd.Timestamp(d, tz="UTC").value // 1_000_000) for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.zeros(n),
        })
        sig = s2.check_entry_signal("FLATUSDT", df, dates[-1])
        self.assertIsNone(sig, "a pure-flat series must never produce an entry signal")

    def test_s8_no_entry_signal_on_pure_flat_series(self):
        import phase_t9b_rsi_mr_funding_paper_engine as s8
        n = 60
        dates = _flat_dates(n, date(2026, 10, 10))
        close = np.full(n, 0.5)
        df = pd.DataFrame({
            "ts_ms": [int(pd.Timestamp(d, tz="UTC").value // 1_000_000) for d in dates],
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.zeros(n),
        })
        sig = s8.check_entry_signal("FLATUSDT", df, dates[-1])
        self.assertIsNone(sig)


class TestDonchianGuard(unittest.TestCase):
    def test_s1_no_entry_on_pure_flat_series(self):
        import phase_t9b_donchian_universev2_paper_engine as s1
        n = 250
        dates = _flat_dates(n, date(2026, 10, 10))
        close = np.full(n, 0.5)
        df = pd.DataFrame({
            "time": pd.to_datetime(dates, utc=True),
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.zeros(n),
        })
        sig = s1.check_entry_signal("FLATUSDT", df, dates[-1])
        self.assertIsNone(sig, "a flat series can never breach a Donchian high -- must stay None")


class TestConsecDownGuard(unittest.TestCase):
    def test_s3_no_entry_on_pure_flat_series(self):
        import phase_t9b_consecdowndays_paper_engine as s3
        n = 250
        dates = _flat_dates(n, date(2026, 10, 10))
        close = np.full(n, 0.5)
        df = pd.DataFrame({
            "time": pd.to_datetime(dates, utc=True),
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.zeros(n),
        })
        sig = s3.check_entry_signal("FLATUSDT", df, dates[-1])
        self.assertIsNone(sig, "consec_down can never accumulate on flat closes -- must stay None")


class TestMomentumRankGuard(unittest.TestCase):
    """The confirmed real bug: a flat symbol's 0% trailing return can rank
    inside the top/bottom basket whenever enough real peers are negative/
    positive -- exactly what put S5's STGUSDT in a basket on 2026-10-09."""

    def test_s5_flat_symbol_excluded_from_basket(self):
        import phase_t9b_momentum_factor_paper_engine as s5
        n = 60
        idx = pd.to_datetime(_flat_dates(n, date(2026, 10, 10)))
        cols = {}
        rng = np.random.default_rng(7)
        for i in range(15):
            # real symbols: random walk, clearly non-flat
            steps = rng.normal(0, 0.01, n)
            cols[f"REAL{i}USDT"] = 1.0 * np.exp(np.cumsum(steps))
        cols["FLATUSDT"] = np.full(n, 0.5)  # the dead symbol
        panel = pd.DataFrame(cols, index=idx)
        long_syms, short_syms, meta = s5.compute_basket(panel, idx[-1].date())
        self.assertNotIn("FLATUSDT", long_syms)
        self.assertNotIn("FLATUSDT", short_syms)


class TestCandidateRankingGuards(unittest.TestCase):
    """Candidate 12 (trailing return) and 19 (proximity-to-high) both rank
    the full universe every rebalance. A flat symbol's trailing return is a
    legitimate-looking 0% (candidate 12); its proximity is exactly 1.0, the
    metric's maximum (candidate 19) -- confirmed the most severe of the two."""

    def _panel(self, n=120):
        idx = pd.to_datetime(_flat_dates(n, date(2026, 10, 10)))
        rng = np.random.default_rng(3)
        panel = {}
        for i in range(15):
            steps = rng.normal(0, 0.01, n)
            panel[f"REAL{i}USDT"] = pd.DataFrame(
                {"close": 1.0 * np.exp(np.cumsum(steps))}, index=idx)
        panel["FLATUSDT"] = pd.DataFrame({"close": np.full(n, 0.5)}, index=idx)
        return panel

    def test_candidate12_excludes_flat_symbol(self):
        from signal_generators import candidate_12_cross_sectional_momentum as c12
        panel = self._panel()
        result = c12.generate_universe_positions(
            panel, {"formation": 30, "quantile": 0.2, "rebalance_n": 30, "side": "long_short"})
        self.assertTrue((result["FLATUSDT"] == 0).all(),
                         "a flat symbol must never be assigned a long or short position")

    def test_candidate19_excludes_flat_symbol(self):
        from signal_generators import candidate_19_proximity_high_momentum as c19
        panel = self._panel()
        result = c19.generate_universe_positions(
            panel, {"lookback": 60, "quantile": 0.2, "rebalance_n": 30, "side": "long_short"})
        self.assertTrue((result["FLATUSDT"] == 0).all(),
                         "proximity=1.0 on flat data must not win the top decile")


if __name__ == "__main__":
    unittest.main()
