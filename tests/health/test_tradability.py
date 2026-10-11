#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exchange-faithful tradability (engines/tradability.py) and its engine integration.
All inputs are synthetic (a temp exchange-status cache, temp config/ledger/health paths); nothing
reads or writes live state. stdlib unittest + unittest.mock only.

Run: python -m unittest discover -s tests/health -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))

import health_emit  # noqa: E402
import tradability as T  # noqa: E402

FETCHED = datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc).isoformat()


def _cache(extra_fut=None, extra_spot=None) -> dict:
    fut = {
        "FUT:GOODUSDT": {"status": "TRADING", "venue": "futures", "delivery_date": None, "onboard_date": "2023-01-01"},
        "FUT:DEADUSDT": {"status": "SETTLING", "venue": "futures", "delivery_date": "2026-06-11", "onboard_date": "2022-01-01"},
        "FUT:NEWUSDT":  {"status": "TRADING", "venue": "futures", "delivery_date": None, "onboard_date": "2026-09-01"},
        "FUT:PENDUSDT": {"status": "PENDING_TRADING", "venue": "futures", "delivery_date": None, "onboard_date": "2026-12-01"},
        "FUT:SOONUSDT": {"status": "SETTLING", "venue": "futures", "delivery_date": "2026-10-20", "onboard_date": "2022-01-01"},
    }
    spot = {
        "GOODUSDT": {"status": "TRADING", "venue": "spot", "delivery_date": None},
        "HALTUSDT": {"status": "BREAK", "venue": "spot", "delivery_date": None},
    }
    fut.update(extra_fut or {})
    spot.update(extra_spot or {})
    return {"fetched_at": FETCHED, "symbols": {**fut, **spot}}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.cache_path = t / "cache.json"
        self.cache_path.write_text(json.dumps(_cache()), encoding="utf-8")
        self.cfg_path = t / "tradability_config.json"
        self.patches = [
            mock.patch.object(T, "CACHE_PATH", self.cache_path),
            mock.patch.object(T, "CONFIG_PATH", self.cfg_path),
            mock.patch.object(T, "VOID_LEDGER", t / "ledger.csv"),
            mock.patch.object(health_emit, "EVENTS_PATH", t / "health_events.jsonl"),
            mock.patch.object(health_emit, "HEALTH_DIR", t),
        ]
        for p in self.patches:
            p.start()
        T.reload(self.cache_path)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        T.reload()
        self.tmp.cleanup()

    def events(self):
        p = Path(self.tmp.name) / "health_events.jsonl"
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


class TestTradableByDate(_Base):
    def test_trading_symbol_is_tradable(self):
        self.assertTrue(T.is_tradable("GOODUSDT", "2026-10-10"))

    def test_delivery_date_itself_is_not_tradable_day_before_is(self):
        self.assertTrue(T.is_tradable("DEADUSDT", "2026-06-10"))
        self.assertFalse(T.is_tradable("DEADUSDT", "2026-06-11"))
        self.assertEqual(T.untradable_reason("DEADUSDT", "2026-07-30"), "on_or_after_delivery")

    def test_before_onboard(self):
        self.assertEqual(T.untradable_reason("NEWUSDT", "2026-08-31"), "before_onboard")
        self.assertTrue(T.is_tradable("NEWUSDT", "2026-09-01"))

    def test_status_rule_applies_in_live_window_only(self):
        # SOON is SETTLING with delivery 2026-10-20: not tradable today (status), but a replay
        # dated a month ago relies on dates only and is tradable.
        self.assertEqual(T.untradable_reason("SOONUSDT", "2026-10-10"), "status_SETTLING")
        self.assertIsNone(T.untradable_reason("SOONUSDT", "2026-08-01"))
        self.assertEqual(T.untradable_reason("PENDUSDT", "2026-10-10"), "before_onboard")

    def test_unknown_symbol_fails_closed(self):
        self.assertEqual(T.untradable_reason("NOPEUSDT", "2026-10-10"), "unknown_symbol")

    def test_symbol_formats_normalise(self):
        self.assertTrue(T.is_tradable("GOOD/USDT", "2026-10-10"))
        self.assertTrue(T.is_tradable("GOOD_USDT", "2026-10-10"))
        self.assertFalse(T.is_tradable("DEAD/USDT", "2026-10-10"))

    def test_spot_venue_uses_status_not_futures_delivery(self):
        self.assertTrue(T.is_tradable("GOOD/USDT", "2026-10-10", "spot"))
        self.assertEqual(T.untradable_reason("HALT/USDT", "2026-10-10", "spot"), "status_BREAK")
        # DEAD exists on futures only: unknown on spot -> fails closed there
        self.assertEqual(T.untradable_reason("DEADUSDT", "2026-10-10", "spot"), "unknown_symbol")

    def test_missing_cache_disables_gate_and_emits_critical(self):
        self.cache_path.write_text("{ not json", encoding="utf-8")
        T.reload(self.cache_path)
        self.assertTrue(T.is_tradable("ANYTHINGUSDT", "2026-10-10"))
        self.assertTrue(any(e.get("check") == "exchange_status_cache_unusable" and e.get("severity") == "CRITICAL"
                            for e in self.events()))

    def test_tradable_universe_filters(self):
        out = T.tradable_universe("T", ["GOODUSDT", "DEADUSDT", "NOPEUSDT", "NEWUSDT"], "2026-10-10")
        self.assertEqual(out, ["GOODUSDT", "NEWUSDT"])


class TestPositionActions(_Base):
    def test_entry_after_delivery_is_void(self):
        a = T.position_action("DEADUSDT", "2026-09-29", "2026-10-10")
        self.assertEqual((a.kind, a.reason), ("void", "on_or_after_delivery"))

    def test_entry_on_delivery_date_is_void(self):
        self.assertEqual(T.position_action("DEADUSDT", "2026-06-11", "2026-06-12").kind, "void")

    def test_entry_before_delivery_then_delivery_reached_settles(self):
        with mock.patch.object(T, "settlement_price", return_value=0.5):
            a = T.position_action("DEADUSDT", "2026-06-01", "2026-06-11")
        self.assertEqual((a.kind, a.price, a.delivery), ("settle", 0.5, date(2026, 6, 11)))

    def test_no_action_before_delivery(self):
        self.assertEqual(T.position_action("DEADUSDT", "2026-06-01", "2026-06-10").kind, "none")

    def test_healthy_position_untouched(self):
        self.assertEqual(T.position_action("GOODUSDT", "2026-10-01", "2026-10-10").kind, "none")

    def test_pending_action_is_inert_until_enabled(self):
        self.assertFalse(T.apply_state_fixes_enabled())
        self.assertIsNone(T.pending_action("S2", "DEADUSDT", "2026-09-29", "2026-10-10", "pid1"))
        pend = [e for e in self.events() if e.get("check") == "tradability_state_fix_pending"]
        self.assertTrue(pend and pend[-1].get("severity") == "WARNING", "a pending fix must be visible in health")

    def test_pending_action_applies_when_enabled(self):
        self.cfg_path.write_text(json.dumps({"apply_state_fixes": True}), encoding="utf-8")
        a = T.pending_action("S2", "DEADUSDT", "2026-09-29", "2026-10-10", "pid1")
        self.assertIsNotNone(a)
        self.assertEqual(a.kind, "void")

    def test_ledger_is_append_only(self):
        T.ledger_append("S2", "void", "DEADUSDT", "2026-09-29", "2026-10-10", "x")
        T.ledger_append("S2", "settle", "DEADUSDT", "2026-06-01", "2026-06-11", "y", 0.5)
        rows = (Path(self.tmp.name) / "ledger.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)  # header + 2


class TestSettlementPrice(unittest.TestCase):
    def test_settlement_is_close_of_delivery_bar(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cache = td / "c.json"
            cache.write_text(json.dumps(_cache()), encoding="utf-8")
            idx = pd.date_range("2026-06-08", periods=6, freq="D", tz="UTC")
            close = [1.0, 0.9, 0.8, 0.7, 0.7, 0.7]
            vol = [10, 10, 10, 10, 0, 0]  # delivery bar 2026-06-11 is the last with volume
            pd.DataFrame({"timestamp": [int(t.value // 1_000_000) for t in idx], "close": close, "volume": vol}
                         ).to_csv(td / "DEADUSDT_1d.csv", index=False)
            with mock.patch.object(T, "CACHE_PATH", cache), mock.patch.object(T, "OHLCV_1D", td):
                T.reload(cache)
                self.assertAlmostEqual(T.settlement_price("DEADUSDT"), 0.7)
            T.reload()


class TestS2Integration(_Base):
    """S2's run_one_day with a synthetic state: voids/settlements only when enabled."""

    def _state(self, entry_date):
        return {"system_name": "T", "paper_equity_usdt": 10000.0, "peak_equity_usdt": 10000.0,
                "kill_switch_triggered": False, "closed_trade_count": 0, "last_run_date": None,
                "open_positions": [{
                    "position_id": f"DEADUSDT_{entry_date.replace('-', '')}", "symbol": "DEADUSDT",
                    "entry_date": entry_date, "entry_price": 0.5, "stop_loss": 0.45,
                    "initial_risk_per_unit": 0.05, "risk_amount_usdt": 25.0, "qty": 1000.0,
                    "bars_held": 3, "atr_at_entry": 0.02, "rsi_at_entry": 20.0}]}

    def test_void_removes_position_no_pnl_and_logs(self):
        import phase_t9b_meanreversion_paper_engine as s2
        self.cfg_path.write_text(json.dumps({"apply_state_fixes": True}), encoding="utf-8")
        st = self._state("2026-09-29")
        res = s2.run_one_day(date(2026, 10, 10), [], st)
        self.assertEqual(res["open_count"], 0)
        self.assertEqual(res["exits"], 0, "a void is not a trade: no exit, no closed_trade_count")
        self.assertEqual(st["closed_trade_count"], 0)
        self.assertEqual(st["paper_equity_usdt"], 10000.0)
        self.assertIn("ENTRY_VOIDED", [e["event"] for e in res["log_events"]])

    def test_disabled_flag_leaves_state_untouched(self):
        import phase_t9b_meanreversion_paper_engine as s2
        st = self._state("2026-09-29")
        with mock.patch.object(s2, "load_ohlcv", return_value=pd.DataFrame()):
            res = s2.run_one_day(date(2026, 10, 10), [], st)
        self.assertEqual(res["open_count"], 1)
        self.assertNotIn("ENTRY_VOIDED", [e["event"] for e in res["log_events"]])

    def test_settlement_books_pnl_at_settlement_price_on_delivery_date(self):
        import phase_t9b_meanreversion_paper_engine as s2
        self.cfg_path.write_text(json.dumps({"apply_state_fixes": True}), encoding="utf-8")
        st = self._state("2026-06-01")
        with mock.patch.object(T, "settlement_price", return_value=0.55):
            res = s2.run_one_day(date(2026, 6, 20), [], st)
        self.assertEqual(res["exits"], 1)
        ex = res["exits_today"][0]
        self.assertEqual(ex["exit_reason"], "exchange_settlement")
        self.assertEqual(ex["exit_date"], "2026-06-11")
        self.assertAlmostEqual(ex["exit_price"], 0.55)
        self.assertAlmostEqual(ex["gross_r"], (0.55 - 0.5) / 0.05, places=5)
        self.assertEqual(res["open_count"], 0)

    def test_entry_scan_excludes_untradable_symbols(self):
        import phase_t9b_meanreversion_paper_engine as s2
        seen = []
        real = s2.check_entry_signal

        def spy(sym, df, run_date):
            seen.append(sym)
            return None
        st = self._state("2026-06-01")
        st["open_positions"] = []
        with mock.patch.object(s2, "check_entry_signal", spy), \
                mock.patch.object(s2, "load_ohlcv", return_value=pd.DataFrame()):
            s2.run_one_day(date(2026, 10, 10), ["GOODUSDT", "DEADUSDT", "NOPEUSDT"], st)
        self.assertEqual(seen, ["GOODUSDT"], "delivered / unknown symbols must never reach the signal check")


class TestS5BasketGate(_Base):
    def test_delivered_symbol_excluded_from_basket_even_with_real_looking_history(self):
        import phase_t9b_momentum_factor_paper_engine as s5
        n = 60
        idx = pd.date_range(end="2026-10-10", periods=n, freq="D")
        rng = np.random.default_rng(11)
        cols = {f"G{i}USDT": 1.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))) for i in range(30)}
        # DEAD crashed hard into delivery (so it would rank in the short basket) but is delivered
        dead = np.linspace(1.0, 0.3, n)
        cols["DEADUSDT"] = dead
        panel = pd.DataFrame(cols, index=idx)
        extra = {f"FUT:G{i}USDT": {"status": "TRADING", "venue": "futures", "delivery_date": None,
                                   "onboard_date": "2023-01-01"} for i in range(30)}
        self.cache_path.write_text(json.dumps(_cache(extra_fut=extra)), encoding="utf-8")
        T.reload(self.cache_path)
        longs, shorts, _ = s5.compute_basket(panel, idx[-1].date())
        self.assertNotIn("DEADUSDT", shorts)
        self.assertNotIn("DEADUSDT", longs)
        self.assertTrue(len(shorts) > 0)


if __name__ == "__main__":
    unittest.main()
