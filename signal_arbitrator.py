"""
T9B Signal Arbitration Manager.

Coordinates all six T9B engines before any position entry.
Each engine instantiates SignalArbitrator(system, run_date) and calls
check_signal(symbol, direction, heat_usdt) per candidate signal.

Decisions are logged to data/t9b_arbitration/daily_log.csv.

Rules enforced:
  Rule 1 — direction_conflict: opposite-direction position exists in another engine
  Rule 2 — duplicate_held:     same-direction position exists in another engine
  Rule 3 — lower_priority_*:   same symbol+direction wanted by higher-priority engine
  Rule 4 — portfolio/system heat limits
  Rule 5 — momentum_short_conflict / {engine}_short_conflict:
           spot LONG blocked by any Futures SHORT (Momentum basket,
           VolContraction System 7, MACross System 8)
  Rule 6 — futures_short_heat_exceeded: System 7 + System 8 combined open
           risk <= 5% of the $20k Futures short pool ($1,000)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import t9b_shared

ROOT    = Path(__file__).resolve().parent
ARB_LOG = ROOT / "data" / "t9b_arbitration" / "daily_log.csv"

# Priority order per regime — lower index wins
# BEAR: VolContraction (System 7) ranks above MACross (System 8) — higher avg_r.
# MIXED: no defined ranks — first signal in session wins.
PRIORITY: dict[str, list[str]] = {
    "BULL": ["donchian", "momentum", "consecdown", "rsi_mr",
             "rsi_mr_funding", "volcontraction", "macross"],
    "BEAR": ["rsi_mr_funding", "rsi_mr", "volcontraction", "macross",
             "momentum", "donchian", "consecdown"],
}

# Engines whose open_positions are SHORT (Futures short specialists)
SHORT_ENGINES = ("volcontraction", "macross")

# Heat limits
PORTFOLIO_HEAT_CAP     = 3_200.0   # 8% of $40k combined portfolio
SYSTEM_HEAT_FRAC       = 0.05      # 5% of each $10k pool = $500
FUTURES_SHORT_POOL     = 20_000.0  # System 7 ($10k) + System 8 ($10k)
FUTURES_SHORT_HEAT_CAP = FUTURES_SHORT_POOL * 0.05   # $1,000 combined

_STATE_PATHS: dict[str, Path] = {
    "donchian":       ROOT / "data" / "t9b_paper"                / "state.json",
    "rsi_mr":         ROOT / "data" / "t9b_mr_paper"             / "state.json",
    "consecdown":     ROOT / "data" / "t9b_consecdowndays_paper"  / "state.json",
    "momentum":       ROOT / "data" / "t9b_momentum_paper"        / "state.json",
    "volcontraction": ROOT / "data" / "t9b_volcontraction_paper"  / "state.json",
    "macross":        ROOT / "data" / "t9b_macross_paper"         / "state.json",
    "rsi_mr_funding": ROOT / "data" / "t9b_rsi_mr_funding_paper"  / "state.json",
}

_BTC_PATHS = [
    ROOT / "data" / "universe"        / "ohlcv_1d" / "BTC_USDT_1d.csv",
    ROOT / "data" / "futures_universe" / "ohlcv_1d" / "BTCUSDT_1d.csv",
]


class SignalArbitrator:
    """
    Per-run arbitration instance.  Create once per engine invocation;
    call check_signal() for each candidate signal before creating the position.
    """

    def __init__(
        self,
        calling_system: str,
        run_date: date,
        fresh_heat: bool = False,
    ) -> None:
        """
        Args:
            calling_system: 'donchian' | 'rsi_mr' | 'consecdown' | 'momentum'
                            | 'volcontraction' | 'macross'
            run_date:       date of the current engine run (for logging)
            fresh_heat:     if True, zero out calling system's heat (use for
                            Momentum rebalance which clears all old positions)
        """
        self.system   = calling_system
        self.run_date = run_date
        self._regime_cache: Optional[str] = None

        ARB_LOG.parent.mkdir(parents=True, exist_ok=True)

        self._eng: dict[str, dict] = self._load_states()
        if fresh_heat and calling_system in self._eng:
            self._eng[calling_system]["heat"] = 0.0

        # In-session approvals: norm_sym -> {system, direction}
        # Tracks what THIS engine has already approved this run (same-process only).
        self._session: dict[str, dict] = {}

    # ── state loading ──────────────────────────────────────────────────────

    def _load_states(self) -> dict:
        result: dict[str, dict] = {}
        for eng, path in _STATE_PATHS.items():
            result[eng] = {"by_dir": {}, "equity": 10_000.0, "heat": 0.0}
            if not path.exists():
                continue
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
                by_dir: dict[str, str] = {}
                heat = 0.0
                # open_positions: LONG for spot engines, SHORT for the
                # Futures short specialists (volcontraction, macross)
                pos_dir = "SHORT" if eng in SHORT_ENGINES else "LONG"
                for pos in s.get("open_positions", []):
                    sym = pos.get("symbol", "")
                    if sym:
                        by_dir[t9b_shared.normalize_sym(sym)] = pos_dir
                        heat += float(pos.get("risk_amount_usdt", 0.0))
                # Momentum longs (1% of notional as heat proxy)
                for pos in s.get("long_positions", []):
                    sym = pos.get("symbol", "")
                    if sym:
                        by_dir[t9b_shared.normalize_sym(sym)] = "LONG"
                        heat += float(pos.get("notional", 0.0)) * 0.01
                # Momentum shorts (2% of notional — shorts carry more tail risk)
                for pos in s.get("short_positions", []):
                    sym = pos.get("symbol", "")
                    if sym:
                        by_dir[t9b_shared.normalize_sym(sym)] = "SHORT"
                        heat += float(pos.get("notional", 0.0)) * 0.02
                result[eng]["by_dir"] = by_dir
                result[eng]["equity"] = float(s.get("paper_equity_usdt", 10_000.0))
                result[eng]["heat"]   = round(heat, 4)
            except Exception:
                pass
        return result

    # ── regime detection ──────────────────────────────────────────────────

    def detect_regime(self) -> str:
        """Return 'BULL', 'BEAR', or 'MIXED' based on BTC vs EMA200."""
        if self._regime_cache is not None:
            return self._regime_cache
        import pandas as pd
        for path in _BTC_PATHS:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, parse_dates=["date"], index_col="date")
                if "close" not in df.columns:
                    continue
                closes = df["close"].sort_index().dropna()
                if len(closes) < 50:
                    continue
                ema200 = closes.ewm(span=200, min_periods=50, adjust=False).mean()
                cutoff = closes.index[closes.index <= pd.Timestamp(self.run_date)]
                if len(cutoff) == 0:
                    continue
                last  = cutoff[-1]
                price = float(closes.loc[last])
                ema   = float(ema200.loc[last])
                regime = "BULL" if price > ema else "BEAR"
                self._regime_cache = regime
                return regime
            except Exception:
                continue
        self._regime_cache = "MIXED"
        return "MIXED"

    # ── priority ──────────────────────────────────────────────────────────

    def _priority_rank(self, system: str) -> int:
        """Lower rank = higher priority.  999 = no defined rank (MIXED)."""
        order = PRIORITY.get(self.detect_regime())
        if order is None:
            return 999
        try:
            return order.index(system)
        except ValueError:
            return 999

    # ── heat helpers ──────────────────────────────────────────────────────

    def _total_heat(self) -> float:
        return sum(v["heat"] for v in self._eng.values())

    def _system_heat(self) -> float:
        return self._eng.get(self.system, {}).get("heat", 0.0)

    def _system_equity(self) -> float:
        return self._eng.get(self.system, {}).get("equity", 10_000.0)

    # ── main API ──────────────────────────────────────────────────────────

    def check_signal(
        self,
        symbol: str,
        direction: str,
        heat_usdt: float,
    ) -> tuple[str, str]:
        """
        Evaluate a proposed signal against all six arbitration rules.

        Args:
            symbol:    raw symbol string ('BTC/USDT', 'BTCUSDT', etc.)
            direction: 'LONG' or 'SHORT'
            heat_usdt: risk dollars this position adds to the system's heat
                       (spot/futures-short: risk_amount_usdt = equity*0.25%;
                        momentum long: alloc*0.01; momentum short: alloc*0.02)

        Returns:
            ('APPROVED', '') or ('REJECTED', reason_string)
        """
        norm      = t9b_shared.normalize_sym(symbol)
        direction = direction.upper()

        # Rule 5 — any Futures SHORT blocks spot LONG (evaluated before generic
        #           check so we can emit the specific '*_short_conflict' reason)
        if direction == "LONG":
            for short_eng in ("momentum",) + SHORT_ENGINES:
                if self.system == short_eng:
                    continue
                eng_by_dir = self._eng.get(short_eng, {}).get("by_dir", {})
                if eng_by_dir.get(norm) == "SHORT":
                    return self._reject(symbol, direction,
                                        f"{short_eng}_short_conflict")

        # Rules 1 + 2 — check held positions in other engines
        for eng, eng_state in self._eng.items():
            if eng == self.system:
                continue
            held = eng_state["by_dir"].get(norm)
            if held is None:
                continue
            if held == direction:
                return self._reject(symbol, direction, "duplicate_held")
            else:
                return self._reject(symbol, direction, "direction_conflict")

        # Rule 3 — in-session priority (within this engine's run)
        if norm in self._session:
            prev = self._session[norm]
            if prev["direction"] != direction:
                return self._reject(symbol, direction, "direction_conflict")
            # Same direction already approved this session.
            # Compare priorities: higher-ranked system should have entered first
            # (bat file order establishes effective priority at cross-engine level).
            # Within a single engine, the same symbol won't appear twice.
            my_rank   = self._priority_rank(self.system)
            prev_rank = self._priority_rank(prev["system"])
            regime    = self.detect_regime()
            if regime == "MIXED":
                # first-in-session wins
                return self._reject(symbol, direction, "lower_priority_mixed")
            if my_rank >= prev_rank:
                return self._reject(symbol, direction, f"lower_priority_{regime.lower()}")

        # Rule 4 — portfolio heat check
        total = self._total_heat()
        if total + heat_usdt > PORTFOLIO_HEAT_CAP:
            return self._reject(
                symbol, direction,
                f"portfolio_heat_exceeded  "
                f"current={total:.0f}  adding={heat_usdt:.0f}  cap={PORTFOLIO_HEAT_CAP:.0f}",
            )
        sys_heat  = self._system_heat()
        sys_limit = self._system_equity() * SYSTEM_HEAT_FRAC
        if sys_heat + heat_usdt > sys_limit:
            return self._reject(
                symbol, direction,
                f"system_heat_exceeded  "
                f"sys_heat={sys_heat:.0f}  adding={heat_usdt:.0f}  limit={sys_limit:.0f}",
            )

        # Rule 6 — combined Futures short heat (System 7 + System 8)
        if self.system in SHORT_ENGINES:
            short_heat = sum(self._eng.get(e, {}).get("heat", 0.0)
                             for e in SHORT_ENGINES)
            if short_heat + heat_usdt > FUTURES_SHORT_HEAT_CAP:
                return self._reject(
                    symbol, direction,
                    f"futures_short_heat_exceeded  "
                    f"combined={short_heat:.0f}  adding={heat_usdt:.0f}  "
                    f"cap={FUTURES_SHORT_HEAT_CAP:.0f}",
                )

        # Approved — record in session + update heat so subsequent calls see it
        self._session[norm] = {"system": self.system, "direction": direction}
        self._eng[self.system]["heat"] = sys_heat + heat_usdt

        return self._approve(symbol, direction)

    # ── logging ───────────────────────────────────────────────────────────

    def _log(self, symbol: str, direction: str, decision: str, reason: str) -> None:
        exists = ARB_LOG.exists()
        with open(ARB_LOG, "a", encoding="utf-8", newline="") as f:
            if not exists:
                f.write("date,symbol,system,direction,decision,reason\n")
            safe_reason = reason.replace(",", ";").replace('"', "'")
            f.write(
                f"{self.run_date},{symbol},{self.system},"
                f"{direction},{decision},{safe_reason}\n"
            )

    def _approve(self, symbol: str, direction: str) -> tuple[str, str]:
        self._log(symbol, direction, "APPROVED", "")
        return ("APPROVED", "")

    def _reject(self, symbol: str, direction: str, reason: str) -> tuple[str, str]:
        self._log(symbol, direction, "REJECTED", reason)
        return ("REJECTED", reason)
