#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exchange-faithful tradability, by date.

A USDT-M futures symbol (venue="futures", the default) is tradable on bar date D iff
    onboard_date <= D < delivery_date            (historical / replay use)
and, for a live decision (bar date within LIVE_WINDOW_DAYS of the cache date),
    status == "TRADING"                           (so a symbol in SETTLING /
                                                   BREAK / PENDING_TRADING
                                                   cannot be entered)
The delivery date itself is NOT tradable: a position in a contract that reaches
delivery is closed at the settlement price on that date (`exchange_settlement`),
and an entry dated on or after delivery is void (annotated, never deleted).

Source of truth: data/exchange_status_cache.json (one bulk exchangeInfo call per
venue, written by tools/health/refresh_exchange_status_cache.py on the host that
can reach fapi). This module only READS it. Replaces the manual
data/halted_symbols_suppression.json list.

Fail-closed rules
  * cache present, symbol absent          -> not tradable (reason "unknown_symbol")
  * cache missing/unreadable              -> everything tradable, one CRITICAL
                                             health finding (the gate is a
                                             refinement; losing the cache must not
                                             silently halt every engine)
  * cache older than CACHE_MAX_AGE_DAYS   -> still used (delivery dates are
                                             monotone), WARNING finding

Spot engines (S1, S3 trade the spot universe) pass venue="spot": spot symbols have no delivery date,
so only the live-window rule applies (status must be TRADING; BREAK/HALT/unknown -> not tradable).

Settlement price: the close of the OHLCV bar dated the delivery date (the last
bar with volume; every later bar in our files is a flat copy of that price).
State-changing application of voids/settlements to EXISTING positions is gated
by data/tradability_config.json {"apply_state_fixes": false} so nothing touches
live state until it is switched on deliberately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from health_emit import emit, resolve
except Exception:  # pragma: no cover - import path differs (package-style import)
    try:
        from engines.health_emit import emit, resolve
    except Exception:
        def emit(*a, **k):  # type: ignore
            return None

        def resolve(*a, **k):  # type: ignore
            return None

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "exchange_status_cache.json"
CONFIG_PATH = ROOT / "data" / "tradability_config.json"
VOID_LEDGER = ROOT / "data" / "tradability_ledger.csv"
OHLCV_1D = ROOT / "data" / "futures_universe" / "ohlcv_1d"
# Cache-age policy (never halts the systems): the exchange-status cache only REFINES the date rule.
#   <= CACHE_FRESH_DAYS      fresh   : full rule (status + dates; an unknown symbol is blocked)
#   <= CACHE_CRITICAL_DAYS   stale   : WARNING; fall back to the date-based rule (status ignored, unknown allowed)
#   <= CACHE_EXPIRED_DAYS    critical: same fallback, CRITICAL finding
#   >  CACHE_EXPIRED_DAYS    expired : same fallback, CRITICAL "expired" message
#   missing / unreadable     gate off, CRITICAL (data backstop below still applies)
# At every age the tradable set stays non-empty, so a stale cache can never stop all engines.
CACHE_FRESH_DAYS = 3
CACHE_CRITICAL_DAYS = 7
CACHE_EXPIRED_DAYS = 30
CACHE_MAX_AGE_DAYS = CACHE_FRESH_DAYS      # legacy alias
LIVE_WINDOW_DAYS = 3
FLAT_BACKSTOP_BARS = 3                     # data backstop: this many trailing flat zero-volume daily bars
SPOT_OHLCV_1D = ROOT / "data" / "universe" / "ohlcv_1d"

_CACHE: Optional[dict] = None          # futures symbol -> info
_SPOT: dict = {}                       # spot symbol -> info
_CACHE_META: dict = {}
_CACHE_BROKEN = False
_FLAT_CACHE: dict = {}                 # (venue, symbol) -> numpy arrays for the data backstop
_CFG_CACHE = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def norm(sym: str) -> str:
    """'BTC/USDT', 'BTC_USDT', 'FUT:BTCUSDT' -> 'BTCUSDT'."""
    s = str(sym).upper().replace("/", "").replace("-", "").replace("_", "")
    return s[4:] if s.startswith("FUT:") else s


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def reload(path: Path | None = None) -> None:
    """Force a re-read (tests, long-running sessions)."""
    global _CACHE, _SPOT, _CACHE_META, _CACHE_BROKEN, _FLAT_CACHE
    _CACHE, _SPOT, _CACHE_META, _CACHE_BROKEN, _FLAT_CACHE = None, {}, {}, False, {}
    _load(path)


def _load(path: Path | None = None) -> None:
    global _CACHE, _SPOT, _CACHE_META, _CACHE_BROKEN
    if _CACHE is not None or _CACHE_BROKEN:
        return
    p = path or CACHE_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        syms, spot = {}, {}
        for k, v in raw.get("symbols", {}).items():
            if k.startswith("FUT:"):
                syms[k[4:]] = v
            else:
                spot[k] = v
        if not syms:
            raise ValueError("no FUT: entries in exchange status cache")
        _CACHE, _SPOT = syms, spot
        _CACHE_META = {"fetched_at": raw.get("fetched_at")}
        resolve("shared/tradability", "exchange_status_cache_unusable")
        try:
            age = (_now() - datetime.fromisoformat(raw["fetched_at"])).total_seconds() / 86400
            mode = _mode_for_age(age)
            ev = {"age_days": round(age, 2), "fetched_at": raw.get("fetched_at"), "mode": mode}
            fix = ("Run tools/refresh_funding_local.py (or tools/health/refresh_exchange_status_cache.py) on the "
                   "funding host (fapi is geo-blocked from CI).")
            if mode == "fresh":
                resolve("shared/tradability", "exchange_status_cache_stale")
            else:
                sev = "WARNING" if mode == "stale" else "CRITICAL"
                tail = {"stale": "", "critical": " -- ESCALATED", "expired": " -- EXPIRED"}[mode]
                emit("shared/tradability", "exchange_status_cache_stale", sev,
                     f"exchange status cache is {age:.1f}d old (fresh limit {CACHE_FRESH_DAYS}d){tail}: falling back to "
                     f"the date-based rule (status ignored, unknown symbols allowed); new delistings are not seen "
                     f"until it is refreshed. Engines keep trading.",
                     evidence=ev, suggested_action=fix)
        except Exception:
            pass
    except Exception as exc:
        _CACHE_BROKEN = True
        emit("shared/tradability", "exchange_status_cache_unusable", "CRITICAL",
             f"data/exchange_status_cache.json unreadable ({exc}); tradability gate DISABLED "
             f"(every symbol treated as tradable) until it is restored",
             evidence={"path": str(p), "error": str(exc)},
             suggested_action="Run tools/health/refresh_exchange_status_cache.py or restore from git.")


def _mode_for_age(age: float) -> str:
    if age <= CACHE_FRESH_DAYS:
        return "fresh"
    if age <= CACHE_CRITICAL_DAYS:
        return "stale"
    if age <= CACHE_EXPIRED_DAYS:
        return "critical"
    return "expired"


def cache_age_days() -> Optional[float]:
    _load()
    f = _CACHE_META.get("fetched_at")
    if _CACHE is None or not f:
        return None
    try:
        return (_now() - datetime.fromisoformat(f)).total_seconds() / 86400
    except Exception:
        return None


def cache_mode() -> str:
    """fresh | stale | critical | expired | missing"""
    age = cache_age_days()
    return "missing" if age is None else _mode_for_age(age)


def _flat_feed(symbol: str, on_date, venue: str) -> bool:
    """Data backstop, independent of the exchange cache and of its age: the last FLAT_BACKSTOP_BARS daily bars on or
    before on_date all have zero volume and high == low. That is exactly what Binance serves for a contract that has
    settled/halted after the cache was last refreshed, and nothing can be traded on it. Point-in-time (uses only
    bars <= on_date), so it is safe in replays."""
    import numpy as np
    import pandas as pd
    key = (venue, norm(symbol))
    if key not in _FLAT_CACHE:
        n = norm(symbol)
        path = (OHLCV_1D / f"{n}_1d.csv") if venue != "spot" else (SPOT_OHLCV_1D / f"{n[:-4]}_USDT_1d.csv")
        arr = None
        try:
            if path.exists():
                cols = pd.read_csv(path, nrows=0).columns
                tcol = "timestamp" if "timestamp" in cols else "time"
                df = pd.read_csv(path, usecols=[tcol, "high", "low", "volume"])
                if tcol == "timestamp":
                    d = pd.to_datetime(df[tcol], unit="ms", utc=True)
                else:
                    d = pd.to_datetime(df[tcol], utc=True)
                flat = ((df["volume"].astype(float) == 0) & (df["high"].astype(float) == df["low"].astype(float))).to_numpy()
                arr = (d.dt.date.to_numpy(), flat)
        except Exception:
            arr = None
        _FLAT_CACHE[key] = arr
    arr = _FLAT_CACHE[key]
    if arr is None:
        return False
    dates, flat = arr
    d = _as_date(on_date)
    idx = int(np.searchsorted(dates, d, side="right"))
    if idx < FLAT_BACKSTOP_BARS:
        return False
    return bool(flat[idx - FLAT_BACKSTOP_BARS:idx].all())


@dataclass(frozen=True)
class Info:
    status: Optional[str]
    onboard: Optional[date]
    delivery: Optional[date]
    known: bool


def info(symbol: str, venue: str = "futures") -> Info:
    _load()
    if _CACHE is None:
        return Info(None, None, None, False)
    v = (_SPOT if venue == "spot" else _CACHE).get(norm(symbol))
    if v is None:
        return Info(None, None, None, False)
    ob = date.fromisoformat(v["onboard_date"]) if v.get("onboard_date") else None
    dl = date.fromisoformat(v["delivery_date"]) if v.get("delivery_date") else None
    return Info(v.get("status"), ob, dl, True)


def delivery_date(symbol: str) -> Optional[date]:
    return info(symbol).delivery


def untradable_reason(symbol: str, on_date, venue: str = "futures") -> Optional[str]:
    """None if tradable, else a short machine-readable reason. See the cache-age policy at the top.
    Order: exact delivery/onboard dates (strongest, survive any cache age) -> flat-feed backstop (needs no cache)
    -> unknown symbol / status (only trusted while the cache is fresh)."""
    _load()
    d = _as_date(on_date)
    mode = cache_mode()
    fresh = mode == "fresh"
    i = info(symbol, venue) if _CACHE is not None else Info(None, None, None, False)
    if i.known:
        if i.onboard is not None and d < i.onboard:
            return "before_onboard"
        if i.delivery is not None and d >= i.delivery:
            return "on_or_after_delivery"
        n = announced_delivery_block_days()
        if n and i.delivery is not None and (i.delivery - d).days <= n:
            return f"delivery_within_{n}d"       # announced delisting: no NEW entries (held positions settle normally)
    if _flat_feed(symbol, d, venue):
        return "flat_zero_volume_feed"          # backstop; needs no cache
    if _CACHE is None:  # cache missing/unreadable -> exchange gate off (CRITICAL already emitted)
        return None
    if not i.known:
        # a fresh cache that lacks the symbol means 'not (yet) a listed contract': block. A stale cache may simply
        # pre-date a new listing, so it must not block it.
        return "unknown_symbol" if fresh else None
    if not fresh:
        return None                              # date-based fallback: a stale status is not trusted
    fetched = _CACHE_META.get("fetched_at")
    try:
        fetched_d = datetime.fromisoformat(fetched).date() if fetched else None
    except Exception:
        fetched_d = None
    # live window: engines run on D+1 for bar D, so 'today' decisions carry a bar date up to
    # LIVE_WINDOW_DAYS before the cache date; older (replay) dates use onboard/delivery only.
    if fetched_d is not None and (fetched_d - d).days <= LIVE_WINDOW_DAYS and i.status != "TRADING":
        return f"status_{i.status}"
    return None


def is_tradable(symbol: str, on_date, venue: str = "futures") -> bool:
    return untradable_reason(symbol, on_date, venue) is None


def filter_tradable(symbols: Iterable[str], on_date, venue: str = "futures") -> list[str]:
    return [s for s in symbols if is_tradable(s, on_date, venue)]


def settlement_price(symbol: str) -> Optional[float]:
    """Close of the bar dated the delivery date (fallback: last bar with volume before it)."""
    dd = delivery_date(symbol)
    if dd is None:
        return None
    import pandas as pd
    path = OHLCV_1D / f"{norm(symbol)}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["timestamp", "close", "volume"])
    df["d"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    on = df[df["d"] == dd]
    if len(on) and float(on["volume"].iloc[-1]) > 0:
        return float(on["close"].iloc[-1])
    before = df[(df["d"] < dd) & (df["volume"] > 0)]
    return float(before["close"].iloc[-1]) if len(before) else None


@dataclass(frozen=True)
class Action:
    kind: str                      # "void" | "settle" | "none"
    symbol: str
    delivery: Optional[date] = None
    price: Optional[float] = None
    reason: str = ""


def position_action(symbol: str, entry_date, run_date, venue: str = "futures") -> Action:
    """
    What the exchange rules require for an open position as of run_date.
      void   : entry_date >= delivery (or before onboard / unknown symbol)
      settle : entry before delivery and run_date >= delivery -> close at settlement
      none   : nothing to do
    """
    why = untradable_reason(symbol, entry_date, venue)
    if why in ("on_or_after_delivery", "before_onboard", "unknown_symbol", "flat_zero_volume_feed"):
        return Action("void", norm(symbol), info(symbol, venue).delivery, None, why)
    dd = info(symbol, venue).delivery
    if dd is not None and _as_date(run_date) >= dd:
        return Action("settle", norm(symbol), dd, settlement_price(symbol), "reached_delivery")
    return Action("none", norm(symbol))


def apply_state_fixes_enabled() -> bool:
    """Existing-position voids/settlements only mutate state when explicitly enabled."""
    try:
        return bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("apply_state_fixes", False))
    except Exception:
        return False


def ledger_append(engine: str, kind: str, symbol: str, entry_date, run_date, detail: str,
                  price: float | None = None) -> None:
    """Append-only record of every void/settlement so nothing is silently deleted."""
    import csv
    new = not VOID_LEDGER.exists()
    VOID_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(VOID_LEDGER, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["logged_utc", "engine", "kind", "symbol", "entry_date", "run_date", "price", "detail"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), engine, kind, norm(symbol),
                    str(entry_date), str(run_date), "" if price is None else price, detail])


def pending_action(engine: str, symbol: str, entry_date, run_date, position_id: str = "",
                   venue: str = "futures") -> Optional[Action]:
    """
    Engine-facing wrapper around position_action(). Returns the Action to APPLY, or None.
    While data/tradability_config.json has apply_state_fixes=false (the default) nothing is
    applied to existing positions: a WARNING finding records that a fix is pending and the
    engine carries on exactly as before.
    """
    act = position_action(symbol, entry_date, run_date, venue)
    if act.kind == "none":
        resolve(engine, "tradability_state_fix_pending", key=position_id or norm(symbol))
        return None
    if not apply_state_fixes_enabled():
        emit(engine, "tradability_state_fix_pending", "WARNING",
             f"{norm(symbol)} (entry {entry_date}): exchange rules require action '{act.kind}' "
             f"({act.reason}, delivery {act.delivery}); NOT applied -- apply_state_fixes is off",
             key=position_id or norm(symbol),
             evidence={"symbol": norm(symbol), "entry_date": str(entry_date), "kind": act.kind,
                       "delivery": str(act.delivery), "reason": act.reason},
             suggested_action="Review docs/TRADABILITY_REPORT.md, then set apply_state_fixes=true "
                              "in data/tradability_config.json.",
             run_date=str(run_date))
        return None
    return act


def tradable_universe(engine: str, symbols: Iterable[str], run_date, venue: str = "futures") -> list[str]:
    """Entry-universe filter. Prints a one-line count; no health finding per excluded symbol."""
    syms = list(symbols)
    ok = [s for s in syms if is_tradable(s, run_date, venue)]
    if len(ok) != len(syms):
        print(f"  [TRADABILITY] {engine}: {len(syms) - len(ok)} of {len(syms)} symbols excluded "
              f"as not tradable on {run_date} (delivered / settling / not onboarded / unknown)", flush=True)
    return ok


def _config() -> dict:
    """data/tradability_config.json, re-read only when the file changes."""
    global _CFG_CACHE
    try:
        m = CONFIG_PATH.stat().st_mtime
        if _CFG_CACHE is None or _CFG_CACHE[0] != (str(CONFIG_PATH), m):
            _CFG_CACHE = ((str(CONFIG_PATH), m), json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        return _CFG_CACHE[1]
    except Exception:
        return {}


def announced_delivery_block_days() -> int:
    """New entries are refused when the exchange has ANNOUNCED a delivery date within this many days
    (default 7; 0 = off). Positions already held are not touched: they settle on the delivery date."""
    try:
        return max(0, int(_config().get("announced_delivery_block_days", 7)))
    except Exception:
        return 7


def regime_exclusion_enabled() -> bool:
    """Switch (default OFF): exclude non-tradable symbols from the regime classifier's breadth and funding inputs."""
    try:
        return bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("regime_exclude_nontradable", False))
    except Exception:
        return False
