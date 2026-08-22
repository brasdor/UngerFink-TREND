#!/usr/bin/env python3
"""
T9B shared utilities — cross-system symbol deduplication.
Imported by all T9B paper engines (Fix 1 support).
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERROR_LOG = ROOT / "logs" / "t9b_task_scheduler.log"

_STATE_PATHS: dict[str, Path] = {
    "donchian":       ROOT / "data" / "t9b_paper"                  / "state.json",
    "rsi_mr":         ROOT / "data" / "t9b_mr_paper"               / "state.json",
    "consecdown":     ROOT / "data" / "t9b_consecdowndays_paper"    / "state.json",
    "momentum":       ROOT / "data" / "t9b_momentum_paper"          / "state.json",
    "volcontraction": ROOT / "data" / "t9b_volcontraction_paper"    / "state.json",
    "macross":        ROOT / "data" / "t9b_macross_paper"           / "state.json",
    "rsi_mr_funding": ROOT / "data" / "t9b_rsi_mr_funding_paper"    / "state.json",
}


def normalize_sym(sym: str) -> str:
    """Normalize to BASEQUOTE: 'BTC/USDT' or 'BTCUSDT' -> 'BTCUSDT'."""
    return sym.replace("/", "").replace("-", "").upper()


def get_cross_system_symbols(exclude: str) -> frozenset[str]:
    """
    Return frozenset of normalized symbols currently open in all T9B engines
    except `exclude` ('donchian', 'rsi_mr', 'consecdown', 'momentum',
    'volcontraction', 'macross').
    Reads state.json files on disk — fast JSON-only operation.
    """
    syms: set[str] = set()
    for engine, path in _STATE_PATHS.items():
        if engine == exclude:
            continue
        if not path.exists():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            # Spot engines use open_positions
            for pos in state.get("open_positions", []):
                s = pos.get("symbol", "")
                if s:
                    syms.add(normalize_sym(s))
            # Momentum engine uses long_positions + short_positions
            for pos in (state.get("long_positions", []) +
                        state.get("short_positions", [])):
                s = pos.get("symbol", "")
                if s:
                    syms.add(normalize_sym(s))
        except Exception:
            pass
    return frozenset(syms)


_REGIME_STATE_PATH   = ROOT / "data" / "regime_state.json"
_REGIME_HISTORY_PATH = ROOT / "data" / "regime_history.csv"

_ENGINE_TO_SYSTEM = {
    "donchian":       "S1",
    "rsi_mr":         "S2",
    "consecdown":     "S3",
    "momentum":       "S5",
    "volcontraction": "S6",
    "macross":        "S7",
    "rsi_mr_funding": "S8",
}

_ENGINE_TO_WCOL = {
    "donchian":       "w_S1",
    "rsi_mr":         "w_S2",
    "consecdown":     "w_S3",
    "momentum":       "w_S5",
    "volcontraction": "w_S6",
    "macross":        "w_S7",
    "rsi_mr_funding": "w_S8",
}

# Equal weight when 7 active systems
_N_SYSTEMS = 7
_EQ_WEIGHT = 1.0 / _N_SYSTEMS

_regime_history_cache = None  # lazily loaded, indexed by date


def _load_regime_history():
    global _regime_history_cache
    if _regime_history_cache is not None:
        return _regime_history_cache
    if not _REGIME_HISTORY_PATH.exists():
        _regime_history_cache = False
        return False
    try:
        import pandas as pd
        hist = pd.read_csv(_REGIME_HISTORY_PATH)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
        _regime_history_cache = hist.set_index("date").sort_index()
    except Exception:
        _regime_history_cache = False
    return _regime_history_cache


def get_regime_weight(engine_name: str, run_date=None) -> tuple[float, float]:
    """
    Return (system_weight, vol_multiplier) for the given engine.

    run_date=None (or today): reads data/regime_state.json, today's live
    snapshot -- unchanged behavior for normal daily runs.

    run_date=<a past date>: reads data/regime_history.csv (a genuine daily
    record) for the most recent row on or before run_date, instead of
    always applying today's static weight. Without this, a --backfill or
    multi-day catch-up run silently retroactively applies TODAY's regime
    weight to every historical entry, mis-sizing any position opened under
    a different historical regime -- confirmed to happen (see the S1
    backfill on 2026-08-22: JST/DEXE were sized off MIXED/0.1429 when the
    actual regime on their entry dates was BEAR/0.05).

    Falls back to equal weight (1/7) / vol_mult=1.0 if neither source is
    available, and to regime_history.csv's earliest row for a run_date
    older than the history file's coverage.
    """
    from datetime import date as _date

    sys_id = _ENGINE_TO_SYSTEM.get(engine_name)
    if sys_id is None:
        return _EQ_WEIGHT, 1.0

    is_historical = run_date is not None and run_date < _date.today()

    if is_historical:
        hist = _load_regime_history()
        wcol = _ENGINE_TO_WCOL.get(engine_name)
        if hist is not False and wcol is not None and not hist.empty:
            avail = hist.index[hist.index <= run_date]
            row_date = avail.max() if len(avail) else hist.index.min()
            try:
                row = hist.loc[row_date]
                return float(row[wcol]), float(row["vol_mult"])
            except Exception:
                pass
        return _EQ_WEIGHT, 1.0

    if not _REGIME_STATE_PATH.exists():
        return _EQ_WEIGHT, 1.0

    try:
        state = json.loads(_REGIME_STATE_PATH.read_text(encoding="utf-8"))
        weights = state.get("weights", {})
        vol_mult = float(state.get("vol_multiplier", 1.0))
        weight = float(weights.get(sys_id, _EQ_WEIGHT))
        return weight, vol_mult
    except Exception:
        return _EQ_WEIGHT, 1.0


def log_engine_error(engine_name: str, exc: BaseException) -> None:
    """Append a timestamped traceback to logs/t9b_task_scheduler.log."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] [{engine_name}] ERROR: {exc}\n")
            fh.write("".join(traceback.format_exception(exc)))
            fh.write("-" * 60 + "\n")
    except Exception:
        pass  # never let logging itself crash the engine


def run_engine(engine_name: str, main_fn) -> None:
    """
    Run an engine's main() with crash logging.
    Unhandled exceptions are written to logs/t9b_task_scheduler.log
    (Task Scheduler runs are headless -- console output is lost),
    printed to stderr, and re-raised as exit code 1 so the daily
    bat file's errorlevel check fires.
    """
    try:
        sys.exit(main_fn())
    except SystemExit:
        raise
    except BaseException as exc:
        log_engine_error(engine_name, exc)
        print(f"[ERROR] {engine_name} failed: {exc} "
              f"(full traceback in {ERROR_LOG})", flush=True)
        traceback.print_exc()
        sys.exit(1)
