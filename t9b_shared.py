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

ROOT = Path(__file__).resolve().parent
ERROR_LOG = ROOT / "logs" / "t9b_task_scheduler.log"

_STATE_PATHS: dict[str, Path] = {
    "donchian":       ROOT / "data" / "t9b_paper"                  / "state.json",
    "rsi_mr":         ROOT / "data" / "t9b_mr_paper"               / "state.json",
    "consecdown":     ROOT / "data" / "t9b_consecdowndays_paper"    / "state.json",
    "momentum":       ROOT / "data" / "t9b_momentum_paper"          / "state.json",
    "volcontraction": ROOT / "data" / "t9b_volcontraction_paper"    / "state.json",
    "macross":        ROOT / "data" / "t9b_macross_paper"           / "state.json",
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
