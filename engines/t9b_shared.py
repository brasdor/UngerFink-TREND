#!/usr/bin/env python3
"""
T9B shared utilities — cross-system symbol deduplication.
Imported by all four T9B paper engines (Fix 1 support).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_STATE_PATHS: dict[str, Path] = {
    "donchian":   ROOT / "data" / "t9b_paper"                / "state.json",
    "rsi_mr":     ROOT / "data" / "t9b_mr_paper"             / "state.json",
    "consecdown": ROOT / "data" / "t9b_consecdowndays_paper"  / "state.json",
    "momentum":   ROOT / "data" / "t9b_momentum_paper"        / "state.json",
}


def normalize_sym(sym: str) -> str:
    """Normalize to BASEQUOTE: 'BTC/USDT' or 'BTCUSDT' -> 'BTCUSDT'."""
    return sym.replace("/", "").replace("-", "").upper()


def get_cross_system_symbols(exclude: str) -> frozenset[str]:
    """
    Return frozenset of normalized symbols currently open in all T9B engines
    except `exclude` ('donchian', 'rsi_mr', 'consecdown', 'momentum').
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
