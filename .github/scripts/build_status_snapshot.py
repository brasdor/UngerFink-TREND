#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions helper -- builds data/status_snapshot.json, the single data
source behind the Telegram /status command.

Why this exists rather than having the Cloudflare Worker compute it live:
the Worker only has the GitHub Contents API (one HTTP fetch per file) and
a tight per-request subrequest budget. Computing "worst-case staleness
across 290+66 OHLCV files" or "unrealized P&L for every open position
across 9 systems" live, in JS, on every /status message would mean
hundreds of fetches per command. This script already has a full repo
checkout and runs once daily (as a step in heartbeat_check.yml, after
both t9b_daily.yml and t9_candidates_daily.yml have had time to finish);
it does all of that work once, and the Worker reads one small JSON file.

Consolidates:
  - regime state (trend/funding/vol_mult/weights) from regime_state.json
  - per-system equity, open count, kill-switch, unrealized P&L, and
    today's realized P&L (vs. the previous run's snapshot) for all 9
    systems -- S1-S8 and both T9 candidates
  - the exact same alert findings check_missed_runs.py computes (imported,
    not duplicated, so the two can never silently drift apart)

Unrealized P&L sources, cheapest-correct available per system:
  - S1/S2/S3/Candidate12/Candidate19: mark_to_market.py already computes
    this daily into mtm_positions.csv -- read it, don't recompute it.
  - S5: its own open_positions.csv already carries an unrealized_pnl
    column (the engine computes it itself for the /positions-style view).
  - S6/S7 (short-only) / S8 (long-only): mark_to_market.py doesn't cover
    these three (never has -- confirmed earlier this session), so this
    script computes it directly from open_positions.csv + each symbol's
    latest close in the already-local futures OHLCV cache. No network
    calls; same cache refresh_futures_data.py already keeps current.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_missed_runs as heartbeat  # noqa: E402 -- reuse, don't duplicate

SNAPSHOT_PATH = ROOT / "data" / "status_snapshot.json"
FUTURES_OHLCV = ROOT / "data" / "futures_universe" / "ohlcv_1d"

# id, label, data_dir, side ("long" / "short" / None where not needed)
SYSTEMS = [
    ("S1", "Donchian",       "data/t9b_paper",                 None),
    ("S2", "RSI-MR",         "data/t9b_mr_paper",               None),
    ("S3", "ConsecDown",     "data/t9b_consecdowndays_paper",   None),
    ("S5", "Momentum",       "data/t9b_momentum_paper",         None),
    ("S6", "VolContraction", "data/t9b_volcontraction_paper",   "short"),
    ("S7", "MACross",        "data/t9b_macross_paper",          "short"),
    ("S8", "RSI-MR-Funding", "data/t9b_rsi_mr_funding_paper",   "long"),
    ("Candidate12", "Candidate 12", "data/t9_candidate12_paper", None),
    ("Candidate19", "Candidate 19", "data/t9_candidate19_paper", None),
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_close(symbol: str) -> float | None:
    """symbol as it appears in open_positions.csv, e.g. 'BTCUSDT'."""
    path = FUTURES_OHLCV / f"{symbol}_1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["close"])
        vals = pd.to_numeric(df["close"], errors="coerce").dropna()
        return float(vals.iloc[-1]) if len(vals) else None
    except Exception:
        return None


def _mtm_unrealized(data_dir: Path) -> float | None:
    path = data_dir / "mtm_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "unrealized_pnl" not in df.columns:
            return 0.0
        return float(pd.to_numeric(df["unrealized_pnl"], errors="coerce").fillna(0).sum())
    except Exception:
        return None


def _own_column_unrealized(data_dir: Path) -> float | None:
    """S5: open_positions.csv already has an unrealized_pnl column."""
    path = data_dir / "open_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "unrealized_pnl" not in df.columns:
            return 0.0
        return float(pd.to_numeric(df["unrealized_pnl"], errors="coerce").fillna(0).sum())
    except Exception:
        return None


def _compute_unrealized(data_dir: Path, side: str) -> float | None:
    """S6/S7/S8: entry_price + qty from open_positions.csv, current close
    from the local futures OHLCV cache. side='short' -> (entry-close)*qty,
    'long' -> (close-entry)*qty."""
    path = data_dir / "open_positions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return 0.0
    total = 0.0
    for _, row in df.iterrows():
        try:
            entry = float(row["entry_price"])
            qty = float(row["qty"])
            close = _latest_close(str(row["symbol"]))
            if close is None:
                continue
            total += (entry - close) * qty if side == "short" else (close - entry) * qty
        except Exception:
            continue
    return round(total, 4)


def build_system_snapshot(sys_id: str, label: str, rel_dir: str, side: str | None,
                          prev_snapshot: dict) -> dict:
    data_dir = ROOT / rel_dir
    state = _load_json(data_dir / "state.json")

    equity = float(state.get("paper_equity_usdt", state.get("closed_equity_usdt", 0.0)))
    open_positions = state.get("open_positions")
    if open_positions is None:
        # S5 splits long/short
        open_positions = state.get("long_positions", []) + state.get("short_positions", [])
    n_open = len(open_positions)
    closed = state.get("closed_trade_count", state.get("total_rebal_count", 0))
    kill_switch = bool(state.get("kill_switch_triggered", False))
    last_run = state.get("last_run_date")

    if side is None and (data_dir / "mtm_positions.csv").exists():
        unrealized = _mtm_unrealized(data_dir)
    elif side is None:
        unrealized = _own_column_unrealized(data_dir)
    else:
        unrealized = _compute_unrealized(data_dir, side)

    prev_equity = None
    prev_sys = (prev_snapshot.get("systems") or {}).get(sys_id)
    if prev_sys is not None:
        prev_equity = prev_sys.get("equity")
    today_realized_pnl = round(equity - prev_equity, 4) if prev_equity is not None else None

    return {
        "label": label,
        "equity": round(equity, 4),
        "open_positions": n_open,
        "closed_trades": closed,
        "kill_switch": kill_switch,
        "unrealized_pnl": unrealized,
        "today_realized_pnl": today_realized_pnl,
        "last_run_date": last_run,
    }


def main() -> int:
    now = datetime.now(timezone.utc)
    prev_snapshot = _load_json(SNAPSHOT_PATH)

    regime_state = _load_json(ROOT / "data" / "regime_state.json")
    regime = {
        "date": regime_state.get("date"),
        "trend": regime_state.get("trend_regime"),
        "funding": regime_state.get("funding_regime"),
        "vol_multiplier": regime_state.get("vol_multiplier"),
        "weights": regime_state.get("weights", {}),
    }

    systems = {}
    for sys_id, label, rel_dir, side in SYSTEMS:
        print(f"[SNAPSHOT] building {sys_id} ({label})...")
        systems[sys_id] = build_system_snapshot(sys_id, label, rel_dir, side, prev_snapshot)

    print("[SNAPSHOT] pulling alert findings (shared with check_missed_runs.py)...")
    alerts = heartbeat.compute_heartbeat()

    snapshot = {
        "generated_utc": now.isoformat(),
        "regime": regime,
        "systems": systems,
        "alerts": alerts,
    }

    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"[SNAPSHOT] wrote {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
