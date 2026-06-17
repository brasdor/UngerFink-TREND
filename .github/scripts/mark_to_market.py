#!/usr/bin/env python3
"""
Mark-to-market: compute unrealized P&L for all open positions and write
a daily equity_curve.csv + mtm_positions.csv per strategy.

Runs after the engines in the GitHub Actions workflow. Reads each engine's
open_positions.csv and the latest close price from the OHLCV cache.

Outputs per strategy dir:
  equity_curve.csv  — date, paper_equity, unrealized_pnl, total_value
  mtm_positions.csv — per-position: symbol, entry_price, current_price, qty,
                       cost, market_value, unrealized_pnl, pnl_pct
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()

SYSTEMS = [
    ("Donchian", ROOT / "data" / "t9b_paper"),
    ("RSI-MR", ROOT / "data" / "t9b_mr_paper"),
    ("ConsecDown", ROOT / "data" / "t9b_consecdowndays_paper"),
]

OHLCV_DIR = ROOT / "data" / "universe" / "ohlcv_1d"


def latest_close(symbol: str) -> float | None:
    """Read the last close price from the committed OHLCV cache."""
    ticker = symbol.replace("/", "_")
    path = OHLCV_DIR / f"{ticker}_1d.csv"
    if not path.exists():
        return None
    last_line = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    parts = last_line.split(",")
    # OHLCV format: timestamp,open,high,low,close,volume
    try:
        return float(parts[4])
    except (IndexError, ValueError):
        return None


def read_positions(data_dir: Path) -> list[dict]:
    op = data_dir / "open_positions.csv"
    if not op.exists():
        return []
    with open(op, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_state(data_dir: Path) -> dict:
    sp = data_dir / "state.json"
    if not sp.exists():
        return {}
    with open(sp, encoding="utf-8") as f:
        return json.load(f)


def process_system(label: str, data_dir: Path, run_date: str) -> None:
    positions = read_positions(data_dir)
    state = read_state(data_dir)
    paper_equity = state.get("paper_equity_usdt", 10000.0)

    mtm_rows = []
    total_unrealized = 0.0
    total_cost = 0.0
    total_market_value = 0.0

    for pos in positions:
        symbol = pos.get("symbol", "")
        try:
            entry_price = float(pos.get("entry_price", 0))
            qty = float(pos.get("qty", 0))
        except (ValueError, TypeError):
            continue

        current_price = latest_close(symbol)
        if current_price is None:
            continue

        cost = entry_price * qty
        market_value = current_price * qty
        pnl = market_value - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0

        total_cost += cost
        total_market_value += market_value
        total_unrealized += pnl

        mtm_rows.append({
            "symbol": symbol,
            "entry_date": pos.get("entry_date", ""),
            "entry_price": f"{entry_price:.6f}",
            "current_price": f"{current_price:.6f}",
            "qty": f"{qty:.6f}",
            "cost_usdt": f"{cost:.2f}",
            "market_value_usdt": f"{market_value:.2f}",
            "unrealized_pnl": f"{pnl:.2f}",
            "pnl_pct": f"{pnl_pct:.2f}",
        })

    total_value = paper_equity + total_unrealized

    # Write mtm_positions.csv
    mtm_path = data_dir / "mtm_positions.csv"
    mtm_fields = ["symbol", "entry_date", "entry_price", "current_price", "qty",
                   "cost_usdt", "market_value_usdt", "unrealized_pnl", "pnl_pct"]
    with open(mtm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mtm_fields)
        w.writeheader()
        w.writerows(mtm_rows)

    # Append to equity_curve.csv
    eq_path = data_dir / "equity_curve.csv"
    write_header = not eq_path.exists() or eq_path.stat().st_size == 0
    with open(eq_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "paper_equity", "unrealized_pnl", "total_value",
                         "open_positions", "total_cost", "total_market_value"])
        w.writerow([run_date, f"{paper_equity:.2f}", f"{total_unrealized:.2f}",
                     f"{total_value:.2f}", len(mtm_rows), f"{total_cost:.2f}",
                     f"{total_market_value:.2f}"])

    print(f"  {label}: {len(mtm_rows)} positions, "
          f"cost=${total_cost:,.2f}, value=${total_market_value:,.2f}, "
          f"unrealized={'+' if total_unrealized >= 0 else ''}{total_unrealized:,.2f}, "
          f"total=${total_value:,.2f}")


def main() -> None:
    run_date = os.environ.get("RUN_DATE", "")
    if not run_date:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Mark-to-market: {run_date}")
    for label, data_dir in SYSTEMS:
        if not data_dir.exists():
            continue
        process_system(label, data_dir, run_date)


if __name__ == "__main__":
    main()
