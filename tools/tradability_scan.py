#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
READ-ONLY audit: every trade/position in every live engine's ledger checked against
exchange tradability by date (engines/tradability.py). Changes nothing.

Flags per trade
  entry_after_delivery : entry_date >= delivery_date          -> would be VOIDED
  before_onboard       : entry_date <  onboard_date           -> would be VOIDED
  unknown_symbol       : not in the exchange status cache     -> would be VOIDED (fail-closed)
  spans_delivery       : entry < delivery <= exit (or still open past delivery)
                         -> would be CLOSED AT SETTLEMENT on the delivery date

Output: data/tradability/historical_scan.csv and a printed summary.
S5 (momentum) has no trade ledger; its baskets are rebuilt from the git history of its state.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
import tradability as T  # noqa: E402

D = ROOT / "data"
TODAY = date(2026, 10, 11)


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _open_from_state(engine: str, state_path: Path, keys=("open_positions",)):
    rows = []
    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return rows
    for k in keys:
        for p in s.get(k, []):
            rows.append(dict(engine=engine, symbol=T.norm(p.get("symbol", "")), side=str(p.get("side") or ("short" if "short" in k else "long")).lower(),
                             entry_date=str(p.get("entry_date"))[:10], exit_date=None, pnl=None, exit_reason="OPEN"))
    return rows


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    # S1 / S2: engine_equity_curve.csv has the closed-trade ledger
    for eng, d in (("S1", "t9b_paper"), ("S2", "t9b_mr_paper")):
        e = _csv(D / d / "engine_equity_curve.csv")
        for r in e.itertuples():
            rows.append(dict(engine=eng, symbol=T.norm(r.symbol), side="long", entry_date=str(r.entry_date)[:10],
                             exit_date=str(r.exit_date)[:10], pnl=float(r.pnl_usdt), gross_r=float(r.gross_r), exit_reason=r.exit_reason))
        rows += _open_from_state(eng, D / d / "state.json")
    # S3: daily_log ENTRY/EXIT pairs
    # daily_log.csv has ragged rows (column layout drifted); pandas on_bad_lines='skip' drops ENTRY/EXIT rows.
    import csv
    _r = list(csv.reader(open(D / "t9b_consecdowndays_paper" / "daily_log.csv", encoding="utf-8")))
    lg = pd.DataFrame([r[:4] for r in _r[1:] if len(r) >= 4], columns=["run_date", "event", "symbol", "detail"])
    open_e = {}
    for r in lg[lg.event.isin(["ENTRY", "EXIT"])].itertuples():
        sym = T.norm(r.symbol)
        if r.event == "ENTRY":
            open_e[sym] = str(r.run_date)[:10]
        elif sym in open_e:
            rows.append(dict(engine="S3", symbol=sym, side="long", entry_date=open_e.pop(sym), exit_date=str(r.run_date)[:10],
                             pnl=None, exit_reason="EXIT"))
    rows += _open_from_state("S3", D / "t9b_consecdowndays_paper" / "state.json")
    # S6 / S7 / S8: equity_curve.csv holds exits
    for eng, d in (("S6", "t9b_volcontraction_paper"), ("S7", "t9b_macross_paper"), ("S8", "t9b_rsi_mr_funding_paper")):
        e = _csv(D / d / "equity_curve.csv")
        if "exit_date" in e.columns:
            for r in e.itertuples():
                rows.append(dict(engine=eng, symbol=T.norm(r.symbol), side="short" if eng in ("S6", "S7") else "long",
                                 entry_date=str(r.entry_date)[:10], exit_date=str(r.exit_date)[:10], pnl=float(r.pnl_usdt),
                                 gross_r=float(r.gross_r), exit_reason=r.exit_reason))
        rows += _open_from_state(eng, D / d / "state.json")
    # candidates 12 / 19
    for eng, c in (("C12", "12"), ("C19", "19")):
        e = _csv(D / f"t9_candidate{c}_paper" / f"closed_trades_c{c}.csv")
        for r in e.itertuples():
            rows.append(dict(engine=eng, symbol=T.norm(r.symbol), side=r.side, entry_date=str(r.entry_date)[:10],
                             exit_date=str(r.exit_date)[:10], pnl=float(r.pnl_usdt), gross_r=float(r.net_r), exit_reason=r.exit_reason))
        rows += _open_from_state(eng, D / f"t9_candidate{c}_paper" / "state.json")
    # S5: baskets from git history of state.json
    seen = set()
    out = subprocess.run(["git", "log", "--format=%H", "--", "data/t9b_momentum_paper/state.json"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    for h in out:
        blob = subprocess.run(["git", "show", f"{h}:data/t9b_momentum_paper/state.json"], cwd=ROOT, capture_output=True, text=True).stdout
        try:
            s = json.loads(blob)
        except Exception:
            continue
        for k, side in (("long_positions", "long"), ("short_positions", "short")):
            for p in s.get(k, []):
                key = (T.norm(p["symbol"]), side, str(p["entry_date"])[:10])
                if key not in seen:
                    seen.add(key)
    cur = {(r["symbol"], r["side"], r["entry_date"]) for r in _open_from_state("S5", D / "t9b_momentum_paper" / "state.json",
                                                                                 ("long_positions", "short_positions"))}
    for sym, side, ed in sorted(seen):
        rows.append(dict(engine="S5", symbol=sym, side=side, entry_date=ed, exit_date=None if (sym, side, ed) in cur else "rebalanced",
                         pnl=None, exit_reason="OPEN" if (sym, side, ed) in cur else "REBALANCE"))
    return pd.DataFrame(rows)


def flag(df: pd.DataFrame) -> pd.DataFrame:
    fl, dd, sp = [], [], []
    for r in df.itertuples():
        venue = "spot" if r.engine in ("S1", "S3") else "futures"
        why = T.untradable_reason(r.symbol, r.entry_date, venue)
        d = T.info(r.symbol, venue).delivery
        dd.append(str(d) if d else "")
        entry_bad = why in ("on_or_after_delivery", "before_onboard", "unknown_symbol", "flat_zero_volume_feed")
        fl.append(why if entry_bad else "")
        ex = None if (r.exit_date in (None, "rebalanced") or pd.isna(r.exit_date)) else str(r.exit_date)[:10]
        end = ex or str(TODAY)
        sp.append(bool(d and not entry_bad and str(d) <= end))
    df = df.copy()
    df["delivery_date"], df["entry_flag"], df["spans_delivery"] = dd, fl, sp
    return df


if __name__ == "__main__":
    df = flag(collect())
    out = D / "tradability"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "historical_scan.csv", index=False)
    print(f"{len(df)} trade/position records scanned")
    g = df.assign(void=df.entry_flag != "", span=df.spans_delivery).groupby("engine").agg(
        n=("symbol", "size"), voided=("void", "sum"), spans=("span", "sum"),
        open_n=("exit_reason", lambda s: int((s == "OPEN").sum())))
    print(g.to_string())
    print()
    print(df[(df.entry_flag != "") | df.spans_delivery][["engine", "symbol", "side", "entry_date", "exit_date", "delivery_date",
                                                          "entry_flag", "spans_delivery", "exit_reason", "pnl"]].to_string(index=False))
