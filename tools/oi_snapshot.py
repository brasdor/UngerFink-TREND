#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append-only BTCUSDT open-interest snapshot log for the risk-off forward test.

See docs/RISKOFF_OI_FORWARD_PREREG.md section 3. Runs from the funding host (fapi.binance.com is HTTP 451 for
GitHub's US runners). Fetches the last 500 five-minute bars (about 41 hours) from /futures/data/openInterestHist and
appends bars not yet in data/riskoff/forward/oi_btcusdt_5m.csv. A stored bar is never overwritten; if the API later
returns a different value for a stored bar, the difference is appended to oi_restatements.csv.

Called by tools/refresh_funding_local.py after the funding refresh; can also be run alone:
    python tools/oi_snapshot.py
Gaps longer than ~40 hours cannot be backfilled from the API (the binance.vision daily metrics files cover them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "riskoff" / "forward"
LOG = OUT_DIR / "oi_btcusdt_5m.csv"
RESTATE = OUT_DIR / "oi_restatements.csv"
URL = "https://fapi.binance.com/futures/data/openInterestHist"
COLS = ["bar_time_ms", "bar_time_utc", "sum_open_interest", "sum_open_interest_value", "first_seen_utc"]


def fetch() -> pd.DataFrame:
    r = requests.get(URL, params={"symbol": "BTCUSDT", "period": "5m", "limit": 500}, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise RuntimeError("openInterestHist returned no rows")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "bar_time_ms": df["timestamp"].astype("int64"),
        "sum_open_interest": df["sumOpenInterest"].astype(float),
        "sum_open_interest_value": df["sumOpenInterestValue"].astype(float),
    })
    out["bar_time_utc"] = pd.to_datetime(out["bar_time_ms"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        new = fetch()
    except Exception as exc:  # network / geo-block: report, do not crash the funding job
        print(f"[OI] snapshot failed: {type(exc).__name__}: {exc}")
        return 1
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    known = old.set_index("bar_time_ms")["sum_open_interest"] if len(old) else pd.Series(dtype=float)

    seen = new[new["bar_time_ms"].isin(known.index)]
    diff = seen[(seen.set_index("bar_time_ms")["sum_open_interest"] - known.reindex(seen["bar_time_ms"]).values).abs().values > 1e-9]
    if len(diff):
        rs = diff.assign(old_sum_open_interest=known.reindex(diff["bar_time_ms"]).values, seen_utc=now)
        rs[["bar_time_ms", "bar_time_utc", "old_sum_open_interest", "sum_open_interest", "seen_utc"]].to_csv(
            RESTATE, mode="a", header=not RESTATE.exists(), index=False)
        print(f"[OI] {len(diff)} stored bars returned a different value (logged to {RESTATE.name}); originals kept")

    add = new[~new["bar_time_ms"].isin(known.index)].copy()
    add["first_seen_utc"] = now
    if len(add):
        add[COLS].to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    last = new["bar_time_utc"].iloc[-1]
    print(f"[OI] fetched {len(new)} bars, appended {len(add)}, log now {len(old) + len(add)} bars, last bar {last}")
    if len(old):
        gap_h = (add["bar_time_ms"].min() - old["bar_time_ms"].max()) / 3.6e6 if len(add) else 0
        if gap_h > 0.2:
            print(f"[OI] WARNING gap of {gap_h:.1f} h between the stored log and the new bars (use the binance.vision files for that span)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
