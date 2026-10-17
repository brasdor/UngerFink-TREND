#!/usr/bin/env python3
"""Why does the S2 shadow replay match only 11 of 36 real entries? Classifies every mismatch (read-only).

Real entries  = data/t9b_mr_paper/engine_equity_curve.csv (entry_date, symbol), since 2026-06-26.
Replay entries = data/tradability/s2_shadow_old_baseline_trades.csv entries (rebuilt from exits) -- the replay
saved only exits, so entry set is rebuilt from (entry_date, symbol) of its exit records.
"""
import sys
from datetime import date
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))
import phase_t9b_meanreversion_paper_engine as s2  # noqa: E402
import tradability as T  # noqa: E402

s2._SKIP_LIVE_FETCH = True
import tempfile, health_emit  # keep this read-only tool out of the live health log
health_emit.EVENTS_PATH = Path(tempfile.mkdtemp()) / 'h.jsonl'
act = pd.read_csv(ROOT / "data/t9b_mr_paper/engine_equity_curve.csv")
act = act[act.entry_date >= "2026-06-26"]
rep = pd.read_csv(ROOT / "data/tradability/s2_shadow_old_baseline_trades.csv")
import csv
_rows = list(csv.reader(open(ROOT / "data/t9b_mr_paper/daily_log.csv", encoding="utf-8")))
log = pd.DataFrame([r[:4] for r in _rows[1:] if len(r) >= 4], columns=["run_date", "event", "symbol", "detail"])
A = {(r.entry_date, r.symbol) for r in act.itertuples()}
R = {(r.entry_date, r.symbol) for r in rep.itertuples()}
print("actual", len(A), "replay (entries that closed before 10-10)", len(R), "match", len(A & R))
log["sym"] = log.symbol.astype(str).str.replace("/", "")
# MIGRATION_DATE = 2026-07-12: engine ran on SPOT symbols/data before that
mig = "2026-07-12"


def _old_rsi(close, n):
    delta = np.diff(close, prepend=close[0]); gain, loss = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    ag = np.full(len(close), np.nan); al = np.full(len(close), np.nan)
    if len(close) < n + 1:
        return np.full(len(close), np.nan)
    ag[n] = gain[1:n+1].mean(); al[n] = loss[1:n+1].mean(); a = 1.0 / n
    for i in range(n + 1, len(close)):
        ag[i] = ag[i-1]*(1-a) + gain[i]*a; al[i] = al[i-1]*(1-a) + loss[i]*a
    rs = np.where(al > 0, ag/al, 100.0); r = 100.0 - 100.0/(1.0+rs); r[:n] = np.nan
    return r


s2._compute_rsi = _old_rsi    # the engine as it ran in production (frozen-RSI bug), so signals are comparable


def sig_in_stored_data(sym, d):
    df = s2.load_ohlcv(sym, up_to_date=date.fromisoformat(d))
    return s2.check_entry_signal(sym, df, date.fromisoformat(d)) is not None


rows = []
for d, sym in sorted(A - R):
    day = log[(log.run_date == d) & (log.sym == sym)]
    ev = ",".join(sorted(set(day.event)))
    rows.append(dict(side="actual-only", date=d, symbol=sym, pre_migration=d < mig,
                     signal_on_stored_data=sig_in_stored_data(sym, d), log_events_that_day=ev,
                     log_detail=(day.detail.iloc[0][:40] if len(day) else "")))
for d, sym in sorted(R - A):
    day = log[(log.run_date == d) & (log.sym == sym)]
    ev = ",".join(sorted(set(day.event)))
    rows.append(dict(side="replay-only", date=d, symbol=sym, pre_migration=d < mig,
                     signal_on_stored_data=True, log_events_that_day=ev,
                     log_detail=(day.detail.iloc[0][:40] if len(day) else "")))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/tradability/s2_replay_leak_diagnosis.csv", index=False)
pd.set_option("display.width", 220)
print(df.groupby(["side", "pre_migration"]).size())
a = df[df.side == "actual-only"]
print("\nactual-only entries: signal reproducible from stored data?\n", a.groupby(["pre_migration", "signal_on_stored_data"]).size())
r = df[df.side == "replay-only"]
print("\nreplay-only entries: what the real engine logged that day for the same symbol\n",
      r.groupby(["pre_migration", "log_events_that_day"]).size().sort_values(ascending=False).head(12))
print("\nlog_detail samples (replay-only):")
print(r[r.log_detail != ""].groupby("log_detail").size().sort_values(ascending=False).head(8))
print("\nmatches by month:", pd.Series([d[:7] for d, _ in (A & R)]).value_counts().sort_index().to_dict(),
      "| actual by month:", pd.Series([d[:7] for d, _ in A]).value_counts().sort_index().to_dict())
