#!/usr/bin/env python3
"""S2 opportunity-cost shadow (READ-ONLY replay; writes only data/tradability/s2_shadow_*.csv).

Replays the real S2 engine (run_one_day) day by day from 2026-06-26 twice from the same start:
  baseline : no tradability gate (what the live engine effectively did)
  gated    : exchange-faithful gate (symbols delivered/unknown excluded by date) with freed slots
             filled by the NEXT valid signals (the engine's own entry loop does this naturally).
Inputs the live engine read from other systems are reconstructed from its own daily_log:
  cross-system duplicate set per date = symbols logged 'duplicate_cross_system' that day;
  the signal arbitrator never rejected in the log (0 'arbitrator:' skips) so it is stubbed to accept.
The baseline is validated against the real ledger (data/t9b_mr_paper/engine_equity_curve.csv).
"""
import sys, json, tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest import mock
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines")); sys.path.insert(0, str(ROOT))
import health_emit, tradability as T
import phase_t9b_meanreversion_paper_engine as s2

START, END = date(2026, 6, 26), date(2026, 10, 10)
START_EQ = 10053.07
log = pd.read_csv(ROOT/"data/t9b_mr_paper/daily_log.csv", on_bad_lines="skip")
dup = log[(log.event == "SIGNAL_SKIPPED") & (log.detail == "duplicate_cross_system")]
CROSS = {d: {s.replace("/", "") for s in g.symbol} for d, g in dup.groupby("run_date")}
T0 = Path(tempfile.mkdtemp())
s2._SKIP_LIVE_FETCH = True

class Accept:
    def __init__(self, *a, **k): pass
    def check_signal(self, *a, **k): return ("ACCEPTED", "")

def _old_rsi(close, n):
    """S2's RSI exactly as it was before the 2026-10-11 flat-window fix (frozen value on flat input)."""
    delta = np.diff(close, prepend=close[0])
    gain, loss = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    avg_g = np.full(len(close), np.nan); avg_l = np.full(len(close), np.nan)
    if len(close) < n + 1:
        return np.full(len(close), np.nan)
    avg_g[n] = gain[1:n+1].mean(); avg_l[n] = loss[1:n+1].mean()
    a = 1.0 / n
    for i in range(n+1, len(close)):
        avg_g[i] = avg_g[i-1]*(1-a) + gain[i]*a
        avg_l[i] = avg_l[i-1]*(1-a) + loss[i]*a
    rs = np.where(avg_l > 0, avg_g/avg_l, 100.0)
    rsi = 100.0 - 100.0/(1.0+rs); rsi[:n] = np.nan
    return rsi

def run(mode):
    old = mode.startswith("old_")
    mode = mode.replace("old_", "")
    st = {"system_name": "shadow", "paper_equity_usdt": START_EQ, "peak_equity_usdt": START_EQ, "kill_switch_triggered": False,
          "closed_trade_count": 0, "last_run_date": None, "open_positions": []}
    exits, entries = [], []
    syms = s2._discover_universe()
    cur_cross = {}
    with mock.patch.object(health_emit, "EVENTS_PATH", T0/"h.jsonl"), mock.patch.object(T, "VOID_LEDGER", T0/"l.csv"), \
         mock.patch.object(s2, "SignalArbitrator", Accept), \
         mock.patch.object(s2.t9b_shared, "get_cross_system_symbols", lambda ex: frozenset(cur_cross.get("s", set()))):
        if mode == "baseline":
            ctx = [mock.patch.object(T, "tradable_universe", lambda e, s, d, v="futures": list(s)),
                   mock.patch.object(T, "pending_action", lambda *a, **k: None)]
        else:
            ctx = [mock.patch.object(T, "apply_state_fixes_enabled", lambda: True)]
        if old: ctx.append(mock.patch.object(s2, "_compute_rsi", _old_rsi))
        for c in ctx: c.start()
        try:
            d = START
            while d <= END:
                cur_cross["s"] = CROSS.get(str(d), set())
                r = s2.run_one_day(d, syms, st)
                for e in r["exits_today"]:
                    exits.append(e)
                for ev in r["log_events"]:
                    if ev["event"] == "ENTRY":
                        entries.append((str(d), ev["symbol"]))
                d += timedelta(days=1)
        finally:
            for c in ctx: c.stop()
    return pd.DataFrame(exits), entries, st

def stats(ex, label):
    ex = ex.copy()
    if ex.empty: return {"mode": label}
    real = ex[ex.gross_r != 0]
    return dict(mode=label, trades=len(ex), nonflat_trades=len(real), hit_rate=round((ex.gross_r > 0).mean(), 4),
                hit_rate_nonflat=round((real.gross_r > 0).mean(), 4) if len(real) else None, avg_R=round(ex.gross_r.mean(), 4),
                avg_R_nonflat=round(real.gross_r.mean(), 4) if len(real) else None, total_pnl=round(ex.pnl_usdt.sum(), 2),
                final_equity=round(START_EQ + ex.pnl_usdt.sum(), 2))

if __name__ == "__main__":
    out = ROOT/"data/tradability"; out.mkdir(exist_ok=True)
    modes = sys.argv[1:] or ["old_baseline", "old_gated"]
    act = pd.read_csv(ROOT/"data/t9b_mr_paper/engine_equity_curve.csv")
    real = act[act.exit_date >= "2026-07-01"]
    S = [stats(real, "ACTUAL ledger (exits >= 2026-07-01)")]
    ae = {(r.entry_date, r.symbol) for r in act.itertuples() if r.entry_date >= "2026-06-26"}
    res = {}
    for m in modes:
        ex, en, _ = run(m)
        ex.to_csv(out/f"s2_shadow_{m}_trades.csv", index=False)
        res[m] = {(d, s) for d, s in en}
        S.append(stats(ex, f"replay {m}"))
        e = res[m]
        print(f"{m}: entries {len(e)} | match actual {len(e & ae)} | replay-only {len(e - ae)} | actual-only {len(ae - e)}", flush=True)
    if "old_baseline" in res and "old_gated" in res:
        print("removed by gate:", sorted(res["old_baseline"] - res["old_gated"]))
        print("added by gate (freed slots refilled):", sorted(res["old_gated"] - res["old_baseline"]))
    print(pd.DataFrame(S).to_string(index=False))
    pd.DataFrame(S).to_csv(out/"s2_shadow_summary_old_rsi.csv", index=False)
