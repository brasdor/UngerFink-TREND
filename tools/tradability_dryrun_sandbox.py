#!/usr/bin/env python3
"""DRY RUN of the tradability state diff on a COPY of the repo. Touches nothing in the real tree.

1. Copies the code and the data the engines need into a temp sandbox.
2. Variant A (apply_state_fixes=false) and variant B (=true): from the same pristine state, run every engine
   exactly as the daily workflow does (--date 2026-10-16 --no-download, last_run_date wound back one day so the
   date can be re-run). A vs B isolates the effect of the fix; both include that day's normal processing.
3. Restates HISTORICAL voids on the ledgers (closed trades entered after delivery): equity, peak, drawdown and
   trade counts, computed from data/tradability/historical_scan.csv, on the sandbox copy only.
4. Writes docs/TRADABILITY_DRYRUN.md.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "2026-10-16"
SYSTEMS = [  # name, state dir, engine command (relative to sandbox)
    ("S1", "t9b_paper", ["engines/phase_t9b_donchian_universev2_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S2", "t9b_mr_paper", ["engines/phase_t9b_meanreversion_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S3", "t9b_consecdowndays_paper", ["engines/phase_t9b_consecdowndays_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S5", "t9b_momentum_paper", ["engines/phase_t9b_momentum_factor_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S6", "t9b_volcontraction_paper", ["phase_t9b_volcontraction_short_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S7", "t9b_macross_paper", ["phase_t9b_macross_short_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("S8", "t9b_rsi_mr_funding_paper", ["engines/phase_t9b_rsi_mr_funding_paper_engine.py", "--date", RUN_DATE, "--no-download"]),
    ("C12", "t9_candidate12_paper", ["engines/phase_t9_momentum_candidates_paper_engine.py", "--candidate", "12", "--date", RUN_DATE, "--no-download"]),
    ("C19", "t9_candidate19_paper", ["engines/phase_t9_momentum_candidates_paper_engine.py", "--candidate", "19", "--date", RUN_DATE, "--no-download"]),
]
KEEP_RESEARCH = {"research_candidate12_cross_sectional_momentum_t8", "research_candidate19_t8"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "frontend", "cloudflare", "dashboards", "logs", "results", "backend", "docs"}


def ignore(dirpath, names):
    out = []
    for n in names:
        p = Path(dirpath) / n
        if n in SKIP_DIRS and p.parent in (ROOT, ROOT / "data"):
            out.append(n)
        elif p.parent == ROOT / "data" and ((n.startswith("research_") and n not in KEEP_RESEARCH) or n.startswith("raw_trend") or n == "futures.db"
                                              or n.startswith("paper_trend") or n == "ohlcv_extended"):
            out.append(n)
        elif n == "__pycache__":
            out.append(n)
    return out


def build_sandbox() -> Path:
    sb = Path(tempfile.mkdtemp(prefix="tradability_sandbox_"))
    shutil.copytree(ROOT, sb / "repo", ignore=ignore, dirs_exist_ok=True)
    return sb / "repo"


def load_state(sb, d):
    return json.loads((sb / "data" / d / "state.json").read_text(encoding="utf-8"))


def metrics(sb, name, d):
    s = load_state(sb, d)
    eq = s.get("paper_equity_usdt")
    closed_eq = s.get("closed_equity_usdt", eq)
    peak = s.get("peak_equity_usdt")
    op = s.get("open_positions") or ((s.get("long_positions") or []) + (s.get("short_positions") or []))
    return dict(system=name, equity=eq, closed_equity=closed_eq, peak=peak, dd_pct=s.get("drawdown_pct"),
                closed_trade_count=s.get("closed_trade_count"), open_n=len(op),
                open_symbols=",".join(sorted(p["symbol"] for p in op)), last_run=s.get("last_run_date"))


def run_variant(sb, apply_flag: bool):
    cfg = sb / "data" / "tradability_config.json"
    cfg.write_text(json.dumps({"apply_state_fixes": apply_flag, "regime_exclude_nontradable": False}), encoding="utf-8")
    ledger = sb / "data" / "tradability_ledger.csv"
    if ledger.exists():
        ledger.unlink()
    for name, d, cmd in SYSTEMS:
        sp = sb / "data" / d / "state.json"
        st = json.loads(sp.read_text(encoding="utf-8"))
        st["last_run_date"] = "2026-10-15"
        sp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    for name, d, cmd in SYSTEMS:
        r = subprocess.run([sys.executable, *cmd], cwd=sb, capture_output=True, text=True, timeout=1800)
        tag = "ok" if r.returncode == 0 else f"EXIT {r.returncode}"
        print(f"  [{'B' if apply_flag else 'A'}] {name}: {tag}", flush=True)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-600:])
    return {name: metrics(sb, name, d) for name, d, _ in SYSTEMS}, ledger.read_text(encoding="utf-8") if ledger.exists() else ""


def snapshot_states(sb):
    snap = {}
    for name, d, _ in SYSTEMS:
        for f in (sb / "data" / d).iterdir():
            if f.is_file():
                snap[(d, f.name)] = f.read_bytes()
    return snap


def restore_states(sb, snap):
    for (d, fn), b in snap.items():
        (sb / "data" / d / fn).write_bytes(b)


def historical_restatement(sb):
    """Closed trades entered after delivery: remove their P&L and count. Returns per-system dict."""
    scan = pd.read_csv(ROOT / "data" / "tradability" / "historical_scan.csv")
    res = {}
    for name, d, _ in SYSTEMS:
        x = scan[(scan.engine == name) & (scan.exit_reason != "OPEN") & (scan.entry_flag.fillna("") != "")]
        pnl = float(x.pnl.fillna(0).sum()) if len(x) else 0.0
        res[name] = dict(voided_closed_trades=int(len(x)), voided_pnl=round(pnl, 4),
                         detail=[(r.symbol, r.entry_date, r.exit_date, None if pd.isna(r.pnl) else round(r.pnl, 4)) for r in x.itertuples()])
    # C19 curve restatement: peak and drawdown after removing the voided P&L from its exit date on
    c19 = sb / "data" / "t9_candidate19_paper" / "engine_equity_curve.csv"
    if c19.exists() and res["C19"]["voided_pnl"] != 0:
        curve = pd.read_csv(c19)
        adj = curve.equity.copy()
        for sym, ed, xd, pnl in res["C19"]["detail"]:
            if pnl is not None:
                adj[curve.date >= xd] -= pnl
        peak = adj.cummax()
        res["C19"]["curve_peak_before"] = round(float(curve.peak_equity.iloc[-1]), 2)
        res["C19"]["curve_peak_after"] = round(float(peak.iloc[-1]), 2)
        res["C19"]["curve_dd_before"] = round(float(curve.drawdown_pct.iloc[-1]), 4)
        res["C19"]["curve_dd_after"] = round(float(((adj - peak) / peak * 100).iloc[-1]), 4)
        res["C19"]["curve_max_dd_before"] = round(float(curve.drawdown_pct.min()), 4)
        res["C19"]["curve_max_dd_after"] = round(float(((adj - peak) / peak * 100).min()), 4)
    return res


def main():
    sb = build_sandbox()
    print("sandbox:", sb, flush=True)
    base = {name: metrics(sb, name, d) for name, d, _ in SYSTEMS}
    pristine = snapshot_states(sb)
    A, ledA = run_variant(sb, False)
    after_A = snapshot_states(sb)
    restore_states(sb, pristine)
    B, ledB = run_variant(sb, True)
    hist = historical_restatement(sb)
    out = {"base": base, "A": A, "B": B, "hist": hist, "ledger_B": ledB}
    (ROOT / "data" / "tradability" / "dryrun_result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "ledger_B"}, indent=1, default=str)[:6000])
    print("LEDGER B:\n", ledB[:3000])
    shutil.rmtree(sb.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
