#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T14 — REALISTIC 6H PORTFOLIO REPLAY

Offline only. T9 paper-live remains frozen.

Uses validated 6H trades from T13 if available, otherwise T10 ACT4_ATR3.
Tests realistic portfolio replay:
- max open = 5
- risk per trade = 0.25%
- max heat = 1.50%
- one position per symbol
- scenarios: no extra cap / same-side cap 4 / same-side cap 3
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
T13_TRADES = ROOT / "data" / "research_trend_t13" / "phase_t13_timeframe_trades.csv"
T10_TRADES = ROOT / "data" / "research_trend_t10" / "phase_t10_trailing_trades.csv"
OUT = ROOT / "data" / "research_trend_t14"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10000.0
TIMEFRAME = "6h"
T10_VARIANT = "ACT4_ATR3"

MAX_OPEN = 5
RISK_PCT = 0.0025
MAX_HEAT = 0.015
ONE_POSITION_PER_SYMBOL = True

SCENARIOS = [
    ("BASE_MAX5_HEAT15", 5),
    ("SAME_SIDE_CAP4", 4),
    ("SAME_SIDE_CAP3", 3),
]


def pf(r):
    s = pd.to_numeric(r, errors="coerce").dropna()
    if s.empty:
        return 0.0
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    if l <= 1e-12:
        return np.inf if g > 0 else 0.0
    return float(g / l)


def max_dd_pct(e):
    s = pd.to_numeric(e, errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    return float(((s - peak) / peak * 100.0).min())


def max_dd_r(r):
    s = pd.to_numeric(r, errors="coerce").fillna(0.0).values
    if len(s) == 0:
        return 0.0
    eq = np.cumsum(s)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def losing_streak(r):
    best = cur = 0
    for x in pd.to_numeric(r, errors="coerce").fillna(0.0):
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def load_trades():
    if T13_TRADES.exists():
        df = pd.read_csv(T13_TRADES)
        df = df[df["timeframe"].astype(str).str.lower() == TIMEFRAME].copy()
        source = str(T13_TRADES)
    elif T10_TRADES.exists():
        df = pd.read_csv(T10_TRADES)
        df = df[df["variant"].astype(str) == T10_VARIANT].copy()
        source = str(T10_TRADES)
    else:
        raise FileNotFoundError(f"Missing input files:\n{T13_TRADES}\n{T10_TRADES}")

    required = ["symbol", "side", "entry_time", "exit_time", "net_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time", "net_r"]).copy()
    df = df[df["exit_time"] >= df["entry_time"]].copy()
    df = df.sort_values(["entry_time", "exit_time", "symbol"]).reset_index(drop=True)
    df["trade_id"] = np.arange(len(df))
    df["source_file"] = source
    return df


def side_eff(open_pos):
    counts = {}
    for p in open_pos.values():
        s = str(p["side"]).upper()
        counts[s] = counts.get(s, 0) + 1
    vals = np.array(list(counts.values()), dtype=float)
    if vals.sum() <= 0:
        return 0.0
    w = vals / vals.sum()
    hhi = np.sum(w*w)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def snapshot(label, t, event, equity, peak, open_pos):
    open_risk = sum(float(p["risk_amount"]) for p in open_pos.values())
    heat = open_risk / max(equity, 1e-12) * 100.0
    dd = (equity - peak) / max(peak, 1e-12) * 100.0
    long_n = sum(1 for p in open_pos.values() if str(p["side"]).upper() == "LONG")
    short_n = sum(1 for p in open_pos.values() if str(p["side"]).upper() == "SHORT")
    return {
        "scenario": label,
        "time": t,
        "event": event,
        "equity": equity,
        "peak_equity": peak,
        "drawdown_pct": dd,
        "open_positions": len(open_pos),
        "long_positions": long_n,
        "short_positions": short_n,
        "max_same_side": max(long_n, short_n),
        "portfolio_heat_pct": heat,
        "open_risk_usdt": open_risk,
        "effective_side_diversification": side_eff(open_pos),
    }


def replay(df, label, max_same_side_allowed):
    events = []
    rows = {}
    for _, row in df.iterrows():
        tid = int(row["trade_id"])
        rows[tid] = row
        events.append((row["entry_time"], 1, tid))
        events.append((row["exit_time"], 0, tid))
    events.sort(key=lambda x: (x[0], x[1]))  # exits first because 0 < 1

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    open_pos = {}
    accepted, skipped, eq_rows = [], [], []

    for t, typ, tid in events:
        if typ == 0:
            pos = open_pos.pop(tid, None)
            if pos is None:
                continue
            r = float(pos["net_r"])
            pnl = float(pos["risk_amount"]) * r
            equity += pnl
            peak = max(peak, equity)
            closed = dict(pos)
            closed.update({
                "exit_time": t,
                "realized_r": r,
                "pnl_usdt": pnl,
                "equity_after_exit": equity,
                "peak_equity": peak,
                "drawdown_pct": (equity - peak) / max(peak, 1e-12) * 100.0,
                "open_positions_after_exit": len(open_pos),
            })
            accepted.append(closed)
            eq_rows.append(snapshot(label, t, "exit", equity, peak, open_pos))
            continue

        row = rows[tid]
        symbol = row["symbol"]
        side = str(row["side"]).upper()
        heat = sum(float(p["risk_amount"]) for p in open_pos.values()) / max(equity, 1e-12)
        same_side_count = sum(1 for p in open_pos.values() if str(p["side"]).upper() == side)
        symbol_open = any(p["symbol"] == symbol for p in open_pos.values())

        reason = None
        if len(open_pos) >= MAX_OPEN:
            reason = "MAX_OPEN"
        elif heat + RISK_PCT > MAX_HEAT + 1e-12:
            reason = "MAX_HEAT"
        elif ONE_POSITION_PER_SYMBOL and symbol_open:
            reason = "SYMBOL_ALREADY_OPEN"
        elif same_side_count >= max_same_side_allowed:
            reason = "MAX_SAME_SIDE"

        if reason:
            skipped.append({
                "scenario": label,
                "trade_id": tid,
                "symbol": symbol,
                "side": side,
                "entry_time": row["entry_time"],
                "skip_reason": reason,
                "open_positions": len(open_pos),
                "same_side_count": same_side_count,
                "current_heat_pct": heat * 100.0,
                "equity": equity,
            })
            continue

        risk_amount = equity * RISK_PCT
        open_pos[tid] = {
            "scenario": label,
            "trade_id": tid,
            "symbol": symbol,
            "side": side,
            "entry_time": row["entry_time"],
            "entry_price": row.get("entry_price", np.nan),
            "exit_price": row.get("exit_price", np.nan),
            "initial_stop": row.get("initial_stop", np.nan),
            "final_stop": row.get("final_stop", np.nan),
            "initial_risk_per_unit": row.get("initial_risk_per_unit", np.nan),
            "net_r": float(row["net_r"]),
            "bars_held": row.get("bars_held", np.nan),
            "max_favorable_r": row.get("max_favorable_r", np.nan),
            "chandelier_active": row.get("chandelier_active", np.nan),
            "risk_amount": risk_amount,
            "equity_before_entry": equity,
            "heat_before_entry_pct": heat * 100.0,
            "heat_after_entry_pct": (heat + RISK_PCT) * 100.0,
            "same_side_before_entry": same_side_count,
            "open_positions_before_entry": len(open_pos),
        }
        eq_rows.append(snapshot(label, t, "entry", equity, peak, open_pos))

    return pd.DataFrame(accepted), pd.DataFrame(skipped), pd.DataFrame(eq_rows)


def summarize(acc, skip, eq, n_input, label, max_same):
    if acc.empty:
        return {
            "scenario": label, "max_same_side_allowed": max_same,
            "input_trades": n_input, "accepted_trades": 0,
            "skipped_trades": len(skip), "acceptance_rate_pct": 0.0,
            "final_equity": INITIAL_CAPITAL, "return_pct": 0.0,
            "max_dd_pct": 0.0, "pf_r": 0.0, "total_r": 0.0,
            "avg_r": 0.0, "win_rate_pct": 0.0,
        }

    r = pd.to_numeric(acc["realized_r"], errors="coerce").dropna()
    final_equity = float(acc["equity_after_exit"].iloc[-1])
    return {
        "scenario": label,
        "max_same_side_allowed": max_same,
        "input_trades": n_input,
        "accepted_trades": len(acc),
        "skipped_trades": len(skip),
        "acceptance_rate_pct": len(acc) / max(n_input, 1) * 100.0,
        "final_equity": final_equity,
        "return_pct": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_dd_pct": max_dd_pct(acc["equity_after_exit"]),
        "pf_r": pf(r),
        "total_r": float(r.sum()) if len(r) else 0.0,
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "median_r": float(r.median()) if len(r) else 0.0,
        "win_rate_pct": float((r > 0).mean() * 100.0) if len(r) else 0.0,
        "max_dd_r": max_dd_r(r),
        "best_r": float(r.max()) if len(r) else 0.0,
        "worst_r": float(r.min()) if len(r) else 0.0,
        "max_losing_streak": losing_streak(r),
        "max_open_observed": int(eq["open_positions"].max()) if not eq.empty else 0,
        "max_heat_observed_pct": float(eq["portfolio_heat_pct"].max()) if not eq.empty else 0.0,
        "max_same_side_observed": int(eq["max_same_side"].max()) if not eq.empty else 0,
        "avg_effective_side_div": float(eq["effective_side_diversification"].mean()) if not eq.empty else 0.0,
    }


def write_report(summary):
    lines = [
        "PHASE T14 — REALISTIC 6H PORTFOLIO REPLAY REPORT",
        "=" * 80,
        "",
        "Offline only. T9 paper-live remains frozen.",
        "",
        "Validated core:",
        "- Timeframe: 6H",
        "- Archetype: Donchian breakout trend following",
        "- Exit: Chandelier ACT4_ATR3",
        "",
        "Portfolio assumptions:",
        f"- Initial capital: {INITIAL_CAPITAL:.2f} USDT",
        f"- Max open: {MAX_OPEN}",
        f"- Risk per trade: {RISK_PCT*100:.2f}%",
        f"- Max heat: {MAX_HEAT*100:.2f}%",
        f"- One position per symbol: {ONE_POSITION_PER_SYMBOL}",
        "",
        "RESULTS",
        "-" * 80,
    ]
    if summary.empty:
        lines.append("No results.")
    else:
        for _, r in summary.sort_values(["return_pct", "pf_r"], ascending=False).iterrows():
            lines.append(
                f"{r['scenario']}: return={r['return_pct']:.2f}% | DD={r['max_dd_pct']:.2f}% | "
                f"PF={r['pf_r']:.2f} | trades={int(r['accepted_trades'])}/{int(r['input_trades'])} | "
                f"avgR={r['avg_r']:.3f} | totalR={r['total_r']:.2f} | "
                f"maxHeat={r['max_heat_observed_pct']:.2f}% | maxSameSide={int(r['max_same_side_observed'])}"
            )
    lines += [
        "",
        "Interpretation rules:",
        "- If BASE is positive and DD is tolerable, do not add filters.",
        "- If SAME_SIDE_CAP3/4 improves DD without destroying PF, it becomes a candidate.",
        "- If all scenarios are weak, do not increase risk or max open.",
        "- Do not update T9 until confirmed by paper-live behavior.",
    ]
    (OUT / "phase_t14_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE T14 — REALISTIC 6H PORTFOLIO REPLAY")
    print("=" * 80)
    print(f"Output: {OUT}")

    trades = load_trades()
    if trades.empty:
        raise ValueError("No 6H trades loaded.")

    print(f"Loaded 6H trades: {len(trades)}")
    print(f"Input source: {trades['source_file'].iloc[0]}")

    summaries, all_acc, all_skip, all_eq = [], [], [], []

    for label, max_same in SCENARIOS:
        print(f"\n[SCENARIO] {label} | max_same_side={max_same}")
        acc, skip, eq = replay(trades, label, max_same)
        summaries.append(summarize(acc, skip, eq, len(trades), label, max_same))
        if not acc.empty:
            all_acc.append(acc)
        if not skip.empty:
            all_skip.append(skip)
        if not eq.empty:
            all_eq.append(eq)
        print(f"  accepted={len(acc)}/{len(trades)} skipped={len(skip)}")

    summary = pd.DataFrame(summaries).sort_values(["return_pct", "pf_r"], ascending=False)
    acc_df = pd.concat(all_acc, ignore_index=True) if all_acc else pd.DataFrame()
    skip_df = pd.concat(all_skip, ignore_index=True) if all_skip else pd.DataFrame()
    eq_df = pd.concat(all_eq, ignore_index=True) if all_eq else pd.DataFrame()

    summary.to_csv(OUT / "phase_t14_realistic_6h_summary.csv", index=False)
    acc_df.to_csv(OUT / "phase_t14_realistic_6h_trades.csv", index=False)
    skip_df.to_csv(OUT / "phase_t14_realistic_6h_skipped.csv", index=False)
    eq_df.to_csv(OUT / "phase_t14_realistic_6h_equity.csv", index=False)
    write_report(summary)

    print("\nT14 SUMMARY")
    cols = [
        "scenario", "accepted_trades", "return_pct", "max_dd_pct", "pf_r",
        "total_r", "avg_r", "win_rate_pct", "max_heat_observed_pct",
        "max_same_side_observed"
    ]
    print(summary[cols].to_string(index=False))
    print("\n[OK] phase_t14_realistic_6h_summary.csv")
    print("[OK] phase_t14_realistic_6h_trades.csv")
    print("[OK] phase_t14_realistic_6h_equity.csv")
    print("[OK] phase_t14_realistic_6h_skipped.csv")
    print("[OK] phase_t14_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
