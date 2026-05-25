#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T11 — TREND PORTFOLIO HEAT & CAPACITY STUDY
=================================================
Offline study only. It does NOT change T9 paper-live.

Input:
  data/research_trend_t10/phase_t10_trailing_trades.csv

Tests:
  variants: ACT4_ATR3, ACT3_ATR3, ACT5_ATR3
  max_open: 5, 8, 10, 12
  risk_per_trade: 0.25%, 0.35%, 0.50%
  max_heat: 1.5%, 2.0%, 3.0%

Outputs:
  data/research_trend_t11/phase_t11_portfolio_heat_summary.csv
  data/research_trend_t11/phase_t11_portfolio_heat_trades.csv
  data/research_trend_t11/phase_t11_portfolio_heat_equity.csv
  data/research_trend_t11/phase_t11_master_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT_TRADES = ROOT / "data" / "research_trend_t10" / "phase_t10_trailing_trades.csv"
OUT = ROOT / "data" / "research_trend_t11"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
VARIANTS = ["ACT4_ATR3", "ACT3_ATR3", "ACT5_ATR3"]
MAX_OPEN_LEVELS = [5, 8, 10, 12]
RISK_LEVELS = [0.0025, 0.0035, 0.0050]
HEAT_LEVELS = [0.015, 0.020, 0.030]
ONE_POSITION_PER_SYMBOL = True
EXTRA_COST_R = 0.00


def profit_factor(x):
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return 0.0
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 1e-12:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def max_dd_pct(equity):
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if e.empty:
        return 0.0
    peak = e.cummax()
    return float(((e - peak) / peak * 100.0).min())


def max_dd_r(r):
    arr = pd.to_numeric(r, errors="coerce").fillna(0.0).values
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def losing_streak(r):
    best = cur = 0
    for v in pd.to_numeric(r, errors="coerce").fillna(0.0):
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def normalize(df):
    required = ["variant", "symbol", "side", "entry_time", "exit_time", "net_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce", utc=True)
    out["net_r"] = pd.to_numeric(out["net_r"], errors="coerce")
    out = out.dropna(subset=["entry_time", "exit_time", "net_r"])
    out = out[out["exit_time"] >= out["entry_time"]].copy()
    out["trade_id"] = np.arange(len(out))
    return out


def simulate(vdf, variant, max_open, risk_pct, max_heat):
    vdf = vdf.sort_values(["entry_time", "exit_time", "symbol"]).reset_index(drop=True)
    events, rows = [], {}
    for _, row in vdf.iterrows():
        tid = int(row["trade_id"])
        rows[tid] = row
        events.append((row["entry_time"], 1, tid))
        events.append((row["exit_time"], 0, tid))
    events.sort(key=lambda x: (x[0], x[1]))  # exits first

    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    open_pos = {}
    accepted, skipped, eq_rows = [], [], []

    for t, event_type, tid in events:
        if event_type == 0:
            pos = open_pos.pop(tid, None)
            if pos is None:
                continue
            r = pos["net_r"] - EXTRA_COST_R
            pnl = pos["risk_amount"] * r
            equity += pnl
            peak = max(peak, equity)
            dd = (equity - peak) / peak * 100.0
            res = dict(pos)
            res.update({
                "exit_time": t, "net_r_after_cost": r, "pnl_usdt": pnl,
                "equity_after_exit": equity, "peak_equity": peak,
                "drawdown_pct": dd, "open_positions_after_exit": len(open_pos),
                "heat_after_exit_pct": sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12) * 100.0,
            })
            accepted.append(res)
            eq_rows.append({
                "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "max_heat": max_heat,
                "time": t, "event": "exit", "equity": equity, "peak": peak, "drawdown_pct": dd,
                "open_positions": len(open_pos),
                "portfolio_heat_pct": sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12) * 100.0,
            })
            continue

        row = rows[tid]
        current_heat = sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12)
        symbol_open = any(p["symbol"] == row["symbol"] for p in open_pos.values())

        reason = None
        if len(open_pos) >= max_open:
            reason = "MAX_OPEN"
        elif current_heat + risk_pct > max_heat + 1e-12:
            reason = "MAX_HEAT"
        elif ONE_POSITION_PER_SYMBOL and symbol_open:
            reason = "SYMBOL_ALREADY_OPEN"

        if reason:
            skipped.append({
                "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "max_heat": max_heat,
                "trade_id": tid, "symbol": row["symbol"], "side": row["side"], "entry_time": row["entry_time"],
                "skip_reason": reason, "open_positions": len(open_pos),
                "portfolio_heat_pct": current_heat * 100.0, "equity": equity,
            })
            continue

        risk_amount = equity * risk_pct
        open_pos[tid] = {
            "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "max_heat": max_heat,
            "trade_id": tid, "symbol": row["symbol"], "side": row["side"], "entry_time": row["entry_time"],
            "entry_price": row.get("entry_price", np.nan), "initial_stop": row.get("initial_stop", np.nan),
            "final_stop": row.get("final_stop", np.nan), "bars_held": row.get("bars_held", np.nan),
            "max_favorable_r": row.get("max_favorable_r", np.nan),
            "chandelier_active": row.get("chandelier_active", np.nan), "net_r": float(row["net_r"]),
            "risk_amount": risk_amount, "equity_before_entry": equity,
            "open_positions_before_entry": len(open_pos), "heat_before_entry_pct": current_heat * 100.0,
            "heat_after_entry_pct": (current_heat + risk_pct) * 100.0,
        }
        eq_rows.append({
            "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "max_heat": max_heat,
            "time": t, "event": "entry", "equity": equity, "peak": peak,
            "drawdown_pct": (equity - peak) / peak * 100.0,
            "open_positions": len(open_pos),
            "portfolio_heat_pct": sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12) * 100.0,
        })

    return pd.DataFrame(accepted), pd.DataFrame(skipped), pd.DataFrame(eq_rows)


def summarize(acc, skip, eq, input_count, variant, max_open, risk_pct, max_heat):
    if acc.empty:
        return {
            "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "risk_pct_label": f"{risk_pct*100:.2f}%",
            "max_heat": max_heat, "max_heat_label": f"{max_heat*100:.2f}%", "input_trades": input_count,
            "accepted_trades": 0, "skipped_trades": len(skip), "acceptance_rate_pct": 0.0,
            "final_equity": INITIAL_CAPITAL, "return_pct": 0.0, "max_dd_pct": 0.0,
            "total_r": 0.0, "avg_r": 0.0, "pf_r": 0.0, "win_rate_pct": 0.0,
            "max_dd_r": 0.0, "best_r": 0.0, "worst_r": 0.0, "max_losing_streak": 0,
            "max_open_observed": 0, "max_heat_observed_pct": 0.0,
        }
    r = pd.to_numeric(acc["net_r_after_cost"], errors="coerce").dropna()
    final_equity = float(acc["equity_after_exit"].iloc[-1])
    return {
        "variant": variant, "max_open": max_open, "risk_pct": risk_pct, "risk_pct_label": f"{risk_pct*100:.2f}%",
        "max_heat": max_heat, "max_heat_label": f"{max_heat*100:.2f}%", "input_trades": input_count,
        "accepted_trades": len(acc), "skipped_trades": len(skip),
        "acceptance_rate_pct": len(acc) / max(input_count, 1) * 100.0,
        "final_equity": final_equity,
        "return_pct": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_dd_pct": max_dd_pct(acc["equity_after_exit"]),
        "total_r": float(r.sum()), "avg_r": float(r.mean()) if len(r) else 0.0,
        "pf_r": profit_factor(r), "win_rate_pct": float((r > 0).mean() * 100.0) if len(r) else 0.0,
        "max_dd_r": max_dd_r(r), "best_r": float(r.max()) if len(r) else 0.0,
        "worst_r": float(r.min()) if len(r) else 0.0,
        "max_losing_streak": losing_streak(r),
        "max_open_observed": int(eq["open_positions"].max()) if not eq.empty else 0,
        "max_heat_observed_pct": float(eq["portfolio_heat_pct"].max()) if not eq.empty else 0.0,
    }


def write_report(summary):
    lines = [
        "PHASE T11 — TREND PORTFOLIO HEAT & CAPACITY REPORT",
        "=" * 80,
        "",
        "Offline study only. T9 paper-live remains frozen.",
        "T10 insight included: ACT3/4/5 with ATR3 are compared.",
        "Main question: can max_open / heat / risk increase without destroying DD?",
        "",
        "TOP BY RETURN WITH DD FILTER <= 15%",
        "-" * 80,
    ]
    if summary.empty:
        lines.append("No results.")
    else:
        filt = summary[summary["max_dd_pct"] >= -15].sort_values(["return_pct", "pf_r"], ascending=False)
        if filt.empty:
            lines.append("No variants passed DD <= 15%.")
        else:
            for _, r in filt.head(25).iterrows():
                lines.append(
                    f"{r['variant']} max{int(r['max_open'])} risk={r['risk_pct']*100:.2f}% heat={r['max_heat']*100:.2f}% | "
                    f"return={r['return_pct']:.2f}% DD={r['max_dd_pct']:.2f}% PF={r['pf_r']:.2f} "
                    f"trades={int(r['accepted_trades'])}/{int(r['input_trades'])} maxHeat={r['max_heat_observed_pct']:.2f}%"
                )
        lines += [
            "", "WARNING", "- Do not increase max open and risk exposure live from one good row.",
            "- Prefer stable regions across nearby max_open/heat/risk values.",
            "- Any live change should be incremental: one risk lever at a time.",
        ]
    (OUT / "phase_t11_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE T11 — TREND PORTFOLIO HEAT & CAPACITY STUDY")
    print("=" * 80)
    print(f"Input:  {INPUT_TRADES}")
    print(f"Output: {OUT}")
    if not INPUT_TRADES.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_TRADES}")
    trades = normalize(pd.read_csv(INPUT_TRADES))
    trades = trades[trades["variant"].isin(VARIANTS)].copy()
    if trades.empty:
        raise ValueError("No trades for selected variants. Check T10 file.")
    all_acc, all_eq, summaries = [], [], []
    for variant in VARIANTS:
        vdf = trades[trades["variant"] == variant].copy()
        print(f"\n[VARIANT] {variant} input trades={len(vdf)}")
        for max_open in MAX_OPEN_LEVELS:
            for risk_pct in RISK_LEVELS:
                for max_heat in HEAT_LEVELS:
                    if max_heat < risk_pct:
                        continue
                    acc, skip, eq = simulate(vdf, variant, max_open, risk_pct, max_heat)
                    summaries.append(summarize(acc, skip, eq, len(vdf), variant, max_open, risk_pct, max_heat))
                    if not acc.empty:
                        all_acc.append(acc)
                    if not eq.empty:
                        all_eq.append(eq)
                    print(f"  max{max_open} risk={risk_pct*100:.2f}% heat={max_heat*100:.2f}% trades={len(acc)}/{len(vdf)}")
    summary = pd.DataFrame(summaries).sort_values(["return_pct", "pf_r"], ascending=False)
    acc_out = pd.concat(all_acc, ignore_index=True) if all_acc else pd.DataFrame()
    eq_out = pd.concat(all_eq, ignore_index=True) if all_eq else pd.DataFrame()
    summary.to_csv(OUT / "phase_t11_portfolio_heat_summary.csv", index=False)
    acc_out.to_csv(OUT / "phase_t11_portfolio_heat_trades.csv", index=False)
    eq_out.to_csv(OUT / "phase_t11_portfolio_heat_equity.csv", index=False)
    write_report(summary)
    print("\nTOP RESULTS")
    cols = ["variant", "max_open", "risk_pct_label", "max_heat_label", "accepted_trades", "return_pct", "max_dd_pct", "pf_r", "avg_r", "max_heat_observed_pct"]
    print(summary[cols].head(25).to_string(index=False))
    print("\n[OK] phase_t11_portfolio_heat_summary.csv")
    print("[OK] phase_t11_portfolio_heat_trades.csv")
    print("[OK] phase_t11_portfolio_heat_equity.csv")
    print("[OK] phase_t11_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
