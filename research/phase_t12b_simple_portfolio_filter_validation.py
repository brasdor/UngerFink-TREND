#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T12B — SIMPLE PORTFOLIO FILTER VALIDATION
===============================================

Offline only. T9 paper-live remains frozen.

Goal:
Validate simple Unger-style portfolio filters after T12 showed crowding:
- max same-side exposure
- max per cluster
- max high-beta exposure

No dynamic heat. No ML. No adaptive parameters. No entry/exit changes.

Input:
    data/research_trend_t10/phase_t10_trailing_trades.csv

Outputs:
    data/research_trend_t12b/
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "data" / "research_trend_t10" / "phase_t10_trailing_trades.csv"
OUT = ROOT / "data" / "research_trend_t12b"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
VARIANTS = ["ACT4_ATR3", "ACT3_ATR3", "ACT5_ATR3"]

MAX_OPEN = 5
RISK_PCT = 0.0025
MAX_HEAT = 0.015
ONE_POSITION_PER_SYMBOL = True

FILTERS = [
    ("BASE_MAX5_ONLY", 5, 99, 99),
    ("SAME3_CLUSTER2_BETA3", 3, 2, 3),
    ("SAME3_CLUSTER3_BETA3", 3, 3, 3),
    ("SAME4_CLUSTER2_BETA3", 4, 2, 3),
    ("SAME4_CLUSTER3_BETA4", 4, 3, 4),
    ("SAME5_CLUSTER2_BETA4", 5, 2, 4),
]


def asset_cluster(symbol: str) -> str:
    base = str(symbol).split("/")[0].upper()
    meme = {"DOGE","SHIB","PEPE","FLOKI","BONK","WIF","MEME","TURBO","BOME","PENGU","TRUMP","1000SATS","LUNC"}
    ai = {"FET","RNDR","RENDER","TAO","WLD","ARKM","AGIX","OCEAN","AI","NFP","PHB","CTXC","GRT"}
    defi = {"UNI","AAVE","MKR","COMP","SNX","CRV","CAKE","LDO","RPL","PENDLE","ENA","JTO","JUP","DYDX","GMX","1INCH","SUSHI"}
    gaming = {"AXS","SAND","MANA","GALA","IMX","RONIN","PIXEL","YGG","ILV","PORTAL","ACE","MAGIC"}
    exchange = {"BNB","OKB","CRO","KCS","GT","MX","LEO","FTT"}
    l1_l2 = {"ETH","SOL","ADA","AVAX","DOT","NEAR","APT","SUI","ICP","ATOM","INJ","SEI","TIA","ARB","OP","STRK","MANTA","METIS","FTM","ALGO","EGLD","KAVA","MINA","CELO","ROSE"}
    rwa = {"ONDO","OM","POLYX","TRU","CFG","MPL"}
    privacy = {"ZEC","DASH","XMR","ZEN","SCRT"}
    if base in meme: return "MEME"
    if base in ai: return "AI"
    if base in defi: return "DEFI"
    if base in gaming: return "GAMING"
    if base in exchange: return "EXCHANGE"
    if base in l1_l2: return "L1_L2"
    if base in rwa: return "RWA"
    if base in privacy: return "PRIVACY"
    return "ALT_OTHER"


def high_beta_bucket(symbol: str) -> str:
    c = asset_cluster(symbol)
    if c in {"MEME", "AI", "DEFI", "GAMING", "L1_L2", "ALT_OTHER"}:
        return "HIGH_CRYPTO_BETA"
    return "OTHER"


def profit_factor(r):
    s = pd.to_numeric(r, errors="coerce").dropna()
    if s.empty: return 0.0
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    if l <= 1e-12:
        return np.inf if g > 0 else 0.0
    return float(g / l)


def max_dd_pct(equity):
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if e.empty: return 0.0
    peak = e.cummax()
    return float(((e - peak) / peak * 100).min())


def max_dd_r(r):
    s = pd.to_numeric(r, errors="coerce").fillna(0).values
    if len(s) == 0: return 0.0
    eq = np.cumsum(s)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def losing_streak(r):
    best = cur = 0
    for x in pd.to_numeric(r, errors="coerce").fillna(0):
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def effective_clusters(open_positions):
    counts = {}
    for p in open_positions.values():
        c = p["cluster"]
        counts[c] = counts.get(c, 0) + 1
    vals = np.array(list(counts.values()), dtype=float)
    if vals.sum() <= 0: return 0.0
    w = vals / vals.sum()
    hhi = np.sum(w*w)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def load_trades():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")
    df = pd.read_csv(INPUT)
    required = ["variant", "symbol", "side", "entry_time", "exit_time", "net_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in T10 trades: {missing}")
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
    df = df.dropna(subset=["entry_time", "exit_time", "net_r"]).copy()
    df = df[df["exit_time"] >= df["entry_time"]].copy()
    df = df[df["variant"].isin(VARIANTS)].copy()
    df["trade_id"] = np.arange(len(df))
    df["cluster"] = df["symbol"].apply(asset_cluster)
    df["beta_bucket"] = df["symbol"].apply(high_beta_bucket)
    return df


def state_row(variant, label, t, event, equity, peak, open_pos):
    heat = sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12) * 100
    dd_pct = (equity - peak) / max(peak, 1e-12) * 100
    long_n = sum(1 for p in open_pos.values() if str(p["side"]).upper() == "LONG")
    short_n = sum(1 for p in open_pos.values() if str(p["side"]).upper() == "SHORT")
    clusters = {}
    for p in open_pos.values():
        clusters[p["cluster"]] = clusters.get(p["cluster"], 0) + 1
    return {
        "variant": variant, "filter_label": label, "time": t, "event": event,
        "equity": equity, "peak": peak, "drawdown_pct": dd_pct,
        "open_positions": len(open_pos), "long_positions": long_n, "short_positions": short_n,
        "portfolio_heat_pct": heat, "effective_clusters": effective_clusters(open_pos),
        "max_cluster_count": max(clusters.values()) if clusters else 0,
    }


def replay(vdf, variant, label, max_same_side, max_cluster, max_beta):
    events, row_by_id = [], {}
    vdf = vdf.sort_values(["entry_time", "exit_time", "symbol"]).reset_index(drop=True)
    for _, row in vdf.iterrows():
        tid = int(row["trade_id"])
        row_by_id[tid] = row
        events.append((row["entry_time"], 1, tid))
        events.append((row["exit_time"], 0, tid))
    events.sort(key=lambda x: (x[0], x[1]))
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    open_pos = {}
    accepted, skipped, equity_rows = [], [], []
    for t, event_type, tid in events:
        if event_type == 0:
            pos = open_pos.pop(tid, None)
            if pos is None: continue
            pnl = pos["risk_amount"] * pos["net_r"]
            equity += pnl
            peak = max(peak, equity)
            dd_pct = (equity - peak) / max(peak, 1e-12) * 100
            out = dict(pos)
            out.update({"exit_time": t, "pnl_usdt": pnl, "equity_after_exit": equity,
                        "drawdown_pct": dd_pct, "open_after_exit": len(open_pos)})
            accepted.append(out)
            equity_rows.append(state_row(variant, label, t, "exit", equity, peak, open_pos))
            continue
        row = row_by_id[tid]
        side = str(row["side"]).upper()
        symbol = row["symbol"]
        cluster = row["cluster"]
        beta = row["beta_bucket"]
        heat = sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12)
        same_side_count = sum(1 for p in open_pos.values() if str(p["side"]).upper() == side)
        cluster_count = sum(1 for p in open_pos.values() if p["cluster"] == cluster)
        beta_count = sum(1 for p in open_pos.values() if p["beta_bucket"] == beta)
        symbol_open = any(p["symbol"] == symbol for p in open_pos.values())
        reason = None
        if len(open_pos) >= MAX_OPEN: reason = "MAX_OPEN"
        elif heat + RISK_PCT > MAX_HEAT + 1e-12: reason = "MAX_HEAT"
        elif ONE_POSITION_PER_SYMBOL and symbol_open: reason = "SYMBOL_ALREADY_OPEN"
        elif same_side_count >= max_same_side: reason = "MAX_SAME_SIDE"
        elif cluster_count >= max_cluster: reason = "MAX_CLUSTER"
        elif beta == "HIGH_CRYPTO_BETA" and beta_count >= max_beta: reason = "MAX_HIGH_BETA"
        if reason:
            skipped.append({"variant": variant, "filter_label": label, "trade_id": tid, "symbol": symbol,
                            "side": side, "cluster": cluster, "beta_bucket": beta, "entry_time": row["entry_time"],
                            "skip_reason": reason, "open_positions": len(open_pos),
                            "same_side_count": same_side_count, "cluster_count": cluster_count,
                            "beta_count": beta_count, "heat_pct": heat * 100})
            continue
        risk_amount = equity * RISK_PCT
        open_pos[tid] = {"variant": variant, "filter_label": label, "trade_id": tid, "symbol": symbol,
                         "side": side, "cluster": cluster, "beta_bucket": beta,
                         "entry_time": row["entry_time"], "entry_price": row.get("entry_price", np.nan),
                         "initial_stop": row.get("initial_stop", np.nan), "final_stop": row.get("final_stop", np.nan),
                         "net_r": float(row["net_r"]), "risk_amount": risk_amount,
                         "equity_before_entry": equity, "heat_before_entry_pct": heat * 100,
                         "same_side_before": same_side_count, "cluster_before": cluster_count,
                         "beta_before": beta_count}
        equity_rows.append(state_row(variant, label, t, "entry", equity, peak, open_pos))
    return pd.DataFrame(accepted), pd.DataFrame(skipped), pd.DataFrame(equity_rows)


def summarize(acc, skip, eq, input_count, variant, label, max_same, max_cluster, max_beta):
    base = {"variant": variant, "filter_label": label, "max_same_side": max_same,
            "max_cluster": max_cluster, "max_high_beta": max_beta,
            "input_trades": input_count, "accepted_trades": 0, "skipped_trades": len(skip),
            "acceptance_rate_pct": 0.0, "return_pct": 0.0, "max_dd_pct": 0.0,
            "pf_r": 0.0, "total_r": 0.0, "avg_r": 0.0, "win_rate_pct": 0.0,
            "max_dd_r": 0.0, "best_r": 0.0, "worst_r": 0.0,
            "max_losing_streak": 0, "max_open_observed": 0, "max_heat_observed_pct": 0.0,
            "avg_effective_clusters": 0.0, "min_effective_clusters": 0.0}
    if acc.empty: return base
    r = pd.to_numeric(acc["net_r"], errors="coerce").dropna()
    final_equity = float(acc["equity_after_exit"].iloc[-1])
    base.update({
        "accepted_trades": len(acc), "skipped_trades": len(skip),
        "acceptance_rate_pct": len(acc) / max(input_count, 1) * 100,
        "return_pct": (final_equity / INITIAL_CAPITAL - 1) * 100,
        "final_equity": final_equity,
        "max_dd_pct": max_dd_pct(acc["equity_after_exit"]),
        "pf_r": profit_factor(r), "total_r": float(r.sum()),
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "win_rate_pct": float((r > 0).mean() * 100) if len(r) else 0.0,
        "max_dd_r": max_dd_r(r), "best_r": float(r.max()) if len(r) else 0.0,
        "worst_r": float(r.min()) if len(r) else 0.0,
        "max_losing_streak": losing_streak(r),
        "max_open_observed": int(eq["open_positions"].max()) if not eq.empty else 0,
        "max_heat_observed_pct": float(eq["portfolio_heat_pct"].max()) if not eq.empty else 0.0,
        "avg_effective_clusters": float(eq["effective_clusters"].mean()) if not eq.empty else 0.0,
        "min_effective_clusters": float(eq["effective_clusters"].min()) if not eq.empty else 0.0,
    })
    return base


def write_report(summary):
    lines = []
    lines.append("PHASE T12B — SIMPLE PORTFOLIO FILTER VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Offline only. T9 paper-live remains frozen.")
    lines.append("Simple rules only: max same-side, max cluster, max high-beta.")
    lines.append("No dynamic heat, no ML, no adaptive logic.")
    lines.append("")
    if summary.empty:
        lines.append("No results.")
    else:
        lines.append("TOP RESULTS BY RETURN")
        lines.append("-" * 80)
        for _, r in summary.sort_values(["return_pct", "pf_r"], ascending=False).head(30).iterrows():
            lines.append(f"{r['variant']} | {r['filter_label']}: return={r['return_pct']:.2f}% "
                         f"DD={r['max_dd_pct']:.2f}% PF={r['pf_r']:.2f} "
                         f"trades={int(r['accepted_trades'])}/{int(r['input_trades'])} "
                         f"avgEffCl={r.get('avg_effective_clusters', 0):.2f}")
        lines.append("")
        lines.append("BEST WITH DD <= 10%")
        lines.append("-" * 80)
        filt = summary[summary["max_dd_pct"] >= -10].sort_values(["return_pct", "pf_r"], ascending=False)
        if filt.empty:
            lines.append("No candidate passed DD <= 10%.")
        else:
            for _, r in filt.head(20).iterrows():
                lines.append(f"{r['variant']} | {r['filter_label']}: return={r['return_pct']:.2f}% "
                             f"DD={r['max_dd_pct']:.2f}% PF={r['pf_r']:.2f} "
                             f"trades={int(r['accepted_trades'])}/{int(r['input_trades'])}")
    lines.append("")
    lines.append("Interpretation rule:")
    lines.append("- If simple filters reduce DD but keep PF below 1, do NOT update T9.")
    lines.append("- If one filter is stable across ACT3/ACT4/ACT5 and improves DD without killing PF, it becomes a candidate.")
    lines.append("- Simplicity is preferred over adaptive portfolio engineering.")
    (OUT / "phase_t12b_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE T12B — SIMPLE PORTFOLIO FILTER VALIDATION")
    print("=" * 80)
    print(f"Input:  {INPUT}")
    print(f"Output: {OUT}")
    trades = load_trades()
    if trades.empty:
        raise ValueError("No trades loaded for selected variants.")
    all_acc, all_skip, all_eq, summaries = [], [], [], []
    for variant in VARIANTS:
        vdf = trades[trades["variant"] == variant].copy()
        if vdf.empty: continue
        print(f"\n[VARIANT] {variant} trades={len(vdf)}")
        for label, max_same, max_cluster, max_beta in FILTERS:
            acc, skip, eq = replay(vdf, variant, label, max_same, max_cluster, max_beta)
            summaries.append(summarize(acc, skip, eq, len(vdf), variant, label, max_same, max_cluster, max_beta))
            if not acc.empty: all_acc.append(acc)
            if not skip.empty: all_skip.append(skip)
            if not eq.empty: all_eq.append(eq)
            print(f"  {label}: accepted={len(acc)}/{len(vdf)}")
    summary = pd.DataFrame(summaries).sort_values(["return_pct", "pf_r"], ascending=False)
    acc_df = pd.concat(all_acc, ignore_index=True) if all_acc else pd.DataFrame()
    skip_df = pd.concat(all_skip, ignore_index=True) if all_skip else pd.DataFrame()
    eq_df = pd.concat(all_eq, ignore_index=True) if all_eq else pd.DataFrame()
    summary.to_csv(OUT / "phase_t12b_simple_filter_summary.csv", index=False)
    acc_df.to_csv(OUT / "phase_t12b_simple_filter_trades.csv", index=False)
    skip_df.to_csv(OUT / "phase_t12b_simple_filter_skipped.csv", index=False)
    eq_df.to_csv(OUT / "phase_t12b_simple_filter_equity.csv", index=False)
    write_report(summary)
    print("\nTOP RESULTS")
    cols = ["variant", "filter_label", "accepted_trades", "return_pct", "max_dd_pct", "pf_r", "avg_r", "avg_effective_clusters"]
    print(summary[cols].head(30).to_string(index=False))
    print("\n[OK] phase_t12b_simple_filter_summary.csv")
    print("[OK] phase_t12b_simple_filter_trades.csv")
    print("[OK] phase_t12b_simple_filter_skipped.csv")
    print("[OK] phase_t12b_simple_filter_equity.csv")
    print("[OK] phase_t12b_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
