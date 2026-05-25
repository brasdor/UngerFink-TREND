#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T12 — TREND CLUSTER & CORRELATION EXPOSURE ENGINE
======================================================

Offline portfolio-level risk study for the Trend Following system.
It does NOT change T9 paper-live.

Input:
    data/research_trend_t10/phase_t10_trailing_trades.csv

Outputs:
    data/research_trend_t12/
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "data" / "research_trend_t10" / "phase_t10_trailing_trades.csv"
OUT = ROOT / "data" / "research_trend_t12"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
VARIANTS_TO_TEST = ["ACT4_ATR3", "ACT3_ATR3", "ACT5_ATR3"]
RISK_PCT = 0.0025
MAX_OPEN = 5
MAX_HEAT = 0.015
MAX_SAME_SIDE_LEVELS = [2, 3, 4, 5]
MAX_PER_CLUSTER_LEVELS = [1, 2, 3]
MAX_BETA_CLUSTER_LEVELS = [1, 2, 3, 5]
ONE_POSITION_PER_SYMBOL = True


def asset_cluster(symbol: str) -> str:
    base = str(symbol).split("/")[0].upper()
    l1_l2 = {"ETH","BNB","SOL","ADA","AVAX","DOT","NEAR","APT","SUI","ICP","ATOM","INJ","SEI","TIA","ARB","OP","STRK","MANTA","METIS","FTM","ALGO","EGLD","KAVA","MINA","CELO","ROSE"}
    meme = {"DOGE","SHIB","PEPE","FLOKI","BONK","WIF","MEME","TURBO","BOME","PENGU","TRUMP","1000SATS","LUNC"}
    ai = {"FET","RNDR","RENDER","TAO","WLD","ARKM","AGIX","OCEAN","AI","NFP","PHB","CTXC","GRT"}
    defi = {"UNI","AAVE","MKR","COMP","SNX","CRV","CAKE","LDO","RPL","PENDLE","ENA","JTO","JUP","DYDX","GMX","1INCH","SUSHI"}
    gaming = {"AXS","SAND","MANA","GALA","IMX","RONIN","PIXEL","YGG","ILV","PORTAL","ACE","MAGIC"}
    exchange = {"BNB","OKB","CRO","KCS","GT","MX","LEO","FTT"}
    rwa = {"ONDO","OM","POLYX","TRU","CFG","MPL","PENDLE"}
    privacy = {"ZEC","DASH","XMR","ZEN","SCRT"}
    btc_beta = {"BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","LINK","TRX","DOT","BCH","LTC","UNI","NEAR","APT","SUI","ICP"}
    if base in meme: return "MEME"
    if base in ai: return "AI"
    if base in defi: return "DEFI"
    if base in gaming: return "GAMING"
    if base in exchange: return "EXCHANGE"
    if base in rwa: return "RWA"
    if base in privacy: return "PRIVACY"
    if base in l1_l2: return "L1_L2"
    if base in btc_beta: return "BTC_BETA"
    return "ALT_OTHER"


def beta_bucket(symbol: str) -> str:
    c = asset_cluster(symbol)
    return "HIGH_CRYPTO_BETA" if c in {"BTC_BETA","L1_L2","MEME","DEFI","AI","GAMING"} else "OTHER_BETA"


def profit_factor(r):
    s = pd.to_numeric(r, errors="coerce").dropna()
    if s.empty: return 0.0
    g, l = s[s > 0].sum(), -s[s < 0].sum()
    return float(g / l) if l > 1e-12 else (np.inf if g > 0 else 0.0)


def max_dd_pct(equity):
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if e.empty: return 0.0
    peak = e.cummax()
    return float(((e - peak) / peak * 100.0).min())


def max_dd_r(r):
    arr = pd.to_numeric(r, errors="coerce").fillna(0).values
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def losing_streak(r):
    best = cur = 0
    for v in pd.to_numeric(r, errors="coerce").fillna(0):
        if v < 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def effective_positions(cluster_counts):
    counts = np.array(list(cluster_counts.values()), dtype=float)
    if counts.sum() <= 0: return 0.0
    w = counts / counts.sum()
    hhi = np.sum(w * w)
    return float(1.0 / hhi) if hhi > 1e-12 else 0.0


def normalize_trades(df):
    out = df.copy()
    req = ["variant","symbol","side","entry_time","exit_time","net_r"]
    missing = [c for c in req if c not in out.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce", utc=True)
    out["net_r"] = pd.to_numeric(out["net_r"], errors="coerce")
    out = out.dropna(subset=["entry_time","exit_time","net_r"])
    out = out[out["exit_time"] >= out["entry_time"]].copy()
    out["trade_id"] = np.arange(len(out))
    out["cluster"] = out["symbol"].apply(asset_cluster)
    out["beta_bucket"] = out["symbol"].apply(beta_bucket)
    return out


def add_state_rows(exposure_rows, equity_rows, variant, t, event, equity, peak, open_pos,
                   max_same_side, max_per_cluster, max_beta_cluster):
    cluster_counts, beta_counts = {}, {}
    side_counts = {"LONG": 0, "SHORT": 0}
    for p in open_pos.values():
        cluster_counts[p["cluster"]] = cluster_counts.get(p["cluster"], 0) + 1
        side_counts[p["side"]] = side_counts.get(p["side"], 0) + 1
        beta_counts[p["beta_bucket"]] = beta_counts.get(p["beta_bucket"], 0) + 1
    heat = sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12) * 100
    row = {
        "variant": variant, "time": t, "event": event, "equity": equity, "peak": peak,
        "drawdown_pct": (equity - peak) / max(peak, 1e-12) * 100,
        "open_positions": len(open_pos), "long_positions": side_counts.get("LONG", 0),
        "short_positions": side_counts.get("SHORT", 0), "portfolio_heat_pct": heat,
        "effective_cluster_positions": effective_positions(cluster_counts),
        "max_same_side": max_same_side, "max_per_cluster": max_per_cluster,
        "max_beta_cluster": max_beta_cluster,
    }
    for k, v in cluster_counts.items(): row[f"cluster_{k}"] = v
    for k, v in beta_counts.items(): row[f"beta_{k}"] = v
    exposure_rows.append(row)
    equity_rows.append(row.copy())


def simulate(vdf, variant, max_same_side, max_per_cluster, max_beta_cluster):
    events, row_by_id = [], {}
    vdf = vdf.sort_values(["entry_time","exit_time","symbol"]).reset_index(drop=True)
    for _, row in vdf.iterrows():
        tid = int(row["trade_id"]); row_by_id[tid] = row
        events.append((row["entry_time"], 1, tid)); events.append((row["exit_time"], 0, tid))
    events.sort(key=lambda x: (x[0], x[1]))
    equity = INITIAL_CAPITAL; peak = INITIAL_CAPITAL
    open_pos, accepted, skipped, equity_rows, exposure_rows = {}, [], [], [], []
    for t, event_type, tid in events:
        if event_type == 0:
            pos = open_pos.pop(tid, None)
            if pos is None: continue
            pnl = pos["risk_amount"] * pos["net_r"]
            equity += pnl; peak = max(peak, equity)
            res = dict(pos)
            res.update({"exit_time": t, "pnl_usdt": pnl, "equity_after_exit": equity,
                        "drawdown_pct": (equity - peak) / peak * 100, "open_after_exit": len(open_pos)})
            accepted.append(res)
            add_state_rows(exposure_rows, equity_rows, variant, t, "exit", equity, peak, open_pos,
                           max_same_side, max_per_cluster, max_beta_cluster)
            continue
        row = row_by_id[tid]
        symbol, side, cluster, beta = row["symbol"], str(row["side"]).upper(), row["cluster"], row["beta_bucket"]
        current_heat = sum(p["risk_amount"] for p in open_pos.values()) / max(equity, 1e-12)
        same_side_count = sum(1 for p in open_pos.values() if p["side"] == side)
        cluster_count = sum(1 for p in open_pos.values() if p["cluster"] == cluster)
        beta_count = sum(1 for p in open_pos.values() if p["beta_bucket"] == beta)
        symbol_open = any(p["symbol"] == symbol for p in open_pos.values())
        reason = None
        if len(open_pos) >= MAX_OPEN: reason = "MAX_OPEN"
        elif current_heat + RISK_PCT > MAX_HEAT + 1e-12: reason = "MAX_HEAT"
        elif ONE_POSITION_PER_SYMBOL and symbol_open: reason = "SYMBOL_ALREADY_OPEN"
        elif same_side_count >= max_same_side: reason = "MAX_SAME_SIDE"
        elif cluster_count >= max_per_cluster: reason = "MAX_PER_CLUSTER"
        elif beta == "HIGH_CRYPTO_BETA" and beta_count >= max_beta_cluster: reason = "MAX_HIGH_BETA"
        if reason:
            skipped.append({"variant": variant, "trade_id": tid, "symbol": symbol, "side": side,
                            "cluster": cluster, "beta_bucket": beta, "entry_time": row["entry_time"],
                            "skip_reason": reason, "open_positions": len(open_pos),
                            "same_side_count": same_side_count, "cluster_count": cluster_count,
                            "beta_count": beta_count, "heat_pct": current_heat * 100,
                            "max_same_side": max_same_side, "max_per_cluster": max_per_cluster,
                            "max_beta_cluster": max_beta_cluster})
            continue
        open_pos[tid] = {"variant": variant, "trade_id": tid, "symbol": symbol, "side": side,
                         "cluster": cluster, "beta_bucket": beta, "entry_time": row["entry_time"],
                         "entry_price": row.get("entry_price", np.nan), "initial_stop": row.get("initial_stop", np.nan),
                         "final_stop": row.get("final_stop", np.nan), "net_r": float(row["net_r"]),
                         "risk_amount": equity * RISK_PCT, "max_same_side": max_same_side,
                         "max_per_cluster": max_per_cluster, "max_beta_cluster": max_beta_cluster,
                         "equity_before_entry": equity, "heat_before_entry_pct": current_heat * 100}
        add_state_rows(exposure_rows, equity_rows, variant, t, "entry", equity, peak, open_pos,
                       max_same_side, max_per_cluster, max_beta_cluster)
    return pd.DataFrame(accepted), pd.DataFrame(skipped), pd.DataFrame(equity_rows), pd.DataFrame(exposure_rows)


def summarize(acc, skip, eq, exposure, input_count, variant, max_same_side, max_per_cluster, max_beta_cluster):
    if acc.empty:
        return {"variant": variant, "max_same_side": max_same_side, "max_per_cluster": max_per_cluster,
                "max_beta_cluster": max_beta_cluster, "input_trades": input_count, "accepted_trades": 0,
                "skipped_trades": len(skip), "acceptance_rate_pct": 0.0, "return_pct": 0.0,
                "final_equity": INITIAL_CAPITAL, "max_dd_pct": 0.0, "total_r": 0.0, "avg_r": 0.0,
                "pf_r": 0.0, "win_rate_pct": 0.0, "max_dd_r": 0.0, "max_open_observed": 0,
                "max_same_side_observed": 0, "max_heat_observed_pct": 0.0,
                "avg_effective_cluster_positions": 0.0, "min_effective_cluster_positions": 0.0,
                "max_losing_streak": 0}
    r = pd.to_numeric(acc["net_r"], errors="coerce").dropna()
    final_equity = float(acc["equity_after_exit"].iloc[-1])
    max_same_side_obs = int(exposure[["long_positions","short_positions"]].max().max()) if not exposure.empty else 0
    return {"variant": variant, "max_same_side": max_same_side, "max_per_cluster": max_per_cluster,
            "max_beta_cluster": max_beta_cluster, "input_trades": input_count, "accepted_trades": len(acc),
            "skipped_trades": len(skip), "acceptance_rate_pct": len(acc)/max(input_count,1)*100,
            "return_pct": (final_equity/INITIAL_CAPITAL - 1)*100, "final_equity": final_equity,
            "max_dd_pct": max_dd_pct(acc["equity_after_exit"]), "total_r": float(r.sum()),
            "avg_r": float(r.mean()) if len(r) else 0.0, "pf_r": profit_factor(r),
            "win_rate_pct": float((r > 0).mean()*100) if len(r) else 0.0, "max_dd_r": max_dd_r(r),
            "best_r": float(r.max()) if len(r) else 0.0, "worst_r": float(r.min()) if len(r) else 0.0,
            "max_losing_streak": losing_streak(r),
            "max_open_observed": int(exposure["open_positions"].max()) if not exposure.empty else 0,
            "max_same_side_observed": max_same_side_obs,
            "max_heat_observed_pct": float(exposure["portfolio_heat_pct"].max()) if not exposure.empty else 0.0,
            "avg_effective_cluster_positions": float(exposure["effective_cluster_positions"].mean()) if not exposure.empty else 0.0,
            "min_effective_cluster_positions": float(exposure["effective_cluster_positions"].min()) if not exposure.empty else 0.0}


def overlap_summary(trades):
    rows = []
    for variant, g in trades.groupby("variant"):
        events, row_by_id = [], {int(r["trade_id"]): r for _, r in g.iterrows()}
        for _, row in g.iterrows():
            tid = int(row["trade_id"]); events += [(row["entry_time"], 1, tid), (row["exit_time"], 0, tid)]
        events.sort(key=lambda x: (x[0], x[1]))
        open_ids, samples, max_open = set(), [], 0
        for t, typ, tid in events:
            if typ == 0: open_ids.discard(tid)
            else: open_ids.add(tid)
            max_open = max(max_open, len(open_ids))
            if open_ids:
                clusters, sides = {}, {"LONG": 0, "SHORT": 0}
                for oid in open_ids:
                    r = row_by_id[oid]
                    clusters[r["cluster"]] = clusters.get(r["cluster"], 0) + 1
                    s = str(r["side"]).upper(); sides[s] = sides.get(s, 0) + 1
                samples.append({"open": len(open_ids), "eff_clusters": effective_positions(clusters),
                                "max_same_side": max(sides.values()),
                                "max_cluster_count": max(clusters.values()) if clusters else 0})
        sdf = pd.DataFrame(samples)
        rows.append({"variant": variant, "raw_trades": len(g), "raw_max_simultaneous_open": max_open,
                     "avg_raw_open": float(sdf["open"].mean()) if not sdf.empty else 0.0,
                     "avg_effective_clusters": float(sdf["eff_clusters"].mean()) if not sdf.empty else 0.0,
                     "max_raw_same_side": int(sdf["max_same_side"].max()) if not sdf.empty else 0,
                     "max_raw_cluster_count": int(sdf["max_cluster_count"].max()) if not sdf.empty else 0})
    return pd.DataFrame(rows)


def write_report(summary, overlap):
    lines = ["PHASE T12 — CLUSTER & CORRELATION EXPOSURE REPORT", "="*80, "",
             "Offline study only. T9 paper-live remains frozen.",
             "Goal: verify if portfolio weakness comes from correlated crypto-beta exposure.", "",
             "RAW OVERLAP SUMMARY", "-"*80]
    if overlap.empty: lines.append("No overlap data.")
    else:
        for _, r in overlap.iterrows():
            lines.append(f"{r['variant']}: rawTrades={int(r['raw_trades'])}, rawMaxOpen={int(r['raw_max_simultaneous_open'])}, avgOpen={r['avg_raw_open']:.2f}, avgEffClusters={r['avg_effective_clusters']:.2f}, maxSameSide={int(r['max_raw_same_side'])}, maxClusterCount={int(r['max_raw_cluster_count'])}")
    lines += ["", "TOP FILTERED RESULTS BY RETURN WITH DD <= 12%", "-"*80]
    filt = summary[summary["max_dd_pct"] >= -12].sort_values(["return_pct", "pf_r"], ascending=False)
    if filt.empty: lines.append("No candidate passed DD <= 12%.")
    else:
        for _, r in filt.head(25).iterrows():
            lines.append(f"{r['variant']} sameSide<={int(r['max_same_side'])} cluster<={int(r['max_per_cluster'])} beta<={int(r['max_beta_cluster'])}: return={r['return_pct']:.2f}% DD={r['max_dd_pct']:.2f}% PF={r['pf_r']:.2f} trades={int(r['accepted_trades'])}/{int(r['input_trades'])} effClustersAvg={r['avg_effective_cluster_positions']:.2f}")
    lines += ["", "Interpretation rules:",
              "- If cluster filters improve DD but kill all returns, they are too restrictive.",
              "- If max same-side improves stability, same-side crowding was the issue.",
              "- Prefer stable regions, not the single best row.",
              "- Do not update T9 until paper-live and offline results agree."]
    (OUT / "phase_t12_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("="*80); print("PHASE T12 — CLUSTER & CORRELATION EXPOSURE ENGINE"); print("="*80)
    print(f"Input:  {INPUT}"); print(f"Output: {OUT}")
    if not INPUT.exists(): raise FileNotFoundError(f"Missing input file: {INPUT}")
    trades = normalize_trades(pd.read_csv(INPUT))
    trades = trades[trades["variant"].isin(VARIANTS_TO_TEST)].copy()
    if trades.empty: raise ValueError("No selected variants found in T10 trades.")
    trades[["symbol","cluster","beta_bucket"]].drop_duplicates().sort_values(["cluster","symbol"]).to_csv(OUT / "phase_t12_asset_cluster_map.csv", index=False)
    overlap = overlap_summary(trades); overlap.to_csv(OUT / "phase_t12_overlap_summary.csv", index=False)
    summaries, all_acc, all_eq, all_exp = [], [], [], []
    for variant in VARIANTS_TO_TEST:
        vdf = trades[trades["variant"] == variant].copy()
        print(f"\n[VARIANT] {variant} trades={len(vdf)}")
        for max_same in MAX_SAME_SIDE_LEVELS:
            for max_cluster in MAX_PER_CLUSTER_LEVELS:
                for max_beta in MAX_BETA_CLUSTER_LEVELS:
                    acc, skip, eq, exp = simulate(vdf, variant, max_same, max_cluster, max_beta)
                    summaries.append(summarize(acc, skip, eq, exp, len(vdf), variant, max_same, max_cluster, max_beta))
                    if not acc.empty: all_acc.append(acc)
                    if not eq.empty: all_eq.append(eq)
                    if not exp.empty: all_exp.append(exp)
                    print(f"  same<={max_same} cluster<={max_cluster} beta<={max_beta} accepted={len(acc)}/{len(vdf)}")
    summary = pd.DataFrame(summaries).sort_values(["return_pct", "pf_r"], ascending=False)
    pd.concat(all_acc, ignore_index=True).to_csv(OUT / "phase_t12_cluster_replay_trades.csv", index=False) if all_acc else pd.DataFrame().to_csv(OUT / "phase_t12_cluster_replay_trades.csv", index=False)
    pd.concat(all_eq, ignore_index=True).to_csv(OUT / "phase_t12_cluster_replay_equity.csv", index=False) if all_eq else pd.DataFrame().to_csv(OUT / "phase_t12_cluster_replay_equity.csv", index=False)
    pd.concat(all_exp, ignore_index=True).to_csv(OUT / "phase_t12_exposure_timeline.csv", index=False) if all_exp else pd.DataFrame().to_csv(OUT / "phase_t12_exposure_timeline.csv", index=False)
    summary.to_csv(OUT / "phase_t12_cluster_replay_summary.csv", index=False)
    write_report(summary, overlap)
    print("\nTOP RESULTS")
    cols = ["variant","max_same_side","max_per_cluster","max_beta_cluster","accepted_trades","return_pct","max_dd_pct","pf_r","avg_effective_cluster_positions","max_heat_observed_pct"]
    print(summary[cols].head(25).to_string(index=False))
    print("\n[OK] phase_t12_cluster_replay_summary.csv")
    print("[OK] phase_t12_cluster_replay_trades.csv")
    print("[OK] phase_t12_cluster_replay_equity.csv")
    print("[OK] phase_t12_exposure_timeline.csv")
    print("[OK] phase_t12_overlap_summary.csv")
    print("[OK] phase_t12_asset_cluster_map.csv")
    print("[OK] phase_t12_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
