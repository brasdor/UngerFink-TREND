#!/usr/bin/env python3
"""
T5 portfolio replay, generic across candidates -- entry AND exit both frozen
(T3/T4 closed). Tests whether the edge survives realistic position sizing and
exposure constraints at $60k target deployment capital, matching the existing
repo's T5 convention (research/phase_t5_trend_portfolio_replay_engine.py:
max_open concurrent positions, signal rejection when caps are breached,
RISK_PER_TRADE_PCT=0.25% risk-per-trade -- same number used in step23 and
phase_t5, kept consistent here).

Three caps, applied together in one chronological replay:
  - max_position_pct: no single position's notional may exceed this fraction
    of capital (position is SIZE-CAPPED, not rejected -- if fixed-risk sizing
    would need a bigger notional than this to risk 0.25%, the position is
    simply smaller and risks less than the target 0.25%, matching how a real
    broker/risk-desk would behave).
  - max_open: no more than this many concurrent positions across the whole
    portfolio -- new entries beyond this are REJECTED (skipped entirely, no
    partial fill / queueing, matching phase_t5's MAX_OPEN_REACHED convention).
  - max_cluster_pct: no more than this fraction of capital concentrated in one
    correlation cluster (reusing modules/asset_clustering.py, built for
    candidate 16) at once -- new entries that would breach it are REJECTED.

Uncapped baseline = same replay with all three caps set to effectively
infinite, for an apples-to-apples dollar comparison (not just R-multiple
avg_r, which wouldn't capture the effect of entirely rejected trades).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.asset_clustering import compute_static_clusters  # noqa: E402

ATR_MULT_R = 2.0
CAPITAL_BASE = 60_000.0
RISK_PCT_PER_TRADE = 0.0025  # matches step23/phase_t5 convention

DEFAULT_MAX_OPEN = 20
DEFAULT_MAX_POSITION_PCT = 0.10
DEFAULT_MAX_CLUSTER_PCT = 0.30
CLUSTER_CORR_THRESHOLD = 0.6  # matches one of candidate 16's own grid values


def build_symbol_cluster_map(panel: dict[str, pd.DataFrame], start, end) -> dict[str, str]:
    """Static clustering (same simplification/caveat as candidate 16 -- full-window
    correlation structure, not walk-forward; defensible here since this is a risk-
    management grouping, not a trading signal, but noted for completeness)."""
    clusters = compute_static_clusters(panel, corr_threshold=CLUSTER_CORR_THRESHOLD,
                                        min_cluster_size=3, start=start, end=end)
    symbol_to_cluster = {}
    for cluster_id, members in clusters.items():
        if cluster_id == "unclustered":
            # asset_clustering's "unclustered" bucket just means "didn't find
            # >=3 correlated peers" -- these symbols are NOT necessarily
            # correlated with each other, so each gets its own singleton
            # cluster rather than being lumped into one shared concentration
            # bucket (which would wrongly cap unrelated-symbol exposure).
            for m in members:
                symbol_to_cluster[m] = f"solo_{m}"
        else:
            for m in members:
                symbol_to_cluster[m] = cluster_id
    return symbol_to_cluster


def replay_with_caps(trades: pd.DataFrame, symbol_cluster_map: dict[str, str],
                      max_open: float = DEFAULT_MAX_OPEN,
                      max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
                      max_cluster_pct: float = DEFAULT_MAX_CLUSTER_PCT) -> dict:
    """trades must have: entry_date, exit_date, entry_px, symbol, net_r, atr_at_entry."""
    t = trades.sort_values("entry_date").reset_index(drop=True)

    initial_risk = ATR_MULT_R * t["atr_at_entry"]
    stop_distance_pct = (initial_risk / t["entry_px"]).clip(lower=1e-9)
    target_notional_pct = (RISK_PCT_PER_TRADE / stop_distance_pct)

    open_positions: list[dict] = []  # {symbol, cluster, notional_pct, exit_date}
    accepted_flags = np.zeros(len(t), dtype=bool)
    dollar_pnl = np.zeros(len(t))
    reject_reasons = []
    max_open_observed = 0

    for i, row in t.iterrows():
        open_positions = [p for p in open_positions if p["exit_date"] > row["entry_date"]]
        max_open_observed = max(max_open_observed, len(open_positions))

        cluster = symbol_cluster_map.get(row["symbol"], "unclustered")
        cluster_pct_open = sum(p["notional_pct"] for p in open_positions if p["cluster"] == cluster)

        reason = None
        if len(open_positions) >= max_open:
            reason = "MAX_OPEN"
        else:
            actual_notional_pct = min(target_notional_pct[i], max_position_pct)
            if cluster_pct_open + actual_notional_pct > max_cluster_pct:
                reason = "MAX_CLUSTER_PCT"

        if reason:
            reject_reasons.append(reason)
            continue

        actual_notional_pct = min(target_notional_pct[i], max_position_pct)
        actual_risk_dollars = actual_notional_pct * stop_distance_pct[i] * CAPITAL_BASE
        dollar_pnl[i] = actual_risk_dollars * row["net_r"]
        accepted_flags[i] = True
        open_positions.append({"symbol": row["symbol"], "cluster": cluster,
                                "notional_pct": actual_notional_pct, "exit_date": row["exit_date"]})

    t = t.copy()
    t["accepted"] = accepted_flags
    t["dollar_pnl"] = dollar_pnl

    accepted = t[t["accepted"]]
    n_rejected = int((~t["accepted"]).sum())
    reject_counts = pd.Series(reject_reasons).value_counts().to_dict() if reject_reasons else {}

    exit_dates = pd.to_datetime(accepted["exit_date"])
    daily_pnl = pd.Series(accepted["dollar_pnl"].to_numpy(), index=exit_dates).groupby(level=0).sum()
    date_min, date_max = t["entry_date"].min(), t["exit_date"].max()
    daily_idx = pd.date_range(date_min, date_max, freq="D")
    daily_pnl = daily_pnl.reindex(daily_idx).fillna(0.0)
    equity = CAPITAL_BASE + daily_pnl.cumsum()
    rets = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    pk = equity.cummax()
    max_dd = ((equity - pk) / pk).min()
    n_years = max((date_max - date_min).days / 365.25, 1e-6)
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan

    return dict(
        n_trades_total=len(t), n_accepted=len(accepted), n_rejected=n_rejected,
        rejection_rate=n_rejected / len(t) if len(t) else 0.0, reject_counts=reject_counts,
        max_open_observed=max_open_observed,
        total_return=total_return, max_dd=max_dd, sharpe=sharpe,
        avg_r_accepted=accepted["net_r"].mean() if len(accepted) else np.nan,
        final_equity=equity.iloc[-1],
    )
