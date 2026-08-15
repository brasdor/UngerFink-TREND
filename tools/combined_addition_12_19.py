#!/usr/bin/env python3
"""
Combined-addition test: baseline S1+S6+S7+S8 vs baseline+12+19 together, same
methodology/infrastructure as tools/portfolio_marginal_contribution.py (step23's real
walk-forward regime engine, Scheme C, frequency-scaled risk-per-trade), extended to two
simultaneous candidate sleeves instead of one.

Allocation: each candidate gets a fixed 20% share (matching the single-candidate
convention in portfolio_marginal_contribution.py), baseline compressed to the
remaining 60% (100% - 20% - 20%), preserving baseline's own relative regime-
conditional tilts among S1/S6/S7/S8 unchanged.
"""
from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.t1_harness import DEFAULT_IS_START, DEFAULT_IS_END  # noqa: E402
from tools.portfolio_marginal_contribution import (  # noqa: E402
    BASELINE_SYSTEMS, BASE_ALLOC, CAPITAL_BASE, ANALYSIS_START, ANALYSIS_END,
    load_universe_panel, pool_candidate_trades_c12, risk_per_trade_for,
)

ANALYSIS_START = DEFAULT_IS_START
ANALYSIS_END = DEFAULT_IS_END
EACH_CANDIDATE_SHARE = 0.20


def pool_candidate_trades_c19(rob_path, panel) -> pd.DataFrame:
    """Candidate 19 uses generate_universe_positions(panel, params) -- same bespoke
    pooling pattern as pool_candidate_trades_c12, adapted for candidate 19's module."""
    from tools.t1_harness import _add_atr, _positions_to_trades
    from signal_generators import candidate_19_proximity_high_momentum as c19

    rob = pd.read_csv(rob_path)
    survivors = rob[rob["survives"]]
    slice_start = DEFAULT_IS_START - pd.Timedelta(days=260)
    prepared = {}
    for sym, df in panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= DEFAULT_IS_END)]
        if len(sub) >= 200:
            prepared[sym] = _add_atr(sub)

    all_trades = []
    for _, row in survivors.iterrows():
        params = dict(lookback=int(row["lookback"]), quantile=float(row["quantile"]),
                      rebalance_n=int(row["rebalance_n"]), side=row["side"])
        positions_by_symbol = c19.generate_universe_positions(
            {sym: df[["open", "high", "low", "close", "volume"]] for sym, df in prepared.items()}, params)
        for sym, sub_atr in prepared.items():
            pos = positions_by_symbol.get(sym)
            if pos is None:
                continue
            pos = pos.reindex(sub_atr.index).fillna(0).astype(int)
            tdf = _positions_to_trades(sub_atr, pos, DEFAULT_IS_START, DEFAULT_IS_END)
            if len(tdf):
                tdf = tdf.copy()
                tdf["symbol"] = sym
                all_trades.append(tdf)
    return pd.concat(all_trades, ignore_index=True)


def get_weights_two_candidates(scheme_c, trend, funding_reg, active_systems,
                                candidate_ids: tuple[str, str], each_share: float):
    base = {s: scheme_c[trend].get(s, 0.0) for s in active_systems}
    if funding_reg == "HIGH":
        for s in ("S6", "S7"):
            if s in base:
                base[s] = base[s] + 0.05
        for s in ("S1", "S5"):
            if s in base:
                base[s] = max(0.0, base[s] - 0.05)
    elif funding_reg == "LOW":
        for s in ("S2", "S3"):
            if s in base:
                base[s] = base[s] + 0.05
        for s in ("S6", "S7"):
            if s in base:
                base[s] = max(0.0, base[s] - 0.05)
    total = sum(base.values())
    base = {k: v / total for k, v in base.items()} if total > 0 else {k: 1 / len(base) for k in base}
    remaining = 1 - 2 * each_share
    for k in base:
        base[k] *= remaining
    for cid in candidate_ids:
        base[cid] = each_share
    return base


def replay_two(trades: pd.DataFrame, regime_at, scheme_c, active_systems,
                alloc: dict, static_frac: dict, candidate_ids: tuple[str, str], each_share: float) -> pd.Series:
    t = trades.sort_values("entry_dt").reset_index(drop=True)
    n_years = max((ANALYSIS_END - ANALYSIS_START).days / 365.25, 1e-6)
    trades_per_year = t.groupby("system")["system"].count() / n_years
    risk_pct = {sysname: risk_per_trade_for(freq) for sysname, freq in trades_per_year.items()}
    print(f"    risk-per-trade by system (frequency-scaled): "
          + ", ".join(f"{s}={trades_per_year[s]:.0f}/yr->{risk_pct[s]:.3%}" for s in trades_per_year.index))

    pnls = np.empty(len(t))
    for k, (sysname, day, r) in enumerate(zip(t["system"], t["entry_day"], t["r"])):
        trend, fund_reg, vol_mult = regime_at(day)
        w = get_weights_two_candidates(scheme_c, trend, fund_reg, active_systems, candidate_ids, each_share)
        mult = (w.get(sysname, 0.0) / static_frac[sysname]) * vol_mult
        risk_amount = alloc[sysname] * mult * risk_pct[sysname]
        pnls[k] = risk_amount * r
    t = t.copy()
    t["pnl"] = pnls

    ev = t.sort_values("exit_dt")
    eq_steps = CAPITAL_BASE + ev["pnl"].cumsum()
    eq_by_day = eq_steps.groupby(pd.to_datetime(ev["exit_dt"]).dt.normalize()).last()
    daily_idx = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D")
    daily_eq = eq_by_day.reindex(daily_idx).ffill().fillna(CAPITAL_BASE)
    return daily_eq


def metrics(daily_eq: pd.Series) -> dict:
    rets = daily_eq.pct_change().dropna()
    total_return = daily_eq.iloc[-1] / daily_eq.iloc[0] - 1
    n_years = (daily_eq.index[-1] - daily_eq.index[0]).days / 365.25
    cagr = (daily_eq.iloc[-1] / daily_eq.iloc[0]) ** (1 / n_years) - 1
    pk = daily_eq.cummax()
    max_dd = ((daily_eq - pk) / pk).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else float("nan")
    downside = rets[rets < 0]
    sortino = (rets.mean() / downside.std() * np.sqrt(365)
               if len(downside) > 1 and downside.std() > 0 else float("nan"))
    ret_2022 = (daily_eq[daily_eq.index.year == 2022].iloc[-1]
                / daily_eq[daily_eq.index.year == 2022].iloc[0] - 1) if (daily_eq.index.year == 2022).any() else float("nan")
    return dict(total_return=total_return, cagr=cagr, max_dd=max_dd, sharpe=sharpe,
                sortino=sortino, ret_2022=ret_2022)


def main():
    print(">>> Running step23 baseline engine for real walk-forward regime state...")
    ns = runpy.run_path(str(ROOT / "step23_combined_equity_replay.py"))
    regime_at = ns["regime_at"]
    scheme_c = ns["SCHEME_C"]
    all_trades = ns["all_trades"]

    baseline_trades = all_trades[all_trades["system"].isin(BASELINE_SYSTEMS)].copy()
    baseline_trades = baseline_trades[(baseline_trades["entry_dt"] >= ANALYSIS_START)
                                       & (baseline_trades["entry_dt"] <= ANALYSIS_END)]
    baseline_trades["entry_day"] = baseline_trades["entry_dt"].dt.normalize()
    print(f"Baseline trades (S1/S6/S7/S8): {len(baseline_trades)}")

    static_frac_baseline = {s: BASE_ALLOC[s] / CAPITAL_BASE for s in BASELINE_SYSTEMS}
    print("\n>>> Replaying baseline (S1+S6+S7+S8)...")
    from tools.portfolio_marginal_contribution import replay as replay_one, get_weights_for_active
    eq_baseline = replay_one(baseline_trades, regime_at, scheme_c, BASELINE_SYSTEMS,
                              BASE_ALLOC, static_frac_baseline, candidate_id=None)
    m_base = metrics(eq_baseline)
    print(f"Baseline: total_return={m_base['total_return']:+.1%}  max_dd={m_base['max_dd']:.1%}  "
          f"sharpe={m_base['sharpe']:.2f}  sortino={m_base['sortino']:.2f}")

    print("\n>>> Pooling candidate 12 and 19 trades...")
    panel = load_universe_panel()
    c12_trades = pool_candidate_trades_c12(
        ROOT / "data/research_candidate12_cross_sectional_momentum_t1/t1_robustness.csv", panel)
    c19_trades = pool_candidate_trades_c19(
        ROOT / "data/research_candidate19_t1/t1_robustness.csv", panel)
    print(f"  12: {len(c12_trades)} pooled trades  |  19: {len(c19_trades)} pooled trades")

    def prep(tdf, sysname):
        tdf = tdf.copy()
        tdf["system"] = sysname
        tdf = tdf.rename(columns={"entry_date": "entry_dt", "exit_date": "exit_dt", "net_r": "r"})
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_dt"])
        tdf = tdf[(tdf["entry_dt"] >= ANALYSIS_START) & (tdf["entry_dt"] <= ANALYSIS_END)]
        tdf["entry_day"] = tdf["entry_dt"].dt.normalize()
        return tdf[["system", "entry_dt", "exit_dt", "r", "entry_day"]]

    c12_p = prep(c12_trades, "C12")
    c19_p = prep(c19_trades, "C19")
    combined = pd.concat([baseline_trades, c12_p, c19_p], ignore_index=True)

    alloc = {s: BASE_ALLOC[s] * (1 - 2 * EACH_CANDIDATE_SHARE) for s in BASELINE_SYSTEMS}
    alloc["C12"] = CAPITAL_BASE * EACH_CANDIDATE_SHARE
    alloc["C19"] = CAPITAL_BASE * EACH_CANDIDATE_SHARE
    static_frac = {s: alloc[s] / CAPITAL_BASE for s in alloc}

    print("\n>>> Replaying baseline + 12 + 19 (each 20% share, baseline compressed to 60%)...")
    eq_combined = replay_two(combined, regime_at, scheme_c, BASELINE_SYSTEMS,
                              alloc, static_frac, candidate_ids=("C12", "C19"), each_share=EACH_CANDIDATE_SHARE)
    m_combined = metrics(eq_combined)

    rows = [dict(variant="baseline (S1+S6+S7+S8)", **m_base),
            dict(variant="baseline + 12 + 19 (20% each)", **m_combined)]

    # also replay each single-candidate addition for reference (reuses existing single-candidate infra)
    for cid, tdf in (("12", c12_trades), ("19", c19_trades)):
        cand = prep(tdf, "CANDIDATE")
        combined_single = pd.concat([baseline_trades, cand], ignore_index=True)
        alloc_single = {s: BASE_ALLOC[s] * (1 - EACH_CANDIDATE_SHARE) for s in BASELINE_SYSTEMS}
        alloc_single["CANDIDATE"] = CAPITAL_BASE * EACH_CANDIDATE_SHARE
        static_frac_single = {s: alloc_single[s] / CAPITAL_BASE for s in alloc_single}
        eq_single = replay_one(combined_single, regime_at, scheme_c, BASELINE_SYSTEMS,
                                alloc_single, static_frac_single, candidate_id="CANDIDATE")
        rows.append(dict(variant=f"baseline + {cid} only (20%)", **metrics(eq_single)))

    summary = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "research_portfolio_marginal_contribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "combined_12_19_summary.csv", index=False)

    print(f"\n{'='*100}\nCOMBINED-ADDITION SUMMARY (2020-2024, $100k base)\n{'='*100}")
    print(f"{'Variant':<32}{'Total Ret':>12}{'CAGR':>10}{'MaxDD':>10}{'Sharpe':>9}{'Sortino':>9}")
    for _, r in summary.iterrows():
        print(f"{r['variant']:<32}{r['total_return']:>+11.1%}{r['cagr']:>+9.1%}"
              f"{r['max_dd']:>9.1%}{r['sharpe']:>9.2f}{r['sortino']:>9.2f}")


if __name__ == "__main__":
    main()
