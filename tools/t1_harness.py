#!/usr/bin/env python3
"""
Standardized T1 test harness for trend candidates -- one parameterized harness
instead of a bespoke T1 script per candidate.

Shared T1 convention (approved 2026-07-27, derived from the most common pattern
across existing S1-S8 T1 scripts -- see phase_t1_dualma_btc_concept_discovery.py
and phase_t1_liqcascade_short_mr_concept_discovery.py):
  - R-unit:        ATR(14) x 2.0 from entry price
  - Cost floor:    avg_r > 0.15R (spot-eligible) or > 0.25R (futures-only)
  - IS window:     2020-01-01 -> 2024-12-31 (extended 2026-07-27 from the original
                    2020-2022; the 3-year window was found to let a single dominant
                    year -- whichever matched a candidate's directional bias -- carry
                    65-100%+ of total R, a pattern confirmed present in the live
                    systems S1/S6 as well as new candidates 02u/03u/06. Forward
                    extension to 2024 is fully supported by existing data (all 290
                    futures / 418 spot universe symbols have clean, continuous data
                    through mid-2026); backward extension before 2020 is NOT used --
                    only ~3/290 futures and ~53/418 spot symbols have any pre-2020
                    data, too sparse for a universe-wide test. Overridable per
                    candidate when data availability forces a shorter window -- see
                    window_adjusted handling below.)
  - Stability gate: neighbourhood zone (+-1 grid step per param) >= 67% profitable
  - Robustness:    drop top-2 trades by net_r, re-verify avg_r > floor and
                    t_stat >= 1.5 (triggered whenever a combo trades <30/yr)
  - Year concentration: no single calendar year may contribute more than
                    YEAR_CONCENTRATION_MAX_SHARE of total R across gate-passing
                    combos (see check_year_concentration below) -- added 2026-07-27
                    after the binary "trades net positive in >=2 of 3 years" gate
                    proved too weak: S1 and S6's live frozen backtests both clear
                    that binary bar while still deriving 83-103% of total R from a
                    single year, which the binary check can't catch but a magnitude
                    check does.

Signal generator interface (signal_generators/candidate_XX_<name>.py):
    def generate_signal(price_data: pd.DataFrame, params: dict) -> pd.Series:
        '''Returns position series: 1 (long), -1 (short), 0 (flat), indexed to price_data.'''
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BATCH_TRACKER = ROOT / "results" / "candidate_batch_tracker.csv"

ATR_MULT_R = 2.0
ATR_WINDOW = 14
COST_FLOOR_SPOT = 0.15
COST_FLOOR_FUTURES = 0.25
STABILITY_ZONE_THRESHOLD = 0.67
ROBUSTNESS_MIN_TRADES_PER_YEAR = 30
ROBUSTNESS_T_FLOOR = 1.5

DEFAULT_IS_START = pd.Timestamp("2020-01-01")
DEFAULT_IS_END = pd.Timestamp("2024-12-31")

YEAR_CONCENTRATION_MAX_SHARE = 0.65  # no single year may exceed this share of total R


def check_window_adjusted(actual_start: pd.Timestamp, actual_end: pd.Timestamp) -> bool:
    return actual_start != DEFAULT_IS_START or actual_end != DEFAULT_IS_END


def _add_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / window, adjust=False).mean()
    return df


def _positions_to_trades(df: pd.DataFrame, positions: pd.Series,
                          is_start: pd.Timestamp, is_end: pd.Timestamp) -> pd.DataFrame:
    """Convert a position series (1/-1/0) into discrete trades. Entries/exits fill at the
    NEXT bar's open (matches existing T1 convention). R-unit fixed at entry using
    ATR(14) x 2.0. Trades gated on ENTRY date falling within [is_start, is_end] so no
    OOS-entered trade contaminates IS stats."""
    positions = positions.reindex(df.index).fillna(0).astype(int)
    trades = []
    pos = 0
    entry_px = entry_r_unit = entry_date = None
    n = len(df)

    for i in range(ATR_WINDOW, n - 1):
        target = positions.iloc[i]
        nxt_open = df["open"].iloc[i + 1]
        nxt_date = df.index[i + 1]

        if pos != 0 and target != pos:
            pnl_pct = (nxt_open - entry_px) / entry_px * pos
            trades.append((entry_date, nxt_date, "long" if pos == 1 else "short",
                            entry_px, nxt_open, pnl_pct / entry_r_unit))
            pos = 0

        if pos == 0 and target != 0:
            atr = df["atr"].iloc[i]
            if np.isnan(atr) or atr <= 0:
                continue
            if is_start <= nxt_date <= is_end:
                pos, entry_px, entry_date = target, nxt_open, nxt_date
                entry_r_unit = ATR_MULT_R * atr / nxt_open

    if pos != 0:
        is_df = df[df.index <= is_end]
        last_px = is_df["close"].iloc[-1]
        last_dt = is_df.index[-1]
        pnl_pct = (last_px - entry_px) / entry_px * pos
        trades.append((entry_date, last_dt, "long" if pos == 1 else "short",
                        entry_px, last_px, pnl_pct / entry_r_unit))

    tdf = pd.DataFrame(trades, columns=["entry_date", "exit_date", "side",
                                         "entry_px", "exit_px", "net_r"])
    return tdf[(tdf["entry_date"] >= is_start) & (tdf["entry_date"] <= is_end)].reset_index(drop=True)


def _combo_stats(tdf: pd.DataFrame, is_years: float) -> dict:
    if len(tdf) == 0:
        return dict(n=0, avg_r=np.nan, total_r=0.0, win_rate=np.nan, pf=np.nan,
                    t_stat=np.nan, trades_per_year=0.0)
    r = tdf["net_r"]
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    return dict(n=len(tdf), avg_r=r.mean(), total_r=r.sum(), win_rate=(r > 0).mean(),
                pf=wins / losses if losses > 0 else np.inf, t_stat=t,
                trades_per_year=len(tdf) / is_years)


def _drop_top2(tdf: pd.DataFrame, cost_floor: float, robustness_triggered: bool) -> dict:
    """Drop the top-2 trades by net_r and re-check. avg_r_ex > cost_floor is ALWAYS
    required, regardless of trade frequency -- the repo's "MANDATORY below 30 trades/yr"
    convention (see phase_t1_dualma_btc_concept_discovery.py Finding #28) only means the
    check must be examined/reported at low sample sizes, not that the return floor is
    waived above 30 trades/yr. What genuinely depends on sample size is the t-stat's
    reliability, so ONLY the t_ex >= ROBUSTNESS_T_FLOOR requirement is conditional on
    robustness_triggered. (Earlier version incorrectly exempted low-frequency combos
    from the floor check entirely whenever trades/yr>=30, which is what produced
    candidate 8's false PASS: avg_r_ex=0.233 < 0.25R floor was overlooked because its
    409 trades/136-per-year exempted it from the check being "mandatory".)"""
    if len(tdf) <= 2:
        return dict(n_ex=max(0, len(tdf) - 2), avg_r_ex=np.nan, t_ex=np.nan, survives=False)
    r = tdf["net_r"].sort_values(ascending=False).iloc[2:]
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    clears_floor = r.mean() > cost_floor
    clears_tstat = (not np.isnan(t)) and t >= ROBUSTNESS_T_FLOOR
    survives = clears_floor and (clears_tstat if robustness_triggered else True)
    return dict(n_ex=len(r), avg_r_ex=r.mean(), t_ex=t, clears_floor=clears_floor,
                clears_tstat=clears_tstat, survives=survives)


def _is_ordinal_param(values: list) -> bool:
    """True for numeric params where a +-1 grid-step neighbour is a meaningful concept
    (e.g. fast=20 -> fast=30). False for categorical toggles (side, use_regime_filter,
    method, ...) where stepping the value is a different strategy, not a nearby one."""
    return all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
               for v in values)


def _zone_frac(grid: pd.DataFrame, row: pd.Series, param_names: list[str],
               param_grids: dict[str, list]) -> float:
    """Fraction of a combo's +-1-grid-step neighbourhood that is also profitable.
    Only numeric/ordinal params are stepped; categorical params (side, boolean toggles,
    etc) are held fixed at the row's own value -- matching the existing repo convention
    (e.g. phase_t1_dualma_btc_concept_discovery.py holds `side` fixed and steps
    fast/slow only). Stepping a categorical dimension would compare genuinely different
    strategies, not "nearby" parameterizations, and silently corrupts the stability gate."""
    ordinal = [p for p in param_names if _is_ordinal_param(param_grids[p])]
    categorical = [p for p in param_names if p not in ordinal]
    if not ordinal:
        return np.nan

    idx = {p: param_grids[p].index(row[p]) for p in ordinal}
    cat_mask = np.ones(len(grid), dtype=bool)
    for p in categorical:
        cat_mask &= (grid[p] == row[p])

    members = []
    for offsets in itertools.product((-1, 0, 1), repeat=len(ordinal)):
        if all(o == 0 for o in offsets):
            continue
        neighbour_idx = {p: idx[p] + o for p, o in zip(ordinal, offsets)}
        if any(not (0 <= neighbour_idx[p] < len(param_grids[p])) for p in ordinal):
            continue
        neighbour_vals = {p: param_grids[p][neighbour_idx[p]] for p in ordinal}
        mask = cat_mask.copy()
        for p, v in neighbour_vals.items():
            mask &= (grid[p] == v)
        m = grid[mask]
        if len(m):
            members.append(bool(m["profitable"].iloc[0]))
    return np.mean(members) if members else np.nan


EDGE_CLUSTERING_RATE_MARGIN = 0.20  # edge survival rate must exceed interior rate by this much


def check_edge_clustering(grid: pd.DataFrame, robustness: pd.DataFrame,
                           param_grids: dict[str, list]) -> dict:
    """Flags whether combos surviving drop-top-2 cluster at the MIN/MAX of a tested
    ordinal param's grid, rather than spreading across the interior. A +-1 stability zone
    cannot distinguish "real neighbourhood optimum" from "the edge of a grid that keeps
    getting better/worse the further you push it" -- both look like a stable zone_frac,
    since the zone check only ever compares values that were actually tested. Edge
    clustering means the candidate wasn't really tested past that parameter's current
    range and shouldn't be read as a confirmed interior optimum (see candidate 6 T1:
    9/11 survivors sat at senkou_b_n=60, the max grid value, with survivor count rising
    monotonically toward that edge -- 0 at 44, 2 at 52, 9 at 60).

    Compares SURVIVAL RATE (survivors / combos tested) at edge values vs interior values,
    not raw survivor counts -- comparing counts is biased against small (e.g. 3-value)
    grids, where a uniform 100%-survival pass across the whole grid trivially puts 2/3 of
    survivors at the two edge values by construction (2 edges vs 1 interior bucket), which
    would incorrectly flag the STRONGEST possible robustness result (everything passes
    equally) as if it were curve-fitting (found via candidate 03u: 108/108 combos survived
    uniformly, tripping the old count-based check purely from grid arithmetic). Rate-based
    comparison correctly reads that case as no differential clustering at all."""
    if len(robustness) == 0:
        return {}
    survivors = robustness[robustness["survives"]]
    if len(survivors) == 0:
        return {}

    flags = {}
    for p, grid_vals in param_grids.items():
        if not _is_ordinal_param(grid_vals) or p not in survivors.columns:
            continue
        sorted_vals = sorted(grid_vals)
        if len(sorted_vals) < 3:
            continue

        tested_counts = grid[p].value_counts()
        survivor_counts = survivors[p].value_counts()
        rate = {v: survivor_counts.get(v, 0) / tested_counts.get(v, 1) for v in sorted_vals
                if tested_counts.get(v, 0) > 0}
        if len(rate) < 3:
            continue

        edge_rate = (rate.get(sorted_vals[0], 0) + rate.get(sorted_vals[-1], 0)) / 2
        interior_vals = sorted_vals[1:-1]
        interior_rate = sum(rate.get(v, 0) for v in interior_vals) / len(interior_vals)

        if edge_rate - interior_rate > EDGE_CLUSTERING_RATE_MARGIN:
            n_at_min = int(survivor_counts.get(sorted_vals[0], 0))
            n_at_max = int(survivor_counts.get(sorted_vals[-1], 0))
            n_interior = len(survivors) - n_at_min - n_at_max
            flags[p] = dict(n_at_min=n_at_min, n_at_max=n_at_max, n_interior=n_interior,
                             edge_rate=round(edge_rate, 3), interior_rate=round(interior_rate, 3),
                             edge=sorted_vals[0] if rate.get(sorted_vals[0], 0) >= rate.get(sorted_vals[-1], 0)
                             else sorted_vals[-1])
    return flags


THIN_SLIVER_MAX_FRACTION = 0.15  # gate-passing combos below this fraction of the grid are flagged


def check_thin_sliver(grid: pd.DataFrame) -> dict:
    """Flags when the primary-gate-passing set is a small fraction of the whole grid
    tested -- a separate concern from check_edge_clustering (which only looks at where
    SURVIVORS of drop-top-2 sit, given at least one gate-passer exists). This catches
    the case where almost nothing passes the primary gate at all, e.g. candidate 8: only
    1/12 combos cleared avg_r>0.25R, by a 0.005R margin, with no neighbouring combo also
    passing -- a single fragile cell, not a real signal, regardless of what its
    drop-top-2 numbers later show. Do not read a PASS as validated when this fires."""
    total = len(grid)
    if total == 0:
        return {}
    n_pass = int(grid["all_gates"].sum())
    if n_pass == 0:
        return {}
    fraction = n_pass / total
    if fraction <= THIN_SLIVER_MAX_FRACTION:
        return dict(n_pass=n_pass, n_total=total, fraction=fraction)
    return {}


YEAR_CONCENTRATION_SENSITIVITY_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75)


def check_year_concentration(grid: pd.DataFrame, trade_store: dict, param_names: list[str],
                              max_share: float = YEAR_CONCENTRATION_MAX_SHARE) -> dict:
    """Magnitude-based concentration gate: pools trades from every primary-gate-passing
    combo and checks whether any single calendar year contributes more than max_share of
    total R. Added 2026-07-27 to replace manual yearly-breakdown inspection and the binary
    "trades net positive in >=2 of 3 years" gate, which proved too weak -- both S1 and S6's
    live frozen backtests clear a >=2/3-years-positive bar while still deriving 83-103% of
    total R from one year (S1: 2021; S6: 2022), which only a magnitude check catches.

    Returns {} if there are no gate-passing combos (nothing to assess) or fewer than 2
    distinct years of trades (concentration is undefined with only one year of data).
    Otherwise returns the yearly breakdown, the dominant year/share, a pass/fail against
    max_share, AND a sensitivity table across YEAR_CONCENTRATION_SENSITIVITY_THRESHOLDS so
    the threshold choice itself is auditable rather than a silent constant."""
    passing = grid[grid["all_gates"]]
    if len(passing) == 0:
        return {}

    pooled = []
    for _, row in passing.iterrows():
        combo_key = tuple(row[p] for p in param_names)
        tdf = trade_store.get(combo_key)
        if tdf is not None and len(tdf):
            pooled.append(tdf)
    if not pooled:
        return {}
    trades = pd.concat(pooled, ignore_index=True)

    years = pd.to_datetime(trades["entry_date"]).dt.year
    yearly = trades.groupby(years)["net_r"].sum()
    if len(yearly) < 2:
        return {}

    total = yearly.sum()
    shares = (yearly / total) if total != 0 else yearly * np.nan
    dominant_year = int(shares.idxmax())
    dominant_share = float(shares.max())

    sensitivity = {t: bool(dominant_share <= t) for t in YEAR_CONCENTRATION_SENSITIVITY_THRESHOLDS}

    return dict(
        yearly_r=yearly.to_dict(),
        yearly_share=shares.round(4).to_dict(),
        dominant_year=dominant_year,
        dominant_share=round(dominant_share, 4),
        max_share_threshold=max_share,
        passes=bool(dominant_share <= max_share),
        sensitivity=sensitivity,
    )


MONTH_CONCENTRATION_MAX_SHARE = 0.25  # no single calendar month may exceed this share of A COMBO'S OWN total R
ASSET_CONCENTRATION_MAX_SHARE = 0.20  # no single asset may exceed this share of A COMBO'S OWN total R (universe variants only)
MONTH_CONCENTRATION_SENSITIVITY_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
ASSET_CONCENTRATION_SENSITIVITY_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)


def _month_concentration(tdf: pd.DataFrame) -> dict:
    """Dominant calendar month's share of ONE COMBO's total R. Returns {} if fewer than 2
    distinct months have trades (concentration undefined with only one month) or no trades."""
    if len(tdf) == 0:
        return {}
    months = pd.to_datetime(tdf["entry_date"]).dt.to_period("M")
    monthly = tdf.groupby(months)["net_r"].sum()
    if len(monthly) < 2:
        return {}
    total = monthly.sum()
    if total == 0:
        return {}
    shares = monthly / total
    return dict(dominant_month=str(shares.idxmax()), dominant_share=float(shares.max()))


def _asset_concentration(tdf: pd.DataFrame) -> dict:
    """Dominant symbol's share of ONE COMBO's total R. Only meaningful for universe/
    cross-sectional combos (tdf has a 'symbol' column spanning >=2 assets) -- returns {}
    for single-asset combos, where concentration is 100% by construction and not a
    meaningful signal, and would otherwise gate out every single-asset candidate."""
    if len(tdf) == 0 or "symbol" not in tdf.columns:
        return {}
    by_asset = tdf.groupby("symbol")["net_r"].sum()
    if len(by_asset) < 2:
        return {}
    total = by_asset.sum()
    if total == 0:
        return {}
    shares = by_asset / total
    return dict(dominant_asset=str(shares.idxmax()), dominant_share=float(shares.max()))


def check_combo_diversification(tdf: pd.DataFrame, max_month_share: float | None,
                                 max_asset_share: float | None) -> dict:
    """Per-combo diversification gate -- added 2026-08-08 after 11u and 13u both passed
    T1/T2/T3/T4 (T4's remove-best-month diagnostic flagged but did not gate: 24.0% and
    39.7% of total R from the SAME single month, 2021-05) only to fail T7 strict
    robustness once the trade set was realistically capped to $60k/20-position sizing --
    a handful of standout trades represented too large a share of the much smaller
    executable set to survive removing top assets/months.

    Unlike check_year_concentration (which pools ALL gate-passing combos into one
    aggregate diagnostic, reported in the tracker as a caution flag but never excluding
    any combo from passing), this evaluates EACH combo's OWN trade set individually and
    is folded directly into all_gates -- a combo whose return is dominated by one month or
    one asset is excluded from T1 at the same stage as the avg_r/stability gates, not
    merely flagged for later discovery at T7.

    Opt-in via max_month_share/max_asset_share (None = disabled, the default -- existing
    candidates' T1 results are unaffected unless explicitly re-run with this gate active)."""
    month = _month_concentration(tdf)
    asset = _asset_concentration(tdf)
    month_ok = max_month_share is None or not month or month["dominant_share"] <= max_month_share
    asset_ok = max_asset_share is None or not asset or asset["dominant_share"] <= max_asset_share
    return dict(month=month, asset=asset, month_ok=month_ok, asset_ok=asset_ok,
                passes=bool(month_ok and asset_ok))


@dataclass
class T1Result:
    candidate_id: str
    candidate_name: str
    grid: pd.DataFrame
    robustness: pd.DataFrame
    actual_is_start: pd.Timestamp
    actual_is_end: pd.Timestamp
    window_adjusted: bool
    cost_floor: float
    n_pass: int = field(init=False)
    edge_clustering: dict = field(init=False)
    thin_sliver: dict = field(init=False)
    year_concentration: dict = field(init=False)

    def __post_init__(self):
        self.n_pass = int(self.grid["all_gates"].sum()) if len(self.grid) else 0
        self.edge_clustering = {}
        self.thin_sliver = {}
        self.year_concentration = {}


def run_t1(
    candidate_id: str,
    candidate_name: str,
    generate_signal: Callable[[pd.DataFrame, dict], pd.Series],
    price_data: pd.DataFrame,
    param_grids: dict[str, list],
    asset_class: str = "futures",
    is_start: pd.Timestamp | None = None,
    is_end: pd.Timestamp | None = None,
    out_dir: Path | None = None,
    max_month_share: float | None = None,
    max_asset_share: float | None = None,
) -> T1Result:
    """Run one candidate's signal generator across a param grid through the shared T1 gates.

    price_data: must have a DatetimeIndex and columns open/high/low/close.
    param_grids: e.g. {"fast": [10,20,30], "slow": [50,100,150]} -- Cartesian product tested.
    asset_class: 'spot' -> 0.15R floor, 'futures' -> 0.25R floor.
    is_start/is_end: override the default 2020-01-01..2022-12-31 window when the candidate's
        data doesn't cover it (funding history, newer listings, etc). Any override is recorded
        as window_adjusted=True in the batch tracker rather than silently compared as-equivalent
        to a full-window run.
    max_month_share/max_asset_share: optional per-combo diversification gate (see
        check_combo_diversification) -- None (default) disables it, unchanged behavior.
    """
    cost_floor = COST_FLOOR_SPOT if asset_class == "spot" else COST_FLOOR_FUTURES

    requested_start = is_start or DEFAULT_IS_START
    requested_end = is_end or DEFAULT_IS_END
    data_start, data_end = price_data.index.min(), price_data.index.max()
    actual_start = max(requested_start, data_start)
    actual_end = min(requested_end, data_end)
    window_adjusted = check_window_adjusted(actual_start, actual_end)

    df = _add_atr(price_data)
    is_years = max((actual_end - actual_start).days / 365.25, 1e-6)

    param_names = list(param_grids.keys())
    rows, trade_store = [], {}
    for combo in itertools.product(*param_grids.values()):
        params = dict(zip(param_names, combo))
        positions = generate_signal(df, params)
        tdf = _positions_to_trades(df, positions, actual_start, actual_end)
        st = _combo_stats(tdf, is_years)
        div = check_combo_diversification(tdf, max_month_share, max_asset_share)
        st["dominant_month_share"] = div["month"].get("dominant_share", np.nan)
        st["dominant_asset_share"] = div["asset"].get("dominant_share", np.nan)
        st["gate_diversification"] = div["passes"]
        st.update(params)
        rows.append(st)
        trade_store[tuple(combo)] = tdf

    return _gate_and_finish(candidate_id, candidate_name, rows, trade_store, param_names,
                             param_grids, cost_floor, actual_start, actual_end,
                             window_adjusted, out_dir, update_tracker=True)


def run_t1_universe(
    candidate_id: str,
    candidate_name: str,
    generate_signal: Callable[[pd.DataFrame, dict], pd.Series],
    price_panel: dict[str, pd.DataFrame],
    param_grids: dict[str, list],
    asset_class: str = "futures",
    is_start: pd.Timestamp | None = None,
    is_end: pd.Timestamp | None = None,
    out_dir: Path | None = None,
    warmup_days: int = 260,
    min_symbol_bars: int = 200,
    update_tracker: bool = False,
    max_month_share: float | None = None,
    max_asset_share: float | None = None,
) -> T1Result:
    """Cross-sectional variant of run_t1: apply generate_signal independently to every
    symbol in price_panel, then pool all resulting trades across the whole universe into
    one grid/gate/robustness evaluation. Use this for time-series-momentum-style candidates
    where diversification across many assets (not a single BTC test) is the actual claim
    being tested -- concentration in 1-2 trades on a single asset is expected to matter less,
    or not survive, once pooled across the full universe.

    update_tracker defaults to False: a universe retest is a distinct T1 result from any
    prior single-asset run for the same candidate_id and should not silently overwrite it
    in results/candidate_batch_tracker.csv. Callers wanting the tracker updated (e.g. under
    a distinct candidate_id like "02u") should pass update_tracker=True explicitly.

    max_month_share/max_asset_share: optional per-combo diversification gate (see
        check_combo_diversification) -- None (default) disables it, unchanged behavior.
    """
    cost_floor = COST_FLOOR_SPOT if asset_class == "spot" else COST_FLOOR_FUTURES
    actual_start = is_start or DEFAULT_IS_START
    actual_end = is_end or DEFAULT_IS_END
    window_adjusted = check_window_adjusted(actual_start, actual_end)
    is_years = max((actual_end - actual_start).days / 365.25, 1e-6)

    slice_start = actual_start - pd.Timedelta(days=warmup_days)
    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in price_panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= actual_end)]
        if len(sub) < min_symbol_bars:
            continue
        prepared[sym] = _add_atr(sub)

    param_names = list(param_grids.keys())
    rows, trade_store = [], {}
    for combo in itertools.product(*param_grids.values()):
        params = dict(zip(param_names, combo))
        combo_trades = []
        for sym, df in prepared.items():
            positions = generate_signal(df, params)
            tdf = _positions_to_trades(df, positions, actual_start, actual_end)
            if len(tdf):
                tdf = tdf.copy()
                tdf["symbol"] = sym
                combo_trades.append(tdf)
        pooled = (pd.concat(combo_trades, ignore_index=True) if combo_trades
                  else pd.DataFrame(columns=["entry_date", "exit_date", "side", "entry_px",
                                              "exit_px", "net_r", "symbol"]))
        st = _combo_stats(pooled, is_years)
        st["n_symbols_traded"] = pooled["symbol"].nunique() if len(pooled) else 0
        div = check_combo_diversification(pooled, max_month_share, max_asset_share)
        st["dominant_month_share"] = div["month"].get("dominant_share", np.nan)
        st["dominant_asset_share"] = div["asset"].get("dominant_share", np.nan)
        st["gate_diversification"] = div["passes"]
        st.update(params)
        rows.append(st)
        trade_store[tuple(combo)] = pooled

    return _gate_and_finish(candidate_id, candidate_name, rows, trade_store, param_names,
                             param_grids, cost_floor, actual_start, actual_end,
                             window_adjusted, out_dir, update_tracker=update_tracker)


def run_t1_universe_ranked(
    candidate_id: str,
    candidate_name: str,
    generate_universe_positions: Callable[[dict[str, pd.DataFrame], dict], dict[str, pd.Series]],
    price_panel: dict[str, pd.DataFrame],
    param_grids: dict[str, list],
    asset_class: str = "futures",
    is_start: pd.Timestamp | None = None,
    is_end: pd.Timestamp | None = None,
    out_dir: Path | None = None,
    warmup_days: int = 260,
    min_symbol_bars: int = 200,
    update_tracker: bool = False,
    max_month_share: float | None = None,
    max_asset_share: float | None = None,
) -> T1Result:
    """Cross-sectional variant of run_t1 for candidates whose signal genuinely requires
    the whole universe simultaneously (ranking/relative-strength constructions using the
    generate_universe_positions(panel, params) -> {symbol: position_series} interface,
    e.g. candidate 12 and 18/19/20) -- distinct from run_t1_universe's per-symbol-
    independent generate_signal(df, params) loop, since ranking needs every symbol's
    data at once to compute cross-sectional order, not just a per-symbol loop that
    happens to run for many symbols.

    generate_universe_positions is called ONCE per combo on the FULL, un-sliced panel
    (so ranking reflects the true tradeable universe at each historical date), then each
    symbol's resulting position series is applied to that symbol's own IS-window/ATR-
    prepared price data for trade conversion -- matching the pattern originally hand-
    rolled in tools/run_candidate12.py's T1 driver, promoted here to the shared harness
    now that three more candidates (18/19/20) use the identical interface.

    max_month_share/max_asset_share: optional per-combo diversification gate (see
        check_combo_diversification) -- None (default) disables it, unchanged behavior.
    """
    cost_floor = COST_FLOOR_SPOT if asset_class == "spot" else COST_FLOOR_FUTURES
    actual_start = is_start or DEFAULT_IS_START
    actual_end = is_end or DEFAULT_IS_END
    window_adjusted = check_window_adjusted(actual_start, actual_end)
    is_years = max((actual_end - actual_start).days / 365.25, 1e-6)

    slice_start = actual_start - pd.Timedelta(days=warmup_days)
    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in price_panel.items():
        sub = df[(df.index >= slice_start) & (df.index <= actual_end)]
        if len(sub) < min_symbol_bars:
            continue
        prepared[sym] = _add_atr(sub)

    param_names = list(param_grids.keys())
    rows, trade_store = [], {}
    for combo in itertools.product(*param_grids.values()):
        params = dict(zip(param_names, combo))
        positions_by_symbol = generate_universe_positions(price_panel, params)

        combo_trades = []
        for sym, sub_atr in prepared.items():
            pos = positions_by_symbol.get(sym)
            if pos is None:
                continue
            pos = pos.reindex(sub_atr.index).fillna(0).astype(int)
            tdf = _positions_to_trades(sub_atr, pos, actual_start, actual_end)
            if len(tdf):
                tdf = tdf.copy()
                tdf["symbol"] = sym
                combo_trades.append(tdf)
        pooled = (pd.concat(combo_trades, ignore_index=True) if combo_trades
                  else pd.DataFrame(columns=["entry_date", "exit_date", "side", "entry_px",
                                              "exit_px", "net_r", "symbol"]))
        st = _combo_stats(pooled, is_years)
        st["n_symbols_traded"] = pooled["symbol"].nunique() if len(pooled) else 0
        div = check_combo_diversification(pooled, max_month_share, max_asset_share)
        st["dominant_month_share"] = div["month"].get("dominant_share", np.nan)
        st["dominant_asset_share"] = div["asset"].get("dominant_share", np.nan)
        st["gate_diversification"] = div["passes"]
        st.update(params)
        rows.append(st)
        trade_store[tuple(combo)] = pooled

    return _gate_and_finish(candidate_id, candidate_name, rows, trade_store, param_names,
                             param_grids, cost_floor, actual_start, actual_end,
                             window_adjusted, out_dir, update_tracker=update_tracker)


def _gate_and_finish(candidate_id, candidate_name, rows, trade_store, param_names,
                      param_grids, cost_floor, actual_start, actual_end, window_adjusted,
                      out_dir, update_tracker: bool) -> T1Result:
    grid = pd.DataFrame(rows)
    grid["profitable"] = grid["avg_r"] > 0
    grid["gate_avgr"] = grid["avg_r"] > cost_floor
    grid["zone_frac"] = grid.apply(lambda r: _zone_frac(grid, r, param_names, param_grids), axis=1)
    grid["gate_stability"] = grid["zone_frac"] >= STABILITY_ZONE_THRESHOLD
    if "gate_diversification" not in grid.columns:
        grid["gate_diversification"] = True
    grid["all_gates"] = grid["gate_avgr"] & grid["gate_stability"] & grid["gate_diversification"]

    robustness_rows = []
    for _, row in grid[grid["all_gates"]].iterrows():
        combo_key = tuple(row[p] for p in param_names)
        tdf = trade_store[combo_key]
        triggered = row["trades_per_year"] < ROBUSTNESS_MIN_TRADES_PER_YEAR
        d = _drop_top2(tdf, cost_floor, robustness_triggered=triggered)
        d.update({p: row[p] for p in param_names})
        d["avg_r_full"], d["t_full"] = row["avg_r"], row["t_stat"]
        d["robustness_triggered"] = triggered
        robustness_rows.append(d)
    robustness = pd.DataFrame(robustness_rows)

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grid.sort_values("avg_r", ascending=False).to_csv(out_dir / "t1_grid.csv", index=False)
        if len(robustness):
            robustness.to_csv(out_dir / "t1_robustness.csv", index=False)

    result = T1Result(candidate_id, candidate_name, grid, robustness,
                       actual_start, actual_end, window_adjusted, cost_floor)
    result.edge_clustering = check_edge_clustering(grid, robustness, param_grids)
    result.thin_sliver = check_thin_sliver(grid)
    result.year_concentration = check_year_concentration(grid, trade_store, param_names)
    if update_tracker:
        _update_batch_tracker(result)
    return result


def _update_batch_tracker(result: T1Result) -> None:
    if not BATCH_TRACKER.exists():
        return
    tracker = pd.read_csv(BATCH_TRACKER, dtype={"candidate_id": str})
    for col in ("t1_status", "current_stage", "window_adjusted",
                "actual_is_start", "actual_is_end", "last_updated"):
        tracker[col] = tracker[col].astype(object)
    mask = tracker["candidate_id"] == result.candidate_id
    if not mask.any():
        return
    passed = result.n_pass > 0
    if not passed:
        survives = False
    elif len(result.robustness) == 0:
        survives = True
    else:
        # _drop_top2 already encodes the full rule (avg_r_ex must always clear the
        # floor; the t-stat requirement is only mandatory when robustness_triggered) --
        # "survives" is authoritative on its own, no OR-with-not-triggered needed.
        survives = bool(result.robustness["survives"].any())

    year_conc_fail = bool(result.year_concentration) and not result.year_concentration.get("passes", True)
    caution = bool(result.edge_clustering) or bool(result.thin_sliver) or year_conc_fail
    stage = "T1_FAILED"
    if survives:
        stage = "T1_PASS_HOLD_NOT_ADVANCED" if caution else "T2"

    tracker.loc[mask, "t1_status"] = "PASS" if survives else "FAIL"
    tracker.loc[mask, "current_stage"] = stage
    tracker.loc[mask, "window_adjusted"] = result.window_adjusted
    tracker.loc[mask, "actual_is_start"] = result.actual_is_start.date().isoformat()
    tracker.loc[mask, "actual_is_end"] = result.actual_is_end.date().isoformat()
    tracker.loc[mask, "last_updated"] = pd.Timestamp.today().date().isoformat()
    if survives and caution:
        existing_notes = tracker.loc[mask, "notes"].iloc[0]
        auto_caution = []
        if result.thin_sliver:
            ts = result.thin_sliver
            auto_caution.append(f"AUTO-FLAG thin_sliver: only {ts['n_pass']}/{ts['n_total']} "
                                 f"combos ({ts['fraction']:.1%}) clear the primary gate.")
        if result.edge_clustering:
            auto_caution.append(f"AUTO-FLAG edge_clustering: {result.edge_clustering}")
        if year_conc_fail:
            yc = result.year_concentration
            auto_caution.append(
                f"AUTO-FLAG year_concentration: {yc['dominant_share']:.1%} of total R from "
                f"{yc['dominant_year']} alone, exceeds {yc['max_share_threshold']:.0%} threshold "
                f"(sensitivity: {yc['sensitivity']})."
            )
        tracker.loc[mask, "notes"] = f"{existing_notes}. {' '.join(auto_caution)}".strip(". ")
    tracker.to_csv(BATCH_TRACKER, index=False)
