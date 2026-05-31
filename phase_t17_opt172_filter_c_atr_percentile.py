#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.2 -- Regime Filter C: ATR Percentile Gate

Filter applied at entry: ATR(14) on the entry bar must be between the 40th and
80th percentile of the trailing 252-bar ATR(14) distribution.

Rationale:
  Low-ATR regime  (<40th pct) → price too quiet; breakouts tend to be false.
  High-ATR regime (>80th pct) → market too noisy/volatile; stop-hunts dominate.
  Mid-ATR regime  (40–80 pct) → healthy trending volatility.

Universe   : 24 symbols (filtered_symbols_v2_included_only.csv)
Base config: Donchian N=20 / ema200_price / ATR(14)x2.0 stop /
             Chandelier ACT+4R trail 3xATR / LONG only / 1D

Tests
  Stability zone : N=[15, 20, 25] -- all must remain profitable
  T4 robustness  : MC (2000 runs), cost stress, period splits,
                   remove-best-asset, remove-best-month
  Trade guard    : N=20 filtered trades must be >= 80
  Portfolio/CAGR : T5 max8 + T6 capital execution

Baseline (DonchianLong_UniverseV2, N=20, no regime filter):
  avg_r=+1.101R  PF=3.072  CAGR=+14.3% (max8, T6)

Output: data/research_donchian_regimeV2_atr_percentile/
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT       = Path(__file__).parent
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OUT_DIR    = ROOT / "data" / "research_donchian_regimeV2_atr_percentile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# Frozen Donchian config
ATR_N      = 14
STOP_MULT  = 2.0
CHAN_ACT_R = 4.0
CHAN_TRAIL = 3.0

# Filter C: ATR percentile gate
ATR_PCT_WINDOW = 252   # trailing bars for percentile distribution
ATR_PCT_LOW    = 0.40  # reject below this percentile (too quiet)
ATR_PCT_HIGH   = 0.80  # reject above this percentile (too noisy)

# Stability zone entry-N values (exit_n = entry_n // 2)
STABILITY_N = [15, 20, 25]
CANONICAL_N = 20

# T4 robustness
MC_RUNS     = 2000
MC_BLOCKS   = [1, 3, 5, 10, 20]
MC_SEED     = 42
EXTRA_COSTS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# T6 capital
START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35
PORT_MAX_OPEN  = 8

# Baseline reference (DonchianLong_UniverseV2, N=20, no filter)
BASELINE = dict(avg_r=1.1011, pf=3.072, cagr_pct=14.3, trades=461,
                label="DonchianLong_UniverseV2 N=20 no-filter")

MIN_TRADES = 80


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(close)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1 : n + 1]))
        for i in range(n + 1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i - 1]):
                atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _atr_pct_rank(atr: np.ndarray, window: int = ATR_PCT_WINDOW) -> np.ndarray:
    """
    For each bar i, compute what percentile atr[i] sits at within
    the prior `window` bars (atr[i-window : i]).  No lookahead.

    Returns array of floats in [0, 1]; NaN where insufficient history.
    rank = fraction of prior-window values that are <= current value.
    """
    n = len(atr)
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        curr = atr[i]
        if not np.isfinite(curr):
            continue
        hist = atr[i - window : i]
        valid = hist[np.isfinite(hist)]
        if len(valid) < window // 2:
            continue
        ranks[i] = float(np.mean(valid <= curr))
    return ranks


# =============================================================================
# DATA
# =============================================================================

def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    return (df[["time", "open", "high", "low", "close"]]
            .dropna(subset=["time", "close"])
            .sort_values("time")
            .reset_index(drop=True))


# =============================================================================
# BACKTEST
# =============================================================================

@dataclass
class Trade:
    symbol:       str
    entry_time:   object
    exit_time:    object
    entry_price:  float
    exit_price:   float
    initial_stop: float
    initial_risk: float
    exit_reason:  str
    bars_held:    int
    net_r:        float
    mae_r:        float
    mfe_r:        float
    atr_pct_at_entry: float   # percentile rank logged for diagnostics


def run_backtest(
    df:        pd.DataFrame,
    symbol:    str,
    entry_n:   int  = CANONICAL_N,
    apply_atr_filter: bool = True,
) -> List[Trade]:
    """Donchian Long backtest with optional ATR percentile gate."""
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    min_bars = max(entry_n, ATR_N, 200) + ATR_PCT_WINDOW + 20
    if nb < min_bars:
        return []

    exit_n = entry_n // 2

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()

    ema200   = _ema(cl, 200)
    atr14    = _atr(hi, lo, cl, ATR_N)
    atr_rank = _atr_pct_rank(atr14, ATR_PCT_WINDOW)

    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades: List[Trade] = []
    pos: Optional[dict] = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        if not (np.isfinite(ema200[i - 1]) and np.isfinite(atr14[i - 1])):
            continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])):
            continue

        # ── MANAGE OPEN POSITION ────────────────────────────────────────────
        if pos is not None:
            raw_gain = hi[i] - pos["entry"]
            raw_loss = lo[i] - pos["entry"]
            pos["mfe_r"] = max(pos["mfe_r"], raw_gain / pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], raw_loss / pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            exit_px = exit_reason = None

            if lo[i] <= pos["stop"]:
                exit_px     = pos["stop"]
                exit_reason = "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px     = pos["chan_stop"]
                exit_reason = "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px     = cl[i]
                exit_reason = "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(Trade(
                    symbol            = symbol,
                    entry_time        = pos["entry_time"],
                    exit_time         = ts[i],
                    entry_price       = pos["entry"],
                    exit_price        = exit_px,
                    initial_stop      = pos["stop"],
                    initial_risk      = pos["risk"],
                    exit_reason       = exit_reason,
                    bars_held         = pos["bars"],
                    net_r             = net_r,
                    mae_r             = min(pos["mae_r"], net_r),
                    mfe_r             = pos["mfe_r"],
                    atr_pct_at_entry  = pos["atr_pct"],
                ))
                pos = None
            else:
                if pos["mfe_r"] >= CHAN_ACT_R:
                    pos["chan_active"] = True
                if pos["chan_active"]:
                    new_cs = pos["hh"] - atr14[i] * CHAN_TRAIL
                    pos["chan_stop"] = max(pos["chan_stop"], new_cs)

        # ── ENTRY ────────────────────────────────────────────────────────────
        if pos is None:
            breakout = (np.isfinite(ema200[i - 1]) and
                        cl[i] > ema200[i - 1] and
                        cl[i] > don_upper[i])
            if not breakout:
                continue

            if apply_atr_filter:
                if not np.isfinite(atr_rank[i]):
                    continue
                if not (ATR_PCT_LOW <= atr_rank[i] <= ATR_PCT_HIGH):
                    continue

            risk = atr14[i - 1] * STOP_MULT
            if risk <= 0:
                continue
            stop_px = cl[i] - risk
            pos = {
                "entry":       cl[i],
                "stop":        stop_px,
                "risk":        risk,
                "entry_time":  ts[i],
                "hh":          hi[i],
                "chan_active": False,
                "chan_stop":   stop_px,
                "mfe_r":       0.0,
                "mae_r":       0.0,
                "bars":        1,
                "atr_pct":     float(atr_rank[i]),
            }

    if pos is not None:
        exit_px = cl[-1]
        net_r   = (exit_px - pos["entry"]) / pos["risk"]
        trades.append(Trade(
            symbol            = symbol,
            entry_time        = pos["entry_time"],
            exit_time         = ts[-1],
            entry_price       = pos["entry"],
            exit_price        = exit_px,
            initial_stop      = pos["stop"],
            initial_risk      = pos["risk"],
            exit_reason       = "end_of_data",
            bars_held         = pos["bars"],
            net_r             = net_r,
            mae_r             = min(pos["mae_r"], net_r),
            mfe_r             = pos["mfe_r"],
            atr_pct_at_entry  = pos["atr_pct"],
        ))

    return trades


def run_universe(entry_n: int, apply_filter: bool, symbols: List[str]) -> pd.DataFrame:
    all_trades: List[Trade] = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 300:
            continue
        trades = run_backtest(df, sym, entry_n=entry_n, apply_atr_filter=apply_filter)
        all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(t) for t in all_trades])
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[EXIT_T, ENTRY_T, R_COL])
    df = df.sort_values(EXIT_T).reset_index(drop=True)
    df["month"] = df[EXIT_T].dt.tz_convert(None).dt.to_period("M").astype(str)
    df["year"]  = df[EXIT_T].dt.year
    return df


# =============================================================================
# STATS HELPERS
# =============================================================================

def _pf(r: np.ndarray) -> float:
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)


def _max_dd(r: np.ndarray) -> float:
    if r.size == 0: return 0.0
    eq = np.cumsum(r); pk = np.maximum.accumulate(eq)
    return float((eq - pk).min())


def summarize(r: np.ndarray) -> dict:
    if r.size == 0:
        return dict(trades=0, total_r=0.0, avg_r=0.0, win_rate=0.0,
                    profit_factor=0.0, max_dd_r=0.0, std_r=0.0, t_score=0.0)
    avg = float(r.mean()); std = float(r.std(ddof=1)) if r.size > 1 else 0.0
    t   = avg / (std / math.sqrt(r.size)) if std > 0 else 0.0
    return dict(trades=int(r.size), total_r=float(r.sum()), avg_r=avg,
                win_rate=float((r > 0).mean()), profit_factor=_pf(r),
                max_dd_r=_max_dd(r), std_r=std, t_score=float(t))


def _block_bootstrap(vals: np.ndarray, bs: int, rng) -> np.ndarray:
    n = len(vals); out = []
    while len(out) < n:
        s = int(rng.integers(0, n)); e = min(s + bs, n)
        blk = vals[s:e]
        if len(blk) < bs:
            blk = np.concatenate([blk, vals[: bs - len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n], dtype=float)


# =============================================================================
# T4 ROBUSTNESS
# =============================================================================

def t4_baseline(df: pd.DataFrame) -> dict:
    r = df[R_COL].to_numpy(dtype=float)
    s = summarize(r)
    s.update(assets=df["symbol"].nunique(), months=df["month"].nunique(),
             start=str(df[EXIT_T].min().date()), end=str(df[EXIT_T].max().date()))
    return s


def t4_montecarlo(df: pd.DataFrame) -> pd.DataFrame:
    vals = df[R_COL].to_numpy(dtype=float)
    rng  = np.random.default_rng(MC_SEED)
    rows = []
    for bs in MC_BLOCKS:
        tots, dds, pfs = [], [], []
        for _ in range(MC_RUNS):
            s = _block_bootstrap(vals, bs, rng)
            tots.append(s.sum()); dds.append(_max_dd(s)); pfs.append(_pf(s))
        tots = np.array(tots); dds = np.array(dds); pfs = np.array(pfs)
        rows.append(dict(
            block_size=bs, mc_runs=MC_RUNS,
            total_r_p05=float(np.percentile(tots, 5)),
            total_r_p50=float(np.percentile(tots, 50)),
            total_r_p95=float(np.percentile(tots, 95)),
            dd_p95=float(np.percentile(dds, 95)),
            pf_p05=float(np.percentile(pfs, 5)),
            pf_p50=float(np.percentile(pfs, 50)),
            prob_positive=float((tots > 0).mean()),
        ))
    return pd.DataFrame(rows)


def t4_cost_stress(df: pd.DataFrame) -> pd.DataFrame:
    vals = df[R_COL].to_numpy(dtype=float)
    rows = []
    for ec in EXTRA_COSTS:
        s = summarize(vals - ec); s["extra_cost"] = ec; rows.append(s)
    return pd.DataFrame(rows)


def t4_period_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(EXIT_T).reset_index(drop=True)
    mid = len(df) // 2; med_t = df[EXIT_T].median()
    slices = {
        "first_half_by_trade":  df.iloc[:mid],
        "second_half_by_trade": df.iloc[mid:],
        "last_100_trades":      df.tail(100),
        "first_half_by_time":   df[df[EXIT_T] <= med_t],
        "second_half_by_time":  df[df[EXIT_T] > med_t],
    }
    rows = []
    for name, sub in slices.items():
        s = summarize(sub[R_COL].to_numpy(dtype=float)); s["split"] = name; rows.append(s)
    return pd.DataFrame(rows)


def t4_remove_best_assets(df: pd.DataFrame) -> pd.DataFrame:
    asset_r = df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    rows = []
    for n in [0, 1, 3, 5]:
        removed = asset_r.head(n).index.tolist()
        sub = df[~df["symbol"].isin(removed)]
        s   = summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n, removed=",".join(removed),
                 assets_remaining=sub["symbol"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)


def t4_remove_best_months(df: pd.DataFrame) -> pd.DataFrame:
    month_r = df.groupby("month")[R_COL].sum().sort_values(ascending=False)
    rows = []
    for n in [0, 1, 2, 3]:
        removed = month_r.head(n).index.tolist()
        sub = df[~df["month"].isin(removed)]
        s   = summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n, removed=",".join(removed),
                 months_remaining=sub["month"].nunique())
        rows.append(s)
    return pd.DataFrame(rows)


def t4_concentration(df: pd.DataFrame) -> dict:
    asset_r = df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    total   = float(asset_r.sum())
    return dict(
        top1_symbol=str(asset_r.index[0]),
        top1_r=float(asset_r.iloc[0]),
        top1_pct=float(asset_r.iloc[0] / total) if total else 0.0,
        top3_r=float(asset_r.head(3).sum()),
        top3_pct=float(asset_r.head(3).sum() / total) if total else 0.0,
        top5_r=float(asset_r.head(5).sum()),
        top5_pct=float(asset_r.head(5).sum() / total) if total else 0.0,
        total_r=total,
        positive_assets=int((asset_r > 0).sum()),
        negative_assets=int((asset_r <= 0).sum()),
        total_assets=len(asset_r),
    )


# =============================================================================
# T5 PORTFOLIO + T6 CAPITAL
# =============================================================================

def t5_replay(df: pd.DataFrame, max_open: int) -> Tuple[pd.DataFrame, dict]:
    df = df.sort_values(ENTRY_T).reset_index(drop=True)
    open_pos: list = []; closed: list = []

    def _flush(now):
        nonlocal open_pos
        still = [p for p in open_pos if p["exit_time"] > now]
        closed.extend(p for p in open_pos if p["exit_time"] <= now)
        open_pos[:] = still

    for _, row in df.iterrows():
        _flush(row[ENTRY_T])
        if any(p["symbol"] == row["symbol"] for p in open_pos):
            continue
        if len(open_pos) >= max_open:
            continue
        open_pos.append({"symbol": row["symbol"],
                          "entry_time": row[ENTRY_T],
                          "exit_time":  row[EXIT_T],
                          R_COL:        row[R_COL]})
    if open_pos:
        last = max(p["exit_time"] for p in open_pos)
        _flush(last + pd.Timedelta(seconds=1))

    acc = pd.DataFrame(closed)
    if acc.empty:
        return acc, dict(accepted=0, total_r=0.0, avg_r=0.0,
                         profit_factor=0.0, max_dd_r=0.0, win_rate=0.0)
    acc = acc.sort_values("exit_time").reset_index(drop=True)
    r   = acc[R_COL].to_numpy(dtype=float)
    return acc, dict(accepted=int(len(r)), total_r=float(r.sum()),
                     avg_r=float(r.mean()), profit_factor=_pf(r),
                     max_dd_r=_max_dd(r), win_rate=float((r > 0).mean()))


def t6_equity(acc: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if acc.empty:
        return pd.DataFrame(), {}
    equity = START_CAP; peak = START_CAP; kill = False; rows = []
    for _, row in acc.sort_values("exit_time").iterrows():
        pnl     = row[R_COL] * equity * RISK_PCT
        equity += pnl; peak = max(peak, equity)
        dd_pct  = (equity - peak) / peak
        if dd_pct <= KILL_SWITCH_DD and not kill:
            kill = True
        rows.append(dict(exit_time=row["exit_time"], symbol=row["symbol"],
                         net_r=row[R_COL], equity=equity, peak=peak,
                         dd_pct=float(dd_pct), kill_fired=kill))
    eq_df  = pd.DataFrame(rows)
    final  = float(eq_df["equity"].iloc[-1])
    start  = acc[ENTRY_T].min(); end = acc["exit_time"].max()
    years  = float((end - start).days / 365.25) if pd.notnull(start) and pd.notnull(end) else 1.0
    cagr   = float((final / START_CAP) ** (1 / years) - 1) if years > 0 else 0.0
    return eq_df, dict(
        start_capital=START_CAP, end_capital=final,
        total_return_pct=float((final - START_CAP) / START_CAP * 100),
        cagr_pct=float(cagr * 100),
        max_dd_pct=float(eq_df["dd_pct"].min() * 100),
        years=round(years, 2),
        kill_switch_fired=kill,
    )


# =============================================================================
# MASTER REPORT
# =============================================================================

def write_master_report(
    stab:       pd.DataFrame,
    baseline:   dict,
    mc:         pd.DataFrame,
    cost:       pd.DataFrame,
    splits:     pd.DataFrame,
    rem_assets: pd.DataFrame,
    rem_months: pd.DataFrame,
    conc:       dict,
    t5_stats:   dict,
    t6_stats:   dict,
    atr_pct_stats: dict,
    trade_guard_pass: bool,
) -> None:

    def gate(cond: bool, label: str) -> str:
        return f"  {'PASS' if cond else 'FAIL'}  {label}"

    mc10      = mc[mc["block_size"] == 10].iloc[0]
    cost10    = cost[cost["extra_cost"] == 0.10].iloc[0]
    sec_half  = splits[splits["split"] == "second_half_by_trade"].iloc[0]
    rem_top1  = rem_assets[rem_assets["removed_n"] == 1].iloc[0]
    rem_top1m = rem_months[rem_months["removed_n"] == 1].iloc[0]

    lines = [
        "PHASE T17 OPT-17.2 -- Regime Filter C: ATR Percentile Gate",
        "=" * 70,
        "",
        f"Filter   : 40th <= ATR(14) percentile rank <= 80th (trailing {ATR_PCT_WINDOW} bars)",
        "Universe : 24 symbols (filtered_symbols_v2_included_only.csv)",
        "Config   : Donchian N=20 / ema200_price / ATR(14)x2.0 stop /",
        "           Chandelier ACT+4R trail 3xATR / LONG only / 1D",
        "",
        f"Baseline (no filter): avg_r=+1.101R  PF=3.072  CAGR=+14.3%  trades=461",
        "",
        "=" * 70,
        f"ATR PERCENTILE FILTER STATS (N=20, filter ON)",
        "=" * 70,
        f"  Signals in 40-80 pct band : {atr_pct_stats['accepted_signals']}  "
        f"({atr_pct_stats['accepted_pct']:.1%} of all breakout signals)",
        f"  Signals rejected <40 pct   : {atr_pct_stats['rejected_low']}  (too quiet)",
        f"  Signals rejected >80 pct   : {atr_pct_stats['rejected_high']}  (too noisy)",
        f"  Avg ATR pct at accepted entries: {atr_pct_stats['avg_pct_accepted']:.3f}",
        "",
        "=" * 70,
        "STABILITY ZONE  (N=[15, 20, 25], ATR percentile filter ON)",
        "=" * 70,
        f"  {'N':>4s}  {'exit_n':>6s}  {'trades':>7s}  {'total_r':>8s}  "
        f"{'avg_r':>8s}  {'PF':>5s}  {'PASS?':>6s}",
        "  " + "-" * 58,
    ]

    all_stab_pass = True
    for _, r in stab.iterrows():
        ok = r["avg_r"] > 0 and r["total_r"] > 0
        if not ok:
            all_stab_pass = False
        lines.append(
            f"  {int(r['entry_n']):>4d}  {int(r['exit_n']):>6d}  "
            f"{int(r['trades']):>7d}  {r['total_r']:>+8.2f}  "
            f"{r['avg_r']:>+8.4f}  {r['profit_factor']:>5.3f}  "
            f"{'YES' if ok else 'NO ':>6s}"
        )

    lines += [
        "",
        f"  Stability zone result: {'ALL PASS' if all_stab_pass else 'FAIL -- not all N profitable'}",
        "",
        "=" * 70, "T4 -- BASELINE (N=20, ATR percentile filter ON)", "=" * 70,
        f"  Trades     : {baseline['trades']}  "
        f"  [guard: >={MIN_TRADES}  {'PASS' if trade_guard_pass else 'FAIL'}]",
        f"  Total R    : {baseline['total_r']:+.2f}R",
        f"  Avg R      : {baseline['avg_r']:+.4f}R   (baseline: +1.101R)",
        f"  PF         : {baseline['profit_factor']:.3f}          (baseline: 3.072)",
        f"  Win rate   : {baseline['win_rate']:.1%}",
        f"  Max DD     : {baseline['max_dd_r']:+.2f}R",
        f"  t-score    : {baseline['t_score']:.2f}",
        f"  Assets     : {baseline['assets']}",
        f"  Period     : {baseline['start']} -> {baseline['end']}",
        "",
        "=" * 70, "T4 -- MONTE CARLO (2000 runs, block_size=10)", "=" * 70,
        f"  totalR  p05/p50/p95 = {mc10['total_r_p05']:.1f} / "
        f"{mc10['total_r_p50']:.1f} / {mc10['total_r_p95']:.1f}",
        f"  prob(totalR > 0)    = {mc10['prob_positive']:.1%}",
        f"  PF      p05/p50     = {mc10['pf_p05']:.2f} / {mc10['pf_p50']:.2f}",
        f"  DD      p95 (worst) = {mc10['dd_p95']:.2f}R",
        "",
        "=" * 70, "T4 -- COST STRESS", "=" * 70,
    ]
    for _, r in cost.iterrows():
        lines.append(
            f"  +{r['extra_cost']:.2f}R/trade -> "
            f"totalR={r['total_r']:+.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}"
        )

    lines += ["", "=" * 70, "T4 -- PERIOD SPLITS", "=" * 70]
    for _, r in splits.iterrows():
        lines.append(
            f"  {r['split']:30s}: trades={int(r['trades']):3d}  "
            f"totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  "
            f"PF={r['profit_factor']:.3f}  win={r['win_rate']:.1%}"
        )

    lines += [
        "", "=" * 70, "T4 -- REMOVE BEST ASSETS", "=" * 70,
        f"  Top asset: {conc['top1_symbol']}  "
        f"({conc['top1_r']:.2f}R = {conc['top1_pct']:.1%} of total)",
    ]
    for _, r in rem_assets.iterrows():
        lines.append(
            f"  remove top {int(r['removed_n'])}: "
            f"totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  "
            f"PF={r['profit_factor']:.3f}  remaining={int(r['assets_remaining'])}  [{r['removed']}]"
        )

    lines += ["", "=" * 70, "T4 -- REMOVE BEST MONTHS", "=" * 70]
    for _, r in rem_months.iterrows():
        lines.append(
            f"  remove top {int(r['removed_n'])}: "
            f"totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  "
            f"PF={r['profit_factor']:.3f}  months_remaining={int(r['months_remaining'])}  [{r['removed']}]"
        )

    lines += [
        "", "=" * 70, "T4 -- ASSET CONCENTRATION", "=" * 70,
        f"  Top 1:  {conc['top1_r']:.1f}R  ({conc['top1_pct']:.1%} of total)",
        f"  Top 3:  {conc['top3_r']:.1f}R  ({conc['top3_pct']:.1%} of total)",
        f"  Top 5:  {conc['top5_r']:.1f}R  ({conc['top5_pct']:.1%} of total)",
        f"  Positive assets: {conc['positive_assets']}/{conc['total_assets']}",
        "",
        "=" * 70, "T5/T6 -- PORTFOLIO (max8) + CAPITAL EXECUTION ($10k / 0.25% risk)", "=" * 70,
        f"  Accepted trades : {t5_stats.get('accepted', 0)}",
        f"  Portfolio avg_r : {t5_stats.get('avg_r', 0):+.4f}R",
        f"  Portfolio PF    : {t5_stats.get('profit_factor', 0):.3f}",
    ]
    if t6_stats:
        ks = "YES ***FIRED***" if t6_stats.get("kill_switch_fired") else "no"
        lines += [
            f"  End capital     : ${t6_stats.get('end_capital', 0):,.0f}",
            f"  Total return    : {t6_stats.get('total_return_pct', 0):+.1f}%",
            f"  CAGR            : {t6_stats.get('cagr_pct', 0):+.1f}%    (baseline: +14.3%)",
            f"  Max DD          : {t6_stats.get('max_dd_pct', 0):+.1f}%    (baseline: -3.4%)",
            f"  Kill switch     : {ks}",
            f"  Duration        : {t6_stats.get('years', 0):.1f} yr",
        ]

    # ── GATE CHECKS ──────────────────────────────────────────────────────────
    avg_r_ok  = baseline["avg_r"] > 0.15
    pf_ok     = baseline["profit_factor"] > 1.0
    mc_ok     = float(mc10["total_r_p05"]) > 0
    cost_ok   = float(cost10["profit_factor"]) > 1.0
    sec_ok    = float(sec_half["avg_r"]) > 0
    rem_a_ok  = float(rem_top1["total_r"]) > 0
    rem_m_ok  = float(rem_top1m["total_r"]) > 0
    kill_ok   = not t6_stats.get("kill_switch_fired", False)
    win_ok    = 0.30 <= baseline["win_rate"] <= 0.45

    better_r    = baseline["avg_r"] > BASELINE["avg_r"]
    better_pf   = baseline["profit_factor"] > BASELINE["pf"]
    better_cagr = t6_stats.get("cagr_pct", 0) > BASELINE["cagr_pct"] if t6_stats else False

    all_gates = all([avg_r_ok, pf_ok, mc_ok, cost_ok, sec_ok,
                     rem_a_ok, rem_m_ok, kill_ok, win_ok,
                     trade_guard_pass, all_stab_pass])

    lines += [
        "",
        "=" * 70, "FINAL SCORECARD -- Filter C vs Baseline", "=" * 70, "",
        gate(trade_guard_pass, f"Trade count >= {MIN_TRADES}          (got {baseline['trades']})"),
        gate(all_stab_pass,    "Stability zone all profitable (N=15/20/25)"),
        gate(win_ok,           f"§4.1 win rate 30-45%          (got {baseline['win_rate']:.1%})"),
        gate(avg_r_ok,         f"§4.2 avg_r > 0.15R            (got {baseline['avg_r']:+.4f}R)"),
        gate(pf_ok,            f"PF > 1.0                      (got {baseline['profit_factor']:.3f})"),
        gate(mc_ok,            f"MC p05 totalR > 0             (got {float(mc10['total_r_p05']):+.1f}R)"),
        gate(cost_ok,          f"Cost +0.10R PF > 1.0          (got {float(cost10['profit_factor']):.3f})"),
        gate(sec_ok,           f"2nd half avg_r > 0            (got {float(sec_half['avg_r']):+.4f}R)"),
        gate(rem_a_ok,         f"Remove top-1 asset profitable (totalR={float(rem_top1['total_r']):+.1f}R)"),
        gate(rem_m_ok,         f"Remove top-1 month profitable (totalR={float(rem_top1m['total_r']):+.1f}R)"),
        gate(kill_ok,          "Kill switch not fired"),
        "",
        "  vs BASELINE (no ATR percentile filter):",
        f"  avg_r  : {baseline['avg_r']:+.4f}R  vs  +1.101R  -> "
        f"{'BETTER' if better_r else 'WORSE'}",
        f"  PF     : {baseline['profit_factor']:.3f}  vs  3.072  -> "
        f"{'BETTER' if better_pf else 'WORSE'}",
    ]
    if t6_stats:
        lines.append(
            f"  CAGR   : {t6_stats.get('cagr_pct', 0):+.1f}%  vs  +14.3%  -> "
            f"{'BETTER' if better_cagr else 'WORSE'}"
        )
    lines += [
        f"  trades : {baseline['trades']}  vs  461  "
        f"(filter removed {461 - baseline['trades']} trades = "
        f"{(461 - baseline['trades']) / 461:.1%} of baseline)",
        "",
        f"  OVERALL: {'PASS -- Filter C improves or maintains quality' if all_gates else 'FAIL -- one or more gates failed'}",
    ]

    rpt = OUT_DIR / "filter_c_master_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Master report: {rpt}")


# =============================================================================
# ATR PERCENTILE DIAGNOSTICS
# =============================================================================

def compute_atr_pct_stats(symbols: List[str]) -> dict:
    """Count how many breakout signals fall in each ATR percentile bucket."""
    accepted = rejected_low = rejected_high = 0
    pcts_accepted: List[float] = []

    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 300:
            continue
        nb = len(df)
        cl = df["close"].to_numpy(dtype=float)
        hi = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        ema200   = _ema(cl, 200)
        atr14    = _atr(hi, lo, cl, ATR_N)
        atr_rank = _atr_pct_rank(atr14, ATR_PCT_WINDOW)
        don_upper = pd.Series(hi).shift(1).rolling(CANONICAL_N).max().to_numpy()
        in_pos = False
        for i in range(1, nb):
            if not (np.isfinite(cl[i]) and np.isfinite(ema200[i - 1]) and
                    np.isfinite(don_upper[i])):
                continue
            breakout = cl[i] > ema200[i - 1] and cl[i] > don_upper[i]
            if in_pos:
                don_lower = pd.Series(lo).shift(1).rolling(CANONICAL_N // 2).min().to_numpy()
                if np.isfinite(don_lower[i]) and cl[i] < don_lower[i]:
                    in_pos = False
                continue
            if breakout and np.isfinite(atr_rank[i]):
                if ATR_PCT_LOW <= atr_rank[i] <= ATR_PCT_HIGH:
                    accepted += 1
                    pcts_accepted.append(float(atr_rank[i]))
                    in_pos = True
                elif atr_rank[i] < ATR_PCT_LOW:
                    rejected_low += 1
                else:
                    rejected_high += 1

    total = accepted + rejected_low + rejected_high
    return dict(
        accepted_signals=accepted,
        rejected_low=rejected_low,
        rejected_high=rejected_high,
        accepted_pct=accepted / total if total else 0.0,
        avg_pct_accepted=float(np.mean(pcts_accepted)) if pcts_accepted else 0.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("Phase T17 Opt-17.2 -- Regime Filter C: ATR Percentile Gate")
    print("=" * 70)

    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()
    print(f"Universe : {len(symbols)} symbols")
    print(f"Filter   : {ATR_PCT_LOW:.0%} <= ATR(14) pct rank <= {ATR_PCT_HIGH:.0%}  "
          f"(trailing {ATR_PCT_WINDOW} bars)")
    print()

    # ── ATR PERCENTILE SIGNAL STATS ─────────────────────────────────────────
    print("[0] Computing ATR percentile signal distribution ...")
    atr_pct_stats = compute_atr_pct_stats(symbols)
    print(f"  Accepted (40-80 pct): {atr_pct_stats['accepted_signals']}  "
          f"({atr_pct_stats['accepted_pct']:.1%})")
    print(f"  Rejected <40 pct:     {atr_pct_stats['rejected_low']}")
    print(f"  Rejected >80 pct:     {atr_pct_stats['rejected_high']}")

    # ── STABILITY ZONE ───────────────────────────────────────────────────────
    print("\n[1] Stability zone (N=15/20/25, ATR percentile filter ON) ...")
    stab_rows = []
    for n in STABILITY_N:
        df_n = run_universe(entry_n=n, apply_filter=True, symbols=symbols)
        if df_n.empty:
            stab_rows.append(dict(entry_n=n, exit_n=n // 2, trades=0,
                                  total_r=0.0, avg_r=0.0, profit_factor=0.0))
            continue
        r = df_n[R_COL].to_numpy(dtype=float)
        s = summarize(r)
        stab_rows.append(dict(entry_n=n, exit_n=n // 2, trades=s["trades"],
                              total_r=s["total_r"], avg_r=s["avg_r"],
                              profit_factor=s["profit_factor"]))
        ok = s["avg_r"] > 0 and s["total_r"] > 0
        print(f"  N={n:2d}: trades={s['trades']:4d}  avg_r={s['avg_r']:+.4f}  "
              f"PF={s['profit_factor']:.3f}  {'PASS' if ok else 'FAIL'}")

    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT_DIR / "filter_c_stability_zone.csv", index=False)

    # ── CANONICAL N=20 ───────────────────────────────────────────────────────
    print(f"\n[2] Canonical N={CANONICAL_N} with ATR percentile filter ...")
    trades = run_universe(entry_n=CANONICAL_N, apply_filter=True, symbols=symbols)
    if trades.empty:
        print("  ERROR: No trades generated.")
        return 1

    trades.to_csv(OUT_DIR / "filter_c_all_trades_N20.csv", index=False)
    trade_count = len(trades)
    trade_guard_pass = trade_count >= MIN_TRADES
    print(f"  Trades : {trade_count}  [guard >={MIN_TRADES}: {'PASS' if trade_guard_pass else 'FAIL'}]")
    print(f"  Symbols: {trades['symbol'].nunique()}")
    print(f"  Period : {trades[EXIT_T].min().date()} -> {trades[EXIT_T].max().date()}")

    # ── T4 ROBUSTNESS ────────────────────────────────────────────────────────
    print("\n[3] T4 robustness battery ...")
    baseline   = t4_baseline(trades)
    mc         = t4_montecarlo(trades)
    cost       = t4_cost_stress(trades)
    splits     = t4_period_splits(trades)
    rem_assets = t4_remove_best_assets(trades)
    rem_months = t4_remove_best_months(trades)
    conc       = t4_concentration(trades)

    mc.to_csv(OUT_DIR / "filter_c_t4_montecarlo.csv", index=False)
    cost.to_csv(OUT_DIR / "filter_c_t4_cost_stress.csv", index=False)
    splits.to_csv(OUT_DIR / "filter_c_t4_period_splits.csv", index=False)
    rem_assets.to_csv(OUT_DIR / "filter_c_t4_remove_best_assets.csv", index=False)
    rem_months.to_csv(OUT_DIR / "filter_c_t4_remove_best_months.csv", index=False)

    mc10 = mc[mc["block_size"] == 10].iloc[0]
    print(f"  avg_r={baseline['avg_r']:+.4f}  PF={baseline['profit_factor']:.3f}  "
          f"t={baseline['t_score']:.2f}")
    print(f"  MC(bs=10): p05={mc10['total_r_p05']:+.1f}  p50={mc10['total_r_p50']:+.1f}  "
          f"prob_pos={mc10['prob_positive']:.1%}")

    # ── T5/T6 ────────────────────────────────────────────────────────────────
    print(f"\n[4] T5 max{PORT_MAX_OPEN} + T6 capital execution ...")
    acc, t5_stats = t5_replay(trades, PORT_MAX_OPEN)
    t6_stats = {}
    if not acc.empty:
        eq_df, t6_stats = t6_equity(acc)
        eq_df.to_csv(OUT_DIR / f"filter_c_t6_equity_max{PORT_MAX_OPEN}.csv", index=False)
    print(f"  Accepted: {t5_stats.get('accepted', 0)}  "
          f"avgR={t5_stats.get('avg_r', 0):+.4f}  PF={t5_stats.get('profit_factor', 0):.3f}")
    if t6_stats:
        print(f"  CAGR={t6_stats.get('cagr_pct', 0):+.1f}%  "
              f"DD={t6_stats.get('max_dd_pct', 0):+.1f}%  "
              f"kill={'YES' if t6_stats.get('kill_switch_fired') else 'no'}")

    # ── MASTER REPORT ────────────────────────────────────────────────────────
    print("\n[5] Writing master report ...")
    write_master_report(
        stab          = stab_df,
        baseline      = baseline,
        mc            = mc,
        cost          = cost,
        splits        = splits,
        rem_assets    = rem_assets,
        rem_months    = rem_months,
        conc          = conc,
        t5_stats      = t5_stats,
        t6_stats      = t6_stats,
        atr_pct_stats = atr_pct_stats,
        trade_guard_pass = trade_guard_pass,
    )

    print()
    print("=" * 70)
    print(f"Filter C complete -> {OUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
