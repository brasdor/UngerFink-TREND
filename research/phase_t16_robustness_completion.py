#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T16 — ROBUSTNESS COMPLETION

Offline only. T9A paper-live remains frozen.

Completes the three robustness tests that T4/T7 never ran (per UNGER recap §4.7):
  1. MONTE CARLO trade-shuffling  — distribution of total_R / max_DD across 2000 shuffles
  2. DATA PERTURBATION stress     — add Gaussian noise to OHLCV, regenerate signals
  3. SLIPPAGE STRESS              — extra cost per trade from 0R to 0.40R

Unger principle §4.7:
  "Robustness tests include parameter stability checks and data perturbations
   to prevent overfitting."  WFA is explicitly NOT the right tool.

Input (in order of preference):
  1. data/research_trend_t15/phase_t15_entry_stability_trades.csv  (canonical T15 trades)
  2. data/research_trend_t13/phase_t13_timeframe_trades.csv         (T13 6H trades)
  3. data/research_trend_t10/phase_t10_trailing_trades.csv          (T10 ACT4_ATR3 trades)

For data perturbation, OHLCV is re-simulated from T13 ohlcv_cache.

Outputs: data/research_trend_t16/
  phase_t16_monte_carlo_summary.csv
  phase_t16_monte_carlo_distribution.csv
  phase_t16_perturbation_summary.csv
  phase_t16_slippage_stress_summary.csv
  phase_t16_robustness_report.txt
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

try:
    import ccxt as _ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
OUT  = ROOT / "data" / "research_trend_t16"
OUT.mkdir(parents=True, exist_ok=True)

T15_TRADES  = ROOT / "data" / "research_trend_t15" / "phase_t15_entry_stability_trades.csv"
T3B_TRADES  = ROOT / "data" / "research_trend_t3b" / "phase_t3b_wide_exit_trades.csv"
CACHE_T15   = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"   # 1D cache from T15 download

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MC_RUNS       = 2000
MC_SEED       = 42

NOISE_LEVELS  = [0.005, 0.010, 0.020, 0.050]   # 0.5%, 1%, 2%, 5% Gaussian σ on OHLCV
NOISE_SEED    = 99

EXTRA_COSTS   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]   # extra R per trade

# Frozen canonical parameters (T8 config — used for perturbation re-simulation)
TIMEFRAME        = "1d"
DONCHIAN_N       = 20
EMA_REGIME_LEN   = 200           # ema200 price-above filter (T1 4yr confirmed)
ATR_N            = 14
INIT_STOP_ATR    = 2.0
ACT_R            = 4.0
CH_MULT          = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    if l <= 1e-12:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def _dd_r(r: np.ndarray) -> float:
    if not len(r):
        return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate=0.0)
    return dict(
        trades=len(r),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        pf=_pf(r),
        max_dd_r=_dd_r(r),
        win_rate=float((r > 0).mean()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CANONICAL TRADES
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical_trades() -> pd.DataFrame:
    for path, filter_fn in [
        (T15_TRADES, None),
        (T3B_TRADES, lambda df: df[df["timeframe"].str.lower() == "1d"]),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if filter_fn:
            df = filter_fn(df)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True, errors="coerce")
        df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
        df = df.dropna(subset=["entry_time","exit_time","net_r"])
        df = df.sort_values("entry_time").reset_index(drop=True)
        if not df.empty:
            print(f"Loaded {len(df)} canonical trades from: {path.name}")
            return df

    raise FileNotFoundError(
        "No canonical trade file found. Run T15, T13, or T10 first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. MONTE CARLO — TRADE SHUFFLING
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle trade order 2000 times, compute key stats for each run.
    Measures how much of the edge depends on sequencing (correlation with equity path)
    vs. the raw distribution of individual trade outcomes.
    """
    r_arr = trades["net_r"].values.astype(float)
    rng = np.random.default_rng(MC_SEED)

    runs = []
    for i in range(MC_RUNS):
        shuffled = rng.permutation(r_arr)
        runs.append(_stats(shuffled))

    dist_df = pd.DataFrame(runs)

    # Baseline (original order)
    baseline = _stats(r_arr)

    # Summary: baseline vs MC percentiles
    summary_rows = []
    for col in ["total_r", "avg_r", "pf", "max_dd_r", "win_rate"]:
        vals = dist_df[col].values
        summary_rows.append({
            "metric":           col,
            "baseline":         baseline[col],
            "mc_mean":          float(np.mean(vals)),
            "mc_p5":            float(np.percentile(vals, 5)),
            "mc_p10":           float(np.percentile(vals, 10)),
            "mc_p25":           float(np.percentile(vals, 25)),
            "mc_p50":           float(np.percentile(vals, 50)),
            "mc_p75":           float(np.percentile(vals, 75)),
            "mc_p90":           float(np.percentile(vals, 90)),
            "mc_p95":           float(np.percentile(vals, 95)),
            "baseline_above_p10": bool(baseline[col] > np.percentile(vals, 10)),
            "baseline_above_p25": bool(baseline[col] > np.percentile(vals, 25)),
            "baseline_above_p50": bool(baseline[col] > np.percentile(vals, 50)),
        })

    return pd.DataFrame(summary_rows), dist_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PERTURBATION — OHLCV NOISE
# ─────────────────────────────────────────────────────────────────────────────

def _safe_sym(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _load_ohlcv_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time","open","high","low","close"])
        df = df.sort_values("time").reset_index(drop=True)
        # Only closed candles
        now = pd.Timestamp.utcnow()
        df["close_time"] = df["time"] + pd.Timedelta(days=1)
        df = df[df["close_time"] <= now].drop(columns=["close_time"])
        return df if len(df) >= 100 else None
    except Exception:
        return None


def _load_ohlcv(sym: str) -> pd.DataFrame | None:
    fname = f"{_safe_sym(sym)}_{TIMEFRAME}.csv"
    df = _load_ohlcv_csv(CACHE_T15 / fname)
    if df is not None:
        return df
    return None


def _add_indicators_pert(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-pc).abs(), (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"]      = tr.rolling(ATR_N).mean()
    d["ema200"]   = d["close"].ewm(span=EMA_REGIME_LEN, adjust=False).mean()
    d["don_high"] = d["high"].shift(1).rolling(DONCHIAN_N).max()
    d["don_low"]  = d["low"].shift(1).rolling(DONCHIAN_N).min()
    exit_n        = max(1, DONCHIAN_N // 2)
    d["don_exit_low"] = d["low"].shift(1).rolling(exit_n).min()
    return d


def _simulate_one(sym: str, df: pd.DataFrame) -> list[float]:
    d = _add_indicators_pert(df)
    warmup = max(DONCHIAN_N, EMA_REGIME_LEN, ATR_N) + 5
    pos = None
    results = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])

        if pos is not None:
            pos["hh"] = max(pos["hh"], high)
            pos["mfe"] = max(pos["mfe"], (high - pos["entry"]) / pos["risk"])
            # 1. Stop check uses stop from START of candle
            if low <= pos["stop"]:
                results.append((pos["stop"] - pos["entry"]) / pos["risk"])
                pos = None
                continue
            # 2. Donchian N//2 exit on close (while Chandelier not yet active)
            if not pos["trail"]:
                don_exit_low = float(row.get("don_exit_low", np.nan))
                if np.isfinite(don_exit_low) and close < don_exit_low:
                    results.append((close - pos["entry"]) / pos["risk"])
                    pos = None
                    continue
            # 3. Update chandelier stop for NEXT candle
            if pos["mfe"] >= ACT_R:
                pos["stop"] = max(pos["stop"], pos["hh"] - CH_MULT * atr)
                pos["trail"] = True

        if pos is None:
            dh   = row.get("don_high")
            e200 = row.get("ema200")
            if not all(np.isfinite(float(x)) for x in [dh, e200]):
                continue
            # LONG only: close breaks Donchian high AND price above EMA200
            if close > float(dh) and close > float(e200):
                risk = atr * INIT_STOP_ATR
                if risk > 0:
                    pos = dict(side="LONG", entry=close, stop=close-risk, risk=risk,
                               hh=close, mfe=0.0, trail=False)

    return results


def data_perturbation(trade_symbols: list[str]) -> pd.DataFrame:
    """
    For each noise level: add Gaussian noise to every OHLCV price column,
    re-run the canonical backtest, compare stats to unperturbed baseline.
    """
    rng = np.random.default_rng(NOISE_SEED)

    # Load OHLCV for symbols that appear in the canonical trade list
    ohlcv: dict[str, pd.DataFrame] = {}
    for sym in set(trade_symbols):
        df = _load_ohlcv(sym)
        if df is not None:
            ohlcv[sym] = df

    if not ohlcv:
        print("  WARNING: no OHLCV available for perturbation test. Skipping.")
        return pd.DataFrame()

    # Baseline (no noise)
    baseline_r: list[float] = []
    for sym, df in ohlcv.items():
        baseline_r.extend(_simulate_one(sym, df))
    baseline = _stats(np.array(baseline_r))

    rows = [{"noise_pct": 0.0, "label": "BASELINE", **baseline}]

    for noise in NOISE_LEVELS:
        perturbed_r: list[float] = []
        for sym, df in ohlcv.items():
            noisy = df.copy()
            for col in ["open","high","low","close"]:
                noise_factors = rng.normal(1.0, noise, size=len(noisy))
                noisy[col] = (noisy[col] * noise_factors).clip(lower=1e-12)
            # Maintain OHLC consistency (high >= open,close,low; low <= open,close)
            noisy["high"] = noisy[["open","close","high"]].max(axis=1)
            noisy["low"]  = noisy[["open","close","low"]].min(axis=1)
            perturbed_r.extend(_simulate_one(sym, noisy))

        s = _stats(np.array(perturbed_r))
        rows.append({"noise_pct": noise * 100, "label": f"NOISE_{noise*100:.1f}pct", **s})

    df_out = pd.DataFrame(rows)
    # Compute pct change vs baseline
    for col in ["total_r","avg_r","pf","max_dd_r","win_rate"]:
        base_val = baseline.get(col, 0.0)
        if abs(base_val) > 1e-12:
            df_out[f"{col}_vs_base_pct"] = (df_out[col] - base_val) / abs(base_val) * 100
        else:
            df_out[f"{col}_vs_base_pct"] = 0.0

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. SLIPPAGE STRESS
# ─────────────────────────────────────────────────────────────────────────────

def slippage_stress(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional cost per trade (in R units) to canonical trade list.
    Unger §4.2: edge must survive realistic execution costs.
    """
    r_baseline = trades["net_r"].values.astype(float)
    rows = []
    for extra in EXTRA_COSTS:
        r_adj = r_baseline - extra
        s = _stats(r_adj)
        pct_pos = float((r_adj > 0).sum()) / max(len(r_adj), 1) * 100
        rows.append({
            "extra_cost_r":        extra,
            "extra_cost_label":    f"{extra:.2f}R/trade",
            "surviving_edge_pct":  pct_pos,
            **s,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(mc_summary: pd.DataFrame, pert_df: pd.DataFrame, slip_df: pd.DataFrame,
                 n_trades: int) -> None:
    lines = [
        "PHASE T16 — ROBUSTNESS COMPLETION",
        "=" * 80,
        "",
        "Three robustness tests missing from T4/T7, now completed per Unger §4.7.",
        "",
        f"Canonical trade input:  {n_trades} trades",
        f"Timeframe:              {TIMEFRAME}",
        f"Canonical params:       Donchian N={DONCHIAN_N}, filter=ema200_price",
        f"Exits:                  Chandelier ACT{ACT_R:g}_ATR{CH_MULT:g}",
        "",
    ]

    # 1. Monte Carlo
    lines += ["1. MONTE CARLO TRADE SHUFFLING", "-" * 40,
              f"   Runs: {MC_RUNS}, seed={MC_SEED}", ""]

    if not mc_summary.empty:
        total_r_row = mc_summary[mc_summary["metric"] == "total_r"]
        dd_row      = mc_summary[mc_summary["metric"] == "max_dd_r"]
        pf_row      = mc_summary[mc_summary["metric"] == "pf"]

        for _, row in mc_summary.iterrows():
            lines.append(
                f"   {row['metric']:15s}  baseline={row['baseline']:+.3f}  "
                f"MC_p10={row['mc_p10']:+.3f}  MC_p50={row['mc_p50']:+.3f}  "
                f"MC_p90={row['mc_p90']:+.3f}  "
                f"above_p25={'YES' if row['baseline_above_p25'] else 'NO'}"
            )

        lines += [
            "",
            "   PASS criteria: baseline total_R above MC_p25 AND baseline max_dd_R above",
            "   MC_p25 (i.e., better drawdown than 75% of random orderings).",
            "   → Sequential ordering not needed for the edge to be real.",
        ]

    # 2. Data perturbation
    lines += ["", "2. DATA PERTURBATION STRESS", "-" * 40,
              "   Gaussian noise added to OHLCV price columns.", ""]

    if not pert_df.empty:
        for _, row in pert_df.iterrows():
            lines.append(
                f"   {row['label']:20s}  trades={int(row['trades']):4d}  "
                f"avg_r={row['avg_r']:+.3f}  pf={row['pf']:.2f}  "
                f"dd={row['max_dd_r']:+.2f}  "
                f"avg_r_vs_base={row.get('avg_r_vs_base_pct',0):+.1f}%"
            )
        lines += [
            "",
            "   PASS criteria: avg_R and PF remain positive under 2% noise.",
            "   WARN: degradation > 50% at 1% noise = signal is fragile.",
            "   FAIL: avg_R turns negative under 1% noise = likely overfit.",
        ]
    else:
        lines.append("   SKIPPED — no OHLCV cache available.")

    # 3. Slippage stress
    lines += ["", "3. SLIPPAGE STRESS", "-" * 40,
              "   Extra cost subtracted from each trade's net_R.", ""]

    if not slip_df.empty:
        for _, row in slip_df.iterrows():
            lines.append(
                f"   {row['extra_cost_label']:12s}  avg_r={row['avg_r']:+.3f}  "
                f"pf={row['pf']:.2f}  dd={row['max_dd_r']:+.2f}  "
                f"trades_positive={row['surviving_edge_pct']:.0f}%"
            )
        lines += [
            "",
            "   Realistic cost for 1D crypto on Binance Spot: ≈ 0.05–0.10R per trade",
            "   (taker fees, no leverage). System must survive at least 0.10R extra.",
            "   FAIL if avg_R ≤ 0 at 0.10R extra cost.",
        ]

    lines += [
        "",
        "INTERPRETATION SUMMARY",
        "-" * 40,
        "  MC PASS + Pert PASS + Slip PASS: edge is robust — proceed to T17/T18.",
        "  Any FAIL: do NOT deploy. Diagnose root cause before T17/T18.",
        "  'WARN' on one test only: note, document, monitor in paper-live.",
        "",
        "  T9A paper-live remains frozen regardless of T16 outcome.",
        "  These tests validate the RESEARCH edge, not the live config.",
    ]

    (OUT / "phase_t16_robustness_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("PHASE T16 — ROBUSTNESS COMPLETION")
    print("=" * 80)
    print(f"Output: {OUT}")

    trades = load_canonical_trades()
    n_trades = len(trades)

    # 1. Monte Carlo
    print(f"\n[1/3] Monte Carlo trade shuffling ({MC_RUNS} runs)...")
    t0 = time.time()
    mc_summary, mc_dist = monte_carlo(trades)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(mc_summary[["metric","baseline","mc_p10","mc_p50","mc_p90","baseline_above_p25"]].to_string(index=False))

    # 2. Data perturbation
    print("\n[2/3] Data perturbation stress...")
    trade_symbols = trades["symbol"].dropna().tolist() if "symbol" in trades.columns else []
    t0 = time.time()
    pert_df = data_perturbation(trade_symbols)
    print(f"  Done in {time.time()-t0:.1f}s")
    if not pert_df.empty:
        print(pert_df[["label","trades","avg_r","pf","max_dd_r"]].to_string(index=False))

    # 3. Slippage stress
    print("\n[3/3] Slippage stress...")
    slip_df = slippage_stress(trades)
    print(slip_df[["extra_cost_label","avg_r","pf","max_dd_r"]].to_string(index=False))

    # Write outputs
    mc_summary.to_csv(OUT / "phase_t16_monte_carlo_summary.csv", index=False)
    mc_dist.to_csv(OUT / "phase_t16_monte_carlo_distribution.csv", index=False)
    if not pert_df.empty:
        pert_df.to_csv(OUT / "phase_t16_perturbation_summary.csv", index=False)
    slip_df.to_csv(OUT / "phase_t16_slippage_stress_summary.csv", index=False)
    write_report(mc_summary, pert_df, slip_df, n_trades)

    print("\n[OK] phase_t16_monte_carlo_summary.csv")
    print("[OK] phase_t16_monte_carlo_distribution.csv")
    print("[OK] phase_t16_perturbation_summary.csv  (if OHLCV available)")
    print("[OK] phase_t16_slippage_stress_summary.csv")
    print("[OK] phase_t16_robustness_report.txt")
    print("\nNext step: run phase_t18_volatility_expansion_gate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
