#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T17 — EMA200 REGIME FILTER  [SUPERSEDED — see T1 redesign]

STATUS: DEPRECATED as a standalone phase.

REASON: The redesigned phase_t1_trend_discovery.py (2026-05-26) now tests BOTH
"ema50_slope" and "ema200_price" filter modes simultaneously across all timeframes
as part of the core concept discovery grid. The T1 stability_report.txt already
recommends which filter is more stable for the canonical timeframe.

T17 is therefore redundant and should NOT be run in the new pipeline.
The EMA200 comparison is done in T1, not T17.

IF you need a post-T3B filter comparison (using the full Chandelier exit rather
than T1's simple exit), that belongs in a new T17B phase — but only AFTER T15
(post-exit stability) has confirmed the canonical entry parameters.

────────────────────────────────────────────────────────────────────────────────
ORIGINAL DOCSTRING (kept for reference):
────────────────────────────────────────────────────────────────────────────────
Unger principle §3.1 (from recap):
  "A 200-period moving average can be used as a market-regime filter:
   only allow long trades when price is above the MA, only allow shorts
   when below. This stabilizes the equity curve significantly, even though
   it generates more signals than a slope-based or crossover approach."

This phase originally tested:
  VAR_A  — EMA50 slope > 0           (frozen T9A config — baseline)
  VAR_B  — price > EMA200            (Unger §3.1 alternative)
  VAR_C  — EMA50 slope > 0 AND price > EMA200  (both combined)
  VAR_D  — No filter                 (lower bound reference)

All other parameters frozen at canonical T8 config:
  Donchian N=20, ATR14, INIT_STOP=2×ATR, Chandelier ACT4_ATR3.

OHLCV source: T13 ohlcv_cache (6H), falls back to T10 cache.

Outputs: data/research_trend_t17/
  phase_t17_regime_filter_trades.csv
  phase_t17_regime_filter_summary.csv
  phase_t17_regime_filter_asset_summary.csv
  phase_t17_regime_filter_report.txt
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path.cwd()
OUT        = ROOT / "data" / "research_trend_t17"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_T13  = ROOT / "data" / "research_trend_t13" / "ohlcv_cache"
CACHE_T10  = ROOT / "data" / "research_trend_t10" / "ohlcv_cache"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    "VAR_D_NO_FILTER",            # raw Donchian — lower bound reference
    "VAR_A_EMA50_SLOPE",          # current T9A (baseline)
    "VAR_B_EMA200_PRICE",         # Unger §3.1 alternative
    "VAR_C_EMA50_AND_EMA200",     # both combined
]

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN PARAMS (all from T8 frozen config)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME        = "6h"
DONCHIAN_N       = 20
EMA_SLOPE_N      = 50      # for VAR_A and VAR_C
EMA_SLOPE_LB     = 10
EMA_REGIME_N     = 200     # for VAR_B and VAR_C
ATR_N            = 14
INIT_STOP_ATR    = 2.0
ACT_R            = 4.0
CH_MULT          = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_sym(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


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


def _stats(r: np.ndarray, label: str) -> dict:
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(variant=label, trades=0, total_r=0.0, avg_r=0.0,
                    pf=0.0, max_dd_r=0.0, win_rate_pct=0.0,
                    median_r=0.0, best_r=0.0, worst_r=0.0)
    return dict(
        variant=label,
        trades=len(r),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        median_r=float(np.median(r)),
        pf=_pf(r),
        max_dd_r=_dd_r(r),
        win_rate_pct=float((r > 0).mean() * 100),
        best_r=float(r.max()),
        worst_r=float(r.min()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time","open","high","low","close"])
        df = df.sort_values("time").reset_index(drop=True)
        now = pd.Timestamp.utcnow()
        df["close_time"] = df["time"] + pd.Timedelta(hours=6)
        df = df[df["close_time"] <= now].drop(columns=["close_time"])
        return df if len(df) >= 250 else None
    except Exception:
        return None


def _load_sym(sym: str) -> pd.DataFrame | None:
    fname = f"{_safe_sym(sym)}_{TIMEFRAME}.csv"
    for cache in (CACHE_T13, CACHE_T10):
        df = _load_csv(cache / fname)
        if df is not None:
            return df
    return None


def load_all() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    # Collect from T13 cache first
    for cache in (CACHE_T13, CACHE_T10):
        if not cache.exists():
            continue
        for f in cache.glob(f"*_{TIMEFRAME}.csv"):
            stem = f.stem
            base_quote = stem[: -len(f"_{TIMEFRAME}")]
            if "_USDT" not in base_quote:
                continue
            sym = base_quote.replace("_USDT", "/USDT", 1)
            if sym not in result:
                df = _load_csv(f)
                if df is not None:
                    result[sym] = df
        if result:
            break
    print(f"Loaded {len(result)} symbols with 6H OHLCV.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # ATR
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-pc).abs(), (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"]          = tr.rolling(ATR_N).mean()
    # Donchian (shifted — no lookahead)
    d["don_high"]     = d["high"].shift(1).rolling(DONCHIAN_N).max()
    d["don_low"]      = d["low"].shift(1).rolling(DONCHIAN_N).min()
    # EMA50 + slope
    d["ema50"]        = d["close"].ewm(span=EMA_SLOPE_N, adjust=False).mean()
    d["ema50_slope"]  = d["ema50"] - d["ema50"].shift(EMA_SLOPE_LB)
    # EMA200
    d["ema200"]       = d["close"].ewm(span=EMA_REGIME_N, adjust=False).mean()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST (single symbol, variant determines entry filter)
# ─────────────────────────────────────────────────────────────────────────────

def _entry_allowed(row: pd.Series, side: str, variant: str) -> bool:
    if variant == "VAR_D_NO_FILTER":
        return True

    slope = float(row.get("ema50_slope", np.nan))
    ema200 = float(row.get("ema200", np.nan))
    close  = float(row.get("close", np.nan))

    if side == "LONG":
        if variant == "VAR_A_EMA50_SLOPE":
            return np.isfinite(slope) and slope > 0
        if variant == "VAR_B_EMA200_PRICE":
            return np.isfinite(ema200) and np.isfinite(close) and close > ema200
        if variant == "VAR_C_EMA50_AND_EMA200":
            return (np.isfinite(slope) and slope > 0 and
                    np.isfinite(ema200) and np.isfinite(close) and close > ema200)
    else:  # SHORT
        if variant == "VAR_A_EMA50_SLOPE":
            return np.isfinite(slope) and slope < 0
        if variant == "VAR_B_EMA200_PRICE":
            return np.isfinite(ema200) and np.isfinite(close) and close < ema200
        if variant == "VAR_C_EMA50_AND_EMA200":
            return (np.isfinite(slope) and slope < 0 and
                    np.isfinite(ema200) and np.isfinite(close) and close < ema200)
    return False


def _backtest(sym: str, df: pd.DataFrame, variant: str) -> list[dict]:
    d = _add_indicators(df)
    # EMA200 needs 200+ bars to warm up
    warmup = max(DONCHIAN_N, EMA_SLOPE_N + EMA_SLOPE_LB, EMA_REGIME_N, ATR_N) + 5
    pos = None
    trades = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        t     = row["time"]

        if pos is not None:
            if pos["side"] == "LONG":
                pos["hh"] = max(pos["hh"], high)
                pos["mfe"] = max(pos["mfe"], (high - pos["entry"]) / pos["risk"])
                if pos["mfe"] >= ACT_R:
                    pos["stop"] = max(pos["stop"], pos["hh"] - CH_MULT * atr)
                    pos["trail"] = True
                if low <= pos["stop"]:
                    r = (pos["stop"] - pos["entry"]) / pos["risk"]
                    trades.append(_trade(sym, pos, t, pos["stop"], r, variant))
                    pos = None
                    continue
            else:
                pos["ll"] = min(pos["ll"], low)
                pos["mfe"] = max(pos["mfe"], (pos["entry"] - low) / pos["risk"])
                if pos["mfe"] >= ACT_R:
                    pos["stop"] = min(pos["stop"], pos["ll"] + CH_MULT * atr)
                    pos["trail"] = True
                if high >= pos["stop"]:
                    r = (pos["entry"] - pos["stop"]) / pos["risk"]
                    trades.append(_trade(sym, pos, t, pos["stop"], r, variant))
                    pos = None
                    continue

        if pos is None:
            dh = row.get("don_high")
            dl = row.get("don_low")
            if not (np.isfinite(float(dh)) and np.isfinite(float(dl))):
                continue

            long_sig  = close > float(dh)
            short_sig = close < float(dl)

            if long_sig and not short_sig and _entry_allowed(row, "LONG", variant):
                risk = atr * INIT_STOP_ATR
                if risk > 0:
                    pos = dict(side="LONG", entry_time=t, entry=close,
                               stop=close-risk, risk=risk, hh=close, ll=close,
                               mfe=0.0, trail=False)
            elif short_sig and not long_sig and _entry_allowed(row, "SHORT", variant):
                risk = atr * INIT_STOP_ATR
                if risk > 0:
                    pos = dict(side="SHORT", entry_time=t, entry=close,
                               stop=close+risk, risk=risk, hh=close, ll=close,
                               mfe=0.0, trail=False)

    return trades


def _trade(sym, pos, exit_time, exit_px, net_r, variant):
    return {
        "variant":      variant,
        "symbol":       sym,
        "timeframe":    TIMEFRAME,
        "side":         pos["side"],
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry"],
        "exit_price":   exit_px,
        "initial_risk_per_unit": pos["risk"],
        "net_r":        float(net_r),
        "max_favorable_r": pos["mfe"],
        "chandelier_active": pos["trail"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-ASSET SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, sym), g in trades_df.groupby(["variant","symbol"]):
        r = g["net_r"].values.astype(float)
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        rows.append({
            "variant": variant, "symbol": sym,
            "trades": len(r), "total_r": float(r.sum()),
            "avg_r": float(r.mean()), "pf": _pf(r),
            "win_rate_pct": float((r > 0).mean() * 100),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(summary: pd.DataFrame, n_syms: int) -> None:
    lines = [
        "PHASE T17 — EMA200 REGIME FILTER",
        "=" * 80,
        "",
        "Unger §3.1: 200-period MA as market-regime filter.",
        "Hypothesis: price > EMA200 stabilizes equity curve vs. EMA50 slope.",
        "",
        f"Symbols tested:  {n_syms}",
        f"Timeframe:       {TIMEFRAME}",
        f"Frozen exits:    Chandelier ACT{ACT_R:g}_ATR{CH_MULT:g}",
        "",
        "RESULTS",
        "-" * 80,
    ]

    for _, row in summary.sort_values("pf", ascending=False).iterrows():
        is_baseline = "  ← CURRENT T9A" if row["variant"] == "VAR_A_EMA50_SLOPE" else ""
        lines.append(
            f"  {row['variant']:35s}  trades={int(row['trades']):4d}  "
            f"avg_r={row['avg_r']:+.4f}  pf={row['pf']:.2f}  "
            f"dd={row['max_dd_r']:+.2f}  win%={row['win_rate_pct']:.1f}{is_baseline}"
        )

    # Find best variant
    best = summary.sort_values("pf", ascending=False).iloc[0] if not summary.empty else None
    lines += [""]

    if best is not None:
        lines += [
            f"Best variant by PF: {best['variant']}",
            f"  trades={int(best['trades'])}  avg_r={best['avg_r']:+.4f}  "
            f"pf={best['pf']:.2f}  dd={best['max_dd_r']:+.2f}",
            "",
        ]

    lines += [
        "DECISION RULES",
        "-" * 40,
        "  Compare VAR_A (baseline) vs VAR_B and VAR_C:",
        "",
        "  KEEP EMA50 slope (VAR_A): if VAR_A has higher PF or fewer correlated losses.",
        "  UPGRADE to EMA200 (VAR_B): if PF improves by ≥ 0.10 with trade count ≥ 80%.",
        "  COMBINE (VAR_C): if max_dd_R improves significantly without destroying trades.",
        "  NO FILTER (VAR_D) is reference only — never deploy without a regime filter.",
        "",
        "  Per Unger: EMA200 generates MORE signals than slope filter (less filtering).",
        "  Fewer trades is NOT automatically better in TF systems.",
        "",
        "  Do NOT update T9A without running T16 robustness on the chosen variant.",
        "  Any filter change requires a new T8-equivalent freeze before T9A update.",
    ]

    (OUT / "phase_t17_regime_filter_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("PHASE T17 — EMA200 REGIME FILTER")
    print("=" * 80)
    print(f"Output: {OUT}")
    print(f"Variants: {VARIANTS}")

    data = load_all()
    if not data:
        raise RuntimeError("No 6H OHLCV found. Run T13 or T10 first.")

    all_trades: list[dict] = []
    summary_rows = []

    for variant in VARIANTS:
        print(f"\n[VARIANT] {variant}")
        variant_trades: list[dict] = []

        for sym, df in data.items():
            t = _backtest(sym, df, variant)
            variant_trades.extend(t)

        r_arr = np.array([t["net_r"] for t in variant_trades], dtype=float)
        s = _stats(r_arr, variant)
        summary_rows.append(s)
        all_trades.extend(variant_trades)
        print(f"  trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
              f"pf={s['pf']:.2f}  dd={s['max_dd_r']:+.2f}  win%={s['win_rate_pct']:.1f}%")

    summary  = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"],  utc=True, errors="coerce")
        trades_df = trades_df.sort_values(["variant","entry_time"]).reset_index(drop=True)

    asset_df = asset_summary(trades_df) if not trades_df.empty else pd.DataFrame()

    # Write outputs
    trades_df.to_csv(OUT / "phase_t17_regime_filter_trades.csv", index=False)
    summary.to_csv(OUT / "phase_t17_regime_filter_summary.csv", index=False)
    asset_df.to_csv(OUT / "phase_t17_regime_filter_asset_summary.csv", index=False)
    write_report(summary, len(data))

    print("\nT17 SUMMARY")
    cols = ["variant","trades","avg_r","pf","max_dd_r","win_rate_pct"]
    print(summary[cols].sort_values("pf", ascending=False).to_string(index=False))

    print("\n[OK] phase_t17_regime_filter_trades.csv")
    print("[OK] phase_t17_regime_filter_summary.csv")
    print("[OK] phase_t17_regime_filter_asset_summary.csv")
    print("[OK] phase_t17_regime_filter_report.txt")
    print("\nNext step: run phase_t18_volatility_expansion_gate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
