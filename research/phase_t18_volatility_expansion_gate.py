#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T18 — VOLATILITY EXPANSION GATE

Offline only. T9A paper-live remains frozen.

Unger sub-archetype §2.6 (from recap):
  "Trading on the Nasdaq: a trading system based on a trend-following breakout
   logic exploiting volatility explosions, using an ATR-based strategy."
  "Enter on breakout only after a contraction phase (low ATR)."

This converts the plain Donchian breakout (T9A) into a SQUEEZE BREAKOUT:
  Only fire the entry signal when the current ATR is below the Nth percentile
  of the trailing ATR window — i.e., after a period of low volatility.

Rationale:
  Low ATR → contraction phase → compression before explosion.
  Classic Donchian fires on every breakout (including volatile, choppy ones).
  Adding a "low-ATR gate" aims to capture clean squeeze breakouts only.

Variants:
  VAR_BASE  — no ATR gate (current T9A baseline)
  VAR_P20   — entry only when ATR < 20th percentile of rolling {WINDOW}-bar ATR
  VAR_P33   — entry only when ATR < 33rd percentile
  VAR_P50   — entry only when ATR < 50th percentile  (median — widest gate)

All other parameters frozen at canonical T8 config:
  Donchian N=20, EMA50 slope filter (VAR_A from T17), ATR14, INIT_STOP=2×ATR,
  Chandelier ACT4_ATR3.

Note: Using EMA50 slope filter (VAR_A) as the regime filter — the T9A baseline.
To combine with T17 winner, substitute the regime filter after reviewing T17.

OHLCV source: T13 ohlcv_cache (6H), falls back to T10 cache.

Outputs: data/research_trend_t18/
  phase_t18_volatility_gate_trades.csv
  phase_t18_volatility_gate_summary.csv
  phase_t18_volatility_gate_asset_summary.csv
  phase_t18_volatility_gate_report.txt
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
OUT        = ROOT / "data" / "research_trend_t18"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_T15  = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"   # 1D data from T15

# ─────────────────────────────────────────────────────────────────────────────
# VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    ("VAR_BASE", None),    # no gate
    ("VAR_P20",  20),      # enter only when ATR < 20th pct of rolling window
    ("VAR_P33",  33),      # enter only when ATR < 33rd pct
    ("VAR_P50",  50),      # enter only when ATR < 50th pct (median gate)
]
ATR_PERCENTILE_WINDOW = 50   # rolling window for ATR percentile calculation

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN PARAMS (T8 canonical)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME      = "1d"
DONCHIAN_N     = 20
EMA_REGIME_LEN = 200           # ema200 price-above (T1 4yr confirmed)
ATR_N          = 14
INIT_STOP_ATR  = 2.0
ACT_R          = 4.0
CH_MULT        = 3.0

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
                    median_r=0.0, best_r=0.0, worst_r=0.0,
                    filter_rate_pct=0.0)
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
        filter_rate_pct=0.0,  # filled in per-variant after counting
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
        df["close_time"] = df["time"] + pd.Timedelta(days=1)
        df = df[df["close_time"] <= now].drop(columns=["close_time"])
        return df if len(df) >= 250 else None
    except Exception:
        return None


def load_all() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if CACHE_T15.exists():
        for f in CACHE_T15.glob(f"*_{TIMEFRAME}.csv"):
            stem = f.stem
            base_quote = stem[: -len(f"_{TIMEFRAME}")]
            if "_USDT" not in base_quote:
                continue
            sym = base_quote.replace("_USDT", "/USDT", 1)
            if sym not in result:
                df = _load_csv(f)
                if df is not None:
                    result[sym] = df
    print(f"Loaded {len(result)} symbols with 1D OHLCV.")
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
    # Rolling ATR percentile (for the squeeze gate)
    d["atr_pct_rank"] = d["atr"].rolling(ATR_PERCENTILE_WINDOW).rank(pct=True) * 100
    # EMA200 price-above regime filter (T8 canonical)
    d["ema200"]       = d["close"].ewm(span=EMA_REGIME_LEN, adjust=False).mean()
    # Donchian entry (shifted)
    d["don_high"]     = d["high"].shift(1).rolling(DONCHIAN_N).max()
    d["don_low"]      = d["low"].shift(1).rolling(DONCHIAN_N).min()
    # Donchian N//2 exit (Turtle rule — active while Chandelier not yet triggered)
    exit_n = max(1, DONCHIAN_N // 2)
    d["don_exit_low"] = d["low"].shift(1).rolling(exit_n).min()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(sym: str, df: pd.DataFrame, variant_label: str,
              pct_threshold: int | None) -> tuple[list[dict], int, int]:
    """
    Returns (trades, n_signals_fired, n_signals_gated_out).
    """
    d = _add_indicators(df)
    warmup = max(DONCHIAN_N, EMA_REGIME_LEN, ATR_N, ATR_PERCENTILE_WINDOW) + 5
    pos = None
    trades = []
    n_fired = 0
    n_gated = 0

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
            pos["hh"] = max(pos["hh"], high)
            pos["mfe"] = max(pos["mfe"], (high - pos["entry"]) / pos["risk"])
            # 1. Stop check uses stop from START of candle
            if low <= pos["stop"]:
                r = (pos["stop"] - pos["entry"]) / pos["risk"]
                trades.append(_trade(sym, pos, t, pos["stop"], r,
                                     variant_label, pct_threshold))
                pos = None
                continue
            # 2. Donchian N//2 exit on close (while Chandelier not yet active)
            if not pos["trail"]:
                don_exit_low = float(row.get("don_exit_low", np.nan))
                if np.isfinite(don_exit_low) and close < don_exit_low:
                    r = (close - pos["entry"]) / pos["risk"]
                    trades.append(_trade(sym, pos, t, close, r,
                                        variant_label, pct_threshold))
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

            long_sig = close > float(dh) and close > float(e200)

            if long_sig:
                # ATR gate check
                atr_rank = float(row.get("atr_pct_rank", np.nan))
                gate_pass = True
                if pct_threshold is not None:
                    if not np.isfinite(atr_rank) or atr_rank >= pct_threshold:
                        gate_pass = False

                if not gate_pass:
                    n_gated += 1
                    continue

                n_fired += 1
                risk = atr * INIT_STOP_ATR
                if risk <= 0:
                    continue

                pos = dict(side="LONG", entry_time=t, entry=close,
                           stop=close-risk, risk=risk, hh=close,
                           mfe=0.0, trail=False, atr_rank=atr_rank)

    return trades, n_fired, n_gated


def _trade(sym, pos, exit_time, exit_px, net_r, variant, pct_thresh):
    return {
        "variant":          variant,
        "atr_gate_pct":     pct_thresh,
        "symbol":           sym,
        "timeframe":        TIMEFRAME,
        "side":             pos["side"],
        "entry_time":       pos["entry_time"],
        "exit_time":        exit_time,
        "entry_price":      pos["entry"],
        "exit_price":       exit_px,
        "initial_risk_per_unit": pos["risk"],
        "net_r":            float(net_r),
        "max_favorable_r":  pos["mfe"],
        "chandelier_active": pos["trail"],
        "atr_rank_at_entry": pos.get("atr_rank", np.nan),
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
        "PHASE T18 — VOLATILITY EXPANSION GATE",
        "=" * 80,
        "",
        "Unger §2.6: enter Donchian breakout only after an ATR contraction phase.",
        "'Low ATR → squeeze → expansion' is the intended trade profile.",
        "",
        f"ATR percentile window: {ATR_PERCENTILE_WINDOW} bars",
        f"Symbols tested:        {n_syms}",
        f"Timeframe:             {TIMEFRAME}",
        f"Regime filter:         EMA200 price-above (T8 canonical — T1 4yr confirmed)",
        "",
        "RESULTS",
        "-" * 80,
    ]

    for _, row in summary.sort_values("pf", ascending=False).iterrows():
        is_baseline = "  ← CURRENT T9A" if row["variant"] == "VAR_BASE" else ""
        gate_str = f"(ATR_rank<{row.get('gate_pct','-')}%)" if row.get('gate_pct') else "(no gate)"
        lines.append(
            f"  {row['variant']:12s} {gate_str:18s}  "
            f"trades={int(row['trades']):4d}  avg_r={row['avg_r']:+.4f}  "
            f"pf={row['pf']:.2f}  dd={row['max_dd_r']:+.2f}  "
            f"win%={row['win_rate_pct']:.1f}  "
            f"filter%={row.get('filter_rate_pct',0):.0f}{is_baseline}"
        )

    best = summary.sort_values("pf", ascending=False).iloc[0] if not summary.empty else None

    lines += [
        "",
        "  'filter_rate_pct' = percentage of raw signals blocked by the ATR gate.",
        "  A high filter rate (>60%) means the gate is very restrictive.",
        "",
        "TRADE PROFILE CHECK (Unger §4.1)",
        "-" * 40,
        "  A healthy TF system has: win_rate 30-45% and avg_winner / avg_loser > 2:1.",
        "  The ATR gate should improve avg_R while keeping win_rate in this range.",
        "",
    ]

    if best is not None:
        lines.append(f"Best variant by PF: {best['variant']}  "
                     f"(trades={int(best['trades'])}, avg_r={best['avg_r']:+.4f}, "
                     f"pf={best['pf']:.2f})")

    lines += [
        "",
        "DECISION RULES",
        "-" * 40,
        "  UPGRADE baseline with ATR gate: if any gate variant improves PF ≥ 0.15",
        "  AND keeps trade count ≥ 50% of baseline AND avg_R improves.",
        "",
        "  PASS with CAUTION: if trades < 50% of baseline, the sample is too thin",
        "  to trust the improvement — the gate may be over-filtering.",
        "",
        "  FAIL: if no gate variant beats VAR_BASE — the squeeze-breakout pattern",
        "  is not present in this crypto universe. Do not add the gate.",
        "",
        "  IMPORTANT: even if a gate variant looks better here, it must survive T16",
        "  (Monte Carlo + data perturbation) before it can be considered for T9A.",
        "",
        "  T9A paper-live remains frozen.",
    ]

    (OUT / "phase_t18_volatility_gate_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("PHASE T18 — VOLATILITY EXPANSION GATE")
    print("=" * 80)
    print(f"Output: {OUT}")
    print(f"ATR percentile window: {ATR_PERCENTILE_WINDOW} bars")
    print(f"Variants: {[v[0] for v in VARIANTS]}")

    data = load_all()
    if not data:
        raise RuntimeError("No 1D OHLCV found in T15 cache. Run phase_t15 first.")

    all_trades: list[dict] = []
    summary_rows = []

    # Track signals fired vs gated per variant
    variant_stats: dict[str, dict] = {}

    for variant_label, pct_thresh in VARIANTS:
        print(f"\n[VARIANT] {variant_label}  gate={pct_thresh}%")
        variant_trades: list[dict] = []
        total_fired = 0
        total_gated = 0

        for sym, df in data.items():
            t, n_fired, n_gated = _backtest(sym, df, variant_label, pct_thresh)
            variant_trades.extend(t)
            total_fired += n_fired
            total_gated += n_gated

        r_arr = np.array([t["net_r"] for t in variant_trades], dtype=float)
        filter_rate = (total_gated / max(total_fired + total_gated, 1)) * 100

        s = _stats(r_arr, variant_label)
        s["filter_rate_pct"] = round(filter_rate, 1)
        s["gate_pct"] = pct_thresh
        s["signals_fired"] = total_fired
        s["signals_gated"] = total_gated
        summary_rows.append(s)
        all_trades.extend(variant_trades)

        print(f"  trades={s['trades']}  avg_r={s['avg_r']:+.4f}  "
              f"pf={s['pf']:.2f}  dd={s['max_dd_r']:+.2f}  "
              f"win%={s['win_rate_pct']:.1f}%  "
              f"gate_filtered={filter_rate:.0f}%")

    summary  = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"],  utc=True, errors="coerce")
        trades_df = trades_df.sort_values(["variant","entry_time"]).reset_index(drop=True)

    asset_df = asset_summary(trades_df) if not trades_df.empty else pd.DataFrame()

    # Write outputs
    trades_df.to_csv(OUT / "phase_t18_volatility_gate_trades.csv", index=False)
    summary.to_csv(OUT / "phase_t18_volatility_gate_summary.csv", index=False)
    asset_df.to_csv(OUT / "phase_t18_volatility_gate_asset_summary.csv", index=False)
    write_report(summary, len(data))

    print("\nT18 SUMMARY")
    cols = ["variant","gate_pct","trades","avg_r","pf","max_dd_r",
            "win_rate_pct","filter_rate_pct"]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].sort_values("pf", ascending=False).to_string(index=False))

    print("\n[OK] phase_t18_volatility_gate_trades.csv")
    print("[OK] phase_t18_volatility_gate_summary.csv")
    print("[OK] phase_t18_volatility_gate_asset_summary.csv")
    print("[OK] phase_t18_volatility_gate_report.txt")
    print("\nWorkflow complete. Review all T15-T18 reports before considering any T9A update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
