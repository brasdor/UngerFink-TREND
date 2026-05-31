#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T17.5 — DONCHIAN RISK SIZING VARIANTS (THEORETICAL)
==========================================================

Tests four risk-per-trade variants against the frozen ExitV2 config
(DonchianLong_UniverseV2_ExitV2 / max8 / 24 symbols).

Variants:
  A: 0.50% fixed per trade  (2× current)
  B: 0.75% fixed per trade  (3× current)
  C: volatility-adjusted    risk = 0.25% × (median_ATR% / current_ATR%), cap 2%
  D: quarter Kelly          f = (edge/odds) × 0.25, cap 2%

For each variant:
  - T6 capital simulation  ($10k, kill-switch -35%)
  - T16 Monte Carlo        (5000 runs, block sizes [1,5,10,20,50])

Flags:
  - Kill-switch fires in T6
  - MC p05 DD < -15% at any block size

ALL RESULTS MARKED THEORETICAL.
Implementation requires T9B paper trading ≥ 3 months.

Input:   data/research_donchian_exitV2_combined/C2_trades.csv
Output:  data/research_donchian_riskV2_{A,B,C,D}/
         data/research_donchian_riskV2_comparison.csv
         data/research_donchian_riskV2_comparison.txt
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()

TRADES_FILE   = ROOT / "data" / "research_donchian_exitV2_combined" / "C2_trades.csv"
UNIVERSE_FILE = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"
OHLCV_CACHE   = ROOT / "data" / "research_donchian_riskV2_C" / "ohlcv_1d"
COMP_CSV      = ROOT / "data" / "research_donchian_riskV2_comparison.csv"
COMP_TXT      = ROOT / "data" / "research_donchian_riskV2_comparison.txt"

# ─── config ───────────────────────────────────────────────────────────────────
PORTFOLIO_CAP    = 8
INITIAL_CAPITAL  = 10_000.0
KILL_SWITCH_DD   = 0.35        # -35% closed-equity drawdown
MC_RUNS          = 5000
MC_BLOCK_SIZES   = [1, 5, 10, 20, 50]
MC_SEED          = 42
ATR_N            = 14

BASELINE_RISK    = 0.0025      # 0.25% frozen config reference
VOL_ADJ_BASE     = 0.0025
VOL_ADJ_CAP      = 0.020
KELLY_CAP        = 0.020
MC_DD_WARN       = -15.0       # flag if MC p05 DD < this value (%)

EPS = 1e-12

# ─── indicators ───────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


# ─── portfolio cap ────────────────────────────────────────────────────────────

def apply_portfolio_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Replay trades chronologically; accept if concurrent open < cap."""
    df = df.sort_values("entry_time").reset_index(drop=True)
    accepted_rows = []
    open_exits: list[pd.Timestamp] = []

    for _, row in df.iterrows():
        et: pd.Timestamp = row["entry_time"]
        # purge positions that exited before this entry
        open_exits = [x for x in open_exits if x > et]
        if len(open_exits) < cap:
            accepted_rows.append(row)
            open_exits.append(row["exit_time"])

    return pd.DataFrame(accepted_rows).reset_index(drop=True)


# ─── T6 simulation ────────────────────────────────────────────────────────────

def simulate_t6(
    trades: pd.DataFrame,
    risk_pcts: np.ndarray,
    initial: float = INITIAL_CAPITAL,
    ks_dd: float = KILL_SWITCH_DD,
) -> tuple[pd.DataFrame, bool]:
    """
    Event-driven closed-equity R-based capital simulation.

    Multiple trades can be open simultaneously (from portfolio cap replay).
    At entry  : reserve risk_amount = current_closed_equity × risk_pct
    At exit   : realise pnl = risk_amount × net_r; update closed equity
    Kill-switch fires when closed_equity drawdown from peak ≥ ks_dd.
    """
    events: list[tuple] = []
    for i, row in trades.iterrows():
        events.append((row["entry_time"], 1, int(i)))   # entries second
        events.append((row["exit_time"],  0, int(i)))   # exits first

    events.sort(key=lambda x: (x[0], x[1]))

    closed_eq   = initial
    peak_eq     = initial
    kill_active = False
    open_risk: dict[int, float] = {}  # trade_idx → committed risk_amount
    results: list[dict] = []

    for _ts, evtype, idx in events:
        if kill_active:
            break
        row = trades.iloc[idx]

        if evtype == 1:          # ── entry ──
            risk_amt = closed_eq * float(risk_pcts[idx])
            open_risk[idx] = risk_amt

        else:                    # ── exit ──
            if idx not in open_risk:
                continue
            risk_amt = open_risk.pop(idx)
            pnl      = risk_amt * float(row["net_r"])
            closed_eq += pnl
            peak_eq   = max(peak_eq, closed_eq)
            dd_pct    = (closed_eq - peak_eq) / max(peak_eq, EPS) * 100.0

            results.append({
                "symbol":     row["symbol"],
                "entry_time": row["entry_time"],
                "exit_time":  row["exit_time"],
                "net_r":      row["net_r"],
                "risk_pct":   float(risk_pcts[idx]),
                "risk_amount": risk_amt,
                "pnl_usdt":   pnl,
                "equity":     closed_eq,
                "peak_equity": peak_eq,
                "dd_pct":     dd_pct,
            })

            if dd_pct <= -ks_dd * 100.0:
                kill_active = True

    return pd.DataFrame(results), kill_active


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def run_mc(
    net_r: np.ndarray,
    risk_pcts: np.ndarray,
    n_runs: int  = MC_RUNS,
    block_sizes: list = MC_BLOCK_SIZES,
    initial: float   = INITIAL_CAPITAL,
    ks_dd: float     = KILL_SWITCH_DD,
    seed: int        = MC_SEED,
) -> pd.DataFrame:
    """
    Block-bootstrap Monte Carlo.
    Resamples (net_r, risk_pct) pairs together to preserve vol-adjusted sizing.
    """
    rng = np.random.default_rng(seed)
    n   = len(net_r)
    records: list[dict] = []

    for bs in block_sizes:
        for _ in range(n_runs):
            # block resample
            n_blocks = math.ceil(n / bs)
            starts   = rng.integers(0, n, size=n_blocks)
            idx_list: list[int] = []
            for s in starts:
                idx_list.extend(range(int(s), min(int(s) + bs, n)))
            idx_arr  = np.array(idx_list[:n], dtype=int)

            boot_r  = net_r[idx_arr]
            boot_rp = risk_pcts[idx_arr]

            # simulate
            eq    = initial
            peak  = initial
            max_dd = 0.0
            kill  = False

            for r, rp in zip(boot_r, boot_rp):
                if kill:
                    break
                risk_amt = eq * rp
                eq      += risk_amt * r
                peak     = max(peak, eq)
                dd       = (eq - peak) / max(peak, EPS) * 100.0
                if dd < max_dd:
                    max_dd = dd
                if dd <= -ks_dd * 100.0:
                    kill = True

            records.append({
                "block_size":   bs,
                "final_equity": eq,
                "return_pct":   (eq / initial - 1.0) * 100.0,
                "max_dd_pct":   max_dd,
                "kill_switch":  kill,
            })

    return pd.DataFrame(records)


# ─── ATR data ─────────────────────────────────────────────────────────────────

def fetch_atr_lookup(
    symbols: list[str],
    accepted: pd.DataFrame,
) -> tuple[dict, float]:
    """
    Download / load 1D OHLCV for each symbol, compute ATR(14)%.
    Returns:
      atr_lookup : dict {(symbol, date_str) -> atr_pct}
      median_atr_pct : float (universe-wide median)
    Falls back gracefully if ccxt unavailable or download fails.
    """
    OHLCV_CACHE.mkdir(parents=True, exist_ok=True)

    try:
        import ccxt  # type: ignore
        HAS_CCXT = True
    except ImportError:
        HAS_CCXT = False
        print("  [WARN] ccxt not installed -- variant C uses median fallback (0.25%)")

    START_MS = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    atr_lookup: dict = {}

    if HAS_CCXT:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        for sym in symbols:
            safe       = sym.replace("/", "_")
            cache_path = OHLCV_CACHE / f"{safe}_1d.csv"

            if cache_path.exists():
                raw = pd.read_csv(cache_path)
            else:
                rows: list = []
                try:
                    # First batch: ~2020-2022
                    batch1 = exchange.fetch_ohlcv(sym, "1d", since=START_MS, limit=1000)
                    if batch1:
                        rows.extend(batch1)
                    time.sleep(0.12)

                    # Second batch: most recent 1000 bars
                    batch2 = exchange.fetch_ohlcv(sym, "1d", limit=1000)
                    if batch2:
                        rows.extend(batch2)
                    time.sleep(0.12)

                except Exception as exc:
                    print(f"  [WARN] {sym}: download error -- {exc}")
                    continue

                if not rows:
                    print(f"  [WARN] {sym}: no OHLCV returned")
                    continue

                raw = pd.DataFrame(
                    rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                raw = raw.drop_duplicates("timestamp").sort_values("timestamp")
                raw.to_csv(cache_path, index=False)

            raw["timestamp"] = pd.to_numeric(raw["timestamp"])
            raw["time"]      = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
            raw              = raw.sort_values("time").reset_index(drop=True)
            raw["atr"]       = compute_atr(raw)
            raw["atr_pct"]   = raw["atr"] / raw["close"].replace(0, np.nan)
            raw["date"]      = raw["time"].dt.date

            date_atr = (
                raw.dropna(subset=["atr_pct"])
                   .set_index("date")["atr_pct"]
                   .to_dict()
            )
            sym_trades = accepted[accepted["symbol"] == sym]
            for _, tr in sym_trades.iterrows():
                d = pd.Timestamp(tr["entry_time"]).date()
                if d in date_atr:
                    atr_lookup[(sym, str(d))] = float(date_atr[d])

    # Compute median across all available values
    vals = [v for v in atr_lookup.values() if np.isfinite(v) and v > 0]
    median_atr_pct = float(np.median(vals)) if vals else 0.02
    return atr_lookup, median_atr_pct


# ─── risk pct computation ─────────────────────────────────────────────────────

def compute_risk_pcts(
    variant: str,
    accepted: pd.DataFrame,
    atr_lookup: dict,
    median_atr_pct: float,
    kelly_f: float,
) -> np.ndarray:
    n = len(accepted)

    if variant == "A":
        return np.full(n, 0.0050)

    if variant == "B":
        return np.full(n, 0.0075)

    if variant == "D":
        return np.full(n, kelly_f)

    # variant C — vol-adjusted
    rp = np.empty(n)
    for i, (_, row) in enumerate(accepted.iterrows()):
        d   = str(pd.Timestamp(row["entry_time"]).date())
        key = (row["symbol"], d)
        if key in atr_lookup and atr_lookup[key] > EPS:
            ratio = median_atr_pct / atr_lookup[key]
            rp[i] = min(VOL_ADJ_BASE * ratio, VOL_ADJ_CAP)
        else:
            rp[i] = VOL_ADJ_BASE   # fallback: same as baseline
    return rp


# ─── write outputs ────────────────────────────────────────────────────────────

def write_variant_outputs(
    variant: str,
    trade_log: pd.DataFrame,
    kill: bool,
    mc_df: pd.DataFrame,
    avg_rp: float,
) -> dict:
    out_dir = ROOT / f"data" / f"research_donchian_riskV2_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trade log
    trade_log.to_csv(out_dir / "phase_t175_trade_log.csv", index=False)

    # Equity curve
    if not trade_log.empty:
        eq_curve = trade_log[["exit_time", "equity", "peak_equity", "dd_pct"]].copy()
        eq_curve.to_csv(out_dir / "phase_t175_equity_curve.csv", index=False)
        final_eq = float(trade_log["equity"].iloc[-1])
        max_dd   = float(trade_log["dd_pct"].min())
        ret_pct  = (final_eq / INITIAL_CAPITAL - 1.0) * 100.0
    else:
        final_eq = INITIAL_CAPITAL
        max_dd   = 0.0
        ret_pct  = 0.0

    # MC summary
    mc_rows: list[dict] = []
    dd_flagged = False
    for bs in MC_BLOCK_SIZES:
        sub      = mc_df[mc_df["block_size"] == bs]
        p05_ret  = float(sub["return_pct"].quantile(0.05))
        p50_ret  = float(sub["return_pct"].quantile(0.50))
        p95_ret  = float(sub["return_pct"].quantile(0.95))
        p05_dd   = float(sub["max_dd_pct"].quantile(0.05))
        prob_pos = float((sub["return_pct"] > 0).mean())
        kill_pct = float(sub["kill_switch"].mean() * 100.0)

        if p05_dd < MC_DD_WARN:
            dd_flagged = True

        mc_rows.append({
            "block_size":    bs,
            "p05_return_pct": p05_ret,
            "p50_return_pct": p50_ret,
            "p95_return_pct": p95_ret,
            "p05_max_dd_pct": p05_dd,
            "prob_positive":  prob_pos,
            "kill_pct":       kill_pct,
        })

    mc_summary = pd.DataFrame(mc_rows)
    mc_summary.to_csv(out_dir / "phase_t175_mc_summary.csv", index=False)

    # Summary txt
    p05_dd_bs50   = mc_rows[-1]["p05_max_dd_pct"]    # block_size=50
    p50_ret_bs50  = mc_rows[-1]["p50_return_pct"]
    prob_pos_bs50 = mc_rows[-1]["prob_positive"]

    lines = [
        f"PHASE T17.5 — RISK SIZING VARIANT {variant}",
        "=" * 52,
        "STATUS: THEORETICAL — requires T9B ≥3 months before implementation",
        f"Date:   2026-05-30",
        "",
        f"Variant description:",
    ]
    desc = {
        "A": "Fixed 0.50% per trade (2× baseline)",
        "B": "Fixed 0.75% per trade (3× baseline)",
        "C": f"Vol-adjusted: 0.25% × (median_ATR% / current_ATR%), cap 2%  [avg={avg_rp*100:.3f}%]",
        "D": f"Quarter Kelly: f=(edge/odds)×0.25, cap 2%  [f={avg_rp*100:.3f}%]",
    }
    lines.append(f"  {desc[variant]}")
    lines += [
        "",
        "T6 Capital Simulation ($10,000)",
        "-" * 44,
        f"  Final equity : ${final_eq:>10,.2f}",
        f"  Total return : {ret_pct:>+8.2f}%",
        f"  Max DD       : {max_dd:>+8.2f}%",
        f"  Kill-switch  : {'FIRED  ⚠️' if kill else 'no'}",
        "",
        "T16 Monte Carlo (5000 runs, block bootstrap)",
        "-" * 44,
        f"  {'bs':>4}  {'p05_ret':>9}  {'p50_ret':>9}  {'p95_ret':>9}  {'p05_DD':>8}  {'prob+':>6}  {'kill%':>6}",
    ]
    for r in mc_rows:
        dd_warn = " ⚠️" if r["p05_max_dd_pct"] < MC_DD_WARN else ""
        lines.append(
            f"  {r['block_size']:>4}  "
            f"{r['p05_return_pct']:>+9.1f}%  "
            f"{r['p50_return_pct']:>+9.1f}%  "
            f"{r['p95_return_pct']:>+9.1f}%  "
            f"{r['p05_max_dd_pct']:>+8.1f}%{dd_warn:<4} "
            f"{r['prob_positive']:>5.1%}  "
            f"{r['kill_pct']:>5.1f}%"
        )
    if kill:
        lines.append("\n⚠️  KILL-SWITCH FIRED in T6 — do not implement this variant")
    if dd_flagged:
        lines.append("\n⚠️  MC p05 DD EXCEEDS -15% — significant tail risk")
    lines.append("\n⚠️  THEORETICAL — implement only after T9B ≥3 months")

    (out_dir / "phase_t175_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Return row for comparison table
    return {
        "variant":         f"Variant {variant}",
        "risk_description": desc[variant].split(":")[1].strip() if ":" in desc[variant] else desc[variant],
        "avg_risk_pct":    f"{avg_rp*100:.3f}%",
        "T6_return_pct":   round(ret_pct, 2),
        "T6_max_dd_pct":   round(max_dd, 2),
        "T6_kill_switch":  "FIRED ⚠️" if kill else "no",
        "MC_p05_DD_bs50":  round(p05_dd_bs50, 2),
        "MC_p50_ret_bs50": round(p50_ret_bs50, 2),
        "MC_prob_pos_bs50": f"{prob_pos_bs50:.1%}",
        "flags":           " | ".join(filter(None, [
            "KILL-SWITCH" if kill else "",
            "DD>-15%" if dd_flagged else "",
        ])) or "OK",
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("PHASE T17.5 -- DONCHIAN RISK SIZING VARIANTS (THEORETICAL)")
    print("=" * 70)

    # ── load trades ──────────────────────────────────────────────────────────
    if not TRADES_FILE.exists():
        print(f"[ERROR] Missing: {TRADES_FILE}")
        return 1

    raw = pd.read_csv(TRADES_FILE)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"], utc=True)
    raw["exit_time"]  = pd.to_datetime(raw["exit_time"],  utc=True)
    print(f"Raw C2 trades:     {len(raw)}")

    # ── portfolio cap ─────────────────────────────────────────────────────────
    accepted = apply_portfolio_cap(raw, PORTFOLIO_CAP)
    print(f"Max{PORTFOLIO_CAP} filtered:     {len(accepted)} trades accepted")
    print(f"Symbols:           {accepted['symbol'].nunique()}")
    print(f"Date range:        {accepted['entry_time'].min().date()} to {accepted['exit_time'].max().date()}")

    symbols = accepted["symbol"].unique().tolist()
    net_r   = np.array(accepted["net_r"], dtype=float)

    # ── variant C: ATR data ───────────────────────────────────────────────────
    print("\n[VAR C] Fetching 1D ATR data for vol-adjusted sizing...")
    atr_lookup, median_atr_pct = fetch_atr_lookup(symbols, accepted)
    coverage = sum(
        1 for _, row in accepted.iterrows()
        if (row["symbol"], str(pd.Timestamp(row["entry_time"]).date())) in atr_lookup
    )
    print(f"  ATR coverage : {coverage}/{len(accepted)} trades ({coverage/len(accepted):.1%})")
    print(f"  Median ATR%  : {median_atr_pct:.4f} ({median_atr_pct*100:.2f}% of price)")

    # ── variant D: Kelly ──────────────────────────────────────────────────────
    winners  = net_r[net_r > 0]
    losers   = np.abs(net_r[net_r < 0])
    win_rate = len(winners) / max(len(net_r), 1)
    avg_w    = winners.mean() if len(winners) > 0 else 1.0
    avg_l    = losers.mean()  if len(losers) > 0  else 1.0
    edge     = net_r.mean()
    odds     = avg_w / max(avg_l, EPS)
    kelly_f  = max(0.0, (edge / odds) * 0.25) if odds > EPS and edge > 0 else BASELINE_RISK
    kelly_f  = min(kelly_f, KELLY_CAP)

    print(f"\n[VAR D] Kelly computation:")
    print(f"  Accepted trades: {len(net_r)} | win_rate={win_rate:.3f}")
    print(f"  avg_winner={avg_w:.3f}R  avg_loser={avg_l:.3f}R")
    print(f"  edge={edge:.4f}R  odds={odds:.4f}")
    print(f"  quarter Kelly f = {kelly_f:.4f}  ({kelly_f*100:.3f}%)")

    # ── run each variant ──────────────────────────────────────────────────────
    comparison_rows: list[dict] = []

    # Baseline reference (0.25% fixed)
    print("\n[BASELINE] Simulating reference config (0.25%)...")
    rp_base        = np.full(len(accepted), BASELINE_RISK)
    tl_base, kb    = simulate_t6(accepted, rp_base)
    ret_base       = (float(tl_base["equity"].iloc[-1]) / INITIAL_CAPITAL - 1) * 100 if not tl_base.empty else 0.0
    dd_base        = float(tl_base["dd_pct"].min()) if not tl_base.empty else 0.0
    comparison_rows.append({
        "variant":         "Baseline (0.25%)",
        "risk_description": "Current frozen config",
        "avg_risk_pct":    "0.250%",
        "T6_return_pct":   round(ret_base, 2),
        "T6_max_dd_pct":   round(dd_base, 2),
        "T6_kill_switch":  "FIRED ⚠️" if kb else "no",
        "MC_p05_DD_bs50":  "—",
        "MC_p50_ret_bs50": "—",
        "MC_prob_pos_bs50": "—",
        "flags":           "KILL-SWITCH" if kb else "reference",
    })

    for var in ["A", "B", "C", "D"]:
        print(f"\n[VAR {var}] Computing risk pcts...")
        rp = compute_risk_pcts(var, accepted, atr_lookup, median_atr_pct, kelly_f)
        avg_rp = float(rp.mean())
        print(f"  avg_risk_pct = {avg_rp*100:.3f}%  "
              f"min={rp.min()*100:.3f}%  max={rp.max()*100:.3f}%")

        print(f"[VAR {var}] Running T6 simulation...")
        trade_log, kill = simulate_t6(accepted, rp)

        print(f"[VAR {var}] Running MC ({MC_RUNS} runs x {len(MC_BLOCK_SIZES)} block sizes)...")
        mc_df = run_mc(net_r, rp)

        print(f"[VAR {var}] Writing outputs...")
        row = write_variant_outputs(var, trade_log, kill, mc_df, avg_rp)
        comparison_rows.append(row)

        # Quick summary
        t6_ret = row["T6_return_pct"]
        t6_dd  = row["T6_max_dd_pct"]
        mc_p05 = row["MC_p05_DD_bs50"]
        print(f"  -> T6: return={t6_ret:+.2f}%  DD={t6_dd:.2f}%  kill={'YES' if kill else 'no'}")
        print(f"  -> MC p05_DD(bs=50): {mc_p05}%")

    # ── comparison table ──────────────────────────────────────────────────────
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(COMP_CSV, index=False)

    header = [
        "PHASE T17.5 -- RISK SIZING COMPARISON TABLE",
        "=" * 70,
        "Config:  DonchianLong_UniverseV2_ExitV2 (frozen ExitV2)",
        "Input:   C2_trades.csv / max8 portfolio cap / 24 symbols",
        "Capital: $10,000 / Kill-switch: -35% / MC: 5000 runs",
        "Date:    2026-05-30",
        "",
        "[!] ALL RESULTS THEORETICAL",
        "    Implementation order: 0.25% -> 0.50% -> 0.75% (never jump)",
        "    Each step requires T9B paper confirmation before going live",
        "",
    ]

    # pretty table
    cols = [
        ("variant",         "Variant",        16),
        ("avg_risk_pct",    "Risk/trade",     10),
        ("T6_return_pct",   "T6 Ret%",        10),
        ("T6_max_dd_pct",   "T6 DD%",          8),
        ("T6_kill_switch",  "Kill-sw",        12),
        ("MC_p05_DD_bs50",  "MC p05DD(bs50)", 15),
        ("MC_prob_pos_bs50","MC prob+(bs50)",  15),
        ("flags",           "Flags",          18),
    ]
    header_line = "  ".join(f"{h:<{w}}" for _, h, w in cols)
    sep_line    = "  ".join("-" * w for _, _, w in cols)
    rows_lines  = []
    for _, r in comp_df.iterrows():
        rows_lines.append("  ".join(
            f"{str(r.get(k,'')):<{w}}" for k, _, w in cols
        ))

    txt_lines = header + [header_line, sep_line] + rows_lines + [
        "",
        "Column definitions:",
        "  T6 Ret%    : total return on $10k over full backtest",
        "  T6 DD%     : max closed-equity drawdown in T6 simulation",
        "  Kill-sw    : whether -35% DD kill-switch fires in T6",
        "  MC p05DD   : 5th-percentile max drawdown across 5000 MC runs (block=50)",
        "  MC prob+   : fraction of 5000 MC runs with positive return (block=50)",
        "  Flags      : OK / KILL-SWITCH / DD>-15%",
        "",
        "Variant C sizing formula:",
        "  risk_pct = 0.25% × (median_ATR% / current_ATR%)  capped at 2%",
        f"  median_ATR% = {median_atr_pct:.4f}",
        "",
        "Variant D Kelly formula:",
        f"  f = (edge/odds) × 0.25  = ({edge:.4f}/{odds:.4f}) × 0.25 = {kelly_f:.4f}",
        f"  edge = avg_r = {edge:.4f}R  |  odds = avg_win/avg_loss = {odds:.4f}",
        "",
        "⚠️  Do not apply any risk sizing increase until T9B has been running ≥3 months.",
    ]
    COMP_TXT.write_text("\n".join(txt_lines), encoding="utf-8")

    # Print table to stdout
    print("\n")
    print("\n".join(header))
    print(header_line)
    print(sep_line)
    for line in rows_lines:
        print(line)
    print(f"\nComparison saved -> {COMP_CSV}")
    print(f"                 -> {COMP_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
