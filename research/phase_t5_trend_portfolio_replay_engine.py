#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T5 — TREND PORTFOLIO REPLAY ENGINE

Purpose
-------
Replay the Phase T3B trade list as a realistic portfolio.

This script does NOT change the signal logic.
It takes the already generated T3B trades and applies portfolio constraints:

- max open positions
- one position per asset
- same-side exposure caps
- signal queue when too many entries arrive on the same candle
- optional focus filters: 6h only, long only, short only, or all
- portfolio heat tracking
- closed-trade equity only
- skipped-signals diagnostics

Input
-----
data/research_trend_t3b/phase_t3b_wide_exit_trades.csv

Outputs
-------
data/research_trend_t5/
    phase_t5_portfolio_summary.csv
    phase_t5_portfolio_equity.csv
    phase_t5_portfolio_trades.csv
    phase_t5_portfolio_skipped_signals.csv
    phase_t5_portfolio_open_count_timeline.csv
    phase_t5_master_report.txt

Notes
-----
This is a research replay, not a live engine.
The engine assumes that each T3B trade already contains a valid entry_time and exit_time.
A trade is accepted only if portfolio constraints allow it at its entry_time.
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_TRADES = PROJECT_DIR / "data" / "research_trend_t3b" / "phase_t3b_wide_exit_trades.csv"
OUT_DIR = PROJECT_DIR / "data" / "research_trend_t5"

TIME_COL_ENTRY = "entry_time"
TIME_COL_EXIT = "exit_time"
R_COL = "net_r"

# T5 deliberately tests several portfolio variants.
# Keep this grid small and interpretable.
PORTFOLIO_VARIANTS = [
    # 1D LONG-only variants (Binance Spot: shorts disabled per T3B config)
    # T4 confirmed ZEC/TRX/XRP are fat-tail carriers — higher max_open captures more of them
    {"name": "T5_1D_LONG_max3",  "timeframe": "1d", "side": "LONG", "max_open": 3,  "max_long": 3,  "max_short": 0},
    {"name": "T5_1D_LONG_max5",  "timeframe": "1d", "side": "LONG", "max_open": 5,  "max_long": 5,  "max_short": 0},
    {"name": "T5_1D_LONG_max8",  "timeframe": "1d", "side": "LONG", "max_open": 8,  "max_long": 8,  "max_short": 0},
    {"name": "T5_1D_LONG_max10", "timeframe": "1d", "side": "LONG", "max_open": 10, "max_long": 10, "max_short": 0},
    # Unconstrained baseline: no position cap (upper bound on capturable R)
    {"name": "T5_1D_LONG_uncap", "timeframe": "1d", "side": "LONG", "max_open": 99, "max_long": 99, "max_short": 0},
]

# If multiple signals occur on the same timestamp, rank them.
# Good simple ranking for trend systems: prefer higher MFE potential observed in research?
# For pure replay we cannot know future MFE live, so default is neutral ordering by:
# 1) timeframe/side filtered already
# 2) symbol alphabetic, deterministic
# A second score can be enabled for research only, but default stays realistic.
USE_RESEARCH_FUTURE_RANKING = False

# Risk assumptions for portfolio heat reporting only.
# Equity is still measured in R.
RISK_PER_TRADE_PCT = 0.25  # example: 0.25% per accepted trade


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_time_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def normalize_side(x: str) -> str:
    x = str(x).upper().strip()
    if x in {"BUY", "LONG"}:
        return "LONG"
    if x in {"SELL", "SHORT"}:
        return "SHORT"
    return x


def max_drawdown_r(r_values: np.ndarray) -> float:
    arr = np.asarray(r_values, dtype=float)
    if arr.size == 0:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def profit_factor(r_values: np.ndarray) -> float:
    arr = np.asarray(r_values, dtype=float)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def losing_streak(r_values: np.ndarray) -> int:
    max_streak = 0
    cur = 0
    for r in r_values:
        if r < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return int(max_streak)


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input trades not found: {path}")

    df = pd.read_csv(path)

    required = ["symbol", "timeframe", "side", TIME_COL_ENTRY, TIME_COL_EXIT, R_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input trades: {missing}")

    df[TIME_COL_ENTRY] = parse_time_series(df[TIME_COL_ENTRY])
    df[TIME_COL_EXIT] = parse_time_series(df[TIME_COL_EXIT])
    df["side"] = df["side"].map(normalize_side)
    df[R_COL] = pd.to_numeric(df[R_COL], errors="coerce")

    df = df.dropna(subset=[TIME_COL_ENTRY, TIME_COL_EXIT, R_COL, "symbol", "timeframe", "side"]).copy()
    df = df[df[TIME_COL_EXIT] > df[TIME_COL_ENTRY]].copy()

    # stable deterministic ID
    df = df.sort_values([TIME_COL_ENTRY, "timeframe", "symbol", "side", TIME_COL_EXIT]).reset_index(drop=True)
    df["source_trade_id"] = np.arange(len(df), dtype=int)
    return df


def filter_variant(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    tf = cfg.get("timeframe", "ALL")
    side = cfg.get("side", "ALL")

    if tf != "ALL":
        out = out[out["timeframe"].astype(str).str.lower() == str(tf).lower()].copy()

    if side != "ALL":
        out = out[out["side"] == side].copy()

    return out.sort_values([TIME_COL_ENTRY, "symbol", "side", TIME_COL_EXIT]).reset_index(drop=True)


@dataclass
class OpenPosition:
    source_trade_id: int
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    net_r: float


def count_sides(open_positions: List[OpenPosition]) -> Tuple[int, int]:
    longs = sum(1 for p in open_positions if p.side == "LONG")
    shorts = sum(1 for p in open_positions if p.side == "SHORT")
    return longs, shorts


def close_due_positions(
    open_positions: List[OpenPosition],
    now: pd.Timestamp,
    accepted_lookup: Dict[int, dict],
    closed_rows: List[dict],
) -> List[OpenPosition]:
    still_open = []
    for p in open_positions:
        if p.exit_time <= now:
            row = accepted_lookup[p.source_trade_id].copy()
            row["portfolio_exit_time"] = p.exit_time
            row["portfolio_net_r"] = p.net_r
            closed_rows.append(row)
        else:
            still_open.append(p)
    return still_open


def can_accept_trade(row: pd.Series, open_positions: List[OpenPosition], cfg: dict) -> Tuple[bool, str]:
    max_open = int(cfg["max_open"])
    max_long = int(cfg["max_long"])
    max_short = int(cfg["max_short"])

    symbol = str(row["symbol"])
    side = normalize_side(row["side"])

    if len(open_positions) >= max_open:
        return False, "MAX_OPEN_REACHED"

    if any(p.symbol == symbol for p in open_positions):
        return False, "ASSET_ALREADY_OPEN"

    longs, shorts = count_sides(open_positions)

    if side == "LONG" and longs >= max_long:
        return False, "MAX_LONG_REACHED"

    if side == "SHORT" and shorts >= max_short:
        return False, "MAX_SHORT_REACHED"

    return True, "ACCEPTED"


def replay_variant(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      accepted_trades_closed_df
      skipped_signals_df
      open_count_timeline_df
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build event timestamps: entry times plus exit times to close positions before evaluating entries.
    entry_times = sorted(df[TIME_COL_ENTRY].unique())

    open_positions: List[OpenPosition] = []
    accepted_lookup: Dict[int, dict] = {}
    closed_rows: List[dict] = []
    skipped_rows: List[dict] = []
    timeline_rows: List[dict] = []

    grouped = dict(tuple(df.groupby(TIME_COL_ENTRY, sort=True)))

    for now in entry_times:
        # Close anything that exits on/before this new entry timestamp.
        open_positions = close_due_positions(open_positions, now, accepted_lookup, closed_rows)

        signals = grouped[now].copy()

        if USE_RESEARCH_FUTURE_RANKING and "mfe_r" in signals.columns:
            signals["_rank"] = pd.to_numeric(signals["mfe_r"], errors="coerce").fillna(0.0)
            signals = signals.sort_values(["_rank", "symbol"], ascending=[False, True])
        else:
            signals = signals.sort_values(["symbol", "side", TIME_COL_EXIT])

        accepted_this_ts = 0
        skipped_this_ts = 0

        for _, row in signals.iterrows():
            ok, reason = can_accept_trade(row, open_positions, cfg)

            base = row.to_dict()
            base["variant"] = cfg["name"]
            base["portfolio_entry_time"] = row[TIME_COL_ENTRY]
            base["open_positions_before_entry"] = len(open_positions)

            if ok:
                p = OpenPosition(
                    source_trade_id=int(row["source_trade_id"]),
                    symbol=str(row["symbol"]),
                    side=normalize_side(row["side"]),
                    entry_time=row[TIME_COL_ENTRY],
                    exit_time=row[TIME_COL_EXIT],
                    net_r=float(row[R_COL]),
                )
                open_positions.append(p)

                base["skip_reason"] = ""
                base["portfolio_status"] = "ACCEPTED"
                accepted_lookup[p.source_trade_id] = base
                accepted_this_ts += 1
            else:
                base["skip_reason"] = reason
                base["portfolio_status"] = "SKIPPED"
                skipped_rows.append(base)
                skipped_this_ts += 1

        longs, shorts = count_sides(open_positions)
        timeline_rows.append({
            "variant": cfg["name"],
            "time": now,
            "open_positions": len(open_positions),
            "open_longs": longs,
            "open_shorts": shorts,
            "accepted_signals_this_time": accepted_this_ts,
            "skipped_signals_this_time": skipped_this_ts,
            "portfolio_heat_pct": len(open_positions) * RISK_PER_TRADE_PCT,
        })

    # Close remaining positions after final signal.
    final_time = max([p.exit_time for p in open_positions], default=None)
    if final_time is not None:
        open_positions = close_due_positions(open_positions, final_time + pd.Timedelta(seconds=1), accepted_lookup, closed_rows)
        timeline_rows.append({
            "variant": cfg["name"],
            "time": final_time,
            "open_positions": 0,
            "open_longs": 0,
            "open_shorts": 0,
            "accepted_signals_this_time": 0,
            "skipped_signals_this_time": 0,
            "portfolio_heat_pct": 0.0,
        })

    accepted_df = pd.DataFrame(closed_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    timeline_df = pd.DataFrame(timeline_rows)

    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values(["portfolio_exit_time", "portfolio_entry_time", "symbol"]).reset_index(drop=True)
        accepted_df["closed_trade_index"] = np.arange(1, len(accepted_df) + 1)
        accepted_df["equity_r"] = accepted_df["portfolio_net_r"].cumsum()
        accepted_df["peak_r"] = accepted_df["equity_r"].cummax()
        accepted_df["dd_r"] = accepted_df["equity_r"] - accepted_df["peak_r"]

    return accepted_df, skipped_df, timeline_df


def summarize_variant(accepted: pd.DataFrame, skipped: pd.DataFrame, timeline: pd.DataFrame, cfg: dict, input_count: int) -> dict:
    if accepted.empty:
        return {
            "variant": cfg["name"],
            "timeframe": cfg.get("timeframe"),
            "side": cfg.get("side"),
            "max_open": cfg.get("max_open"),
            "max_long": cfg.get("max_long"),
            "max_short": cfg.get("max_short"),
            "input_signals": input_count,
            "accepted_trades": 0,
            "skipped_signals": len(skipped),
            "acceptance_rate": 0.0,
            "total_r": 0.0,
            "avg_r": np.nan,
            "median_r": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_dd_r": np.nan,
            "best_trade_r": np.nan,
            "worst_trade_r": np.nan,
            "losing_streak": np.nan,
            "max_open_observed": 0,
            "avg_open_positions": 0.0,
            "max_heat_pct": 0.0,
            "start": "",
            "end": "",
            "assets_accepted": 0,
        }

    r = accepted["portfolio_net_r"].astype(float).to_numpy()
    win = float((r > 0).mean()) if len(r) else np.nan
    pf = profit_factor(r)
    dd = max_drawdown_r(r)
    ls = losing_streak(r)

    max_open_observed = int(timeline["open_positions"].max()) if not timeline.empty else 0
    avg_open = float(timeline["open_positions"].mean()) if not timeline.empty else 0.0
    max_heat = float(timeline["portfolio_heat_pct"].max()) if not timeline.empty else 0.0

    return {
        "variant": cfg["name"],
        "timeframe": cfg.get("timeframe"),
        "side": cfg.get("side"),
        "max_open": cfg.get("max_open"),
        "max_long": cfg.get("max_long"),
        "max_short": cfg.get("max_short"),
        "input_signals": input_count,
        "accepted_trades": int(len(accepted)),
        "skipped_signals": int(len(skipped)),
        "acceptance_rate": float(len(accepted) / input_count) if input_count else 0.0,
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "win_rate": win,
        "profit_factor": pf,
        "max_dd_r": dd,
        "best_trade_r": float(np.max(r)),
        "worst_trade_r": float(np.min(r)),
        "losing_streak": int(ls),
        "max_open_observed": max_open_observed,
        "avg_open_positions": avg_open,
        "max_heat_pct": max_heat,
        "start": str(accepted["portfolio_entry_time"].min()),
        "end": str(accepted["portfolio_exit_time"].max()),
        "assets_accepted": int(accepted["symbol"].nunique()),
    }


def write_master_report(summary: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("PHASE T5 — TREND PORTFOLIO REPLAY MASTER REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated variants: {len(summary)}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 70)

    if summary.empty:
        lines.append("No summary rows.")
    else:
        cols = [
            "variant", "accepted_trades", "skipped_signals", "acceptance_rate",
            "total_r", "avg_r", "profit_factor", "max_dd_r",
            "win_rate", "max_open_observed", "assets_accepted"
        ]
        for _, row in summary.sort_values("total_r", ascending=False).iterrows():
            lines.append(
                f"{row['variant']}: "
                f"trades={int(row['accepted_trades'])}, "
                f"skipped={int(row['skipped_signals'])}, "
                f"accept={row['acceptance_rate']:.1%}, "
                f"totalR={row['total_r']:.2f}, "
                f"avgR={row['avg_r']:.3f}, "
                f"PF={row['profit_factor']:.2f}, "
                f"DD={row['max_dd_r']:.2f}, "
                f"win={row['win_rate']:.1%}, "
                f"assets={int(row['assets_accepted'])}"
            )

    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 70)
    lines.append("Good signs:")
    lines.append("- Portfolio total R stays positive after max_open constraints.")
    lines.append("- DD falls materially compared with raw T3B.")
    lines.append("- Acceptance rate is not too tiny; otherwise the queue is too restrictive.")
    lines.append("- Fat-tail assets (ZEC/TRX/XRP) get accepted and are not chronically skipped.")
    lines.append("- Max open observed reaches the cap sometimes, but not constantly.")
    lines.append("")
    lines.append("Bad signs:")
    lines.append("- Positive raw edge disappears completely after portfolio constraints.")
    lines.append("- System accepts too few trades to be statistically useful.")
    lines.append("- One direction dominates so much that ALL variants are worse than side-specific variants.")
    lines.append("- DD remains excessive relative to total R.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(OUT_DIR)

    print("=" * 80)
    print("PHASE T5 — TREND PORTFOLIO REPLAY ENGINE")
    print("=" * 80)
    print(f"Input: {INPUT_TRADES}")

    trades = load_trades(INPUT_TRADES)

    print(f"Trades loaded: {len(trades)}")
    print(f"Assets: {trades['symbol'].nunique()}")
    print(f"Date range: {trades[TIME_COL_ENTRY].min()} -> {trades[TIME_COL_EXIT].max()}")
    print(f"Output dir: {OUT_DIR}")
    print("")

    all_accepted = []
    all_skipped = []
    all_timeline = []
    summary_rows = []

    for cfg in PORTFOLIO_VARIANTS:
        vdf = filter_variant(trades, cfg)
        print(f"[RUN] {cfg['name']} | input signals={len(vdf)}")

        accepted, skipped, timeline = replay_variant(vdf, cfg)
        summary = summarize_variant(accepted, skipped, timeline, cfg, input_count=len(vdf))
        summary_rows.append(summary)

        if not accepted.empty:
            all_accepted.append(accepted)
        if not skipped.empty:
            all_skipped.append(skipped)
        if not timeline.empty:
            all_timeline.append(timeline)

        print(
            f"     accepted={summary['accepted_trades']} skipped={summary['skipped_signals']} "
            f"totalR={summary['total_r']:.2f} PF={summary['profit_factor']:.2f} "
            f"DD={summary['max_dd_r']:.2f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    accepted_df = pd.concat(all_accepted, ignore_index=True) if all_accepted else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    timeline_df = pd.concat(all_timeline, ignore_index=True) if all_timeline else pd.DataFrame()

    summary_path = OUT_DIR / "phase_t5_portfolio_summary.csv"
    trades_path = OUT_DIR / "phase_t5_portfolio_trades.csv"
    skipped_path = OUT_DIR / "phase_t5_portfolio_skipped_signals.csv"
    timeline_path = OUT_DIR / "phase_t5_portfolio_open_count_timeline.csv"
    equity_path = OUT_DIR / "phase_t5_portfolio_equity.csv"
    report_path = OUT_DIR / "phase_t5_master_report.txt"

    summary_df.to_csv(summary_path, index=False)
    accepted_df.to_csv(trades_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    timeline_df.to_csv(timeline_path, index=False)

    if not accepted_df.empty:
        equity_cols = [
            "variant", "closed_trade_index", "portfolio_exit_time",
            "symbol", "side", "timeframe", "portfolio_net_r",
            "equity_r", "peak_r", "dd_r"
        ]
        eq_cols = [c for c in equity_cols if c in accepted_df.columns]
        accepted_df[eq_cols].to_csv(equity_path, index=False)
    else:
        pd.DataFrame().to_csv(equity_path, index=False)

    write_master_report(summary_df, report_path)

    print("")
    print("[OK] Wrote:")
    print(f"  {summary_path}")
    print(f"  {trades_path}")
    print(f"  {skipped_path}")
    print(f"  {timeline_path}")
    print(f"  {equity_path}")
    print(f"  {report_path}")
    print("")
    print("Please send back:")
    print("  phase_t5_master_report.txt")
    print("  phase_t5_portfolio_summary.csv")
    print("  phase_t5_portfolio_trades.csv")
    print("  phase_t5_portfolio_equity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
