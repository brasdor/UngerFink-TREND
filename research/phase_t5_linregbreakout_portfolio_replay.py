#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T5 -- LinearRegressionBreakout Config B Portfolio Replay

Replays the T2 Config B trades (N=55/mult=3.0/sm=2.0/midline exit) under
realistic portfolio constraints. T3B found chandelier degrades performance,
so T2 midline trades are used directly as the signal source.

Portfolio constraints applied:
  - max open positions cap
  - one position per asset at any time
  - deterministic signal queue (alphabetic by symbol when simultaneous entries)
  - closed-trade equity tracking

Input:  data/research_linregbreakout_t2/phase_t2_linregbreakout_cfgB_trades.csv
Output: data/research_linregbreakout_t5/
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_DIR  = Path(__file__).resolve().parents[1]
INPUT_TRADES = PROJECT_DIR / "data" / "research_linregbreakout_t2" / "phase_t2_linregbreakout_cfgB_trades.csv"
OUT_DIR      = PROJECT_DIR / "data" / "research_linregbreakout_t5"

TIME_COL_ENTRY = "entry_time"
TIME_COL_EXIT  = "exit_time"
R_COL          = "net_r"

PORTFOLIO_VARIANTS = [
    {"name": "T5_LINREG_B_max3",  "timeframe": "1d", "side": "LONG", "max_open": 3,  "max_long": 3},
    {"name": "T5_LINREG_B_max5",  "timeframe": "1d", "side": "LONG", "max_open": 5,  "max_long": 5},
    {"name": "T5_LINREG_B_max8",  "timeframe": "1d", "side": "LONG", "max_open": 8,  "max_long": 8},
    {"name": "T5_LINREG_B_max10", "timeframe": "1d", "side": "LONG", "max_open": 10, "max_long": 10},
    {"name": "T5_LINREG_B_uncap", "timeframe": "1d", "side": "LONG", "max_open": 99, "max_long": 99},
]

RISK_PER_TRADE_PCT = 0.25


# =============================================================================
# HELPERS
# =============================================================================

def fix_epoch(ts: pd.Series) -> pd.Series:
    """Re-interpret stored Unix-ms integers that were loaded as nanoseconds."""
    if not ts.empty and ts.dt.year.max() < 2000:
        ns_vals = ts.astype("int64")
        return pd.to_datetime(ns_vals, unit="ms", utc=True)
    return ts


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input trades not found: {path}")

    df = pd.read_csv(path)
    required = ["symbol", "timeframe", "side", TIME_COL_ENTRY, TIME_COL_EXIT, R_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TIME_COL_ENTRY] = pd.to_datetime(df[TIME_COL_ENTRY], utc=True, errors="coerce", format="mixed")
    df[TIME_COL_EXIT]  = pd.to_datetime(df[TIME_COL_EXIT],  utc=True, errors="coerce", format="mixed")

    df[TIME_COL_ENTRY] = fix_epoch(df[TIME_COL_ENTRY])
    df[TIME_COL_EXIT]  = fix_epoch(df[TIME_COL_EXIT])

    df["side"]      = df["side"].astype(str).str.upper()
    df[R_COL]       = pd.to_numeric(df[R_COL], errors="coerce")
    df["timeframe"] = df["timeframe"].astype(str).str.lower()

    df = df.dropna(subset=[TIME_COL_ENTRY, TIME_COL_EXIT, R_COL, "symbol", "timeframe", "side"]).copy()
    df = df[df[TIME_COL_EXIT] > df[TIME_COL_ENTRY]].copy()
    df = df.sort_values([TIME_COL_ENTRY, "symbol", "side", TIME_COL_EXIT]).reset_index(drop=True)
    df["source_trade_id"] = np.arange(len(df), dtype=int)
    return df


def filter_variant(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    tf   = cfg.get("timeframe", "ALL")
    side = cfg.get("side", "ALL")
    if tf != "ALL":
        out = out[out["timeframe"] == str(tf).lower()].copy()
    if side != "ALL":
        out = out[out["side"] == str(side).upper()].copy()
    return out.sort_values([TIME_COL_ENTRY, "symbol", TIME_COL_EXIT]).reset_index(drop=True)


def max_drawdown_r(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    eq   = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def profit_factor(r: np.ndarray) -> float:
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    if l <= 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def losing_streak(r: np.ndarray) -> int:
    mx = cur = 0
    for v in r:
        if v < 0:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return int(mx)


@dataclass
class OpenPosition:
    source_trade_id: int
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    net_r: float


def close_due(
    open_pos: List[OpenPosition],
    now: pd.Timestamp,
    lookup: Dict[int, dict],
    closed: List[dict],
) -> List[OpenPosition]:
    still_open = []
    for p in open_pos:
        if p.exit_time <= now:
            row = lookup[p.source_trade_id].copy()
            row["portfolio_exit_time"] = p.exit_time
            row["portfolio_net_r"]     = p.net_r
            closed.append(row)
        else:
            still_open.append(p)
    return still_open


def can_accept(row: pd.Series, open_pos: List[OpenPosition], cfg: dict) -> Tuple[bool, str]:
    if len(open_pos) >= int(cfg["max_open"]):
        return False, "MAX_OPEN_REACHED"
    if any(p.symbol == str(row["symbol"]) for p in open_pos):
        return False, "ASSET_ALREADY_OPEN"
    n_long = sum(1 for p in open_pos if p.side == "LONG")
    if row["side"] == "LONG" and n_long >= int(cfg["max_long"]):
        return False, "MAX_LONG_REACHED"
    return True, "ACCEPTED"


def replay_variant(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    open_pos: List[OpenPosition] = []
    lookup:   Dict[int, dict]    = {}
    closed:   List[dict]         = []
    skipped:  List[dict]         = []
    timeline: List[dict]         = []

    grouped = dict(tuple(df.groupby(TIME_COL_ENTRY, sort=True)))

    for now in sorted(grouped):
        open_pos = close_due(open_pos, now, lookup, closed)

        signals = grouped[now].sort_values(["symbol", TIME_COL_EXIT])

        for _, row in signals.iterrows():
            ok, reason = can_accept(row, open_pos, cfg)
            base = row.to_dict()
            base["variant"]                    = cfg["name"]
            base["portfolio_entry_time"]       = row[TIME_COL_ENTRY]
            base["open_positions_before_entry"] = len(open_pos)

            if ok:
                p = OpenPosition(
                    source_trade_id=int(row["source_trade_id"]),
                    symbol=str(row["symbol"]),
                    side=str(row["side"]),
                    entry_time=row[TIME_COL_ENTRY],
                    exit_time=row[TIME_COL_EXIT],
                    net_r=float(row[R_COL]),
                )
                open_pos.append(p)
                base["skip_reason"]     = ""
                base["portfolio_status"] = "ACCEPTED"
                lookup[p.source_trade_id] = base
            else:
                base["skip_reason"]     = reason
                base["portfolio_status"] = "SKIPPED"
                skipped.append(base)

        n_long = sum(1 for p in open_pos if p.side == "LONG")
        timeline.append({
            "variant":                    cfg["name"],
            "time":                       now,
            "open_positions":             len(open_pos),
            "open_longs":                 n_long,
            "accepted_signals_this_time": sum(1 for b in [base] if b.get("portfolio_status") == "ACCEPTED"),
            "portfolio_heat_pct":         len(open_pos) * RISK_PER_TRADE_PCT,
        })

    # flush remaining open positions
    if open_pos:
        final = max(p.exit_time for p in open_pos)
        open_pos = close_due(open_pos, final + pd.Timedelta(seconds=1), lookup, closed)

    accepted_df = pd.DataFrame(closed)
    skipped_df  = pd.DataFrame(skipped)
    timeline_df = pd.DataFrame(timeline)

    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values(
            ["portfolio_exit_time", "portfolio_entry_time", "symbol"]
        ).reset_index(drop=True)
        accepted_df["closed_trade_index"] = np.arange(1, len(accepted_df) + 1)
        accepted_df["equity_r"] = accepted_df["portfolio_net_r"].cumsum()
        accepted_df["peak_r"]   = accepted_df["equity_r"].cummax()
        accepted_df["dd_r"]     = accepted_df["equity_r"] - accepted_df["peak_r"]

    return accepted_df, skipped_df, timeline_df


def summarize_variant(accepted: pd.DataFrame, skipped: pd.DataFrame,
                      timeline: pd.DataFrame, cfg: dict, n_input: int) -> dict:
    base = dict(
        variant=cfg["name"], timeframe=cfg.get("timeframe"),
        side=cfg.get("side"), max_open=cfg.get("max_open"),
        input_signals=n_input,
    )
    if accepted.empty:
        return {**base, "accepted_trades": 0, "skipped_signals": len(skipped),
                "acceptance_rate": 0.0, "total_r": 0.0, "avg_r": np.nan,
                "win_rate": np.nan, "profit_factor": np.nan, "max_dd_r": np.nan,
                "best_trade_r": np.nan, "worst_trade_r": np.nan,
                "losing_streak": np.nan, "max_open_observed": 0,
                "avg_open_positions": 0.0, "assets_accepted": 0}

    r = accepted["portfolio_net_r"].astype(float).to_numpy()
    return {
        **base,
        "accepted_trades":    int(len(r)),
        "skipped_signals":    int(len(skipped)),
        "acceptance_rate":    float(len(r) / n_input) if n_input else 0.0,
        "total_r":            float(r.sum()),
        "avg_r":              float(r.mean()),
        "win_rate":           float((r > 0).mean()),
        "profit_factor":      profit_factor(r),
        "max_dd_r":           max_drawdown_r(r),
        "best_trade_r":       float(r.max()),
        "worst_trade_r":      float(r.min()),
        "losing_streak":      losing_streak(r),
        "max_open_observed":  int(timeline["open_positions"].max()) if not timeline.empty else 0,
        "avg_open_positions": float(timeline["open_positions"].mean()) if not timeline.empty else 0.0,
        "assets_accepted":    int(accepted["symbol"].nunique()),
    }


def write_master_report(summary: pd.DataFrame, t2_total_r: float, t2_avg_r: float,
                        t2_pf: float, out_path: Path) -> None:
    lines = [
        "PHASE T5 -- LinearRegressionBreakout Config B PORTFOLIO REPLAY MASTER REPORT",
        "=" * 70,
        "",
        "Method:  LinReg(55) + 3.0*StdErr breakout  /  midline exit  /  1D LONG",
        f"T2 baseline (no portfolio filter): trades=196  totalR={t2_total_r:.2f}  avgR={t2_avg_r:.4f}  PF={t2_pf:.3f}",
        "T4 note:  Second-half decay and Dec-2024 concentration flagged.",
        "          T5 portfolio constraints reduce XRP/ALGO/ADA over-exposure.",
        "",
        "=" * 70,
        "VARIANT SUMMARY",
        "=" * 70,
    ]

    for _, row in summary.sort_values("total_r", ascending=False).iterrows():
        lines.append(
            f"  {row['variant']:25s}: "
            f"trades={int(row['accepted_trades']):3d}  "
            f"skip={int(row['skipped_signals']):3d}  "
            f"accept={row['acceptance_rate']:.0%}  "
            f"totalR={row['total_r']:+7.2f}  "
            f"avgR={row['avg_r']:+.3f}  "
            f"PF={row['profit_factor']:.2f}  "
            f"DD={row['max_dd_r']:.2f}  "
            f"win={row['win_rate']:.1%}  "
            f"assets={int(row['assets_accepted'])}"
        )

    lines += [
        "",
        "=" * 70,
        "INTERPRETATION GUIDE",
        "=" * 70,
        "",
        "  PROCEED TO T6 if:",
        "  - At least one capped variant (max_open <= 10) retains meaningful positive R",
        "  - Portfolio DD materially less than uncapped raw DD",
        "  - Acceptance rate > 50% (queue not chronically blocked)",
        "",
        "  CONCERN if:",
        "  - All capped variants erase the edge (XRP/ALGO/ADA always take the slots)",
        "  - Acceptance rate very low (< 40%) at realistic caps",
        "  - avg_r degrades sharply when constrained (concentration not diversified)",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE T5 -- LinearRegressionBreakout Config B Portfolio Replay")
    print("=" * 70)
    print(f"Input:  {INPUT_TRADES}")

    trades = load_trades(INPUT_TRADES)
    print(f"Trades: {len(trades)}")
    print(f"Assets: {trades['symbol'].nunique()}")
    print(f"Range:  {trades[TIME_COL_ENTRY].min()} -> {trades[TIME_COL_EXIT].max()}")
    print(f"Output: {OUT_DIR}")
    print()

    all_accepted = []
    all_skipped  = []
    all_timeline = []
    summary_rows = []

    for cfg in PORTFOLIO_VARIANTS:
        vdf = filter_variant(trades, cfg)
        print(f"[RUN] {cfg['name']:25s} | input={len(vdf)}")
        accepted, skipped, timeline = replay_variant(vdf, cfg)
        s = summarize_variant(accepted, skipped, timeline, cfg, n_input=len(vdf))
        summary_rows.append(s)
        if not accepted.empty:
            all_accepted.append(accepted)
        if not skipped.empty:
            all_skipped.append(skipped)
        if not timeline.empty:
            all_timeline.append(timeline)
        print(
            f"       accepted={s['accepted_trades']:3d}  skipped={s['skipped_signals']:3d}  "
            f"totalR={s['total_r']:+7.2f}  PF={s['profit_factor']:.2f}  DD={s['max_dd_r']:.2f}"
        )

    summary_df  = pd.DataFrame(summary_rows)
    accepted_df = pd.concat(all_accepted, ignore_index=True) if all_accepted else pd.DataFrame()
    skipped_df  = pd.concat(all_skipped,  ignore_index=True) if all_skipped  else pd.DataFrame()
    timeline_df = pd.concat(all_timeline, ignore_index=True) if all_timeline else pd.DataFrame()

    prefix = OUT_DIR / "phase_t5_linregbreakout_cfgB"

    summary_df.to_csv(f"{prefix}_portfolio_summary.csv",  index=False)
    accepted_df.to_csv(f"{prefix}_portfolio_trades.csv",  index=False)
    skipped_df.to_csv(f"{prefix}_portfolio_skipped.csv",  index=False)
    timeline_df.to_csv(f"{prefix}_open_count_timeline.csv", index=False)

    if not accepted_df.empty:
        eq_cols = [c for c in ["variant", "closed_trade_index", "portfolio_exit_time",
                                "symbol", "side", "timeframe", "portfolio_net_r",
                                "equity_r", "peak_r", "dd_r"] if c in accepted_df.columns]
        accepted_df[eq_cols].to_csv(f"{prefix}_equity_curve.csv", index=False)

    write_master_report(
        summary=summary_df,
        t2_total_r=71.04,
        t2_avg_r=0.3624,
        t2_pf=1.528,
        out_path=Path(f"{prefix}_master_report.txt"),
    )

    print()
    print("=" * 70)
    print("T5 COMPLETE -- review master report before T6")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
