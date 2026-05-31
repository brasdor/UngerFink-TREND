#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T6 — CAPITAL & EXECUTION ENGINE V2
=======================================

Purpose
-------
Transforms Phase T5 portfolio replay trades into a realistic capital/execution simulation.

This version follows a more robust "Unger-style" research discipline:
- no new signal logic
- no optimisation/tuning of entries
- closed-trade equity accounting
- realistic capital allocation
- portfolio heat control based on concurrent open risk
- margin/cash reservation
- kill-switch and drawdown protection
- variant-by-variant evaluation, without mixing systems

Input
-----
data/research_trend_t5/phase_t5_portfolio_trades.csv

Expected columns from Phase T5:
- variant
- symbol
- side
- timeframe
- portfolio_entry_time / entry_time
- portfolio_exit_time / exit_time
- entry_price
- initial_stop
- initial_risk
- portfolio_net_r / net_r

Outputs
-------
data/research_trend_t6/
- phase_t6_master_report.txt
- phase_t6_variant_summary.csv
- phase_t6_trade_log.csv
- phase_t6_equity_curve.csv
- phase_t6_heat_timeline.csv
- phase_t6_skipped_entries.csv

Notes
-----
This is NOT a live trading engine.
It is a research-grade capital simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path.cwd()

INPUT_FILE = PROJECT_ROOT / "data" / "research_trend_t5" / "phase_t5_portfolio_trades.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "research_trend_t6"

INITIAL_CAPITAL_USDT = 10_000.0

# Fixed fractional risk model.
# 0.25% is intentionally conservative and coherent with a paper/live transition mindset.
RISK_PER_TRADE_PCT = 0.0025

# Portfolio heat = sum of open trade initial risks / current equity.
MAX_PORTFOLIO_HEAT_PCT = 0.015   # 1.5% total open risk, e.g. 6 trades * 0.25%

# Margin / notional model.
# Binance Spot LONG only — no leverage, no margin.
LEVERAGE = 1.0

# Do not allow a single trade to consume too much notional/margin.
MAX_NOTIONAL_PER_TRADE_PCT = 0.35     # max 35% of equity as notional
MAX_MARGIN_USAGE_PCT = 0.85           # do not reserve more than 85% of equity as margin

# Execution realism.
# Phase T5 net_r already includes a cost_r column in most cases.
# Keep EXTRA_EXECUTION_COST_R at 0 for first pass, then stress later if needed.
EXTRA_EXECUTION_COST_R = 0.00

# Kill-switch / research safety layer.
STOP_DD_PCT = 35.0                    # stop accepting new entries after -35% closed-equity DD
MIN_EQUITY_FLOOR_PCT = 0.50           # stop if equity falls below 50% initial capital

# Numerical safeguards.
MIN_STOP_PCT = 0.002                  # reject trades whose initial stop distance is unrealistically tiny
MAX_STOP_PCT = 0.50                   # reject trades whose initial stop is absurdly wide
EPS = 1e-12


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class OpenPosition:
    trade_id: int
    variant: str
    symbol: str
    side: str
    timeframe: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    initial_stop: float
    stop_pct: float
    risk_pct: float
    risk_amount: float
    notional: float
    margin_reserved: float
    net_r: float


@dataclass
class TradeResult:
    variant: str
    trade_id: int
    symbol: str
    side: str
    timeframe: str
    entry_time: str
    exit_time: str
    entry_price: float
    initial_stop: float
    stop_pct: float
    risk_pct: float
    risk_amount: float
    notional: float
    margin_reserved: float
    net_r_raw: float
    extra_cost_r: float
    net_r_after_extra_cost: float
    pnl_usdt: float
    equity_before_entry: float
    equity_after_exit: float
    peak_equity: float
    drawdown_pct: float
    portfolio_heat_before_entry_pct: float
    portfolio_heat_after_entry_pct: float
    cash_after_exit: float


@dataclass
class SkippedEntry:
    variant: str
    trade_id: int
    symbol: str
    side: str
    timeframe: str
    entry_time: str
    reason: str
    equity: float
    cash: float
    reserved_margin: float
    open_risk_pct: float
    stop_pct: float
    requested_risk_pct: float
    requested_notional: float
    requested_margin: float


# ============================================================
# HELPERS
# ============================================================

def _first_existing_col(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def max_drawdown_pct(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100.0
    return float(dd.min())


def profit_factor_from_pnl(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= EPS:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def profit_factor_from_r(r: pd.Series) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= EPS:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def losing_streak(values: List[float]) -> int:
    best = 0
    cur = 0
    for x in values:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    entry_col = _first_existing_col(out, ["portfolio_entry_time", "entry_time"])
    exit_col = _first_existing_col(out, ["portfolio_exit_time", "exit_time"])
    r_col = _first_existing_col(out, ["portfolio_net_r", "net_r", "gross_r", "r_value", "R"])

    if entry_col is None:
        raise ValueError("Cannot find entry time column. Expected portfolio_entry_time or entry_time.")
    if exit_col is None:
        raise ValueError("Cannot find exit time column. Expected portfolio_exit_time or exit_time.")
    if r_col is None:
        raise ValueError("Cannot find R column. Expected portfolio_net_r or net_r.")

    required = ["symbol", "side", "entry_price", "initial_stop"]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    if "variant" not in out.columns:
        out["variant"] = "ALL"

    if "timeframe" not in out.columns:
        out["timeframe"] = "unknown"

    out["entry_time_t6"] = pd.to_datetime(out[entry_col], utc=True, errors="coerce")
    out["exit_time_t6"] = pd.to_datetime(out[exit_col], utc=True, errors="coerce")
    out["net_r_t6"] = pd.to_numeric(out[r_col], errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["initial_stop"] = pd.to_numeric(out["initial_stop"], errors="coerce")

    out = out.dropna(subset=["entry_time_t6", "exit_time_t6", "net_r_t6", "entry_price", "initial_stop"]).copy()

    out = out[out["exit_time_t6"] >= out["entry_time_t6"]].copy()

    out["trade_id_t6"] = np.arange(len(out))

    # Initial stop distance as % of entry.
    out["stop_pct_t6"] = (out["entry_price"] - out["initial_stop"]).abs() / out["entry_price"].abs()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["stop_pct_t6"])

    return out


def current_equity(cash: float, open_positions: Dict[int, OpenPosition]) -> float:
    # Closed-equity style: no unrealized PnL, but reserved margin remains part of equity.
    return cash + sum(p.margin_reserved for p in open_positions.values())


def current_reserved_margin(open_positions: Dict[int, OpenPosition]) -> float:
    return sum(p.margin_reserved for p in open_positions.values())


def current_open_risk_amount(open_positions: Dict[int, OpenPosition]) -> float:
    return sum(p.risk_amount for p in open_positions.values())


def simulate_variant(variant_name: str, trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Event-driven simulation:
    - exits are processed before entries at the same timestamp
    - each accepted entry reserves margin
    - each exit releases margin and applies PnL = original risk_amount * net_r
    - portfolio heat is based on concurrent open initial risk
    """

    vdf = trades[trades["variant"] == variant_name].copy()
    vdf = vdf.sort_values(["entry_time_t6", "exit_time_t6", "symbol"]).reset_index(drop=True)

    # Build event table.
    events = []

    for _, row in vdf.iterrows():
        tid = int(row["trade_id_t6"])
        events.append((row["entry_time_t6"], 1, tid))  # entries second
        events.append((row["exit_time_t6"], 0, tid))   # exits first

    events.sort(key=lambda x: (x[0], x[1]))

    rows_by_id = {int(r["trade_id_t6"]): r for _, r in vdf.iterrows()}

    cash = float(INITIAL_CAPITAL_USDT)
    peak_equity = float(INITIAL_CAPITAL_USDT)
    stopped = False

    open_positions: Dict[int, OpenPosition] = {}
    trade_results: List[TradeResult] = []
    skipped: List[SkippedEntry] = []
    equity_records = []
    heat_records = []

    for timestamp, event_type, tid in events:
        eq_before_event = current_equity(cash, open_positions)
        reserved_before_event = current_reserved_margin(open_positions)
        open_risk_before_event = current_open_risk_amount(open_positions)
        heat_before_pct = open_risk_before_event / max(eq_before_event, EPS)

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------
        if event_type == 0:
            pos = open_positions.pop(tid, None)
            if pos is None:
                continue

            net_r_after_cost = pos.net_r - EXTRA_EXECUTION_COST_R
            pnl = pos.risk_amount * net_r_after_cost

            cash += pos.margin_reserved + pnl

            eq_after_exit = current_equity(cash, open_positions)
            peak_equity = max(peak_equity, eq_after_exit)
            dd_pct = (eq_after_exit - peak_equity) / max(peak_equity, EPS) * 100.0

            result = TradeResult(
                variant=variant_name,
                trade_id=tid,
                symbol=pos.symbol,
                side=pos.side,
                timeframe=pos.timeframe,
                entry_time=str(pos.entry_time),
                exit_time=str(pos.exit_time),
                entry_price=pos.entry_price,
                initial_stop=pos.initial_stop,
                stop_pct=pos.stop_pct,
                risk_pct=pos.risk_pct,
                risk_amount=pos.risk_amount,
                notional=pos.notional,
                margin_reserved=pos.margin_reserved,
                net_r_raw=pos.net_r,
                extra_cost_r=EXTRA_EXECUTION_COST_R,
                net_r_after_extra_cost=net_r_after_cost,
                pnl_usdt=pnl,
                equity_before_entry=np.nan,  # filled from accepted entry snapshot not stored here
                equity_after_exit=eq_after_exit,
                peak_equity=peak_equity,
                drawdown_pct=dd_pct,
                portfolio_heat_before_entry_pct=np.nan,
                portfolio_heat_after_entry_pct=np.nan,
                cash_after_exit=cash,
            )
            trade_results.append(result)

            equity_records.append({
                "variant": variant_name,
                "time": timestamp,
                "event": "exit",
                "trade_id": tid,
                "symbol": pos.symbol,
                "side": pos.side,
                "equity": eq_after_exit,
                "cash": cash,
                "reserved_margin": current_reserved_margin(open_positions),
                "open_positions": len(open_positions),
                "open_risk_amount": current_open_risk_amount(open_positions),
                "portfolio_heat_pct": current_open_risk_amount(open_positions) / max(eq_after_exit, EPS) * 100.0,
                "peak_equity": peak_equity,
                "drawdown_pct": dd_pct,
                "pnl_usdt": pnl,
                "net_r": net_r_after_cost,
            })

            heat_records.append({
                "variant": variant_name,
                "time": timestamp,
                "event": "exit",
                "open_positions": len(open_positions),
                "open_risk_amount": current_open_risk_amount(open_positions),
                "equity": eq_after_exit,
                "portfolio_heat_pct": current_open_risk_amount(open_positions) / max(eq_after_exit, EPS) * 100.0,
                "reserved_margin": current_reserved_margin(open_positions),
                "cash": cash,
            })

            # Kill-switch after closed trade, not intra-trade.
            if dd_pct <= -STOP_DD_PCT or eq_after_exit <= INITIAL_CAPITAL_USDT * MIN_EQUITY_FLOOR_PCT:
                stopped = True

            continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------
        row = rows_by_id[tid]

        eq = current_equity(cash, open_positions)
        reserved = current_reserved_margin(open_positions)
        open_risk = current_open_risk_amount(open_positions)
        heat_pct = open_risk / max(eq, EPS)

        stop_pct = float(row["stop_pct_t6"])
        net_r = float(row["net_r_t6"])

        symbol = str(row["symbol"])
        side = str(row["side"])
        timeframe = str(row["timeframe"])

        requested_risk_pct = RISK_PER_TRADE_PCT
        requested_risk_amount = eq * requested_risk_pct
        requested_notional = requested_risk_amount / max(stop_pct, EPS)
        max_notional = eq * MAX_NOTIONAL_PER_TRADE_PCT
        requested_notional = min(requested_notional, max_notional)
        requested_margin = requested_notional / LEVERAGE

        reason = None

        if stopped:
            reason = "KILL_SWITCH_ACTIVE"
        elif stop_pct < MIN_STOP_PCT:
            reason = "STOP_TOO_TIGHT"
        elif stop_pct > MAX_STOP_PCT:
            reason = "STOP_TOO_WIDE"
        elif eq <= INITIAL_CAPITAL_USDT * MIN_EQUITY_FLOOR_PCT:
            reason = "EQUITY_FLOOR"
        elif heat_pct + requested_risk_pct > MAX_PORTFOLIO_HEAT_PCT + 1e-9:
            reason = "MAX_PORTFOLIO_HEAT"
        elif requested_margin > cash:
            reason = "INSUFFICIENT_CASH"
        elif (reserved + requested_margin) > eq * MAX_MARGIN_USAGE_PCT:
            reason = "MAX_MARGIN_USAGE"
        elif not math.isfinite(net_r):
            reason = "INVALID_R"

        if reason is not None:
            skipped.append(SkippedEntry(
                variant=variant_name,
                trade_id=tid,
                symbol=symbol,
                side=side,
                timeframe=timeframe,
                entry_time=str(row["entry_time_t6"]),
                reason=reason,
                equity=eq,
                cash=cash,
                reserved_margin=reserved,
                open_risk_pct=heat_pct * 100.0,
                stop_pct=stop_pct,
                requested_risk_pct=requested_risk_pct * 100.0,
                requested_notional=requested_notional,
                requested_margin=requested_margin,
            ))
            continue

        # Accept entry.
        cash -= requested_margin

        pos = OpenPosition(
            trade_id=tid,
            variant=variant_name,
            symbol=symbol,
            side=side,
            timeframe=timeframe,
            entry_time=row["entry_time_t6"],
            exit_time=row["exit_time_t6"],
            entry_price=float(row["entry_price"]),
            initial_stop=float(row["initial_stop"]),
            stop_pct=stop_pct,
            risk_pct=requested_risk_pct,
            risk_amount=requested_risk_amount,
            notional=requested_notional,
            margin_reserved=requested_margin,
            net_r=net_r,
        )
        open_positions[tid] = pos

        eq_after_entry = current_equity(cash, open_positions)
        heat_after_pct = current_open_risk_amount(open_positions) / max(eq_after_entry, EPS)

        equity_records.append({
            "variant": variant_name,
            "time": timestamp,
            "event": "entry",
            "trade_id": tid,
            "symbol": symbol,
            "side": side,
            "equity": eq_after_entry,
            "cash": cash,
            "reserved_margin": current_reserved_margin(open_positions),
            "open_positions": len(open_positions),
            "open_risk_amount": current_open_risk_amount(open_positions),
            "portfolio_heat_pct": heat_after_pct * 100.0,
            "peak_equity": peak_equity,
            "drawdown_pct": (eq_after_entry - peak_equity) / max(peak_equity, EPS) * 100.0,
            "pnl_usdt": 0.0,
            "net_r": np.nan,
        })

        heat_records.append({
            "variant": variant_name,
            "time": timestamp,
            "event": "entry",
            "open_positions": len(open_positions),
            "open_risk_amount": current_open_risk_amount(open_positions),
            "equity": eq_after_entry,
            "portfolio_heat_pct": heat_after_pct * 100.0,
            "reserved_margin": current_reserved_margin(open_positions),
            "cash": cash,
        })

    # Convert.
    trade_log = pd.DataFrame([asdict(x) for x in trade_results])
    skipped_df = pd.DataFrame([asdict(x) for x in skipped])
    equity_df = pd.DataFrame(equity_records)
    heat_df = pd.DataFrame(heat_records)

    # Fill entry snapshots in trade log from heat/equity records.
    if not trade_log.empty and not equity_df.empty:
        entry_snap = equity_df[equity_df["event"] == "entry"][
            ["trade_id", "equity", "portfolio_heat_pct"]
        ].rename(columns={
            "equity": "equity_before_entry",
            "portfolio_heat_pct": "portfolio_heat_after_entry_pct"
        })
        # heat before entry is not stored directly, leave it NaN for now.
        trade_log = trade_log.drop(columns=["equity_before_entry", "portfolio_heat_after_entry_pct"], errors="ignore")
        trade_log = trade_log.merge(entry_snap, on="trade_id", how="left")

    return trade_log, equity_df, heat_df, skipped_df


def summarize_variant(variant: str, trade_log: pd.DataFrame, equity_df: pd.DataFrame, skipped_df: pd.DataFrame, original_count: int) -> dict:
    if trade_log.empty:
        return {
            "variant": variant,
            "input_signals": original_count,
            "executed_trades": 0,
            "skipped_entries": len(skipped_df),
            "acceptance_rate_pct": 0.0,
            "initial_capital": INITIAL_CAPITAL_USDT,
            "final_equity": INITIAL_CAPITAL_USDT,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor_pnl": 0.0,
            "profit_factor_r": 0.0,
            "win_rate_pct": 0.0,
            "avg_net_r": 0.0,
            "median_net_r": 0.0,
            "best_net_r": 0.0,
            "worst_net_r": 0.0,
            "max_losing_streak": 0,
            "max_open_positions_observed": 0,
            "max_heat_pct_observed": 0.0,
            "max_reserved_margin_pct": 0.0,
            "kill_switch_triggered": False,
        }

    final_equity = float(trade_log["equity_after_exit"].iloc[-1])
    dd = max_drawdown_pct(trade_log["equity_after_exit"])
    pnl_pf = profit_factor_from_pnl(trade_log["pnl_usdt"])
    r_pf = profit_factor_from_r(trade_log["net_r_after_extra_cost"])

    if not equity_df.empty:
        max_open = int(equity_df["open_positions"].max())
        max_heat = float(equity_df["portfolio_heat_pct"].max())
        max_reserved_margin_pct = float((equity_df["reserved_margin"] / equity_df["equity"].replace(0, np.nan) * 100).max())
    else:
        max_open = 0
        max_heat = 0.0
        max_reserved_margin_pct = 0.0

    return {
        "variant": variant,
        "input_signals": original_count,
        "executed_trades": len(trade_log),
        "skipped_entries": len(skipped_df),
        "acceptance_rate_pct": len(trade_log) / max(original_count, 1) * 100.0,
        "initial_capital": INITIAL_CAPITAL_USDT,
        "final_equity": final_equity,
        "return_pct": (final_equity / INITIAL_CAPITAL_USDT - 1.0) * 100.0,
        "max_drawdown_pct": dd,
        "profit_factor_pnl": pnl_pf,
        "profit_factor_r": r_pf,
        "win_rate_pct": (trade_log["pnl_usdt"] > 0).mean() * 100.0,
        "avg_net_r": float(trade_log["net_r_after_extra_cost"].mean()),
        "median_net_r": float(trade_log["net_r_after_extra_cost"].median()),
        "best_net_r": float(trade_log["net_r_after_extra_cost"].max()),
        "worst_net_r": float(trade_log["net_r_after_extra_cost"].min()),
        "max_losing_streak": losing_streak(trade_log["pnl_usdt"].tolist()),
        "max_open_positions_observed": max_open,
        "max_heat_pct_observed": max_heat,
        "max_reserved_margin_pct": max_reserved_margin_pct,
        "kill_switch_triggered": bool((trade_log["drawdown_pct"] <= -STOP_DD_PCT).any() or (trade_log["equity_after_exit"] <= INITIAL_CAPITAL_USDT * MIN_EQUITY_FLOOR_PCT).any()),
    }


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("PHASE T6 — CAPITAL & EXECUTION ENGINE V2")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Initial capital: {INITIAL_CAPITAL_USDT:,.2f} USDT")
    lines.append(f"Risk per trade: {RISK_PER_TRADE_PCT*100:.3f}%")
    lines.append(f"Max portfolio heat: {MAX_PORTFOLIO_HEAT_PCT*100:.3f}%")
    lines.append(f"Leverage proxy: {LEVERAGE:.2f}x")
    lines.append(f"Max notional per trade: {MAX_NOTIONAL_PER_TRADE_PCT*100:.1f}% equity")
    lines.append(f"Max margin usage: {MAX_MARGIN_USAGE_PCT*100:.1f}% equity")
    lines.append(f"Extra execution cost: {EXTRA_EXECUTION_COST_R:.3f}R")
    lines.append(f"Kill-switch drawdown: -{STOP_DD_PCT:.1f}%")
    lines.append("")

    if summary.empty:
        lines.append("No results.")
    else:
        lines.append("VARIANT SUMMARY")
        lines.append("-" * 70)
        for _, r in summary.sort_values("return_pct", ascending=False).iterrows():
            lines.append(
                f"{r['variant']}: "
                f"trades={int(r['executed_trades'])}/{int(r['input_signals'])}, "
                f"return={r['return_pct']:.2f}%, "
                f"final={r['final_equity']:.2f}, "
                f"DD={r['max_drawdown_pct']:.2f}%, "
                f"PF_R={r['profit_factor_r']:.2f}, "
                f"avgR={r['avg_net_r']:.3f}, "
                f"win={r['win_rate_pct']:.1f}%, "
                f"maxHeat={r['max_heat_pct_observed']:.2f}%"
            )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)
    lines.append("This phase does not validate live trading yet.")
    lines.append("It checks whether the portfolio replay can survive realistic capital constraints.")
    lines.append("Good signs: positive return, DD below kill-switch, PF_R > 1, heat within limits.")
    lines.append("Bad signs: kill-switch triggered, large skipped rate due to cash/margin, or DD too high.")
    lines.append("")
    lines.append("Next logical phase: T7 Strict Robustness on the best capital variants.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=" * 80)
    print("PHASE T6 — CAPITAL & EXECUTION ENGINE V2")
    print("=" * 80)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(INPUT_FILE)
    trades = normalize_trades(raw)

    if trades.empty:
        raise ValueError("No valid trades after normalization.")

    print(f"Loaded valid trades: {len(trades)}")
    print(f"Variants: {trades['variant'].nunique()}")
    print(f"Date range: {trades['entry_time_t6'].min()} -> {trades['exit_time_t6'].max()}")
    print("")

    all_trade_logs = []
    all_equity = []
    all_heat = []
    all_skipped = []
    summaries = []

    for variant in sorted(trades["variant"].unique()):
        print(f"[RUN] {variant}")

        trade_log, equity_df, heat_df, skipped_df = simulate_variant(variant, trades)
        original_count = int((trades["variant"] == variant).sum())

        summaries.append(summarize_variant(variant, trade_log, equity_df, skipped_df, original_count))

        if not trade_log.empty:
            all_trade_logs.append(trade_log)
        if not equity_df.empty:
            all_equity.append(equity_df)
        if not heat_df.empty:
            all_heat.append(heat_df)
        if not skipped_df.empty:
            all_skipped.append(skipped_df)

    summary_df = pd.DataFrame(summaries)

    trade_log_df = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    heat_df = pd.concat(all_heat, ignore_index=True) if all_heat else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()

    # Save.
    summary_path = OUTPUT_DIR / "phase_t6_variant_summary.csv"
    trade_log_path = OUTPUT_DIR / "phase_t6_trade_log.csv"
    equity_path = OUTPUT_DIR / "phase_t6_equity_curve.csv"
    heat_path = OUTPUT_DIR / "phase_t6_heat_timeline.csv"
    skipped_path = OUTPUT_DIR / "phase_t6_skipped_entries.csv"
    report_path = OUTPUT_DIR / "phase_t6_master_report.txt"

    summary_df.to_csv(summary_path, index=False)
    trade_log_df.to_csv(trade_log_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    heat_df.to_csv(heat_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    write_report(summary_df, report_path)

    print("")
    print("[OK] phase_t6_variant_summary.csv")
    print("[OK] phase_t6_trade_log.csv")
    print("[OK] phase_t6_equity_curve.csv")
    print("[OK] phase_t6_heat_timeline.csv")
    print("[OK] phase_t6_skipped_entries.csv")
    print("[OK] phase_t6_master_report.txt")
    print("")
    print("Done.")

    if not summary_df.empty:
        print("")
        print(summary_df.sort_values("return_pct", ascending=False)[[
            "variant", "executed_trades", "return_pct", "max_drawdown_pct", "profit_factor_r", "avg_net_r"
        ]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
