#!/usr/bin/env python3
"""
Stop-loss variant engine -- extends tools/t3_exit_engine.py's re-simulation approach
(entry frozen, only the exit re-simulated against actual intrabar high/low/close over
the trade's original holding window; a new rule can only trigger EARLIER than the
original flat/opposite-signal exit, never later). Built for testing whether an actual
executed protective stop (candidates 12 and 19 currently have NONE -- ATR x 2.0 is only
a position-sizing/R-normalization unit, exits are purely signal-driven) reduces
liquidation exposure without destroying the edge.

Four mechanisms, five variant families:
  - fixed_r:      stop at entry_px -/+ r_mult * (ATR_MULT_R * atr_at_entry), intrabar
                  low/high touch triggers exit at the stop price. Used for the
                  catastrophic-tier (-4R/-6R/-8R) and percentile-calibrated stops.
  - atr_mult:     stop at entry_px -/+ atr_mult * atr_at_entry directly (NOT R-multiple
                  -- since this system's own R-unit is DEFINED as ATR_MULT_R(2.0) x atr,
                  an "atr_mult" stop of 4.0/6.0 is mathematically identical to a fixed_r
                  stop of 2.0R/3.0R. This equivalence is intentional to surface, not
                  hidden -- see run_stop_loss_test.py's reporting, which states the R-
                  equivalent explicitly so "wider ATR multiple" isn't misread as wider
                  than the -4R catastrophic tier when in R-terms it's actually tighter.
  - bar_close_r:  same fixed_r mechanism, but requires the BAR'S CLOSE (not an intrabar
                  wick) to have breached the level before triggering -- exit executes at
                  that close, not at the stop price itself.
  - time_stop:    exit N days after entry regardless of P&L, at that bar's close, IF the
                  original exit hadn't already happened by then (a time stop can only
                  shorten a trade, same "never later than original" invariant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATR_MULT_R = 2.0  # matches t1_harness/t3_exit_engine convention: 1R = ATR_MULT_R * atr


def _entry_risk(row) -> float | None:
    atr = row.atr_at_entry
    if not np.isfinite(atr) or atr <= 0:
        return None
    return ATR_MULT_R * atr


def compute_mae_r_distribution(trades: pd.DataFrame, prepared: dict[str, pd.DataFrame]) -> np.ndarray:
    """Per-trade max adverse excursion, in R-units, over each trade's ORIGINAL (no-stop)
    holding window -- the candidate's own empirical tail-risk distribution, used to
    calibrate the percentile-based stop rather than assuming an ATR-normal distribution."""
    mae = []
    for row in trades.itertuples():
        risk = _entry_risk(row)
        if risk is None:
            continue
        df = prepared[row.symbol]
        window = df[(df.index > row.entry_date) & (df.index <= row.exit_date)]
        if len(window) == 0:
            continue
        side = 1 if row.side == "long" else -1
        if side == 1:
            worst = (window["low"].min() - row.entry_px) / risk
        else:
            worst = (row.entry_px - window["high"].max()) / risk
        mae.append(worst)
    return np.asarray(mae, dtype=float)


def _simulate_fixed_stop(row, prepared: dict[str, pd.DataFrame], stop_distance_fn, confirm_close: bool):
    risk = _entry_risk(row)
    if risk is None:
        return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"
    side = 1 if row.side == "long" else -1
    entry_px = row.entry_px
    stop_distance = stop_distance_fn(risk, row.atr_at_entry)
    stop_level = entry_px - stop_distance if side == 1 else entry_px + stop_distance

    df = prepared[row.symbol]
    window = df[(df.index > row.entry_date) & (df.index <= row.exit_date)]
    if len(window) == 0:
        return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"

    for dt, bar in window.iterrows():
        if confirm_close:
            breached = (bar["close"] <= stop_level) if side == 1 else (bar["close"] >= stop_level)
            exit_px = bar["close"]
        else:
            breached = (bar["low"] <= stop_level) if side == 1 else (bar["high"] >= stop_level)
            exit_px = stop_level
        if breached:
            pnl = (exit_px - entry_px) if side == 1 else (entry_px - exit_px)
            return pnl / risk, dt, exit_px, ("STOP_CLOSE" if confirm_close else "STOP")

    return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"


def _simulate_time_stop(row, prepared: dict[str, pd.DataFrame], n_days: int):
    risk = _entry_risk(row)
    if risk is None:
        return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"
    side = 1 if row.side == "long" else -1
    entry_px = row.entry_px
    cutoff = row.entry_date + pd.Timedelta(days=n_days)
    if cutoff >= row.exit_date:
        return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"

    df = prepared[row.symbol]
    window = df[(df.index > row.entry_date) & (df.index <= cutoff)]
    if len(window) == 0:
        return row.net_r, row.exit_date, row.exit_px, "ORIGINAL"
    exit_dt = window.index[-1]
    exit_px = window["close"].iloc[-1]
    pnl = (exit_px - entry_px) if side == 1 else (entry_px - exit_px)
    return pnl / risk, exit_dt, exit_px, "TIME_STOP"


def apply_fixed_r_stop(trades: pd.DataFrame, prepared: dict, r_mult: float,
                        confirm_close: bool = False) -> pd.DataFrame:
    fn = lambda risk, atr: r_mult * risk
    results = [_simulate_fixed_stop(row, prepared, fn, confirm_close) for row in trades.itertuples()]
    out = trades.copy()
    out["net_r"], out["exit_date"], out["exit_px"], out["exit_reason"] = zip(*results)
    return out


def apply_atr_mult_stop(trades: pd.DataFrame, prepared: dict, atr_mult: float) -> pd.DataFrame:
    fn = lambda risk, atr: atr_mult * atr
    results = [_simulate_fixed_stop(row, prepared, fn, confirm_close=False) for row in trades.itertuples()]
    out = trades.copy()
    out["net_r"], out["exit_date"], out["exit_px"], out["exit_reason"] = zip(*results)
    return out


def apply_time_stop(trades: pd.DataFrame, prepared: dict, n_days: int) -> pd.DataFrame:
    results = [_simulate_time_stop(row, prepared, n_days) for row in trades.itertuples()]
    out = trades.copy()
    out["net_r"], out["exit_date"], out["exit_px"], out["exit_reason"] = zip(*results)
    return out
