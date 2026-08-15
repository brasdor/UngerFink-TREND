#!/usr/bin/env python3
"""
T6 capital/execution realism engine, generic across candidates -- entry AND
exit both frozen (T3/T4/T5 closed). Builds on T5's confirmed caps ($60k,
max_open=20, max_position_pct=10%, max_cluster_pct=30%) and layers on
execution-realism checks the repo's existing T6 script
(research/phase_t6_capital_execution_engine_V2.py) does NOT cover -- that
script is explicitly spot/1x/no-slippage ("Binance Spot LONG only -- no
leverage, no margin", "Keep EXTRA_EXECUTION_COST_R at 0 for first pass"). Our
candidates trade the futures universe with both sides, so genuinely new
infrastructure is needed for:

  1. Liquidity-tiered slippage (not a flat bps assumption): 20-day rolling
     dollar ADV (close*volume, shifted 1 day to avoid lookahead) buckets each
     symbol at each entry into tier1 (>=$50M ADV, e.g. BTC/ETH/majors,
     2bps/fill), tier2 ($5M-$50M, 8bps/fill), tier3 (<$5M, illiquid alts,
     25bps/fill). Slippage applied at BOTH entry and exit (round-trip = 2x
     tier bps), converted into R-units via the trade's own stop distance --
     the same mechanic as T4's cost_stress, so tight-stop (large notional)
     trades correctly bear proportionally more slippage cost.
  2. Market vs limit fill test: baseline convention already fills at next-bar
     open (T1 convention). A limit alternative --resting order placed
     limit_edge_bps inside the open, filled only if that bar's low/high
     reaches it-- is tested as a diagnostic: better price when filled, but
     some entries are simply missed.
  3. Futures margin/liquidation mechanics: isolated margin, liquidation price
     = entry_px * (1 -/+ 1/leverage -+ maintenance_margin_rate). Scans each
     trade's actual intrabar low/high (not just close-to-close) over its full
     holding window to check whether liquidation would have triggered before
     the modeled stop/exit. Tested across a leverage sensitivity sweep
     (3x/5x/10x) since liquidation buffer is a pure function of leverage,
     independent of the $60k portfolio-level position sizing.
  4. Position-sizing sanity check: reports actual $ notional distribution at
     entry to confirm the 0.25%-risk convention never produces a notional
     below a realistic Binance futures minimum-notional floor ($5-100).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ATR_MULT_R = 2.0
CAPITAL_BASE = 60_000.0
RISK_PCT_PER_TRADE = 0.0025

TIER1_ADV_USD = 50_000_000.0
TIER2_ADV_USD = 5_000_000.0
TIER_BPS = {"tier1": 2.0, "tier2": 8.0, "tier3": 25.0}

LIMIT_EDGE_BPS = 15.0
MIN_NOTIONAL_USD = 100.0  # conservative Binance USDT-M futures floor

MAINTENANCE_MARGIN_RATE = 0.005  # flat conservative proxy across tiers
LEVERAGE_SWEEP = (3, 5, 10)


def add_adv(prepared: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """20-day rolling dollar ADV, shifted 1 day so entry-day sizing never
    looks ahead into same-day volume."""
    out = {}
    for sym, df in prepared.items():
        df = df.copy()
        df["adv_usd"] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean().shift(1)
        out[sym] = df
    return out


def liquidity_tier(adv_usd: float) -> str:
    if not np.isfinite(adv_usd):
        return "tier3"
    if adv_usd >= TIER1_ADV_USD:
        return "tier1"
    if adv_usd >= TIER2_ADV_USD:
        return "tier2"
    return "tier3"


def apply_execution_realism(trades: pd.DataFrame, prepared_adv: dict[str, pd.DataFrame],
                             leverage: int) -> pd.DataFrame:
    """Per-trade: liquidity tier, slippage-adjusted net_r, limit-fillability,
    liquidation check against intrabar low/high over the full holding window."""
    t = trades.copy()
    stop_distance_pct = (ATR_MULT_R * t["atr_at_entry"] / t["entry_px"]).clip(lower=1e-9)

    tiers, slip_bps_rt, net_r_slip = [], [], []
    limit_fillable = []
    liq_triggered = []

    mmr = MAINTENANCE_MARGIN_RATE
    liq_buffer = 1.0 / leverage - mmr

    for i, row in t.iterrows():
        sym = row["symbol"]
        adv_df = prepared_adv[sym]
        adv_at_entry = adv_df.loc[row["entry_date"], "adv_usd"] if row["entry_date"] in adv_df.index else np.nan
        tier = liquidity_tier(adv_at_entry)
        bps_rt = 2.0 * TIER_BPS[tier]
        extra_cost_r = (bps_rt / 10_000.0) / stop_distance_pct[i]
        r_after_slip = row["net_r"] - extra_cost_r

        tiers.append(tier)
        slip_bps_rt.append(bps_rt)
        net_r_slip.append(r_after_slip)

        side = 1 if row["side"] == "long" else -1
        entry_px = row["entry_px"]
        window = adv_df[(adv_df.index > row["entry_date"]) & (adv_df.index <= row["exit_date"])]

        limit_px = entry_px * (1 - LIMIT_EDGE_BPS / 10_000.0 * side)
        entry_bar = adv_df[adv_df.index == row["entry_date"]]
        if side == 1:
            fillable = bool(len(entry_bar) and entry_bar["low"].iloc[0] <= limit_px)
        else:
            fillable = bool(len(entry_bar) and entry_bar["high"].iloc[0] >= limit_px)
        limit_fillable.append(fillable)

        if len(window) == 0 or liq_buffer <= 0:
            liq_triggered.append(False)
        elif side == 1:
            liq_price = entry_px * (1 - liq_buffer)
            liq_triggered.append(bool(window["low"].min() <= liq_price))
        else:
            liq_price = entry_px * (1 + liq_buffer)
            liq_triggered.append(bool(window["high"].max() >= liq_price))

    t["liquidity_tier"] = tiers
    t["slippage_bps_roundtrip"] = slip_bps_rt
    t["net_r_after_slippage"] = net_r_slip
    t["limit_fillable"] = limit_fillable
    t[f"liquidation_lev{leverage}x"] = liq_triggered
    t["stop_distance_pct"] = stop_distance_pct.values
    return t


def position_sizing_check(trades: pd.DataFrame) -> dict:
    stop_distance_pct = (ATR_MULT_R * trades["atr_at_entry"] / trades["entry_px"]).clip(lower=1e-9)
    target_notional_pct = (RISK_PCT_PER_TRADE / stop_distance_pct).clip(upper=0.10)
    notional_usd = target_notional_pct * CAPITAL_BASE
    return dict(
        min_notional_usd=float(notional_usd.min()), p05_notional_usd=float(notional_usd.quantile(0.05)),
        median_notional_usd=float(notional_usd.median()), max_notional_usd=float(notional_usd.max()),
        pct_below_min_floor=float((notional_usd < MIN_NOTIONAL_USD).mean()),
    )
