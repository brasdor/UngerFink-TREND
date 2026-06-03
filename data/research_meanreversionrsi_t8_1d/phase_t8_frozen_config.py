#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FROZEN CONFIG -- MeanReversionRSI 1D
UngerFink Pipeline Phase T8
Generated: 2026-06-01

DO NOT MODIFY after freeze. Any changes require full restart from T1.
"""

SYSTEM_NAME    = "MeanReversionRSI"
SYSTEM_TYPE    = "MEAN_REVERSION"
FROZEN_DATE    = "2026-06-01"
PIPELINE_PHASE = "T8_FROZEN"

# =============================================================================
# ENTRY
# =============================================================================
TIMEFRAME      = "1D"
RSI_N          = 14
OVERSOLD_THR   = 25              # RSI(14) crosses BELOW 25 -> enter LONG at next bar open
FILTER_MODE    = "none"          # EMA200 filter confirmed net-negative across ALL timeframes

# =============================================================================
# EXIT
# =============================================================================
EXIT_MODE      = "time_only"     # Variant E: fixed hold period - NO RSI exit, NO profit target
TIME_EXIT_BARS = 20              # hold exactly 20 daily bars then close at market open

ATR_STOP_MULT  = 3.0             # safety stop: entry_price - 3 * ATR(14)
ATR_STOP_NOTE  = (               # SAFETY NET ONLY -- never the primary exit trigger
    "ATR stop is a catastrophic-loss guard only. "
    "Primary exit is ALWAYS time_exit (20 bars). "
    "Do NOT use ATR stop as primary exit -- Variant B showed 0% win rate."
)

# =============================================================================
# PORTFOLIO
# =============================================================================
MAX_CONCURRENT = 10              # max 10 simultaneous open positions
SIDE           = "LONG"
LEVERAGE       = 1.0             # Binance Spot -- no leverage

# =============================================================================
# CAPITAL
# =============================================================================
STARTING_CAPITAL    = 10_000    # USD reference capital
RISK_PER_TRADE      = 0.0025    # 0.25% of current equity per trade (current setting)
KILL_SWITCH_DD      = 0.35      # halt ALL trading if equity drops 35% from peak

RISK_SIZING_NOTE = (
    "0.25% risk produces +2.65% CAGR standalone. "
    "Primary value is portfolio diversification: near-zero DD (1.99%), "
    "negative correlation to Donchian trend system. "
    "REVIEW to 1.0% risk after T9B >= 3 months live confirmation. "
    "At 1.0% risk: estimated +10-11% CAGR, max DD ~8%."
)

# =============================================================================
# UNIVERSE
# =============================================================================
EXCHANGE       = "binance_spot"
UNIVERSE       = "60-symbol original crypto universe (LONG only)"
DATA_DIR       = "data/raw_trend_t1"

# =============================================================================
# VERIFIED PERFORMANCE (T1-T7 pipeline)
# =============================================================================
PERFORMANCE = {
    "risk_per_trade":              "0.25%",
    "cagr":                        "+2.65%",
    "estimated_cagr_at_1pct_risk": "+10-11%",
    "max_dd_pct":                  "1.99%",
    "max_dd_usd":                  "$231 on $10k",
    "kill_switch_fired":           False,
    "avg_r":                       "+0.285R",
    "win_rate":                    "57.4%",
    "profit_factor":               2.37,
    "t_score":                     4.96,
    "mc_p05_total_r":              "+33.8R (block=20, 2000 runs)",
    "prob_positive_mc":            "99.9-100%",
    "2022_bear_result":            "+$495 (positive in bear market)",
    "2022_bear_note":              "Time exit captures bounces, exits before next leg down",
}

# =============================================================================
# CAVEAT FLAGS (mandatory monitoring)
# =============================================================================
CAVEATS = {
    "2026_weakness": (
        "Win rate 28.6% in 2026 partial year -- below the 50% floor. "
        "Equity impact: -$157 on $10k. "
        "ACTION: Monitor in T9B. If 2026 full-year win rate stays below 50%, "
        "investigate regime change before raising risk to 1.0%."
    ),
    "risk_sizing": (
        "0.25% risk produces low standalone CAGR (+2.65%). "
        "Primary value is portfolio diversification: near-zero DD, "
        "negative correlation to Donchian trend-following system. "
        "Raise to 1.0% ONLY after T9B >= 3 months live confirmation."
    ),
    "max_concurrent_bear": (
        "Max concurrent positions ever reached: 33 during 2022 bear market selloff. "
        "max10 cap filtered this appropriately. "
        "Bear-market signal quality is HIGH (avg_r +0.432R in 2022). "
        "Consider raising cap to max15 during confirmed bear regimes "
        "(requires regime-detection rule to be defined first)."
    ),
    "2024_win_rate": (
        "Win rate 76.9% in 2024 -- anomalously high vs 57.4% average. "
        "Reversion to mean (~57-60%) expected in future years."
    ),
    "filter_locked": (
        "EMA200 filter tested across 5 timeframes and rejected. "
        "Do NOT re-add without full T1-T4 re-run."
    ),
    "exit_locked": (
        "Variant E (time exit only) beat combined RSI+ATR+time exit in T3MR. "
        "Do NOT add RSI exit condition without T3MR re-run."
    ),
}

# =============================================================================
# PORTFOLIO COMBINATION: MR + DONCHIAN LONG
# =============================================================================
DONCHIAN_LONG_REF = {
    "cagr":    "+16.0%",
    "max_dd":  "-4.4%",
    "avg_r":   "+1.511R",
}

COMBINATION_THESIS = (
    "MR enters on weakness (RSI oversold); Donchian enters on strength (breakout). "
    "Structural negative correlation -> combined portfolio produces smoother equity curve "
    "and lower combined drawdown than either system alone. "
    "MR CAGR is lower standalone but the DD reduction and Sharpe improvement "
    "justify running both systems on the same capital account with separate position limits. "
    "Donchian: up to 3-5 concurrent trend positions. "
    "MeanReversionRSI: up to 10 concurrent MR positions. "
    "Total simultaneous positions on account: up to 15."
)

# =============================================================================
# STATUS
# =============================================================================
STATUS    = "FROZEN_T8 -- approved for T15/T16 then T9B paper trading"
NEXT_STEP = (
    "T15 parameter stability + T16 Monte Carlo (5000 runs). "
    "Then T9B paper trading at 0.25% risk for >= 3 months. "
    "After 3 months: review 2026 win rate, consider raising risk to 1.0%."
)
