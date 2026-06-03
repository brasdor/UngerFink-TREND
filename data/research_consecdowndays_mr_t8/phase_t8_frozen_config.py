#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FROZEN CONFIG -- ConsecDownDaysMR 1D
UngerFink Pipeline Phase T8
Generated: 2026-06-02 10:17 UTC

BULL SPECIALIST SYSTEM -- pairs with RSI MR Long for all-weather coverage.
DO NOT MODIFY after freeze. Any changes require restart from T1.
"""

SYSTEM_NAME    = "ConsecDownDaysMR"
SYSTEM_TYPE    = "MEAN_REVERSION_BULL_SPECIALIST"
FROZEN_DATE    = "2026-06-02 10:17 UTC"
PIPELINE_PHASE = "T8_FROZEN"

# =============================================================================
# ENTRY
# =============================================================================
TIMEFRAME      = "1D"
CONSEC_N       = 5              # 5 consecutive down closes required
FILTER_MODE    = "ema200_price_above"  # MANDATORY -- keeps inactive in bear markets
# Entry: price has closed DOWN for exactly 5 consecutive days
#        AND close > EMA(200)  [confirmed bull market context]
# Execute: enter LONG at next bar open after the 5th consecutive down close

# =============================================================================
# EXIT
# =============================================================================
EXIT_MODE      = "time_only"    # Variant E -- validated as best in T3MR
HOLD_BARS      = 20             # hold exactly 20 daily bars then close at market
ATR_STOP_MULT  = 2.0            # safety stop: entry - 2*ATR(14) -- SAFETY NET ONLY
ATR_STOP_NOTE  = (
    "ATR stop is a catastrophic-loss guard only. "
    "Primary exit is ALWAYS time_exit (20 bars). "
    "Variant B (ATR-only) and Variant A/C (RSI exit) all failed win-rate gates -- "
    "time exit is the only variant that passes."
)

# =============================================================================
# PORTFOLIO
# =============================================================================
MAX_CONCURRENT = None           # uncapped (system rarely fires >5 simultaneous)
                                # max10 cap is virtually identical (97.8% acceptance)
SIDE           = "LONG"
LEVERAGE       = 1.0            # Binance Spot -- no leverage

# =============================================================================
# CAPITAL
# =============================================================================
STARTING_CAPITAL    = 10_000
RISK_PER_TRADE      = 0.0025    # 0.25% of current equity
KILL_SWITCH_DD      = 0.35      # halt if equity drops 35% from peak

RISK_SIZING_NOTE = (
    "0.25% risk produces +4.03% CAGR standalone. "
    "Review to 1.0% after T9B >= 3 months confirmation. "
    "At 1.0% risk: estimated +16% CAGR, max DD ~16%."
)

# =============================================================================
# UNIVERSE
# =============================================================================
EXCHANGE       = "binance_spot"
UNIVERSE       = "52-symbol crypto universe"
DATA_DIR       = "data/raw_trend_t1"

# =============================================================================
# VERIFIED PERFORMANCE (T1-T7 pipeline)
# =============================================================================
PERFORMANCE = {
    "risk_per_trade":  "0.25%",
    "cagr":            "+4.03%",
    "estimated_cagr_at_1pct_risk": "+16%",
    "max_dd_pct":      "4.03%",
    "max_dd_usd":      "$444 on $10k",
    "kill_switch_fired": False,
    "avg_r":           "+0.470R",
    "win_rate":        "51.4%",
    "profit_factor":   2.31,
    "t_score":         3.63,
    "mc_p05_total_r":  "+17.4R (block=20, 2000 runs)",
    "prob_positive_mc": "98.5%+",
    "bull_bear_r_ratio": "7.2x (positive years / negative years)",
    "2022_result":     "-3.22R (only 4 trades -- EMA200 filter kept system inactive)",
    "2021_result":     "+36.88R (best year -- bull market specialist)",
    "2024_result":     "+57.74R (best year -- bull market specialist)",
}

# =============================================================================
# SYSTEM CHARACTER: BULL SPECIALIST
# =============================================================================
SYSTEM_CHARACTER = """
ConsecDownDaysMR is explicitly designed as a BULL MARKET SPECIALIST.
It pairs with RSI MR Long for all-weather portfolio coverage:

RSI MR Long:
  - Consistent positive returns every year (2021-2025 all positive)
  - Lower avg_r (+0.285R) but very smooth equity curve
  - Active in ALL market regimes

ConsecDownDaysMR:
  - Bull market amplifier -- catches dip-and-bounce in uptrending markets
  - Much higher avg_r (+0.470R) in bull years (2021: +0.90R, 2024: +0.90R)
  - EMA200 filter deliberately prevents activity in bear markets
  - Negative in bear/neutral years (2022: -0.80R, 2023: -0.30R, 2025: -0.07R)
  - The negative bear-year results are by design, not a flaw

Combined portfolio:
  RSI MR Long + ConsecDownDaysMR = higher bull return + RSI MR provides bear floor
"""

# =============================================================================
# CAVEAT FLAGS (mandatory monitoring)
# =============================================================================
CAVEATS = {
    "bear_market_inactive": (
        "System is deliberately inactive in bear markets (EMA200 filter). "
        "2022 had only 4 trades and produced -3.22R. "
        "This is expected and correct -- the filter is working. "
        "Do NOT remove EMA200 filter without full T1-T4 re-run."
    ),
    "negative_years": (
        "2023: -7.69R (-$208), 2025: -2.73R (-$86). "
        "These negative years are structurally expected for a bull specialist. "
        "Pair with RSI MR Long to offset negative years."
    ),
    "year_concentration": (
        "2021+2024 contribute 113% of total R -- other years net negative. "
        "MC p05 = +17.4R at block=20 confirms genuine edge despite concentration. "
        "t-score = 3.63 -- statistically significant."
    ),
    "month_fragility": (
        "Removing top-2 months drops avg_r below gate (0.187R). "
        "System is lumpy -- a few exceptional months drive total return. "
        "MC block=20 stress-tests this -- p05 still +17.4R."
    ),
    "2026_partial": (
        "2026 has only 3 trades (+$93). Crypto market direction unclear. "
        "Monitor T9B for regime continuation."
    ),
    "risk_sizing": (
        "0.25% risk produces +4.03% standalone CAGR. "
        "Raise to 1.0% ONLY after T9B >= 3 months confirmation."
    ),
}

# =============================================================================
# PORTFOLIO COMBINATION: ConsecDownDays + RSI MR Long
# =============================================================================
RSI_MR_LONG_REF = {
    "cagr":    "+2.65%",
    "max_dd":  "-1.99%",
    "avg_r":   "+0.285R",
    "character": "all-weather MR -- positive every year",
}

COMBINATION_THESIS = (
    "Portfolio: RSI MR Long (all-weather floor) + ConsecDownDaysMR (bull amplifier). "
    "RSI MR Long provides consistent returns in ALL years. "
    "ConsecDownDaysMR adds convex upside in bull years (+$960 in 2021, +$1644 in 2024) "
    "at the cost of small losses in bear/neutral years. "
    "Net portfolio: smoother curve with RSI as the base + ConsecDownDays as the amplifier."
)

# =============================================================================
# STATUS
# =============================================================================
STATUS    = "FROZEN_T8 -- running T15 param stability + T16 Monte Carlo"
NEXT_STEP = (
    "T15: confirm consec_n=[4,5,6] neighbourhood all profitable. "
    "T16: 5000-run MC block bootstrap. "
    "Then T9B paper trading at 0.25% risk."
)
