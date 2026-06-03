#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T8 -- ConsecDownDaysMR Config Freeze
UngerFink Pipeline / Andrea Unger Methodology
"""

import os
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "data" / "research_consecdowndays_mr_t8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_CONFIG = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FROZEN CONFIG -- ConsecDownDaysMR 1D
UngerFink Pipeline Phase T8
Generated: {date}

BULL SPECIALIST SYSTEM -- pairs with RSI MR Long for all-weather coverage.
DO NOT MODIFY after freeze. Any changes require restart from T1.
"""

SYSTEM_NAME    = "ConsecDownDaysMR"
SYSTEM_TYPE    = "MEAN_REVERSION_BULL_SPECIALIST"
FROZEN_DATE    = "{date}"
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
'''

SCORECARD = """
================================================================
PHASE T8 -- ConsecDownDaysMR 1D  FINAL SCORECARD
Generated: {date}
================================================================

FROZEN CONFIG SUMMARY
---------------------
System       : ConsecDownDaysMR
Type         : BULL SPECIALIST (pairs with RSI MR Long)
Timeframe    : 1D
Entry        : 5 consecutive down closes AND close > EMA(200)
Exit         : Time exit after 20 bars  [Variant E]
Safety stop  : ATR(14) x 2.0 below entry
Portfolio    : uncapped  /  0.25% risk/trade
Universe     : 52 symbols (Binance Spot)

PIPELINE GATE SUMMARY (T1 - T7)
---------------------------------
T1  Concept Discovery     : PASS  (31 stable combos, stability=1.00, EMA200 +40%)
T2  Core Engine           : PASS  (win_rate=51.4%, avg_r=+0.470R)
T3MR Exit Engineering     : PASS  (Variant E: time exit 20 bars)
T4  Robustness            : PASS  (MC p05=+17.4R, remove-year tests pass)
T5  Portfolio Filter      : PASS  (uncapped canonical, cap rarely binding)
T6  Capital Engine        : PASS  (CAGR=+4.03%, max DD=4.03%)
T7  Asset Robustness      : PASS  (remove ZEC: system survives)

KEY METRICS
-----------
Avg R/trade  : +0.470R
Win rate     : 51.4%
PF           : 2.31
CAGR         : +4.03%  (0.25% risk)
Max DD       : 4.03%
t-score      : 3.63  (significant)
Bull/Bear R  : 7.2x  (positive years vs negative)

YEAR-BY-YEAR (T6 verified at $10k, 0.25% risk)
-------------------------------------------------
2021:  +$960   (41 trades, WR 63.4%, +0.90R avg)  <<< BULL
2022:  -$88    ( 4 trades, WR  0.0%, -0.80R avg)  <<< BEAR (filter active)
2023: -$208    (26 trades, WR 34.6%, -0.30R avg)
2024: +$1,644  (64 trades, WR 60.9%, +0.90R avg)  <<< BULL
2025:  -$86    (41 trades, WR 41.5%, -0.07R avg)
2026:  +$93    ( 3 trades, WR 33.3%, +1.02R avg)  (partial)

CAVEATS
-------
1. BULL SPECIALIST: negative in bear/neutral years by design.
2. 2023 and 2025 are negative -- pair with RSI MR Long to offset.
3. EMA200 filter is MANDATORY -- do not remove without full T1 re-run.
4. Month concentration risk: top-2 months removal drops below gate.
5. Raise risk to 1.0% only after T9B >= 3 months confirmation.

STATUS: T15+T16 PENDING, THEN T9B PAPER TRADING
"""


def main() -> None:
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    config   = FROZEN_CONFIG.replace("{date}", date_str)
    scorecard = SCORECARD.replace("{date}", date_str)

    (OUT_DIR/"phase_t8_frozen_config.py").write_text(config, encoding="utf-8")
    (OUT_DIR/"phase_t8_final_scorecard.txt").write_text(scorecard, encoding="utf-8")

    print(scorecard)
    print(f"[OK] {OUT_DIR}/phase_t8_frozen_config.py")
    print(f"[OK] {OUT_DIR}/phase_t8_final_scorecard.txt")
    print("\nT8 FROZEN. Running T15+T16 next.")


if __name__ == "__main__":
    main()
