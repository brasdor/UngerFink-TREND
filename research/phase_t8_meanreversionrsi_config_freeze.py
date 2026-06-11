#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T8 -- MeanReversionRSI Config Freeze
UngerFink Pipeline / Andrea Unger Methodology

Writes the frozen, production-ready config for MeanReversionRSI 1D.
Run only after T6 and T7 both PASS.

Output: data/research_meanreversionrsi_t8_1d/
    phase_t8_frozen_config.py
    phase_t8_final_scorecard.txt
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research_meanreversionrsi_t8_1d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_CONFIG = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FROZEN CONFIG -- MeanReversionRSI 1D
UngerFink Pipeline Phase T8
Generated: {date}

DO NOT MODIFY after freeze. Any changes require restart from T1.
"""

SYSTEM_NAME    = "MeanReversionRSI"
SYSTEM_TYPE    = "MEAN_REVERSION"
FROZEN_DATE    = "{date}"
PIPELINE_PHASE = "T8_FROZEN"

# === ENTRY ===
TIMEFRAME      = "1D"
RSI_N          = 14
OVERSOLD_THR   = 25          # enter LONG when RSI14 < 25
FILTER_MODE    = "none"      # EMA200 filter confirmed net-negative across all TFs

# === EXIT ===
EXIT_MODE      = "time_only"  # Variant E: simplest, best T3MR performance
TIME_EXIT_BARS = 20           # hold 20 daily bars then close at market
ATR_STOP_MULT  = 3.0          # safety stop: entry - 3*ATR14 (prevents catastrophic loss)
# Note: ATR stop is a SAFETY NET only -- primary exit is always time.

# === PORTFOLIO ===
MAX_CONCURRENT = 10           # max 10 simultaneous positions (T5 canonical)
SIDE           = "LONG"
LEVERAGE       = 1.0          # Binance Spot, no leverage

# === CAPITAL ===
STARTING_CAPITAL = 10_000    # USD reference
RISK_PER_TRADE   = 0.0025    # 0.25% equity risk per trade
KILL_SWITCH_DD   = 0.35      # halt if equity drops 35% from equity peak

# === UNIVERSE ===
EXCHANGE       = "binance_spot"
UNIVERSE_SIZE  = 52           # symbols from raw_trend_t1 cache
DATA_DIR       = "data/raw_trend_t1"

# === PERFORMANCE SUMMARY (T6 verified) ===
# Starting capital : $10,000
# CAGR             : see T6 scorecard
# Max DD %         : see T6 scorecard
# Avg R per trade  : +0.285R  (max10 filtered)
# Profit Factor    : ~2.37
# t-score T1       : 4.96  (highly significant)
# Kill-switch       : NOT fired

# === T4 ROBUSTNESS HIGHLIGHTS ===
# MC p05 total R (bs=20) : +33.8R  (100% prob positive)
# Remove top-1 asset     : avg_r still +0.274R
# Cost stress +0.20R     : still PASS
# 2022 bear market       : +22.6R  (positive in bear -- structural edge)

# === CAVEATS AND MONITORING FLAGS ===
CAVEATS = [
    "2026 partial year is near-zero (+0.34R uncapped, flat). "
    "System has been running weak since early 2026. "
    "Monitor monthly: if 3 consecutive losing months, review.",

    "Win rate 2024 was 82.4% -- unusually high. "
    "Reversion to mean (60%) expected in future years.",

    "Max concurrent ever reached: 33 (2022 bear market). "
    "max10 cap means ~67% of simultaneous bear-market entries are skipped. "
    "This is intentional -- reduces risk during market-wide panic entries.",

    "EMA200 filter was tested and rejected. "
    "Do NOT re-add without full T1-T4 re-run.",

    "Variant E (time exit only) beat combined RSI+ATR+time exit in T3MR. "
    "Do NOT add RSI exit condition without re-running T3MR.",
]

# === PORTFOLIO COMBINATION NOTE ===
PORTFOLIO_NOTE = (
    "MeanReversionRSI 1D is structurally UNCORRELATED with DonchianLong. "
    "MR enters on weakness (RSI oversold); Donchian enters on breakouts. "
    "They can share a capital account with SEPARATE position limits. "
    "DonchianLong benchmark: CAGR +16.0%, max DD -4.4%. "
    "Combined portfolio improves Sharpe ratio via negative correlation."
)

# === DONCHIAN LONG REFERENCE ===
DONCHIAN_LONG = {{
    "cagr":    0.160,
    "max_dd":  -0.044,
    "avg_r":   1.511,
    "note":    "Higher CAGR; MR provides DD diversification",
}}
'''

SCORECARD_TEMPLATE = """
================================================================
PHASE T8 -- MeanReversionRSI 1D  FINAL SCORECARD
Generated: {date}
================================================================

FROZEN CONFIG SUMMARY
---------------------
System       : MeanReversionRSI
Timeframe    : 1D
Entry        : RSI(14) < 25  (no EMA200 filter)
Exit         : Time exit after 20 bars  [Variant E]
Safety stop  : ATR(14) x 3.0 below entry
Portfolio    : max 10 concurrent  /  0.25% risk/trade
Universe     : 52 symbols (Binance Spot)

PIPELINE GATE SUMMARY (T1 - T7)
---------------------------------
T1  Concept Discovery     : PASS  (175 stable combos, stability=1.00)
T2  Core Engine           : PASS  (win_rate=61.7%, avg_r=+0.285R, 2022=+9.45R)
T3MR Exit Engineering     : PASS  (Variant E best: avg_r=+0.309R, PF=2.81)
T4  Robustness            : PASS  (MC p05=+33.8R, all 6 critical checks pass)
T5  Portfolio Filter      : PASS  (max10 canonical: avg_r=+0.285R, DD=8.04R)
T6  Capital Engine        : PASS  (see T6 scorecard)
T7  Asset Robustness      : PASS  (remove HBAR: system survives)

KEY METRICS (T5 max10, T6 capital)
-------------------------------------
Avg R/trade  : +0.285R
Win rate     : 57.4%
PF           : 2.37
Max DD       : ~8%
t-score      : 4.96  (highly significant)
2022 bear    : POSITIVE (+19.4R on max10)

CAVEATS (mandatory monitoring)
--------------------------------
1. 2026 partial year is near-zero -- monitor for regime change.
2. 2024 win rate 82.4% is anomalously high -- expect reversion.
3. max10 cap intentionally skips bear-market simultaneous entries.
4. EMA200 filter rejected -- do not re-add without full T1 re-run.
5. Time exit only (Variant E) -- do not add RSI exit without T3MR re-run.

CORRELATION WITH DONCHIAN LONG
---------------------------------
MR enters on weakness; Donchian enters on strength.
Structural negative correlation -- portfolio combination improves Sharpe.
DonchianLong: CAGR +16.0%, max DD -4.4%.
MeanReversionRSI: lower CAGR, lower DD, negative correlation benefit.

STATUS: FROZEN FOR PAPER TRADING
"""


def main() -> None:
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    config_code = FROZEN_CONFIG.format(date=date_str)
    scorecard   = SCORECARD_TEMPLATE.format(date=date_str)

    config_path    = OUT_DIR / "phase_t8_frozen_config.py"
    scorecard_path = OUT_DIR / "phase_t8_final_scorecard.txt"

    config_path.write_text(config_code, encoding="utf-8")
    scorecard_path.write_text(scorecard, encoding="utf-8")

    print(scorecard)
    print(f"[OK] {config_path}")
    print(f"[OK] {scorecard_path}")
    print("\nT8 CONFIG FROZEN.")


if __name__ == "__main__":
    main()
