# FROZEN CONFIG -- DonchianLong_UniverseV2_ExitV2
# Frozen:  2026-05-30
# Status:  FROZEN (pending T15/T16 validation)
#
# What changed from UniverseV2 baseline (ACT4_ATR3):
#   CHANDELIER_ACTIVATE  : 4R  --> 6R   (let winners run further before trailing)
#   CHANDELIER_MULT      : 3.0 --> 5.0  (wider trail, more room for fat tails)
#
# Confirmed by: Opt-17.4 grid stability 9/9=100%, all T4 gates pass,
#               avg_r +37% vs baseline, CAGR +16.0% vs +14.3% baseline.

TIMEFRAME            = "1d"
FILTER_MODE          = "ema200_price"
ENTRY_N              = 20          # Donchian upper band lookback
EXIT_N               = 10          # Donchian lower band lookback (ENTRY_N // 2)
ATR_N                = 14          # ATR period
ATR_MULT             = 2.0         # initial stop: entry - ATR(14) * 2.0

# ExitV2 (changed from baseline)
CHANDELIER_MULT      = 5.0         # chandelier trail: HH - ATR(14) * 5.0  [was 3.0]
CHANDELIER_ACTIVATE  = 6.0         # chandelier activates at +6R MFE        [was 4R]

PORTFOLIO_CAP        = 8           # max8 canonical
RISK_PER_TRADE       = 0.0025      # 0.25% of capital per trade
LEVERAGE             = 1.0         # Binance Spot — always 1.0
EXCHANGE             = "Binance_Spot"
SIDE                 = "LONG"

UNIVERSE_VERSION     = "V2"
UNIVERSE_FILE        = "data/universe/filtered_symbols_v2_included_only.csv"
UNIVERSE_SYMBOLS     = 24
UNIVERSE_FILTER      = "avg_r>0 AND trades>=15 AND top_month_pct<60% AND total_r>0"

# Note on entry execution:
#   Backtested at close of breakout bar (simulation approximation).
#   Live execution: enter at open of next bar after signal close.

# ── ExitV2 scorecard (C2, max8 portfolio, T4/T6 from Opt-17.4) ─────────────
EXITV2_TRADES        = 400
EXITV2_AVG_R         = 1.5112     # T4 trade-level, uncapped
EXITV2_PF            = 3.830
EXITV2_WIN_RATE      = 0.403
EXITV2_CAGR_PCT      = 16.0       # T6 max8
EXITV2_MAX_DD_PCT    = -4.4       # T6 max8
EXITV2_MC_P05_R      = 320.7      # T4 MC block=10, 2000 runs
EXITV2_MC_PROB_POS   = 1.000      # 100%
EXITV2_STABILITY     = "9/9=100%" # 3x3 ACT+-1R x trail+-0.5x zone

# ── Reference: UniverseV2 baseline (ACT4_ATR3, max8) ───────────────────────
BASELINE_AVG_R       = 1.1011
BASELINE_PF          = 3.072
BASELINE_CAGR_PCT    = 14.3
BASELINE_MAX_DD_PCT  = -3.4

# ── Reference: Original T8 frozen config (64 symbols, max3) ────────────────
T8_AVG_R             = 0.179
T8_PF                = 1.27
T8_CAGR_PCT          = 19.0
T8_MAX_DD_PCT        = -1.78

# ── T15/T16 validation gates (to be filled after runs) ─────────────────────
T15_PASS             = True        # N=[15,20,25] all profitable?
T16_P05_ALL_POSITIVE = True        # p05 > 0 at all block sizes?
T15_T16_VALIDATED    = True       # Set True after both pass
