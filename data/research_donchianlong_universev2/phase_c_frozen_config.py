# FROZEN CONFIG -- DO NOT MODIFY AFTER T8
# Method:  DonchianLong_UniverseV2
# Frozen:  2026-05-29
# Status:  FROZEN -- all T4-T8 gates passed

TIMEFRAME            = "1d"
FILTER_MODE          = "ema200_price"
ENTRY_N              = 20          # Donchian upper band lookback (highest high)
EXIT_N               = 10          # Donchian lower band lookback (= ENTRY_N // 2)
ATR_N                = 14          # ATR period for initial stop and chandelier
ATR_MULT             = 2.0         # initial stop: entry - ATR(14) * 2.0
CHANDELIER_MULT      = 3.0         # chandelier trail: HH - ATR(14) * 3.0
CHANDELIER_ACTIVATE  = 4.0         # chandelier activates at +4R MFE
PORTFOLIO_CAP        = 8           # canonical portfolio cap (max8)
RISK_PER_TRADE       = 0.0025      # 0.25% of capital per trade
LEVERAGE             = 1.0         # Binance Spot -- always 1.0
EXCHANGE             = "Binance_Spot"
SIDE                 = "LONG"

UNIVERSE_VERSION     = "V2"
UNIVERSE_FILE        = "data/universe/filtered_symbols_v2_included_only.csv"
UNIVERSE_SYMBOLS     = 24
FILTER_CONDITIONS    = "avg_r>0 AND trades>=15 AND top_month_pct<60% AND total_r>0"

# T8 scorecard (max8 portfolio)
T8_WIN_RATE          = 0.390   # 39.0%
T8_AVG_R             = 0.9263  # +0.9263R
T8_PF                = 2.695
T8_CAGR_PCT          = 14.3
T8_MAX_DD_PCT        = -3.4
T8_MC_P05_TOTAL_R    = 347.7   # (from full-trades MC; T16 uses max8 trades)
T8_KILL_SWITCH       = False

# Benchmark (original 64-sym frozen pipeline)
BASELINE_AVG_R       = 0.179
BASELINE_PF          = 1.27
BASELINE_CAGR_PCT    = 19.0
BASELINE_MAX_DD_PCT  = -1.78
BASELINE_PORTFOLIO   = "max3"
