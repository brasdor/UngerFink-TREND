# FROZEN CONFIG -- DO NOT MODIFY AFTER T8
# Method  : MACrossShort_FundingGate (System 8)
# Frozen  : 2026-06-12
# Research: T1 sweep -> T2 -> T3 -> T4 (8/9) -> T15 -> re-canonicalized
#           -> T2v2 (7/7) -> T3v2 -> T4v2 (8/9) -> T5 -> T6 -> T7 -> T8
# CAVEAT  : 2026 partial -31.6R (monitor in T9B); win rate 47.2% above 45%
#           gate -- structural (funding gate selects crowded shorts, S7=45.6%)

METHOD             = "MACrossShort_FundingGate"
TIMEFRAME          = "4H"
EXCHANGE           = "binance_futures_usdm"
SIDE               = "SHORT"
LEVERAGE           = 1.0

FAST_EMA           = 20        # fast EMA crosses below slow EMA on bar close
SLOW_EMA           = 30
EMA_N              = 200       # close must be BELOW EMA200
FUNDING_THRESHOLD  = 0.0001    # 0.01% per 8h -- discovery constraint (System 7 class)
ATR_N              = 14
ATR_STOP_MULT      = 2.0       # stop = ATR x 2.0 ABOVE entry
HOLD_BARS          = 35        # time exit after 35 x 4H bars (~5.8 calendar days)

PORTFOLIO_CAP      = None      # uncapped (System 7 precedent -- caps cut total R)
RISK_PER_TRADE_PCT = 0.0025
CAPITAL_CEILING    = 150.0
INITIAL_CAPITAL    = 10_000.0
KILL_SWITCH_DD_PCT = 35.0

UNIVERSE_SIZE      = 106
DATA_PATH_4H       = "data/futures_universe/ohlcv_4h/"
DATA_PATH_FUNDING  = "data/futures_universe/funding_rates/"

PERF_TRADES        = 3572
PERF_AVG_R         = 0.3145
PERF_WIN_RATE      = 0.4720
PERF_PROFIT_FACTOR = 1.6594
PERF_T_STAT        = 11.3
PERF_TOTAL_R       = 1123.5
PERF_CAGR          = 0.4635
PERF_MAX_DD        = -0.2096
PERF_FINAL_EQ      = 119845.48
