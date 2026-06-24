# FROZEN CONFIG -- DO NOT MODIFY AFTER T8
# Method  : RSI_MR_Futures_FundingGate
# Frozen  : 2026-06-22
# Research: T1 (68/72 pass) -> T2 (t=7.22) -> T3B (time exit wins) -> T4 (all pass) -> T5/T6/T7 -> T8

METHOD             = "RSI_MR_Futures_FundingGate"
TIMEFRAME          = "1D"
EXCHANGE           = "binance_futures_usdm"
SIDE               = "LONG"
LEVERAGE           = 1.0

RSI_N              = 10
OVERSOLD_THRESHOLD = 20
FUNDING_GATE       = 'negative'    # funding_rate < 0 required at entry
ATR_N              = 14
ATR_STOP_MULT      = 2.0
HOLD_BARS          = 25
EXIT_MODE          = 'time_only'    # T3B confirmed: time exit dominates MR

FILTER_MODE        = 'none'
PORTFOLIO_CAP      = None           # uncapped
RISK_PER_TRADE_PCT = 0.0025
CAPITAL_CEILING    = 150.0
INITIAL_CAPITAL    = 10000.0
KILL_SWITCH_DD_PCT = 35.0

PERF_TRADES        = 257
PERF_AVG_R         = 0.6131
PERF_WIN_RATE      = 0.6693
PERF_PROFIT_FACTOR = 3.4626
PERF_T_STAT        = 5.28
PERF_TOTAL_R       = 157.6
PERF_CAGR          = 0.1026
PERF_MAX_DD        = -0.0626
PERF_FINAL_EQ      = 14782.45