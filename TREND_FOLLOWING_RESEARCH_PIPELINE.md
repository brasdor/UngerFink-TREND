# Trend Following Research Pipeline
## UngerFink Framework — T1 → T8 Systematic Research Protocol

**Version:** 1.1  
**Based on:** Andrea Unger methodology, validated on 1D Donchian/crypto pipeline  
**Purpose:** Repeatable research protocol for any new trend following method  
**Target:** Claude Code or any AI coding assistant  
**Last updated:** T15→T18 validation complete — first pipeline fully validated

---

## VALIDATED REFERENCE CONFIG (Donchian — 1D Crypto)
*(This is the first completed run of this pipeline. Use as benchmark for all future methods.)*

```
METHOD:             Donchian channel breakout
TIMEFRAME:          1d
FILTER:             ema200_price
ENTRY:              Price closes above Donchian N=20 upper band
EXIT:               Donchian N=10 lower band OR Chandelier ACT4_ATR3 (whichever first)
ATR_MULT:           2.0 (initial stop)
CHANDELIER:         Activates at +4R, trails at 3×ATR
PORTFOLIO_CAP:      max3 (preferred) / max5 (alternative)
RISK_PER_TRADE:     0.25% of capital
MAX_HEAT:           0.75% (max3) / 1.25% (max5)
LEVERAGE:           1.0 (Binance Spot, long only)
BACKTEST:           ~4 years (~1500 bars)
TOTAL RETURN:       ~+19% over 4 years (spot, no leverage)
MAX DD:             -1.78% (max3) / -2.73% (max5)
T15 RESULT:         PASS — all 7 N values profitable, zone 100%
T16 RESULT:         PASS with WARN — perturbation warn at 2%, slippage PASS at 0.10R
T18 RESULT:         FAIL — no ATR gate added (structural reason: see Section 11)
STATUS:             FROZEN — ready for T9B paper trading
```

---

## 0. HOW TO USE THIS DOCUMENT

This pipeline is a step-by-step research protocol. Each phase produces output files
that feed the next phase. You never skip a phase. You never proceed to the next phase
unless the current phase passes its gate checks.

**Before starting any new method, fill in Section 1 (Configuration Block).
All scripts must read their parameters from this block — no hardcoded values.**

---

## 1. CONFIGURATION BLOCK
*(Fill this in before writing any code)*

```
METHOD_NAME:        e.g. "DualMA", "KeltnerBreakout", "ATRChannel"
ENTRY_LOGIC:        e.g. "EMA50 crosses above EMA200", "price closes above Keltner upper band"
EXIT_LOGIC:         e.g. "Donchian N//2 lower band OR Chandelier 3×ATR trailing"
FILTER_MODES:       e.g. ["ema200_price", "ema50_slope"]
TIMEFRAMES:         e.g. ["1d", "4h", "6h", "8h"]
PARAM_GRID:         e.g. {fast_n: [5,10,20,30,50], slow_n: [50,100,200]}
ATR_MULTS:          e.g. [2.0, 3.0]
STABILITY_ZONE:     e.g. ±1 step around canonical param (min 3 values)
ASSET_UNIVERSE:     e.g. "same 60-symbol crypto universe as Donchian pipeline"
DATA_ROOT:          e.g. "C:/Users/Jean/UngerFink-TREND/data"
PROJECT_ROOT:       e.g. "C:/Users/Jean/UngerFink-TREND"
EXCHANGE:           e.g. "Binance Spot" (affects leverage, long-only constraint)
RISK_PER_TRADE:     e.g. 0.0025 (0.25%)
MAX_CONCURRENT:     e.g. 3 or 5
BACKTEST_BARS:      e.g. 1500 (≈4 years on 1D)
```

---

## 2. UNGER GATE CHECKS REFERENCE
*(Applied at every phase — never waive these)*

| Gate | Rule | Source |
|------|------|--------|
| §2.1 Stability | Edge must be profitable across ≥67% of the stability zone, not just at the canonical param | Unger §2.1 |
| §4.1 Win rate | 30–45% for trend following. Above 55% = suspect overfitting | Unger §4.1 |
| §4.2 Cost floor | avg_r > 0.15R to cover Binance taker fees | Unger §4.2 |
| §4.7 Concentration | Flag if top-1 asset > 50% of total R. Not a disqualifier but must be noted | Unger §4.7 |
| Drawdown | Max DD must be within acceptable range for the risk profile | General |

---

## 3. PHASE T1 — CONCEPT DISCOVERY

### Purpose
Grid search across all parameter combinations to find if the method has a stable edge.
This is the only phase where you test many combinations. After T1, everything locks down.

### Script name
`phase_t1_{method_name}_concept_discovery.py`

### Input
- OHLCV data for all symbols in ASSET_UNIVERSE
- Bar count: BACKTEST_BARS

### Parameter grid to test
```
For each combination of:
  - timeframe in TIMEFRAMES
  - filter_mode in FILTER_MODES
  - atr_mult in ATR_MULTS
  - each param set in PARAM_GRID

Total combos = len(TIMEFRAMES) × len(FILTER_MODES) × len(ATR_MULTS) × len(PARAM_GRID combos)
```

### Logic to implement per combo
1. Apply filter (e.g. ema200_price: only take longs when close > EMA200)
2. Apply entry logic from ENTRY_LOGIC
3. Apply exit logic from EXIT_LOGIC
4. Record: trades, avg_r, profit_factor, max_dd per combo

### Stability zone check (§2.1)
For each (timeframe, filter_mode, atr_mult) group:
- Identify the canonical param (highest avg_r in zone)
- Define stability zone: canonical ± 1 step in each param dimension
- Count how many zone combos are profitable
- PASS if ≥ 67% of zone combos are profitable
- FAIL otherwise — do NOT proceed with this combo

### Cost floor check (§4.2)
- For every PASS combo: check avg_r > 0.15R
- If avg_r ≤ 0.15R: mark as COST_FAIL even if stability passes

### Output files
```
data/research_{method}_t1/
    phase_t1_stability_ranking.csv      # All combos ranked by stability + avg_r
    phase_t1_n_sensitivity.csv          # Per-param sensitivity for each TF/filter/ATR group
    phase_t1_summary.txt                # Human-readable report with recommended config
```

### T1 gate to pass before proceeding
- At least one (timeframe, filter, atr_mult) combo has PASS stability AND passes §4.2
- If nothing passes: STOP. The method has no edge on this universe. Document and archive.

### Recommended config output
```
CANONICAL_TIMEFRAME:   (from T1 results)
CANONICAL_FILTER:      (from T1 results)
CANONICAL_PARAM:       (from T1 results)
CANONICAL_ATR_MULT:    (from T1 results)
CANONICAL_EXIT_N:      (e.g. CANONICAL_PARAM // 2 for Donchian-style exit)
```

---

## 4. PHASE T2 — CORE ENGINE (PER-SYMBOL RESOLUTION)

### Purpose
Run the canonical config from T1 at full per-symbol resolution.
Validate §4.1 win rate and §4.2 cost floor on real trade-by-trade data.

### Script name
`phase_t2_{method_name}_core_engine.py`

### Input
- T1 canonical config (CANONICAL_TIMEFRAME, CANONICAL_FILTER, CANONICAL_PARAM, CANONICAL_ATR_MULT)
- Full OHLCV data

### Logic
- Run the full backtest with canonical config only
- Record every individual trade with: symbol, entry_time, exit_time, entry_price, exit_price, net_r, side

### Gate checks
| Check | Target | Action if fail |
|-------|--------|----------------|
| §4.1 Win rate | 30–45% | STOP if outside range |
| §4.2 avg_r | > 0.15R | STOP if below |
| Profit factor | > 1.0 | Flag if below |
| §4.7 Concentration | Note top-3 assets % of total R | Flag, do not stop |

### Output files
```
data/research_{method}_t2/
    phase_t2_trades.csv                 # All trades (one row per trade)
    phase_t2_asset_summary.csv          # Per-symbol breakdown
    phase_t2_equity.csv                 # Cumulative R over time
    phase_t2_summary.csv                # Aggregate metrics
```

---

## 5. PHASE T3B — EXIT ENGINEERING

### Purpose
Improve the exit logic to capture more of the fat tail (big winners).
The Chandelier trailing stop is the primary tool. Combined exit = Donchian channel exit OR Chandelier, whichever triggers first.

### Script name
`phase_t3b_{method_name}_wide_exit_engineering.py`

### Input
- T2 canonical config
- T2 trades as baseline comparison

### Parameters to test
```
CHANDELIER_ATR_MULTS:   e.g. [2.0, 3.0, 4.0]
CHANDELIER_ACTIVATION:  e.g. only activate after trade reaches +X R (e.g. 2R, 3R, 4R)
```

### Logic
- Combined exit: exit when EITHER the Donchian exit channel OR the Chandelier trailing stop is hit
- Chandelier: trailing stop = highest_high_since_entry - (ATR × CHANDELIER_ATR_MULT)
- Compare each variant against T2 baseline on: total_r, avg_r, win_rate, profit_factor, max_dd, trade_count

### Gate checks (must beat T2 baseline)
| Metric | Target |
|--------|--------|
| Total R | > T2 total_r |
| avg_r | > 0.15R (§4.2 still applies) |
| Win rate | Still 30–45% (§4.1) |
| Max DD | Not catastrophically worse than T2 |

### Key diagnostic
- Check fat tail impact: compare T2 vs T3B on top-5 symbols by R
- If Chandelier genuinely extends big winners → architecture is correct
- If total R improves only because of more re-entries (lower avg_r) → note the trade-off

### Output files
```
data/research_{method}_t3b/
    phase_t3b_wide_exit_trades.csv
    phase_t3b_wide_exit_asset_summary.csv
    phase_t3b_wide_exit_equity.csv
    phase_t3b_wide_exit_summary.csv
```

### Freeze canonical exit config after T3B
```
CANONICAL_EXIT:         combined Donchian + Chandelier
CHANDELIER_ATR_MULT:    (from T3B results)
CHANDELIER_ACTIVATION:  (from T3B results)
```

---

## 6. PHASE T4 — ROBUSTNESS ENGINE

### Purpose
Try to break the system. Stress-test the T3B results across multiple dimensions.
This phase does NOT change entry/exit logic — it only analyzes closed trades from T3B.

### Script name
`phase_t4_{method_name}_robustness_engine.py`

### Input
`data/research_{method}_t3b/phase_t3b_wide_exit_trades.csv`

### Tests to run

**1. Baseline summary**
- Total R, avg R, win rate, PF, max DD, losing streak, t-score
- Grouped by: timeframe, side (LONG/SHORT), ALL

**2. Block bootstrap Monte Carlo**
- 2000 runs, block sizes: [1, 3, 5, 10, 20]
- Output: total_r p05/p50/p95, DD p05/p50/p95, PF p05/p50/p95, prob_positive

**3. Extra cost stress**
- Apply additional cost per trade: [0.00, 0.02, 0.05, 0.10, 0.15, 0.20] R
- Edge should survive at least 0.05R extra cost

**4. Rolling calendar windows**
- Window: 180 days, step: 90 days
- Shows if edge is consistent over time or concentrated in one period

**5. Period splits**
- First half vs second half (by trade count and by time)
- Last 100 trades, last 200 trades
- Recent performance should not be structurally worse than historical

**6. Remove best assets**
- Remove top [1, 3, 5, 10] assets by total R
- System should survive removal of top 1-3 assets

**7. Remove best months**
- Remove top [1, 2, 3] months by total R
- System should not depend on a single month

**8. Asset concentration**
- top1/top3/top5/top10 as % of total R
- Flag if top1 > 50% of total R (§4.7)

### Gate checks
| Test | Pass condition |
|------|----------------|
| MC p05 total R | Positive or near zero |
| Cost stress 0.05R | PF still > 1.0 |
| Remove top-1 asset | System still profitable |
| Remove top-1 month | System still profitable |
| Rolling windows | Multiple profitable windows, not one isolated spike |

### Output files
```
data/research_{method}_t4/
    phase_t4_baseline_summary.csv
    phase_t4_montecarlo_summary.csv
    phase_t4_cost_stress_summary.csv
    phase_t4_rolling_windows.csv
    phase_t4_period_splits.csv
    phase_t4_remove_best_assets.csv
    phase_t4_remove_best_months.csv
    phase_t4_asset_concentration.csv
    phase_t4_master_report.txt
```

---

## 7. PHASE T5 — PORTFOLIO FILTER

### Purpose
Apply a concurrent position cap to simulate realistic portfolio management.
Tests whether limiting simultaneous open trades improves risk-adjusted returns.

### Script name
`phase_t5_{method_name}_portfolio_filter.py`

### Input
`data/research_{method}_t3b/phase_t3b_wide_exit_trades.csv`

### Variants to test
```
PORTFOLIO_CAPS:     ["uncapped", "max3", "max5", "max8", "max10"]
SIDE:               LONG only (for Binance Spot)
```

### Logic
- For each cap variant: replay trades in chronological order
- When a new signal fires, only accept it if current open positions < cap
- Queue or discard signals that exceed the cap
- Track: accepted trades, total R, avg R, PF, max DD

### Scorecard format
```
Variant | Trades | Accept% | Total R | Avg R | PF | DD
```

### Selection criteria
- Best balance of: total R retained, drawdown reduction, trade count (statistical confidence)
- Minimum 80 trades for statistical confidence at this stage
- Note: max3 often has best per-trade metrics but fewest trades; sweet spot typically max5

### Output files
```
data/research_{method}_t5/
    phase_t5_portfolio_summary.csv      # All variants scorecard
    phase_t5_trades_{variant}.csv       # Accepted trades per variant
```

### Freeze canonical portfolio config
```
CANONICAL_PORTFOLIO_CAP:    (from T5 results, e.g. max3 or max5)
MAX_PORTFOLIO_HEAT:         CANONICAL_PORTFOLIO_CAP × RISK_PER_TRADE
```

---

## 8. PHASE T6 — CAPITAL EXECUTION ENGINE

### Purpose
Translate R-based results into real USDT equity curves with realistic position sizing.
Validates the system with actual dollar amounts and checks the kill-switch.

### Script name
`phase_t6_{method_name}_capital_execution_engine.py`

### Input
- T5 accepted trades per variant
- Starting capital: e.g. $10,000
- RISK_PER_TRADE: e.g. 0.0025 (0.25%)
- LEVERAGE: **always 1.0 for Binance Spot** (long-only, no margin)
- KILL_SWITCH_DD: e.g. -0.35 (halt trading if equity drops 35% from peak)

### CRITICAL CHECK BEFORE RUNNING
```
Verify LEVERAGE = 1.0 in script config
Binance Spot = no margin = LEVERAGE must be 1.0, never higher
```

### Logic
- Position size per trade = (capital × RISK_PER_TRADE) / (entry_price × ATR_stop_distance)
- Simulate equity curve trade by trade
- Apply kill-switch: if drawdown from peak exceeds KILL_SWITCH_DD, flag and stop

### Key outputs to review
- Equity curve shape: smooth growth or single spike?
- Max DD in USDT and % of capital
- Whether kill-switch ever fires
- Compare max3 vs max5: which produces better USDT risk-adjusted return?

### Output files
```
data/research_{method}_t6/
    phase_t6_equity_{variant}.csv       # USDT equity curve per variant
    phase_t6_variant_summary.csv        # All variants: return%, max DD%, Sharpe
    phase_t6_master_report.txt
```

---

## 9. PHASE T7 — ASSET-LEVEL ROBUSTNESS

### Purpose
Verify the edge is not carried by a single asset at the portfolio level.
Extends T4's remove-best-assets test with full capital simulation.

### Script name
`phase_t7_{method_name}_asset_robustness.py`

### Input
- T5 canonical variant trades
- T6 capital parameters

### Tests
- Remove top-1, top-3, top-5 assets → re-run T6 capital simulation
- Result must remain profitable after removing top-1 asset
- If removing top-1 asset turns system negative: flag as concentration risk, do not proceed to T8

### Output files
```
data/research_{method}_t7/
    phase_t7_remove_asset_summary.csv
    phase_t7_equity_remove_top1.csv
```

---

## 10. PHASE T8 — FINAL CONFIG FREEZE

### Purpose
Lock the complete configuration. This is the last step before paper trading (T9B).
No further parameter changes are permitted after T8.

### Script name
`phase_t8_{method_name}_config_freeze.py`

### What T8 does
1. Reads T1→T7 output files
2. Validates all gate checks passed
3. Writes the frozen config file
4. Generates the final scorecard

### Frozen config format
```python
# FROZEN CONFIG — DO NOT MODIFY AFTER T8
# Method: {METHOD_NAME}
# Frozen: {date}

TIMEFRAME           = "{CANONICAL_TIMEFRAME}"
FILTER_MODE         = "{CANONICAL_FILTER}"
ENTRY_PARAM         = {CANONICAL_PARAM}         # e.g. N=20 for Donchian
ATR_MULT            = {CANONICAL_ATR_MULT}       # initial stop
CHANDELIER_MULT     = {CHANDELIER_ATR_MULT}      # trailing stop
CHANDELIER_ACTIVATE = {CHANDELIER_ACTIVATION}    # R threshold to activate
PORTFOLIO_CAP       = {CANONICAL_PORTFOLIO_CAP}  # max concurrent positions
RISK_PER_TRADE      = {RISK_PER_TRADE}           # fraction of capital
LEVERAGE            = 1.0                        # Binance Spot, always 1.0
EXCHANGE            = "{EXCHANGE}"
SIDE                = "LONG"
```

### Final scorecard (T8 must produce this)
```
METHOD:             {METHOD_NAME}
TIMEFRAME:          {CANONICAL_TIMEFRAME}
BACKTEST_BARS:      {BACKTEST_BARS}
TRADES (T5 canon):  {n}
WIN RATE:           {pct}%     [gate: 30–45%]
AVG R:              {r}R       [gate: >0.15R]
PROFIT FACTOR:      {pf}       [gate: >1.0]
MAX DD (R):         {dd}R
MAX DD (%):         {pct}%
TOTAL RETURN:       +{pct}% over {n} years
REMOVE TOP-1 ASSET: {pass/fail}
MC P05 TOTAL R:     {r}R       [gate: positive or near zero]
COST STRESS 0.05R:  PF={pf}    [gate: >1.0]
KILL SWITCH FIRED:  {yes/no}
STATUS:             FROZEN / READY FOR T9B
```

### Output files
```
data/research_{method}_t8/
    phase_t8_frozen_config.py
    phase_t8_final_scorecard.txt
```

---

## 11. PENDING VALIDATION PHASES
*(Run after T8, before graduating to live trading)*

### T15 — Parameter Stability Check
- Re-run T2 with param variants around canonical (e.g. N=15, N=20, N=25 for Donchian)
- All variants must show positive avg_r and pass §4.2
- Win rate must stay in 30–45% range across all variants
- **Critical if win rate at T3B was at the §4.1 floor (30%)**

### T16 — Monte Carlo Completion
- Full Monte Carlo on T5 canonical trades
- 5000+ runs with block sizes [1, 5, 10, 20, 50]
- p05 total R must be positive
- p05 max DD must be within acceptable range for the risk profile
- **This is the gate for reconsidering R sizing (0.25% → 0.5%)**

### T18 — Volatility Gate (optional)
- Add a volatility filter: only accept signals when ATR(14) is in a defined range
- Tests whether filtering out low/high volatility periods improves signal quality
- Run as a T1-style grid on the T3B trades

**LESSON FROM DONCHIAN VALIDATION — T18 FAILED:**
All ATR gate variants (P20, P33, P50) underperformed the no-gate baseline on every
metric. Structural reason: best trades emerge from elevated-ATR trending markets —
the squeeze gate blocks exactly those signals. EMA200 already selects expansionary
regimes; an ATR contraction gate contradicts it.
Rule: if FILTER_MODE = ema200_price, expect T18 to fail. Still run it, but set
expectations accordingly. T18 is more likely to add value with volatility-neutral
filter modes.

---

## 12. AFTER T15/T16/T18 — T9B PAPER TRADING

Once T15, T16, and T18 pass:
1. Configure live signal generator using frozen T8 config
2. Run parallel to any existing paper config (T9A)
3. Minimum paper observation period: 3–6 months before considering live capital
4. Do not modify config during paper period

---

## 13. METHODS BACKLOG — COMPLETE TREND FOLLOWING INVESTIGATION
*(Full scope before switching to Mean Reversion)*

### Timeframe policy (updated)
All methods start on 1D. Expand to shorter timeframes (4H, 6H, 8H, 2H, 1H)
when either:
- The method has intraday-native logic (previous session, opening range, bias)
- 1D fails entirely AND there is theoretical reason to expect shorter TF edge
- The method explicitly requires intraday data (e.g. session high/low breakout)

For daily breakout methods: only expand to shorter TFs if 1D fails completely.
For intraday methods: test all relevant TFs from the start.

---

### TREND FOLLOWING — STATUS BOARD

| # | Method | Category | Timeframes | Status |
|---|--------|----------|------------|--------|
| 1 | Donchian Long | Channel breakout | 1D | ✓ FROZEN |
| 2 | DualMA BTC Long | MA crossover | 1D | ✓ FROZEN (n=5, low confidence) |
| 3 | Donchian Short | Channel breakout | 1D | ✗ HALTED T1 — bear data insufficient |
| 4 | Keltner Long | Channel breakout | 1D | ✗ HALTED T4 — Dec 2024 concentration |
| 5 | Linear Regression | Adaptive breakout | 1D, 4H, 8H | ✗ HALTED T4 — Dec 2024 concentration |
| 6 | ATR Channel | Channel breakout | 1D | ✗ PRE-HALTED — same family as Keltner/LinReg |
| 7 | Previous Session H/L | Intraday breakout | 1H, 2H, 4H | ✗ HALTED T1 — no session structure in crypto |
| 8 | Opening Range Breakout | Intraday breakout | 1H, 2H | ✗ PRE-HALTED — same structural reason as Prev Session |
| 9 | **Intraday Bias** | Time/calendar | 1H, 2H, 4H | ← NEXT (config in 13F) |
| 10 | Intraweek Bias | Calendar | 1D | Pending — after Intraday Bias |
| 11 | DualMA Short | MA crossover | 1D | ✗ Deprioritized — bear data insufficient |
| 12 | Keltner Short | Channel breakout | 1D | ✗ Deprioritized — bear data insufficient |
| — | **Mean Reversion** | Counter-trend | TBD | AFTER all trend following resolved |

---

**NOTE ON SHORT-SIDE METHODS:**
All short-side methods are structurally untestable on the current 4-year
dataset (2021–2025) — crypto bear markets represent only ~25% of the period.
Revisit when either:
- Backtest extended to include 2018–2020 full bear cycle
- Sufficient live bear market data accumulates (12+ months below EMA200)

**NOTE ON ATR CHANNEL:**
Keltner T4 failed due to December 2024 concentration and second-half decay.
ATR Channel uses similar logic (price vs volatility-adjusted band) and will
likely hit the same failure. Still run T1 to confirm — if T1 passes stability
and §4.2, proceed to T2. If T4 shows same concentration pattern, document and
halt. Do not skip — absence of evidence is not evidence of absence.

---

## 13E. NEXT METHOD — LINEAR REGRESSION BREAKOUT
### Full Configuration Block for Claude Code

**What it is:**
Linear regression fits a straight line to the last N closing prices using
least-squares regression. The entry fires when price breaks above the upper
regression channel (regression line + X standard errors). This is more
adaptive than Donchian or Keltner because the channel slope adjusts to the
current trend direction — in a rising market the channel tilts upward,
naturally filtering out weak breakouts.

Unger Academy explicitly covers linear regression adapted to financial markets
to calculate entry levels, presented as a trend following strategy on Gold futures
on a 60-minute timeframe.

```
METHOD_NAME:        LinearRegressionBreakout
ENTRY_LOGIC:        Price closes ABOVE the upper linear regression channel
                    Upper channel = LinReg(close, N) + mult × StdErr(close, N)
                    LinReg = least-squares regression line over last N bars
                    StdErr = standard error of the regression
                    Long entry: close > LinReg(N) + mult × StdErr(N)
FILTER_MODES:       ["ema200_price", "linreg_slope_positive"]
                    ema200_price: same bull regime gate as Donchian
                    linreg_slope_positive: only enter when regression slope > 0
                    (price is trending up within the regression window)
TIMEFRAMES:         ["1d", "4h", "8h"]
                    Start with 1D. Also test 4H and 8H because linear regression
                    is more noise-resistant than pure price channels at shorter
                    timeframes — may pass §4.2 where Donchian failed
PARAM_GRID:
    n:              [10, 15, 20, 25, 30, 40, 55]
                    Regression lookback period
    mult:           [1.0, 1.5, 2.0, 2.5, 3.0]
                    Standard error multiplier — controls channel width
STABILITY_ZONE:     N ∈ [15, 20, 25] for 1D
                    N ∈ [20, 25, 30] for 4H/8H (longer lookback needed for noise)
                    PASS if ≥ 67% of zone combos profitable
ATR_MULTS:          [2.0, 3.0]
                    Initial stop ATR multiplier
EXIT_LOGIC:         Linear regression lower channel OR Chandelier trailing stop
                    Lower channel = LinReg(N) - mult × StdErr(N)
                    Chandelier: activates at +4R, trails at 3×ATR
ASSET_UNIVERSE:     Same 60-symbol crypto universe
DATA_ROOT:          C:/Users/Jean/UngerFink-TREND/data
PROJECT_ROOT:       C:/Users/Jean/UngerFink-TREND
OUTPUT_ROOT:        data/research_linregbreakout_t{N}/
EXCHANGE:           Binance Spot
LEVERAGE:           1.0 (LONG only)
RISK_PER_TRADE:     0.0025 (0.25%)
MAX_CONCURRENT:     3 or 5
BACKTEST_BARS:      1500 (1D) / 6000 (4H) / 4500 (8H) — equivalent 4yr history
SIDE:               LONG only
BENCHMARK:          Donchian Long: avg_r=+0.179R, PF=1.27, CAGR=+19%, DD=-1.78%
```

### Why Linear Regression may succeed where Keltner failed

| Aspect | Keltner | Linear Regression | Why it matters |
|--------|---------|-------------------|----------------|
| Band anchor | EMA (lagged) | Regression line (best fit) | LinReg adapts slope to trend direction |
| False breakout filtering | Moderate | Better — slope filter adds confirmation | linreg_slope_positive gate may reduce December 2024 spike entries |
| Shorter TF viability | Unlikely (failed Donchian) | Possible — regression smooths noise | Worth testing 4H/8H explicitly |
| Parameter sensitivity | 2D (N + km) | 2D (N + mult) | Similar complexity |

### Critical T4 watch items (given Keltner failure)
- **Remove best months**: December 2024 must NOT be a single-month dependency
- **Period splits**: second half must not be negative
- **Remove best assets**: system must survive removal of top 3 assets
- If any of these repeat the Keltner pattern → HALT immediately, document

### Claude Code instructions

```
python engines/pipeline_agent.py --method LinearRegressionBreakout --start-phase T1
```

Or paste this document to Claude Code and say:

> "Start Phase T1 for LinearRegressionBreakout using the configuration in
> Section 13E. Test timeframes 1D, 4H, and 8H — linear regression is more
> noise-resistant than pure price channels so shorter timeframes are worth
> testing. Entry: price closes above LinReg(N) + mult × StdErr(N).
> Filters: ema200_price and linreg_slope_positive.
> §4.2 cost floor: 0.15R (Spot). Output to data/research_linregbreakout_t1/.
> Do not proceed to T2 until I review the stability ranking."

---

## 13F. NEXT METHOD — INTRADAY BIAS
### Full Configuration Block for Claude Code

**What it is:**
Intraday bias exploits recurring price movements at specific hours of the day.
Instead of looking for breakout levels, it asks: does this asset statistically
tend to go up (or down) during a specific hour window, consistently across
the backtest period? If yes, that recurring pattern is the edge.

Unger Academy explicitly identifies this as a distinct strategy category and
notes that Ethereum has a strong intraday bias. Crypto is particularly well-suited
because it trades 24/7 — every hour of the day has data, unlike equity futures
which have defined sessions.

**Critical Unger §2.1 application for bias:**
The stability principle applies differently here. Instead of testing parameter
zones (N ± 1 step), stability is tested across:
- Adjacent hour windows (if hour 14 is best, hours 13 and 15 must also be profitable)
- Multiple assets (bias must appear in multiple symbols, not just 1–2)
- Multiple years (bias must be consistent across 2022, 2023, 2024, 2025)
A bias that only appears in one asset, one year, or one isolated hour is noise.

```
METHOD_NAME:        IntradayBias
ENTRY_LOGIC:        Enter long (or short) at the OPEN of a specific UTC hour
                    Exit at the CLOSE of the same bar (single-bar trade)
                    OR hold for N bars (multi-bar variant)
                    This is NOT a breakout — it is a time-of-day entry
FILTER_MODES:       ["none", "ema200_price", "weekday"]
                    none: pure time bias, no price filter
                    ema200_price: only take long bias signals in bull regime
                    weekday: only trade on specific days (Mon-Fri vs weekend)
TIMEFRAMES:         ["1H", "2H", "4H"]
                    1H: tests all 24 hourly windows individually
                    2H: tests 12 two-hour windows
                    4H: tests 6 four-hour windows
                    Start with 1H — most granular, reveals exact bias hours
PARAM_GRID:
    entry_hour_utc: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
                    For 1H: test each hour independently
                    For 2H: test windows [0-2, 2-4, 4-6, ..., 22-0]
                    For 4H: test windows [0-4, 4-8, 8-12, 12-16, 16-20, 20-0]
    hold_bars:      [1, 2, 3, 4]
                    Number of bars to hold after entry
                    hold_bars=1: enter open, exit close of same bar
                    hold_bars=2: hold for 2 bars, etc.
    side:           ["LONG", "SHORT"]
                    Test both independently — some hours may favor shorts
STABILITY_ZONE:     For each (hour, hold_bars, side) combo:
                    Adjacent hours ± 1 must also be profitable (§2.1 principle)
                    AND bias must appear in ≥ 30% of assets (not just 1–2)
                    AND bias must be profitable in ≥ 3 out of 4 years
                    ALL THREE conditions required for PASS
ATR_MULTS:          N/A for single-bar trades
                    For multi-bar (hold_bars > 1): ATR stop [1.5, 2.0] optional
EXIT_LOGIC:         Primary: bar close after hold_bars
                    Optional: ATR stop for hold_bars > 1 only
ASSET_UNIVERSE:     Same 60-symbol crypto universe
                    Also test BTC/USDT, ETH/USDT separately — Unger notes
                    ETH specifically has strong intraday bias
DATA_ROOT:          C:/Users/Jean/UngerFink-TREND/data
PROJECT_ROOT:       C:/Users/Jean/UngerFink-TREND
OUTPUT_ROOT:        data/research_intradaybias_t{N}/
EXCHANGE:           Binance Spot
LEVERAGE:           1.0
RISK_PER_TRADE:     0.0025 (0.25%)
BACKTEST_BARS:      35040 (1H × 4yr) / 17520 (2H × 4yr) / 8760 (4H × 4yr)
SIDE:               LONG and SHORT (test both — bias can favor either direction)
COST_FLOOR_R:       0.15R §4.2 — critical for single-bar intraday trades
                    Single-bar trades have very short holding period —
                    avg_r must comfortably exceed fees on both entry and exit
```

### What the T1 scan should produce

The T1 output for bias is a **heatmap**, not a ranked list:
- Rows: entry hours (0–23 for 1H)
- Columns: assets or asset groups
- Values: avg_r per (hour, asset) combination
- Color: green = positive, red = negative

Look for:
- **Horizontal bands** (same hour profitable across many assets) → genuine time bias
- **Isolated green cells** (one hour profitable on one asset) → noise, ignore
- **Consistent patterns across years** → robust bias

### Known crypto intraday patterns to watch for
- **US market open (13:30–15:00 UTC)** — BTC often moves with equity open
- **Asian session (01:00–05:00 UTC)** — historically lower volume, mean-reverting
- **Weekend effect** — crypto often sells off Friday evening, recovers Sunday
- **ETH specifically** — Unger Academy notes a strong documented intraday bias

### §4.2 cost floor note for single-bar bias trades
Single-bar intraday trades are the hardest to clear §4.2 because:
- Holding period = 1 bar = 1H/2H/4H
- Two fee events (entry + exit) relative to very short price movement
- avg_r must be > 0.15R on a trade that lasts 1–4 hours

If single-bar trades fail §4.2, try hold_bars=2 or hold_bars=4 before halting.

### Claude Code instructions

```
python engines/pipeline_agent.py --method IntradayBias --start-phase T1
```

Or paste this document to Claude Code and say:

> "Start Phase T1 for IntradayBias using the configuration in Section 13F.
> This is a time-of-day bias scan, NOT a price breakout. For each combination
> of (entry_hour_utc, hold_bars, side, timeframe, filter_mode):
> enter at bar open of that hour, exit after hold_bars bars.
> Test timeframes 1H, 2H, 4H. Test all 24 hourly entry windows on 1H,
> 12 two-hour windows on 2H, 6 four-hour windows on 4H.
> Stability check: adjacent hours ± 1 must also be profitable AND bias must
> appear in ≥ 30% of assets AND must be profitable in ≥ 3 of 4 years.
> §4.2 cost floor: 0.15R. Output to data/research_intradaybias_t1/.
> Produce a heatmap CSV of avg_r by (hour, asset) in addition to the
> standard stability ranking. Do not proceed to T2 until I review."

```
METHOD_NAME:        KeltnerLong
ENTRY_LOGIC:        Price closes ABOVE the upper Keltner band on daily close
                    Upper band = EMA(N) + ATR(N) × keltner_mult
                    Long entry: close > EMA(N) + ATR(N) × keltner_mult
FILTER_MODES:       ["ema200_price"]
                    Same as Donchian — price above EMA200 bull regime gate
                    ema50_slope tested as secondary filter
TIMEFRAMES:         ["1d"]
                    1D only — 4H/6H/8H failed §4.2 on Donchian for structural
                    reasons (fees exceed edge). Same logic applies to Keltner.
                    Only expand if 1D fails entirely.
PARAM_GRID:
    n:              [10, 15, 20, 25, 30, 40, 55]
                    Same N values as Donchian for direct comparison
    keltner_mult:   [1.5, 2.0, 2.5, 3.0]
                    Controls band width — lower = more signals, higher = fewer
STABILITY_ZONE:     N ∈ [15, 20, 25] — same zone as Donchian
                    PASS if ≥ 67% of zone combos profitable
ATR_MULTS:          [2.0, 3.0]
                    Initial stop ATR multiplier — same as Donchian
EXIT_LOGIC:         Keltner lower band (EMA(N) - ATR(N) × keltner_mult)
                    OR Chandelier trailing stop — whichever triggers first
                    Mirror of Donchian exit architecture
CHANDELIER:         Activates at +4R, trails at 3×ATR — same as Donchian
ASSET_UNIVERSE:     Same 60-symbol crypto universe as Donchian pipeline
DATA_ROOT:          C:/Users/Jean/UngerFink-TREND/data
PROJECT_ROOT:       C:/Users/Jean/UngerFink-TREND
OUTPUT_ROOT:        data/research_keltnerlong_t{N}/
EXCHANGE:           Binance Spot
LEVERAGE:           1.0 (LONG only, no margin)
RISK_PER_TRADE:     0.0025 (0.25%)
MAX_CONCURRENT:     3 or 5 (test both in T5)
BACKTEST_BARS:      1500 (≈4 years on 1D)
SIDE:               LONG only
BENCHMARK:          Donchian Long frozen config:
                    avg_r=+0.179R, PF=1.27, CAGR=+19%, max_dd=-1.78% (max3)
```

### Key differences vs Donchian to watch in T1

| Aspect | Donchian | Keltner | Why it matters |
|--------|----------|---------|----------------|
| Band calculation | Price extremes (highest high/lowest low) | EMA ± ATR×mult (volatility-adjusted) | Keltner bands adapt to volatility — may filter false breakouts better |
| Signal frequency | 187–304 trades (4yr, N=20) | Likely similar or fewer | Keltner upper band is harder to breach in low-volatility periods |
| False breakouts | Moderate | Potentially fewer | ATR-adjusted bands are inherently wider during volatile markets |
| Parameter sensitivity | N only | N + keltner_mult (2D grid) | More combinations — stability zone needs both dimensions |
| Key risk | Already validated | Unknown — test needed | May behave similarly to Donchian or find complementary signals |

### What a good T1 result looks like

- At least one (N, keltner_mult) combo passes §2.1 stability AND §4.2 cost floor
- ema200_price filter dominates (consistent with Donchian finding)
- ATR×2.0 initial stop likely dominates (consistent with Donchian finding)
- avg_r comparable to or better than Donchian +0.179R benchmark
- If keltner_mult=1.5 dominates: Keltner is generating more signals than Donchian
- If keltner_mult=3.0 dominates: Keltner is a more selective breakout filter

### Claude Code instructions — how to start

```
python engines/pipeline_agent.py --method KeltnerLong --start-phase T1
```

Or paste this document to Claude Code and say:

> "Start Phase T1 for KeltnerLong using the configuration in Section 13D.
> Script name: phase_t1_keltnerlong_concept_discovery.py
> Entry: price closes above EMA(N) + ATR(N) × keltner_mult.
> Test N = [10,15,20,25,30,40,55] × keltner_mult = [1.5,2.0,2.5,3.0] × 
> filter = [ema200_price, ema50_slope] × ATR_stop = [2.0, 3.0].
> Stability zone: N ∈ [15,20,25], PASS ≥ 67%.
> §4.2 cost floor: 0.15R (Binance Spot).
> Output to data/research_keltnerlong_t1/.
> Do not proceed to T2 until I review the stability ranking."

---

## 13A. NEXT METHOD — DUAL MA CROSSOVER
### Full Configuration Block for Claude Code

```
METHOD_NAME:        DualMA
ENTRY_LOGIC:        Fast EMA crosses above Slow EMA (crossover on daily close)
                    Long entry: fast_ema crosses above slow_ema on bar close
                    Exit: fast_ema crosses below slow_ema OR ATR trailing stop
FILTER_MODES:       ["ema200_price", "ema50_slope"]
                    NOTE: ema200_price here means slow_ema IS the filter —
                    consider using an independent longer EMA (e.g. EMA300) as
                    regime filter to avoid circularity with the entry signal
TIMEFRAMES:         ["1d"]
                    NOTE: 4H/6H/8H failed §4.2 on Donchian — test 1D first.
                    Only expand to other timeframes if 1D fails entirely.
PARAM_GRID:
    fast_ema:       [10, 20, 30, 50]
    slow_ema:       [50, 100, 150, 200]
    CONSTRAINT:     fast_ema must always be < slow_ema
    Total combos:   only valid pairs (10/50, 10/100, 10/150, 10/200,
                    20/50, 20/100, 20/150, 20/200,
                    30/50, 30/100, 30/150, 30/200,
                    50/100, 50/150, 50/200) = 15 combos
ATR_MULTS:          [2.0, 3.0]
STABILITY_ZONE:     For each canonical (fast, slow) pair:
                    zone = fast ± 1 step AND slow ± 1 step
                    e.g. canonical (20, 100): zone includes (10/100, 30/100, 20/50, 20/150)
                    PASS if ≥ 67% of zone combos profitable
EXIT_LOGIC:         Option A (simple): exit when fast_ema crosses below slow_ema
                    Option B (combined): fast_ema cross below slow_ema OR ATR×mult trailing stop
                    Test Option B first — matches T3B Chandelier architecture
CHANDELIER:         Same as Donchian pipeline — activates at +4R, trails at 3×ATR
                    Keep identical to Donchian to allow direct comparison
ASSET_UNIVERSE:     Same 60-symbol crypto universe as Donchian pipeline
DATA_ROOT:          C:/Users/Jean/UngerFink-TREND/data
PROJECT_ROOT:       C:/Users/Jean/UngerFink-TREND
OUTPUT_ROOT:        data/research_dualma_t{N}/
EXCHANGE:           Binance Spot
LEVERAGE:           1.0 (LONG only, no margin — same as Donchian)
RISK_PER_TRADE:     0.0025 (0.25% — same as Donchian for direct comparison)
MAX_CONCURRENT:     3 or 5 (test both in T5 — same as Donchian)
BACKTEST_BARS:      1500 (≈4 years on 1D — same as Donchian)
BENCHMARK:          Compare all results against Donchian frozen config:
                    avg_r=+0.179R, PF=1.27, total_r=+82R, max_dd=-1.78% (max3)
```

### Key differences vs Donchian to watch in T1

| Aspect | Donchian | Dual MA | Why it matters |
|--------|----------|---------|----------------|
| Entry timing | Breakout of N-period high (sharp) | EMA crossover (lagged) | MA crossover enters later — lower avg R expected |
| Win rate | 33–35% | Likely higher (40–50%) | MA systems hold longer, more winners but smaller |
| Signal frequency | 187–304 trades (4yr) | Likely fewer | Crossovers are rare on daily timeframe |
| False signals | Moderate | Higher in ranging markets | EMA crossovers whipsaw badly sideways |
| Fat tail capture | Strong (Chandelier extends) | Uncertain | Test whether Chandelier still helps |

### Critical §4.2 risk for Dual MA
MA crossover systems typically have **lower avg_r than breakout systems** because:
- Entry is lagged (you enter after the move has started)
- Exit is also lagged (you exit after the trend has reversed)

If avg_r < 0.15R at T1, the method fails §4.2 on this universe.
This is the most likely failure mode — be prepared for it.

### Claude Code instructions — how to start

Paste this entire document to Claude Code and say:

> "Implement Phase T1 for the Dual MA crossover method using the configuration
> in Section 13A. Script name: phase_t1_dualma_concept_discovery.py
> Follow the T1 logic in Section 3 exactly. Use the same data loading
> infrastructure as the existing Donchian pipeline scripts. Output to
> data/research_dualma_t1/. Do not proceed to T2 until I review the T1
> stability ranking and confirm at least one combo passes both §2.1 and §4.2."

---

## 13B. NEXT METHOD — SHORT-SIDE DONCHIAN
### Full Configuration Block for Claude Code

**Sequencing note:** Run this pipeline independently (Option A).
Validate the short-side edge in isolation through full T1→T8 before combining
with the long-side Donchian system. Do not assume the short side works just
because the long side does — crypto bear markets have different dynamics.

```
METHOD_NAME:        DonchianShort
ENTRY_LOGIC:        Price closes BELOW Donchian N-period LOW on daily close
                    Short entry: close < lowest_low(N) on bar close
FILTER_MODES:       ["ema200_price_below"]
                    Bear regime gate: only take shorts when close < EMA200
                    This is the MIRROR of the long-side ema200_price filter
TIMEFRAMES:         ["1d"]
                    Same reasoning as long-side — test 1D first
PARAM_GRID:
    donchian_n:     [10, 15, 20, 25, 30, 40, 55]
                    Same N values as long-side T1 for direct comparison
ATR_MULTS:          [2.0, 3.0]
                    Same as long-side for direct comparison
STABILITY_ZONE:     N ∈ [15, 20, 25] — same zone definition as long-side
                    PASS if ≥ 67% of zone profitable
EXIT_LOGIC:         Donchian N//2 UPPER band (mirror of long-side lower band exit)
                    OR Chandelier trailing stop (highest high since entry - ATR×mult)
                    Combined exit — whichever triggers first
CHANDELIER:         Activates at +4R profit, trails at 3×ATR above highest high
                    Mirror of long-side: tracks highest high instead of lowest low
ASSET_UNIVERSE:     Same 60-symbol crypto universe as long-side Donchian
DATA_ROOT:          C:/Users/Jean/UngerFink-TREND/data
PROJECT_ROOT:       C:/Users/Jean/UngerFink-TREND
OUTPUT_ROOT:        data/research_donchianshort_t{N}/
EXCHANGE:           Binance Futures (USD-M perpetuals)
                    NOT Binance Spot — shorting requires margin account
LEVERAGE:           1.0x to start (validate edge before adding leverage)
                    See Section 13C for Binance Futures migration details
RISK_PER_TRADE:     0.0025 (0.25% — same as long-side for direct comparison)
MAX_CONCURRENT:     3 or 5 (test both in T5)
BACKTEST_BARS:      1500 (≈4 years on 1D)
SIDE:               SHORT only (validate in isolation — Option A)
BENCHMARK:          Compare against long-side Donchian frozen config
                    Also compare: are the best short trades in months where
                    long-side is losing? (correlation check at T4)
```

### Key differences vs long-side Donchian to watch in T1

| Aspect | Long-side | Short-side | Why it matters |
|--------|-----------|------------|----------------|
| Market regime | Bull (above EMA200) | Bear (below EMA200) | Crypto spends less time in bear — fewer signals expected |
| Move speed | Slow grind up | Fast crash down | Bear moves are sharper — potentially higher avg R per trade |
| Signal frequency | 187–304 trades (4yr) | Likely 30–50% fewer | Bear periods shorter in 4yr backtest (2021–2025 mostly bull) |
| Funding rates | N/A (Spot) | Paid when short on Futures | Additional cost — raises §4.2 threshold above 0.15R |
| Fat tail profile | ZEC +64R, XRP +44R | Unknown — test needed | Crash moves can be violent and fast |

### Critical §4.2 adjustment for Futures

On Binance Futures, the effective cost per trade is higher than Spot:
- Taker fee: ~0.04% per side (vs ~0.04% Spot) — similar
- **Funding rate: paid every 8 hours when short during contango** — adds ~0.01–0.03R per day held
- For a 1D system holding trades for average 5–10 days: add ~0.05–0.30R extra cost

**Revised §4.2 threshold for Futures short-side: avg_r > 0.25R** (not 0.15R)
Apply this stricter threshold at T1 cost floor check.

### Claude Code instructions — how to start

Paste this entire document to Claude Code and say:

> "Implement Phase T1 for the Short-side Donchian method using the configuration
> in Section 13B. Script name: phase_t1_donchianshort_concept_discovery.py
> The logic is the MIRROR of the existing long-side T1 script — entry on
> Donchian lower band break, filter on price below EMA200, ATR stop above entry.
> Use Futures cost model: §4.2 threshold is 0.25R not 0.15R (funding rate adjustment).
> Output to data/research_donchianshort_t1/. Do not proceed to T2 until I
> review the T1 stability ranking."

---

## 13C. BINANCE FUTURES MIGRATION
### What changes when moving from Spot to Futures

**This section applies to all short-side research and any future leveraged configs.**

#### Account setup differences

| Item | Binance Spot | Binance Futures (USD-M) |
|------|-------------|------------------------|
| Collateral | USDT held directly | USDT margin account (separate) |
| Leverage | 1.0× only | 1×–125× (use 1× to start) |
| Shorting | Not available | Available via perpetual contracts |
| Funding rate | None | Paid/received every 8h |
| Liquidation risk | None | Exists even at 1× if margin falls |
| Fee structure | Spot taker ~0.04% | Futures taker ~0.04% + funding |

#### Script changes required for Futures

Every pipeline script that handles position sizing or P&L must be updated:

```python
# Spot (current)
EXCHANGE        = "binance_spot"
LEVERAGE        = 1.0
FUNDING_RATE    = 0.0
COST_FLOOR_R    = 0.15        # §4.2 threshold

# Futures (short-side)
EXCHANGE        = "binance_futures_usdm"
LEVERAGE        = 1.0         # Always start at 1.0 — validate edge first
FUNDING_RATE    = 0.0001      # 0.01% per 8h = ~0.03% per day estimate
                              # Recalculate from actual Binance data before T2
COST_FLOOR_R    = 0.25        # Raised from 0.15R to account for funding cost
SIDE            = "SHORT"
```

#### Position sizing on Futures

Same ATR-based formula as Spot, but with one addition:
```
position_size = (capital × RISK_PER_TRADE) / (entry_price × ATR × ATR_MULT)
margin_required = position_size × entry_price / LEVERAGE
```
At LEVERAGE=1.0, margin_required = full notional value — same as Spot effectively.

#### Kill-switch on Futures

Keep the same -35% DD kill-switch from T6.
On Futures, also add: **halt if margin ratio drops below 20%** (liquidation buffer).

#### Leverage increase decision tree

Do NOT increase leverage until all of these are true:
1. Short-side pipeline T1→T8 complete and config frozen
2. T9B paper trading running for minimum 3 months
3. T16 Monte Carlo p05 DD confirmed within acceptable range
4. Long-side T9B also running (so you have the hedge before adding leverage)

When ready, increase in steps: 1.0× → 1.5× → 2.0× — never jump to 3×+
Each step requires 1 month of paper observation before the next.

---

## 17. DONCHIAN LONG — OPTIMIZATION EXTENSIONS
*(Formal research extensions of the frozen config — follow pipeline discipline)*

**Rules for all optimizations:**
- Test one change at a time — never combine multiple optimizations in one run
- Each variant benchmarks against frozen config: avg_r=+0.179R, PF=1.27, CAGR=+19%, DD=-1.78%
- A variant only replaces the frozen config if it passes ALL T2→T8 gate checks
- The frozen config remains the production config until a variant is formally frozen
- Correct investigation order: 1 → 2 → 3 → 4 → 5 (each builds on the previous)

---

### 17.1 ASSET UNIVERSE FILTERING & EXPANSION — COMPLETE ✓

**Result: FULLY VALIDATED — replaces original frozen config**

```
DonchianLong_UniverseV2 — FROZEN CONFIG
Universe:    24 symbols (filtered_symbols_v2.csv)
Filter:      avg_r>0, trades≥15, top_month_pct<60%, total_r>0
Timeframe:   1D / ema200_price / N=20 / ATR×2.0 / Chandelier ACT4_ATR3
Portfolio:   max8 (canonical) / max3 (conservative)
Risk/trade:  0.25% (review to 0.50% unlocked — pending T9B ≥3 months)
avg_r:       +1.101R (vs original +0.179R)
PF:          3.072   (vs original 1.27)
CAGR:        +14.3%  (max8, 5.5yr)
Max DD:      -3.4%   (max8)
MC p05:      +148R   (bs=50, 100% prob positive)
T15:         PASS    (N=15/20/25 all profitable, stable zone)
T16:         PASS    (100% prob positive at all block sizes)
Status:      FROZEN — replaces original 64-sym config
Original:    ARCHIVED (not deleted)
R sizing:    0.25%→0.50% review UNLOCKED after T9B ≥3 months
Frozen at:   data/research_donchianlong_universev2/phase_c_frozen_config.py
```

**All subsequent optimizations (17.2→17.5) use UniverseV2 as the new baseline.**

**Step A — Expand universe first (Option C):**
```
ACTION:         Add all available Binance Spot USDT pairs with sufficient history
CRITERIA:       Minimum 1500 1D bars (≈4 years)
                Minimum average daily volume > $1M USDT (liquidity filter)
                Exclude stablecoins, wrapped tokens, leveraged tokens
TARGET:         From current 60 symbols → aim for 100–150 symbols
SCRIPT:         universe_expansion.py
OUTPUT:         data/universe/expanded_symbols.csv
```

**Step B — Filter to profitable subset:**
```
ACTION:         Run T2 on expanded universe, then apply profitability filter
FILTER LOGIC:   Remove any symbol where:
                - avg_r < 0 over full backtest period
                - total_r < 0 AND trades < 5 (insufficient data)
                - max_dd > 20R on that symbol alone (catastrophic single-asset risk)
BENCHMARK:      Current: 17 profitable / 34 losing / 9 neutral out of 60
TARGET:         Filtered universe of ~40–60 symbols with positive expectancy
SCRIPT:         phase_t2_donchian_universe_filter.py
OUTPUT:         data/universe/filtered_symbols.csv
```

**Step C — Rerun full T2→T8 on filtered+expanded universe:**
```
METHOD_NAME:    DonchianLong_UniverseV2
INPUT:          data/universe/filtered_symbols.csv
PIPELINE:       Full T2→T8 with frozen canonical config
                (1D / ema200_price / N=20 / ATR×2.0 / Chandelier ACT4_ATR3)
GATE CHECK:     Must beat frozen config on: avg_r, PF, T4 remove-top-1, T4 remove-top-month
```

**Claude Code instructions:**
> "Implement universe expansion and filtering for DonchianLong following
> Section 17.1 of the pipeline document. Step A: scan all Binance Spot USDT
> pairs, filter to ≥1500 1D bars and >$1M avg daily volume, exclude
> stablecoins/wrapped/leveraged tokens. Step B: run T2 on expanded universe
> using frozen Donchian config, then filter to profitable symbols only.
> Step C: rerun T2→T8 on filtered universe as DonchianLong_UniverseV2.
> Benchmark all results against frozen config."

---

### 17.2 REGIME FILTER VARIANTS
*(After 17.1 — builds on the filtered universe)*

**Test each filter independently against ema200_price baseline:**

```
METHOD_NAME:    DonchianLong_RegimeV2
BASE CONFIG:    Frozen Donchian (1D / N=20 / ATR×2.0 / Chandelier ACT4_ATR3)
UNIVERSE:       Output of 17.1 (filtered+expanded)

FILTER VARIANTS TO TEST (one at a time):

  A. BTC Dominance filter
     Entry only when BTC.D (BTC dominance %) is FALLING (7-day slope < 0)
     Rationale: altcoins trend strongest when BTC dominance is declining

  B. Volume confirmation filter
     Entry only when breakout bar volume > 1.5× average volume (20-bar MA)
     Rationale: genuine breakouts have above-average volume participation

  C. ATR percentile filter
     Entry only when ATR(14) is in 40th–80th percentile of trailing 252 bars
     Rationale: avoid entries in extreme low-vol (false breakout) or
     extreme high-vol (whipsaw) regimes
     Note: T18 failed a simpler ATR gate — this percentile version is
     more nuanced and worth testing separately

  D. Multi-timeframe confirmation
     1W EMA200 must also be in bull regime (price above weekly EMA200)
     Rationale: weekly trend alignment reduces counter-trend daily entries

STABILITY CHECK:
  Each filter variant must be tested across N ∈ [15, 20, 25] (stability zone)
  PASS if all 3 N values remain profitable with the new filter
  FAIL if any zone N turns negative

GATE CHECK:
  avg_r must improve vs frozen config (+0.179R)
  Trade count must not fall below 100 trades (statistical floor)
  T4 remove-top-month must survive (December 2024 test)
```

**Claude Code instructions:**
> "Test regime filter variants A, B, C, D for DonchianLong following
> Section 17.2. Test each filter independently on the 17.1 filtered universe.
> For each: run T2, check stability zone N=[15,20,25], run T4 robustness.
> Benchmark against frozen config avg_r=+0.179R. Output to
> data/research_donchian_regimeV2_{filter}/. Do not combine filters."

---

### 17.3 ENTRY REFINEMENT
*(After 17.2 — builds on best regime filter from 17.2)*

```
METHOD_NAME:    DonchianLong_EntryV2
BASE CONFIG:    Frozen config + best regime filter from 17.2
UNIVERSE:       Output of 17.1

ENTRY VARIANTS TO TEST (one at a time):

  A. Limit order entry (pullback entry)
     Instead of entering at next bar open after breakout signal:
     Place limit order at breakout_level - 0.25×ATR
     If not filled within 2 bars, cancel and wait for next signal
     Rationale: slightly better average entry price, reduces slippage

  B. Volume-confirmed entry
     Enter only on the bar AFTER breakout if that bar also closes above
     the breakout level (confirmation bar)
     Rationale: reduces false breakouts at the cost of slightly worse entry

  C. Partial entry + add-on
     Enter 50% position at breakout
     Add remaining 50% if price moves +1R in favor within 5 bars
     Exit full position on stop
     Rationale: reduces initial risk, adds on confirmed momentum

GATE CHECK:
  avg_r must improve or match frozen config
  Win rate must stay in 24–45% range (crypto-adjusted §4.1)
  T4 period splits: second half must not be negative
```

**Claude Code instructions:**
> "Test entry refinement variants A, B, C for DonchianLong following
> Section 17.3. Use frozen config + best regime filter from 17.2.
> Test each entry variant independently. Run T2→T4 for each.
> Output to data/research_donchian_entryV2_{variant}/."

---

### 17.4 EXIT REFINEMENT — COMPLETE ✓

**Result: C2 adopted — ACT=6R + trail=5.0×ATR**

```
DonchianLong_UniverseV2_ExitV2 — FROZEN CONFIG
Universe:    24 symbols (filtered_symbols_v2.csv)
Timeframe:   1D / ema200_price / N=20 / ATR×2.0 initial stop
Exit:        Donchian N=10 OR Chandelier ACT=6R trail=5.0×ATR
Portfolio:   max8 (canonical) / max3 (conservative)
Risk/trade:  0.25% (review to 0.50% pending T9B ≥3 months)
avg_r:       +1.511R  (vs original +0.179R — 8.4× improvement)
PF:          3.830    (vs original 1.27 — 3× improvement)
CAGR:        +16.0%   (max8, 5.5yr)
Max DD:      -4.4%    (max8)
MC p05:      +200.5R  (bs=50, 100% prob positive at ALL block sizes)
T15:         PASS     (N=15/20/25 monotonically improving)
T16:         PASS     (100% prob positive, p05>+200R at bs=50)
Stability:   9/9 = 100% (strongest in entire pipeline)
Status:      FROZEN — replaces UniverseV2 baseline
Frozen at:   data/research_donchian_exitV2_combined/phase_exitv2_frozen_config.py
```

**Variants tested and rejected:**
- A (ACT=8R alone): PASS but C2 dominates on all metrics
- B (trail=5.0× alone): PASS, good improvement, C2 strictly better
- C (partial profit at +4R): REJECT — cuts fat-tail winners prematurely
- D (60-day backstop): REJECT — cuts long-running chandelier winners
- C3 (C2 + backstop): REJECT — backstop hurts C2 on all metrics

**Optimization cycle summary (17.1→17.4):**

| Stage | avg_r | PF | CAGR | Key finding |
|-------|-------|-----|------|-------------|
| Original 64-sym | +0.179R | 1.27 | +19.0% | Starting point |
| 17.1 UniverseV2 | +1.101R | 3.072 | +14.3% | Universe filter = dominant optimization |
| 17.2 Regime filters | No improvement | — | — | Universe filter made regime filters redundant |
| 17.3 Entry refinement | No improvement | — | — | All variants parameter peaks or inferior |
| 17.4 ExitV2 | +1.511R | 3.830 | +16.0% | Wider chandelier captures fat tails correctly |

**Key lesson:** On a pre-filtered high-quality universe, the exit architecture matters more than entry refinement or regime filtering. The fat-tail edge needs room to run — ACT=6R + trail=5.0× provides that room robustly.

```
METHOD_NAME:    DonchianLong_ExitV2
BASE CONFIG:    Best config from 17.3
UNIVERSE:       Output of 17.1

EXIT VARIANTS TO TEST (one at a time):

  A. Chandelier activation threshold variants
     Current: ACT4 (activates at +4R profit)
     Test:    ACT2 (activates at +2R)
              ACT3 (activates at +3R)
              ACT6 (activates at +6R)
              ACT8 (activates at +8R)
     Rationale: ACT4 was chosen at T3B — test adjacent values for stability

  B. Chandelier ATR multiplier variants
     Current: ATR×3.0 trailing
     Test:    ATR×2.0 (tighter trail — exits sooner)
              ATR×4.0 (wider trail — holds longer)
              ATR×5.0 (very wide — only for extreme runners)
     Rationale: verify ATR×3.0 is stable, not a lucky peak

  C. Partial profit taking
     Take 50% off at +4R fixed profit target
     Trail remaining 50% with Chandelier
     Rationale: locks in some R while keeping exposure on fat tail

  D. Time-based exit backstop
     Exit any trade still open after 60 calendar days regardless of price
     Rationale: prevents capital being tied up in stalled positions

STABILITY CHECK:
  Chandelier variants must form a stable zone — ACT±1 and ATR±0.5 must
  all be profitable. If only one activation threshold works, it's a peak.

GATE CHECK:
  Total R and avg R must beat or match best config from 17.3
  Win rate must stay in 24–45% range
  Fat tail preservation: ZEC/XRP/TRX type trades must still run fully
```

**Claude Code instructions:**
> "Test exit refinement variants A, B, C, D for DonchianLong following
> Section 17.4. Use best config from 17.3. Test each variant independently.
> For Chandelier variants, apply §2.1 stability check across adjacent values.
> Run T2→T4 for each. Output to data/research_donchian_exitV2_{variant}/."

---

### 17.5 RISK SIZING VARIANTS
*(After 17.4 — only after T9B paper running. Theoretical backtest now, live decision later)*

```
METHOD_NAME:    DonchianLong_RiskV2
BASE CONFIG:    Best config from 17.4
UNIVERSE:       Output of 17.1

RISK VARIANTS TO TEST:

  A. Conservative increase
     0.25% → 0.50% per trade (2× current)
     Max heat: 1.50% (max3) / 2.50% (max5)
     Rationale: T16 MC confirmed p05 DD acceptable — this is the
     already-flagged post-T16 review

  B. Moderate increase
     0.25% → 0.75% per trade (3× current)
     Max heat: 2.25% (max3)
     Only test if 0.50% passes T6 kill-switch and MC checks

  C. Volatility-adjusted sizing
     Risk = 0.25% × (median_ATR / current_ATR)
     Reduce size when current ATR > median (high volatility)
     Increase size when current ATR < median (low volatility)
     Rationale: same R target but adapts to market conditions

  D. Kelly criterion (theoretical only)
     Fractional Kelly: f = (edge / odds) × 0.25 (quarter Kelly)
     Compute from T2 win rate and avg win/loss ratio
     Cap at 2% per trade maximum
     NOTE: Kelly sizing is aggressive — treat as theoretical reference only

CRITICAL RULES:
  - Theoretical backtest results are for planning only
  - Do NOT implement risk increases before T9B has run for ≥3 months
  - Implement in steps: 0.25% → 0.50% → 0.75% (never jump)
  - Each live step requires 1 month paper observation before next step
  - T16 MC p05 DD must remain acceptable at each new risk level

GATE CHECK:
  T6 kill-switch must not fire at new risk level
  T16 MC p05 DD must stay within -15% of capital (conservative bound)
  Equity curve must remain smooth — no single-spike profile
```

**Claude Code instructions:**
> "Test risk sizing variants A, B, C, D for DonchianLong following
> Section 17.5. Use best config from 17.4. Run T6 capital simulation
> and T16 Monte Carlo for each variant. Flag if kill-switch fires or
> MC p05 DD exceeds -15%. Output to data/research_donchian_riskV2_{variant}/.
> Mark all results as THEORETICAL — live implementation requires T9B
> paper confirmation per Section 17.5 rules."

---

### 17. OPTIMIZATION SUMMARY — SEQUENCE FOR CLAUDE CODE

```
Step 1: python engines/pipeline_agent.py --method DonchianLong_UniverseV2 (Section 17.1)
        → Review expanded+filtered universe before proceeding

Step 2: python engines/pipeline_agent.py --method DonchianLong_RegimeV2_BTC_Dominance
        python engines/pipeline_agent.py --method DonchianLong_RegimeV2_Volume
        python engines/pipeline_agent.py --method DonchianLong_RegimeV2_ATR_Percentile
        python engines/pipeline_agent.py --method DonchianLong_RegimeV2_MultiTF
        → Review each independently, select best for Step 3

Step 3: python engines/pipeline_agent.py --method DonchianLong_EntryV2_Limit
        python engines/pipeline_agent.py --method DonchianLong_EntryV2_Confirmed
        python engines/pipeline_agent.py --method DonchianLong_EntryV2_Partial
        → Review each, select best for Step 4

Step 4: python engines/pipeline_agent.py --method DonchianLong_ExitV2_ChandelierACT
        python engines/pipeline_agent.py --method DonchianLong_ExitV2_ChandelierATR
        python engines/pipeline_agent.py --method DonchianLong_ExitV2_PartialProfit
        python engines/pipeline_agent.py --method DonchianLong_ExitV2_TimeStop
        → Review each, select best for Step 5

Step 5: python engines/pipeline_agent.py --method DonchianLong_RiskV2_0.50pct
        (only after T9B running ≥3 months)
```

```
{PROJECT_ROOT}/
    data/
        research_{method}_t1/
        research_{method}_t2/
        research_{method}_t3b/
        research_{method}_t4/
        research_{method}_t5/
        research_{method}_t6/
        research_{method}_t7/
        research_{method}_t8/
    phase_t1_{method}_concept_discovery.py
    phase_t2_{method}_core_engine.py
    phase_t3b_{method}_wide_exit_engineering.py
    phase_t4_{method}_robustness_engine.py
    phase_t5_{method}_portfolio_filter.py
    phase_t6_{method}_capital_execution_engine.py
    phase_t7_{method}_asset_robustness.py
    phase_t8_{method}_config_freeze.py
```

---

## 15. QUICK REFERENCE — GATE SUMMARY

| Phase | Must pass to proceed |
|-------|----------------------|
| T1 | ≥1 combo: stability PASS + §4.2 cost floor |
| T2 | §4.1 win rate 30–45% + avg_r > 0.15R |
| T3B | Total R > T2 baseline + §4.1 + §4.2 still hold |
| T4 | MC p05 positive + remove-top-1 profitable + cost stress 0.05R PF>1.0 |
| T5 | ≥80 trades in canonical variant |
| T6 | Kill switch never fires + DD within risk profile |
| T7 | Profitable after removing top-1 asset |
| T8 | All above passed → config frozen |
| T15 | Win rate stable across N variants (critical if T3B win rate was at 30% floor) |
| T16 | MC p05 total R positive → then reconsider R sizing |
| T17 | Deprecated — EMA200 regime filter validated at T1 level |
| T18 | Optional — if FAIL, document structural reason and proceed without gate |

---

## 16. CLAUDE-POWERED PIPELINE AGENT

### What it does

`pipeline_agent.py` runs the full T1→T8 pipeline autonomously for any method.
For each phase it:
1. Checks if the phase script exists — if not, asks Claude to write it
2. Runs the script
3. Reads all output files
4. Calls Claude to analyze results and check all gate conditions
5. Proceeds if gates pass, halts with a detailed report if they fail
6. Pauses for human review at key phases (T1, T4, T8, T16)

### Usage

```bash
# Set API key first
export ANTHROPIC_API_KEY=your_key_here

# Run Dual MA from T1 (long-side, Spot)
python engines/pipeline_agent.py --method DualMA --start-phase T1

# Run Short-side Donchian from T1 (Futures, higher cost floor)
python engines/pipeline_agent.py --method DonchianShort --start-phase T1 --futures

# Resume a pipeline from a specific phase
python engines/pipeline_agent.py --method DualMA --start-phase T4

# Skip human review checkpoints (fully automated — use with caution)
python engines/pipeline_agent.py --method DualMA --start-phase T1 --auto

# Use a custom config JSON file
python engines/pipeline_agent.py --method DualMA --config my_config.json
```

### Human review checkpoints

The agent pauses and asks for confirmation at:
- **T1** — before committing to a canonical config
- **T4** — after robustness testing (concentration, MC, cost stress)
- **T8** — before freezing the final config
- **T16** — before deciding on R sizing

At each checkpoint the agent prints the full gate results, key findings,
warnings, and identified canonical config. You type `y` to proceed or `n` to halt.

### What Claude does at each phase

| Phase | Claude's job |
|-------|-------------|
| T1 | Writes script if missing. After run: identifies canonical config, checks §2.1 stability and §4.2 cost floor across all combos |
| T2 | Writes script if missing. After run: checks §4.1 win rate, §4.2 avg_r, flags §4.7 concentration |
| T3B | Writes script if missing. After run: compares against T2 baseline, checks fat-tail improvement |
| T4 | Writes script if missing. After run: reviews MC p05, cost stress, remove-best-asset and remove-best-month results |
| T5 | Writes script if missing. After run: selects canonical portfolio cap (max3 vs max5) |
| T6 | Writes script if missing. After run: checks kill-switch, equity curve shape, USDT return |
| T7 | Writes script if missing. After run: confirms system survives top-1 asset removal |
| T8 | Writes script if missing. After run: generates frozen config, final scorecard |
| T15–T18 | Same pattern — writes, runs, analyzes, gates |

### Agent log

Every run produces `pipeline_agent_{method}_log.json` with:
- Phase-by-phase gate results
- Claude's analysis for each phase
- Script stdout tails
- Final pass/fail status

### Planned improvements (v2+)

- Parallel phase runs where dependencies allow
- Automatic parameter suggestions when T1 partially fails
- Slack/email notification on gate failures
- Web dashboard for equity curve visualization
- Multi-method comparison report (Donchian vs DualMA vs Keltner)
- Automatic T9B signal generator activation after T8 freeze

---

## 19. RESEARCH BACKLOG — PENDING AFTER CURRENT MR INVESTIGATIONS

### 19A. IntradayVolatilityBreakout (Trend Following, 4H)

**What it is:**
Price breaks above the highest high of the last N 4H bars, but ONLY when the
market has been in a low-volatility compression phase (ATR percentile < 30th
of trailing window). This is distinct from Donchian (1D channel, no volatility
filter) — this system specifically exploits the volatility expansion that
follows a compression squeeze.

Theoretical basis: Markets alternate between compression (consolidation) and
expansion (trending). Entering at the start of an expansion after confirmed
compression should improve signal quality vs. entering any channel breakout.

The ATR percentile < 30th filter means: only fire when current ATR(14) is in
the bottom 30% of its trailing 252-bar distribution (approximately 1 year on 4H).

```
METHOD_NAME:        IntradayVolatilityBreakout
ENTRY_LOGIC:        close > highest_high(N, 4H bars) AND ATR(14) < ATR_percentile_30th
                    Compression filter: ATR14 / ATR14.rolling(252).quantile(0.30)
                    Entry fires only in low-volatility compression zone
TIMEFRAME:          4H (native intraday logic — not expanding from 1D)
PARAM_GRID:
    n_bars:         [10, 15, 20, 30, 40]  (4H channel breakout window)
    atr_pct:        [0.20, 0.30, 0.40]    (compression threshold: bottom X%)
    atr_lookback:   [126, 252, 504]       (rolling window for percentile: ~3mo/6mo/1yr)
STABILITY_ZONE:     n_bars +/- 1 step AND atr_pct +/- 0.10 step
                    PASS if >= 67% of zone profitable
FILTER_MODES:       ["ema200_price", "none"]
ATR_MULTS:          [2.0, 3.0]
EXIT_LOGIC:         Same Chandelier architecture as Donchian
                    ACT=4R, trail=3xATR OR Donchian N//2 lower channel
ASSET_UNIVERSE:     60-symbol crypto universe
EXCHANGE:           Binance Futures (USD-M) — 4H intraday on Futures
LEVERAGE:           1.0 to start
RISK_PER_TRADE:     0.0025 (0.25%)
COST_FLOOR_R:       0.15R (Futures, higher than 1D Spot threshold)
BACKTEST_BARS:      8760 (4H x ~4yr equivalent)
SIDE:               LONG only initially

Key distinction vs Donchian:
  Donchian 1D: breakout of any N-day high regardless of volatility context
  IntradayVol: breakout of N-4H-bar high ONLY when coming out of compression
  Expected: fewer signals, better signal quality, higher avg_r per trade
  Risk: low-ATR periods may precede continued consolidation, not expansion
  T18 lesson: Donchian T18 FAILED ATR gate because Donchian's best trades are
  in HIGH-ATR expansionary markets. IntradayVol uses ATR gate differently:
  entry in LOW-ATR zone but expecting subsequent expansion — test separately.

STATUS:             PENDING — after current MR investigations complete
PRIORITY:           LOW — run only after RSI MR 1D T9B confirms (Sep 2026)
```

---

---

## 18. DEPLOYMENT ROADMAP

### Phase 1 — Long-only live (Binance Spot)

**Prerequisites before any live capital:**
1. T9B paper trading ≥ 3 months on DonchianLong_UniverseV2_ExitV2
2. 17.5 risk sizing theoretical complete
3. Signal generator built and tested
4. Mean Reversion T1→T8 complete (run in parallel)

**Systems for Phase 1:**
```
DonchianLong_UniverseV2_ExitV2   (24 symbols, max8, 0.25% risk)
DualMA BTC Long                   (BTC only, max1, separate allocation)
Mean Reversion Long               (after T1→T8 complete)
```

**Start at 10–20% of intended capital for first 3 months live.**

### Phase 2 — Add short side (Binance Futures)

**Prerequisites:**
1. Phase 1 running successfully for ≥ 3 months
2. Historical data extended to 2018–2020 (bear cycle data collected)
3. Short-side T1→T8 validated on extended dataset
4. Futures account set up and paper-tested
5. Leverage decision: start at 1.0×, increase per Section 13C decision tree

**Systems for Phase 2:**
```
DonchianShort_UniverseV2          (requires new T1→T8 with bear data)
Mean Reversion Short              (requires new T1→T8 with bear data)
```

### Immediate actions (start now in parallel):
```
1. Start T9B paper — DonchianLong_UniverseV2_ExitV2 (start the clock)
2. Collect extended historical data 2017–2020 for all 24 symbols
3. Run 17.5 risk sizing theoretical
4. Start Mean Reversion T1
```

---

*Pipeline version 2.0 — DonchianLong_UniverseV2_ExitV2 fully validated (17.4 complete)*
*Full optimization: avg_r +0.179R → +1.511R (8.4×), PF 1.27 → 3.830 (3×)*
*MC p05 +200.5R at bs=50, 100% prob positive — strongest validation in pipeline*
*Next: 17.5 risk sizing theoretical + Mean Reversion T1 + T9B paper trading*
*Short-side deployment requires extended 2017–2020 historical data — collect now*

---

## SYSTEM 7 + 8 T9B CAVEAT NOTE (added 2026-06-13)

```
System 7 (VolContractionShort) and System 8 (MACrossShort) both entered T9B
paper trading in June 2026. A key limitation to note for the September review:

CAVEAT_6 (S7 and S8):
  Cross-system dedup (signal_arbitrator.py duplicate_cross_system rule) is
  blocking S7 and S8 entries in early T9B because the Momentum Factor engine
  holds a 22-symbol footprint (12 longs + 10 shorts) on Binance Futures.

  When S7 or S8 generates a short signal on a symbol Momentum already holds
  (as a short or long), the arbitrator blocks the duplicate. In the first weeks
  of T9B this suppressed all S7 signals and is expected to suppress S8 signals.

  September 2026 review must account for this:
    - S7 sample size will be artificially small (not representative)
    - S8 sample size similarly reduced
    - Do NOT compare S7/S8 T9B trade counts directly to backtest signal frequency
    - Consider: separate Futures heat pool for S7+S8 vs Momentum in live deployment
      so short specialists can trade independently of Momentum's basket
```
