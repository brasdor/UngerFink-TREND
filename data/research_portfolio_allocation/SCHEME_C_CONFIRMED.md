# Scheme C (Regime-Tilted) -- CONFIRMED PRODUCTION ALLOCATION

**Confirmed:** 2026-06-17
**Total Capital:** $60,000 ($30k Spot + $30k Futures)

## Allocation

| Pool    | System | Allocation | Weight |
|---------|--------|-----------|--------|
| Spot    | S1 Donchian Long       | $8,000  | 27% |
| Spot    | S2 RSI MR Long         | $12,000 | 40% |
| Spot    | S3 ConsecDownDays MR   | $10,000 | 33% |
| Futures | S6 Momentum Factor     | $8,000  | 27% |
| Futures | S7 VolContraction Short | $11,000 | 37% |
| Futures | S8 MA Cross Short      | $11,000 | 37% |

## Basis

Comparison of 5 allocation schemes (see allocation_comparison.txt).
Scheme C tied best Sharpe (1.74) with D, but has more practical round-number allocations.
Scheme D allocates only $2,705 to S1 (too small for practical 0.25% risk sizing).

## Key Logic

**Spot ($30k):**
- S2 + S3 = $22k (MR floor, covers consolidation/ranging years)
- S1 = $8k (trend, active in bull years: 2021 +84.7%, 2024 +28.7%)

**Futures ($30k):**
- S7 + S8 = $22k (short specialists, carry 2022: S7 +84.2%, S8 +114.4%)
- S6 = $8k (momentum factor, amplifies bull years: 2021 +183.9%, 2024 +74.0%)

## Year-by-Year Portfolio Returns (Scheme C backtest)

```
2021: +38.9%   (S1+S6 bull amplifiers dominate)
2022: +28.8%   (S7+S8 short specialists dominate)
2023: +22.9%   (S7+S8 continued + S2 MR floor)
2024: +57.3%   (all systems positive)
2025: +34.8%   (broad positive, S8 +61.6%)
2026:  -0.5%   (partial year, essentially flat)
```

## Caveats

1. S8 returns are theoretical maximums (uncapped T6 equity curve).
2. Regime detection not yet automated -- manual review monthly.
3. Monitor allocation drift via t9b_combined_summary.py.
4. S6 documented CAGR +47.77% from Phase 3 canonical backtest.

## Implementation

- Engine equity constants updated to Scheme C allocations (2026-06-18)
- Existing open positions keep original sizing
- New positions from today onward use Scheme C allocation
- t9b_combined_summary.py updated with target allocation and drift tracking
