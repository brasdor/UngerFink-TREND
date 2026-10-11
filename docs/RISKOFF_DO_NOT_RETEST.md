# Risk-off research: do not retest

These specifications were tested once, on BTC data from 2019-09-08 to 2026-09-28 (H7: 2020-09-01 to 2026-08-28), under `docs/RISKOFF_PREREG.md` (commit `12e10e2e`). Re-running them with a different threshold, lag, window, transformation, subperiod or label on the same history is a forking path: the data has already been used to pick winners, and every extra variant adds to the multiplicity that the Benjamini-Hochberg numbers did not account for.

## Closed (verdict, source)

| spec | verdict | where |
|---|---|---|
| H4-VIX: ln VIX vs 60-obs mean, lag 1 | DEAD | `RISKOFF_STAGE1.md` |
| H4-NDQ: drawdown from 20-obs Nasdaq high, lag 1 | DEAD | same |
| H4-JPY: 10-obs change of ECB USD/JPY, lag 1 | DEAD | same |
| H4-BAA: Baa-10y spread vs 60-obs mean, lag 2 | DEAD | same |
| H7-BAS: 7-day mean perp premium index, lag 1 | DEAD | same |
| H7-FUND: 3-day mean BTC funding, lag 1 | WEAK (unstable: 0.69 pre-2022, 0.52 after) | same |
| H7-OI: ln OI vs 60-day mean, lag 1 | WEAK; only forward retest allowed, see `RISKOFF_OI_FORWARD_PREREG.md` | same |
| H8-BRD: 10-day change of classifier breadth | DEAD | same |
| H8-ETHBTC: 10-day log change of ETH/BTC | DEAD | same |
| H1-HP, H1-HR, H2-NL, H2-RY, H2-USD, H3-STB, H3-USDT (event study) | DEAD | same |
| UP label (>= +10% in 10 days) for all of the above | DEAD under the registered orientation | same |
| Label sensitivity (7% and 15%; 5 and 20 days) | Stage 0 base rates only; never used for Stage 1 | `RISKOFF_STAGE0.md` |

## Observations that are NOT leads

Do not build hypotheses on these; they were seen after the fact in tables of many numbers.

- VIX and Nasdaq AUC were 0.57 and 0.59 from 2022 on but 0.49 and 0.51 before.
- Funding and basis AUC were 0.69 and 0.71 before 2022 and about 0.5 after.
- VIX predicts UP moves as well as DOWN moves (a two-sided volatility indicator) and had an incremental UP dAUC of +0.065 (BH q 0.22).
- Stablecoin UP raw p of 0.05-0.07 in the post-hoc sensitivity, from a non-point-in-time series.
- The September 2026 dip (about -7%) is not a 10%/10-day event.

## What is allowed

1. **Forward data only,** under a pre-registration committed before the data exist (the open-interest test is the only one registered).
2. **A genuinely new hypothesis** (different variable or mechanism, or a different label) with its own pre-registration committed first. Count the 9 DOWN and 9 UP Stage 1 tests and the 14 event-study tests already run when interpreting any new p-value.
3. **A bug fix** with a written diff of what changed and why, run once.
4. **A new, independent sample for the same spec** (another asset, another venue), pre-registered, if the spec is unchanged.

## What is not allowed

Tuning lags, windows, thresholds or transformations; dropping or adding episodes; changing the merge gap or the label to rescue a result; combining variables into a flag using this history; reporting a variant that passes without listing every variant tried.
