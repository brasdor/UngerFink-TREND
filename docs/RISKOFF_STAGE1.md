# Risk-off flag - Stage 1 report (univariate tests, pre-registered)

Pre-registration: `docs/RISKOFF_PREREG.md`, commit `12e10e2e` (2026-10-11 10:44 +02:00), committed before any feature or statistic was computed.
Code: `tools/riskoff/stage1_fetch.py`, `stage1_run.py`, `stage1_calibration.py`, `stage1_report.py`. Results in `data/riskoff/stage1_*.csv`. Not touched: engines, workflows, allocator, Telegram. No variables were combined and no risk model was built; the logistic regressions in the incremental test are evaluation scaffolding only.

## 1. Verdicts

Computed mechanically from the rules in the pre-registration (section 8). Target: BTC down 10% within 10 days. A hypothesis takes the verdict of its best card.

| hypothesis | verdict | cards |
|---|---|---|
| H4 cross-asset (VIX, Nasdaq, USD/JPY, Baa spread) | DEAD | H4-VIX DEAD, H4-NDQ DEAD, H4-JPY DEAD, H4-BAA DEAD |
| H7 derivatives crowding (OI, basis, funding) | WEAK | H7-OI WEAK, H7-FUND WEAK, H7-BAS DEAD |
| H8 crypto internals (breadth change, ETH/BTC) | DEAD | H8-BRD DEAD, H8-ETHBTC DEAD |
| H1 miner margin (event study) | DEAD | H1-HP DEAD, H1-HR DEAD |
| H2 liquidity (event study) | DEAD | H2-NL DEAD, H2-RY DEAD, H2-USD DEAD |
| H3 stablecoins (event study, not point-in-time) | DEAD | H3-STB DEAD, H3-USDT DEAD |

**Nothing survives.** The strongest candidates are two H7 cards (open interest, funding), both rated WEAK. In plain terms:

- **H4 (VIX, Nasdaq, USD/JPY, Baa spread): dead.** AUC 0.49-0.53, all p >= 0.23, none adds to the regime classifier's inputs. USD/JPY points the wrong way in 19 of 21 leave-one-episode-out runs.
- **H8 (breadth change, ETH/BTC): dead.** AUC 0.54 and 0.50. Breadth change has a small, non-significant incremental AUC gain (+0.012, p 0.10, BH q 0.43).
- **H7: weak.** Open interest (AUC 0.603) and funding (0.598) have raw one-sided p of 0.03 and 0.02, but after Benjamini-Hochberg over the 9 tests q = 0.13 for both, so neither passes the pre-registered 0.10. Basis is dead (p 0.14). Details and why this is not a result in section 4.
- **H1, H2, H3: dead at event-study level**, with a power caveat in section 7. H3 would have been capped at WEAK anyway because the stablecoin series is not point-in-time.
- **UP label (for the short systems): dead for all 9 cards** under the pre-registered orientation (section 5).

## 2. What was tested

| item | value |
|---|---|
| Label (primary) | BTC close falls >= 10% within 10 days; 367 flagged days of 2,574 labeled (14.3%) |
| Label (secondary) | BTC close rises >= 10% within 10 days; 530 flagged days (20.6%) |
| Windows | H4, H8: 2019-09-08 to 2026-09-28 (H8-BRD from 2020-02-14, ETH/BTC from 2019-12-07). H7: 2020-09-01 to 2026-08-28 (H7-OI effectively from 2020-10-31 after its 60-day warm-up). |
| 30-day-merge episodes in sample | H4: 21. H8: 20-21. H7: 18-19. |
| Bootstrap | 2,000 cluster-bootstrap draws, clusters = 30-day-merge episodes plus the quiet period before the next one; seed 20261011 |
| Baseline for incremental test | B1 = ln(BTC / EMA200), B2 = classifier's 3-day-EMA breadth, B3 = ATR14 252-day percentile |
| Lags | applied per card (VIX, Nasdaq, USD/JPY, OI, basis, funding: 1 day; Baa: 2; breadth and ETH/BTC: 0 because they are our own closed bars) |

## 3. Univariate results, DOWN label

| var | days | 30d episodes | AUC | CI lo | CI hi | p (1-sided) | BH q | Brier skill | BS lo | BS hi |
|---|---|---|---|---|---|---|---|---|---|---|
| H4-VIX | 2574 | 21 | 0.530 | 0.419 | 0.631 | 0.299 | 0.392 | -0.015 | -0.047 | 0.005 |
| H4-NDQ | 2574 | 21 | 0.534 | 0.433 | 0.603 | 0.235 | 0.392 | -0.011 | -0.020 | -0.002 |
| H4-JPY | 2574 | 21 | 0.493 | 0.428 | 0.580 | 0.525 | 0.525 | -0.019 | -0.039 | -0.005 |
| H4-BAA | 2574 | 21 | 0.531 | 0.420 | 0.647 | 0.305 | 0.392 | -0.010 | -0.046 | 0.010 |
| H7-OI | 2124 | 18 | 0.603 | 0.496 | 0.685 | 0.029 | 0.133 | 0.005 | -0.033 | 0.065 |
| H7-BAS | 2184 | 19 | 0.579 | 0.414 | 0.663 | 0.143 | 0.327 | -0.032 | -0.059 | -0.006 |
| H7-FUND | 2184 | 19 | 0.598 | 0.505 | 0.671 | 0.019 | 0.133 | -0.014 | -0.040 | 0.025 |
| H8-BRD | 2415 | 20 | 0.536 | 0.474 | 0.610 | 0.145 | 0.327 | -0.011 | -0.022 | -0.001 |
| H8-ETHBTC | 2480 | 21 | 0.500 | 0.420 | 0.569 | 0.515 | 0.525 | -0.011 | -0.031 | 0.011 |

The "too small" flag (fewer than 6 episodes in a subperiod) applies to all three H7 cards; it blocks SURVIVES. AUC is oriented so that > 0.5 means the variable moves in the pre-registered direction before drops. "Brier skill" is `1 - Brier(model)/Brier(base rate)` from the walk-forward quintile calibration; positive = beats always predicting the base rate. **Only H7-OI has a positive point estimate (+0.005) and its interval spans zero.** Even where the AUC is above 0.5, none of these variables improves probability forecasts over the base rate. The walk-forward AUC (after the 365-day burn-in) is 0.625 for OI, 0.558 for funding and below 0.55 for everything else.

### Stability: subperiods and leave-one-episode-out

| var | AUC pre-2022 | eps pre | AUC 2022-on | eps post | LOEO min | LOEO max | LOEO <0.5 | of |
|---|---|---|---|---|---|---|---|---|
| H4-VIX | 0.489 | 7 | 0.572 | 14 | 0.512 | 0.558 | 0 | 21 |
| H4-NDQ | 0.507 | 7 | 0.586 | 14 | 0.511 | 0.548 | 0 | 21 |
| H4-JPY | 0.462 | 7 | 0.483 | 14 | 0.482 | 0.522 | 19 | 21 |
| H4-BAA | 0.563 | 7 | 0.547 | 14 | 0.517 | 0.565 | 0 | 21 |
| H7-OI | 0.601 | 4 | 0.592 | 14 | 0.571 | 0.616 | 0 | 18 |
| H7-BAS | 0.710 | 5 | 0.473 | 14 | 0.513 | 0.598 | 0 | 19 |
| H7-FUND | 0.687 | 5 | 0.519 | 14 | 0.550 | 0.613 | 0 | 19 |
| H8-BRD | 0.553 | 6 | 0.516 | 14 | 0.520 | 0.552 | 0 | 20 |
| H8-ETHBTC | 0.479 | 7 | 0.522 | 14 | 0.481 | 0.521 | 10 | 21 |

- **H7-OI is the most stable variable**: 0.601 before 2022, 0.592 after, and never below 0.571 when any single episode is removed.
- **H7-FUND and H7-BAS are not stable.** Funding: 0.687 before 2022, 0.519 after. Basis: 0.710 and 0.473. Removing the Jan-Jul 2021 episode alone drops basis from 0.579 to 0.513 and funding from 0.598 to 0.550.
- **Pre-2022 holds only 4-5 episodes for H7** (below the pre-registered minimum of 6), so the subperiod comparison for H7 cannot support a stability claim either way. That is the "too small" flag.

### Incremental value over the regime classifier's inputs (walk-forward, sign-free)

| var | OOS days | AUC B1-B3 | AUC +var | dAUC | lo | hi | p | BH q | dBrier skill |
|---|---|---|---|---|---|---|---|---|---|
| H4-VIX | 1914 | 0.535 | 0.532 | -0.003 | -0.066 | 0.038 | 0.560 | 0.727 | -0.000 |
| H4-NDQ | 1914 | 0.535 | 0.535 | 0.000 | -0.032 | 0.022 | 0.561 | 0.727 | 0.001 |
| H4-JPY | 1914 | 0.535 | 0.529 | -0.006 | -0.056 | 0.030 | 0.646 | 0.727 | -0.007 |
| H4-BAA | 1914 | 0.535 | 0.559 | 0.024 | -0.070 | 0.090 | 0.311 | 0.700 | -0.014 |
| H7-OI | 1734 | 0.553 | 0.571 | 0.018 | -0.017 | 0.056 | 0.142 | 0.426 | 0.010 |
| H7-BAS | 1794 | 0.543 | 0.541 | -0.002 | -0.069 | 0.030 | 0.581 | 0.727 | 0.000 |
| H7-FUND | 1794 | 0.543 | 0.588 | 0.044 | -0.019 | 0.072 | 0.096 | 0.426 | -0.001 |
| H8-BRD | 1914 | 0.535 | 0.547 | 0.012 | -0.006 | 0.048 | 0.098 | 0.426 | 0.008 |
| H8-ETHBTC | 1910 | 0.535 | 0.532 | -0.003 | -0.010 | 0.005 | 0.827 | 0.827 | -0.001 |

Only funding (+0.044, p 0.096) and breadth change (+0.012, p 0.098) come near raw significance and neither survives BH (q 0.43). Note this test fits the coefficient freely, so it is sign-blind by design (pre-registered).

### Lead time before the 41 primary episodes (alarm = score in its own top quintile so far)

| var | episodes | hits | hit rate | chance rate | lift | median lead (days, hits only) |
|---|---|---|---|---|---|---|
| H4-VIX | 34 | 14 | 0.412 | 0.421 | 0.980 | 8.500 |
| H4-NDQ | 34 | 10 | 0.294 | 0.355 | 0.830 | 18.500 |
| H4-JPY | 34 | 13 | 0.382 | 0.436 | 0.880 | 12.000 |
| H4-BAA | 34 | 13 | 0.382 | 0.304 | 1.260 | 18.000 |
| H7-OI | 26 | 8 | 0.308 | 0.280 | 1.100 | 7.000 |
| H7-BAS | 28 | 5 | 0.179 | 0.119 | 1.500 | 20.000 |
| H7-FUND | 28 | 5 | 0.179 | 0.097 | 1.840 | 12.000 |
| H8-BRD | 32 | 7 | 0.219 | 0.317 | 0.690 | 13.000 |
| H8-ETHBTC | 33 | 14 | 0.424 | 0.490 | 0.870 | 8.000 |

"Chance rate" = share of all days that had an alarm in the preceding 11 days. No variable gives a useful lead: hit rates are 18-42% against chance rates of 10-49%. The best lift is funding (5 hits of 28 episodes, 1.8x chance, median lead 12 days), which rests on five hits: one in Nov 2021 and four in 2024, the 2024 ones being small drops of 10-16%. Open interest hit 8 of 26 episodes (6 of them in 2021-12 to 2023-04, lead 0-11 days). The funding and basis alarms did not fire before the large 2020-21 and 2022 drops in a way that beats chance. Lead times are not interpretable where the lift is near 1.

## 4. Why H7 is "weak" and not more

- Open interest: raw p 0.03, but BH q 0.13 (fails the 0.10 gate), Brier skill indistinguishable from 0, no significant incremental value (dAUC +0.018, p 0.14), 4 pre-2022 episodes. It passes subperiod and leave-one-out checks, which is the reason it is the first thing to look at again.
- Funding: raw p 0.02, but q 0.13, negative Brier skill, signal concentrated before 2022.
- Basis: p 0.14, and dependent on a single 2021 episode.
- These are 9 tests on about 20 episodes. A top p of 0.02-0.03 among 9 is not rare under no effect (probability that the best of 9 independent tests has p <= 0.02 is about 17%).

## 5. UP label (secondary, for the short systems)

| var | 30d episodes | AUC | lo | hi | p | BH q | Brier skill | dAUC incr. | p incr. | BH q incr. | verdict (UP) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H4-VIX | 22 | 0.440 | 0.378 | 0.498 | 0.978 | 0.978 | -0.005 | 0.065 | 0.024 | 0.216 | DEAD |
| H4-NDQ | 22 | 0.494 | 0.403 | 0.548 | 0.606 | 0.978 | -0.017 | -0.016 | 0.912 | 0.912 | DEAD |
| H4-JPY | 22 | 0.454 | 0.404 | 0.504 | 0.964 | 0.978 | -0.013 | -0.001 | 0.510 | 0.755 | DEAD |
| H4-BAA | 22 | 0.529 | 0.368 | 0.618 | 0.368 | 0.978 | -0.019 | -0.003 | 0.587 | 0.755 | DEAD |
| H7-OI | 19 | 0.530 | 0.427 | 0.597 | 0.260 | 0.978 | -0.015 | 0.001 | 0.379 | 0.755 | DEAD |
| H7-BAS | 19 | 0.410 | 0.360 | 0.568 | 0.860 | 0.978 | -0.001 | 0.007 | 0.235 | 0.705 | DEAD |
| H7-FUND | 19 | 0.415 | 0.371 | 0.556 | 0.884 | 0.978 | 0.002 | 0.000 | 0.494 | 0.755 | DEAD |
| H8-BRD | 21 | 0.510 | 0.424 | 0.574 | 0.413 | 0.978 | -0.015 | -0.005 | 0.677 | 0.761 | DEAD |
| H8-ETHBTC | 22 | 0.469 | 0.428 | 0.534 | 0.866 | 0.978 | -0.005 | 0.009 | 0.066 | 0.297 | DEAD |

All nine are DEAD under the pre-registered orientation (expected sign opposite to the DOWN sign). One observation, not a hypothesis test: H4-VIX has oriented AUC 0.44 for UP, i.e. a high VIX is followed by *more* 10% up-moves, and its incremental dAUC is +0.065 (raw p 0.024, BH q 0.22, not significant after correction). That is consistent with VIX acting as a two-sided volatility indicator, but it is exploratory. The pre-registration text gave a rationale for the UP signs ("stress extremes precede rebounds") that contradicts its own rule ("opposite of DOWN") for stress variables; I implemented the rule as written and the contradiction is listed in section 8.

## 6. Event study (H1, H2, H3), DOWN label

Statistic: mean oriented z-score over days -10..0 before each episode start, against a circular-shift null (5,000 draws). Window starts 2020-09-06 because the expanding z-score needs 365 days of history inside our label calendar.

| var | window start | episodes | T observed | null mean | null sd | p (1-sided) | BH q | verdict |
|---|---|---|---|---|---|---|---|---|
| H1-HP | 2020-09-06 | 18 | -0.1603 | -0.136 | 0.1731 | 0.5097 | 0.9394 | DEAD |
| H1-HR | 2020-09-06 | 18 | -0.027 | 0.0739 | 0.148 | 0.767 | 0.9394 | DEAD |
| H2-NL | 2020-09-06 | 18 | 0.256 | 0.3504 | 0.0644 | 0.9394 | 0.9394 | DEAD |
| H2-RY | 2020-09-06 | 18 | -0.0837 | 0.1243 | 0.168 | 0.9026 | 0.9394 | DEAD |
| H2-USD | 2020-09-06 | 18 | 0.0831 | 0.0139 | 0.1633 | 0.3485 | 0.9394 | DEAD |
| H3-STB | 2020-09-06 | 18 | 0.5637 | 0.5259 | 0.0936 | 0.4157 | 0.9394 | DEAD |
| H3-USDT | 2020-09-06 | 18 | 0.4639 | 0.5295 | 0.0725 | 0.7692 | 0.9394 | DEAD |

UP label (secondary): | var | episodes | T obs | null mean | p | BH q | verdict |
|---|---|---|---|---|---|---|
| H1-HP | 19 | -0.191 | 0.1376 | 0.9388 | 0.9388 | DEAD |
| H1-HR | 19 | -0.1464 | -0.0737 | 0.7041 | 0.9388 | DEAD |
| H2-NL | 19 | -0.3825 | -0.3498 | 0.7111 | 0.9388 | DEAD |
| H2-RY | 19 | -0.3641 | -0.1231 | 0.9128 | 0.9388 | DEAD |
| H2-USD | 19 | -0.1414 | -0.0058 | 0.8342 | 0.9388 | DEAD |
| H3-STB | 19 | -0.6193 | -0.5241 | 0.8336 | 0.9388 | DEAD |
| H3-USDT | 19 | -0.6118 | -0.532 | 0.8724 | 0.9388 | DEAD |

Sensitivity (post-hoc, A1): z-scores computed from the full vendor history (2017-12 onward), which adds the 2019-2020 episodes:

| var | window start | episodes | T obs | null mean | p | BH q | verdict |
|---|---|---|---|---|---|---|---|
| H1-HP | 2018-11-30 | 21 | -0.2596 | -0.2341 | 0.5405 | 0.9544 | DEAD |
| H1-HR | 2018-11-30 | 21 | 0.1396 | 0.2626 | 0.8388 | 0.9544 | DEAD |
| H2-NL | 2018-11-30 | 21 | -0.0195 | -0.1848 | 0.2102 | 0.9544 | DEAD |
| H2-RY | 2018-11-30 | 21 | -0.04 | 0.0314 | 0.6323 | 0.9544 | DEAD |
| H2-USD | 2018-11-30 | 21 | 0.0478 | -0.0556 | 0.3165 | 0.9544 | DEAD |
| H3-STB | 2018-12-29 | 21 | 0.1316 | 0.1416 | 0.9544 | 0.9544 | DEAD |
| H3-USDT | 2018-12-29 | 21 | 0.1314 | 0.1407 | 0.9538 | 0.9544 | DEAD |

| var | episodes | p | BH q | verdict (UP, sensitivity) |
|---|---|---|---|---|
| H1-HP | 22 | 0.9602 | 0.9602 | DEAD |
| H1-HR | 22 | 0.4861 | 0.9602 | DEAD |
| H2-NL | 22 | 0.8762 | 0.9602 | DEAD |
| H2-RY | 22 | 0.8418 | 0.9602 | DEAD |
| H2-USD | 22 | 0.8824 | 0.9602 | DEAD |
| H3-STB | 22 | 0.0522 | 0.2513 | WEAK |
| H3-USDT | 22 | 0.0718 | 0.2513 | WEAK |

Nothing is significant for the DOWN label in either version (best raw p 0.35 in the pre-registered run, 0.21 in the sensitivity). In the post-hoc version the two stablecoin cards reach raw p 0.05-0.07 for the UP label (BH q 0.25); since H3 is not point-in-time and these were not the pre-registered run, they carry no weight.

## 7. Where the sample is too small

- **Episodes.** About 20 independent 30-day episodes in every sample (and in the Stage 0 15-day list only 13 of 41 episodes fell 20% or more). A single test of the AUC has a null sd of 0.04-0.07 for these persistent features (section 8, A2), so only AUCs of roughly 0.57-0.61 or more can be told apart from persistent noise, and after correcting for 9 tests, higher. **"Dead" means "no AUC above about 0.58 is visible"; it does not exclude real effects of 0.52-0.57.**
- **Subperiods.** Pre-2022 holds 4-7 episodes. No subperiod stability claim is supportable for H7 (4-5 episodes) and only weakly for H4/H8 (7 and 6-7).
- **Event studies.** With 18-21 episodes the null sd of the mean z is 0.06-0.19 for most cards, so only mean standardized moves of roughly 0.2-0.3 or more are detectable. Slow variables (H1, H2, H3) change by regime, so their real sample is closer to the number of regime changes than to 18.
- **H3** cannot be judged on a point-in-time basis at all until a forward snapshot log exists.
- **H7 funding.** The history ends 2026-08-28, so the September-October 2026 period is not tested.

## 8. Amendments and caveats (post-hoc unless stated)

- **A0 (before testing).** Three zero open-interest snapshots (2022-03-07, 2024-07-10, 2024-07-12) in the binance.vision files were treated as missing and forward-filled for at most 3 days. Found during feature QA, before any statistic.
- **A1 (post-hoc sensitivity).** Event-study z-scores from the full vendor history; reported alongside the pre-registered run, which is the one the verdicts use.
- **A2 (post-hoc check of the machinery).** For each Stage 1 variable I simulated 2,000 Gaussian AR(1) noise series with the same lag-1 autocorrelation, on the same days and labels, and computed the empirical p of the observed AUC. It broadly agrees with the cluster-bootstrap p-values, so the bootstrap is not badly miscalibrated, except for OI (0.007 vs 0.030). Under that null the OI p-value would pass a BH gate over 9 tests (0.007 x 9 = 0.063), but this is not the pre-registered test and the Gaussian AR(1) null ignores fat tails and regime shifts, so it does not change the verdict.

| var | n days | lag-1 autocorr | AUC obs | AUC null sd | AUC null 95th pct | p vs AR(1) noise |
|---|---|---|---|---|---|---|
| H4-VIX | 2574 | 0.953 | 0.5302 | 0.0428 | 0.5706 | 0.2244 |
| H4-NDQ | 2574 | 0.954 | 0.5343 | 0.0427 | 0.5709 | 0.2024 |
| H4-JPY | 2574 | 0.92 | 0.4931 | 0.0387 | 0.5641 | 0.5672 |
| H4-BAA | 2574 | 0.991 | 0.5309 | 0.058 | 0.5967 | 0.3078 |
| H7-OI | 2124 | 0.919 | 0.6028 | 0.0434 | 0.57 | 0.007 |
| H7-BAS | 2184 | 0.992 | 0.5787 | 0.071 | 0.6104 | 0.1369 |
| H7-FUND | 2184 | 0.969 | 0.5976 | 0.0506 | 0.5852 | 0.03 |
| H8-BRD | 2415 | 0.95 | 0.5362 | 0.0442 | 0.5722 | 0.2044 |
| H8-ETHBTC | 2480 | 0.907 | 0.5 | 0.0392 | 0.5632 | 0.4893 |

- **Orientation text.** The UP-sign rationale in the pre-registration is inconsistent with its rule for stress variables (section 5). Rule implemented.
- **Vintages.** All FRED series are latest-vintage values (no ALFRED). For VIX, Nasdaq, Baa, real yields this is immaterial; for WALCL/TGA minor. Stablecoin, hashrate and revenue series are as published today.
- **Data choices.** USD/JPY uses the ECB reference rate (Frankfurter) to avoid the week-long FRED lag; breadth is computed on the 290-symbol futures universe, which is survivor-biased and not identical to the classifier's own panel. Funding is BTCUSDT only.
- **Calibration method.** The Brier calibration uses expanding quintile edges that include the current day's score (no labels), and a smoothing constant of 20. The classifier's funding axis is not in the baseline (per your list), so the funding card's incremental result is against B1-B3 only.
- **Multiplicity.** BH families are as pre-registered: 9 DOWN AUC p-values, 9 UP, 9 incremental DOWN, 9 incremental UP, 7 event-study DOWN, 7 UP. No pooling.
- **Fetch.** Python requests hung on FRED for a while; the eight FRED files were downloaded with curl (same URLs). A background python fetch job left over from that was ignored.

## 9. Decisions for you

1. Whether open interest deserves a second, independent look. It is the only variable with consistent sign across subperiods and no single-episode dependence, but it fails multiplicity, Brier skill and the incremental test here. A fair next step is not a bigger model but a different sample: e.g. other venues/assets (ETH, the 290-symbol OI from our Coinglass DB up to 2026-06-15) to see whether the 0.60 reproduces, with the same pre-registration discipline.
2. Whether to keep spending on H1-H3 given the event-study results and the power limits. The cheaper conclusion is to park them until a vintaged forward log exists.
3. The 2026-09 dip (about -7%) is invisible to the 10%/10-day label; if that kind of move is what the flag is for, the label definition, not the features, is the issue, and that would be a new pre-registration.
