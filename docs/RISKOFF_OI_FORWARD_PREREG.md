# Risk-off: open-interest forward pre-registration

Committed BEFORE the forward snapshot log exists and before any statistic is computed on data after 2026-08-28. The commit timestamp is the evidence.

Purpose: the one Stage 1 variable that was stable (H7-OI, verdict WEAK, see `docs/RISKOFF_STAGE1.md`) gets one out-of-sample test on data that was not in the Stage 1 sample. **The specification is exactly the one tested. No parameter, window, transformation, lag, label or threshold may change.** Any change makes this a new hypothesis that needs its own pre-registration and does not count as a confirmation.

Does not build a flag, a model or an alert. Does not touch engines, the allocator or Telegram.

## 1. Specification (copied from `docs/RISKOFF_PREREG.md`, commit `12e10e2e`, card H7-OI)

| item | value |
|---|---|
| Variable | BTCUSDT perpetual open interest in coins (`sum_open_interest`), the LAST 5-minute row of each UTC day in the Binance public data daily `metrics` file (`data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/`) |
| Cleaning | a daily value <= 0 is treated as missing; missing days are forward-filled for at most 3 days (rule A0 of Stage 1, kept) |
| Transformation | `x_t = ln(OI_t) - ln(mean of OI over the last 60 calendar days ending at t)` |
| Lag | 1 day: the value dated t is usable from day t+1 |
| Sign | + (higher x = more risk-off); score `s = x` |
| Label | DOWN: `y[D] = 1` if `min(close[D+1..D+10]) <= 0.90 * close[D]`, BTCUSDT perpetual daily close (Binance), same data source as Stage 0 |
| Primary statistic | oriented AUC of `s` vs `y` (Mann-Whitney, ties = 0.5) |
| Cluster bootstrap | 2,000 draws, clusters = 30-day-merge episodes plus the quiet period before the next one; one-sided p = (1 + #draws with AUC <= 0.5) / 2001; seed 20261012 |
| Secondary | walk-forward Brier skill (quintile calibration, smoothing 20, >= 365 resolved training days, expanding, training days <= D-11) and walk-forward incremental dAUC over the regime classifier inputs B1-B3 (logistic, C = 1, refit every 30 days), exactly as in Stage 1 section 6 |
| In-sample reference (not to be re-estimated) | AUC 0.603 [0.496, 0.685], p 0.030, BH q 0.133; pre-2022 0.601, 2022-on 0.592; leave-one-out min 0.571; Brier skill +0.005; incremental dAUC +0.018 (p 0.14). Sample 2020-10-31 to 2026-08-28, 18 episodes |

Training history for the walk-forward pieces is the Stage 1 sample plus accruing forward days; the AUC itself is computed on forward days only.

## 2. Samples

- **Forward sample F2 (the decision sample):** days D >= 2026-10-12 whose label is fully resolved (D + 10 days observed). Features for these days come only from data published after this commit's date or computed by the unchanged pipeline.
- **Gap segment F1 (informational only):** D in 2026-08-29 to 2026-09-28. Stage 1 ended its H7 window on 2026-08-28 (funding ends there), so F1 was not evaluated. Disclosure: the OI feature values for F1 exist in `data/riskoff/stage1_features_daily.csv` and the labels exist; **no statistic has been computed on F1** as of this commit. F1 is reported once, at the analysis date, next to F2 but does NOT enter the decision, because the data were already published when this was written.
- Days 2026-09-29 to 2026-10-11 belong to neither segment.

## 3. Data collection (audit trail)

- The authoritative series is the Binance public data file above (same as tested), downloaded when available (about 1 day after the day ends).
- An append-only snapshot log, `data/riskoff/forward/oi_btcusdt_5m.csv`, is written by the daily data job on the funding host: Binance API `/futures/data/openInterestHist` (period 5m, last 500 bars) for BTCUSDT. Columns: bar time, `sum_open_interest`, `sum_open_interest_value`, `first_seen_utc`. A value already in the log is never overwritten; a later different value is written to `oi_restatements.csv`.
- Use of the log: (a) fallback only if the public file for a day is missing after 7 days; (b) audit. Any day where the log's last bar differs from the public file by more than 0.1% is listed in the report.
- The job runs on a machine that can reach `fapi.binance.com` (US runners get HTTP 451). Gaps longer than ~40 hours in the 5m log cannot be backfilled from the API; the public file covers them.

## 4. Decision rule (one analysis, no peeking)

- **Analysis date:** the first date on which F2 contains >= 10 resolved 30-day-merge DOWN episodes, or 2029-10-12, whichever comes first. The performance script is not run before then. Allowed before: data-completeness checks that compute no AUC, Brier or p-value.
- **CONFIRMED:** oriented AUC on F2 >= 0.55 AND one-sided cluster-bootstrap p <= 0.05 AND walk-forward Brier skill point estimate > 0. This is a single pre-specified test, so no multiplicity correction applies; it does not remove the multiplicity already attached to the Stage 1 selection (open interest was chosen as the best of 9 tests, and CONFIRMED is evidence only if the forward result is clearly separate from that).
- **REFUTED:** oriented AUC on F2 < 0.53 with >= 10 episodes.
- **INCONCLUSIVE:** anything else, including fewer than 6 episodes at 2029-10-12 (then extend once to 2031-10-12 with no other change).
- Incremental dAUC and F1 are reported but do not change the verdict.

## 5. Power, stated in advance

The event rate is about 3 independent 30-day episodes per year (21 in 7 years), so 10 forward episodes take about 3.3 years (analysis around early 2030). With about 10 episodes the cluster-bootstrap interval for an AUC is roughly +/- 0.10 wide, so only a true AUC around 0.65 or higher is likely to reach p <= 0.05; if the real value is the in-sample 0.60 the most likely outcome is INCONCLUSIVE, and the in-sample 0.60 is itself probably optimistic (best of 9). This test can refute or detect a large effect; it cannot cheaply confirm a modest one. The decision to wait is Jean's; nothing here is gating the live systems.

## 6. Amendments

Deviations, data outages and any analysis added later are listed in the forward report under "Amendments" and are not pre-registered.
