# Risk-off research, Stage 1: pre-registration

Written and committed BEFORE any Stage 1 feature, AUC, Brier score, bootstrap or event-study statistic has been computed. The commit timestamp is the evidence. Anything not specified here is decided in the Stage 1 report under "Amendments" and labelled as post-hoc.

Scope: Stage 1 tests H4, H7, H8. Event study only for H1, H2, H3. H5 and H6 are skipped. No variables are combined into a flag and no risk model is fitted. The only fitted objects are evaluation scaffolding (section 6) and are never exported.
Not touched: engines, workflows, the allocator, Telegram.

## 0. Frozen inputs

| item | value |
|---|---|
| Stage 0 labels | `data/riskoff/labels_daily.csv` sha256 prefix `e870c710840a7d15` |
| Stage 0 events | `data/riskoff/events.csv` sha256 prefix `240a03223d282553` |
| Stage 0 BTC closes | `data/riskoff/target_series.csv` sha256 prefix `6234536e51c0c181` |
| Target series | BTCUSDT perpetual daily close, 2019-09-08 to 2026-10-08 |
| Valid label days | D <= 2026-09-28 (last 10 days unlabeled) |
| Seeds | numpy `default_rng(20261011)` for every bootstrap/permutation |

## 1. Labels (fixed)

- **DOWN (primary):** `y_dn[D] = 1` if `min(close[D+1..D+10]) <= 0.90 * close[D]`. This is the Stage 0 `BTC_10pct_10d` column.
- **UP (secondary, for the short systems):** `y_up[D] = 1` if `max(close[D+1..D+10]) >= 1.10 * close[D]`.
- Both are labels only and never enter a feature.
- **Episodes:** flagged days merged when <= 30 days apart (the cluster unit for all bootstraps). For lead times only, the 15-day-merge primary episode list in `events.csv` is used (41 episodes). Up episodes are built the same way from `y_up`.

## 2. Point-in-time rules

A feature value for date D may use only observations whose publication time is before the end of D (UTC). Implementation: the transformation is computed on the series' native observations, then each observation dated t becomes usable from calendar day `t + L` (lag L below) and is forward-filled to calendar days until the next observation becomes usable. Weekends therefore carry the last business-day value. No backfill. Series are used as downloaded on the test date; vintages exist only for FRED and were not fetched, so revision risk is stated per card.

## 3. Hypothesis cards

Orientation: every card has an expected sign for the DOWN label. The oriented score `s = sign * x` is high when risk-off is predicted. For the UP label the expected sign is the OPPOSITE of the DOWN sign (stress extremes precede rebounds; crowding precedes down-moves, not up-moves). The UP signs are a weak prior and exploratory.

Global kill condition (applies to every Stage 1 card, per label): dead if oriented AUC < 0.53, or the one-sided cluster-bootstrap p > 0.10, or the sign is opposite to the card.

### Stage 1 (AUC, Brier, bootstrap, LOEO, subperiods, incremental, lead time)

| id | channel | variable | transformation x (on native observations) | lag L | sign (DOWN) | sample window | revision risk |
|---|---|---|---|---|---|---|---|
| H4-VIX | cross-asset fear | VIX close (FRED VIXCLS) | `ln(VIX_t) - ln(mean of last 60 obs)` | 1 | + | 2019-09-08 to 2026-09-28 | none |
| H4-NDQ | equity risk appetite | Nasdaq Composite (FRED NASDAQCOM) | `NDQ_t / max(last 20 obs) - 1` (drawdown from 20-obs high) | 1 | - (deeper drawdown = more risk) | same | none |
| H4-JPY | carry unwind | USD/JPY ECB reference rate (Frankfurter API; FRED DEXJPUS is lagged ~1 week and is NOT used) | `ln(USDJPY_t / USDJPY_{t-10 obs})` | 1 | - (yen strength) | same | none |
| H4-BAA | credit stress | Baa-10y Treasury spread (FRED BAA10Y), in percentage points | `BAA10Y_t - mean(last 60 obs)` | 2 | + | same | none |
| H7-OI | leverage build-up | BTCUSDT open interest, coins (`sum_open_interest`, last 5-min snapshot of day t; binance.vision daily metrics) | `ln(OI_t) - ln(mean of last 60 days)` | 1 | + | **2020-09-01 to 2026-08-28** | none |
| H7-BAS | long crowding via basis | BTCUSDT perp premium index, 1d kline close (binance.vision premiumIndexKlines) | mean of last 7 daily closes | 1 | + | same | none |
| H7-FUND | long crowding via funding | BTCUSDT funding rate per settlement (our local file; last print 2026-08-28 08:00) | mean of all settlements in the last 3 days | 1 | + | same | none |
| H8-BRD | internal breadth | `% of the 290 futures symbols above their own EMA200` (min_periods 20), EMA span 3, i.e. exactly the regime classifier's `breadth_ema3` | `breadth_t - breadth_{t-10}` (a CHANGE, because the level is a baseline input) | 0 (own data, bar closed) | - | 2019-09-08 to 2026-09-28 (from the first day with >= 20 symbols) | none; survivor-biased universe |
| H8-ETHBTC | alt/majors rotation | ETHUSDT / BTCUSDT daily close | `ln(ratio_t / ratio_{t-10})` | 0 | - | same | none |

Window note: H7 uses 2020-09 through 2026-08-28 because OI/basis files begin 2020-09 and funding ends 2026-08-28; labels for these days exist. H7 tests are therefore not comparable in sample to H4/H8, and are reported with their own episode counts.

### Event study only (H1, H2, H3)

| id | channel | variable | transformation x | lag L | sign (DOWN) | revision / PIT caveat |
|---|---|---|---|---|---|---|
| H1-HP | miner margin | hashprice in BTC = (miners revenue USD / BTC price) / hashrate, blockchain.com | `ln(7d mean_t / 7d mean_{t-30})` | 1 | - | not vintaged; derived series |
| H1-HR | miner capitulation | hashrate, blockchain.com | `ln(7d mean_t / 7d mean_{t-30})` | 1 | - | estimate, not restated |
| H2-NL | dollar liquidity | `WALCL - WTREGEN - 1000*RRPONTSYD` (millions USD), Wednesday grid | 13-week log change | 2 (Wednesday value usable from Friday) | - | latest-revised FRED values, small revisions; no vintage |
| H2-RY | real rates | 10y TIPS yield (FRED DFII10), percentage points | 10-obs change | 2 | + | none |
| H2-USD | dollar strength | broad dollar index (FRED DTWEXBGS) | 20-obs % change | 8 (H.10 weekly lag) | + | minor annual reweighting |
| H3-STB | stablecoin dry powder | total stablecoin circulating USD (DefiLlama) | 30-day % change | 1 | - | **history restated by vendor; NOT point-in-time** |
| H3-USDT | issuer-level | USDT circulating (DefiLlama) | 30-day % change | 1 | - | as above |

Not tested, by design: power-cost proxy for H1 (no usable daily free series, see Stage 0); H5; H6.

## 4. Tests (exact list; nothing else is run on Stage 1 variables)

For each Stage 1 card, each label (DOWN primary, UP secondary):

1. **AUC** of oriented score vs label on all valid days in the card's window where the score exists (Mann-Whitney, ties = 0.5).
2. **Walk-forward Brier skill** vs the expanding base rate (section 6.1).
3. **Cluster bootstrap** of AUC and Brier skill: 2,000 draws, resampling clusters with replacement (section 5). Report 95% percentile CI and the one-sided p = (1 + #draws with AUC <= 0.5) / (B + 1).
4. **Leave-one-episode-out (LOEO)** AUC: for each 30-day-merge episode k remove all days in `[start_k - 10, trough_k]`, recompute AUC. Report min, max, count of AUC < 0.5, and the most influential episode.
5. **Subperiods:** AUC for D < 2022-01-01 and D >= 2022-01-01, with the number of 30-day episodes in each.
6. **Incremental value** over the regime classifier inputs (section 6.2): walk-forward paired change in AUC and in Brier skill, cluster-bootstrap p.
7. **Lead time** (section 7).
8. **Benjamini-Hochberg** at q = 0.10 over: (a) the 9 DOWN-label AUC p-values (primary family); (b) the 9 UP-label AUC p-values (secondary family); (c) the 9 DOWN-label incremental ΔAUC p-values; (d) the 9 UP incremental p-values. Families are never pooled and never re-defined after seeing results.

For each event-study card (H1-H3), each label: build the episode list (30-day merge) inside the variable's PIT-usable window; compute the standardized feature `z = (x - expanding mean) / expanding std` (min 365 days, PIT) ; statistic `T` = mean over episodes of the mean of oriented `z` over event days -10..0 relative to the episode start; null = 5,000 draws of a common circular shift of the episode start set within the window (spacing preserved); one-sided p. Also save the event-time path -30..+10. BH at q = 0.10 over the 7 DOWN tests (family e) and, separately, the 7 UP tests (family f). A minimum of 12 episodes in the window is required for any verdict beyond "too small".

## 5. Clusters

Cluster id of day D = index of the latest 30-day-merge episode start <= D within the evaluation sample (days before the first start form cluster 0). Bootstrap resamples whole clusters, so serial correlation in labels and features stays inside a cluster. Episodes are those of the label being tested.

## 6. Evaluation scaffolding (not a deliverable)

**6.1 Walk-forward calibration for Brier.** At day D: bin edges = quintiles of the oriented score over days <= D. Training days = days j <= D - 11 (label resolved) with a score; `p_D = (k + m * base) / (n + m)` where `k, n` = positives and count among training days in D's bin, `base` = expanding positive rate over training days, `m = 20`. Needs >= 365 resolved training days. Brier skill `= 1 - Brier(p) / Brier(base)`, both on the same evaluated days.

**6.2 Baseline = the current regime classifier's inputs, as defined in `regime_allocator_v4.py`:**
- B1 = `ln(BTC close / EMA200)`, EMA span 200, `adjust=False`;
- B2 = `breadth_ema3` (the classifier's 3-day-EMA'd % of symbols above their EMA200), computed from the 290 futures closes (the classifier's own panel may differ slightly);
- B3 = ATR14 percentile over the trailing 252 days.
The classifier's funding axis is NOT part of the baseline (the brief names three inputs), so H7-FUND is not disadvantaged by incremental testing against itself only if it adds beyond B1-B3; this is stated, not corrected.
Model A = L2 logistic regression (C = 1, standardized inputs) on B1-B3; model B = A plus the oriented standardized candidate score. Refit every 30 days on days j <= D - 11, >= 365 resolved days. Out-of-sample paired ΔAUC = AUC(B) - AUC(A) and ΔBrier skill, with a cluster bootstrap p (one-sided, ΔAUC <= 0). For H8-BRD the candidate is the 10-day CHANGE in breadth, so it is not collinear with B2 by construction.

## 7. Lead time

Alarm is ON on day D if the oriented score is >= its 80th percentile over days <= D (min 365 days). For each of the 41 primary 15-day-merge DOWN episodes with a score available: HIT if an alarm day exists in `[start - 10, start]`; lead = start minus the onset of the contiguous alarm run containing the first such day (capped at 60). Report per-episode lead, hit rate, median lead, and the reference rate = share of all days with an alarm in the trailing 11 days (the chance hit rate); lift = hit rate / reference. The alarm rule is for measuring lead time only and is not a flag proposal.

## 8. Verdict rules (per hypothesis, per label)

- **SURVIVES** (Stage 1 cards), all required: oriented AUC >= 0.55 with the expected sign; BH q <= 0.10 in its family; walk-forward Brier skill > 0; AUC > 0.5 in both subperiods and each subperiod holds >= 6 episodes; LOEO AUC > 0.5 for every dropped episode; incremental ΔAUC with BH q <= 0.10 in its family.
- **WEAK:** expected sign, AUC >= 0.53 and raw one-sided p <= 0.10, but at least one SURVIVES condition fails.
- **DEAD:** otherwise (kill condition met).
- **TOO SMALL overlay:** if the sample has < 12 episodes, or either subperiod < 6, the report states "no conclusion on stability" and the verdict cannot be SURVIVES.
- Event-study cards: **SURVIVES (event-study level)** if BH q <= 0.10, expected sign, >= 12 episodes; **WEAK** if raw p <= 0.10 otherwise; **DEAD** otherwise. H3 cards can never exceed WEAK (not point-in-time).
- A hypothesis verdict is the verdict of its best card for the DOWN label; UP verdicts are reported separately and are secondary.

## 9. Amendments

Deviations, data problems and any analysis added after the first results are listed in the Stage 1 report under "Amendments" with the reason, and are not counted as pre-registered.
