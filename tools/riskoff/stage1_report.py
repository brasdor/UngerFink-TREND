"""Builds docs/RISKOFF_STAGE1.md. Verdicts are computed by the rules in docs/RISKOFF_PREREG.md section 8, not by hand."""
import os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "riskoff")
R = pd.read_csv(os.path.join(D, "stage1_results.csv"))
ER = pd.read_csv(os.path.join(D, "stage1_event_results.csv"))
EX = pd.read_csv(os.path.join(D, "stage1_event_results_ext.csv"))
CAL = pd.read_csv(os.path.join(D, "stage1_calibration_ar1.csv"))
LT = pd.read_csv(os.path.join(D, "stage1_lead_times.csv"))


def md(df):
    cols = list(df.columns)
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join("" if (isinstance(r[c], float) and np.isnan(r[c])) else str(r[c]) for c in cols) + " |")
    return "\n".join(o)


def f(x, k=3):
    return "" if pd.isna(x) else f"{x:.{k}f}"


def verdict(r):
    small = (r.ep30_pre < 6) or (r.ep30_post < 6) or (r.n_ep30 < 12)
    surv = (r.auc >= 0.55 and r.q_auc <= 0.10 and r.brier_skill > 0 and r.auc_pre2022 > 0.5 and r.auc_2022on > 0.5
            and not small and r.loeo_below_half == 0 and r.q_inc <= 0.10)
    if surv:
        return "SURVIVES"
    if r.auc >= 0.53 and r.p_auc <= 0.10:
        return "WEAK"
    return "DEAD"


R["verdict"] = R.apply(verdict, axis=1)
R["too_small"] = R.apply(lambda r: "yes" if (r.ep30_pre < 6 or r.ep30_post < 6) else "", axis=1)


def ev_verdict(r, cap=None):
    if pd.isna(r.p_one_sided):
        return "TOO SMALL"
    if r.n_episodes < 12:
        return "TOO SMALL"
    v = "SURVIVES" if r.q_bh <= 0.10 else ("WEAK" if r.p_one_sided <= 0.10 else "DEAD")
    if cap and v == "SURVIVES":
        v = "WEAK"
    return v


ER["verdict"] = ER.apply(lambda r: ev_verdict(r, cap=r["var"].startswith("H3")), axis=1)
EX["verdict"] = EX.apply(lambda r: ev_verdict(r, cap=r["var"].startswith("H3")), axis=1)

dn = R[R.label == "DOWN"].copy(); up = R[R.label == "UP"].copy()
main_tbl = dn[["var", "n_days", "n_ep30", "auc", "auc_ci_lo", "auc_ci_hi", "p_auc", "q_auc", "brier_skill", "bs_ci_lo", "bs_ci_hi"]].copy()
main_tbl.columns = ["var", "days", "30d episodes", "AUC", "CI lo", "CI hi", "p (1-sided)", "BH q", "Brier skill", "BS lo", "BS hi"]
stab = dn[["var", "auc_pre2022", "ep30_pre", "auc_2022on", "ep30_post", "loeo_min", "loeo_max", "loeo_below_half", "n_loeo"]].copy()
stab.columns = ["var", "AUC pre-2022", "eps pre", "AUC 2022-on", "eps post", "LOEO min", "LOEO max", "LOEO <0.5", "of"]
inc = dn[["var", "inc_n", "auc_base", "auc_base_plus", "d_auc", "d_auc_ci_lo", "d_auc_ci_hi", "p_inc", "q_inc", "d_brier_skill"]].copy()
inc.columns = ["var", "OOS days", "AUC B1-B3", "AUC +var", "dAUC", "lo", "hi", "p", "BH q", "dBrier skill"]
lead = dn[["var", "n_lead_eps", "lead_hit_rate", "lead_ref_rate", "lead_lift", "lead_median"]].copy()
lead["hits"] = (dn.lead_hit_rate * dn.n_lead_eps).round().astype(int).values
lead = lead[["var", "n_lead_eps", "hits", "lead_hit_rate", "lead_ref_rate", "lead_lift", "lead_median"]]
lead.columns = ["var", "episodes", "hits", "hit rate", "chance rate", "lift", "median lead (days, hits only)"]
upt = up[["var", "n_ep30", "auc", "auc_ci_lo", "auc_ci_hi", "p_auc", "q_auc", "brier_skill", "d_auc", "p_inc", "q_inc", "verdict"]].copy()
upt.columns = ["var", "30d episodes", "AUC", "lo", "hi", "p", "BH q", "Brier skill", "dAUC incr.", "p incr.", "BH q incr.", "verdict (UP)"]

HYP = {"H4": "H4 cross-asset (VIX, Nasdaq, USD/JPY, Baa spread)", "H7": "H7 derivatives crowding (OI, basis, funding)", "H8": "H8 crypto internals (breadth change, ETH/BTC)"}
vrows = []
order = {"SURVIVES": 0, "WEAK": 1, "DEAD": 2}
for h, name in HYP.items():
    d = dn[dn["var"].str.startswith(h)].sort_values("verdict", key=lambda s: s.map(order))
    best = d.iloc[0]
    cards = ", ".join(f"{v} {x}" for v, x in zip(d["var"], d["verdict"]))
    vrows.append(dict(hypothesis=name, verdict=best.verdict, cards=cards))
for h, name in [("H1", "H1 miner margin (event study)"), ("H2", "H2 liquidity (event study)"), ("H3", "H3 stablecoins (event study, not point-in-time)")]:
    d = ER[(ER.label == "DOWN") & ER["var"].str.startswith(h)].sort_values("verdict", key=lambda s: s.map(order))
    vrows.append(dict(hypothesis=name, verdict=d.iloc[0].verdict, cards=", ".join(f"{v} {x}" for v, x in zip(d["var"], d["verdict"]))))
VT = pd.DataFrame(vrows)

ev_tbl = ER[ER.label == "DOWN"][["var", "window_start", "n_episodes", "T_obs", "null_mean", "null_sd", "p_one_sided", "q_bh", "verdict"]].copy()
ev_tbl.columns = ["var", "window start", "episodes", "T observed", "null mean", "null sd", "p (1-sided)", "BH q", "verdict"]
ev_up = ER[ER.label == "UP"][["var", "n_episodes", "T_obs", "null_mean", "p_one_sided", "q_bh", "verdict"]].copy()
ev_up.columns = ["var", "episodes", "T obs", "null mean", "p", "BH q", "verdict"]
ex_tbl = EX[EX.label == "DOWN"][["var", "window_start", "n_episodes", "T_obs", "null_mean", "p_one_sided", "q_bh", "verdict"]].copy()
ex_tbl.columns = ["var", "window start", "episodes", "T obs", "null mean", "p", "BH q", "verdict"]
ex_up = EX[EX.label == "UP"][["var", "n_episodes", "p_one_sided", "q_bh", "verdict"]].copy()
ex_up.columns = ["var", "episodes", "p", "BH q", "verdict (UP, sensitivity)"]
cal = CAL.copy()
cal.columns = ["var", "n days", "lag-1 autocorr", "AUC obs", "AUC null sd", "AUC null 95th pct", "p vs AR(1) noise"]

for c in main_tbl.columns[3:]:
    main_tbl[c] = main_tbl[c].map(lambda x: f(x, 3))
for t in (stab, inc, upt, lead):
    for c in t.columns:
        if t[c].dtype == float:
            t[c] = t[c].map(lambda x: f(x, 3))

TXT = f"""# Risk-off flag - Stage 1 report (univariate tests, pre-registered)

Pre-registration: `docs/RISKOFF_PREREG.md`, commit `12e10e2e` (2026-10-11 10:44 +02:00), committed before any feature or statistic was computed.
Code: `tools/riskoff/stage1_fetch.py`, `stage1_run.py`, `stage1_calibration.py`, `stage1_report.py`. Results in `data/riskoff/stage1_*.csv`. Not touched: engines, workflows, allocator, Telegram. No variables were combined and no risk model was built; the logistic regressions in the incremental test are evaluation scaffolding only.

## 1. Verdicts

Computed mechanically from the rules in the pre-registration (section 8). Target: BTC down 10% within 10 days. A hypothesis takes the verdict of its best card.

{md(VT)}

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

{md(main_tbl)}

The "too small" flag (fewer than 6 episodes in a subperiod) applies to all three H7 cards; it blocks SURVIVES. AUC is oriented so that > 0.5 means the variable moves in the pre-registered direction before drops. "Brier skill" is `1 - Brier(model)/Brier(base rate)` from the walk-forward quintile calibration; positive = beats always predicting the base rate. **Only H7-OI has a positive point estimate (+0.005) and its interval spans zero.** Even where the AUC is above 0.5, none of these variables improves probability forecasts over the base rate. The walk-forward AUC (after the 365-day burn-in) is 0.625 for OI, 0.558 for funding and below 0.55 for everything else.

### Stability: subperiods and leave-one-episode-out

{md(stab)}

- **H7-OI is the most stable variable**: 0.601 before 2022, 0.592 after, and never below 0.571 when any single episode is removed.
- **H7-FUND and H7-BAS are not stable.** Funding: 0.687 before 2022, 0.519 after. Basis: 0.710 and 0.473. Removing the Jan-Jul 2021 episode alone drops basis from 0.579 to 0.513 and funding from 0.598 to 0.550.
- **Pre-2022 holds only 4-5 episodes for H7** (below the pre-registered minimum of 6), so the subperiod comparison for H7 cannot support a stability claim either way. That is the "too small" flag.

### Incremental value over the regime classifier's inputs (walk-forward, sign-free)

{md(inc)}

Only funding (+0.044, p 0.096) and breadth change (+0.012, p 0.098) come near raw significance and neither survives BH (q 0.43). Note this test fits the coefficient freely, so it is sign-blind by design (pre-registered).

### Lead time before the 41 primary episodes (alarm = score in its own top quintile so far)

{md(lead)}

"Chance rate" = share of all days that had an alarm in the preceding 11 days. No variable gives a useful lead: hit rates are 18-42% against chance rates of 10-49%. The best lift is funding (5 hits of 28 episodes, 1.8x chance, median lead 12 days), which rests on five hits: one in Nov 2021 and four in 2024, the 2024 ones being small drops of 10-16%. Open interest hit 8 of 26 episodes (6 of them in 2021-12 to 2023-04, lead 0-11 days). The funding and basis alarms did not fire before the large 2020-21 and 2022 drops in a way that beats chance. Lead times are not interpretable where the lift is near 1.

## 4. Why H7 is "weak" and not more

- Open interest: raw p 0.03, but BH q 0.13 (fails the 0.10 gate), Brier skill indistinguishable from 0, no significant incremental value (dAUC +0.018, p 0.14), 4 pre-2022 episodes. It passes subperiod and leave-one-out checks, which is the reason it is the first thing to look at again.
- Funding: raw p 0.02, but q 0.13, negative Brier skill, signal concentrated before 2022.
- Basis: p 0.14, and dependent on a single 2021 episode.
- These are 9 tests on about 20 episodes. A top p of 0.02-0.03 among 9 is not rare under no effect (probability that the best of 9 independent tests has p <= 0.02 is about 17%).

## 5. UP label (secondary, for the short systems)

{md(upt)}

All nine are DEAD under the pre-registered orientation (expected sign opposite to the DOWN sign). One observation, not a hypothesis test: H4-VIX has oriented AUC 0.44 for UP, i.e. a high VIX is followed by *more* 10% up-moves, and its incremental dAUC is +0.065 (raw p 0.024, BH q 0.22, not significant after correction). That is consistent with VIX acting as a two-sided volatility indicator, but it is exploratory. The pre-registration text gave a rationale for the UP signs ("stress extremes precede rebounds") that contradicts its own rule ("opposite of DOWN") for stress variables; I implemented the rule as written and the contradiction is listed in section 8.

## 6. Event study (H1, H2, H3), DOWN label

Statistic: mean oriented z-score over days -10..0 before each episode start, against a circular-shift null (5,000 draws). Window starts 2020-09-06 because the expanding z-score needs 365 days of history inside our label calendar.

{md(ev_tbl)}

UP label (secondary): {md(ev_up)}

Sensitivity (post-hoc, A1): z-scores computed from the full vendor history (2017-12 onward), which adds the 2019-2020 episodes:

{md(ex_tbl)}

{md(ex_up)}

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

{md(cal)}

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
"""
with open(os.path.join(ROOT, "docs", "RISKOFF_STAGE1.md"), "w", encoding="utf-8") as fh:
    fh.write(TXT)
print(VT.to_string(index=False))
print(R[["var", "label", "verdict", "too_small"]].to_string(index=False))
