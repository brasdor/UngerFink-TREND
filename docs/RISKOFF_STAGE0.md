# Risk-off flag - Stage 0 report (event catalogue + data inventory)

Status: read-only research. No flag, model, alert or Telegram call was built; no live system, engine, workflow or allocator file was touched.
Generated 2026-10-10 by `tools/riskoff/stage0_*.py`. Outputs in `data/riskoff/`: `events.csv`, `labels_daily.csv`, `base_rates.csv`,
`known_episodes_check.csv`, `episode_counts_by_window.csv`, `target_series.csv`, `data_inventory.csv`.

## Headline

- **Primary definition (10% / 10d): 41 BTC episodes, 36 basket episodes** at a 15-day merge gap. This number is fragile: re-merging the *same* flags at 30 / 45 days gives **21 / 16 (BTC)** and **13 / 9 (basket)**. Only **13** BTC episodes fell 20% or more, only **5** fell 30%+.
- **No vintage data:** H3 stablecoins (DefiLlama restates history), H6 (EPU, GPR are recomputed), H8 BTC dominance, and H5 consensus (no free history at all). Hashrate, difficulty and the settled/derived market series are not restated but also have no vintage archive. Only FRED series (H2, H4, H5 actuals) have real vintages, and the keyless ALFRED host answered 1 of 4 requests, so a free FRED API key is needed.
- **Unreachable from CI:** Binance funding (HTTP 451, already known) and everything Coinglass (paid key, local). Unverified: Yahoo-sourced series (gold, TTF) are likely to be blocked. **I did not test any host from a GitHub runner**, because that needs a workflow; reachability ratings other than the two documented cases are inference.
- **Blocks for point-in-time testing:** see section 5. The sharpest: ICE BofA high-yield spread is only available for 3 years (from 2023-10-10), and no free liquidation history exists.

## 1. Events

**Rule (fixed before looking at results, not tuned):** day D is risk-off if `min(close[D+1..D+h]) <= (1-thr) * close[D]`. Daily bars, calendar days (crypto has no non-trading days). The last `h` days of each series are unlabeled (NaN, not 0): labels end 2026-09-28 for the 10d horizon. The label uses future data by construction and lives only in `labels_daily.csv`; nothing in the repo imports it as a feature.

**Targets and real start dates**

| target | construction | first date | last date | days |
|---|---|---|---|---|
| BTC | BTCUSDT perpetual daily close (`data/futures_universe/ohlcv_1d`) | 2019-09-08 | 2026-10-08 | 2584 |
| EW basket | daily-rebalanced mean of simple returns over the 290-symbol futures universe, needing >=5 constituents | 2020-01-10 | 2026-10-08 | 2458 |

Caveats on the basket (not corrected, stated so you can judge the number):
- **Survivorship:** the universe is the 290 symbols alive today. Dead tokens are absent, which flatters the basket's early history; new listings join late and tend to bleed after launch, which drags it down. The two effects do not cancel in a known way.
- **67 symbol-day returns with |return| > 80% were dropped** as probable bad prints or redenominations (guard in `stage0_events.py`). I did not audit them one by one.
- The basket is *much* noisier than BTC: 30.4% of days flagged vs 14.3%.

**Base rates and episode counts** (cell = share of labeled days flagged / episodes at a 15-day merge gap). Only 10%/10d is the primary definition; the rest are sensitivity and none was chosen on results.

BTC (labeled 2574 days):

| horizon | 7% | 10% | 15% |
|---|---|---|---|
| 5d | 13.3% / 44 ep | 6.8% / 35 ep | 2.1% / 17 ep |
| 10d | 24.6% / 44 ep | 14.3% / 41 ep **(primary)** | 6.0% / 27 ep |
| 20d | 38.4% / 29 ep | 26.7% / 30 ep | 14.7% / 28 ep |

EW basket (labeled 2448 days):

| horizon | 7% | 10% | 15% |
|---|---|---|---|
| 5d | 27.1% / 46 ep | 17.0% / 42 ep | 7.4% / 39 ep |
| 10d | 40.8% / 37 ep | 30.4% / 36 ep **(primary)** | 17.6% / 37 ep |
| 20d | 55.4% / 20 ep | 45.6% / 29 ep | 32.9% / 30 ep |

**Declustering:** flagged days at most 15 days apart are merged. Start = first flagged day. Trough = lowest close in [start, last flagged day + h]. Size = trough close / close at start - 1 (so it is measured from the first flagged day, not the peak).

**Episode count depends on the merge gap, which the brief fixed at 15 days as an example.** Same primary flags, different gaps (reporting only, not a choice):

| merge gap | BTC episodes | basket episodes |
|---|---|---|
| 15 d | 41 | 36 |
| 30 d | 21 | 13 |
| 45 d | 16 | 9 |

Episodes by window (primary definition) - what each data history can actually see:

| window | target | episodes | ge20pct | ge30pct | ge40pct |
|---|---|---|---|---|---|
| full history | BTC | 41 | 13 | 5 | 2 |
| full history | EW_BASKET | 36 | 22 | 13 | 9 |
| from 2020-09 (Binance vision OI/premium) | BTC | 34 | 10 | 4 | 1 |
| from 2020-09 (Binance vision OI/premium) | EW_BASKET | 33 | 20 | 11 | 7 |
| from 2021-01 | BTC | 33 | 10 | 4 | 1 |
| from 2021-01 | EW_BASKET | 32 | 20 | 11 | 7 |
| from 2023-01 | BTC | 17 | 3 | 0 | 0 |
| from 2023-01 | EW_BASKET | 21 | 13 | 7 | 4 |
| from 2023-10-10 (FRED ICE BofA HY OAS) | BTC | 14 | 3 | 0 | 0 |
| from 2023-10-10 (FRED ICE BofA HY OAS) | EW_BASKET | 16 | 10 | 7 | 4 |

### Sanity check against known episodes (rule not adjusted)

| episode | window | btc_max_drawdown_in_window_pct | BTC_10pct_10d | EW_BASKET_10pct_10d | BTC_7pct_10d | BTC_15pct_10d |
|---|---|---|---|---|---|---|
| May 2021 (China mining ban / 19-May crash) | 2021-05-10..2021-05-25 | -38.9 | 11/16 days flagged | 13/16 days flagged | 13/16 days flagged | 9/16 days flagged |
| May 2022 (Terra/LUNA) | 2022-05-04..2022-05-14 | -26.9 | 5/11 days flagged | 7/11 days flagged | 6/11 days flagged | 5/11 days flagged |
| Nov 2022 (FTX) | 2022-11-05..2022-11-11 | -25.3 | 4/7 days flagged | 5/7 days flagged | 6/7 days flagged | 3/7 days flagged |
| Aug 2024 (yen carry unwind, 5-Aug) | 2024-07-30..2024-08-06 | -18.4 | 5/8 days flagged | 5/8 days flagged | 6/8 days flagged | 3/8 days flagged |
| 2025-26 decline (6-Oct-25 peak -> Feb-26) | 2025-10-06..2026-02-06 | -49.6 | 31/124 days flagged | 57/124 days flagged | 53/124 days flagged | 15/124 days flagged |
| Sep 2026 dip (as seen in our data) | 2026-09-01..2026-10-08 | -6.9 | 0/28 days flagged | 0/28 days flagged | 0/28 days flagged | 0/28 days flagged |

Reading: the rule captured May 2021, May 2022, Nov 2022, Aug 2024 and the 2025-26 decline (several episodes inside it) on BTC and on the basket. It **missed the September 2026 dip**: in our data BTC's largest drawdown inside 1 Sep - 8 Oct is -6.9%, below even the 7% threshold, and the last 10 days are unlabeled anyway. If the dip you have in mind was intraday or in another instrument, it is not visible in daily futures closes. Also note the Aug 2026 jump (BTC 63k -> 78k in 3 days, 17-21 Aug) is continuous in the source file, not a data gap.

### Episode tables (primary 10% / 10d, 15-day merge)

**BTC - 41 episodes**

| start | trough | size_% (start close -> trough) | days start->trough | flagged days |
|---|---|---|---|---|
| 2019-09-14 | 2019-09-29 | -22.2 | 15 | 10 |
| 2019-10-14 | 2019-10-24 | -11.2 | 10 | 1 |
| 2019-11-10 | 2019-11-24 | -23.7 | 14 | 11 |
| 2019-12-07 | 2019-12-17 | -11.6 | 10 | 2 |
| 2020-02-16 | 2020-03-12 | -52.0 | 25 | 22 |
| 2020-05-07 | 2020-05-11 | -14.3 | 4 | 6 |
| 2020-08-24 | 2020-09-08 | -13.8 | 15 | 10 |
| 2020-11-24 | 2020-11-27 | -10.6 | 3 | 1 |
| 2021-01-07 | 2021-01-27 | -23.1 | 20 | 13 |
| 2021-02-17 | 2021-02-28 | -13.5 | 11 | 6 |
| 2021-03-13 | 2021-03-25 | -16.2 | 12 | 7 |
| 2021-04-11 | 2021-06-25 | -47.5 | 75 | 43 |
| 2021-07-10 | 2021-07-20 | -11.1 | 10 | 2 |
| 2021-09-03 | 2021-09-21 | -18.6 | 18 | 11 |
| 2021-10-20 | 2021-10-27 | -11.5 | 7 | 1 |
| 2021-11-08 | 2021-12-13 | -31.0 | 35 | 19 |
| 2021-12-26 | 2022-01-22 | -31.0 | 27 | 19 |
| 2022-02-10 | 2022-02-21 | -14.9 | 11 | 12 |
| 2022-03-29 | 2022-04-11 | -16.7 | 13 | 7 |
| 2022-04-28 | 2022-05-18 | -27.8 | 20 | 11 |
| 2022-06-02 | 2022-06-18 | -37.7 | 16 | 18 |
| 2022-08-09 | 2022-08-28 | -15.5 | 19 | 10 |
| 2022-09-10 | 2022-09-21 | -14.8 | 11 | 3 |
| 2022-10-29 | 2022-11-09 | -23.5 | 11 | 11 |
| 2023-02-27 | 2023-03-10 | -14.2 | 11 | 6 |
| 2023-04-13 | 2023-04-21 | -10.2 | 8 | 4 |
| 2023-08-08 | 2023-08-18 | -12.5 | 10 | 8 |
| 2024-01-08 | 2024-01-18 | -12.0 | 10 | 4 |
| 2024-03-10 | 2024-03-19 | -10.2 | 9 | 6 |
| 2024-04-07 | 2024-05-01 | -15.9 | 24 | 8 |
| 2024-06-30 | 2024-07-07 | -11.0 | 7 | 3 |
| 2024-07-25 | 2024-08-05 | -17.9 | 11 | 10 |
| 2024-08-23 | 2024-09-05 | -12.3 | 13 | 4 |
| 2024-12-16 | 2024-12-27 | -11.1 | 11 | 2 |
| 2025-02-16 | 2025-03-10 | -18.2 | 22 | 11 |
| 2025-03-27 | 2025-04-08 | -12.5 | 12 | 2 |
| 2025-10-05 | 2025-10-17 | -13.8 | 12 | 5 |
| 2025-10-26 | 2025-11-22 | -26.0 | 27 | 12 |
| 2026-01-21 | 2026-02-05 | -29.7 | 15 | 15 |
| 2026-03-17 | 2026-03-27 | -10.2 | 10 | 1 |
| 2026-05-23 | 2026-06-06 | -20.7 | 14 | 10 |

**EW basket - 36 episodes**

| start | trough | size_% (start close -> trough) | days start->trough | flagged days |
|---|---|---|---|---|
| 2020-02-12 | 2020-03-12 | -64.2 | 29 | 32 |
| 2020-05-02 | 2020-05-11 | -11.5 | 9 | 3 |
| 2020-08-24 | 2020-11-04 | -43.9 | 72 | 40 |
| 2020-11-23 | 2020-12-23 | -12.3 | 30 | 17 |
| 2021-01-09 | 2021-01-11 | -11.2 | 2 | 6 |
| 2021-02-13 | 2021-02-28 | -18.8 | 15 | 10 |
| 2021-03-19 | 2021-03-25 | -10.9 | 6 | 2 |
| 2021-04-12 | 2021-07-20 | -55.3 | 99 | 64 |
| 2021-09-02 | 2021-09-21 | -21.8 | 19 | 16 |
| 2021-11-08 | 2022-01-24 | -52.9 | 77 | 47 |
| 2022-02-07 | 2022-02-21 | -28.4 | 14 | 17 |
| 2022-03-29 | 2022-06-18 | -68.7 | 81 | 62 |
| 2022-08-08 | 2022-09-21 | -29.3 | 44 | 20 |
| 2022-10-29 | 2022-11-09 | -30.8 | 11 | 12 |
| 2022-12-06 | 2022-12-19 | -19.2 | 13 | 10 |
| 2023-02-07 | 2023-03-11 | -27.1 | 32 | 20 |
| 2023-04-13 | 2023-05-11 | -23.9 | 28 | 15 |
| 2023-05-28 | 2023-06-14 | -27.3 | 17 | 13 |
| 2023-08-07 | 2023-08-17 | -16.8 | 10 | 9 |
| 2023-10-01 | 2023-10-11 | -10.9 | 10 | 1 |
| 2023-11-11 | 2023-11-21 | -10.5 | 10 | 4 |
| 2023-12-27 | 2024-01-23 | -15.8 | 27 | 12 |
| 2024-03-09 | 2024-05-14 | -33.4 | 66 | 28 |
| 2024-06-01 | 2024-07-05 | -41.3 | 34 | 26 |
| 2024-07-20 | 2024-08-05 | -34.5 | 16 | 14 |
| 2024-08-22 | 2024-09-06 | -15.6 | 15 | 7 |
| 2024-09-23 | 2024-11-04 | -15.9 | 42 | 21 |
| 2024-12-03 | 2024-12-21 | -28.5 | 18 | 15 |
| 2025-01-02 | 2025-04-08 | -66.0 | 96 | 62 |
| 2025-04-26 | 2025-06-22 | -23.3 | 57 | 27 |
| 2025-07-20 | 2025-08-02 | -20.4 | 13 | 18 |
| 2025-09-12 | 2025-11-22 | -47.5 | 71 | 42 |
| 2025-12-06 | 2025-12-18 | -18.1 | 12 | 9 |
| 2026-01-13 | 2026-02-05 | -40.1 | 23 | 24 |
| 2026-03-16 | 2026-03-27 | -13.7 | 11 | 2 |
| 2026-05-08 | 2026-06-10 | -32.4 | 33 | 18 |

## 2. Data inventory

Full detail (URLs, licence, PIT plan, probe result) is in `data/riskoff/data_inventory.csv`, 40 rows. Summary:

| H | variable | freq | history start | publication lag | vintages? | CI? | rating |
|---|---|---|---|---|---|---|---|
| H1 | hashrate (TH/s) | daily | 2009 | ~1d (day D final after D) | no | unknown | green |
| H1 | difficulty | per 2016 blocks (~14d) | 2009 | 0 (block timestamp exact) | not needed | unknown | green |
| H1 | hashprice (USD/TH/day) | daily | 2009 | ~1d | no | unknown | amber |
| H1 | power-cost proxy: US gas (Henry Hub) | daily | 1997-01-07 | ~1-3d | ALFRED (see H5 note) | likely | amber |
| H1 | power-cost proxy: EU gas (TTF) | daily (Yahoo) / monthly (FRED) | 2017-10-23 (Yahoo); 1992 (monthly) | 0-1d (Yahoo); >=1 month (FRED) | no | unlikely | amber |
| H1 | power-cost proxy: ERCOT wholesale price | 15-min/hourly | 2010s | ~same day | no | unknown | red |
| H1 | PAID: miner economics (hashprice, miner flows, hash ribbons) | daily | 2018+ | ~1d | partial | unknown | nan |
| H2 | Fed balance sheet (WALCL, Wed level) | weekly (Wednesday) | 2002-12-18 | released Thu ~16:30 ET; use from D+2 (Fri) | ALFRED | likely | green |
| H2 | Treasury General Account (WTREGEN) | weekly (Wed avg) | 2002-12-18 | as WALCL | ALFRED | likely | green |
| H2 | Reverse repo (RRPONTSYD) | daily | 2003-02-07 (≈0 until 2021) | same day ~13:00 ET | not needed | likely | green |
| H2 | Net liquidity = WALCL - TGA - RRP | weekly | 2003 (RRP-meaningful 2021+) | max of components | partial | likely | green |
| H2 | Dollar index (broad, DTWEXBGS) | daily (published weekly) | 2006-01-02 | ~1 week (last obs 2026-10-02 on 10-09) | ALFRED | likely | amber |
| H2 | US 10y real yield (DFII10) | daily | 2003-01-02 | ~1d | not needed | likely | green |
| H3 | total stablecoin supply | daily | 2017-11-29 | ~0-1d | NO | likely | amber |
| H3 | USDT supply | daily | 2017-11 | as above | NO | likely | amber |
| H3 | USDC supply | daily | 2018-09-11 | as above | NO | likely | amber |
| H3 | issuer mint/burn events | per event | 2019+ | event time | nan | unknown | nan |
| H4 | VIX close | daily | 1990-01-02 | close 16:15 ET; FRED posts next morning | not needed | likely | green |
| H4 | US high-yield OAS (ICE BofA) | daily | 2023-10-10 ONLY (ICE licence caps FRED at ~3y) | ~1d | nan | likely | red |
| H4 | Baa-10y Treasury spread (BAA10Y), credit proxy | daily | 1986-01-02 | ~1d | nan | likely | green |
| H4 | Nasdaq Composite | daily | 1971-02-05 | ~1d | nan | likely | green |
| H4 | Gold | daily | 2000-08-30 | 0-1d | no | unlikely | amber |
| H4 | USD/JPY | daily | 1999 (ECB) / 1971 (FRED) | ECB ref ~16:00 CET same day; FRED ~1 week | nan | unknown | green |
| H5 | FOMC meeting dates | ~8/yr | 2021 on current page (+ historical pages) | scheduled a year ahead | nan | likely | green |
| H5 | CPI / NFP / PCE release dates | monthly | 1990s | scheduled ex ante | nan | likely | amber |
| H5 | actual values (first print) | monthly | 1990s+ vintages | release 08:30 ET | YES (full vintages) | unknown | amber |
| H5 | consensus / forecast | per release | current week only | nan | NO | nan | red |
| H6 | Economic Policy Uncertainty, daily | daily | 1985-01-01 | assume +1d (file stamped 2026-10-09 23:55 GMT already holds the 9-Oct row) | NO | likely | red |
| H6 | EPU monthly (US) | monthly | 1900 | month-end + 1-4w | NO | likely | red |
| H6 | Geopolitical Risk index (Caldara-Iacoviello) | daily + monthly | 1985 (per source docs; xls not parsed, xlrd absent) | file updated ~weekly (last-modified 2026-10-05): shift 8d | NO | likely | red |
| H7 | funding rates (290 USDT-perps) | 8h | BTCUSDT 2019-09-08 | settled at 00/08/16 UTC; use a settlement for D only if <= D close | not needed | no | amber |
| H7 | funding, alternative venues | 8h/1h | varies | as above | not needed | unknown | amber |
| H7 | open interest (5-min, daily files) | 5-min | 2020-09 (404 for 2020-06 and earlier) | ~1d (file for 10-09 present on 10-10) | not needed | yes | green |
| H7 | open interest, all-venue / deeper | 1d/4h | 2020-02-27; DB ends 2026-06-15 | ~1d | NO | no | amber |
| H7 | liquidations | daily | 2019-09-08 (DB) | ~1d | NO | no | red |
| H7 | basis (perp premium index) | 1d (also finer) | 2020-01 | daily file ~1d; monthly zip after month end | not needed | yes | green |
| H7 | long/short, taker ratios | 5-min | 2020-09 | ~1d | not needed | yes | green |
| H8 | breadth (share of 290 symbols above 50d/200d MA, advance/decline) | daily | 2019-09-08 (BTC); >=5 symbols from 2020-01-10 | bar of D complete at D+1 00:00 UTC | not needed | yes | green |
| H8 | BTC dominance | daily | free tier: 365 days | ~0d | NO | unknown | amber |
| H8 | ETH/BTC | daily | 2019-11 | 0 | not needed | yes | green |

`CI?` meanings: yes = already used by a CI job here; no = documented blocked here; likely = static gov/CDN host, untested; unknown = API/WAF host, untested; unlikely = host known to block datacenter IPs, untested.

## 3. Feasibility by hypothesis (R/A/G)

| H | rating | why | judged by |
|---|---|---|---|
| H1 miner margin | **AMBER** | hashrate, difficulty, hashprice are free and un-restated, history to 2009. Power cost has no usable free daily proxy (ERCOT is behind registration, gas is a weak stand-in). Hashprice embeds BTC price, so it needs a price-neutral form. | event study; slow-moving, effective N is a handful of regimes |
| H2 liquidity | **GREEN** (data) | all FRED, free, real vintages exist. DXY has a 1-week H.10 lag. RRP is ~0 before 2021. | event study + coarse regime split; weekly and highly autocorrelated |
| H3 stablecoins | **AMBER** | free and reachable, but DefiLlama restates history and has no vintages, so a backtest is not point-in-time. | event study until a forward snapshot log exists |
| H4 cross-asset | **GREEN, with one RED component** | VIX, Nasdaq, USDJPY, BAA10Y are daily, free, unrevised. HY OAS exists only from 2023-10-10; the substitutes are different variables. Gold depends on Yahoo. | Stage 1 univariate test is possible for VIX/Nasdaq/USDJPY/BAA10Y |
| H5 scheduled events | **AMBER** (dates) / **RED** (consensus) | FOMC and release dates are ex ante and free. First-print actuals need ALFRED. No free consensus history. | event study only; no surprise claims |
| H6 policy / geopolitical | **RED** | EPU and GPR are revised retroactively and have no vintages; GPR lags up to ~1 week; monthly EPU is too coarse. | contemporaneous event study only, with an explicit caveat |
| H7 derivatives crowding | **AMBER** | OI, long/short, taker and basis are free from binance.vision from 2020-09 and reachable in CI. Funding is local-only and our files are stale (below). Liquidations have no free source. | Stage 1 possible for OI/basis/funding on 2020-09+; liquidations frozen at 2026-06-16 |
| H8 crypto internals | **GREEN** | own data, reachable, no lag. Breadth is survivor-biased. True BTC dominance is paid beyond 1 year; use a volume-share proxy. | Stage 1 univariate |

## 4. Is the episode count enough?

I do not have a formal power calculation. Rule of thumb used here: a univariate test with a block bootstrap wants >=30 independent episodes, and a subperiod split wants roughly >=10 in each half. Day-level N (about 2,500 labeled days) is not the sample size, because flags come in runs and the underlying series are autocorrelated.

- **BTC, full history:** 41 episodes at a 15-day gap, 21 at 30 days, 16 at 45 days. Borderline for one univariate test; **too thin for subperiod stability** at the 30-day gap and above, especially since 2021-22 holds 16 of the 41 (9 + 7 by start year), so any "stable across periods" result will be driven by that bear market.
- **BTC with size >=20%:** 13 episodes. Too few to test anything.
- **Since 2023-10-10 (where HY OAS starts):** 14 BTC episodes (3 of >=20%). **H4's HY spread cannot be evaluated** on this window.
- **Basket:** 36 at 15 days but 13 at 30 days; it spends 30% of days flagged, so a "risk-off" signal would be on a third of the time. A 10%/10d basket flag is close to "alts are in a drawdown", not an event.
- **Slow variables (H1, H2, H3, H6):** they change over months, so their effective sample is the number of regime changes (single digits to low tens), not the number of episodes. **These can only be judged by event study.**
- **H5:** event study by construction (about 8 FOMC meetings a year, ~12 CPI and NFP releases a year each); the question is the window around the date, not prediction of episodes.
- **Candidates for Stage 1 univariate tests:** H4 (VIX, Nasdaq, USDJPY, BAA10Y), H7 (OI, basis, funding, long/short), H8 (breadth, ETH/BTC), and the H2 weekly series with caveats. Report all episode-count sensitivities with them.

## 5. What blocks point-in-time testing

1. **No vintages** for DefiLlama stablecoins, EPU, GPR, CoinGecko dominance. History we download today is not what was known at D. Fix: start append-only daily snapshot jobs now (new workflow, needs your OK); strict PIT testing is then possible only for data collected from today on.
2. **ALFRED/FRED vintages need a free API key.** The keyless ALFRED host succeeded once and then refused three requests. Without vintages, H2 uses latest-revised values (small effect for WALCL/TGA, but not zero).
3. **HY OAS: 3 years only** (2023-10-10 -> ), ICE licence on FRED. Substitutes are different variables.
4. **No free consensus history** (H5). Surprise tests are not possible; do not claim them.
5. **No free liquidation history.** We hold Coinglass data to 2026-06-16 only; Binance liquidation endpoints are gone (404).
6. **Funding is stale and local-only.** In `data/futures_universe/funding_rates`, the median last print is 2026-08-28 and the oldest is 2026-06-11 (BTCUSDT 2019-09-08 start). So the repaired funding series does not cover the Sep-Oct 2026 window and cannot be refreshed in CI (HTTP 451).
7. **Publication lags that a naive join would ignore:** DXY and USDJPY from FRED (H.10) lag about a week; GPR about a week; H.4.1 is Thursday for a Wednesday level.
8. **Survivorship** in breadth and the basket (current 290 symbols only); the basket starts 2020-01-10, not 2019.
9. **Host reliability:** Yahoo (gold, TTF) is unofficial and is likely to be blocked in CI; BLS returned 403 and ERCOT needs registration even locally. Not yet fetched, by design.
10. **Paid items** (not fetched, prices not checked): Coinglass plan (we hold a key; gain: liquidations and all-venue OI going forward), consensus data (Bloomberg / Refinitiv / TradingEconomics / Econoday; gain: true surprise), miner-economics and on-chain suites (Glassnode / CryptoQuant / Hashrate Index; gain: exchange-quoted hashprice, miner flows, power cost), long-history BTC dominance (CoinGecko paid). I did not look up current prices, so figures need checking before any purchase decision.

## 6. Decisions needed from you before Stage 1

- **Merge gap:** keep 15 days (41 BTC episodes) or move to 30 days (21)? The gap changes the answer to "is there enough data" more than any variable does; I kept 15 as given.
- **Free FRED API key** (for vintages): OK to register?
- **CI reachability probe:** a one-off workflow that only does HEAD/GET on the listed hosts and writes a log, so ratings stop being inference. It touches `.github/workflows`, so I did not do it.
- **Forward snapshot job** for stablecoins, EPU and GPR: OK to add after Stage 1?
