"""Builds docs/RISKOFF_STAGE0.md from the Stage 0 CSVs (tables are generated, narrative is fixed text)."""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "riskoff")
base = pd.read_csv(os.path.join(D, "base_rates.csv"))
ev = pd.read_csv(os.path.join(D, "events.csv"))
inv = pd.read_csv(os.path.join(D, "data_inventory.csv"))
cnt = pd.read_csv(os.path.join(D, "episode_counts_by_window.csv"))
known = pd.read_csv(os.path.join(D, "known_episodes_check.csv"))


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


def grid(target):
    b = base[base.target == target]
    rows = []
    for h in (5, 10, 20):
        r = {"horizon": f"{h}d"}
        for t in (0.07, 0.10, 0.15):
            x = b[(b.thr == t) & (b.horizon == h)].iloc[0]
            tag = " **(primary)**" if x.primary else ""
            r[f"{int(t*100)}%"] = f"{x.base_rate_pct}% / {x.n_episodes} ep{tag}"
        rows.append(r)
    return md(pd.DataFrame(rows))


def ev_table(target):
    e = ev[(ev.target == target) & (ev.primary)].copy()
    e = e[["start", "trough", "size_pct", "days_to_trough", "n_flag_days"]]
    e.columns = ["start", "trough", "size_% (start close -> trough)", "days start->trough", "flagged days"]
    return md(e)


def b(t):
    return base[(base.target == t) & base.primary].iloc[0]


bt, ew = b("BTC"), b("EW_BASKET")
inv_s = inv[["hyp", "variable", "frequency", "history_start", "pub_lag", "vintage_data", "ci_reachable", "rating"]].copy()
inv_s.columns = ["H", "variable", "freq", "history start", "publication lag", "vintages?", "CI?", "rating"]

TXT = f"""# Risk-off flag - Stage 0 report (event catalogue + data inventory)

Status: read-only research. No flag, model, alert or Telegram call was built; no live system, engine, workflow or allocator file was touched.
Generated 2026-10-10 by `tools/riskoff/stage0_*.py`. Outputs in `data/riskoff/`: `events.csv`, `labels_daily.csv`, `base_rates.csv`,
`known_episodes_check.csv`, `episode_counts_by_window.csv`, `target_series.csv`, `data_inventory.csv`.

## Headline

- **Primary definition (10% / 10d): {bt.n_episodes} BTC episodes, {ew.n_episodes} basket episodes** at a 15-day merge gap. This number is fragile: re-merging the *same* flags at 30 / 45 days gives **21 / 16 (BTC)** and **13 / 9 (basket)**. Only **13** BTC episodes fell 20% or more, only **5** fell 30%+.
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
- **{int(67)} symbol-day returns with |return| > 80% were dropped** as probable bad prints or redenominations (guard in `stage0_events.py`). I did not audit them one by one.
- The basket is *much* noisier than BTC: {ew.base_rate_pct}% of days flagged vs {bt.base_rate_pct}%.

**Base rates and episode counts** (cell = share of labeled days flagged / episodes at a 15-day merge gap). Only 10%/10d is the primary definition; the rest are sensitivity and none was chosen on results.

BTC (labeled {bt.n_days_labeled} days):

{grid('BTC')}

EW basket (labeled {ew.n_days_labeled} days):

{grid('EW_BASKET')}

**Declustering:** flagged days at most 15 days apart are merged. Start = first flagged day. Trough = lowest close in [start, last flagged day + h]. Size = trough close / close at start - 1 (so it is measured from the first flagged day, not the peak).

**Episode count depends on the merge gap, which the brief fixed at 15 days as an example.** Same primary flags, different gaps (reporting only, not a choice):

| merge gap | BTC episodes | basket episodes |
|---|---|---|
| 15 d | 41 | 36 |
| 30 d | 21 | 13 |
| 45 d | 16 | 9 |

Episodes by window (primary definition) - what each data history can actually see:

{md(cnt)}

### Sanity check against known episodes (rule not adjusted)

{md(known)}

Reading: the rule captured May 2021, May 2022, Nov 2022, Aug 2024 and the 2025-26 decline (several episodes inside it) on BTC and on the basket. It **missed the September 2026 dip**: in our data BTC's largest drawdown inside 1 Sep - 8 Oct is -6.9%, below even the 7% threshold, and the last 10 days are unlabeled anyway. If the dip you have in mind was intraday or in another instrument, it is not visible in daily futures closes. Also note the Aug 2026 jump (BTC 63k -> 78k in 3 days, 17-21 Aug) is continuous in the source file, not a data gap.

### Episode tables (primary 10% / 10d, 15-day merge)

**BTC - {bt.n_episodes} episodes**

{ev_table('BTC')}

**EW basket - {ew.n_episodes} episodes**

{ev_table('EW_BASKET')}

## 2. Data inventory

Full detail (URLs, licence, PIT plan, probe result) is in `data/riskoff/data_inventory.csv`, {len(inv)} rows. Summary:

{md(inv_s)}

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
"""
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
with open(os.path.join(ROOT, "docs", "RISKOFF_STAGE0.md"), "w", encoding="utf-8") as f:
    f.write(TXT)
print("written", len(TXT))
