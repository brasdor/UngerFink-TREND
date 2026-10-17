# Exchange-faithful tradability, flat-data guards and S2 opportunity cost (2026-10-11)

Status: code committed locally, **nothing applied to live state, nothing pushed**. New-entry gating is on; voiding/settling of
EXISTING positions is off (`data/tradability_config.json` `apply_state_fixes: false`) until you confirm. Exact state diff:
`docs/TRADABILITY_STATE_DIFF.md`.

## 1. The rule (engines/tradability.py)

- **Futures engines (S2, S5, S6, S7, S8, C12, C19):** a symbol is tradable on bar date D iff `onboard_date <= D < delivery_date`, and for a
  live decision (bar date within 3 days of the cache date) its exchange status is `TRADING`. The delivery date itself is not tradable.
- **Spot engines (S1, S3 trade the spot universe):** spot symbols have no delivery date; only `status == TRADING` applies. This matters:
  the futures cache would wrongly block S3's `PEPE/USDT` (futures name `1000PEPEUSDT`) and S1's `SC/USDT`. With spot status the rule blocks
  exactly `ARDR`, `ICX` (S1) and `TON` (S3), all spot `BREAK`.
- A position in a contract that reaches delivery closes at the **settlement price on the delivery date**, `exit_reason = exchange_settlement`.
  Settlement price = close of the delivery-date bar (last bar with volume; every later bar in our files is a flat copy of it).
- An entry dated on/after delivery is **voided**: removed from open state, annotated `ENTRY_VOIDED` in the engine log and appended to
  `data/tradability_ledger.csv` (append-only). A void books no P&L and is not a trade (`closed_trade_count` unchanged).
- Source: `data/exchange_status_cache.json` (now carries `onboard_date`, `contract_type`; covers all 290 universe symbols; 13 are `SETTLING`).
  Fail-closed: unknown symbol -> not tradable. Unreadable cache -> gate off with a CRITICAL finding (so losing the cache cannot halt every
  engine silently). The funding-host job now refreshes the cache daily.

Where it hooks in: S1/S2/S3/S8 entry loops and exit loops; S5 basket ranking (`compute_basket`) plus a state step; S6/S7 pending-fill, signal
and a pre-pass over positions; C12/C19 proposals plus a pre-pass. Entry/exit rules are otherwise unchanged.

## 2. Exact state diff (dry run)

7 open positions are affected, all **voids** (entries after delivery), none settlements; equity is unchanged for every engine:

| engine | positions | delivery |
|---|---|---|
| S2 | COSUSDT, DUSDT, HIGHUSDT, MBOXUSDT (entered 2026-09-29), SCRTUSDT (2026-10-05) | 2026-06-11 (x4), 2026-08-26 |
| S5 | STGUSDT short (2026-10-09; its basket slot stays as cash, 9 shorts remain) | 2026-09-24 |
| C19 | COSUSDT long (2026-10-10) | 2026-06-11 |

Full JSON of each removed position is in `docs/TRADABILITY_STATE_DIFF.md`. I smoke-tested the real S2/S8/S1/S3/S6/S7 states in memory with
the flag on: S2 goes 5 -> 0 open with five `ENTRY_VOIDED`, the others are unchanged. Open positions in non-tradable symbols keep alerting
(CRITICAL `exchange_status_not_trading`) until you apply this.

## 3. Historical entries (read-only scan, `data/tradability/historical_scan.csv`)

699 trade/position records across 9 engines. Entry on/after delivery (would be voided) / held through delivery (would settle earlier):

| engine | records | void | spans delivery | P&L in the voided closed trades |
|---|---|---|---|---|
| S2 | 41 | 17 (12 closed + 5 open) | 1 (ICXUSDT) | 0.00 |
| S5 | 212 basket rows | 6 | 8 | not tracked per position |
| S8 | 27 | 2 (DUSDT, HIGHUSDT, 2026-07-10) | 0 | 0.00 |
| C19 | 71 | 2 (ACXUSDT closed, COSUSDT open) | 0 | **-6.07 USDT** (ACX, entered 8 days after delivery, closed by rebalance) |
| S1, S3, S6, S7, C12 | - | 0 | 0 | - |

S5 voids: DUSDT/HIGHUSDT shorts 2026-06-26, HFTUSDT long 2026-08-14, STGUSDT shorts 2026-09-25 and 2026-10-09, VANRYUSDT short 2026-08-14.
S5 baskets are rebuilt from the git history of its state file (it keeps no trade ledger).

## 4. S2 corrected statistics and opportunity cost

Corrected from the real ledger (`engine_equity_curve.csv`, 36 closed trades): voiding the 12 closed post-delivery trades (all zero-R flat
fills) gives

| | trades | hit rate | avg R | P&L |
|---|---|---|---|---|
| as logged | 36 | 36.1% | 0.109 | +98.67 |
| corrected | **24** | **54.2%** | **0.163** | +98.67 (voids carried 0) |

The 5 open positions are also voided (not counted above). ICXUSDT (entered 08-20, closed 09-09) would settle on 08-26 instead; same frozen
price, so the same -7.999 USDT, only the slot frees 14 days earlier.

**Opportunity-cost shadow** (`tools/s2_opportunity_cost_shadow.py`, rerun 2026-10-17 with a corrected log parser, see below): the real S2
`run_one_day` replayed day by day from 2026-06-26 with the cross-system duplicate set rebuilt per date from S2's own log. The cap (10) was
genuinely binding: 183 `cap 10 reached` skips on 39 days.

| replay | trades | non-flat | hit rate | avg R | P&L |
|---|---|---|---|---|---|
| old engine (frozen-RSI bug, no gate) | 43 | 23 | 25.6% | 0.193 | +188.1 |
| tradability gate only, freed slots refilled by the next valid signals | 33 | 33 | 57.6% | 0.481 | +339.4 |
| current engine (RSI fix + gate) | 34 | 34 | 55.9% | 0.451 | +336.9 |

The gate removed 28 entries (dead-symbol re-entries and the signals they displaced) and 13 lower-ranked valid signals took the slots
(NEO/SKL/SPELL/S on 06-26, CELR/MOVE 07-27, BAT/MOVE 08-11, GRT/HIVE 08-12, KAIA/MTL/POLYX 08-16). **Difference: +151.3 USDT, about +1.5% of S2's
10,000 start** (the first run said +129.7 with a defective log parser). The RSI-freeze fix alone captures essentially all of it; the
tradability gate adds only SCRTUSDT of 2026-08-26 once the RSI fix is in.

### Why the replay matches only 11 of 36 real entries (diagnosed 2026-10-17)

`tools/s2_replay_leak_diagnosis.py`, output `data/tradability/s2_replay_leak_diagnosis.csv`. What it is NOT: the data. All 11 real ENTRY
rows since 07-12 reproduce from the stored OHLCV exactly (price identical, RSI within 0.05), and with the pre-fix RSI all 25 real entries
that the replay missed do signal on the stored data. What leaks in:

1. **The real ledger is partly a reconstruction, not a live record.** Commits `6c3219895` ("recover 11 interleaved trades") and `6a4f7c8f4`
   ("S2: +8 closed trades recovered from a 07-16/08-06 gap", replayed 2026-08-23) rebuilt S2's trade history by date-sliced re-runs. The
   log shows the same symbol both ENTERED and "cap 10 reached" on the same day (8 cases on 07-15, 07-17, 07-30): the day was run twice
   against different states. A clean forward replay cannot reproduce that path.
2. **Start state.** The replay begins 2026-06-26 with zero positions. The real engine was already at the cap that day (it logged
   `cap 10 reached` for DIAUSDT and KSMUSDT) because it still carried the spot-era positions that the 06-25 remediation closed after the fact.
3. **Cap occupancy cascades.** Once one entry differs, every later cap decision differs: 19 of the 25 real-only entries fall on days the
   replay is at the cap, and 15 are symbols the replay already holds; the 20-day dead-symbol batches also run on a different phase
   (real 07-30/08-20/09-09/09-29, replay 06-26/07-17/08-06/08-26/09-15/10-05).
4. **Cross-system occupancy** is rebuilt from S2's own skip log, which only lists symbols S2 did not already hold, so it is approximate.
5. **My own first-version bug (fixed):** `pd.read_csv(on_bad_lines="skip")` silently dropped 490 of 6,311 rows of the ragged
   `daily_log.csv`, including 43 of 55 ENTRY rows and every cross-system skip on six dead-symbol entry dates. Fixing it did not change
   the match (11 of 36 both times), which is how I know it was not the main cause. The same drop affected S3's log in the historical scan
   (now fixed) and any consumer using that option.

Conclusion: read the shadow as a controlled comparison of the same replay with and without the gate (+151 USDT), not as a restatement of
the real ledger.

## 5. Flat-data behaviour per engine (zero-range input)

| engine | what the indicator returned on flat input | guard |
|---|---|---|
| S1 Donchian | breakout needs close > prior high: impossible on constant input | immune; validity check + `emit` added |
| S2 RSI(14) | **frozen at its pre-flat value forever** (COSUSDT stuck at 16.067355 for 122+ days = permanent "oversold") | RSI -> NaN when the trailing 15 closes are identical; ATR 0 blocks the signal |
| S3 consec-down | strict `<` never counts a flat bar: 0 | immune; check added |
| S5 momentum | trailing return exactly 0.0, ranks inside baskets | zero-range lookback excluded from ranking |
| S6 vol-compression | ATR -> 0 after 14 bars; compression counter could run during the transition | **new:** ATR NaN on zero-range window, counter resets, `flat_window` refused in `check_signal`, `emit` |
| S7 EMA cross | EMAs converge; a cross could fire in the first 14 flat bars | **new:** same guard |
| S8 RSI + funding | same formula as S2 | NaN fix on top of its existing 5-bar volume/range filter |
| C12 | trailing return 0.0 | excluded from ranking |
| C19 | proximity = close/high = 1.0 exactly, the metric's maximum, guaranteeing a top-decile pick | excluded from ranking |

Tests: 56 pass (`python -m unittest discover -s tests/health`), including flat-series tests for every engine (new S6/S7 tests), the
tradability rule by date/venue, void/settle/inert-flag behaviour on S2 and the S5 basket gate. Flat-window `emit()` now fires only for
symbols the exchange still lists as tradable (settled symbols never reach the signal code), so it stays a real-anomaly signal.

## 6. Suppression list replaced; staleness alerts

`data/halted_symbols_suppression.json`, `check_halted_symbols.py` and `monthly_halted_symbol_check.yml` are deleted. Staleness, funding
staleness, flat-price and exchange-status checks now exempt any symbol the cache says is not tradable. Checked against today's data:
OHLCV staleness OK (13 futures + 3 spot symbols exempt), funding staleness 0 findings (277 tradable symbols, max age 0.17 d), flat-price 0;
the only remaining CRITICALs are `exchange_status_not_trading` for the 7 voidable positions above, which disappear when you apply the diff.
A real stale tradable symbol still alerts (unchanged threshold 3 days). Note one thing the old list never covered: spot ICX is now `BREAK`
and is exempt too.

## 7. Fail-closed stale inputs (done in commit 253b18bc1 by the parallel session)

I did not redo this. Verified: funding for all 277 tradable symbols is current (max age 0.17 d); the 13 stale files are exactly the settled
contracts; the rebuilt regime history (43 rows) changed no trend/funding label. The S6/S7/S8 funding and OHLCV gates and the regime
funding fallback are in HEAD.

## 8. Data job (funding host)

Committed: `tools/refresh_funding_local.py` now also runs (each step isolated; a failure never blocks the funding commit) the OI log,
`tools/forward_snapshots.py` and the exchange-status refresh, and commits `data/riskoff/forward/` and the cache.

| snapshot | running? | where | note |
|---|---|---|---|
| stablecoin supply (total, USDT, USDC) | seeded 2026-10-11, **not scheduled anywhere yet** | `data/riskoff/forward/stablecoins/stablecoins_<fetchdate>.csv` (last 45 rows per series) | DefiLlama restates history, so the dated copy is the only vintage |
| EPU daily | seeded | `forward/epu/epu_daily_<fetchdate>.csv` (last 150 rows + file sha256) and `epu_full_<YYYY-MM>.csv.gz` monthly | |
| GPR daily | seeded | `forward/gpr/gpr_daily_<fetchdate>.csv` (last 150 rows) | needs `xlrd` (`requirements-datahost.txt`); without it only a hash row and a weekly raw copy are kept |
| BTC dominance | seeded | `forward/dominance/btc_dominance.csv` (one appended row/day) | CoinGecko `/global`, current value only |
| BTCUSDT open interest 5m | seeded | `forward/oi_btcusdt_5m.csv` | from the earlier task |

Everything seeded today ran from this laptop once, by hand. **None of it runs on a schedule: no host has been chosen**, and nothing was
registered on this laptop. Health check `forward_snapshots_fresh` (replaces the retired suppression-list check) raises CRITICAL if any
source, the OI log, is older than 26 h; B's `funding_refresh_job_alive` covers the funding commits themselves. Setup for the host
(self-hosted runner and cron+deploy key, hardening) is `docs/FUNDING_HOST_SETUP.md`; the runner workflow runs the same script, so the new
steps come for free. For the cron option the host needs `pip install -r requirements.txt -r requirements-datahost.txt`.

## 9. Tradability by date for the multi-year replay

`engines/tradability.py` is the single function for it: `is_tradable(symbol, date, venue)` and `untradable_reason(...)`. Onboard dates
come from the cache for every currently listed symbol. Limits to know before the 2022-to-today replay: the cache holds only currently
listed and `SETTLING` contracts, so symbols delisted and purged from `exchangeInfo` before today are "unknown" (fail-closed, excluded) and
the 290-symbol universe is survivor-biased by construction; spot symbols have no historical status, so the spot engines' replay can only
apply today's status.

## 10. Stale exchange-status cache: behaviour and policy (added 2026-10-17)

Policy implemented in `engines/tradability.py` (`CACHE_FRESH_DAYS=3`, `CACHE_CRITICAL_DAYS=7`, `CACHE_EXPIRED_DAYS=30`): **the cache only refines
the date rule, so no cache age can halt the engines.** All nine systems read the same module (S1/S3 through the spot entries, the rest through
the futures entries), so the behaviour is identical for every engine; only the entry gate and the settle/void pre-pass differ in where they sit.

| cache age | rule used | emitted | engines |
|---|---|---|---|
| <= 3 d (fresh) | full: dates + status TRADING inside the live window; unknown symbol blocked | nothing (any old finding is resolved) | all gate normally |
| 3-7 d (stale) | date-based fallback: onboard/delivery dates only, status ignored, unknown symbol allowed | WARNING `shared/tradability:exchange_status_cache_stale` (from the engines) and `exchange_status_cache_stale` (health runner) | no engine blocks anything new; SETTLING symbols with a known delivery date are still blocked once that date passes |
| 7-30 d | same fallback | same finding escalated to CRITICAL | same |
| > 30 d | same fallback | CRITICAL, message says EXPIRED | same |
| missing / unreadable | exchange gate off | CRITICAL `exchange_status_cache_unusable` | all engines keep trading; only the flat-feed backstop below remains |

Tested for 1, 3, 4, 7, 8, 30, 31 and 400 days: the tradable set stays non-empty at every age, a delivered symbol is still voided from its
known delivery date even with a 40-day-old cache, and S2's entry scan is unchanged at 30 days (`TestCacheAgePolicy`).

**Symbol missing from the cache (a new listing).** Fresh cache: blocked by every futures engine as `unknown_symbol` (the engines print a count,
no finding), until the next refresh, at most 3 days. Stale cache: allowed, because the cache may simply pre-date the listing. S5/C12/C19 and
the 4H engines need history windows anyway.

**Symbol that enters SETTLING after the last refresh.** The cache cannot know. Entries are possible until one of two backstops trips:
(1) new `flat_zero_volume_feed` rule in the tradability module itself: the last 3 daily bars on or before the decision date all have zero
volume and high == low (what Binance serves after settlement), at any cache age and even with no cache, point-in-time safe; this blocks
every engine's entries and rankings; (2) the indicator-level flat guards (S2/S8 RSI, S5/C12/C19 rankings, S6/S7 ATR window) once their full
window is flat. The exposure window for a new entry is therefore at most the first 3 flat bars after delivery. An already open position
is only voided/settled once the refreshed cache shows the delivery date; its P&L is unaffected because the price is frozen at settlement.
I assumed Binance puts the delivery date into `exchangeInfo` when the delisting is announced; I have not verified that, and if it does not,
the backstop is the only protection before the refresh.

**Health check on cache age.** `check_exchange_status` (tools/health/run_checks.py) now uses the same bands: WARNING at 3-7 d, CRITICAL
beyond 7 d (it was a 35-day WARNING). Beside it: `forward_snapshots_fresh` (any snapshot or the open-interest log older than 26 h) and the
parallel session's `funding_refresh_job_alive`.

## 11. Regime inputs: exclusion switch and replay (added 2026-10-17)

Switch: `regime_exclude_nontradable` in `data/tradability_config.json`, **default false**; when true `engines/regime_daily.py` leaves
non-tradable symbols out of the breadth and funding averages. Replay both ways (`tools/regime_exclusion_replay.py`, 2022-01-01 to
2026-10-10, 1,744 days, survivor-biased because the cache only holds currently listed or settling contracts):

- Validation: the "include" replay reproduces the committed `regime_history.csv` on all 163 overlapping days, trend and funding labels 100%,
  breadth within 0.18 points on average.
- Excluding changes **11 trend-label days, all between 2026-09-27 and 2026-10-07 (MIXED -> BULL)**, and no funding-label day. 2022-2025 are
  identical (few contracts had settled then). BULL occupancy 12.3% -> 13.0%; trend switches 49 -> 51.
- Cause: the exclusion lifts smoothed breadth by about 2 points (57.9 vs 60.3 on 09-27), across the 58 BULL entry threshold, with BTC
  above its EMA200. During those 11 days BULL weights (S1 0.25, S5 0.25 vs 0.125 each) would have applied instead of MIXED.

That is the whole difference, and it sits in the most recent two weeks, so it is a real but small behavioural change. Left off, as agreed.

## 12. Open items

1. **Apply the state diff:** decided yes, after the dry run, on a tagged commit. Dry run done (`docs/TRADABILITY_DRYRUN.md`); tag
   `pre-tradability-apply` is created locally on the commit that carries this report; the flag is still false; nothing pushed.
2. **Regime inputs:** switch built (section 11); the choice is yours after reading the replay.
3. **Host:** until you pick one, run `python tools/refresh_funding_local.py --no-push` by hand each day. I did that on 2026-10-17 (funding was
   6 days old, the exchange cache 6.1 days, snapshots 5.9 days; the CI funding gates had blocked every S6/S7/S8 entry since 10-14). Your
   own notes (project memory, Blueprint doc) list a Hetzner Nuremberg CPX21 as the planned dev box and an AWS Tokyo t3.small for
   production; I cannot see either from here, so I do not know whether they exist or are on.
4. **Push:** not done. The stale-cache behaviour is now tested (section 10), which was the stated condition.

## 13. Concurrency note

Another session was editing the same repo; `docs/SESSION_OWNERSHIP.md` lists the split. I committed only explicit paths. Their commits
(`253b18bc1`, `39b0bbc30`) are untouched; the contested engine files now contain both lines of work and the full test suite passes.
