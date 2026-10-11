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

**Opportunity-cost shadow** (`tools/s2_opportunity_cost_shadow.py`): the real S2 `run_one_day` replayed day by day from 2026-06-26 with
the cross-system duplicate set rebuilt per date from S2's own log. The cap (10) was genuinely binding: 165 `cap 10 reached` skips logged, concentrated in 06-27..07-06 and 07-17..08-12.

| replay | trades | non-flat | hit rate | avg R | P&L |
|---|---|---|---|---|---|
| old engine (frozen-RSI bug, no gate) | 44 | 24 | 27.3% | 0.200 | +207.2 |
| tradability gate only, freed slots refilled by the next valid signals | 34 | 34 | 55.9% | 0.451 | +336.9 |
| current engine (RSI fix + gate) | 34 | 34 | 55.9% | 0.451 | +336.9 |

The gate removed 28 entries (dead-symbol re-entries and the signals they displaced) and 13 lower-ranked valid signals took the slots
(e.g. NEO/ONE/SKL/SPELL 06-26, BAT/MOVE 08-11, GRT/HIVE 08-12, KAIA/MTL/POLYX 08-16). **Difference: +129.7 USDT, about +1.3% of S2's
10,000 start, from refilled slots.** The RSI-freeze fix alone (committed earlier today) already captures all of it; the tradability gate
adds only SCRTUSDT of 2026-08-26 on top once the RSI fix is in.

**How far to trust the replay.** It does not reproduce the real ledger entry by entry (11 of 36 entries match; 24 non-flat trades each side;
non-flat hit rate 50% vs 54%, avg R 0.37 vs 0.16). Causes I know: the real engine went through remediation/back-dating in June and a July
data-gap period, my start equity is approximate, and regime-weight history was rebuilt today. Read the old-vs-gated difference as an
order-of-magnitude estimate, not an exact P&L restatement.

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

## 10. Open items needing your decision

1. **Apply the state diff?** Set `apply_state_fixes` true in `data/tradability_config.json`; the engines then perform exactly section 2 on
   their next run. Reversible only from git history.
2. **Regime inputs include dead symbols.** The regime breadth counts the 13 settled contracts as "below EMA200" (2026-10-10: 49.66% with
   them, 51.62% without), and the funding average includes their last frozen rates (-6.3e-6 with, +8.6e-6 without). Labels are unaffected
   today (MIXED/LOW either way) but a threshold crossing could differ. Not changed because it moves allocation weights.
3. **Which host runs the data job** (still open) - until then snapshots, the OI log and the exchange-status cache refresh only when someone
   runs `tools/refresh_funding_local.py` by hand, and the cache goes stale (WARNING after 3 days).
4. Push: nothing has been pushed. The CI engines will gate new entries as soon as this lands on the remote.

## 11. Concurrency note

Another session was editing the same repo; `docs/SESSION_OWNERSHIP.md` lists the split. I committed only explicit paths. Their commits
(`253b18bc1`, `39b0bbc30`) are untouched; the contested engine files now contain both lines of work and the full test suite passes.
