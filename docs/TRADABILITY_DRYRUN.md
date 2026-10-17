# Tradability state diff: dry run on a copy (2026-10-17)

Run by `tools/tradability_dryrun_sandbox.py`. The repo was copied to a temp directory; nothing in the real tree changed and
`apply_state_fixes` stays **false** there. In the copy every engine was run exactly as the workflow runs it
(`--date 2026-10-16 --no-download`, `last_run_date` wound back one day) twice from the same pristine state: variant A with the flag
off, variant B with it on. Historical voids were then restated from `data/tradability/historical_scan.csv`.

## Equity, trades and open positions, every system

| system | equity before | equity, flag OFF (A) | equity, flag ON (B) | equity after history restated | d equity | closed trades before | after | voided closed | open before | open after |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10215.9 | 10215.9 | 10215.9 | 10215.9 | 0.0 | 12 | 12 | 0 | 4 | 4 |
| S2 | 10151.74 | 10151.74 | 10151.74 | 10151.74 | 0.0 | 45 | 33 | 12 | 9 | 4 |
| S3 | 9940.09 | 9940.09 | 9940.09 | 9940.09 | 0.0 | 2 | 2 | 0 | 6 | 6 |
| S5 | 10728.82 | 10728.82 | 10728.82 | 10728.82 | -0.0 | n/a | n/a (no ledger) | 5 | 30 | 29 |
| S6 | 10166.42 | 10166.42 | 10166.42 | 10166.42 | 0.0 | 102 | 102 | 0 | 1 | 1 |
| S7 | 9666.64 | 9666.64 | 9666.64 | 9666.64 | -0.0 | 160 | 160 | 0 | 6 | 6 |
| S8 | 10396.81 | 10396.81 | 10396.81 | 10396.81 | -0.0 | 27 | 25 | 2 | 0 | 0 |
| C12 | 63948.38 | 63948.38 | 63948.38 | 63948.38 | 0.0 | 46 | 46 | 0 | 20 | 20 |
| C19 | 66483.15 | 66483.15 | 66483.15 | 66489.22 | 6.07 | 51 | 50 | 1 | 20 | 19 |

Reading it:
- **Equity does not move in A vs B for any system**, including S8 and C19: voiding an open position books no P&L.
- **Open positions that change (B vs A):**
- **S2**: removed ['COSUSDT', 'DUSDT', 'HIGHUSDT', 'MBOXUSDT', 'SCRTUSDT']; added []
- **S5**: removed ['STGUSDT']; added []
- **C19**: removed ['COSUSDT']; added []
  Everything else (S1, S3, S6, S7, S8, C12) is untouched; no open position is added anywhere.
- **Historical voids move equity only where a voided closed trade carried P&L.** That is C19 alone: ACXUSDT (entered 2026-08-15, seven
  days after its 2026-08-07 delivery, closed 2026-09-12 by rebalance) lost 6.07 USDT, so removing it raises C19 closed equity by
  **+6.07** (66,483.15 -> 66,489.22) and its curve peak by the same amount; max drawdown is 0.0 before and after. The other voided closed
  trades (S2: 12, S8: 2) were zero-R flat fills with 0.00 P&L, so only their counts change. S5 has no trade ledger (its baskets are
  rebuilt from git history), so its 5 voided closed basket rows cannot be restated in P&L; the effect is limited to turnover cost on a
  rebalance and is not computed.
- `closed trades before` is the engine state's `closed_trade_count`; S2's 45 includes 9 older remediation exits. "After" subtracts the
  voided closed trades (S2 -12, S8 -2, C19 -1).

## Exact changes the flag performs on open positions

Ledger written by the engines in variant B (`data/tradability_ledger.csv` in the copy):

```
logged_utc,engine,kind,symbol,entry_date,run_date,price,detail
2026-10-17T12:15:40+00:00,S2,void,COSUSDT,2026-09-29,2026-10-16,,on_or_after_delivery; delivery 2026-06-11; entry_price 0.000485
2026-10-17T12:15:40+00:00,S2,void,DUSDT,2026-09-29,2026-10-16,,on_or_after_delivery; delivery 2026-06-11; entry_price 0.005192
2026-10-17T12:15:40+00:00,S2,void,HIGHUSDT,2026-09-29,2026-10-16,,on_or_after_delivery; delivery 2026-06-11; entry_price 0.0478
2026-10-17T12:15:40+00:00,S2,void,MBOXUSDT,2026-09-29,2026-10-16,,on_or_after_delivery; delivery 2026-06-11; entry_price 0.0051
2026-10-17T12:15:40+00:00,S2,void,SCRTUSDT,2026-10-05,2026-10-16,,on_or_after_delivery; delivery 2026-08-26; entry_price 0.01262
2026-10-17T12:15:51+00:00,S5,void,STGUSDT,2026-10-09,2026-10-16,,SHORT on_or_after_delivery; delivery 2026-09-24; entry_price 0.1283
2026-10-17T12:16:26+00:00,C19,void,COSUSDT,2026-10-10,2026-10-16,,long on_or_after_delivery; delivery 2026-06-11; entry_price 0.0004862125
```

## How to apply and how to reverse

1. Tag the commit that is live: `git tag pre-tradability-apply` (done at the end of this session on the commit that contains this file).
2. Set `apply_state_fixes` to true in `data/tradability_config.json`, commit, push. The next daily run of each engine performs the voids above.
3. Restating the historical closed trades (C19 +6.07, count corrections) is NOT done by the engines; it is a separate, deliberate edit of
   the ledgers that I have not written to the live tree. Say if you want it.
4. Reverse: `git revert` the config commit (the voided positions come back only from git history, which the tag preserves).
