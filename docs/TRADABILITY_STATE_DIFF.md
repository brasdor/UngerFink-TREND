# Exchange-tradability: exact live-state diff (DRY RUN, nothing applied)

Computed 2026-10-11 against data/exchange_status_cache.json. Apply by setting `apply_state_fixes` to true in `data/tradability_config.json`; the engines then perform exactly these changes on their next run.
**7 open positions affected in total.**


## S1 (t9b_paper) - 3 open, 0 affected

No change.

## S2 (t9b_mr_paper) - 5 open, 5 affected

| list | symbol | entry | delivery | action | reason | settlement px | entry px | qty | P&L effect |
|---|---|---|---|---|---|---|---|---|---|
| open_positions | COSUSDT | 2026-09-29 | 2026-06-11 | VOID | on_or_after_delivery |  | 0.000485 | 236873818.9999999 | 0 (void: no P&L booked) |
| open_positions | DUSDT | 2026-09-29 | 2026-06-11 | VOID | on_or_after_delivery |  | 0.005192 | 23375705.822368413 | 0 (void: no P&L booked) |
| open_positions | HIGHUSDT | 2026-09-29 | 2026-06-11 | VOID | on_or_after_delivery |  | 0.0478 | 1645718.983325613 | 0 (void: no P&L booked) |
| open_positions | MBOXUSDT | 2026-09-29 | 2026-06-11 | VOID | on_or_after_delivery |  | 0.0051 | 16759940.023584899 | 0 (void: no P&L booked) |
| open_positions | SCRTUSDT | 2026-10-05 | 2026-08-26 | VOID | on_or_after_delivery |  | 0.01262 | 35416.323960368405 | 0 (void: no P&L booked) |

Equity `10151.735099999996` unchanged by voids; `closed_trade_count` `45` unchanged (a void is not a trade). Positions removed from state:

```json
{
 "open_positions": {
  "position_id": "COSUSDT_20260929",
  "symbol": "COSUSDT",
  "entry_date": "2026-09-29",
  "entry_price": 0.000485,
  "stop_loss": 0.00048485,
  "initial_risk_per_unit": 1.5e-07,
  "risk_amount_usdt": 35.53107284999999,
  "qty": 236873818.9999999,
  "bars_held": 11,
  "atr_at_entry": 5e-08,
  "rsi_at_entry": 16.0674
 }
}
{
 "open_positions": {
  "position_id": "DUSDT_20260929",
  "symbol": "DUSDT",
  "entry_date": "2026-09-29",
  "entry_price": 0.005192,
  "stop_loss": 0.00519048,
  "initial_risk_per_unit": 1.52e-06,
  "risk_amount_usdt": 35.53107284999999,
  "qty": 23375705.822368413,
  "bars_held": 11,
  "atr_at_entry": 5.1e-07,
  "rsi_at_entry": 21.8912
 }
}
{
 "open_positions": {
  "position_id": "HIGHUSDT_20260929",
  "symbol": "HIGHUSDT",
  "entry_date": "2026-09-29",
  "entry_price": 0.0478,
  "stop_loss": 0.04777841,
  "initial_risk_per_unit": 2.159e-05,
  "risk_amount_usdt": 35.53107284999999,
  "qty": 1645718.983325613,
  "bars_held": 11,
  "atr_at_entry": 7.2e-06,
  "rsi_at_entry": 17.9655
 }
}
{
 "open_positions": {
  "position_id": "MBOXUSDT_20260929",
  "symbol": "MBOXUSDT",
  "entry_date": "2026-09-29",
  "entry_price": 0.0051,
  "stop_loss": 0.00509788,
  "initial_risk_per_unit": 2.12e-06,
  "risk_amount_usdt": 35.53107284999999,
  "qty": 16759940.023584899,
  "bars_held": 11,
  "atr_at_entry": 7.1e-07,
  "rsi_at_entry": 23.093
 }
}
{
 "open_positions": {
  "position_id": "SCRTUSDT_20261005",
  "symbol": "SCRTUSDT",
  "entry_date": "2026-10-05",
  "entry_price": 0.01262,
  "stop_loss": 0.01186757,
  "initial_risk_per_unit": 0.00075243,
  "risk_amount_usdt": 26.648304637499997,
  "qty": 35416.323960368405,
  "bars_held": 5,
  "atr_at_entry": 0.00025081,
  "rsi_at_entry": 23.6459
 }
}
```

## S3 (t9b_consecdowndays_paper) - 6 open, 0 affected

No change.

## S5 (t9b_momentum_paper) - 30 open, 1 affected

| list | symbol | entry | delivery | action | reason | settlement px | entry px | qty | P&L effect |
|---|---|---|---|---|---|---|---|---|---|
| short_positions | STGUSDT | 2026-10-09 | 2026-09-24 | VOID | on_or_after_delivery |  | 0.1283 | 3136.80109737 | 0 (void: no P&L booked) |

Equity `10728.8235` unchanged by voids; `closed_trade_count` `n/a` unchanged (a void is not a trade). Positions removed from state:

```json
{
 "short_positions": {
  "symbol": "STGUSDT",
  "side": "SHORT",
  "entry_date": "2026-10-09",
  "entry_price": 0.1283,
  "quantity": 3136.80109737,
  "notional": 402.4516
 }
}
```

## S6 (t9b_volcontraction_paper) - 3 open, 0 affected

No change.

## S7 (t9b_macross_paper) - 6 open, 0 affected

No change.

## S8 (t9b_rsi_mr_funding_paper) - 0 open, 0 affected

No change.

## C12 (t9_candidate12_paper) - 20 open, 0 affected

No change.

## C19 (t9_candidate19_paper) - 20 open, 1 affected

| list | symbol | entry | delivery | action | reason | settlement px | entry px | qty | P&L effect |
|---|---|---|---|---|---|---|---|---|---|
| open_positions | COSUSDT | 2026-10-10 | 2026-06-11 | VOID | on_or_after_delivery |  | 0.0004862125 | None | 0 (void: no P&L booked) |

Equity `66483.1463` unchanged by voids; `closed_trade_count` `51` unchanged (a void is not a trade). Positions removed from state:

```json
{
 "open_positions": {
  "symbol": "COSUSDT",
  "side": "long",
  "combo_idx": 0,
  "entry_date": "2026-10-10",
  "entry_price": 0.0004862125,
  "stop_distance_px": 4.883912210946095e-08,
  "atr_at_entry": 2.4419561054730473e-08,
  "notional_usdt": 6648.3146268312985,
  "notional_pct": 0.1,
  "margin_reserved_usdt": 3324.1573134156492,
  "risk_amount_usdt": 0.6694801028493382,
  "cluster": "solo_COSUSDT",
  "liquidity_tier": "tier3"
 }
}
```
