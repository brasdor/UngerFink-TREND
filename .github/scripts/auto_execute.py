#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-execute today's ACCEPTED T9B signals on Binance.

Runs in the daily workflow after the engines produce signals. For each signal
that the paper engine actually entered (event=ENTRY in daily_log), places a
sized, capped spot BUY. Mirrors the web app's execution safety rules.

HARD SAFETY GATES (all must be deliberately set):
  AUTO_EXECUTE_ENABLED=true   master switch — default OFF, nothing runs otherwise
  EXCHANGE_TESTNET            default 'true' -> fake money; set 'false' for real
  EXCHANGE_API_KEY / SECRET   required for any placement

Idempotency: a committed ledger (data/auto_orders/placed.csv) keyed by
run_date+strategy+symbol means re-running the workflow never double-places.

Never fails the workflow: any error is logged and the script exits 0.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER = ROOT / "data" / "auto_orders" / "placed.csv"

# strategy label + which CSV column holds the protective stop
SYSTEMS = [
    {"strategy": "DonchianLong",   "data_dir": ROOT / "data" / "t9b_paper",                 "stop_field": "initial_stop"},
    {"strategy": "MeanReversionRSI", "data_dir": ROOT / "data" / "t9b_mr_paper",            "stop_field": "stop_loss"},
    {"strategy": "ConsecDownDaysMR", "data_dir": ROOT / "data" / "t9b_consecdowndays_paper", "stop_field": "stop_loss"},
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _enabled() -> bool:
    return _env_bool("AUTO_EXECUTE_ENABLED", False)


def _testnet() -> bool:
    return _env_bool("EXCHANGE_TESTNET", True)


def _accepted_rows(system: dict):
    """Yield (symbol, entry, stop) for signals the paper engine actually entered."""
    data_dir = system["data_dir"]
    signals_csv = data_dir / "signals_today.csv"
    daily_log = data_dir / "daily_log.csv"
    state_json = data_dir / "state.json"
    if not signals_csv.exists():
        return

    try:
        sig = pd.read_csv(signals_csv)
    except Exception:
        return
    if sig.empty:
        return

    run_date = ""
    if state_json.exists():
        try:
            run_date = json.loads(state_json.read_text(encoding="utf-8")).get("last_run_date", "")
        except Exception:
            pass

    accepted: set = set()
    if daily_log.exists() and run_date:
        try:
            log = pd.read_csv(daily_log)
            today = log[log["run_date"] == run_date]
            accepted = set(today[today["event"] == "ENTRY"]["symbol"].dropna().tolist())
        except Exception:
            pass

    for _, row in sig.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol or symbol not in accepted:
            continue  # only auto-execute signals the engine actually entered
        try:
            entry = float(row.get("close"))
            stop = float(row.get(system["stop_field"]))
        except (TypeError, ValueError):
            continue
        if entry <= 0 or stop <= 0 or entry <= stop:
            continue
        yield run_date, symbol, entry, stop


def _load_ledger_keys() -> set:
    if not LEDGER.exists():
        return set()
    try:
        df = pd.read_csv(LEDGER)
        return set(df["key"].astype(str).tolist())
    except Exception:
        return set()


def _append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def _telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    import urllib.parse
    import urllib.request
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data),
            timeout=15,
        )
    except Exception:
        pass


def main() -> int:
    if not _enabled():
        print("[AUTO] AUTO_EXECUTE_ENABLED not set -- no orders placed.")
        return 0

    key_id = os.environ.get("EXCHANGE_API_KEY", "")
    secret = os.environ.get("EXCHANGE_SECRET", "")
    if not key_id or not secret:
        print("[AUTO] EXCHANGE_API_KEY/SECRET missing -- skipping.")
        return 0

    testnet = _testnet()
    risk_pct = float(os.environ.get("RISK_PCT", "0.0025"))
    max_order = float(os.environ.get("MAX_ORDER_USDT", "100"))
    equity_env = os.environ.get("EQUITY_USDT")

    import ccxt

    client = ccxt.binance({
        "apiKey": key_id,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    if testnet:
        client.set_sandbox_mode(True)
    client.load_markets()

    mode = "TESTNET (fake money)" if testnet else "LIVE (REAL MONEY)"
    print(f"[AUTO] enabled -- mode={mode}  risk={risk_pct}  cap={max_order} USDT")

    # equity: explicit, else free USDT balance
    if equity_env:
        equity = float(equity_env)
    else:
        try:
            equity = float(client.fetch_balance().get("free", {}).get("USDT", 0.0))
        except Exception as e:
            print(f"[AUTO] could not read balance: {e}")
            return 0
    if equity <= 0:
        print("[AUTO] no equity -- skipping.")
        return 0

    placed_keys = _load_ledger_keys()
    placed_count = 0

    for system in SYSTEMS:
        for run_date, symbol, entry, stop in _accepted_rows(system):
            key = f"{run_date}:{system['strategy']}:{symbol}"
            if key in placed_keys:
                print(f"[AUTO] {key} already placed -- skip (idempotent)")
                continue

            # size: loss-to-stop == risk_pct of equity
            per_unit_risk = entry - stop
            raw_qty = (equity * risk_pct) / per_unit_risk
            try:
                qty = float(client.amount_to_precision(symbol, raw_qty))
            except Exception as e:
                print(f"[AUTO] {symbol} precision error: {e}")
                continue

            notional = qty * entry
            limits = client.markets.get(symbol, {}).get("limits", {})
            min_qty = (limits.get("amount") or {}).get("min")
            min_notional = (limits.get("cost") or {}).get("min")
            if min_qty is not None and qty < min_qty:
                print(f"[AUTO] {symbol} qty {qty} below min {min_qty} -- skip")
                continue
            if min_notional is not None and notional < min_notional:
                print(f"[AUTO] {symbol} notional {notional:.2f} below min {min_notional} -- skip")
                continue
            if notional > max_order:
                print(f"[AUTO] {symbol} notional {notional:.2f} over cap {max_order} -- skip")
                continue

            try:
                order = client.create_order(symbol, "market", "buy", qty)
            except Exception as e:
                print(f"[AUTO] {symbol} placement FAILED: {e}")
                continue

            filled = float(order.get("filled") or 0)
            avg = order.get("average") or order.get("price")
            record = {
                "key": key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": system["strategy"],
                "symbol": symbol,
                "qty": qty,
                "avg_price": avg,
                "notional_usdt": round((filled * float(avg)) if avg else notional, 4),
                "exchange_order_id": str(order.get("id")),
                "testnet": testnet,
                "status": order.get("status"),
            }
            _append_ledger(record)
            placed_keys.add(key)
            placed_count += 1
            print(f"[AUTO] PLACED {key}  qty={qty}  avg={avg}  id={order.get('id')}")
            _telegram(
                f"⚡ <b>Auto-exec</b> ({'testnet' if testnet else 'LIVE'})\n"
                f"{system['strategy']}: BUY <code>{symbol}</code> qty {qty} @ {avg}"
            )

    print(f"[AUTO] done -- {placed_count} order(s) placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
