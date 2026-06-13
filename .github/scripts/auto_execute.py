#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-execute T9B paper-engine activity on Binance (entries AND exits).

Runs in the daily workflow after the engines run. Mirrors the paper engines:
  ENTRY: for each signal the engine actually entered (event=ENTRY), place a
         sized, capped market BUY.
  EXIT : for each position we hold (from our ledger) that the engine has
         CLOSED (no longer in open_positions.csv), place a market SELL.

There is no EXIT event in the logs, so exits are detected by a position
leaving the engine's open_positions.csv.

HARD SAFETY GATES (all deliberate):
  AUTO_EXECUTE_ENABLED=true   master switch — default OFF, nothing runs otherwise
  EXCHANGE_TESTNET            default 'true' -> fake money; set 'false' for real
  EXCHANGE_API_KEY / SECRET   required for any placement

Ledger: data/auto_orders/placed.csv is an append-only BUY/SELL record (committed
back). Our net open position per (strategy, symbol) is the replay of BUYs minus
SELLs, which makes both entries and exits idempotent across re-runs.

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
LEDGER_FIELDS = [
    "key", "timestamp", "action", "strategy", "symbol", "qty",
    "avg_price", "notional_usdt", "exchange_order_id", "testnet", "status",
]

SYSTEMS = [
    {"strategy": "DonchianLong",     "data_dir": ROOT / "data" / "t9b_paper",                  "stop_field": "initial_stop"},
    {"strategy": "MeanReversionRSI", "data_dir": ROOT / "data" / "t9b_mr_paper",               "stop_field": "stop_loss"},
    {"strategy": "ConsecDownDaysMR", "data_dir": ROOT / "data" / "t9b_consecdowndays_paper",   "stop_field": "stop_loss"},
]


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _enabled() -> bool:
    return _env_bool("AUTO_EXECUTE_ENABLED", False)


def _testnet() -> bool:
    return _env_bool("EXCHANGE_TESTNET", True)


# --------------------------------------------------------------------------- #
# data readers
# --------------------------------------------------------------------------- #
def _run_date(system: dict) -> str:
    state = system["data_dir"] / "state.json"
    if state.exists():
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("last_run_date", "")
        except Exception:
            pass
    return ""


def _accepted_entries(system: dict, run_date: str):
    """Yield (symbol, entry, stop) for signals the engine actually entered."""
    signals_csv = system["data_dir"] / "signals_today.csv"
    daily_log = system["data_dir"] / "daily_log.csv"
    if not signals_csv.exists():
        return
    try:
        sig = pd.read_csv(signals_csv)
    except Exception:
        return
    if sig.empty:
        return

    accepted: set = set()
    if daily_log.exists() and run_date:
        try:
            log = pd.read_csv(daily_log, on_bad_lines="skip")
            today = log[log["run_date"] == run_date]
            accepted = set(today[today["event"] == "ENTRY"]["symbol"].dropna().tolist())
        except Exception:
            pass

    for _, row in sig.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol or symbol not in accepted:
            continue
        try:
            entry = float(row.get("close"))
            stop = float(row.get(system["stop_field"]))
        except (TypeError, ValueError):
            continue
        if entry <= 0 or stop <= 0 or entry <= stop:
            continue
        yield symbol, entry, stop


def _engine_open_symbols(system: dict) -> set:
    """Symbols the engine currently holds (from open_positions.csv)."""
    op = system["data_dir"] / "open_positions.csv"
    if not op.exists():
        return set()
    try:
        df = pd.read_csv(op, on_bad_lines="skip")
        if "symbol" not in df.columns:
            return set()
        return set(df["symbol"].dropna().astype(str).str.strip().tolist())
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# ledger (append-only; net position derived by replay)
# --------------------------------------------------------------------------- #
def _ledger_df() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_FIELDS)
    try:
        return pd.read_csv(LEDGER)
    except Exception:
        return pd.DataFrame(columns=LEDGER_FIELDS)


def _net_positions(df: pd.DataFrame) -> dict:
    """(strategy, symbol) -> net qty held (sum BUY - sum SELL)."""
    net: dict = {}
    for _, r in df.iterrows():
        k = (str(r.get("strategy")), str(r.get("symbol")))
        try:
            q = float(r.get("qty"))
        except (TypeError, ValueError):
            continue
        action = str(r.get("action", "")).upper()
        if action == "BUY":
            net[k] = net.get(k, 0.0) + q
        elif action == "SELL":
            net[k] = net.get(k, 0.0) - q
    return net


def _append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


# --------------------------------------------------------------------------- #
# telegram
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def _place(client, side: str, symbol: str, qty: float):
    """Market order. Returns (filled_qty, avg_price, order_id, status) or None."""
    try:
        order = client.create_order(symbol, "market", side, qty)
    except Exception as e:
        print(f"[AUTO] {side} {symbol} FAILED: {e}")
        return None
    avg = order.get("average") or order.get("price")
    return float(order.get("filled") or 0), avg, str(order.get("id")), order.get("status")


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
        "apiKey": key_id, "secret": secret,
        "enableRateLimit": True, "options": {"defaultType": "spot"},
    })
    if testnet:
        client.set_sandbox_mode(True)
    client.load_markets()
    print(f"[AUTO] enabled -- mode={'TESTNET (fake money)' if testnet else 'LIVE (REAL MONEY)'}  "
          f"risk={risk_pct}  cap={max_order} USDT")

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

    ledger = _ledger_df()
    existing_keys = set(ledger["key"].astype(str).tolist()) if not ledger.empty else set()
    net = _net_positions(ledger)
    n_buy = n_sell = 0

    for system in SYSTEMS:
        strat = system["strategy"]
        run_date = _run_date(system)
        open_syms = _engine_open_symbols(system)

        # ---- EXITS: we hold it, engine no longer does -> SELL ----
        for (s_strat, s_sym), held in list(net.items()):
            if s_strat != strat or held <= 0:
                continue
            if s_sym in open_syms:
                continue  # engine still holds it -> keep
            qty = float(client.amount_to_precision(s_sym, held))
            if qty <= 0:
                continue
            res = _place(client, "sell", s_sym, qty)
            if res is None:
                continue
            filled, avg, oid, status = res
            _append_ledger({
                "key": f"{run_date}:{strat}:{s_sym}:SELL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "SELL", "strategy": strat, "symbol": s_sym, "qty": qty,
                "avg_price": avg, "notional_usdt": round(filled * float(avg), 4) if avg else "",
                "exchange_order_id": oid, "testnet": testnet, "status": status,
            })
            net[(s_strat, s_sym)] = held - qty
            n_sell += 1
            print(f"[AUTO] EXIT  SELL {strat} {s_sym} qty={qty} avg={avg} id={oid}")
            _telegram(f"⚪ <b>Auto-exit</b> ({'testnet' if testnet else 'LIVE'})\n"
                      f"{strat}: SELL <code>{s_sym}</code> qty {qty} @ {avg}")

        # ---- ENTRIES: engine entered it -> BUY ----
        for symbol, entry, stop in _accepted_entries(system, run_date):
            key = f"{run_date}:{strat}:{symbol}:BUY"
            if key in existing_keys:
                print(f"[AUTO] {key} already placed -- skip (idempotent)")
                continue

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

            res = _place(client, "buy", symbol, qty)
            if res is None:
                continue
            filled, avg, oid, status = res
            _append_ledger({
                "key": key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "BUY", "strategy": strat, "symbol": symbol, "qty": qty,
                "avg_price": avg, "notional_usdt": round(filled * float(avg), 4) if avg else round(notional, 4),
                "exchange_order_id": oid, "testnet": testnet, "status": status,
            })
            existing_keys.add(key)
            net[(strat, symbol)] = net.get((strat, symbol), 0.0) + qty
            n_buy += 1
            print(f"[AUTO] ENTRY BUY  {strat} {symbol} qty={qty} avg={avg} id={oid}")
            _telegram(f"⚡ <b>Auto-entry</b> ({'testnet' if testnet else 'LIVE'})\n"
                      f"{strat}: BUY <code>{symbol}</code> qty {qty} @ {avg}")

    print(f"[AUTO] done -- {n_buy} buy(s), {n_sell} sell(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
