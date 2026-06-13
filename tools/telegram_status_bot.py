#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram /status bot — a local long-polling listener.

Run this on a machine that stays on (like the daily .bat). While running, it
answers /status in your Telegram chat with a live snapshot: open paper
positions per strategy, last engine run, the auto-execution ledger, and the
(testnet) exchange balance.

  python tools/telegram_status_bot.py

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (and optional EXCHANGE_* for the
balance line) from the environment or from backend/.env. Responds only to the
configured chat id. Ctrl+C to stop. It does NOT place or change anything.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SYSTEMS = [
    ("Donchian",   ROOT / "data" / "t9b_paper"),
    ("RSI-MR",     ROOT / "data" / "t9b_mr_paper"),
    ("ConsecDown", ROOT / "data" / "t9b_consecdowndays_paper"),
]
LEDGER = ROOT / "data" / "auto_orders" / "placed.csv"


# --------------------------------------------------------------------------- #
def load_env() -> None:
    """Populate os.environ from backend/.env for any keys not already set."""
    envf = ROOT / "backend" / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _last_run(d: Path) -> str:
    sj = d / "state.json"
    if sj.exists():
        try:
            return json.loads(sj.read_text(encoding="utf-8")).get("last_run_date", "?")
        except Exception:
            pass
    return "?"


def _open_positions(d: Path):
    op = d / "open_positions.csv"
    if not op.exists():
        return 0, []
    try:
        df = pd.read_csv(op, on_bad_lines="skip")
        syms = df["symbol"].dropna().astype(str).tolist() if "symbol" in df.columns else []
        return len(syms), syms
    except Exception:
        return 0, []


def _last_equity(d: Path):
    eq = d / "equity_curve.csv"
    if not eq.exists():
        return None
    try:
        df = pd.read_csv(eq, on_bad_lines="skip")
        if df.empty:
            return None
        for col in ("equity", "closed_equity", "equity_usdt"):
            if col in df.columns:
                return float(df[col].iloc[-1])
    except Exception:
        pass
    return None


def _ledger_net():
    if not LEDGER.exists():
        return {}
    try:
        df = pd.read_csv(LEDGER)
    except Exception:
        return {}
    net: dict = {}
    for _, r in df.iterrows():
        k = (str(r.get("strategy")), str(r.get("symbol")))
        try:
            q = float(r.get("qty"))
        except (TypeError, ValueError):
            continue
        net[k] = net.get(k, 0.0) + (q if str(r.get("action")).upper() == "BUY" else -q)
    return {k: v for k, v in net.items() if v > 1e-9}


def _testnet_balance():
    key = os.environ.get("EXCHANGE_API_KEY", "")
    sec = os.environ.get("EXCHANGE_SECRET", "")
    if not key or not sec:
        return None, None
    testnet = os.environ.get("EXCHANGE_TESTNET", "true").strip().lower() != "false"
    try:
        import ccxt
        ex = ccxt.binance({"apiKey": key, "secret": sec, "enableRateLimit": True,
                           "options": {"defaultType": "spot"}})
        if testnet:
            ex.set_sandbox_mode(True)
        free = ex.fetch_balance().get("free", {})
        usdt = float(free.get("USDT", 0.0))
        return usdt, testnet
    except Exception:
        return None, testnet


def build_status() -> str:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"📊 <b>UngerFink status</b> — {now}", ""]

    for label, d in SYSTEMS:
        n, syms = _open_positions(d)
        eq = _last_equity(d)
        eq_s = f"  ·  equity ${eq:,.0f}" if eq is not None else ""
        shown = ", ".join(syms[:6]) + ("…" if len(syms) > 6 else "")
        lines.append(f"<b>{label}</b>: {n} open{(' — ' + shown) if shown else ''}{eq_s}")
        lines.append(f"   last run: {_last_run(d)}")

    net = _ledger_net()
    lines.append("")
    if net:
        lines.append(f"<b>Auto-exec</b>: {len(net)} open position(s)")
        for (strat, sym), q in list(net.items())[:8]:
            lines.append(f"   {strat}: {sym} qty {q}")
    else:
        lines.append("<b>Auto-exec</b>: no open positions")

    usdt, testnet = _testnet_balance()
    if usdt is not None:
        tag = "testnet" if testnet else "LIVE"
        lines.append(f"<b>Balance</b> ({tag}): {usdt:,.2f} USDT free")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _api(token: str, method: str, params: dict, timeout: int = 35):
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=timeout) as r:
        return json.load(r)


def _send(token: str, chat: str, text: str) -> None:
    try:
        _api(token, "sendMessage",
             {"chat_id": chat, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": "true"}, timeout=15)
    except Exception as e:
        print(f"[BOT] send failed: {e}")


def main() -> int:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("[BOT] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set "
              "(env or backend/.env). Exiting.")
        return 1

    # register the command in Telegram's menu (best effort)
    try:
        _api(token, "setMyCommands",
             {"commands": json.dumps([{"command": "status",
                                       "description": "Show open positions, equity, auto-exec, balance"}])},
             timeout=15)
    except Exception:
        pass

    # Drain any messages queued before startup so we don't reply to stale
    # commands — only answer what arrives while we're listening.
    offset = None
    try:
        drained = _api(token, "getUpdates", {"timeout": 0}, timeout=15)
        results = drained.get("result", [])
        if results:
            offset = results[-1]["update_id"] + 1
    except Exception:
        pass

    print("[BOT] listening for /status  (Ctrl+C to stop)")
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = _api(token, "getUpdates", params, timeout=40)
        except KeyboardInterrupt:
            print("\n[BOT] stopped.")
            return 0
        except Exception as e:
            print(f"[BOT] poll error: {e}")
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if str((msg.get("chat") or {}).get("id")) != str(chat):
                continue  # only answer the configured chat
            text = (msg.get("text") or "").strip().lower()
            if text.startswith("/status"):
                print("[BOT] /status -> replying")
                _send(token, chat, build_status())
            elif text.startswith("/start") or text.startswith("/help"):
                _send(token, chat, "Commands:\n/status — open positions, equity, auto-exec, balance")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[BOT] stopped.")
