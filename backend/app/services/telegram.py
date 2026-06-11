"""Telegram push notifications.

A send failure must never break a trade or a workflow, so every error is
swallowed and reported as a ``False`` return rather than raised.
"""
from __future__ import annotations

import httpx

from app.config import Settings


async def send_message(settings: Settings, text: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message. Returns True on success, False otherwise.

    No-ops (returns False) when the bot token or chat id is unconfigured.
    """
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            return resp.status_code == 200
    except Exception:
        return False
