@echo off
:: Starts the Telegram /status bot. Keep this window open to answer /status.
:: Reads credentials from backend\.env (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).
cd /d %~dp0
echo Starting UngerFink /status bot -- keep this window open. Ctrl+C to stop.
python tools\telegram_status_bot.py
pause
