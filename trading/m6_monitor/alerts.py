"""
M6 Alerts
=========
Telegram notification system.  All sends are fire-and-forget -- wrapped in
try/except so a Telegram failure never crashes M5 or M6.
Plain text only -- no Markdown or HTML (Binance symbol names break parsers).
"""
import os
import sys

from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_LEVEL_TAG = {
    "info":     "[INFO]",
    "warning":  "[WARNING]",
    "critical": "[CRITICAL]",
}


def _send(text: str) -> bool:
    """Post plain text to Telegram. Returns True on HTTP 200."""
    if not _TOKEN or not _CHAT_ID:
        return False
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            data={"chat_id": _CHAT_ID, "text": text},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning(f"Telegram send failed -- skipping ({exc})")
        return False


def send_alert(message: str, level: str = "info") -> bool:
    tag = _LEVEL_TAG.get(level, "[INFO]")
    return _send(f"{tag} UngerFink-TREND\n{message}")


def send_rebalancing_summary(
    opened: int,
    closed: int,
    kept: int,
    pnl: float,
    regime_score: float,
) -> bool:
    text = (
        f"[M5] Weekly Rebalancing Complete\n"
        f"Opened: {opened}  Closed: {closed}  Kept: {kept}\n"
        f"Total P&L: {pnl:+.2f} USDT\n"
        f"Regime score: {regime_score:.3f}"
    )
    return _send(text)


def send_circuit_breaker_alert(cb_state: str, triggers: list) -> bool:
    trig_str = "; ".join(str(t) for t in triggers) if triggers else "none"
    text = (
        f"[CIRCUIT BREAKER] State: {cb_state.upper()}\n"
        f"Triggers: {trig_str}"
    )
    return _send(text)


def send_heartbeat_alert(message: str) -> bool:
    return _send(f"[HEARTBEAT ALERT] UngerFink-TREND\n{message}")


def send_weekly_report(report: dict) -> bool:
    week    = report.get("week_start", "?")
    pnl     = float(report.get("weekly_pnl", 0.0))
    ret_pct = float(report.get("weekly_return_pct", 0.0))
    wr      = float(report.get("win_rate", 0.0))
    sharpe  = report.get("sharpe_live")
    mdd     = float(report.get("max_drawdown", 0.0))
    best    = report.get("best_trade", "n/a")
    worst   = report.get("worst_trade", "n/a")
    notes   = report.get("notes", "")

    sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, float) and sharpe == sharpe else "n/a"

    text = (
        f"[WEEKLY REPORT] Week of {week}\n"
        f"P&L: {pnl:+.2f} USDT ({ret_pct:+.2f}%)\n"
        f"Win rate: {wr:.1f}%\n"
        f"Sharpe (live): {sharpe_str}  Backtest: 1.132\n"
        f"Max drawdown: {mdd:.2f}%\n"
        f"Best trade: {best}\n"
        f"Worst trade: {worst}"
    )
    if notes:
        text += f"\nNotes: {notes}"
    return _send(text)


# ── CLI (used by run_daily.bat) ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send a Telegram alert")
    parser.add_argument("--message", required=True, help="Alert message text")
    parser.add_argument("--level", default="info",
                        choices=["info", "warning", "critical"],
                        help="Alert level")
    args = parser.parse_args()
    ok = send_alert(args.message, level=args.level)
    import sys
    sys.exit(0 if ok else 1)
