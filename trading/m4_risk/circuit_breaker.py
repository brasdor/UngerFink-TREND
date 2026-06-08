"""
M4 Circuit Breaker
==================
Automated kill switches that reduce or halt trading when pre-defined
risk thresholds are breached.  Rules are evaluated in severity order
and the worst state wins.
"""
from enum import Enum


class CBState(Enum):
    OK      = "ok"
    HALVED  = "halved"   # position sizes x 0.5
    STOPPED = "stopped"  # no new trades, close all


# ─── threshold constants ───────────────────────────────────────────────────────
_DAILY_LOSS_HALT     = -0.03   # -3%  daily P&L  -> HALVED
_DD_HALT             = -0.10   # -10% peak-to-trough -> HALVED
_DD_STOP             = -0.20   # -20% peak-to-trough -> STOPPED
_BTC_MOVE_HALT       =  0.08   # 8%  absolute BTC 24h move -> HALVED
_CONSEC_LOSS_WEEKS   =  3      # 3 losing weeks -> HALVED
_LOW_VOL_RATIO       =  0.30   # worst held coin < 30% of 30d avg vol -> HALVED


def check_circuit_breakers(
    portfolio_state: dict,
) -> tuple[CBState, list[str]]:
    """
    Evaluate all circuit breaker rules and return the worst triggered state
    together with a list of human-readable descriptions for each trigger.

    portfolio_state keys
    --------------------
    daily_pnl_pct          : float   today's P&L as % of equity (e.g. -0.04 = -4%)
    peak_to_trough_pct     : float   current drawdown from equity peak (negative)
    btc_24h_move           : float   BTC price % move last 24 h (signed, e.g. -0.09)
    consecutive_losing_weeks : int   number of consecutive weeks with negative return
    lowest_volume_coin_pct : float   worst held coin's volume vs 30d avg (0-1)
    """
    daily_pnl     = float(portfolio_state.get("daily_pnl_pct", 0.0))
    peak_to_trough = float(portfolio_state.get("peak_to_trough_pct", 0.0))
    btc_move      = float(portfolio_state.get("btc_24h_move", 0.0))
    consec_loss   = int(portfolio_state.get("consecutive_losing_weeks", 0))
    low_vol       = float(portfolio_state.get("lowest_volume_coin_pct", 1.0))

    worst   = CBState.OK
    triggers: list[str] = []

    def _escalate(new_state: CBState, msg: str) -> None:
        nonlocal worst
        triggers.append(msg)
        # Severity: STOPPED > HALVED > OK
        if new_state == CBState.STOPPED or (
            new_state == CBState.HALVED and worst == CBState.OK
        ):
            worst = new_state

    # Rules checked in order; _escalate keeps worst state
    if daily_pnl < _DAILY_LOSS_HALT:
        _escalate(CBState.HALVED, f"Daily loss > 3%  (actual {daily_pnl:.1%})")

    if peak_to_trough < _DD_STOP:
        _escalate(CBState.STOPPED, f"Drawdown > 20%  (actual {peak_to_trough:.1%})")
    elif peak_to_trough < _DD_HALT:
        _escalate(CBState.HALVED, f"Drawdown > 10%  (actual {peak_to_trough:.1%})")

    if abs(btc_move) > _BTC_MOVE_HALT:
        _escalate(CBState.HALVED, f"BTC moved > 8% in 24h  (actual {btc_move:+.1%})")

    if consec_loss >= _CONSEC_LOSS_WEEKS:
        _escalate(CBState.HALVED, f"3 consecutive losing weeks  ({consec_loss} weeks)")

    if low_vol < _LOW_VOL_RATIO:
        _escalate(
            CBState.HALVED,
            f"Liquidity warning on a held coin  ({low_vol:.0%} of 30d avg vol)",
        )

    return worst, triggers


def get_position_multiplier(state: CBState) -> float:
    """Map CBState to a position size multiplier."""
    return {
        CBState.OK:      1.0,
        CBState.HALVED:  0.5,
        CBState.STOPPED: 0.0,
    }[state]
