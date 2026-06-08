"""
M4 Regime Filter
================
Combines macro (FRED) and Binance on-chain proxies (funding rates,
long/short ratio, open interest) into a single regime_score 0.0-1.0.
Falls back to neutral defaults if any table is empty or sparse so M4
never crashes from missing data.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_MACRO_FALLBACK = 0.7
_WHALE_FALLBACK = 0.8

# ─── private helpers that return (score, notes) ───────────────────────────────

def _macro_score(conn) -> tuple[float, list[str]]:
    try:
        df = pd.read_sql_query(
            "SELECT series_id, date, value FROM macro ORDER BY date", conn
        )
    except Exception as exc:
        logger.warning(f"macro table read error ({exc}) -- using neutral macro score {_MACRO_FALLBACK}")
        return _MACRO_FALLBACK, [f"macro table error -- fallback {_MACRO_FALLBACK}"]

    if df.empty:
        logger.info(f"FRED key not configured -- using neutral macro score {_MACRO_FALLBACK}")
        return _MACRO_FALLBACK, [f"FRED key not configured -- fallback {_MACRO_FALLBACK}"]

    notes: list[str] = []
    score = 0.0

    def _series(*ids):
        upper = [s.upper() for s in ids]
        return df[df["series_id"].str.upper().isin(upper)].sort_values("date")

    # ── Component 1: M2 YoY vs historical median ──────────────────────────────
    m2 = _series("M2", "M2SL")
    if len(m2) >= 13:
        vals = m2.set_index("date")["value"]
        latest_yoy = vals.iloc[-1] / vals.iloc[-13] - 1
        yoys = [vals.iloc[i] / vals.iloc[i - 12] - 1 for i in range(12, len(vals))]
        median_yoy = pd.Series(yoys).median()
        if latest_yoy > median_yoy:
            score += 0.3
            notes.append(
                f"M2 YoY {latest_yoy:.1%} above median {median_yoy:.1%} -> +0.3"
            )
        else:
            score += 0.1
            notes.append(
                f"M2 YoY {latest_yoy:.1%} below median {median_yoy:.1%} -> +0.1"
            )
    else:
        score += 0.2
        notes.append("M2: insufficient data -> +0.2 (neutral)")

    # ── Component 2: DXY level and trend ─────────────────────────────────────
    dxy = _series("DXY", "DTWEXBGS", "DTWEXM")
    if len(dxy) >= 30:
        dxy_vals = dxy.set_index("date")["value"]
        latest_dxy = float(dxy_vals.iloc[-1])
        lookback = dxy_vals.iloc[-63] if len(dxy_vals) >= 63 else dxy_vals.iloc[0]
        is_falling  = latest_dxy < float(lookback)
        is_below_100 = latest_dxy < 100.0
        if is_falling and is_below_100:
            score += 0.3
            notes.append(f"DXY {latest_dxy:.1f} falling and below 100 -> +0.3")
        else:
            score += 0.1
            notes.append(
                f"DXY {latest_dxy:.1f} "
                f"({'falling' if is_falling else 'rising'}, "
                f"{'below' if is_below_100 else 'above'} 100) -> +0.1"
            )
    else:
        score += 0.2
        notes.append("DXY: insufficient data -> +0.2 (neutral)")

    # ── Component 3: FEDFUNDS cutting vs hiking ───────────────────────────────
    ff = _series("FEDFUNDS", "DFF")
    if len(ff) >= 3:
        ff_vals = ff.set_index("date")["value"]
        current   = float(ff_vals.iloc[-1])
        three_ago = float(ff_vals.iloc[-3])
        if current < three_ago:
            score += 0.2
            notes.append(
                f"FEDFUNDS cutting {three_ago:.2f}% -> {current:.2f}% -> +0.2"
            )
        else:
            score += 0.1
            notes.append(
                f"FEDFUNDS stable/hiking {three_ago:.2f}% -> {current:.2f}% -> +0.1"
            )
    else:
        score += 0.15
        notes.append("FEDFUNDS: insufficient data -> +0.15 (neutral)")

    # ── Component 4: ETF flows (not yet implemented) ──────────────────────────
    score += 0.2
    notes.append("ETF flows: not implemented -> +0.2 (fixed)")

    clamped = max(0.3, min(1.0, score))
    if clamped != score:
        notes.append(f"macro clamped {score:.3f} -> {clamped:.3f}")
    return clamped, notes


def _whale_score(conn) -> tuple[float, dict, list[str]]:
    """
    Returns (combined_score, subscores_dict, notes).
    subscores_dict keys: funding_proxy, ls_proxy, oi_proxy.
    All data sourced from Binance tables — no external API required.
    """
    notes: list[str] = []

    # ── Sub-score 1: Funding rate proxy ───────────────────────────────────────
    # Thresholds in decimal fraction form (DB storage format).
    # +0.0100% = 0.0001, +0.0050% = 0.00005, etc.
    _FR_HIGH  =  0.0001    # +0.0100%
    _FR_MID   =  0.00005   # +0.0050%
    _FR_LOW   = -0.00005   # -0.0050%
    _FR_VLOW  = -0.0001    # -0.0100%

    try:
        fr_rows = conn.execute(
            "SELECT funding_rate FROM funding_rates "
            "WHERE symbol='BTCUSDT' ORDER BY funding_time DESC LIMIT 21"
        ).fetchall()
    except Exception as exc:
        fr_rows = []
        logger.warning(f"funding_rates read error ({exc}) -- funding_proxy fallback 0.80")

    if len(fr_rows) < 5:
        funding_proxy = 0.80
        logger.warning("funding_rates has < 5 rows for BTCUSDT -- funding_proxy fallback 0.80")
        notes.append("whale.funding_proxy = 0.80 (insufficient data -- fallback)")
    else:
        mean_fr = sum(r[0] for r in fr_rows) / len(fr_rows)
        mean_fr_pct = mean_fr * 100   # convert to % for display

        if mean_fr > _FR_HIGH:
            funding_proxy = 0.65
            label = "overcrowded longs -- caution"
        elif mean_fr > _FR_MID:
            funding_proxy = 0.80
            label = "mildly bullish -- neutral"
        elif mean_fr >= _FR_LOW:
            funding_proxy = 0.90
            label = "balanced -- healthy"
        elif mean_fr >= _FR_VLOW:
            funding_proxy = 1.00
            label = "overcrowded shorts -- squeeze potential"
        else:
            funding_proxy = 1.00
            label = "extreme short bias -- strong squeeze setup"

        notes.append(
            f"whale.funding_proxy = {funding_proxy:.2f}"
            f" (7d avg rate: {mean_fr_pct:+.4f}% -- {label})"
        )

    # ── Sub-score 2: Long/short ratio proxy ───────────────────────────────────
    try:
        ls_row = conn.execute(
            "SELECT long_ratio FROM ls_ratio "
            "WHERE symbol='BTCUSDT' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    except Exception as exc:
        ls_row = None
        logger.warning(f"ls_ratio read error ({exc}) -- ls_proxy fallback 0.80")

    ls_count = conn.execute(
        "SELECT COUNT(*) FROM ls_ratio WHERE symbol='BTCUSDT'"
    ).fetchone()[0] if ls_row is not None else 0

    if ls_count < 5 or ls_row is None:
        ls_proxy = 0.80
        logger.warning("ls_ratio has < 5 rows for BTCUSDT -- ls_proxy fallback 0.80")
        notes.append("whale.ls_proxy = 0.80 (insufficient data -- fallback)")
    else:
        long_ratio = float(ls_row[0])

        if long_ratio > 0.60:
            ls_proxy = 0.65
            label = "too many longs -- crowded"
        elif long_ratio >= 0.50:
            ls_proxy = 0.80
            label = "mild long bias -- neutral"
        elif long_ratio >= 0.45:
            ls_proxy = 0.90
            label = "balanced -- healthy"
        else:
            ls_proxy = 1.00
            label = "too many shorts -- bullish contrarian"

        notes.append(
            f"whale.ls_proxy = {ls_proxy:.2f}"
            f" (long_ratio: {long_ratio:.3f} -- {label})"
        )

    # ── Sub-score 3: OI trend proxy ───────────────────────────────────────────
    try:
        oi_rows = conn.execute(
            "SELECT open_interest FROM open_interest "
            "WHERE symbol='BTCUSDT' ORDER BY timestamp DESC LIMIT 14"
        ).fetchall()
        px_rows = conn.execute(
            "SELECT close FROM ohlcv "
            "WHERE symbol='BTCUSDT' ORDER BY open_time DESC LIMIT 14"
        ).fetchall()
    except Exception as exc:
        oi_rows = []
        px_rows = []
        logger.warning(f"open_interest/ohlcv read error ({exc}) -- oi_proxy fallback 0.85")

    if len(oi_rows) < 5 or len(px_rows) < 5:
        oi_proxy = 0.85
        logger.warning("open_interest has < 5 rows for BTCUSDT -- oi_proxy fallback 0.85")
        notes.append("whale.oi_proxy = 0.85 (insufficient data -- fallback)")
    else:
        oi_latest = float(oi_rows[0][0])
        oi_oldest = float(oi_rows[-1][0])
        oi_change = (oi_latest - oi_oldest) / oi_oldest if oi_oldest else 0.0

        px_latest = float(px_rows[0][0])
        px_oldest = float(px_rows[-1][0])
        px_change = (px_latest - px_oldest) / px_oldest if px_oldest else 0.0

        if oi_change > 0.10 and px_change > 0.05:
            oi_proxy = 1.00
            label = "conviction rally"
        elif oi_change > 0.10 and px_change < -0.05:
            oi_proxy = 0.60
            label = "bearish divergence"
        elif oi_change < -0.10 and px_change < -0.05:
            oi_proxy = 0.70
            label = "deleveraging"
        elif oi_change < -0.10 and px_change > 0.05:
            oi_proxy = 0.90
            label = "short squeeze"
        else:
            oi_proxy = 0.85
            label = "neutral"

        notes.append(
            f"whale.oi_proxy = {oi_proxy:.2f}"
            f" (OI {oi_change:+.1%}, price {px_change:+.1%} -- {label})"
        )

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = max(0.50, min(1.00, (funding_proxy + ls_proxy + oi_proxy) / 3.0))
    subscores = {
        "funding_proxy": round(funding_proxy, 4),
        "ls_proxy":      round(ls_proxy, 4),
        "oi_proxy":      round(oi_proxy, 4),
    }
    notes.append(f"whale.combined = {combined:.4f}")

    return combined, subscores, notes


# ─── public API ───────────────────────────────────────────────────────────────

def macro_score(conn) -> float:
    """Reads macro table.  Returns 0.0-1.0.  Falls back to 0.7 if no data."""
    score, _ = _macro_score(conn)
    return score


def whale_score(conn) -> float:
    """Binance on-chain proxy score.  Returns 0.0-1.0.  Falls back gracefully."""
    score, _, _ = _whale_score(conn)
    return score


def regime_score(conn) -> dict:
    """
    Returns dict with macro, whale, whale_subscores, combined, timestamp, notes.
    combined = macro * whale, clamped [0.2, 1.0].
    """
    macro, macro_notes       = _macro_score(conn)
    whale, subscores, whale_notes = _whale_score(conn)
    combined = max(0.2, min(1.0, macro * whale))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_notes = macro_notes + whale_notes + [
        f"combined = {macro:.3f} x {whale:.3f} = {combined:.3f}"
    ]

    logger.info(
        f"regime_score: macro={macro:.3f} whale={whale:.3f} "
        f"(fr={subscores['funding_proxy']:.2f} ls={subscores['ls_proxy']:.2f} "
        f"oi={subscores['oi_proxy']:.2f}) combined={combined:.3f}"
    )
    return {
        "macro":           macro,
        "whale":           whale,
        "whale_subscores": subscores,
        "combined":        combined,
        "timestamp":       ts,
        "notes":           all_notes,
    }
