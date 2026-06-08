"""
M7 Squeeze Scorer
=================
Combines the four M7 scanner signals into a composite squeeze score 0-100.

Filters applied before returning candidates:
  1. Exclude symbols already open in paper_trades (factor OR squeeze)
  2. Exclude symbols with 30d avg quote_volume < $10M
  3. Return empty if M4 circuit breaker state == 'stopped'

Thresholds:
  score >= 85  HIGH_CONVICTION  (Telegram alert + auto-open)
  score >= 70  CANDIDATE        (watch list)
  score <  70  WATCH            (logged only)
"""
import os
import sys

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from m7_squeeze.crypto_scanner import (
    scan_funding_signal,
    scan_ls_signal,
    scan_oi_signal,
    scan_volume_spike,
)

_MIN_VOL_30D = 10_000_000   # $10M USDT per day


# ── composite scorer ───────────────────────────────────────────────────────────

def compute_squeeze_scores(conn) -> pd.DataFrame:
    """
    Returns DataFrame sorted by total_score descending.

    Columns: symbol, total_score, funding_score, ls_score, oi_score,
             volume_score, threshold, avg_funding_rate, long_ratio,
             oi_change_pct, caution_flag
    """
    # Circuit breaker gate — no new squeeze trades when STOPPED
    cb_row = conn.execute(
        "SELECT cb_state FROM risk_state ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    cb_stopped = cb_row and cb_row[0] == "stopped"
    if cb_stopped:
        logger.warning("M4 circuit breaker STOPPED — squeeze scanner blocked from opening new trades")

    # Universe base
    latest_date = conn.execute("SELECT MAX(date) FROM universe_history").fetchone()[0]
    if not latest_date:
        logger.warning("compute_squeeze_scores: no universe snapshot found")
        return pd.DataFrame()

    universe = pd.read_sql_query(
        "SELECT symbol FROM universe_history WHERE date = ?",
        conn, params=[latest_date],
    )
    if universe.empty:
        return pd.DataFrame()

    # Run all four scanners
    funding_df = scan_funding_signal(conn)
    ls_df      = scan_ls_signal(conn)
    oi_df      = scan_oi_signal(conn)
    volume_df  = scan_volume_spike(conn)

    base = universe[["symbol"]].copy()

    # Merge each signal; fill gaps with 0
    for sig_df, score_col in [
        (funding_df, "funding_score"),
        (ls_df,      "ls_score"),
        (oi_df,      "oi_score"),
        (volume_df,  "volume_score"),
    ]:
        if not sig_df.empty and score_col in sig_df.columns:
            base = base.merge(sig_df, on="symbol", how="left")
        if score_col not in base.columns:
            base[score_col] = 0.0

    for col in ["funding_score", "ls_score", "oi_score", "volume_score"]:
        base[col] = base[col].fillna(0.0)

    # Pull through extra detail columns
    for col, default in [
        ("avg_funding_rate", float("nan")),
        ("long_ratio",       float("nan")),
        ("oi_change_pct",    float("nan")),
        ("caution_flag",     0),
    ]:
        if col not in base.columns:
            base[col] = default
        else:
            base[col] = base[col].fillna(default)
    base["caution_flag"] = base["caution_flag"].astype(int)

    # ── filter 1: exclude all open paper_trades (factor + squeeze) ────────────
    try:
        open_syms = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM paper_trades WHERE status='open'"
            ).fetchall()
        }
    except Exception:
        open_syms = set()
    base = base[~base["symbol"].isin(open_syms)].copy()

    # ── filter 2: exclude low-volume symbols ──────────────────────────────────
    # Compute 30d avg quote_volume from ohlcv for remaining symbols
    if not base.empty:
        syms = base["symbol"].tolist()
        ph = ",".join("?" * len(syms))
        vol_rows = conn.execute(
            f"""
            SELECT symbol, AVG(quote_volume) AS avg_vol
            FROM (
                SELECT symbol, quote_volume,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) AS rn
                FROM ohlcv WHERE symbol IN ({ph})
            )
            WHERE rn <= 30
            GROUP BY symbol
            """,
            syms,
        ).fetchall()
        vol_map = {r[0]: float(r[1]) for r in vol_rows}
        base["avg_vol_30d"] = base["symbol"].map(vol_map).fillna(0.0)
        base = base[base["avg_vol_30d"] >= _MIN_VOL_30D].drop(columns=["avg_vol_30d"])

    # ── filter 3: CB STOPPED → return empty (no candidates) ──────────────────
    if cb_stopped:
        return pd.DataFrame(columns=[
            "symbol", "total_score", "funding_score", "ls_score", "oi_score",
            "volume_score", "threshold", "avg_funding_rate", "long_ratio",
            "oi_change_pct", "caution_flag",
        ])

    # Compute total (cap at 100)
    base["total_score"] = (
        base["funding_score"] + base["ls_score"] + base["oi_score"] + base["volume_score"]
    ).clip(upper=100.0)

    def _threshold(s: float) -> str:
        if s >= 85:
            return "HIGH_CONVICTION"
        if s >= 70:
            return "CANDIDATE"
        return "WATCH"

    base["threshold"] = base["total_score"].apply(_threshold)

    ordered_cols = [
        "symbol", "total_score", "funding_score", "ls_score", "oi_score",
        "volume_score", "threshold", "avg_funding_rate", "long_ratio",
        "oi_change_pct", "caution_flag",
    ]
    avail = [c for c in ordered_cols if c in base.columns]
    return base[avail].sort_values("total_score", ascending=False).reset_index(drop=True)


# ── per-symbol re-scorer (used by executor for score-decay exit check) ─────────

def rescore_symbol(symbol: str, conn) -> float:
    """
    Returns raw squeeze score for a single symbol, ignoring all exclusion filters.
    Used by executor.check_exits for score-decay exit condition.
    Expensive (runs all 4 scanners) but called at most MAX_POSITIONS times/day.
    """
    total = 0.0
    for scan_fn, score_col in [
        (scan_funding_signal, "funding_score"),
        (scan_ls_signal,      "ls_score"),
        (scan_oi_signal,      "oi_score"),
        (scan_volume_spike,   "volume_score"),
    ]:
        try:
            df = scan_fn(conn)
            if not df.empty and symbol in df["symbol"].values:
                val = df.loc[df["symbol"] == symbol, score_col].iloc[0]
                total += float(val) if pd.notna(val) else 0.0
        except Exception as exc:
            logger.debug(f"rescore_symbol {symbol} {score_col}: {exc}")
    return min(total, 100.0)
