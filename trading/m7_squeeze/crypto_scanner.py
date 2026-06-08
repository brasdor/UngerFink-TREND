"""
M7 Crypto Scanner
=================
Four independent signal scanners that source data from M1 tables.
Each function is defensive: returns an empty DataFrame on missing data or errors,
never crashes the daily pipeline.

Funding rates stored as decimal fractions: 0.0001 = 0.01%  (Binance convention).
Thresholds in docstrings are written as percentages for readability.
"""
import os
import sys

import pandas as pd
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


# ── shared helper ──────────────────────────────────────────────────────────────

def _universe_symbols(conn) -> list[str]:
    """All symbols from the latest universe_history snapshot."""
    row = conn.execute("SELECT MAX(date) FROM universe_history").fetchone()
    if not row or not row[0]:
        return []
    rows = conn.execute(
        "SELECT symbol FROM universe_history WHERE date = ? ORDER BY quote_volume DESC",
        (row[0],),
    ).fetchall()
    return [r[0] for r in rows]


# ── signal 1: funding rate ─────────────────────────────────────────────────────

def scan_funding_signal(conn) -> pd.DataFrame:
    """
    For each symbol: avg funding rate over the last 21 rows (~7 days × 3/day).

    Score thresholds (rate in decimal, e.g. -0.000100 == -0.0100%):
      avg_rate < -0.000100  → 40 pts   extreme short bias
      avg_rate < -0.000050  → 25 pts
      avg_rate < -0.000010  → 10 pts
      avg_rate >= -0.000010 →  0 pts

    Returns: DataFrame(symbol, funding_score, avg_funding_rate)
    """
    empty = pd.DataFrame(columns=["symbol", "funding_score", "avg_funding_rate"])
    try:
        symbols = _universe_symbols(conn)
        if not symbols:
            return empty

        ph = ",".join("?" * len(symbols))
        df = pd.read_sql_query(
            f"""
            SELECT symbol, funding_rate
            FROM (
                SELECT symbol, funding_rate,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY funding_time DESC) AS rn
                FROM funding_rates
                WHERE symbol IN ({ph})
            )
            WHERE rn <= 21
            """,
            conn, params=symbols,
        )
        if df.empty:
            return empty

        agg = df.groupby("symbol")["funding_rate"].mean().reset_index()
        agg.columns = ["symbol", "avg_funding_rate"]

        def _score(r: float) -> int:
            if r < -0.000100:
                return 40
            if r < -0.000050:
                return 25
            if r < -0.000010:
                return 10
            return 0

        agg["funding_score"] = agg["avg_funding_rate"].apply(_score)
        return agg[["symbol", "funding_score", "avg_funding_rate"]]
    except Exception as exc:
        logger.warning(f"scan_funding_signal failed: {exc}")
        return empty


# ── signal 2: long/short ratio ─────────────────────────────────────────────────

def scan_ls_signal(conn) -> pd.DataFrame:
    """
    Latest long/short ratio per symbol.

    Score thresholds:
      long_ratio < 0.35 → 30 pts   extreme short dominance
      long_ratio < 0.40 → 20 pts
      long_ratio < 0.45 → 10 pts
      long_ratio >= 0.45 →  0 pts

    Returns: DataFrame(symbol, ls_score, long_ratio)
    """
    empty = pd.DataFrame(columns=["symbol", "ls_score", "long_ratio"])
    try:
        symbols = _universe_symbols(conn)
        if not symbols:
            return empty

        ph = ",".join("?" * len(symbols))
        df = pd.read_sql_query(
            f"""
            SELECT symbol, long_ratio
            FROM (
                SELECT symbol, long_ratio,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM ls_ratio
                WHERE symbol IN ({ph})
            )
            WHERE rn = 1
            """,
            conn, params=symbols,
        )
        if df.empty:
            return empty

        def _score(r: float) -> int:
            if r < 0.35:
                return 30
            if r < 0.40:
                return 20
            if r < 0.45:
                return 10
            return 0

        df = df.copy()
        df["ls_score"] = df["long_ratio"].apply(_score)
        return df[["symbol", "ls_score", "long_ratio"]]
    except Exception as exc:
        logger.warning(f"scan_ls_signal failed: {exc}")
        return empty


# ── signal 3: open interest change ────────────────────────────────────────────

def scan_oi_signal(conn) -> pd.DataFrame:
    """
    OI change over the last 14 rows (14 daily snapshots from M1).

    Symbols with fewer than 14 OI rows are skipped (oi_score=0).

    Score:
      oi_change > +20% AND price flat/down → 30 pts  shorts building into resistance
      oi_change > +10% AND price flat/down → 15 pts
      all other                            →  0 pts

    Caution flag (squeeze may already be starting):
      oi_change > +15% AND price rising    → caution_flag=1

    Returns: DataFrame(symbol, oi_score, oi_change_pct, caution_flag)
    """
    empty = pd.DataFrame(columns=["symbol", "oi_score", "oi_change_pct", "caution_flag"])
    try:
        symbols = _universe_symbols(conn)
        if not symbols:
            return empty

        ph = ",".join("?" * len(symbols))

        # Rank OI rows per symbol newest-first
        oi_ranked = pd.read_sql_query(
            f"""
            SELECT symbol, open_interest,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
            FROM open_interest
            WHERE symbol IN ({ph})
            """,
            conn, params=symbols,
        )
        if oi_ranked.empty:
            return empty

        # Only score symbols with ≥ 14 OI rows
        row_counts = oi_ranked.groupby("symbol")["rn"].max()
        valid_syms = row_counts[row_counts >= 14].index.tolist()
        short_syms = row_counts[row_counts < 14].index.tolist()
        if short_syms:
            logger.debug(f"OI scorer: {len(short_syms)} symbol(s) skipped (<14 rows)")

        if not valid_syms:
            # Return all as 0-score
            all_known = row_counts.index.tolist()
            return pd.DataFrame({
                "symbol":       all_known,
                "oi_score":     0,
                "oi_change_pct": float("nan"),
                "caution_flag": 0,
            })

        oi_latest = (
            oi_ranked[oi_ranked["rn"] == 1][["symbol", "open_interest"]]
            .rename(columns={"open_interest": "oi_latest"})
        )
        oi_old = (
            oi_ranked[oi_ranked["rn"] == 14][["symbol", "open_interest"]]
            .rename(columns={"open_interest": "oi_14d_ago"})
        )
        oi_m = oi_latest.merge(oi_old, on="symbol")
        oi_m = oi_m[oi_m["symbol"].isin(valid_syms)].copy()
        oi_m["oi_change_pct"] = (
            (oi_m["oi_latest"] - oi_m["oi_14d_ago"])
            / oi_m["oi_14d_ago"].replace(0.0, float("nan"))
        )
        oi_m = oi_m.dropna(subset=["oi_change_pct"])

        # Fetch latest and 14th-most-recent close price for valid symbols
        ph2 = ",".join("?" * len(valid_syms))
        price_ranked = pd.read_sql_query(
            f"""
            SELECT symbol, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) AS rn
            FROM ohlcv
            WHERE symbol IN ({ph2})
            """,
            conn, params=valid_syms,
        )
        p_now = (
            price_ranked[price_ranked["rn"] == 1][["symbol", "close"]]
            .rename(columns={"close": "price_now"})
        )
        p_old = (
            price_ranked[price_ranked["rn"] == 14][["symbol", "close"]]
            .rename(columns={"close": "price_14d_ago"})
        )
        price_m = p_now.merge(p_old, on="symbol", how="left")
        # If fewer than 14 OHLCV rows available, treat price as flat (conservative)
        price_m["price_flat_or_down"] = (
            price_m["price_now"] <= price_m["price_14d_ago"].fillna(price_m["price_now"])
        )

        final = oi_m.merge(
            price_m[["symbol", "price_flat_or_down"]], on="symbol", how="left"
        )
        final["price_flat_or_down"] = final["price_flat_or_down"].fillna(True)

        def _oi_score(row) -> int:
            chg  = row["oi_change_pct"]
            flat = bool(row["price_flat_or_down"])
            if chg > 0.20 and flat:
                return 30
            if chg > 0.10 and flat:
                return 15
            return 0

        def _caution(row) -> int:
            return 1 if (row["oi_change_pct"] > 0.15 and not bool(row["price_flat_or_down"])) else 0

        final["oi_score"]    = final.apply(_oi_score, axis=1)
        final["caution_flag"] = final.apply(_caution, axis=1)

        result = final[["symbol", "oi_score", "oi_change_pct", "caution_flag"]].copy()

        # Append skipped (< 14 rows) as 0-score entries
        if short_syms:
            extras = pd.DataFrame({
                "symbol":       short_syms,
                "oi_score":     0,
                "oi_change_pct": float("nan"),
                "caution_flag": 0,
            })
            result = pd.concat([result, extras], ignore_index=True)

        return result
    except Exception as exc:
        logger.warning(f"scan_oi_signal failed: {exc}")
        return empty


# ── signal 4: volume spike ────────────────────────────────────────────────────

def scan_volume_spike(conn) -> pd.DataFrame:
    """
    Today's quote_volume vs the 30-candle rolling average.

    Score:
      volume > 3× avg → 10 pts
      volume > 2× avg →  5 pts
      volume ≤ 2× avg →  0 pts

    Returns: DataFrame(symbol, volume_score, volume_ratio)
    """
    empty = pd.DataFrame(columns=["symbol", "volume_score", "volume_ratio"])
    try:
        symbols = _universe_symbols(conn)
        if not symbols:
            return empty

        ph = ",".join("?" * len(symbols))

        latest_df = pd.read_sql_query(
            f"""
            SELECT symbol, quote_volume AS latest_vol
            FROM (
                SELECT symbol, quote_volume,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) AS rn
                FROM ohlcv WHERE symbol IN ({ph})
            )
            WHERE rn = 1
            """,
            conn, params=symbols,
        )
        avg_df = pd.read_sql_query(
            f"""
            SELECT symbol, AVG(quote_volume) AS avg_vol_30d
            FROM (
                SELECT symbol, quote_volume,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) AS rn
                FROM ohlcv WHERE symbol IN ({ph})
            )
            WHERE rn <= 30
            GROUP BY symbol
            """,
            conn, params=symbols,
        )
        if latest_df.empty or avg_df.empty:
            return empty

        merged = latest_df.merge(avg_df, on="symbol")
        merged["volume_ratio"] = (
            merged["latest_vol"] / merged["avg_vol_30d"].replace(0.0, float("nan"))
        )
        merged = merged.dropna(subset=["volume_ratio"])

        def _score(r: float) -> int:
            if r > 3.0:
                return 10
            if r > 2.0:
                return 5
            return 0

        merged["volume_score"] = merged["volume_ratio"].apply(_score)
        return merged[["symbol", "volume_score", "volume_ratio"]]
    except Exception as exc:
        logger.warning(f"scan_volume_spike failed: {exc}")
        return empty
