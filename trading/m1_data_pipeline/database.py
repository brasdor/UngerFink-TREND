import sqlite3
import os
import pandas as pd
from contextlib import contextmanager
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH


def _resolve_db_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, DB_PATH)


@contextmanager
def get_conn():
    path = _resolve_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS universe_history (
                date            TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                base_asset      TEXT,
                quote_volume    REAL,
                price_precision INT,
                qty_precision   INT,
                min_qty         REAL,
                tick_size       REAL,
                snapshot_type   TEXT    NOT NULL DEFAULT 'daily',
                PRIMARY KEY (date, symbol)
            );

            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol       TEXT    NOT NULL,
                open_time    INT     NOT NULL,
                open         REAL,
                high         REAL,
                low          REAL,
                close        REAL,
                volume       REAL,
                quote_volume REAL,
                trades       INT,
                PRIMARY KEY (symbol, open_time)
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
                ON ohlcv (symbol, open_time);

            CREATE TABLE IF NOT EXISTS funding_rates (
                symbol       TEXT    NOT NULL,
                funding_time INT     NOT NULL,
                funding_rate REAL,
                PRIMARY KEY (symbol, funding_time)
            );

            CREATE TABLE IF NOT EXISTS open_interest (
                symbol        TEXT    NOT NULL,
                timestamp     INT     NOT NULL,
                open_interest REAL,
                PRIMARY KEY (symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS ls_ratio (
                symbol      TEXT    NOT NULL,
                timestamp   INT     NOT NULL,
                long_ratio  REAL,
                short_ratio REAL,
                PRIMARY KEY (symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS macro (
                series_id TEXT    NOT NULL,
                date      TEXT    NOT NULL,
                value     REAL,
                PRIMARY KEY (series_id, date)
            );

            CREATE TABLE IF NOT EXISTS onchain (
                metric    TEXT    NOT NULL,
                timestamp INT     NOT NULL,
                value     REAL,
                PRIMARY KEY (metric, timestamp)
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time      INT     NOT NULL,
                mode          TEXT    NOT NULL,
                symbols_ok    INT     DEFAULT 0,
                symbols_err   INT     DEFAULT 0,
                rows_inserted INT     DEFAULT 0,
                notes         TEXT
            );
        """)


def upsert(table: str, rows: list[dict]) -> int:
    """Insert or replace rows into table. Returns number of rows written."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
    values = [tuple(r[c] for c in cols) for r in rows]
    with get_conn() as conn:
        conn.executemany(sql, values)
    return len(values)


def read_ohlcv(symbol: str, start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
    """Return OHLCV rows for symbol between UTC-ms timestamps start..end (inclusive)."""
    clauses = ["symbol = ?"]
    params: list = [symbol]
    if start is not None:
        clauses.append("open_time >= ?")
        params.append(start)
    if end is not None:
        clauses.append("open_time <= ?")
        params.append(end)
    where = " AND ".join(clauses)
    sql = f"SELECT * FROM ohlcv WHERE {where} ORDER BY open_time"
    with get_conn() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def read_universe(date: str) -> pd.DataFrame:
    """Return universe snapshot for a given date string (YYYY-MM-DD)."""
    sql = "SELECT * FROM universe_history WHERE date = ? ORDER BY quote_volume DESC"
    with get_conn() as conn:
        df = pd.read_sql_query(sql, conn, params=[date])
    return df


def get_last_date(
    table: str,
    symbol: Optional[str] = None,
    date_col: str = "open_time",
    as_text: bool = False,
) -> Optional[int | str]:
    """
    Return the MAX value of date_col for the given table/symbol.
    Returns an int (UTC-ms) by default, or a TEXT string when as_text=True.
    Returns None if the table is empty or no matching rows exist.

    Callers must supply the correct date_col for their table:
        ohlcv          → date_col='open_time'   (default)
        funding_rates  → date_col='funding_time'
        open_interest  → date_col='timestamp'
        ls_ratio       → date_col='timestamp'
        onchain        → date_col='timestamp'
        macro          → date_col='date', as_text=True
    """
    symbol_col = "metric" if table == "onchain" else "symbol"

    with get_conn() as conn:
        if symbol:
            row = conn.execute(
                f"SELECT MAX({date_col}) FROM {table} WHERE {symbol_col} = ?", (symbol,)
            ).fetchone()
        else:
            row = conn.execute(f"SELECT MAX({date_col}) FROM {table}").fetchone()

    val = row[0] if row and row[0] is not None else None
    if val is None:
        return None
    return str(val) if as_text else int(val)


def log_pipeline_run(
    mode: str,
    symbols_ok: int = 0,
    symbols_err: int = 0,
    rows_inserted: int = 0,
    notes: Optional[str] = None,
) -> int:
    """Insert a pipeline_runs record. Returns the new run_id."""
    import time as _time
    run_time = int(_time.time() * 1000)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pipeline_runs (run_time, mode, symbols_ok, symbols_err, rows_inserted, notes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_time, mode, symbols_ok, symbols_err, rows_inserted, notes),
        )
    return cur.lastrowid
