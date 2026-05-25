#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T9C V2 — Streamlit Dashboard for Trend Paper Engine
===================================================

Adds:
- live Binance price for open positions
- unrealized paper PnL / R estimate
- chart for each open position:
    * recent 6H close price
    * entry price
    * current trailing/current stop
    * initial stop

Run:
    streamlit run dashboard_trend_t9c_V2.py

Dependencies:
    pip install streamlit pandas numpy ccxt
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

try:
    import ccxt
except Exception:
    ccxt = None


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "paper_trend_t9a"
CACHE_DIR = DATA_DIR / "ohlcv_cache"

HEALTH_JSON = DATA_DIR / "system_health_trend_t9a.json"
STATE_JSON = DATA_DIR / "trend_t9a_state.json"

EQUITY_CSV = DATA_DIR / "equity_trend_t9a.csv"
OPEN_POSITIONS_CSV = DATA_DIR / "open_positions_trend_t9a.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_trend_t9a.csv"
SIGNALS_CSV = DATA_DIR / "signals_trend_t9a.csv"
SKIPPED_CSV = DATA_DIR / "skipped_signals_trend_t9a.csv"

TIMEFRAME = "6h"
CHART_BARS = 120


# ============================================================
# HELPERS
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "-"


def parse_time_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def profit_factor(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 1e-12:
        return np.inf if gains > 0 else 0.0
    return gains / losses


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


@st.cache_resource(ttl=300)
def get_exchange():
    if ccxt is None:
        return None
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


@st.cache_data(ttl=60)
def fetch_live_price(symbol: str):
    exchange = get_exchange()
    if exchange is None:
        return None
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last") or ticker.get("close"))
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_recent_ohlcv(symbol: str, timeframe: str = TIMEFRAME, limit: int = CHART_BARS) -> pd.DataFrame:
    exchange = get_exchange()

    if exchange is not None:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df.sort_values("time").reset_index(drop=True)
        except Exception:
            pass

    cache_path = CACHE_DIR / f"{safe_symbol_name(symbol)}_{timeframe}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.tail(limit).reset_index(drop=True)

    return pd.DataFrame()


def calc_unrealized_r(row, live_price: float):
    try:
        side = str(row.get("side", "")).upper()
        entry = float(row.get("entry_price"))
        risk = float(row.get("initial_risk_per_unit"))
        if risk <= 0:
            return np.nan
        if side == "LONG":
            return (live_price - entry) / risk
        if side == "SHORT":
            return (entry - live_price) / risk
    except Exception:
        return np.nan
    return np.nan


def calc_unrealized_pnl(row, live_price: float):
    try:
        r = calc_unrealized_r(row, live_price)
        risk_amount = float(row.get("risk_amount_usdt", 0))
        return r * risk_amount
    except Exception:
        return np.nan


def build_position_chart_df(symbol: str, row: pd.Series, live_price):
    df = fetch_recent_ohlcv(symbol)
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df.tail(CHART_BARS)

    entry = float(row.get("entry_price", np.nan))
    current_stop = float(row.get("current_stop", np.nan))
    initial_stop = float(row.get("initial_stop", np.nan))

    chart_df = pd.DataFrame(index=df["time"])
    chart_df["close"] = pd.to_numeric(df["close"], errors="coerce").values
    chart_df["entry"] = entry
    chart_df["current_stop"] = current_stop
    chart_df["initial_stop"] = initial_stop

    if live_price is not None and len(chart_df) > 0:
        live_time = pd.Timestamp.utcnow()
        extra = pd.DataFrame(
            {
                "close": [float(live_price)],
                "entry": [entry],
                "current_stop": [current_stop],
                "initial_stop": [initial_stop],
            },
            index=[live_time],
        )
        chart_df = pd.concat([chart_df, extra])

    return chart_df


# ============================================================
# LOAD DATA
# ============================================================

st.set_page_config(
    page_title="Trend T9C Dashboard",
    layout="wide",
)

st.title("📈 Trend Following T9C V2 — Paper Live Dashboard")
st.caption("Binance paper engine · 6H · Donchian breakout · wide exit · closed-trade equity only")

health = read_json(HEALTH_JSON)
state = read_json(STATE_JSON)

equity = read_csv(EQUITY_CSV)
open_pos = read_csv(OPEN_POSITIONS_CSV)
closed = read_csv(CLOSED_TRADES_CSV)
signals = read_csv(SIGNALS_CSV)
skipped = read_csv(SKIPPED_CSV)

equity = parse_time_col(equity, "timestamp_utc")
signals = parse_time_col(signals, "timestamp_utc")
skipped = parse_time_col(skipped, "timestamp_utc")

if "exit_time" in closed.columns:
    closed = parse_time_col(closed, "exit_time")
if "entry_time" in closed.columns:
    closed = parse_time_col(closed, "entry_time")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=False)
refresh_seconds = st.sidebar.selectbox("Refresh interval", [30, 60, 120, 300], index=1)

if auto_refresh:
    st.sidebar.caption(f"Auto-refresh ogni {refresh_seconds}s")
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
        unsafe_allow_html=True,
    )

if st.sidebar.button("Manual refresh"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# SYSTEM HEALTH
# ============================================================

st.subheader("🟢 System Health")

col1, col2, col3, col4, col5 = st.columns(5)

status = health.get("status", "UNKNOWN")
last_run = state.get("last_run_utc", "-")
last_bar = health.get("last_closed_bar_time", state.get("last_closed_bar_time", "-"))

with col1:
    st.metric("Status", status)

with col2:
    st.metric("Open Positions", health.get("open_positions", len(open_pos)))

with col3:
    st.metric("Closed Equity", f"{fmt_num(health.get('closed_equity_usdt', state.get('closed_equity_usdt', 0)))} USDT")

with col4:
    st.metric("Drawdown", f"{fmt_num(health.get('drawdown_pct', state.get('drawdown_pct', 0)))} %")

with col5:
    st.metric("Portfolio Heat", f"{fmt_num(health.get('portfolio_heat_pct', 0))} %")

st.write(f"**Last run UTC:** {last_run}")
st.write(f"**Last closed 6H candle:** {last_bar}")

if health.get("errors_this_run"):
    st.error("Errors detected in last run:")
    st.write(health.get("errors_this_run"))

if health.get("status") == "KILL_SWITCH" or state.get("kill_switch_triggered"):
    st.error("KILL-SWITCH ACTIVE — no new paper entries should be accepted.")


# ============================================================
# EQUITY
# ============================================================

st.subheader("📊 Closed-Trade Equity")

if equity.empty:
    st.info("No equity data yet.")
else:
    eq_plot = equity.copy()
    if "timestamp_utc" in eq_plot.columns:
        eq_plot = eq_plot.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    if "closed_equity_usdt" in eq_plot.columns and "timestamp_utc" in eq_plot.columns:
        st.line_chart(eq_plot.set_index("timestamp_utc")["closed_equity_usdt"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Equity", f"{fmt_num(eq_plot['closed_equity_usdt'].iloc[-1])} USDT")
    with c2:
        st.metric("Latest DD", f"{fmt_num(eq_plot['drawdown_pct'].iloc[-1])} %")
    with c3:
        st.metric("Closed Trades", int(eq_plot["closed_trade_count"].iloc[-1]) if "closed_trade_count" in eq_plot.columns else len(closed))
    with c4:
        st.metric("Latest Heat", f"{fmt_num(eq_plot['portfolio_heat_pct'].iloc[-1])} %")


# ============================================================
# OPEN POSITIONS WITH LIVE PRICE
# ============================================================

st.subheader("📌 Open Positions — Live Price & Trailing Stop")

if open_pos.empty:
    st.info("No open positions.")
else:
    enriched = open_pos.copy()

    live_prices = []
    unrealized_rs = []
    unrealized_pnls = []
    distance_to_stop_pcts = []

    for _, row in enriched.iterrows():
        symbol = row.get("symbol")
        live = fetch_live_price(symbol)
        live_prices.append(live)

        if live is None:
            unrealized_rs.append(np.nan)
            unrealized_pnls.append(np.nan)
            distance_to_stop_pcts.append(np.nan)
            continue

        unr = calc_unrealized_r(row, live)
        unpnl = calc_unrealized_pnl(row, live)

        try:
            current_stop = float(row.get("current_stop"))
            side = str(row.get("side", "")).upper()
            if side == "LONG":
                dist = (live - current_stop) / live * 100
            else:
                dist = (current_stop - live) / live * 100
        except Exception:
            dist = np.nan

        unrealized_rs.append(unr)
        unrealized_pnls.append(unpnl)
        distance_to_stop_pcts.append(dist)

    enriched["live_price"] = live_prices
    enriched["unrealized_R"] = unrealized_rs
    enriched["unrealized_pnl_usdt"] = unrealized_pnls
    enriched["distance_to_stop_pct"] = distance_to_stop_pcts

    show_cols = [
        "symbol", "side", "timeframe", "entry_time",
        "entry_price", "live_price", "current_stop", "initial_stop",
        "unrealized_R", "unrealized_pnl_usdt", "distance_to_stop_pct",
        "max_favorable_r", "chandelier_active",
        "risk_amount_usdt", "notional_usdt", "margin_reserved_usdt", "qty"
    ]
    available_cols = [c for c in show_cols if c in enriched.columns]
    st.dataframe(enriched[available_cols], use_container_width=True)

    st.markdown("### 📉 Position Charts")

    for _, row in enriched.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        live = row.get("live_price")

        with st.expander(f"{symbol} — {side} — live {fmt_num(live, 8)}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Live Price", fmt_num(live, 8))
            with c2:
                st.metric("Unrealized R", fmt_num(row.get("unrealized_R"), 2))
            with c3:
                st.metric("Unrealized PnL", f"{fmt_num(row.get('unrealized_pnl_usdt'), 2)} USDT")
            with c4:
                st.metric("Distance to Stop", f"{fmt_num(row.get('distance_to_stop_pct'), 2)} %")

            chart_df = build_position_chart_df(symbol, row, live)
            if chart_df.empty:
                st.warning("No chart data available for this position.")
            else:
                st.line_chart(chart_df[["close", "entry", "current_stop", "initial_stop"]])

            st.caption(
                "Il grafico usa close 6H recenti + prezzo live indicativo. "
                "Entry, initial stop e current/trailing stop sono linee orizzontali allo stato attuale."
            )


# ============================================================
# CLOSED TRADES
# ============================================================

st.subheader("✅ Closed Trades")

if closed.empty:
    st.info("No closed trades yet.")
else:
    r_col = "net_r" if "net_r" in closed.columns else "gross_r"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Closed Trades", len(closed))
    with c2:
        st.metric("Total R", fmt_num(pd.to_numeric(closed[r_col], errors="coerce").sum()))
    with c3:
        st.metric("Avg R", fmt_num(pd.to_numeric(closed[r_col], errors="coerce").mean()))
    with c4:
        st.metric("PF", fmt_num(profit_factor(closed[r_col])))
    with c5:
        st.metric("Win Rate", f"{fmt_num((pd.to_numeric(closed[r_col], errors='coerce') > 0).mean() * 100)} %")

    if "exit_time" in closed.columns and r_col in closed.columns:
        closed_sorted = closed.sort_values("exit_time").copy()
        closed_sorted["cum_r"] = pd.to_numeric(closed_sorted[r_col], errors="coerce").fillna(0).cumsum()
        st.line_chart(closed_sorted.set_index("exit_time")["cum_r"])

    show_cols = [
        "symbol", "side", "entry_time", "exit_time",
        "entry_price", "exit_price", "gross_r", "net_r",
        "pnl_usdt", "exit_reason", "max_favorable_r",
        "chandelier_active"
    ]
    available_cols = [c for c in show_cols if c in closed.columns]
    st.dataframe(closed[available_cols].tail(100).iloc[::-1], use_container_width=True)


# ============================================================
# SIGNALS
# ============================================================

st.subheader("📡 Signals")

if signals.empty:
    st.info("No signals logged yet.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Signals logged", len(signals))
    with c2:
        if "side" in signals.columns:
            st.metric("Long signals", int((signals["side"].str.upper() == "LONG").sum()))
    with c3:
        if "side" in signals.columns:
            st.metric("Short signals", int((signals["side"].str.upper() == "SHORT").sum()))

    show_cols = [
        "timestamp_utc", "symbol", "side", "bar_time",
        "entry_price", "initial_stop", "atr", "ema50_slope",
        "signal"
    ]
    available_cols = [c for c in show_cols if c in signals.columns]
    st.dataframe(signals[available_cols].tail(100).iloc[::-1], use_container_width=True)


# ============================================================
# SKIPPED SIGNALS
# ============================================================

st.subheader("⏭️ Skipped Signals")

if skipped.empty:
    st.info("No skipped signals.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Skipped total", len(skipped))
    with c2:
        if "skip_reason" in skipped.columns and not skipped["skip_reason"].empty:
            st.metric("Most common reason", skipped["skip_reason"].mode().iloc[0])

    if "skip_reason" in skipped.columns:
        st.bar_chart(skipped["skip_reason"].value_counts())

    show_cols = [
        "timestamp_utc", "symbol", "side", "bar_time",
        "skip_reason", "closed_equity_usdt",
        "open_positions", "portfolio_heat_pct"
    ]
    available_cols = [c for c in show_cols if c in skipped.columns]
    st.dataframe(skipped[available_cols].tail(100).iloc[::-1], use_container_width=True)


# ============================================================
# RAW FILE STATUS
# ============================================================

st.subheader("🗂️ File Status")

file_rows = []
for label, path in [
    ("Health", HEALTH_JSON),
    ("State", STATE_JSON),
    ("Equity", EQUITY_CSV),
    ("Open Positions", OPEN_POSITIONS_CSV),
    ("Closed Trades", CLOSED_TRADES_CSV),
    ("Signals", SIGNALS_CSV),
    ("Skipped", SKIPPED_CSV),
]:
    file_rows.append({
        "file": label,
        "path": str(path),
        "exists": path.exists(),
        "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else 0,
    })

st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

st.caption("T9C V2 legge i CSV/JSON locali prodotti da T9A e usa Binance solo per prezzo live/grafici. Non invia ordini.")
