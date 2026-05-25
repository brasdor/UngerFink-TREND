#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TREND T9 — Recovery-Synced Candlestick Dashboard
================================================

Compatible with:
- T9A / T9A V2 Binance Paper Sim Engine
- T9D Trend Recovery & Reconciliation Engine

Run:
    streamlit run dashboard_trend_t9_RECOVERY_SYNC.py

Dependencies:
    pip install streamlit pandas numpy plotly ccxt
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import ccxt
except Exception:
    ccxt = None


ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "paper_trend_t9a"
CANDLES_DIR = DATA_DIR / "ohlcv_cache"

STATE_JSON = DATA_DIR / "trend_t9a_state.json"
HEALTH_JSON = DATA_DIR / "system_health_trend_t9a.json"

OPEN_POSITIONS_CSV = DATA_DIR / "open_positions_trend_t9a.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_trend_t9a.csv"
EQUITY_CSV = DATA_DIR / "equity_trend_t9a.csv"
SIGNALS_CSV = DATA_DIR / "signals_trend_t9a.csv"
SKIPPED_CSV = DATA_DIR / "skipped_signals_trend_t9a.csv"

RECOVERY_EVENTS_CSV = DATA_DIR / "phase_t9d_recovery_events.csv"
RECOVERY_REPORT_CSV = DATA_DIR / "phase_t9d_recovery_report.csv"

TIMEFRAME = "6h"
CHART_BARS = 180
INITIAL_CAPITAL_USDT = 10_000.0


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def read_json_safe(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def fmt(x, n=2):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.{n}f}"
    except Exception:
        return "-"


def safe_symbol_file(symbol: str):
    return str(symbol).replace("/", "_").replace(":", "_")


def profit_factor(values):
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return 0.0
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 1e-12:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def max_dd_from_cum(cum):
    if len(cum) == 0:
        return 0.0
    arr = np.asarray(cum, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = arr - peak
    return float(dd.min())


def parse_time(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def candle_file_for_symbol(symbol: str):
    base = safe_symbol_file(symbol)
    candidates = [
        CANDLES_DIR / f"{base}_{TIMEFRAME}.csv",
        CANDLES_DIR / f"{base}_6h.csv",
        CANDLES_DIR / f"{base}_6H.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(CANDLES_DIR.glob(f"*{base}*.csv"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return None


@st.cache_resource(ttl=300)
def get_exchange():
    if ccxt is None:
        return None
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


@st.cache_data(ttl=60)
def fetch_live_price(symbol: str):
    ex = get_exchange()
    if ex is None:
        return None
    try:
        ticker = ex.fetch_ticker(symbol)
        return float(ticker.get("last") or ticker.get("close"))
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_recent_ohlcv(symbol: str, limit: int = CHART_BARS):
    ex = get_exchange()
    if ex is not None:
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df.sort_values("time").reset_index(drop=True)
        except Exception:
            pass

    p = candle_file_for_symbol(symbol)
    if p is None:
        return pd.DataFrame()
    df = read_csv_safe(p)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df.tail(limit).reset_index(drop=True)


def calc_unrealized_r(row, live_price):
    try:
        side = str(row.get("side", "")).upper()
        entry = float(row.get("entry_price"))
        risk = float(row.get("initial_risk_per_unit"))
        if risk <= 0 or live_price is None:
            return np.nan
        if side == "LONG":
            return (float(live_price) - entry) / risk
        if side == "SHORT":
            return (entry - float(live_price)) / risk
    except Exception:
        return np.nan
    return np.nan


def calc_unrealized_pnl(row, live_price):
    try:
        r = calc_unrealized_r(row, live_price)
        risk_amount = float(row.get("risk_amount_usdt", 0))
        return r * risk_amount
    except Exception:
        return np.nan


def build_equity_events(closed: pd.DataFrame, recovery_events: pd.DataFrame) -> pd.DataFrame:
    events = []

    if not closed.empty:
        c = closed.copy()
        r_col = None
        for col in ["net_r", "gross_r", "realized_r", "r", "R"]:
            if col in c.columns:
                r_col = col
                break
        time_col = None
        for col in ["exit_time", "bar_time", "timestamp_utc", "timestamp"]:
            if col in c.columns:
                time_col = col
                break
        if r_col:
            c["r"] = pd.to_numeric(c[r_col], errors="coerce")
            c["time"] = pd.to_datetime(c[time_col], errors="coerce", utc=True) if time_col else pd.NaT
            c["source"] = "CLOSED_TRADES"
            events.append(c)

    if not recovery_events.empty:
        r = recovery_events.copy()
        if "event_type" in r.columns:
            r = r[r["event_type"].astype(str).str.contains("POSITION_CLOSED_RECOVERY", case=False, na=False)].copy()
        if "realized_r" in r.columns:
            r["r"] = pd.to_numeric(r["realized_r"], errors="coerce")
            if "bar_time" in r.columns:
                r["time"] = pd.to_datetime(r["bar_time"], errors="coerce", utc=True)
            elif "timestamp_utc" in r.columns:
                r["time"] = pd.to_datetime(r["timestamp_utc"], errors="coerce", utc=True)
            else:
                r["time"] = pd.NaT
            r["source"] = "T9D_RECOVERY"
            events.append(r)

    if not events:
        return pd.DataFrame()

    out = pd.concat(events, ignore_index=True, sort=False)
    out = out.dropna(subset=["r"]).copy()

    if "position_id" in out.columns:
        out = out.drop_duplicates(subset=["position_id", "r"], keep="last")

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
    out = out.sort_values("time").reset_index(drop=True)

    out["cum_r"] = out["r"].cumsum()
    out["peak_r"] = out["cum_r"].cummax()
    out["dd_r"] = out["cum_r"] - out["peak_r"]
    return out


def build_candlestick_figure(symbol, row, live_price=None):
    if go is None:
        return None

    df = fetch_recent_ohlcv(symbol)
    if df.empty:
        return None

    if "timestamp" in df.columns and "time" not in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).tail(CHART_BARS)
    if df.empty:
        return None

    entry = float(row.get("entry_price", np.nan))
    current_stop = float(row.get("current_stop", np.nan))
    initial_stop = float(row.get("initial_stop", np.nan))
    side = str(row.get("side", "")).upper()

    entry_time = pd.to_datetime(row.get("entry_time"), errors="coerce", utc=True)
    if pd.isna(entry_time):
        entry_time = pd.to_datetime(row.get("entry_bar_time"), errors="coerce", utc=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="6H candles",
    ))

    if np.isfinite(entry):
        fig.add_hline(y=entry, line_dash="solid", annotation_text="ENTRY", annotation_position="top left")
        if not pd.isna(entry_time):
            idx = (df["time"] - entry_time).abs().idxmin()
            marker_time = df.loc[idx, "time"]
            fig.add_trace(go.Scatter(
                x=[marker_time],
                y=[entry],
                mode="markers+text",
                marker=dict(size=13, symbol="triangle-up" if side == "LONG" else "triangle-down"),
                text=["ENTRY"],
                textposition="top center" if side == "LONG" else "bottom center",
                name="Entry marker",
            ))

    if np.isfinite(initial_stop):
        fig.add_hline(y=initial_stop, line_dash="dot", annotation_text="Initial stop", annotation_position="bottom left")

    if np.isfinite(current_stop):
        fig.add_hline(y=current_stop, line_dash="dash", annotation_text="Current / trailing stop", annotation_position="bottom right")

    if live_price is not None:
        try:
            live_price = float(live_price)
            if np.isfinite(live_price):
                fig.add_hline(y=live_price, line_dash="dashdot", annotation_text="Live price", annotation_position="top right")
        except Exception:
            pass

    active = str(row.get("chandelier_active", "False")).lower() in ["true", "1", "yes"]
    act_time = pd.to_datetime(row.get("chandelier_activation_time"), errors="coerce", utc=True)
    act_price = row.get("chandelier_activation_price")
    try:
        act_price = float(act_price)
    except Exception:
        act_price = np.nan

    if active:
        if not pd.isna(act_time) and np.isfinite(act_price):
            idx = (df["time"] - act_time).abs().idxmin()
            marker_time = df.loc[idx, "time"]
            fig.add_trace(go.Scatter(
                x=[marker_time],
                y=[act_price],
                mode="markers+text",
                marker=dict(size=14, symbol="star"),
                text=["TRAIL ON"],
                textposition="top center",
                name="Chandelier activation",
            ))
        else:
            fig.add_annotation(
                x=df["time"].iloc[-1],
                y=current_stop if np.isfinite(current_stop) else df["close"].iloc[-1],
                text="TRAILING ACTIVE<br>activation time not available",
                showarrow=True,
            )

    fig.update_layout(
        title=f"{symbol} — {side} — {TIMEFRAME}",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


# ============================================================
# LOAD DATA
# ============================================================

st.set_page_config(page_title="Trend T9 Recovery Dashboard", layout="wide")
st.title("Trend T9 — Recovery-Synced Candlestick Dashboard")
st.caption("T9A V2 + T9D · 6H · closed-trade equity · recovery synced · paper only")

state = read_json_safe(STATE_JSON)
health = read_json_safe(HEALTH_JSON)

open_pos = read_csv_safe(OPEN_POSITIONS_CSV)
closed = read_csv_safe(CLOSED_TRADES_CSV)
equity = read_csv_safe(EQUITY_CSV)
signals = read_csv_safe(SIGNALS_CSV)
skipped = read_csv_safe(SKIPPED_CSV)
recovery_events = read_csv_safe(RECOVERY_EVENTS_CSV)
recovery_report = read_csv_safe(RECOVERY_REPORT_CSV)

equity_events = build_equity_events(closed, recovery_events)

if not open_pos.empty:
    live_prices = []
    unrealized_rs = []
    unrealized_pnls = []
    distance_to_stop = []

    for _, row in open_pos.iterrows():
        symbol = row.get("symbol")
        live = fetch_live_price(symbol)
        live_prices.append(live)

        unr = calc_unrealized_r(row, live)
        unpnl = calc_unrealized_pnl(row, live)
        unrealized_rs.append(unr)
        unrealized_pnls.append(unpnl)

        try:
            side = str(row.get("side", "")).upper()
            current_stop = float(row.get("current_stop"))
            if live is None:
                dist = np.nan
            elif side == "LONG":
                dist = (float(live) - current_stop) / float(live) * 100
            else:
                dist = (current_stop - float(live)) / float(live) * 100
        except Exception:
            dist = np.nan

        distance_to_stop.append(dist)

    open_pos["live_price"] = live_prices
    open_pos["unrealized_R"] = unrealized_rs
    open_pos["unrealized_pnl_usdt"] = unrealized_pnls
    open_pos["distance_to_stop_pct"] = distance_to_stop


# ============================================================
# TOP METRICS
# ============================================================

closed_r = float(equity_events["r"].sum()) if not equity_events.empty else 0.0
max_dd_r = max_dd_from_cum(equity_events["cum_r"]) if not equity_events.empty else 0.0
pf = profit_factor(equity_events["r"]) if not equity_events.empty else 0.0
win = (equity_events["r"] > 0).mean() * 100 if not equity_events.empty else 0.0

floating_r = float(pd.to_numeric(open_pos.get("unrealized_R", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not open_pos.empty else 0.0
total_r = closed_r + floating_r

closed_equity_usdt = health.get("closed_equity_usdt", state.get("closed_equity_usdt", INITIAL_CAPITAL_USDT))
dd_pct = health.get("drawdown_pct", state.get("drawdown_pct", 0))
portfolio_heat = health.get("portfolio_heat_pct", 0)
status = health.get("status", "UNKNOWN")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Status", status)
c2.metric("Open", len(open_pos))
c3.metric("Closed trades", len(equity_events))
c4.metric("Closed R", fmt(closed_r, 2))
c5.metric("Floating R", fmt(floating_r, 2))
c6.metric("Total R", fmt(total_r, 2))
c7.metric("DD R", fmt(max_dd_r, 2))

c8, c9, c10, c11 = st.columns(4)
c8.metric("Closed Equity", f"{fmt(closed_equity_usdt, 2)} USDT")
c9.metric("DD %", f"{fmt(dd_pct, 2)}%")
c10.metric("Portfolio Heat", f"{fmt(portfolio_heat, 2)}%")
c11.metric("PF", fmt(pf, 3))

st.write(f"**Last run UTC:** {state.get('last_run_utc', '-')}")
st.write(f"**Last recovery UTC:** {state.get('last_recovery_utc', '-')}")
st.write(f"**Last closed 6H candle:** {health.get('last_closed_bar_time', state.get('last_closed_bar_time', '-'))}")

if health.get("errors") or health.get("errors_this_run"):
    st.error("Errors detected:")
    st.write(health.get("errors", health.get("errors_this_run")))

if status == "KILL_SWITCH" or state.get("kill_switch_triggered"):
    st.error("KILL-SWITCH ACTIVE")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Open positions candles",
    "Equity / Drawdown",
    "Recovery events",
    "Closed trades",
    "Signals",
    "Skipped",
    "Monitor",
])

with tab1:
    st.subheader("Open positions — candles with Entry / Initial Stop / Current Stop / Trail activation")

    if go is None:
        st.error("Plotly non installato. Esegui: pip install plotly")

    if open_pos.empty:
        st.info("No open positions.")
    else:
        show_cols = [
            "symbol", "side", "entry_time", "entry_price", "live_price",
            "current_stop", "initial_stop", "unrealized_R", "unrealized_pnl_usdt",
            "distance_to_stop_pct", "max_favorable_r", "chandelier_active",
            "chandelier_activation_time", "chandelier_activation_price",
            "risk_amount_usdt", "notional_usdt"
        ]
        available = [c for c in show_cols if c in open_pos.columns]
        st.dataframe(open_pos[available], use_container_width=True)

        labels = []
        for i, row in open_pos.iterrows():
            labels.append(
                f"{i} | {row.get('symbol', 'UNKNOWN')} | {row.get('side', '')} | "
                f"uR={fmt(row.get('unrealized_R'), 2)} | trail={row.get('chandelier_active', False)}"
            )

        choice = st.selectbox("Select open position", labels)
        idx = int(choice.split("|")[0].strip())
        row = open_pos.iloc[idx]
        symbol = row.get("symbol")

        fig = build_candlestick_figure(symbol, row, row.get("live_price"))
        if fig is None:
            st.warning("No candlestick chart available.")
        else:
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Closed-trade equity R — synced with T9D recovery")

    if equity_events.empty:
        st.info("No closed/recovered trades yet.")
    else:
        plot = equity_events.dropna(subset=["time"]).copy()
        st.line_chart(plot.set_index("time")[["cum_r", "dd_r"]])
        st.dataframe(equity_events.tail(300).iloc[::-1], use_container_width=True)

    st.subheader("Raw equity snapshots")
    if equity.empty:
        st.info("No equity snapshot file yet.")
    else:
        parse_time(equity, "timestamp_utc")
        st.dataframe(equity.tail(300).iloc[::-1], use_container_width=True)

with tab3:
    st.subheader("T9D Recovery Events")
    if recovery_events.empty:
        st.info("No T9D recovery events yet.")
    else:
        st.dataframe(recovery_events.tail(500).iloc[::-1], use_container_width=True)

    st.subheader("T9D Recovery Report")
    if recovery_report.empty:
        st.info("No T9D recovery report yet.")
    else:
        st.dataframe(recovery_report.tail(100).iloc[::-1], use_container_width=True)

with tab4:
    st.subheader("Closed Trades")
    if closed.empty:
        st.info("No closed trades yet.")
    else:
        st.dataframe(closed.tail(500).iloc[::-1], use_container_width=True)

with tab5:
    st.subheader("Signals")
    if signals.empty:
        st.info("No signals yet.")
    else:
        st.dataframe(signals.tail(500).iloc[::-1], use_container_width=True)

with tab6:
    st.subheader("Skipped Signals")
    if skipped.empty:
        st.info("No skipped signals.")
    else:
        if "skip_reason" in skipped.columns:
            st.bar_chart(skipped["skip_reason"].value_counts())
        st.dataframe(skipped.tail(500).iloc[::-1], use_container_width=True)

with tab7:
    st.subheader("Monitor / File Sources")

    rows = []
    for name, path in [
        ("DATA_DIR", DATA_DIR),
        ("STATE_JSON", STATE_JSON),
        ("HEALTH_JSON", HEALTH_JSON),
        ("OPEN_POSITIONS_CSV", OPEN_POSITIONS_CSV),
        ("CLOSED_TRADES_CSV", CLOSED_TRADES_CSV),
        ("EQUITY_CSV", EQUITY_CSV),
        ("SIGNALS_CSV", SIGNALS_CSV),
        ("SKIPPED_CSV", SKIPPED_CSV),
        ("RECOVERY_EVENTS_CSV", RECOVERY_EVENTS_CSV),
        ("RECOVERY_REPORT_CSV", RECOVERY_REPORT_CSV),
        ("CANDLES_DIR", CANDLES_DIR),
    ]:
        rows.append({
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() and path.is_file() else "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.warning(
        "Equity R viene calcolata da closed_trades_trend_t9a.csv + POSITION_CLOSED_RECOVERY. "
        "Le posizioni aperte sono separate come Floating R. Questo evita mismatch tra recovery, state e dashboard."
    )

st.caption("Read-only dashboard. Usa Binance solo per prezzo live/candele se disponibile. Non invia ordini.")
