import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# TREND FOLLOWING CANDLE DASHBOARD 4H
# ============================================================
# Run:
# streamlit run dashboard_trend_candles_entry_trailing_4h.py
#
# Reads, when available:
# data/live_trend/sim_positions_trend.csv
# data/live_trend/sim_trades_trend.csv
# data/live_trend/signals_trend_live.csv
# data/live_trend/open_positions_stop_timeline_15m.csv
# candle cache folders under data/
#
# Read-only dashboard. No orders.
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIRS = [
    BASE_DIR / "data" / "live_trend",
    BASE_DIR / "data" / "paper_live_trend",
    BASE_DIR / "data" / "trend_live",
    BASE_DIR / "data",
]

CACHE_DIRS = [
    BASE_DIR / "data" / "raw_cache",
    BASE_DIR / "data" / "cache",
    BASE_DIR / "data" / "live_trend" / "raw_cache",
    BASE_DIR / "data" / "live_trend" / "candles_cache",
    BASE_DIR / "data" / "paper_live_trend" / "candles_cache",
]


def read_csv_safe(path):
    if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_file(names):
    for d in DATA_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None


def parse_dates(df, cols):
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], utc=True, errors="coerce")
    return x


def to_num(df, cols):
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def normalize_symbol(symbol):
    return str(symbol).replace("/", "_").replace(":", "_").replace("-", "_")


def infer_cols(df):
    x = df.copy()
    aliases = {
        "symbol": ["symbol", "pair", "market"],
        "side": ["side", "direction"],
        "entry_dt": ["entry_dt", "entry_time", "entry_time_utc", "opened_at"],
        "exit_dt": ["exit_dt", "exit_time", "exit_time_utc", "closed_at"],
        "entry_price": ["entry_price", "entry", "entry_px", "open_price"],
        "exit_price": ["exit_price", "exit", "exit_px", "close_price"],
        "initial_stop": ["initial_stop", "initial_stop_price", "sl", "stop_loss"],
        "stop": ["stop", "stop_price", "current_stop", "trail_stop", "trailing_stop"],
        "trail_activated_at": ["trail_activated_at", "trailing_activated_at", "trail_activation_dt"],
        "trail_activated_price": ["trail_activated_price", "trailing_activated_price"],
        "be_activated_at": ["be_activated_at", "breakeven_activated_at"],
        "be_price": ["be_price", "breakeven_price"],
        "realized_r": ["realized_r", "r", "R"],
    }
    for std, opts in aliases.items():
        if std in x.columns:
            continue
        for o in opts:
            if o in x.columns:
                x[std] = x[o]
                break

    x = parse_dates(x, ["entry_dt", "exit_dt", "trail_activated_at", "be_activated_at"])
    x = to_num(x, ["entry_price", "exit_price", "initial_stop", "stop", "trail_activated_price", "be_price", "realized_r"])
    if "side" in x.columns:
        x["side"] = x["side"].astype(str).str.upper()
    return x


def find_candle_file(symbol):
    s = normalize_symbol(symbol)
    compact = str(symbol).replace("/", "").replace(":USDT", "")
    for d in CACHE_DIRS:
        if not d.exists():
            continue
        candidates = [
            d / f"{s}_4h.csv",
            d / f"{s}_4H.csv",
            d / f"{compact}_4h.csv",
            d / f"{compact}_4H.csv",
            d / f"{compact}USDT_4h.csv",
            d / f"{compact}USDT_4H.csv",
            d / f"{s}.csv",
            d / f"{compact}.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        for p in d.glob("*.csv"):
            name = p.name.upper()
            if compact.upper() in name and "4H" in name:
                return p
    return None


def normalize_candles(df):
    x = df.copy()
    if "close_time" not in x.columns:
        for c in ["datetime", "time", "timestamp", "open_time", "date"]:
            if c in x.columns:
                if c == "timestamp" and pd.api.types.is_numeric_dtype(x[c]):
                    x["close_time"] = pd.to_datetime(x[c], unit="ms", utc=True, errors="coerce")
                else:
                    x["close_time"] = pd.to_datetime(x[c], utc=True, errors="coerce")
                break

    ren = {}
    for raw, std in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"),
                     ("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")]:
        if raw in x.columns and std not in x.columns:
            ren[raw] = std
    x = x.rename(columns=ren)

    need = ["close_time", "open", "high", "low", "close"]
    for c in need:
        if c not in x.columns:
            return pd.DataFrame()
    x["close_time"] = pd.to_datetime(x["close_time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=need).sort_values("close_time").reset_index(drop=True)


def load_candles(symbol):
    p = find_candle_file(symbol)
    if p is None:
        return pd.DataFrame(), None
    return normalize_candles(read_csv_safe(p)), p


def load_data():
    positions_path = find_file(["sim_positions_trend.csv", "positions_trend.csv", "open_positions_trend.csv"])
    trades_path = find_file(["sim_trades_trend.csv", "trades_trend.csv", "closed_trades_trend.csv"])
    signals_path = find_file(["signals_trend_live.csv", "signals_trend.csv", "trend_signals.csv"])
    timeline_path = find_file(["open_positions_stop_timeline_15m.csv", "stop_timeline_15m.csv", "trend_stop_timeline.csv"])

    positions = infer_cols(read_csv_safe(positions_path)) if positions_path else pd.DataFrame()
    trades = infer_cols(read_csv_safe(trades_path)) if trades_path else pd.DataFrame()
    signals = infer_cols(read_csv_safe(signals_path)) if signals_path else pd.DataFrame()
    timeline = read_csv_safe(timeline_path) if timeline_path else pd.DataFrame()

    if not timeline.empty:
        timeline = parse_dates(timeline, ["timestamp", "time", "dt", "bar_time", "entry_dt"])
        timeline = to_num(timeline, ["stop", "stop_price", "trail_stop", "trailing_stop", "current_stop"])

    return positions, trades, signals, timeline, {
        "positions": positions_path,
        "trades": trades_path,
        "signals": signals_path,
        "timeline": timeline_path,
    }


def equity_stats(trades):
    if trades.empty or "realized_r" not in trades.columns:
        return 0, 0.0, 0.0, 0.0
    r = pd.to_numeric(trades["realized_r"], errors="coerce").dropna().to_numpy(float)
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(eq)
    gp = r[r > 0].sum() if np.any(r > 0) else 0.0
    gl = r[r < 0].sum() if np.any(r < 0) else 0.0
    pf = gp / abs(gl) if gl < 0 else (999.0 if gp > 0 else 0.0)
    return len(r), float(r.sum()), float(dd.min()), float(pf)


def stop_path_for(timeline, symbol, entry_dt):
    if timeline.empty:
        return pd.DataFrame()
    x = timeline.copy()
    if "symbol" in x.columns:
        x = x[x["symbol"].astype(str) == str(symbol)]
    if "entry_dt" in x.columns and pd.notna(entry_dt):
        ed = pd.to_datetime(entry_dt, utc=True, errors="coerce")
        x = x[pd.to_datetime(x["entry_dt"], utc=True, errors="coerce") == ed]

    tcol = next((c for c in ["timestamp", "time", "dt", "bar_time"] if c in x.columns), None)
    scol = next((c for c in ["stop", "stop_price", "trail_stop", "trailing_stop", "current_stop"] if c in x.columns), None)
    if tcol is None or scol is None:
        return pd.DataFrame()
    x[tcol] = pd.to_datetime(x[tcol], utc=True, errors="coerce")
    x[scol] = pd.to_numeric(x[scol], errors="coerce")
    return x.dropna(subset=[tcol, scol]).sort_values(tcol)[[tcol, scol]].rename(columns={tcol: "time", scol: "stop"})


def crop_candles(candles, pos):
    if candles.empty or "entry_dt" not in pos or pd.isna(pos.get("entry_dt")):
        return candles.tail(140)
    start = pos["entry_dt"] - pd.Timedelta(hours=4 * 70)
    if "exit_dt" in pos and pd.notna(pos.get("exit_dt")):
        end = pos["exit_dt"] + pd.Timedelta(hours=4 * 40)
    else:
        end = candles["close_time"].max()
    return candles[(candles["close_time"] >= start) & (candles["close_time"] <= end)]


def add_marker(fig, x, y, name, color, symbol):
    if pd.isna(x) or pd.isna(y):
        return
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        text=[name], textposition="top center",
        marker=dict(color=color, size=13, symbol=symbol),
        name=name
    ))


def make_chart(candles, pos, timeline):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=candles["close_time"],
        open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"],
        name="4H candles"
    ))

    side = str(pos.get("side", "")).upper()
    entry_color = "green" if side in ["BUY", "LONG"] else "red"
    entry_symbol = "triangle-up" if side in ["BUY", "LONG"] else "triangle-down"

    add_marker(fig, pos.get("entry_dt"), pos.get("entry_price"), "ENTRY", entry_color, entry_symbol)
    add_marker(fig, pos.get("exit_dt"), pos.get("exit_price"), "EXIT", "black", "x")

    if pd.notna(pos.get("initial_stop", np.nan)):
        fig.add_hline(y=float(pos["initial_stop"]), line_dash="dash", annotation_text="Initial Stop")
    if pd.notna(pos.get("stop", np.nan)):
        fig.add_hline(y=float(pos["stop"]), line_dash="dot", annotation_text="Current Stop")
    if pd.notna(pos.get("be_activated_at", pd.NaT)):
        add_marker(fig, pos.get("be_activated_at"), pos.get("be_price", pos.get("entry_price")), "BE", "blue", "diamond")
    if pd.notna(pos.get("trail_activated_at", pd.NaT)):
        add_marker(fig, pos.get("trail_activated_at"), pos.get("trail_activated_price", pos.get("stop", pos.get("entry_price"))), "TRAIL ON", "orange", "star")

    sp = stop_path_for(timeline, pos.get("symbol"), pos.get("entry_dt"))
    if not sp.empty:
        fig.add_trace(go.Scatter(x=sp["time"], y=sp["stop"], mode="lines", name="Stop / trailing path"))

    fig.update_layout(
        title=f"{pos.get('symbol')} — {side}",
        height=720,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        legend=dict(orientation="h"),
    )
    return fig


st.set_page_config(page_title="Trend Candle Dashboard", layout="wide")
st.title("Trend Following — Candlestick Entry & Trailing Dashboard")
st.caption("Read-only. Shows 4H candles, entry, exit, stops, BE/trailing activation if logged.")

positions, trades, signals, timeline, paths = load_data()

with st.expander("Detected files"):
    for k, v in paths.items():
        st.write(f"{k}: {v}")

ntr, eqr, ddr, pf = equity_stats(trades)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open positions", len(positions))
c2.metric("Closed trades", ntr)
c3.metric("Equity R", f"{eqr:.2f}")
c4.metric("Max DD R", f"{ddr:.2f}")
c5.metric("PF", f"{pf:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Open position chart", "Closed trade chart", "Signals", "Equity"])

with tab1:
    if positions.empty:
        st.info("No open positions found.")
    elif "symbol" not in positions.columns:
        st.warning("Positions file found, but no symbol column.")
        st.dataframe(positions, use_container_width=True)
    else:
        labels = [
            f"{i} | {r.get('symbol','')} | {r.get('side','')} | {r.get('entry_dt','')}"
            for i, r in positions.iterrows()
        ]
        choice = st.selectbox("Select open position", labels)
        idx = int(choice.split("|")[0].strip())
        pos = positions.iloc[idx].to_dict()
        candles, cpath = load_candles(pos["symbol"])
        st.caption(f"Candle file: {cpath}")
        if candles.empty:
            st.warning("No 4H candle cache found for this symbol.")
            st.dataframe(positions, use_container_width=True)
        else:
            st.plotly_chart(make_chart(crop_candles(candles, pos), pos, timeline), use_container_width=True)
        st.json({k: str(v) for k, v in pos.items()})

with tab2:
    if trades.empty:
        st.info("No closed trades found.")
    elif "symbol" not in trades.columns:
        st.warning("Trades file found, but no symbol column.")
        st.dataframe(trades, use_container_width=True)
    else:
        t = trades.sort_values("exit_dt", ascending=False) if "exit_dt" in trades.columns else trades.copy()
        t = t.reset_index(drop=True)
        labels = [
            f"{i} | {r.get('symbol','')} | {r.get('side','')} | R={r.get('realized_r','')} | {r.get('entry_dt','')}"
            for i, r in t.iterrows()
        ]
        choice = st.selectbox("Select closed trade", labels)
        idx = int(choice.split("|")[0].strip())
        pos = t.iloc[idx].to_dict()
        candles, cpath = load_candles(pos["symbol"])
        st.caption(f"Candle file: {cpath}")
        if candles.empty:
            st.warning("No 4H candle cache found for this symbol.")
            st.dataframe(t, use_container_width=True)
        else:
            st.plotly_chart(make_chart(crop_candles(candles, pos), pos, timeline), use_container_width=True)
        st.json({k: str(v) for k, v in pos.items()})

with tab3:
    if signals.empty:
        st.info("No signal file found.")
    else:
        st.dataframe(signals.tail(300), use_container_width=True)

with tab4:
    if trades.empty or "realized_r" not in trades.columns:
        st.info("No realized_r available.")
    else:
        t = trades.sort_values("exit_dt").copy() if "exit_dt" in trades.columns else trades.copy()
        t["realized_r"] = pd.to_numeric(t["realized_r"], errors="coerce").fillna(0.0)
        t["equity_r"] = t["realized_r"].cumsum()
        t["drawdown_r"] = t["equity_r"] - t["equity_r"].cummax()
        x = t["exit_dt"] if "exit_dt" in t.columns else np.arange(len(t))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=t["equity_r"], mode="lines", name="Equity R"))
        fig.add_trace(go.Scatter(x=x, y=t["drawdown_r"], mode="lines", name="Drawdown R"))
        fig.update_layout(height=520, title="Closed-trade Equity / Drawdown")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(t.tail(300), use_container_width=True)
