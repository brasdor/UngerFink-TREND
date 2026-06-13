# -*- coding: utf-8 -*-
"""
Phase T3B — Trend Following Wide Exit Engineering
Author: Jean / ChatGPT

Purpose
-------
Take the Phase T2 core trend concept and test wider trend-following exits after Phase T3 showed that early breakeven/trailing was too aggressive:

- Binance Spot USDT universe, default 70 assets
- Timeframes: 2h, 4h and 6h
- Entry concept unchanged from T2:
    LONG  = close breaks Donchian high + EMA50 slope positive
    SHORT = close breaks Donchian low  + EMA50 slope negative
- Initial stop = 2 ATR
- No early breakeven
- Chandelier trailing stop activated only after +4R
- Wider Chandelier ATR multiple
- Closed candles only
- No real orders. Research/backtest only.

Outputs
-------
data/research_trend_t3b/phase_t3b_wide_exit_trades.csv
data/research_trend_t3b/phase_t3b_wide_exit_equity.csv
data/research_trend_t3b/phase_t3b_wide_exit_summary.csv
data/research_trend_t3b/phase_t3b_wide_exit_asset_summary.csv
data/research_trend_t3b/phase_t3b_wide_exit_stop_timeline.csv

Dependencies
------------
pip install ccxt pandas numpy
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "research_trend_t3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ASSETS = 70

# ── UPDATE THESE based on T2 output (phase_t2_core_trend_summary.csv) ──────
# Set TIMEFRAMES to [T2 --timeframe], EMA_FILTER_MODE to [T2 --filter-mode],
# DONCHIAN_ENTRY to T2 --donchian-entry, INITIAL_STOP_ATR_MULT to T2 --atr-mult.
TIMEFRAMES = ["1d"]                 # T1→T2 confirmed: 1d only PASS + cost-check
EMA_FILTER_MODE = "ema200_price"    # T1→T2 confirmed: EMA200 price-above (Unger §3.1)
DONCHIAN_ENTRY = 20                 # T1 stability zone 15/20/25 all profitable (100%)
INITIAL_STOP_ATR_MULT = 2.0        # T1 4yr confirmed: ATR×2.0 avg_r=+0.359R, pf=1.57 (T1 canonical)
# ───────────────────────────────────────────────────────────────────────────

OHLCV_LIMIT = 1500

EMA_LEN = 50
EMA_SLOPE_LOOKBACK = 10
EMA_REGIME_LEN = 200   # for EMA_FILTER_MODE = "ema200_price"
ATR_LEN = 14

USE_BREAKEVEN = False
BREAKEVEN_TRIGGER_R = 999.0  # disabled in T3B
CHAND_ACTIVATE_R = 4.0
CHAND_ATR_MULT = 3.0          # T10 stability finding: 3.0 is most robust (was 4.0)
CHAND_LOOKBACK = 22

ALLOW_LONG = True
ALLOW_SHORT = False   # Binance Spot: LONG only for paper-live; keep True for research

# Conservative research costs in R. Keep small here; T4 will stress costs harder.
COST_R_PER_TRADE = 0.02

QUOTE = "USDT"
STABLE_OR_FIAT_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD", "USDD", "DAI",
    "USD1", "RLUSD", "EUR", "AEUR", "EURI", "TRY", "BRL", "ARS", "GBP",
}
LEVERAGED_KEYWORDS = ["UP", "DOWN", "BULL", "BEAR"]


# ============================================================
# DATA HELPERS
# ============================================================

def make_exchange() -> ccxt.binance:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def get_top_spot_usdt_symbols(exchange: ccxt.binance, n: int = N_ASSETS) -> List[str]:
    """Return the top liquid USDT spot symbols from Binance by quote volume."""
    print("Loading Binance markets and tickers...")
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    rows = []
    for symbol, market in markets.items():
        if not market.get("spot"):
            continue
        if not market.get("active"):
            continue
        if market.get("quote") != QUOTE:
            continue
        base = market.get("base")
        if base in STABLE_OR_FIAT_BASES:
            continue
        if ":" in symbol:
            continue
        # Exclude leveraged tokens conservatively.
        base_upper = str(base).upper()
        if any(base_upper.endswith(k) for k in LEVERAGED_KEYWORDS):
            continue

        ticker = tickers.get(symbol, {}) or {}
        qv = ticker.get("quoteVolume")
        if qv is None or not np.isfinite(qv):
            qv = 0.0
        rows.append((symbol, float(qv)))

    rows.sort(key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in rows[:n]]
    print(f"Selected {len(selected)} symbols:")
    print(", ".join(selected))
    return selected


def fetch_ohlcv_df(exchange: ccxt.binance, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["datetime", "timestamp", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()

    df["ema50"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    df["ema_slope"] = df["ema50"] - df["ema50"].shift(EMA_SLOPE_LOOKBACK)

    # EMA200 price-above filter (used when EMA_FILTER_MODE == "ema200_price")
    df["ema200"] = df["close"].ewm(span=EMA_REGIME_LEN, adjust=False).mean()

    # Use shifted bands so the current candle cannot define its own breakout.
    df["donchian_high_entry"] = df["high"].shift(1).rolling(DONCHIAN_ENTRY).max()
    df["donchian_low_entry"] = df["low"].shift(1).rolling(DONCHIAN_ENTRY).min()

    # Turtle exit band: N//2 period. Active while Chandelier has not taken over.
    _exit_n = max(1, DONCHIAN_ENTRY // 2)
    df["donchian_high_exit"] = df["high"].shift(1).rolling(_exit_n).max()
    df["donchian_low_exit"] = df["low"].shift(1).rolling(_exit_n).min()

    # Chandelier uses previous completed bars, updated while in trade.
    df["chand_high"] = df["high"].shift(1).rolling(CHAND_LOOKBACK).max()
    df["chand_low"] = df["low"].shift(1).rolling(CHAND_LOOKBACK).min()

    return df


# ============================================================
# BACKTEST CORE
# ============================================================

def safe_r_multiple(side: str, entry: float, exit_price: float, initial_risk: float) -> float:
    if initial_risk <= 0 or not np.isfinite(initial_risk):
        return np.nan
    if side == "LONG":
        return (exit_price - entry) / initial_risk
    return (entry - exit_price) / initial_risk


def simulate_symbol_timeframe(symbol: str, timeframe: str, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Single-position sequential simulation for one symbol/timeframe."""
    trades: List[Dict] = []
    timeline: List[Dict] = []

    df = add_indicators(df)
    warmup = max(DONCHIAN_ENTRY, EMA_LEN + EMA_SLOPE_LOOKBACK,
                 EMA_REGIME_LEN, ATR_LEN, CHAND_LOOKBACK) + 5
    if len(df) <= warmup + 5:
        return trades, timeline

    in_pos = False
    pos: Optional[Dict] = None

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        # Skip incomplete indicator rows.
        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue

        # ----------------------------------------------------
        # Manage open position on closed candle i
        # ----------------------------------------------------
        if in_pos and pos is not None:
            side = pos["side"]
            entry = pos["entry_price"]
            initial_risk = pos["initial_risk"]
            old_stop = pos["current_stop"]

            # MFE / MAE based on completed candle extremes.
            if side == "LONG":
                candle_mfe_r = (row["high"] - entry) / initial_risk
                candle_mae_r = (row["low"] - entry) / initial_risk
                pos["mfe_r"] = max(pos["mfe_r"], candle_mfe_r)
                pos["mae_r"] = min(pos["mae_r"], candle_mae_r)
            else:
                candle_mfe_r = (entry - row["low"]) / initial_risk
                candle_mae_r = (entry - row["high"]) / initial_risk
                pos["mfe_r"] = max(pos["mfe_r"], candle_mfe_r)
                pos["mae_r"] = min(pos["mae_r"], candle_mae_r)

            # Breakeven disabled in T3B: Phase T3 showed early BE damaged trend asymmetry.
            be_activated_this_bar = False
            if USE_BREAKEVEN and (not pos["breakeven_active"]) and pos["mfe_r"] >= BREAKEVEN_TRIGGER_R:
                pos["breakeven_active"] = True
                be_activated_this_bar = True
                if side == "LONG":
                    pos["current_stop"] = max(pos["current_stop"], entry)
                else:
                    pos["current_stop"] = min(pos["current_stop"], entry)

            # Wide Chandelier activation after +4R.
            chand_activated_this_bar = False
            chandelier_candidate = np.nan
            if pos["mfe_r"] >= CHAND_ACTIVATE_R:
                if not pos["chandelier_active"]:
                    pos["chandelier_active"] = True
                    chand_activated_this_bar = True

                if side == "LONG" and np.isfinite(row["chand_high"]):
                    chandelier_candidate = row["chand_high"] - CHAND_ATR_MULT * row["atr"]
                    pos["current_stop"] = max(pos["current_stop"], chandelier_candidate)
                elif side == "SHORT" and np.isfinite(row["chand_low"]):
                    chandelier_candidate = row["chand_low"] + CHAND_ATR_MULT * row["atr"]
                    pos["current_stop"] = min(pos["current_stop"], chandelier_candidate)

            current_stop = pos["current_stop"]
            stop_changed = bool(abs(current_stop - old_stop) > 1e-12)

            timeline.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "datetime": row["datetime"],
                "side": side,
                "entry_time": pos["entry_time"],
                "entry_price": entry,
                "initial_stop": pos["initial_stop"],
                "current_stop": current_stop,
                "old_stop": old_stop,
                "stop_changed": stop_changed,
                "breakeven_active": pos["breakeven_active"],
                "chandelier_active": pos["chandelier_active"],
                "be_activated_this_bar": be_activated_this_bar,
                "chand_activated_this_bar": chand_activated_this_bar,
                "chandelier_candidate": chandelier_candidate,
                "mfe_r": pos["mfe_r"],
                "mae_r": pos["mae_r"],
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
                "atr": row["atr"],
            })

            # Exit check after stop update, using closed candle high/low.
            exit_reason = None
            exit_price = None

            if side == "LONG":
                if row["low"] <= current_stop:
                    exit_reason = "STOP_BE_CHAND" if pos["breakeven_active"] or pos["chandelier_active"] else "INITIAL_STOP"
                    exit_price = current_stop
                elif not pos["chandelier_active"]:
                    don_exit_low = row.get("donchian_low_exit", np.nan)
                    if np.isfinite(don_exit_low) and row["close"] < don_exit_low:
                        exit_reason = "DONCHIAN_EXIT"
                        exit_price = row["close"]
            else:
                if row["high"] >= current_stop:
                    exit_reason = "STOP_BE_CHAND" if pos["breakeven_active"] or pos["chandelier_active"] else "INITIAL_STOP"
                    exit_price = current_stop
                elif not pos["chandelier_active"]:
                    don_exit_high = row.get("donchian_high_exit", np.nan)
                    if np.isfinite(don_exit_high) and row["close"] > don_exit_high:
                        exit_reason = "DONCHIAN_EXIT"
                        exit_price = row["close"]

            if exit_reason is not None and exit_price is not None:
                gross_r = safe_r_multiple(side, entry, exit_price, initial_risk)
                net_r = gross_r - COST_R_PER_TRADE
                bars_held = i - pos["entry_index"]

                trades.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "side": side,
                    "entry_time": pos["entry_time"],
                    "exit_time": row["datetime"],
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "initial_stop": pos["initial_stop"],
                    "final_stop": current_stop,
                    "initial_risk": initial_risk,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "breakeven_touched": pos["breakeven_active"],
                    "chandelier_touched": pos["chandelier_active"],
                    "mfe_r": pos["mfe_r"],
                    "mae_r": pos["mae_r"],
                    "gross_r": gross_r,
                    "cost_r": COST_R_PER_TRADE,
                    "net_r": net_r,
                    "donchian_entry": DONCHIAN_ENTRY,
                    "ema_len": EMA_LEN,
                    "ema_slope_lookback": EMA_SLOPE_LOOKBACK,
                    "atr_len": ATR_LEN,
                    "initial_stop_atr_mult": INITIAL_STOP_ATR_MULT,
                    "be_trigger_r": BREAKEVEN_TRIGGER_R,
                    "chand_activate_r": CHAND_ACTIVATE_R,
                    "chand_atr_mult": CHAND_ATR_MULT,
                    "chand_lookback": CHAND_LOOKBACK,
                })
                in_pos = False
                pos = None
                continue

        # ----------------------------------------------------
        # Entry logic, only if flat
        # ----------------------------------------------------
        if not in_pos:
            close = row["close"]
            atr = row["atr"]
            ema_slope = row["ema_slope"]
            dh = row["donchian_high_entry"]
            dl = row["donchian_low_entry"]

            if not all(np.isfinite(x) for x in [close, atr, ema_slope, dh, dl]):
                continue

            # Regime filter: EMA50 slope or EMA200 price-above (set via EMA_FILTER_MODE)
            if EMA_FILTER_MODE == "ema200_price":
                ema200 = row.get("ema200", np.nan)
                filter_long  = np.isfinite(ema200) and close > float(ema200)
                filter_short = np.isfinite(ema200) and close < float(ema200)
            else:  # ema50_slope (default)
                filter_long  = float(ema_slope) > 0
                filter_short = float(ema_slope) < 0

            long_signal  = ALLOW_LONG  and close > dh and filter_long
            short_signal = ALLOW_SHORT and close < dl and filter_short

            # If both somehow happen, skip to avoid ambiguity.
            if long_signal and short_signal:
                continue

            if long_signal:
                entry = close
                initial_risk = INITIAL_STOP_ATR_MULT * atr
                initial_stop = entry - initial_risk
                if initial_stop <= 0 or initial_risk <= 0:
                    continue
                pos = {
                    "side": "LONG",
                    "entry_index": i,
                    "entry_time": row["datetime"],
                    "entry_price": entry,
                    "initial_stop": initial_stop,
                    "current_stop": initial_stop,
                    "initial_risk": initial_risk,
                    "breakeven_active": False,
                    "chandelier_active": False,
                    "mfe_r": 0.0,
                    "mae_r": 0.0,
                }
                in_pos = True

            elif short_signal:
                entry = close
                initial_risk = INITIAL_STOP_ATR_MULT * atr
                initial_stop = entry + initial_risk
                if initial_risk <= 0:
                    continue
                pos = {
                    "side": "SHORT",
                    "entry_index": i,
                    "entry_time": row["datetime"],
                    "entry_price": entry,
                    "initial_stop": initial_stop,
                    "current_stop": initial_stop,
                    "initial_risk": initial_risk,
                    "breakeven_active": False,
                    "chandelier_active": False,
                    "mfe_r": 0.0,
                    "mae_r": 0.0,
                }
                in_pos = True

    # Force close final open position at last close for research accounting.
    if in_pos and pos is not None:
        row = df.iloc[-1]
        side = pos["side"]
        entry = pos["entry_price"]
        initial_risk = pos["initial_risk"]
        exit_price = row["close"]
        gross_r = safe_r_multiple(side, entry, exit_price, initial_risk)
        net_r = gross_r - COST_R_PER_TRADE
        trades.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "entry_time": pos["entry_time"],
            "exit_time": row["datetime"],
            "entry_price": entry,
            "exit_price": exit_price,
            "initial_stop": pos["initial_stop"],
            "final_stop": pos["current_stop"],
            "initial_risk": initial_risk,
            "bars_held": len(df) - 1 - pos["entry_index"],
            "exit_reason": "FORCED_END_OF_DATA",
            "breakeven_touched": pos["breakeven_active"],
            "chandelier_touched": pos["chandelier_active"],
            "mfe_r": pos["mfe_r"],
            "mae_r": pos["mae_r"],
            "gross_r": gross_r,
            "cost_r": COST_R_PER_TRADE,
            "net_r": net_r,
            "donchian_entry": DONCHIAN_ENTRY,
            "ema_len": EMA_LEN,
            "ema_slope_lookback": EMA_SLOPE_LOOKBACK,
            "atr_len": ATR_LEN,
            "initial_stop_atr_mult": INITIAL_STOP_ATR_MULT,
            "be_trigger_r": BREAKEVEN_TRIGGER_R,
            "chand_activate_r": CHAND_ACTIVATE_R,
            "chand_atr_mult": CHAND_ATR_MULT,
            "chand_lookback": CHAND_LOOKBACK,
        })

    return trades, timeline


# ============================================================
# STATS
# ============================================================

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def profit_factor(r: pd.Series) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return float(gains / losses)


def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    groups = [("ALL", "ALL", trades_df)]
    for tf, g in trades_df.groupby("timeframe"):
        groups.append((tf, "ALL", g))
    for side, g in trades_df.groupby("side"):
        groups.append(("ALL", side, g))
    for (tf, side), g in trades_df.groupby(["timeframe", "side"]):
        groups.append((tf, side, g))

    for tf, side, g in groups:
        g = g.sort_values("exit_time").copy()
        eq = g["net_r"].cumsum()
        rows.append({
            "timeframe": tf,
            "side": side,
            "trades": len(g),
            "total_net_r": g["net_r"].sum(),
            "avg_net_r": g["net_r"].mean(),
            "median_net_r": g["net_r"].median(),
            "win_rate": (g["net_r"] > 0).mean(),
            "profit_factor": profit_factor(g["net_r"]),
            "max_drawdown_r": max_drawdown(eq),
            "best_trade_r": g["net_r"].max(),
            "worst_trade_r": g["net_r"].min(),
            "avg_bars_held": g["bars_held"].mean(),
            "median_bars_held": g["bars_held"].median(),
            "be_touch_rate": g["breakeven_touched"].mean(),
            "chand_touch_rate": g["chandelier_touched"].mean(),
            "avg_mfe_r": g["mfe_r"].mean(),
            "avg_mae_r": g["mae_r"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["timeframe", "side"]).reset_index(drop=True)


def build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    rows = []
    for (symbol, timeframe), g in trades_df.groupby(["symbol", "timeframe"]):
        g = g.sort_values("exit_time").copy()
        eq = g["net_r"].cumsum()
        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "trades": len(g),
            "total_net_r": g["net_r"].sum(),
            "avg_net_r": g["net_r"].mean(),
            "win_rate": (g["net_r"] > 0).mean(),
            "profit_factor": profit_factor(g["net_r"]),
            "max_drawdown_r": max_drawdown(eq),
            "best_trade_r": g["net_r"].max(),
            "worst_trade_r": g["net_r"].min(),
            "be_touch_rate": g["breakeven_touched"].mean(),
            "chand_touch_rate": g["chandelier_touched"].mean(),
        })
    return pd.DataFrame(rows).sort_values("total_net_r", ascending=False).reset_index(drop=True)


def build_equity(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    eq = trades_df.sort_values("exit_time").copy()
    eq["equity_r"] = eq["net_r"].cumsum()
    eq["peak_r"] = eq["equity_r"].cummax()
    eq["drawdown_r"] = eq["equity_r"] - eq["peak_r"]
    return eq[[
        "exit_time", "symbol", "timeframe", "side", "exit_reason",
        "net_r", "equity_r", "peak_r", "drawdown_r",
        "breakeven_touched", "chandelier_touched", "mfe_r", "mae_r",
    ]]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Phase T3B — Wide Trend Exit Engineering")
    print("70 asset default | 2H/4H/6H | Donchian/EMA entry | NO early BE | Chandelier +4R, ATR x4")
    print("=" * 80)

    exchange = make_exchange()
    symbols = get_top_spot_usdt_symbols(exchange, N_ASSETS)

    all_trades: List[Dict] = []
    all_timeline: List[Dict] = []

    for tf in TIMEFRAMES:
        print(f"\n--- Timeframe: {tf} ---")
        for idx, symbol in enumerate(symbols, start=1):
            try:
                print(f"[{idx:02d}/{len(symbols)}] Fetching {symbol} {tf}...")
                df = fetch_ohlcv_df(exchange, symbol, tf, OHLCV_LIMIT)
                if df.empty or len(df) < 200:
                    print(f"  Skipped {symbol}: insufficient data ({len(df)} bars)")
                    continue
                trades, timeline = simulate_symbol_timeframe(symbol, tf, df)
                all_trades.extend(trades)
                all_timeline.extend(timeline)
                print(f"  Trades: {len(trades)} | Timeline rows: {len(timeline)}")
                time.sleep(exchange.rateLimit / 1000.0)
            except Exception as exc:
                print(f"  ERROR {symbol} {tf}: {exc}")
                continue

    trades_df = pd.DataFrame(all_trades)
    timeline_df = pd.DataFrame(all_timeline)

    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
        trades_df = trades_df.sort_values(["exit_time", "symbol", "timeframe"]).reset_index(drop=True)

    if not timeline_df.empty:
        timeline_df["datetime"] = pd.to_datetime(timeline_df["datetime"], utc=True)
        timeline_df["entry_time"] = pd.to_datetime(timeline_df["entry_time"], utc=True)
        timeline_df = timeline_df.sort_values(["datetime", "symbol", "timeframe"]).reset_index(drop=True)

    summary_df = build_summary(trades_df)
    asset_summary_df = build_asset_summary(trades_df)
    equity_df = build_equity(trades_df)

    trades_path = OUT_DIR / "phase_t3b_wide_exit_trades.csv"
    equity_path = OUT_DIR / "phase_t3b_wide_exit_equity.csv"
    summary_path = OUT_DIR / "phase_t3b_wide_exit_summary.csv"
    asset_summary_path = OUT_DIR / "phase_t3b_wide_exit_asset_summary.csv"
    timeline_path = OUT_DIR / "phase_t3b_wide_exit_stop_timeline.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    asset_summary_df.to_csv(asset_summary_path, index=False)
    timeline_df.to_csv(timeline_path, index=False)

    print("\n" + "=" * 80)
    print("DONE — Phase T3B outputs written:")
    print(trades_path)
    print(equity_path)
    print(summary_path)
    print(asset_summary_path)
    print(timeline_path)
    print("=" * 80)

    if not summary_df.empty:
        print("\nSUMMARY:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
