#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase T2 — Core Trend Engine
Trend Following System — Binance Spot — Donchian/EMA Core

Scopo della Phase T2:
- Congelare il concept grezzo più promettente della T1.
- Costruire un core engine pulito e leggibile.
- Usare 70 asset Binance spot USDT di default.
- Usare solo candele chiuse.
- NON introdurre ancora exit engineering avanzata.

Logica T2 default:
LONG only, adatta a Binance Spot:
    trend filter  : EMA50 slope > 0
    entry         : close > Donchian High 20
    initial stop  : entry - 2 * ATR14
    exit          : initial stop oppure close < Donchian Low 10

Nota metodologica:
- Breakeven +1R e Chandelier dopo +2R appartengono alla Phase T3.
- Qui vogliamo un motore core stabile, non ancora ottimizzato.

Output:
    data/research_trend_t2/phase_t2_core_trend_trades.csv
    data/research_trend_t2/phase_t2_core_trend_equity.csv
    data/research_trend_t2/phase_t2_core_trend_summary.csv
    data/research_trend_t2/phase_t2_core_trend_asset_summary.csv

Author: ChatGPT per Jean
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError as exc:
    raise SystemExit(
        "Manca ccxt. Installa con:\n\n"
        "pip install ccxt pandas numpy\n"
    ) from exc


# ============================================================
# CONFIG DEFAULT — PHASE T2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_trend_t2"
OUT_DIR = DATA_DIR / "research_trend_t2"

DEFAULT_TOP_N = 70
DEFAULT_LIMIT = 1200
DEFAULT_TIMEFRAMES = ["2h", "4h"]

# Core T2 default, derivato dal concept migliore T1
DEFAULT_DONCHIAN_ENTRY = 20
DEFAULT_DONCHIAN_EXIT = 10
DEFAULT_ATR_MULT = 2.0
DEFAULT_EMA_LEN = 50
DEFAULT_EMA_SLOPE_LOOKBACK = 10
DEFAULT_SIDE_MODE = "long"  # spot-compatible

ATR_LEN = 14

EXCLUDED_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI"
}
LEVERAGED_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    trade_id: int
    symbol: str
    timeframe: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price_initial: float
    exit_reason: str
    bars_held: int
    risk_per_unit: float
    pnl_R: float
    pnl_pct_on_price: float
    mae_R: float
    mfe_R: float
    donchian_entry: int
    donchian_exit: int
    atr_mult: float
    ema_len: int
    ema_slope_lookback: int


@dataclass
class Summary:
    phase: str
    symbols_requested: int
    symbols_with_data: int
    timeframes: str
    side_mode: str
    donchian_entry: int
    donchian_exit: int
    atr_mult: float
    ema_len: int
    ema_slope_lookback: int
    trades: int
    total_R: float
    avg_R: float
    median_R: float
    win_rate_pct: float
    profit_factor: float
    max_dd_R: float
    best_trade_R: float
    worst_trade_R: float
    avg_bars_held: float
    median_bars_held: float
    max_bars_held: int
    expectancy_R: float


# ============================================================
# UTILS
# ============================================================

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_symbol_filename(symbol: str, timeframe: str) -> str:
    return symbol.replace("/", "_").replace(":", "_") + f"_{timeframe}.csv"


def is_good_spot_usdt_symbol(symbol: str, market: Dict) -> bool:
    if not market.get("spot", False):
        return False
    if not market.get("active", True):
        return False
    if not symbol.endswith("/USDT"):
        return False
    if ":" in symbol:
        return False

    base = str(market.get("base", "")).upper()
    if base in EXCLUDED_BASES:
        return False

    for kw in LEVERAGED_KEYWORDS:
        if base.endswith(kw):
            return False

    return True


def init_exchange() -> "ccxt.binance":
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })


def get_top_usdt_spot_symbols(exchange: "ccxt.binance", top_n: int) -> List[str]:
    print("Loading Binance markets...")
    markets = exchange.load_markets()

    print("Loading Binance tickers...")
    tickers = exchange.fetch_tickers()

    rows = []
    for symbol, market in markets.items():
        if not is_good_spot_usdt_symbol(symbol, market):
            continue
        ticker = tickers.get(symbol, {})
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None or not np.isfinite(quote_volume):
            quote_volume = 0.0
        rows.append((symbol, float(quote_volume)))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in rows[:top_n]]

    print(f"\nSelected {len(symbols)} Binance spot USDT symbols:")
    print(", ".join(symbols))
    return symbols


def fetch_ohlcv_cached(
    exchange: "ccxt.binance",
    symbol: str,
    timeframe: str,
    limit: int,
    refresh: bool,
) -> Optional[pd.DataFrame]:
    path = RAW_DIR / safe_symbol_filename(symbol, timeframe)

    if path.exists() and not refresh:
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception as exc:
            print(f"[WARN] Cache read failed {path}: {exc}. Refetching...")

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        time.sleep(exchange.rateLimit / 1000.0)
    except Exception as exc:
        print(f"[WARN] Fetch failed for {symbol} {timeframe}: {exc}")
        return None

    if not ohlcv or len(ohlcv) < 250:
        print(f"[WARN] Insufficient candles for {symbol} {timeframe}")
        return None

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Strict closed candles only: drop the currently forming candle.
    df = df.iloc[:-1].reset_index(drop=True)

    df.to_csv(path, index=False)
    return df


def add_indicators(
    df: pd.DataFrame,
    donchian_entry: int,
    donchian_exit: int,
    ema_len: int,
    ema_slope_lookback: int,
) -> pd.DataFrame:
    d = df.copy()

    prev_close = d["close"].shift(1)
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - prev_close).abs()
    tr3 = (d["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_LEN).mean()

    d["ema"] = d["close"].ewm(span=ema_len, adjust=False).mean()
    d["ema_slope"] = d["ema"] - d["ema"].shift(ema_slope_lookback)

    # Shifted levels: levels known before the current candle closes.
    d["donchian_high_entry"] = d["high"].shift(1).rolling(donchian_entry).max()
    d["donchian_low_entry"] = d["low"].shift(1).rolling(donchian_entry).min()
    d["donchian_high_exit"] = d["high"].shift(1).rolling(donchian_exit).max()
    d["donchian_low_exit"] = d["low"].shift(1).rolling(donchian_exit).min()

    return d


def profit_factor(r_values: List[float]) -> float:
    wins = [x for x in r_values if x > 0]
    losses = [x for x in r_values if x < 0]
    gross_profit = float(np.sum(wins)) if wins else 0.0
    gross_loss = abs(float(np.sum(losses))) if losses else 0.0
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown_r(r_values: List[float]) -> float:
    if not r_values:
        return 0.0
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def backtest_symbol_core(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    side_mode: str,
    donchian_entry: int,
    donchian_exit: int,
    atr_mult: float,
    ema_len: int,
    ema_slope_lookback: int,
    first_trade_id: int,
) -> List[Trade]:
    d = add_indicators(
        df=df,
        donchian_entry=donchian_entry,
        donchian_exit=donchian_exit,
        ema_len=ema_len,
        ema_slope_lookback=ema_slope_lookback,
    )

    trades: List[Trade] = []
    position = None
    trade_id = first_trade_id

    for i in range(len(d)):
        row = d.iloc[i]

        required = [
            row.get("atr"), row.get("ema_slope"), row.get("donchian_high_entry"),
            row.get("donchian_low_entry"), row.get("donchian_high_exit"), row.get("donchian_low_exit"),
        ]
        if any((not np.isfinite(x)) for x in required):
            continue
        if float(row["atr"]) <= 0:
            continue

        ts = row["timestamp"]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])

        if position is None:
            allow_long = side_mode in ("long", "both")
            allow_short = side_mode in ("short", "both")

            long_signal = (
                allow_long
                and close > float(row["donchian_high_entry"])
                and float(row["ema_slope"]) > 0
            )
            short_signal = (
                allow_short
                and close < float(row["donchian_low_entry"])
                and float(row["ema_slope"]) < 0
            )

            # If both somehow trigger, skip. No discretionary choice.
            if long_signal and not short_signal:
                risk = atr * atr_mult
                position = {
                    "side": "long",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": close,
                    "stop_initial": close - risk,
                    "risk_per_unit": risk,
                    "mae_R": 0.0,
                    "mfe_R": 0.0,
                }
            elif short_signal and not long_signal:
                risk = atr * atr_mult
                position = {
                    "side": "short",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": close,
                    "stop_initial": close + risk,
                    "risk_per_unit": risk,
                    "mae_R": 0.0,
                    "mfe_R": 0.0,
                }
            continue

        side = position["side"]
        entry = float(position["entry_price"])
        stop_initial = float(position["stop_initial"])
        risk = float(position["risk_per_unit"])
        bars_held = i - int(position["entry_idx"])

        # Track rough intra-trade MAE/MFE using candle high/low.
        if side == "long":
            adverse_R = (low - entry) / risk
            favorable_R = (high - entry) / risk
        else:
            adverse_R = (entry - high) / risk
            favorable_R = (entry - low) / risk
        position["mae_R"] = min(float(position["mae_R"]), float(adverse_R))
        position["mfe_R"] = max(float(position["mfe_R"]), float(favorable_R))

        exit_price = None
        exit_reason = None

        if side == "long":
            # Conservative order: stop first, then close-based Donchian exit.
            if low <= stop_initial:
                exit_price = stop_initial
                exit_reason = "initial_stop"
            elif close < float(row["donchian_low_exit"]):
                exit_price = close
                exit_reason = "donchian_exit"

            if exit_price is not None:
                pnl_R = (float(exit_price) - entry) / risk
                pnl_pct = (float(exit_price) / entry - 1.0) * 100.0
                trades.append(Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    entry_time=str(position["entry_time"]),
                    exit_time=str(ts),
                    entry_price=entry,
                    exit_price=float(exit_price),
                    stop_price_initial=stop_initial,
                    exit_reason=exit_reason,
                    bars_held=int(bars_held),
                    risk_per_unit=risk,
                    pnl_R=float(pnl_R),
                    pnl_pct_on_price=float(pnl_pct),
                    mae_R=float(position["mae_R"]),
                    mfe_R=float(position["mfe_R"]),
                    donchian_entry=donchian_entry,
                    donchian_exit=donchian_exit,
                    atr_mult=atr_mult,
                    ema_len=ema_len,
                    ema_slope_lookback=ema_slope_lookback,
                ))
                trade_id += 1
                position = None

        elif side == "short":
            if high >= stop_initial:
                exit_price = stop_initial
                exit_reason = "initial_stop"
            elif close > float(row["donchian_high_exit"]):
                exit_price = close
                exit_reason = "donchian_exit"

            if exit_price is not None:
                pnl_R = (entry - float(exit_price)) / risk
                pnl_pct = (entry / float(exit_price) - 1.0) * 100.0
                trades.append(Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    entry_time=str(position["entry_time"]),
                    exit_time=str(ts),
                    entry_price=entry,
                    exit_price=float(exit_price),
                    stop_price_initial=stop_initial,
                    exit_reason=exit_reason,
                    bars_held=int(bars_held),
                    risk_per_unit=risk,
                    pnl_R=float(pnl_R),
                    pnl_pct_on_price=float(pnl_pct),
                    mae_R=float(position["mae_R"]),
                    mfe_R=float(position["mfe_R"]),
                    donchian_entry=donchian_entry,
                    donchian_exit=donchian_exit,
                    atr_mult=atr_mult,
                    ema_len=ema_len,
                    ema_slope_lookback=ema_slope_lookback,
                ))
                trade_id += 1
                position = None

    return trades


def build_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["exit_time", "pnl_R", "equity_R", "peak_R", "dd_R"])

    eq = trades_df.copy()
    eq["exit_time"] = pd.to_datetime(eq["exit_time"], utc=True)
    eq = eq.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)
    eq["equity_R"] = eq["pnl_R"].cumsum()
    eq["peak_R"] = eq["equity_R"].cummax()
    eq["dd_R"] = eq["equity_R"] - eq["peak_R"]
    return eq[["exit_time", "trade_id", "symbol", "timeframe", "side", "pnl_R", "equity_R", "peak_R", "dd_R"]]


def summarize_all(
    trades_df: pd.DataFrame,
    args: argparse.Namespace,
    symbols_requested: int,
    symbols_with_data: int,
) -> Summary:
    r = trades_df["pnl_R"].tolist() if not trades_df.empty else []
    bars = trades_df["bars_held"].tolist() if not trades_df.empty else []

    if not r:
        return Summary(
            phase="T2_CORE_TREND_ENGINE",
            symbols_requested=symbols_requested,
            symbols_with_data=symbols_with_data,
            timeframes=",".join(args.timeframes),
            side_mode=args.side_mode,
            donchian_entry=args.donchian_entry,
            donchian_exit=args.donchian_exit,
            atr_mult=args.atr_mult,
            ema_len=args.ema_len,
            ema_slope_lookback=args.ema_slope_lookback,
            trades=0,
            total_R=0.0,
            avg_R=0.0,
            median_R=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            max_dd_R=0.0,
            best_trade_R=0.0,
            worst_trade_R=0.0,
            avg_bars_held=0.0,
            median_bars_held=0.0,
            max_bars_held=0,
            expectancy_R=0.0,
        )

    return Summary(
        phase="T2_CORE_TREND_ENGINE",
        symbols_requested=symbols_requested,
        symbols_with_data=symbols_with_data,
        timeframes=",".join(args.timeframes),
        side_mode=args.side_mode,
        donchian_entry=args.donchian_entry,
        donchian_exit=args.donchian_exit,
        atr_mult=args.atr_mult,
        ema_len=args.ema_len,
        ema_slope_lookback=args.ema_slope_lookback,
        trades=len(r),
        total_R=float(np.sum(r)),
        avg_R=float(np.mean(r)),
        median_R=float(np.median(r)),
        win_rate_pct=float(100.0 * np.mean([x > 0 for x in r])),
        profit_factor=float(profit_factor(r)),
        max_dd_R=float(max_drawdown_r(r)),
        best_trade_R=float(np.max(r)),
        worst_trade_R=float(np.min(r)),
        avg_bars_held=float(np.mean(bars)),
        median_bars_held=float(np.median(bars)),
        max_bars_held=int(np.max(bars)),
        expectancy_R=float(np.mean(r)),
    )


def build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for (symbol, timeframe), g in trades_df.groupby(["symbol", "timeframe"]):
        r = g["pnl_R"].tolist()
        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "trades": len(r),
            "total_R": float(np.sum(r)),
            "avg_R": float(np.mean(r)),
            "median_R": float(np.median(r)),
            "win_rate_pct": float(100.0 * np.mean([x > 0 for x in r])),
            "profit_factor": float(profit_factor(r)),
            "max_dd_R_local": float(max_drawdown_r(r)),
            "best_trade_R": float(np.max(r)),
            "worst_trade_R": float(np.min(r)),
            "avg_bars_held": float(g["bars_held"].mean()),
            "avg_mfe_R": float(g["mfe_R"].mean()),
            "avg_mae_R": float(g["mae_R"].mean()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["total_R", "profit_factor", "trades"], ascending=[False, False, False]).reset_index(drop=True)


def write_outputs(trades: List[Trade], args: argparse.Namespace, symbols_requested: int, symbols_with_data: int) -> None:
    trades_path = OUT_DIR / "phase_t2_core_trend_trades.csv"
    equity_path = OUT_DIR / "phase_t2_core_trend_equity.csv"
    summary_path = OUT_DIR / "phase_t2_core_trend_summary.csv"
    asset_summary_path = OUT_DIR / "phase_t2_core_trend_asset_summary.csv"

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
        trades_df = trades_df.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)

    equity_df = build_equity_curve(trades_df)
    summary = summarize_all(trades_df, args, symbols_requested, symbols_with_data)
    summary_df = pd.DataFrame([asdict(summary)])
    asset_summary_df = build_asset_summary(trades_df)

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    asset_summary_df.to_csv(asset_summary_path, index=False)

    print("\nSaved outputs:")
    print(f"  Trades       : {trades_path}")
    print(f"  Equity       : {equity_path}")
    print(f"  Summary      : {summary_path}")
    print(f"  Asset summary: {asset_summary_path}")

    print("\n=== PHASE T2 SUMMARY ===")
    print(summary_df.to_string(index=False))

    if not asset_summary_df.empty:
        print("\nTop 15 assets by total_R:")
        cols = ["symbol", "timeframe", "trades", "total_R", "avg_R", "profit_factor", "max_dd_R_local", "best_trade_R"]
        print(asset_summary_df[cols].head(15).to_string(index=False))


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    exchange = init_exchange()

    symbols = get_top_usdt_spot_symbols(exchange, top_n=args.top_n)

    data: Dict[Tuple[str, str], pd.DataFrame] = {}
    for timeframe in args.timeframes:
        print(f"\n=== Fetch/cache candles timeframe={timeframe} ===")
        for idx, symbol in enumerate(symbols, start=1):
            print(f"[{idx}/{len(symbols)}] {symbol} {timeframe}")
            df = fetch_ohlcv_cached(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                limit=args.limit,
                refresh=args.refresh,
            )
            if df is not None and len(df) >= 250:
                data[(symbol, timeframe)] = df

    all_trades: List[Trade] = []
    next_trade_id = 1

    for timeframe in args.timeframes:
        symbols_with_tf_data = [s for s in symbols if (s, timeframe) in data]
        print(f"\n=== Core trend backtest timeframe={timeframe}, symbols={len(symbols_with_tf_data)} ===")

        for symbol in symbols_with_tf_data:
            trades = backtest_symbol_core(
                df=data[(symbol, timeframe)],
                symbol=symbol,
                timeframe=timeframe,
                side_mode=args.side_mode,
                donchian_entry=args.donchian_entry,
                donchian_exit=args.donchian_exit,
                atr_mult=args.atr_mult,
                ema_len=args.ema_len,
                ema_slope_lookback=args.ema_slope_lookback,
                first_trade_id=next_trade_id,
            )
            if trades:
                next_trade_id = max(t.trade_id for t in trades) + 1
                all_trades.extend(trades)
            print(f"{symbol:12s} {timeframe:>3s} trades={len(trades):3d} R={sum(t.pnl_R for t in trades):8.2f}")

    symbols_with_data = len(set(symbol for symbol, _tf in data.keys()))
    write_outputs(
        trades=all_trades,
        args=args,
        symbols_requested=len(symbols),
        symbols_with_data=symbols_with_data,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase T2 — Core Trend Engine — Binance Spot")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Numero asset Binance USDT spot. Default: 70")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Numero candele richieste. Default: 1200")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES, help="Esempio: --timeframes 2h 4h")
    parser.add_argument("--refresh", action="store_true", help="Riscarica dati anche se cache esiste")

    parser.add_argument("--side-mode", choices=["long", "short", "both"], default=DEFAULT_SIDE_MODE)
    parser.add_argument("--donchian-entry", type=int, default=DEFAULT_DONCHIAN_ENTRY)
    parser.add_argument("--donchian-exit", type=int, default=DEFAULT_DONCHIAN_EXIT)
    parser.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT)
    parser.add_argument("--ema-len", type=int, default=DEFAULT_EMA_LEN)
    parser.add_argument("--ema-slope-lookback", type=int, default=DEFAULT_EMA_SLOPE_LOOKBACK)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
