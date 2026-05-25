#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase T1 — Clean Trend Concept Discovery
Trend Following System 2H/4H — Binance Spot

Obiettivo:
- Testare concetti trend-following semplici, senza overfitting.
- Donchian breakout + EMA slope filter + ATR stop + exit semplice.
- Usare solo candele chiuse.
- Salvare risultati e trades in CSV.

Logica base:
LONG:
    close > Donchian High N
    EMA50 slope > 0
    initial stop = entry - ATR * atr_mult
    exit = stop oppure close < Donchian Low exit_N

SHORT:
    close < Donchian Low N
    EMA50 slope < 0
    initial stop = entry + ATR * atr_mult
    exit = stop oppure close > Donchian High exit_N

Nota:
- Su Binance Spot gli short non sono eseguibili direttamente.
- In questa fase di research li simuliamo separatamente per capire se esiste edge teorico.
- Prima del paper/live spot reale potremo congelare solo LONG o usare short solo come informazione di regime.

Author: ChatGPT per Jean
"""

from __future__ import annotations

import argparse
import math
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
# CONFIG DEFAULT
# ============================================================

DEFAULT_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = DEFAULT_BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_trend_t1"
OUT_DIR = DATA_DIR / "research_trend_t1"

EXCLUDED_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
    "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "EURI"
}

LEVERAGED_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR")

DEFAULT_TIMEFRAMES = ["2h", "4h"]
DEFAULT_LIMIT = 1200
DEFAULT_TOP_N = 50

DONCHIAN_ENTRY_LIST = [20, 40, 55]
DONCHIAN_EXIT_LIST = [10, 20]
ATR_MULT_LIST = [2.0, 3.0]
EMA_LEN_LIST = [50]
EMA_SLOPE_LOOKBACK_LIST = [5, 10]

ATR_LEN = 14
STARTING_EQUITY_R = 0.0


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Trade:
    symbol: str
    timeframe: str
    side: str
    concept: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price_initial: float
    exit_reason: str
    bars_held: int
    risk_per_unit: float
    pnl_R: float
    donchian_entry: int
    donchian_exit: int
    atr_mult: float
    ema_len: int
    ema_slope_lookback: int


@dataclass
class SummaryRow:
    concept: str
    timeframe: str
    side_mode: str
    donchian_entry: int
    donchian_exit: int
    atr_mult: float
    ema_len: int
    ema_slope_lookback: int
    symbols_tested: int
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

    base = market.get("base", "")
    if base in EXCLUDED_BASES:
        return False

    upper_base = base.upper()
    for kw in LEVERAGED_KEYWORDS:
        # evita token leveraged tipo BTCUP, ETHDOWN ecc.
        if upper_base.endswith(kw):
            return False

    return True


def init_exchange() -> "ccxt.binance":
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    return exchange


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

    print(f"Selected {len(symbols)} USDT spot symbols:")
    print(", ".join(symbols))
    return symbols


def fetch_ohlcv_cached(
    exchange: "ccxt.binance",
    symbol: str,
    timeframe: str,
    limit: int,
    refresh: bool = False,
) -> Optional[pd.DataFrame]:
    path = RAW_DIR / safe_symbol_filename(symbol, timeframe)

    if path.exists() and not refresh:
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception as exc:
            print(f"[WARN] Could not read cache {path}: {exc}. Refetching...")

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        time.sleep(exchange.rateLimit / 1000.0)
    except Exception as exc:
        print(f"[WARN] fetch failed for {symbol} {timeframe}: {exc}")
        return None

    if not ohlcv or len(ohlcv) < 200:
        print(f"[WARN] insufficient candles for {symbol} {timeframe}")
        return None

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Closed candles only:
    # Binance often returns the currently forming candle as the last row.
    # We drop the last row for strict research consistency.
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

    # Shifted Donchian levels:
    # Entry on close crossing above/below levels known before current candle close.
    d["donchian_high_entry"] = d["high"].shift(1).rolling(donchian_entry).max()
    d["donchian_low_entry"] = d["low"].shift(1).rolling(donchian_entry).min()

    d["donchian_high_exit"] = d["high"].shift(1).rolling(donchian_exit).max()
    d["donchian_low_exit"] = d["low"].shift(1).rolling(donchian_exit).min()

    return d


def backtest_symbol(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    donchian_entry: int,
    donchian_exit: int,
    atr_mult: float,
    ema_len: int,
    ema_slope_lookback: int,
    side_mode: str,
) -> List[Trade]:
    """
    side_mode: "long", "short", "both"
    """
    d = add_indicators(
        df=df,
        donchian_entry=donchian_entry,
        donchian_exit=donchian_exit,
        ema_len=ema_len,
        ema_slope_lookback=ema_slope_lookback,
    )

    trades: List[Trade] = []
    position = None

    for i in range(len(d)):
        row = d.iloc[i]

        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue
        if not np.isfinite(row["ema_slope"]):
            continue
        if not np.isfinite(row["donchian_high_entry"]) or not np.isfinite(row["donchian_low_entry"]):
            continue
        if not np.isfinite(row["donchian_high_exit"]) or not np.isfinite(row["donchian_low_exit"]):
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

            # In caso rarissimo di doppio segnale, non entra.
            if long_signal and not short_signal:
                entry_price = close
                risk_per_unit = atr * atr_mult
                stop_price = entry_price - risk_per_unit
                position = {
                    "side": "long",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "stop_initial": stop_price,
                    "risk_per_unit": risk_per_unit,
                }

            elif short_signal and not long_signal:
                entry_price = close
                risk_per_unit = atr * atr_mult
                stop_price = entry_price + risk_per_unit
                position = {
                    "side": "short",
                    "entry_idx": i,
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "stop_initial": stop_price,
                    "risk_per_unit": risk_per_unit,
                }

            continue

        # Manage open position
        side = position["side"]
        entry_price = position["entry_price"]
        stop_initial = position["stop_initial"]
        risk_per_unit = position["risk_per_unit"]
        bars_held = i - position["entry_idx"]

        exit_price = None
        exit_reason = None

        if side == "long":
            # Stop check first, conservative assumption.
            if low <= stop_initial:
                exit_price = stop_initial
                exit_reason = "initial_stop"
            elif close < float(row["donchian_low_exit"]):
                exit_price = close
                exit_reason = "donchian_exit"

            if exit_price is not None:
                pnl_R = (exit_price - entry_price) / risk_per_unit
                trades.append(Trade(
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    concept="donchian_ema_clean",
                    entry_time=str(position["entry_time"]),
                    exit_time=str(ts),
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    stop_price_initial=stop_initial,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    risk_per_unit=risk_per_unit,
                    pnl_R=float(pnl_R),
                    donchian_entry=donchian_entry,
                    donchian_exit=donchian_exit,
                    atr_mult=atr_mult,
                    ema_len=ema_len,
                    ema_slope_lookback=ema_slope_lookback,
                ))
                position = None

        elif side == "short":
            if high >= stop_initial:
                exit_price = stop_initial
                exit_reason = "initial_stop"
            elif close > float(row["donchian_high_exit"]):
                exit_price = close
                exit_reason = "donchian_exit"

            if exit_price is not None:
                pnl_R = (entry_price - exit_price) / risk_per_unit
                trades.append(Trade(
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    concept="donchian_ema_clean",
                    entry_time=str(position["entry_time"]),
                    exit_time=str(ts),
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    stop_price_initial=stop_initial,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    risk_per_unit=risk_per_unit,
                    pnl_R=float(pnl_R),
                    donchian_entry=donchian_entry,
                    donchian_exit=donchian_exit,
                    atr_mult=atr_mult,
                    ema_len=ema_len,
                    ema_slope_lookback=ema_slope_lookback,
                ))
                position = None

    return trades


def compute_max_drawdown_r(r_values: List[float]) -> float:
    if not r_values:
        return 0.0
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def profit_factor(r_values: List[float]) -> float:
    wins = [x for x in r_values if x > 0]
    losses = [x for x in r_values if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def summarize_trades(
    trades: List[Trade],
    timeframe: str,
    side_mode: str,
    donchian_entry: int,
    donchian_exit: int,
    atr_mult: float,
    ema_len: int,
    ema_slope_lookback: int,
    symbols_tested: int,
) -> SummaryRow:
    r = [t.pnl_R for t in trades]
    n = len(r)

    if n == 0:
        return SummaryRow(
            concept="donchian_ema_clean",
            timeframe=timeframe,
            side_mode=side_mode,
            donchian_entry=donchian_entry,
            donchian_exit=donchian_exit,
            atr_mult=atr_mult,
            ema_len=ema_len,
            ema_slope_lookback=ema_slope_lookback,
            symbols_tested=symbols_tested,
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
            expectancy_R=0.0,
        )

    return SummaryRow(
        concept="donchian_ema_clean",
        timeframe=timeframe,
        side_mode=side_mode,
        donchian_entry=donchian_entry,
        donchian_exit=donchian_exit,
        atr_mult=atr_mult,
        ema_len=ema_len,
        ema_slope_lookback=ema_slope_lookback,
        symbols_tested=symbols_tested,
        trades=n,
        total_R=float(np.sum(r)),
        avg_R=float(np.mean(r)),
        median_R=float(np.median(r)),
        win_rate_pct=float(100.0 * np.mean([x > 0 for x in r])),
        profit_factor=profit_factor(r),
        max_dd_R=compute_max_drawdown_r(r),
        best_trade_R=float(np.max(r)),
        worst_trade_R=float(np.min(r)),
        avg_bars_held=float(np.mean([t.bars_held for t in trades])),
        expectancy_R=float(np.mean(r)),
    )


def write_csvs(all_trades: List[Trade], summary_rows: List[SummaryRow]) -> None:
    trades_path = OUT_DIR / "phase_t1_trend_discovery_trades.csv"
    summary_path = OUT_DIR / "phase_t1_trend_discovery_summary.csv"

    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    summary_df = pd.DataFrame([asdict(s) for s in summary_rows])

    if not trades_df.empty:
        trades_df = trades_df.sort_values(["timeframe", "symbol", "entry_time"]).reset_index(drop=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["timeframe", "total_R", "profit_factor", "trades"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  Trades : {trades_path}")
    print(f"  Summary: {summary_path}")

    if not summary_df.empty:
        print("\nTop 10 configurations by total_R:")
        cols = [
            "timeframe", "side_mode", "donchian_entry", "donchian_exit",
            "atr_mult", "ema_slope_lookback", "trades", "total_R",
            "avg_R", "profit_factor", "max_dd_R"
        ]
        print(summary_df[cols].head(10).to_string(index=False))


def run(args: argparse.Namespace) -> None:
    ensure_dirs()

    exchange = init_exchange()

    symbols = get_top_usdt_spot_symbols(exchange, top_n=args.top_n)

    # Load/cache data first
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
            if df is None or len(df) < 250:
                continue
            data[(symbol, timeframe)] = df

    all_trades: List[Trade] = []
    summary_rows: List[SummaryRow] = []

    side_modes = []
    if args.long:
        side_modes.append("long")
    if args.short:
        side_modes.append("short")
    if args.both:
        side_modes.append("both")

    if not side_modes:
        side_modes = ["long", "short", "both"]

    for timeframe in args.timeframes:
        symbols_with_data = [s for s in symbols if (s, timeframe) in data]
        print(f"\n=== Backtest timeframe={timeframe}, symbols={len(symbols_with_data)} ===")

        for side_mode in side_modes:
            for donchian_entry in DONCHIAN_ENTRY_LIST:
                for donchian_exit in DONCHIAN_EXIT_LIST:
                    if donchian_exit >= donchian_entry:
                        continue
                    for atr_mult in ATR_MULT_LIST:
                        for ema_len in EMA_LEN_LIST:
                            for ema_slope_lookback in EMA_SLOPE_LOOKBACK_LIST:
                                combo_trades: List[Trade] = []

                                for symbol in symbols_with_data:
                                    df = data[(symbol, timeframe)]
                                    trades = backtest_symbol(
                                        df=df,
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        donchian_entry=donchian_entry,
                                        donchian_exit=donchian_exit,
                                        atr_mult=atr_mult,
                                        ema_len=ema_len,
                                        ema_slope_lookback=ema_slope_lookback,
                                        side_mode=side_mode,
                                    )
                                    combo_trades.extend(trades)

                                all_trades.extend(combo_trades)
                                summary_rows.append(summarize_trades(
                                    trades=combo_trades,
                                    timeframe=timeframe,
                                    side_mode=side_mode,
                                    donchian_entry=donchian_entry,
                                    donchian_exit=donchian_exit,
                                    atr_mult=atr_mult,
                                    ema_len=ema_len,
                                    ema_slope_lookback=ema_slope_lookback,
                                    symbols_tested=len(symbols_with_data),
                                ))

                                print(
                                    f"{timeframe} {side_mode:5s} "
                                    f"DE={donchian_entry:2d} DX={donchian_exit:2d} "
                                    f"ATR={atr_mult:.1f} slope={ema_slope_lookback:2d} "
                                    f"trades={len(combo_trades):4d} "
                                    f"R={sum(t.pnl_R for t in combo_trades):8.2f}"
                                )

    write_csvs(all_trades=all_trades, summary_rows=summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase T1 — Clean Trend Concept Discovery for Binance Spot"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Numero massimo di simboli USDT spot da testare. Default: {DEFAULT_TOP_N}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Numero candele richieste da Binance. Default: {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help='Timeframe da testare. Esempio: --timeframes 2h 4h',
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Riscarica i dati anche se già presenti in cache.",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Testa solo long.",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Testa solo short teorici.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Testa long+short nello stesso motore.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
