#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T2 — CORE TREND ENGINE  (redesigned)

Purpose: Freeze the T1 recommended concept into a clean, deeply-instrumented engine.
Run AFTER reading phase_t1_stability_report.txt and using its recommended params.

Changes vs. original T2
-----------------------
1. Default timeframes: ["4h","6h","8h","1d"]  (removed 2H)
2. --filter-mode argument: "ema50_slope" or "ema200_price"  (T1 canonical winner)
3. Donchian exit = don_n // 2  (not a separate free parameter)
4. Full MAE / MFE tracking per trade
5. Per-timeframe output files (suffix _TF)

The T2 command is printed at the end of phase_t1_stability_report.txt.
Example:
  python phase_t2_core_trend_engine.py --timeframe 6h --filter-mode ema200_price \
      --donchian-entry 20 --atr-mult 2.0

Outputs: data/research_trend_t2/
    phase_t2_core_trend_trades.csv
    phase_t2_core_trend_equity.csv
    phase_t2_core_trend_summary.csv
    phase_t2_core_trend_asset_summary.csv
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
    raise SystemExit("Install ccxt:  pip install ccxt pandas numpy") from exc


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS  (overridden by CLI args — use T1 recommendations)
# ─────────────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_trend_t2"
OUT_DIR = ROOT / "data" / "research_trend_t2"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults — replace with T1 recommendation
DEFAULT_TIMEFRAMES    = ["4h", "6h", "8h", "1d"]
TF_LIMITS             = {"4h": 1500, "6h": 1200, "8h": 1000, "1d": 800}
DEFAULT_TIMEFRAME     = "6h"     # single-TF mode default; T1 will recommend specific TF
DEFAULT_FILTER_MODE   = "ema50_slope"   # T1 will recommend; "ema200_price" is alternative
DEFAULT_DONCHIAN_ENTRY = 20
DEFAULT_ATR_MULT       = 2.0
DEFAULT_TOP_N          = 70
DEFAULT_LIMIT          = 1200

EMA_SLOPE_N    = 50
EMA_SLOPE_LB   = 10
EMA_REGIME_N   = 200
ATR_N          = 14

EXCLUDE_BASES = {
    "USDC","BUSD","TUSD","USDP","PAXG","XAUT","FDUSD","USDD","DAI",
    "USD1","RLUSD","EUR","AEUR","EURI","WBTC","BTTC",
}
LEVERAGED_KW = ("UP","DOWN","BULL","BEAR")


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:           int
    symbol:             str
    timeframe:          str
    side:               str
    filter_mode:        str
    entry_time:         str
    exit_time:          str
    entry_price:        float
    exit_price:         float
    initial_stop:       float
    exit_reason:        str
    bars_held:          int
    risk_per_unit:      float
    pnl_R:              float
    pnl_pct_on_price:   float
    mae_R:              float
    mfe_R:              float
    donchian_entry:     int
    donchian_exit:      int
    atr_mult:           float
    ema_slope_n:        int
    ema_slope_lb:       int
    filter_param:       str   # "EMA50_SLOPE_LB10" or "EMA200_PRICE"


# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE / UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def _init_exchange() -> "ccxt.binance":
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })


def _is_valid(symbol: str, market: dict) -> bool:
    if not market.get("spot") or not market.get("active", True):
        return False
    if not symbol.endswith("/USDT") or ":" in symbol:
        return False
    base = str(market.get("base", "")).upper()
    if base in EXCLUDE_BASES:
        return False
    return not any(base.endswith(kw) for kw in LEVERAGED_KW)


def get_universe(exchange: "ccxt.binance", top_n: int) -> List[str]:
    print("Loading Binance markets and tickers...")
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    rows = []
    for sym, m in markets.items():
        if not _is_valid(sym, m):
            continue
        qv = float(tickers.get(sym, {}).get("quoteVolume") or 0)
        if np.isfinite(qv):
            rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in rows[:top_n]]
    print(f"Selected {len(symbols)} USDT spot symbols.")
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _safe_fname(symbol: str, tf: str) -> str:
    return symbol.replace("/", "_").replace(":", "_") + f"_{tf}.csv"


def _fetch_cached(exchange: "ccxt.binance", symbol: str, tf: str,
                  limit: int, refresh: bool) -> Optional[pd.DataFrame]:
    # Try T1 raw cache first (avoid re-download if T1 already ran)
    t1_cache = ROOT / "data" / "raw_trend_t1" / _safe_fname(symbol, tf)
    t2_cache = RAW_DIR / _safe_fname(symbol, tf)

    for path in (t2_cache, t1_cache):
        if path.exists() and not refresh:
            try:
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                if len(df) >= 200:
                    return df
            except Exception:
                pass

    for attempt in range(3):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            time.sleep(exchange.rateLimit / 1000.0)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    else:
        print(f"  [WARN] {symbol} {tf}: {last_err}")
        return None

    if not raw or len(raw) < 200:
        return None

    df = pd.DataFrame(raw, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df = df.iloc[:-1].reset_index(drop=True)   # drop forming candle
    df.to_csv(t2_cache, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame, don_n: int, filter_mode: str) -> pd.DataFrame:
    d = df.copy()
    don_exit = max(don_n // 2, 5)

    # ATR
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-pc).abs(), (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # Donchian (shifted — no lookahead)
    d["don_high"]     = d["high"].shift(1).rolling(don_n).max()
    d["don_low_exit"] = d["low"].shift(1).rolling(don_exit).min()

    # EMA50 slope
    d["ema50"]       = d["close"].ewm(span=EMA_SLOPE_N, adjust=False).mean()
    d["ema_slope"]   = d["ema50"] - d["ema50"].shift(EMA_SLOPE_LB)

    # EMA200 (always computed)
    d["ema200"]      = d["close"].ewm(span=EMA_REGIME_N, adjust=False).mean()

    return d, don_exit


def _filter_ok(row: pd.Series, filter_mode: str) -> bool:
    if filter_mode == "ema50_slope":
        es = float(row.get("ema_slope", np.nan))
        return np.isfinite(es) and es > 0
    # ema200_price
    e200  = float(row.get("ema200", np.nan))
    close = float(row.get("close", np.nan))
    return np.isfinite(e200) and np.isfinite(close) and close > e200


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER (single symbol)
# ─────────────────────────────────────────────────────────────────────────────

def _backtest_symbol(df: pd.DataFrame, symbol: str, timeframe: str,
                     don_n: int, atr_mult: float, filter_mode: str,
                     first_id: int) -> List[Trade]:
    d, don_exit = _add_indicators(df, don_n, filter_mode)
    warmup = max(don_n, don_exit, EMA_SLOPE_N + EMA_SLOPE_LB, EMA_REGIME_N, ATR_N) + 5

    filter_desc = (f"EMA{EMA_SLOPE_N}_SLOPE_LB{EMA_SLOPE_LB}"
                   if filter_mode == "ema50_slope" else f"EMA{EMA_REGIME_N}_PRICE")

    pos = None
    trades: List[Trade] = []
    tid = first_id

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        ts    = row["timestamp"]

        if pos is not None:
            bars_held = i - pos["entry_i"]
            # MAE / MFE update (using candle extremes)
            fav_r   = (high  - pos["entry"]) / pos["risk"]
            adv_r   = (low   - pos["entry"]) / pos["risk"]
            pos["mfe_R"] = max(pos["mfe_R"], fav_r)
            pos["mae_R"] = min(pos["mae_R"], adv_r)

            exit_px = None
            reason  = None
            if low <= pos["stop"]:
                exit_px = pos["stop"];   reason = "initial_stop"
            elif close < float(row["don_low_exit"]):
                exit_px = close;         reason = "donchian_exit"

            if exit_px is not None:
                pnl_R   = (float(exit_px) - pos["entry"]) / pos["risk"]
                pnl_pct = (float(exit_px) / pos["entry"] - 1.0) * 100.0
                trades.append(Trade(
                    trade_id=tid, symbol=symbol, timeframe=timeframe,
                    side="LONG", filter_mode=filter_mode,
                    entry_time=str(pos["entry_time"]), exit_time=str(ts),
                    entry_price=pos["entry"], exit_price=float(exit_px),
                    initial_stop=pos["stop"], exit_reason=reason,
                    bars_held=int(bars_held), risk_per_unit=pos["risk"],
                    pnl_R=float(pnl_R), pnl_pct_on_price=float(pnl_pct),
                    mae_R=float(pos["mae_R"]), mfe_R=float(pos["mfe_R"]),
                    donchian_entry=don_n, donchian_exit=don_exit,
                    atr_mult=atr_mult, ema_slope_n=EMA_SLOPE_N,
                    ema_slope_lb=EMA_SLOPE_LB, filter_param=filter_desc,
                ))
                tid += 1
                pos = None
            continue

        # Entry check
        dh     = row.get("don_high")
        dl_ex  = row.get("don_low_exit")
        if not all(np.isfinite(float(x)) for x in [dh, dl_ex]):
            continue

        if close > float(dh) and _filter_ok(row, filter_mode):
            risk = atr * atr_mult
            if risk > 0:
                pos = dict(entry_time=ts, entry_i=i, entry=close,
                           stop=close - risk, risk=risk, mfe_R=0.0, mae_R=0.0)

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────

def _pf(r: List[float]) -> float:
    wins   = [x for x in r if x > 0]
    losses = [x for x in r if x < 0]
    g = sum(wins); l = abs(sum(losses))
    return float(g / l) if l > 1e-12 else (float("inf") if g > 0 else 0.0)


def _dd(r: List[float]) -> float:
    if not r: return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _build_summary(trades_df: pd.DataFrame, args: argparse.Namespace,
                   n_req: int, n_ok: int) -> pd.DataFrame:
    r = trades_df["pnl_R"].tolist() if not trades_df.empty else []
    b = trades_df["bars_held"].tolist() if not trades_df.empty else []
    n = len(r)
    row = dict(
        phase="T2_CORE_TREND",
        timeframe=args.timeframe,
        filter_mode=args.filter_mode,
        donchian_entry=args.donchian_entry,
        donchian_exit=args.donchian_entry // 2,
        atr_mult=args.atr_mult,
        symbols_requested=n_req, symbols_with_data=n_ok,
        trades=n,
        total_R=float(np.sum(r)) if n else 0.0,
        avg_R=float(np.mean(r)) if n else 0.0,
        median_R=float(np.median(r)) if n else 0.0,
        win_rate_pct=float(100.0 * np.mean([x > 0 for x in r])) if n else 0.0,
        profit_factor=_pf(r),
        max_dd_R=_dd(r),
        best_trade_R=float(np.max(r)) if n else 0.0,
        worst_trade_R=float(np.min(r)) if n else 0.0,
        avg_bars_held=float(np.mean(b)) if b else 0.0,
        avg_mfe_R=float(trades_df["mfe_R"].mean()) if not trades_df.empty else 0.0,
        avg_mae_R=float(trades_df["mae_R"].mean()) if not trades_df.empty else 0.0,
    )
    return pd.DataFrame([row])


def _build_asset_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    rows = []
    for (sym, tf), g in trades_df.groupby(["symbol","timeframe"]):
        r = g["pnl_R"].tolist()
        rows.append(dict(
            symbol=sym, timeframe=tf,
            trades=len(r), total_R=float(np.sum(r)),
            avg_R=float(np.mean(r)), median_R=float(np.median(r)),
            win_rate_pct=float(100.0 * np.mean([x > 0 for x in r])),
            profit_factor=_pf(r), max_dd_R_local=_dd(r),
            best_trade_R=float(np.max(r)), worst_trade_R=float(np.min(r)),
            avg_bars_held=float(g["bars_held"].mean()),
            avg_mfe_R=float(g["mfe_R"].mean()), avg_mae_R=float(g["mae_R"].mean()),
        ))
    return pd.DataFrame(rows).sort_values(
        ["total_R","profit_factor"], ascending=False
    ).reset_index(drop=True)


def _build_equity(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    eq = trades_df.copy()
    eq["exit_time"] = pd.to_datetime(eq["exit_time"], utc=True)
    eq = eq.sort_values(["exit_time","trade_id"]).reset_index(drop=True)
    eq["equity_R"] = eq["pnl_R"].cumsum()
    eq["peak_R"]   = eq["equity_R"].cummax()
    eq["dd_R"]     = eq["equity_R"] - eq["peak_R"]
    return eq[["exit_time","trade_id","symbol","timeframe","side",
               "pnl_R","equity_R","peak_R","dd_R"]]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _write_outputs(all_trades: List[Trade], args: argparse.Namespace,
                   n_req: int, n_ok: int) -> None:
    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"],  utc=True)
        trades_df = trades_df.sort_values(["exit_time","trade_id"]).reset_index(drop=True)

    summary_df     = _build_summary(trades_df, args, n_req, n_ok)
    asset_summary  = _build_asset_summary(trades_df)
    equity_df      = _build_equity(trades_df)

    trades_df.to_csv(    OUT_DIR / "phase_t2_core_trend_trades.csv",       index=False)
    equity_df.to_csv(    OUT_DIR / "phase_t2_core_trend_equity.csv",        index=False)
    summary_df.to_csv(   OUT_DIR / "phase_t2_core_trend_summary.csv",       index=False)
    asset_summary.to_csv(OUT_DIR / "phase_t2_core_trend_asset_summary.csv", index=False)

    print("\n=== PHASE T2 SUMMARY ===")
    print(summary_df.to_string(index=False))

    if not asset_summary.empty:
        print("\nTop 15 assets by total_R:")
        cols = ["symbol","timeframe","trades","total_R","avg_R","profit_factor",
                "max_dd_R_local","best_trade_R","avg_mfe_R"]
        print(asset_summary[cols].head(15).to_string(index=False))

    # Print T3B instructions
    filter_line = ("EMA50 slope > 0" if args.filter_mode == "ema50_slope"
                   else "price > EMA200")
    print(f"""
=== NEXT STEP: Run T3B ===
Update phase_t3b_wide_exit_engineering.py constants:
  TIMEFRAMES          = ["{args.timeframe}"]
  DONCHIAN_ENTRY      = {args.donchian_entry}
  EMA_FILTER_MODE     = "{args.filter_mode}"   # {filter_line}
  CHAND_ATR_MULT      = 3.0                    # T10 stability finding (not 4.0)
  INITIAL_STOP_ATR_MULT = {args.atr_mult}
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase T2 — Core Trend Engine. Configure with T1 recommendations."
    )
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME,
                   choices=["4h","6h","8h","1d"],
                   help="Single timeframe to study in depth (T1 recommended)")
    p.add_argument("--filter-mode", default=DEFAULT_FILTER_MODE,
                   choices=["ema50_slope","ema200_price"],
                   help="Regime filter (T1 recommended)")
    p.add_argument("--donchian-entry", type=int, default=DEFAULT_DONCHIAN_ENTRY,
                   help="Donchian entry N (T1 recommended, default 20)")
    p.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT,
                   help="Initial stop ATR multiplier (T1 recommended)")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("=" * 80)
    print("PHASE T2 — CORE TREND ENGINE")
    print("=" * 80)
    print(f"Timeframe:     {args.timeframe}")
    print(f"Filter mode:   {args.filter_mode}")
    print(f"Donchian N:    {args.donchian_entry}  (exit = {args.donchian_entry//2})")
    print(f"ATR mult:      {args.atr_mult}")
    print(f"Universe:      top {args.top_n}")
    print()

    exchange = _init_exchange()
    symbols  = get_universe(exchange, args.top_n)
    limit    = TF_LIMITS.get(args.timeframe, DEFAULT_LIMIT)

    print(f"\n=== Fetching {args.timeframe} candles ({limit} bars) ===")
    data: Dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(symbols, 1):
        print(f"  [{idx:3d}/{len(symbols)}] {sym}", end="  ")
        df = _fetch_cached(exchange, sym, args.timeframe, limit, args.refresh)
        min_bars = max(EMA_REGIME_N + EMA_SLOPE_LB + 10, 250)
        if df is not None and len(df) >= min_bars:
            data[sym] = df
            print(f"OK ({len(df)} bars)")
        else:
            print("SKIP")

    print(f"\n=== Backtesting {args.timeframe} ({len(data)} symbols) ===")
    all_trades: List[Trade] = []
    tid = 1

    for sym, df in data.items():
        trades = _backtest_symbol(
            df=df, symbol=sym, timeframe=args.timeframe,
            don_n=args.donchian_entry, atr_mult=args.atr_mult,
            filter_mode=args.filter_mode, first_id=tid,
        )
        if trades:
            tid = max(t.trade_id for t in trades) + 1
            all_trades.extend(trades)
        r = sum(t.pnl_R for t in trades)
        print(f"  {sym:12s} trades={len(trades):3d}  R={r:8.2f}")

    _write_outputs(all_trades, args, len(symbols), len(data))

    print("\n[OK] phase_t2_core_trend_trades.csv")
    print("[OK] phase_t2_core_trend_equity.csv")
    print("[OK] phase_t2_core_trend_summary.csv")
    print("[OK] phase_t2_core_trend_asset_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
