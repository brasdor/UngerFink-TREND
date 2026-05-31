#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T1 -- PreviousSessionBreakout Concept Discovery

Entry (no lookahead):
  LONG:  close > prev_session_high  AND  close > EMA(200)
  SHORT: close < prev_session_low   AND  close < EMA(200)

Session = UTC calendar day.
Exit: initial stop hit  OR  end of same UTC session (last bar of day).

Parameter grid:
  Timeframes:   1h, 2h, 4h
  Stop mults:   [1.5, 2.0, 3.0]  (ATR(14) multiple)
  Time filters: full | first_half (entry bar hour UTC < 12) | second_half (>= 12)
  Sides:        LONG, SHORT

One trade per symbol per session (no re-entry after exit within same UTC day).
Entry price = close of signal bar.

§4.2 cost floor: 0.15R
Output: data/research_prevsessionbreakout_t1/
"""

from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    sys.exit("ccxt not installed: pip install ccxt")


# =============================================================================
# CONFIG
# =============================================================================

ROOT      = Path(__file__).parent
D1_CACHE  = ROOT / "data" / "research_trend_t15"   / "ohlcv_cache"
H4_CACHE  = ROOT / "data" / "research_linregbreakout_t1" / "ohlcv_cache"
CACHE_DIR = ROOT / "data" / "research_prevsessionbreakout_t1" / "ohlcv_cache"
OUT_DIR   = ROOT / "data" / "research_prevsessionbreakout_t1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAMES     = ["1h", "2h", "4h"]
TF_TARGET_BARS = {"1h": 25_000, "2h": 13_000, "4h": 7_000}
TF_MS          = {"1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000}

STOP_MULTS   = [1.5, 2.0, 3.0]
TIME_FILTERS = ["full", "first_half", "second_half"]
SIDES        = ["LONG", "SHORT"]

ATR_N        = 14
EMA_N        = 200
COST_FLOOR_R = 0.15   # §4.2


# =============================================================================
# INDICATORS
# =============================================================================

def compute_ema(close: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with SMA of first n bars."""
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothed ATR."""
    nb = len(close)
    tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1 : n + 1]))
        for i in range(n + 1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i - 1]):
                atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


# =============================================================================
# DATA LOADING
# =============================================================================

def get_symbols() -> List[str]:
    """Symbol list from 1D cache (*.csv  →  BTC/USDT etc.)."""
    syms = []
    for f in sorted(D1_CACHE.glob("*_1d.csv")):
        stem = f.stem[: -len("_1d")]   # strip trailing _1d
        sym  = stem.replace("_", "/", 1)
        if sym.endswith("/USDT"):
            syms.append(sym)
    return syms


def _load_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV CSV; handle both Unix-ms and datetime-string timestamps."""
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    df = df.rename(columns={"open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]].dropna(subset=["time"])


def _download(exchange, symbol: str, tf: str, target_bars: int) -> Optional[pd.DataFrame]:
    bar_ms = TF_MS[tf]
    since  = int(pd.Timestamp.utcnow().timestamp() * 1_000) - target_bars * bar_ms
    rows: List = []
    limit = 1_000
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        except Exception as exc:
            print(f"    [WARN] {symbol} {tf}: {exc}")
            break
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1
        time_mod.sleep(0.05)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def get_ohlcv(symbol: str, tf: str, exchange) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    fname = f"{clean}_{tf}.csv"
    path  = CACHE_DIR / fname

    if path.exists():
        return _load_csv(path)

    # Reuse existing 4H or 1D cache
    if tf == "4h":
        alt = H4_CACHE / fname
        if alt.exists():
            df = _load_csv(alt)
            df.to_csv(path, index=False)
            return df

    if tf == "1d":
        alt = D1_CACHE / f"{clean}_1d.csv"
        if alt.exists():
            return _load_csv(alt)
        return None

    # Download
    df = _download(exchange, symbol, tf, TF_TARGET_BARS[tf])
    if df is not None:
        df.to_csv(path, index=False)
    return df


# =============================================================================
# INDICATOR PREP
# =============================================================================

def prepare(raw: pd.DataFrame, symbol: str, tf: str) -> Optional[pd.DataFrame]:
    df = raw.copy()
    df["symbol"]    = symbol
    df["timeframe"] = tf
    df = df.sort_values("time").reset_index(drop=True)

    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour

    # Last-bar-of-day flag (per date group, iloc-based)
    df["is_last_bar"] = False
    for _, g in df.groupby("date"):
        df.loc[g.index[-1], "is_last_bar"] = True

    # Precompute day-end iloc for fast exit scan
    day_end = np.zeros(len(df), dtype=np.intp)
    for _, g in df.groupby("date"):
        end_iloc = g.index[-1]
        day_end[g.index] = end_iloc
    df["day_end"] = day_end

    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)

    df["ema200"] = compute_ema(close, EMA_N)
    df["atr14"]  = compute_atr(high, low, close, ATR_N)

    # Prev-session high/low: daily aggregation then shift-by-1
    daily = (
        df.groupby("date", sort=True)
        .agg(sh=("high", "max"), sl=("low", "min"))
        .reset_index()
    )
    daily["prev_sh"] = daily["sh"].shift(1)
    daily["prev_sl"] = daily["sl"].shift(1)
    df = df.merge(daily[["date", "prev_sh", "prev_sl"]], on="date", how="left")

    # Integer date for vectorised deduplication (day-level uniqueness)
    df["date_int"] = pd.to_datetime(df["date"]).astype("int64")

    if len(df) < EMA_N + 100:
        return None
    return df.reset_index(drop=True)


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_symbol(df: pd.DataFrame) -> List[dict]:
    close   = df["close"].to_numpy(dtype=float)
    high    = df["high"].to_numpy(dtype=float)
    low     = df["low"].to_numpy(dtype=float)
    ema200  = df["ema200"].to_numpy(dtype=float)
    atr14   = df["atr14"].to_numpy(dtype=float)
    prev_sh = df["prev_sh"].to_numpy(dtype=float)
    prev_sl = df["prev_sl"].to_numpy(dtype=float)
    hours   = df["hour"].to_numpy(dtype=int)
    is_last = df["is_last_bar"].to_numpy(dtype=bool)
    day_end = df["day_end"].to_numpy(dtype=np.intp)
    date_i  = df["date_int"].to_numpy(dtype=np.int64)
    times   = df["time"].to_numpy()

    symbol = str(df["symbol"].iloc[0])
    tf     = str(df["timeframe"].iloc[0])
    n      = len(close)

    # Shared validity: warm-up done, all indicators finite, not last bar
    valid = np.zeros(n, dtype=bool)
    for i in range(EMA_N, n):
        valid[i] = (
            np.isfinite(close[i]) and np.isfinite(ema200[i]) and
            np.isfinite(atr14[i]) and atr14[i] > 0 and
            np.isfinite(prev_sh[i]) and np.isfinite(prev_sl[i]) and
            not is_last[i]
        )

    trades: List[dict] = []

    for side in SIDES:
        # Signal array (ignoring time-filter and one-per-day)
        if side == "LONG":
            sig = valid & (close > ema200) & (close > prev_sh)
        else:
            sig = valid & (close < ema200) & (close < prev_sl)
        sig_idx = np.where(sig)[0]

        for tf_filt in TIME_FILTERS:
            # Apply time filter
            if tf_filt == "first_half":
                filt = sig_idx[hours[sig_idx] < 12]
            elif tf_filt == "second_half":
                filt = sig_idx[hours[sig_idx] >= 12]
            else:
                filt = sig_idx

            if len(filt) == 0:
                continue

            # One entry per UTC day: keep first signal of each day
            sig_dates = date_i[filt]
            _, first  = np.unique(sig_dates, return_index=True)
            entry_bars = filt[first]

            for entry_i in entry_bars:
                entry_px = close[entry_i]
                atr_val  = atr14[entry_i]
                j_start  = entry_i + 1
                j_end    = int(day_end[entry_i]) + 1  # exclusive upper bound

                for sm in STOP_MULTS:
                    risk = atr_val * sm
                    if risk <= 0:
                        continue

                    if side == "LONG":
                        stop_px = entry_px - risk
                        window  = low[j_start:j_end]
                        cond    = window <= stop_px
                    else:
                        stop_px = entry_px + risk
                        window  = high[j_start:j_end]
                        cond    = window >= stop_px

                    if j_start >= j_end:
                        # Entry was last bar — skip (shouldn't happen due to valid mask)
                        continue

                    if cond.any():
                        hit_rel  = int(np.argmax(cond))
                        exit_i   = j_start + hit_rel
                        exit_px  = float(stop_px)
                        exit_rsn = "initial_stop"
                    else:
                        exit_i   = j_end - 1
                        exit_px  = float(close[j_end - 1])
                        exit_rsn = "session_end"

                    if side == "LONG":
                        net_r = (exit_px - entry_px) / risk
                    else:
                        net_r = (entry_px - exit_px) / risk

                    trades.append({
                        "symbol":       symbol,
                        "timeframe":    tf,
                        "side":         side,
                        "time_filter":  tf_filt,
                        "stop_mult":    sm,
                        "entry_time":   times[entry_i],
                        "exit_time":    times[exit_i],
                        "entry_price":  float(entry_px),
                        "exit_price":   float(exit_px),
                        "initial_stop": float(stop_px),
                        "exit_reason":  exit_rsn,
                        "net_r":        float(net_r),
                    })

    return trades


# =============================================================================
# AGGREGATION & STABILITY
# =============================================================================

def aggregate(all_trades: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_trades)
    rows = []
    for key, g in df.groupby(["timeframe", "side", "time_filter", "stop_mult"], sort=True):
        r  = g["net_r"].to_numpy(dtype=float)
        g_r = r[r > 0].sum()
        l_r = -r[r < 0].sum()
        pf  = (g_r / l_r) if l_r > 0 else (float("inf") if g_r > 0 else 0.0)
        eq  = np.cumsum(r)
        dd  = float((eq - np.maximum.accumulate(eq)).min())
        rows.append({
            "timeframe":       key[0], "side": key[1],
            "time_filter":     key[2], "stop_mult": key[3],
            "trades":          int(len(r)),
            "total_r":         float(r.sum()),
            "avg_r":           float(r.mean()),
            "win_rate":        float((r > 0).mean()),
            "profit_factor":   pf,
            "max_dd_r":        dd,
            "pass_cost_floor": bool(r.mean() >= COST_FLOOR_R),
        })
    return pd.DataFrame(rows)


def stability_zones(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["zone_pass_rate"]      = 0.0
    df["stability_zone_pass"] = False

    for (tf, side, tf_f), g in df.groupby(["timeframe", "side", "time_filter"]):
        pass_map = dict(zip(g["stop_mult"], g["pass_cost_floor"]))
        for idx, row in g.iterrows():
            sm_i  = STOP_MULTS.index(row["stop_mult"])
            zone  = [STOP_MULTS[sm_i]]
            if sm_i > 0:
                zone.insert(0, STOP_MULTS[sm_i - 1])
            if sm_i < len(STOP_MULTS) - 1:
                zone.append(STOP_MULTS[sm_i + 1])
            rate = sum(pass_map.get(z, False) for z in zone) / len(zone)
            df.at[idx, "zone_pass_rate"]      = rate
            df.at[idx, "stability_zone_pass"] = rate >= 2 / 3

    return df


# =============================================================================
# REPORT
# =============================================================================

def write_report(results: pd.DataFrame) -> None:
    lines = [
        "PHASE T1 -- PreviousSessionBreakout Concept Discovery",
        "=" * 70,
        "",
        "Entry:  LONG  if close > prev-UTC-day high AND close > EMA(200)",
        "        SHORT if close < prev-UTC-day low  AND close < EMA(200)",
        "Exit:   Initial stop (ATR(14)*mult) OR last bar of same UTC session",
        "Rule:   One trade per symbol per UTC session (no intraday re-entry)",
        f"§4.2 cost floor: {COST_FLOOR_R}R",
        "",
    ]

    for tf in TIMEFRAMES:
        lines += [f"{'='*70}", f"TIMEFRAME: {tf}", f"{'='*70}", ""]
        sub = results[results["timeframe"] == tf]
        if sub.empty:
            lines += ["  (no results)", ""]
            continue
        lines.append(
            f"  {'TF_FILTER':12s} {'SIDE':5s} {'SM':4s}  {'N':6s}  "
            f"{'AVG_R':7s}  {'PF':5s}  {'WIN%':5s}  {'DD':7s}  PASS  ZONE"
        )
        lines.append("  " + "-" * 72)
        for _, r in sub.sort_values(["side", "time_filter", "stop_mult"]).iterrows():
            lines.append(
                f"  {r['time_filter']:12s} {r['side']:5s} {r['stop_mult']:.1f}"
                f"  {int(r['trades']):6d}  {r['avg_r']:+.4f}  {r['profit_factor']:.3f}"
                f"  {r['win_rate']:.1%}  {r['max_dd_r']:+.2f}"
                f"  {'PASS' if r['pass_cost_floor'] else '    '}"
                f"  {'ZONE' if r['stability_zone_pass'] else '    '}"
            )
        lines.append("")

    # Stability ranking
    lines += ["=" * 70, "STABILITY RANKING  (avg_r ≥ 0.15R  +  zone_pass)", "=" * 70, ""]
    top = results[results["stability_zone_pass"]].sort_values("avg_r", ascending=False)
    if top.empty:
        lines.append("  No combo passes both §4.2 and stability zone.")
    else:
        for _, r in top.iterrows():
            lines.append(
                f"  [{r['timeframe']:3s}|{r['side']:5s}|{r['time_filter']:12s}|sm={r['stop_mult']:.1f}]"
                f"  avg_r={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}"
                f"  trades={int(r['trades'])}  zone={r['zone_pass_rate']:.0%}"
            )

    lines += [
        "",
        "=" * 70,
        "NEXT STEPS",
        "=" * 70,
        "",
        "  T2 candidates: rows where avg_r >= 0.15R AND stability_zone_pass = True",
        "  Do not proceed to T2 without human review.",
    ]

    rpt = OUT_DIR / "phase_t1_prevsessionbreakout_summary.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Report: {rpt}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE T1 -- PreviousSessionBreakout Concept Discovery")
    print("=" * 70)

    symbols = get_symbols()
    print(f"Universe: {len(symbols)} symbols")
    print(f"Grid: TFs={TIMEFRAMES}  stops={STOP_MULTS}  filters={TIME_FILTERS}  sides={SIDES}")
    print(f"Output: {OUT_DIR}")
    print()

    exchange = ccxt.binance({"enableRateLimit": True})

    all_trades: List[dict] = []

    for tf in TIMEFRAMES:
        print(f"[TF={tf}] loading + backtesting...")
        loaded = skipped = 0
        for sym in symbols:
            raw = get_ohlcv(sym, tf, exchange)
            if raw is None or len(raw) < EMA_N + 100:
                skipped += 1
                continue
            df = prepare(raw, sym, tf)
            if df is None:
                skipped += 1
                continue
            trades = backtest_symbol(df)
            all_trades.extend(trades)
            loaded += 1
        print(f"  Loaded={loaded}  Skipped={skipped}  Trades_so_far={len(all_trades)}")

    if not all_trades:
        print("[ERROR] No trades generated.")
        return 1

    print(f"\nTotal trades across all combos: {len(all_trades)}")

    results = stability_zones(aggregate(all_trades))

    pd.DataFrame(all_trades).to_csv(
        OUT_DIR / "phase_t1_prevsessionbreakout_trades.csv", index=False
    )
    results.to_csv(
        OUT_DIR / "phase_t1_prevsessionbreakout_results.csv", index=False
    )
    write_report(results)

    # Console summary
    print("\n" + "=" * 70)
    print("TOP 15 COMBOS BY avg_r  (pass_cost_floor=True only)")
    print("=" * 70)
    top = results[results["pass_cost_floor"]].sort_values("avg_r", ascending=False).head(15)
    if top.empty:
        print("  None pass §4.2 cost floor.")
    else:
        for _, r in top.iterrows():
            print(
                f"  [{r['timeframe']:2s}|{r['side']:5s}|{r['time_filter']:12s}|sm={r['stop_mult']:.1f}]"
                f"  avg_r={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}"
                f"  trades={int(r['trades'])}"
                f"  {'ZONE' if r['stability_zone_pass'] else '    '}"
            )

    print()
    print("=" * 70)
    print("T1 COMPLETE -- awaiting human review before T2")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
