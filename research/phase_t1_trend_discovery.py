#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T1 — DONCHIAN CHANNEL BREAKOUT CONCEPT DISCOVERY  (redesigned)

Unger §2.1: "The Donchian Channel is a price channel formed by two bands tracing
the highest high and the lowest low of the last N periods, and is basically
employed in trend-following systems."

Key Unger principle: "optimizing the lookback parameter may improve the system
but the STABILITY AROUND the chosen value is what matters."

Changes vs. original T1
-----------------------
1. Timeframes: ["4h","6h","8h","1d"]  (removed 2H — too noisy per Unger §4.8)
2. Donchian N grid: [10,15,20,25,30,40,55]  (stability analysis around N=20)
3. Filter modes: ["ema50_slope","ema200_price"]  (tests both; Unger §3.1 recommends EMA200)
4. Donchian exit = don_n // 2  (classic Turtle rule, not a free parameter)
5. Universe: top 70 USDT spot symbols (consistent with T2–T14)
6. New output: phase_t1_stability_report.txt  (canonical parameter recommendation)

Side: LONG only (Binance Spot — no direct shorting)

Grid size: 4 TF × 7 N × 2 filters × 2 ATR mults = 112 combos × 70 symbols

Outputs: data/research_trend_t1/
    phase_t1_trend_discovery_trades.csv
    phase_t1_trend_discovery_summary.csv
    phase_t1_stability_report.txt    ← READ THIS to configure T2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError as exc:
    raise SystemExit("Install ccxt:  pip install ccxt pandas numpy") from exc


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_trend_t1"
OUT_DIR = ROOT / "data" / "research_trend_t1"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Timeframes — multi-day, aligned with Unger §2.1 Donchian channel approach
TIMEFRAMES    = ["4h", "6h", "8h", "1d"]
TF_LIMITS     = {"4h": 1500, "6h": 1200, "8h": 1000, "1d": 1500}  # Option B: ~4yr 1d history
TF_HOURS      = {"4h": 4, "6h": 6, "8h": 8, "1d": 24}

# Donchian N stability grid around the canonical N=20
DONCHIAN_NS   = [10, 15, 20, 25, 30, 40, 55]
# Exit = don_n // 2  (Turtle Trading canonical: 20-entry / 10-exit)
# This is not a free parameter — it is derived from entry N.

# Regime filter variants — both tested simultaneously
FILTER_MODES  = ["ema50_slope", "ema200_price"]

# Initial stop ATR multiples
ATR_MULTS     = [2.0, 3.0]

# Fixed
EMA_SLOPE_N   = 50
EMA_SLOPE_LB  = 10
EMA_REGIME_N  = 200
ATR_N         = 14
UNIVERSE_SIZE = 70

# Stability zone: N values "around" the canonical N=20
STABILITY_ZONE_NS = [15, 20, 25]    # ±1 step from canonical
STABILITY_PASS_PCT = 67.0           # ≥ 67% of zone combos profitable = PASS

# Universe exclusions
EXCLUDE_BASES = {
    "USDC","BUSD","TUSD","USDP","PAXG","XAUT","FDUSD","USDD","DAI",
    "USD1","RLUSD","EUR","AEUR","EURI","WBTC","BTTC",
}
LEVERAGED_KW = ("UP","DOWN","BULL","BEAR")


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
    if any(base.endswith(kw) for kw in LEVERAGED_KW):
        return False
    return True


def get_universe(exchange: "ccxt.binance") -> List[str]:
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
    symbols = [s for s, _ in rows[:UNIVERSE_SIZE]]
    print(f"Selected {len(symbols)} USDT spot symbols.")
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _safe_fname(symbol: str, tf: str) -> str:
    return symbol.replace("/", "_").replace(":", "_") + f"_{tf}.csv"


def fetch_cached(exchange: "ccxt.binance", symbol: str, tf: str,
                 refresh: bool = False) -> Optional[pd.DataFrame]:
    path = RAW_DIR / _safe_fname(symbol, tf)
    limit = TF_LIMITS[tf]

    if path.exists() and not refresh:
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df if len(df) >= 200 else None
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
    # Drop the last (currently forming) candle
    df = df.iloc[:-1].reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame, don_n: int) -> pd.DataFrame:
    d = df.copy()
    don_exit = max(don_n // 2, 5)

    # ATR (fixed at ATR_N=14)
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-pc).abs(), (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()

    # Donchian entry bands (shifted — no lookahead)
    d["don_high"]     = d["high"].shift(1).rolling(don_n).max()
    d["don_low"]      = d["low"].shift(1).rolling(don_n).min()

    # Donchian exit band (N//2 — Turtle Trading canonical)
    d["don_low_exit"] = d["low"].shift(1).rolling(don_exit).min()

    # EMA50 slope filter (ema50_slope mode)
    d["ema50"]        = d["close"].ewm(span=EMA_SLOPE_N, adjust=False).mean()
    d["ema_slope"]    = d["ema50"] - d["ema50"].shift(EMA_SLOPE_LB)

    # EMA200 price filter (ema200_price mode)
    d["ema200"]       = d["close"].ewm(span=EMA_REGIME_N, adjust=False).mean()

    return d, don_exit


def _filter_ok(row: pd.Series, filter_mode: str) -> bool:
    """Return True if the regime filter allows a LONG entry on this bar."""
    if filter_mode == "ema50_slope":
        es = float(row.get("ema_slope", np.nan))
        return np.isfinite(es) and es > 0
    # ema200_price
    e200 = float(row.get("ema200", np.nan))
    close = float(row.get("close", np.nan))
    return np.isfinite(e200) and np.isfinite(close) and close > e200


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER (single symbol)
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(sym: str, df: pd.DataFrame, don_n: int, filter_mode: str,
              atr_mult: float) -> list[dict]:
    d, don_exit = _add_indicators(df, don_n)

    # Warmup: need enough bars for all indicators
    warmup = max(don_n, don_exit, EMA_SLOPE_N + EMA_SLOPE_LB, EMA_REGIME_N, ATR_N) + 5

    pos = None
    trades = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        t     = row["timestamp"]
        dh    = row.get("don_high")
        dl    = row.get("don_low")
        dl_ex = row.get("don_low_exit")

        if not all(np.isfinite(float(x)) for x in [dh, dl, dl_ex]):
            continue

        if pos is not None:
            # Exit conditions: initial stop OR Donchian exit
            if low <= pos["stop"]:
                r = (pos["stop"] - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(sym, pos, t, pos["stop"], r,
                                          "initial_stop", don_n, don_exit,
                                          filter_mode, atr_mult))
                pos = None
                continue
            elif close < float(dl_ex):
                r = (close - pos["entry"]) / pos["risk"]
                trades.append(_make_trade(sym, pos, t, close, r,
                                          "donchian_exit", don_n, don_exit,
                                          filter_mode, atr_mult))
                pos = None
                continue

        if pos is None:
            long_signal = close > float(dh) and _filter_ok(row, filter_mode)
            if long_signal:
                risk = atr * atr_mult
                if risk > 0:
                    pos = dict(entry_time=t, entry=close,
                               stop=close - risk, risk=risk)

    return trades


def _make_trade(sym, pos, exit_time, exit_px, net_r, reason,
                don_n, don_exit, filter_mode, atr_mult):
    return {
        "symbol":       sym,
        "side":         "LONG",
        "filter_mode":  filter_mode,
        "don_n":        don_n,
        "don_exit_n":   don_exit,
        "atr_mult":     atr_mult,
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "entry_price":  pos["entry"],
        "exit_price":   float(exit_px),
        "initial_stop": pos["entry"] - pos["risk"],
        "initial_risk": pos["risk"],
        "net_r":        float(net_r),
        "exit_reason":  reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pf(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    return float(g / l) if l > 1e-12 else (float("inf") if g > 0 else 0.0)


def _dd(r: np.ndarray) -> float:
    if not len(r): return 0.0
    eq = np.cumsum(r)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, median_r=0.0)
    r = np.array([t["net_r"] for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0, median_r=0.0)
    return dict(
        trades=len(r), total_r=float(r.sum()), avg_r=float(r.mean()),
        median_r=float(np.median(r)), pf=_pf(r), max_dd_r=_dd(r),
        win_rate_pct=float((r > 0).mean() * 100),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    """
    For each (timeframe, filter_mode, atr_mult) group:
    - Count % of STABILITY_ZONE_NS combos that are profitable (PF > 1.0)
    - Rank groups by zone stability %
    """
    rows = []
    for (tf, fm, am), g in grid.groupby(["timeframe","filter_mode","atr_mult"]):
        zone = g[g["don_n"].isin(STABILITY_ZONE_NS)]
        total_zone  = len(zone)
        pf_gt1      = int((zone["pf"] > 1.0).sum())
        avg_r_pos   = int((zone["avg_r"] > 0).sum())
        strong      = int(((zone["pf"] > 1.2) & (zone["avg_r"] > 0.05)).sum())
        pct         = 100.0 * pf_gt1 / max(total_zone, 1)

        # Stats for canonical N=20 specifically
        can20 = g[g["don_n"] == 20]
        can_trades   = int(can20["trades"].sum()) if not can20.empty else 0
        can_avg_r    = float(can20["avg_r"].mean()) if not can20.empty else 0.0
        can_pf       = float(can20["pf"].mean()) if not can20.empty else 0.0
        can_dd       = float(can20["max_dd_r"].mean()) if not can20.empty else 0.0

        rows.append(dict(
            timeframe=tf, filter_mode=fm, atr_mult=am,
            zone_total=total_zone, zone_pf_gt1=pf_gt1,
            zone_avg_r_positive=avg_r_pos, zone_strong=strong,
            zone_stability_pct=round(pct, 1),
            stability_verdict="PASS" if pct >= STABILITY_PASS_PCT else (
                "WARN" if pct >= 40 else "FAIL"),
            can_n20_trades=can_trades, can_n20_avg_r=round(can_avg_r, 4),
            can_n20_pf=round(can_pf, 3), can_n20_max_dd_r=round(can_dd, 2),
        ))

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct","can_n20_pf"], ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(grid: pd.DataFrame, sa: pd.DataFrame) -> None:
    lines = [
        "PHASE T1 — DONCHIAN CHANNEL BREAKOUT CONCEPT DISCOVERY",
        "=" * 80,
        "",
        "Unger §2.1: entry on Donchian channel breakout.",
        "Unger §3.1: EMA200 price-above filter as regime gate.",
        "Stability principle: edge must be consistent AROUND the chosen N, not just AT it.",
        "",
        f"Grid: {len(TIMEFRAMES)} timeframes × {len(DONCHIAN_NS)} Donchian N × "
        f"{len(FILTER_MODES)} filters × {len(ATR_MULTS)} ATR mults = "
        f"{len(TIMEFRAMES)*len(DONCHIAN_NS)*len(FILTER_MODES)*len(ATR_MULTS)} combos",
        f"Stability zone: N ∈ {STABILITY_ZONE_NS}  (PASS threshold: ≥{STABILITY_PASS_PCT:.0f}%)",
        "",
        "═" * 80,
        "STABILITY RANKING — best (TF, filter) combos by zone stability",
        "═" * 80,
    ]

    for _, row in sa.head(10).iterrows():
        verdict = row["stability_verdict"]
        lines.append(
            f"  [{verdict:4s}] TF={row['timeframe']:3s}  filter={row['filter_mode']:14s}  "
            f"ATR×{row['atr_mult']:.1f}  "
            f"zone={row['zone_pf_gt1']}/{row['zone_total']} profitable "
            f"({row['zone_stability_pct']:.0f}%)  "
            f"N=20→ trades={row['can_n20_trades']}  avg_r={row['can_n20_avg_r']:+.4f}  "
            f"pf={row['can_n20_pf']:.2f}  dd={row['can_n20_max_dd_r']:+.2f}"
        )

    # Best candidate
    best = sa.iloc[0] if not sa.empty else None
    if best is not None:
        lines += [
            "",
            "─" * 80,
            "RECOMMENDED CANONICAL CONFIGURATION",
            "─" * 80,
            f"  Timeframe:    {best['timeframe']}",
            f"  Filter mode:  {best['filter_mode']}",
            f"  Donchian N:   20  (stability zone: {best['zone_stability_pct']:.0f}% pass)",
            f"  ATR mult:     {best['atr_mult']:.1f}",
            f"  Donchian exit: {20 // 2} (= N // 2)",
            "",
            "  → Use these values in T2 command:",
            f"  python phase_t2_core_trend_engine.py "
            f"--timeframe {best['timeframe']} "
            f"--filter-mode {best['filter_mode']} "
            f"--donchian-entry 20 --atr-mult {best['atr_mult']:.1f}",
        ]

    # Per-TF detail
    lines += ["", "═" * 80, "DONCHIAN N SENSITIVITY PER TIMEFRAME + FILTER", "═" * 80]

    for (tf, fm, am), g in grid.groupby(["timeframe","filter_mode","atr_mult"]):
        g_sorted = g.sort_values("don_n")
        lines += [f"", f"  TF={tf}  filter={fm}  ATR×{am:.1f}"]
        for _, row in g_sorted.iterrows():
            n = int(row["don_n"])
            star = " ← STABILITY ZONE" if n in STABILITY_ZONE_NS else ""
            can  = " ← CANONICAL"      if n == 20 else ""
            sign = "[+]" if row["pf"] > 1.0 and row["avg_r"] > 0 else "[-]"
            lines.append(
                f"    {sign} N={n:2d}  exit={n//2:2d}  trades={int(row['trades']):4d}  "
                f"avg_r={row['avg_r']:+.3f}  pf={row['pf']:.2f}  "
                f"dd={row['max_dd_r']:+.2f}{star}{can}"
            )

    lines += [
        "",
        "═" * 80,
        "INTERPRETATION (Unger §2.1 methodology)",
        "═" * 80,
        "",
        f"  PASS  (stability ≥ {STABILITY_PASS_PCT:.0f}%): Edge is robust. Use recommended config in T2.",
        "  WARN  (stability 40–67%): Edge exists but parameter-sensitive. Consider widening N.",
        "  FAIL  (stability < 40%):  Edge is fragile. Do NOT proceed. Consider §2.6 (ATR gate).",
        "",
        "  Win rate check (Unger §4.1): a healthy TF system has win rate 30–45%.",
        "  A win rate above 55% is suspicious — likely overfitting or holding losers.",
        "",
        "  Cost check (Unger §4.2): avg_r must be > 0.15R to cover Binance taker fees.",
        "  If avg_r < 0.15R on the recommended combo, the edge is too thin for live trading.",
        "",
        "  T9A paper-live remains FROZEN regardless of T1 findings.",
        "  T1 findings feed T2 → T3B → T4–T8 → new paper config (separate from T9A).",
    ]

    (OUT_DIR / "phase_t1_stability_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(refresh: bool = False) -> int:
    print("=" * 80)
    print("PHASE T1 — DONCHIAN CHANNEL BREAKOUT CONCEPT DISCOVERY")
    print("=" * 80)
    print(f"Timeframes:    {TIMEFRAMES}")
    print(f"Donchian N:    {DONCHIAN_NS}  (exit = N // 2)")
    print(f"Filter modes:  {FILTER_MODES}")
    print(f"ATR mults:     {ATR_MULTS}")
    print(f"Universe:      top {UNIVERSE_SIZE} USDT spot symbols")
    print()

    exchange = _init_exchange()
    symbols  = get_universe(exchange)

    # Fetch and cache OHLCV for all symbols × timeframes
    data: Dict[Tuple[str, str], pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        print(f"\n=== Fetching {tf} candles ===")
        for idx, sym in enumerate(symbols, 1):
            print(f"  [{idx:3d}/{len(symbols)}] {sym}", end="  ")
            df = fetch_cached(exchange, sym, tf, refresh=refresh)
            if df is not None and len(df) >= max(EMA_REGIME_N + EMA_SLOPE_LB + 10, 250):
                data[(sym, tf)] = df
                print(f"OK ({len(df)} bars)")
            else:
                print("SKIP")

    # Run grid sweep
    all_trades: list[dict] = []
    summary_rows: list[dict] = []
    total_combos = len(TIMEFRAMES) * len(DONCHIAN_NS) * len(FILTER_MODES) * len(ATR_MULTS)
    done = 0

    for tf in TIMEFRAMES:
        symbols_with_data = [s for s in symbols if (s, tf) in data]
        if not symbols_with_data:
            print(f"\nWARN: no data for {tf}, skipping.")
            continue
        print(f"\n=== Backtesting {tf} ({len(symbols_with_data)} symbols) ===")

        for don_n in DONCHIAN_NS:
            for fm in FILTER_MODES:
                for am in ATR_MULTS:
                    combo_trades: list[dict] = []
                    for sym in symbols_with_data:
                        combo_trades.extend(_backtest(sym, data[(sym, tf)], don_n, fm, am))

                    # Tag with timeframe
                    for t in combo_trades:
                        t["timeframe"] = tf

                    s = _stats(combo_trades)
                    row = dict(timeframe=tf, don_n=don_n, don_exit_n=don_n//2,
                               filter_mode=fm, atr_mult=am, symbols=len(symbols_with_data))
                    row.update(s)
                    summary_rows.append(row)
                    all_trades.extend(combo_trades)
                    done += 1

                    print(
                        f"  [{done:3d}/{total_combos}] TF={tf} N={don_n:2d} "
                        f"exit={don_n//2:2d} {fm:14s} ATR×{am:.1f}  "
                        f"trades={s['trades']:4d}  avg_r={s['avg_r']:+.3f}  pf={s['pf']:.2f}"
                    )

    if not summary_rows:
        print("ERROR: no trades produced. Check data / exchange connectivity.")
        return 1

    grid = pd.DataFrame(summary_rows)

    # Stability analysis
    sa = stability_analysis(grid)
    print("\n=== STABILITY RANKING (top 5) ===")
    cols = ["timeframe","filter_mode","atr_mult","zone_stability_pct",
            "stability_verdict","can_n20_avg_r","can_n20_pf"]
    print(sa[cols].head(5).to_string(index=False))

    # Write outputs
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"],  utc=True, errors="coerce")
        trades_df = trades_df.sort_values(["timeframe","entry_time","symbol"]).reset_index(drop=True)

    trades_df.to_csv(OUT_DIR / "phase_t1_trend_discovery_trades.csv", index=False)
    grid.to_csv(OUT_DIR / "phase_t1_trend_discovery_summary.csv", index=False)
    sa.to_csv(OUT_DIR / "phase_t1_stability_ranking.csv", index=False)
    write_report(grid, sa)

    if not sa.empty:
        best = sa.iloc[0]
        print(f"\nRECOMMENDED: TF={best['timeframe']}  filter={best['filter_mode']}  "
              f"N=20  ATR×{best['atr_mult']:.1f}  "
              f"stability={best['zone_stability_pct']:.0f}%  "
              f"verdict={best['stability_verdict']}")
        print(f"\nRun T2 with:")
        print(f"  python phase_t2_core_trend_engine.py "
              f"--timeframe {best['timeframe']} "
              f"--filter-mode {best['filter_mode']} "
              f"--donchian-entry 20 --atr-mult {best['atr_mult']:.1f}")

    print("\n[OK] phase_t1_trend_discovery_trades.csv")
    print("[OK] phase_t1_trend_discovery_summary.csv")
    print("[OK] phase_t1_stability_ranking.csv")
    print("[OK] phase_t1_stability_report.txt  ← READ THIS before running T2")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Re-download OHLCV even if cached")
    args = p.parse_args()
    raise SystemExit(main(refresh=args.refresh))
