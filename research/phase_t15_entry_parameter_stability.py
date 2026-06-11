#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T15 — POST-EXIT ENTRY STABILITY STUDY

Offline only. T9A paper-live remains frozen.

ROLE IN REDESIGNED PIPELINE (updated after T1/T2 redesign)
-----------------------------------------------------------
T1 (redesigned) already performs entry parameter stability discovery across
timeframes and filter modes. T15 therefore plays a DIFFERENT, narrower role:

  T15 = post-exit entry stability check, run AFTER T3B
  Purpose: confirm the T1-canonical entry parameters still hold stability
  when the FULL exit architecture (Chandelier ACT4_ATR3) is applied,
  not just the simple initial-stop + Donchian-exit used in T1/T2.

  Why needed: exit engineering (T3B) changes which trades are held and how
  long — it can shift which entry N looks best. This check validates that
  the T1 canonical N is still the stable choice post-Chandelier.

  The cost check (Unger §4.2) is also valuable here because T3B's Chandelier
  produces larger avg_R trades, making the cost-vs-edge calculation meaningful.

Parameter sweep (exits frozen at T3B canonical ACT4_ATR3):
  Donchian N (entry):  [10, 15, 20, 25, 30, 40, 55]
  EMA length:          [20, 30, 50, 75, 100]
  EMA slope lookback:  [5, 10, 15]
  → 105 combos × 70 symbols × 6H

OHLCV source: T13 ohlcv_cache (6H), falls back to T10, then downloads fresh.

Outputs: data/research_trend_t15/
  phase_t15_entry_stability_grid.csv      — stats for all 105 combos
  phase_t15_entry_stability_trades.csv    — canonical combo trades (used by T16)
  phase_t15_entry_stability_sensitivity.csv — per-param sensitivity slices
  phase_t15_entry_cost_check.csv          — fee analysis for canonical combo
  phase_t15_entry_stability_report.txt    — master report
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

# ── optional ccxt (only needed if OHLCV cache is absent) ─────────────────────
try:
    import ccxt as _ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
OUT = ROOT / "data" / "research_trend_t15"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_OWN = OUT / "ohlcv_cache"
CACHE_OWN.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SWEEP
# Filter is fixed at ema200_price (T1 4yr confirmed). Only Donchian N varies.
# ─────────────────────────────────────────────────────────────────────────────
DONCHIAN_NS    = [10, 15, 20, 25, 30, 40, 55]
CANONICAL      = (20,)          # (donchian_n,) — T8 frozen config

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN EXIT & STOP PARAMS (T3B/T8 canonical)
# ─────────────────────────────────────────────────────────────────────────────
ACT_R          = 4.0
CH_MULT        = 3.0
ATR_N          = 14
INIT_STOP_ATR  = 2.0
EMA_REGIME_LEN = 200            # ema200 price-above filter (Unger §3.1, T1 confirmed)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE & EXCHANGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME      = "1d"
LIMIT_BARS     = 1500           # ~4 years of 1D data
UNIVERSE_SIZE  = 70
FEE_PER_SIDE   = 0.001          # 0.10% Binance taker per leg
LEVERAGE_PROXY = 1.0            # Binance Spot — no leverage
ALLOW_SHORT    = False          # Binance Spot LONG only

EXCLUDE_BASES = {
    "USDC","BUSD","TUSD","USDP","PAXG","XAUT","FDUSD","USDD","DAI",
    "USD1","RLUSD","EUR","AEUR","WBTC","BTTC"
}
PREFERRED = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "DOGE/USDT","AVAX/USDT","LINK/USDT","TRX/USDT","DOT/USDT","BCH/USDT",
    "LTC/USDT","UNI/USDT","NEAR/USDT","APT/USDT","SUI/USDT","ICP/USDT",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_sym(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _pf(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    if l <= 1e-12:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def _dd_r(r: pd.Series) -> float:
    vals = pd.to_numeric(r, errors="coerce").fillna(0.0).values
    if not len(vals):
        return 0.0
    eq = np.cumsum(vals)
    return float((eq - np.maximum.accumulate(eq)).min())


def _stats(r: pd.Series) -> dict:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return dict(trades=0, total_r=0.0, avg_r=0.0, median_r=0.0,
                    win_rate_pct=0.0, pf=0.0, max_dd_r=0.0,
                    best_r=0.0, worst_r=0.0)
    return dict(
        trades=len(r),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        median_r=float(r.median()),
        win_rate_pct=float((r > 0).mean() * 100),
        pf=_pf(r),
        max_dd_r=_dd_r(r),
        best_r=float(r.max()),
        worst_r=float(r.min()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time", "open", "high", "low", "close"])
        df = df.sort_values("time").reset_index(drop=True)
        # Keep only fully closed candles (close_time <= now)
        now = pd.Timestamp.utcnow()
        df["close_time"] = df["time"] + pd.Timedelta(days=1)
        df = df[df["close_time"] <= now].drop(columns=["close_time"])
        return df if len(df) >= 100 else None
    except Exception:
        return None


def _load_symbol(sym: str) -> pd.DataFrame | None:
    fname = f"{safe_sym(sym)}_{TIMEFRAME}.csv"
    # T15 uses its own OHLCV cache (no T13/T10 1D data exists)
    for cache in (CACHE_OWN,):
        df = _load_csv(cache / fname)
        if df is not None:
            return df
    return None


def _download_universe() -> list[str]:
    if not HAS_CCXT:
        raise RuntimeError("ccxt not available and no cached OHLCV found.")
    ex = _ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    markets = ex.load_markets()
    syms = []
    for sym, m in markets.items():
        if not m.get("spot", False) or not m.get("active", True):
            continue
        if m.get("quote") != "USDT" or ":" in sym:
            continue
        if m.get("base") in EXCLUDE_BASES:
            continue
        if any(x in sym for x in ["UP/","DOWN/","BULL/","BEAR/","3L/","3S/"]):
            continue
        syms.append(sym)
    ordered = [s for s in PREFERRED if s in syms] + [s for s in sorted(syms) if s not in PREFERRED]
    return ordered[:UNIVERSE_SIZE]


def _download_and_cache(ex, sym: str) -> pd.DataFrame | None:
    """Fetch 1D OHLCV from Binance, save to CACHE_OWN, return DataFrame."""
    try:
        raw = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=LIMIT_BARS)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["timestamp","open","high","low","close"])
        fname = CACHE_OWN / f"{safe_sym(sym)}_{TIMEFRAME}.csv"
        df.to_csv(fname, index=False)
        return _load_csv(fname)
    except Exception as e:
        print(f"  Download failed {sym}: {e}")
        return None


def load_all_ohlcv() -> dict[str, pd.DataFrame]:
    """
    Return dict of symbol → DataFrame for all 1D symbols.
    Uses CACHE_OWN if present; downloads from Binance on first run.
    """
    # Use existing cache if populated
    cached_files = list(CACHE_OWN.glob(f"*_{TIMEFRAME}.csv"))
    if cached_files:
        result: dict[str, pd.DataFrame] = {}
        for f in cached_files:
            base_quote = f.stem[: -len(f"_{TIMEFRAME}")]
            if "_USDT" not in base_quote:
                continue
            sym = base_quote.replace("_USDT", "/USDT", 1)
            df = _load_csv(f)
            if df is not None:
                result[sym] = df
        print(f"Loaded {len(result)} symbols from T15 1D cache.")
        return result

    # Cold start — download from Binance
    if not HAS_CCXT:
        raise RuntimeError("ccxt not available and no 1D OHLCV cache found. Run T15 first or install ccxt.")
    print("No 1D OHLCV cache found — downloading universe from Binance...")
    ex = _ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    candidates = _download_universe()
    result = {}
    for i, sym in enumerate(candidates, 1):
        df = _download_and_cache(ex, sym)
        if df is not None:
            result[sym] = df
        if i % 10 == 0:
            print(f"  Downloaded {i}/{len(candidates)} symbols...")
        time.sleep(0.1)

    print(f"Downloaded and cached {len(result)} symbols with 1D data.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame, don_n: int) -> pd.DataFrame:
    d = df.copy()
    # ATR
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()
    # EMA200 price-above regime filter (Unger §3.1, T1 4yr confirmed)
    d["ema200"] = d["close"].ewm(span=EMA_REGIME_LEN, adjust=False).mean()
    # Donchian entry (shifted — no lookahead)
    d["don_high"] = d["high"].shift(1).rolling(don_n).max()
    d["don_low"]  = d["low"].shift(1).rolling(don_n).min()
    # Donchian exit: N//2 (Turtle rule — active while Chandelier not yet triggered)
    exit_n = max(1, don_n // 2)
    d["don_exit_low"]  = d["low"].shift(1).rolling(exit_n).min()
    d["don_exit_high"] = d["high"].shift(1).rolling(exit_n).max()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER (single symbol, single parameter combo)
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(sym: str, df: pd.DataFrame, don_n: int) -> list[dict]:
    d = _add_indicators(df, don_n)
    warmup = max(don_n, EMA_REGIME_LEN, ATR_N) + 5
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
        t     = row["time"]

        if pos is not None:
            pos["hh"] = max(pos["hh"], high)
            mfe = (high - pos["entry"]) / pos["risk"]
            pos["mfe"] = max(pos["mfe"], mfe)
            # 1. Stop check uses the stop from the START of this candle (before update)
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                r = (exit_px - pos["entry"]) / pos["risk"]
                fee_r = 2 * FEE_PER_SIDE * LEVERAGE_PROXY * pos["entry"] / pos["risk"]
                trades.append(_make_trade(sym, pos, t, exit_px, r, fee_r, don_n))
                pos = None
                continue
            # 2. Donchian N//2 exit on close (only while Chandelier not yet active)
            if not pos["trail"]:
                don_exit_low = row.get("don_exit_low", np.nan)
                if np.isfinite(float(don_exit_low)) and close < float(don_exit_low):
                    exit_px = close
                    r = (exit_px - pos["entry"]) / pos["risk"]
                    fee_r = 2 * FEE_PER_SIDE * LEVERAGE_PROXY * pos["entry"] / pos["risk"]
                    trades.append(_make_trade(sym, pos, t, exit_px, r, fee_r, don_n))
                    pos = None
                    continue
            # 3. Update chandelier stop for NEXT candle (after all exit checks pass)
            if pos["mfe"] >= ACT_R:
                new_stop = pos["hh"] - CH_MULT * atr
                pos["stop"] = max(pos["stop"], new_stop)
                pos["trail"] = True

        if pos is None:
            dh   = float(row["don_high"]) if np.isfinite(float(row["don_high"])) else None
            e200 = float(row["ema200"])   if np.isfinite(float(row["ema200"]))   else None
            if dh is None or e200 is None:
                continue
            # Entry: close breaks Donchian high AND price above EMA200 (LONG only)
            if close > dh and close > e200:
                risk = atr * INIT_STOP_ATR
                if risk > 0:
                    pos = dict(side="LONG", entry_time=t, entry_i=i,
                               entry=close, stop=close - risk, risk=risk,
                               hh=close, mfe=0.0, trail=False)

    return trades


def _make_trade(sym, pos, exit_time, exit_px, net_r, fee_r, don_n):
    return {
        "symbol":                sym,
        "timeframe":             TIMEFRAME,
        "side":                  pos["side"],
        "don_n":                 don_n,
        "entry_time":            pos["entry_time"],
        "exit_time":             exit_time,
        "entry_price":           pos["entry"],
        "exit_price":            exit_px,
        "initial_stop":          pos["entry"] - pos["risk"],
        "initial_risk_per_unit": pos["risk"],
        "net_r":                 float(net_r),
        "fee_r":                 float(fee_r),
        "net_r_after_fee":       float(net_r - fee_r),
        "max_favorable_r":       pos["mfe"],
        "chandelier_active":     pos["trail"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = len(DONCHIAN_NS)
    print(f"Grid: {total} Donchian N values × {len(data)} symbols  "
          f"(filter fixed: ema200_price, exit: Donchian N//2 + Chandelier ACT{ACT_R:g}_ATR{CH_MULT:g})")

    rows = []
    canonical_trades: list[dict] = []

    for idx, don_n in enumerate(DONCHIAN_NS, 1):
        combo_trades: list[dict] = []
        for sym, df in data.items():
            combo_trades.extend(_backtest(sym, df, don_n))

        r_ser = pd.Series([t["net_r"] for t in combo_trades])
        s = _stats(r_ser)
        row = {"don_n": don_n, "is_canonical": (don_n,) == CANONICAL}
        row.update(s)
        rows.append(row)

        if (don_n,) == CANONICAL:
            canonical_trades = combo_trades

        print(f"  [{idx}/{total}]  don_n={don_n:2d}  "
              f"trades={s['trades']:4d}  avg_r={s['avg_r']:+.3f}  pf={s['pf']:.2f}  "
              f"dd_r={s['max_dd_r']:+.2f}{'  ← CANONICAL' if (don_n,) == CANONICAL else ''}")

    return pd.DataFrame(rows), canonical_trades


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY SLICES (one param varies, others fixed at canonical)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_slices(grid: pd.DataFrame) -> pd.DataFrame:
    sub = grid.copy()
    sub["param"] = "donchian_n"
    sub["value"] = sub["don_n"].astype(str)
    return sub[["param","value","don_n","trades","total_r",
                "avg_r","pf","max_dd_r","win_rate_pct","is_canonical"]]


# ─────────────────────────────────────────────────────────────────────────────
# COST CHECK (Unger §4.2)
# ─────────────────────────────────────────────────────────────────────────────

def cost_check(canonical_trades: list[dict]) -> pd.DataFrame:
    """
    Per Unger §4.2: avg_trade_usdt must exceed round-trip fees.
    fee_R for each trade = 2 × FEE_PER_SIDE × LEVERAGE_PROXY × entry_price / initial_risk_per_unit
    """
    rows = []
    for t in canonical_trades:
        fee_r = t.get("fee_r", 0.0)
        net_r = t.get("net_r", 0.0)
        rows.append({
            "symbol":   t["symbol"],
            "side":     t["side"],
            "entry_price": t["entry_price"],
            "initial_risk_per_unit": t["initial_risk_per_unit"],
            "net_r":    net_r,
            "fee_r":    fee_r,
            "net_r_after_fee": net_r - fee_r,
            "stop_pct_of_price": t["initial_risk_per_unit"] / max(t["entry_price"], 1e-12),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary_rows = []
    for sym, g in df.groupby("symbol"):
        r_after = g["net_r_after_fee"]
        summary_rows.append({
            "symbol": sym,
            "trades": len(g),
            "avg_fee_r": float(g["fee_r"].mean()),
            "avg_net_r_gross": float(g["net_r"].mean()),
            "avg_net_r_after_fee": float(r_after.mean()),
            "avg_stop_pct": float(g["stop_pct_of_price"].mean() * 100),
            "fee_covers": float(g["fee_r"].mean()) < float(g["net_r"].mean()),
            "net_r_after_fee_positive": float(r_after.mean()) > 0.0,
        })
    return pd.DataFrame(summary_rows).sort_values("avg_net_r_after_fee", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def stability_analysis(grid: pd.DataFrame) -> dict:
    """
    Unger stability test: how many Donchian N values near canonical are profitable?
    Zone = don_n in [15, 20, 25] (±1 step from canonical N=20).
    """
    zone = grid[grid["don_n"].isin([15, 20, 25])]
    total_zone  = len(zone)
    prof_zone   = int((zone["pf"] > 1.0).sum())
    pos_avg_r   = int((zone["avg_r"] > 0.0).sum())
    strong_zone = int(((zone["pf"] > 1.2) & (zone["avg_r"] > 0.05)).sum())

    can = grid[grid["don_n"] == CANONICAL[0]]
    can_row = can.iloc[0].to_dict() if not can.empty else {}

    return {
        "canonical":            CANONICAL,
        "canonical_trades":     can_row.get("trades", 0),
        "canonical_avg_r":      can_row.get("avg_r", 0.0),
        "canonical_pf":         can_row.get("pf", 0.0),
        "canonical_max_dd_r":   can_row.get("max_dd_r", 0.0),
        "canonical_win_rate":   can_row.get("win_rate_pct", 0.0),
        "zone_combos":          total_zone,
        "zone_pf_gt1":          prof_zone,
        "zone_avg_r_positive":  pos_avg_r,
        "zone_strong":          strong_zone,
        "stability_pct":        100.0 * prof_zone / max(total_zone, 1),
        "total_grid_combos":    len(grid),
        "grid_pf_gt1":          int((grid["pf"] > 1.0).sum()),
        "grid_stability_pct":   100.0 * (grid["pf"] > 1.0).sum() / max(len(grid), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(sa: dict, grid: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    can = sa["canonical"]
    lines = [
        "PHASE T15 — ENTRY PARAMETER STABILITY STUDY",
        "=" * 80,
        "",
        "Unger principle: pick parameters by stability around chosen value, not by",
        "best single value.  This study stress-tests the frozen Donchian N.",
        "",
        f"Canonical config: Donchian N={can[0]}, filter=ema200_price (fixed)",
        f"Exits frozen at:  Donchian N//2 fallback + Chandelier ACT{ACT_R:g}_ATR{CH_MULT:g}",
        f"Timeframe:        {TIMEFRAME}",
        "",
        "CANONICAL COMBO STATS",
        "-" * 40,
        f"  Trades:         {sa['canonical_trades']}",
        f"  Avg R:          {sa['canonical_avg_r']:+.4f}",
        f"  Profit factor:  {sa['canonical_pf']:.2f}",
        f"  Max drawdown R: {sa['canonical_max_dd_r']:+.2f}",
        f"  Win rate:       {sa['canonical_win_rate']:.1f}%",
        "",
        "STABILITY ZONE (don_n∈{15,20,25} — ±1 step from canonical N=20)",
        "-" * 40,
        f"  Total combos in zone:  {sa['zone_combos']}",
        f"  PF > 1.0:              {sa['zone_pf_gt1']} / {sa['zone_combos']}  "
        f"({sa['stability_pct']:.0f}%)",
        f"  Avg R > 0:             {sa['zone_avg_r_positive']} / {sa['zone_combos']}",
        f"  Strong (PF>1.2, avgR>0.05): {sa['zone_strong']} / {sa['zone_combos']}",
        "",
        "FULL GRID",
        "-" * 40,
        f"  Total combos:   {sa['total_grid_combos']}",
        f"  PF > 1.0:       {sa['grid_pf_gt1']} / {sa['total_grid_combos']}  "
        f"({sa['grid_stability_pct']:.0f}%)",
        "",
    ]

    # Donchian N sweep results
    lines += [
        "DONCHIAN N SWEEP  (filter=ema200_price fixed, exit=Donchian N//2 + Chandelier)",
        "-" * 60,
    ]
    for _, row in grid.sort_values("don_n").iterrows():
        marker = " ← CANONICAL" if row["don_n"] == can[0] else ""
        lines.append(
            f"  don_n={int(row['don_n']):3d}  trades={int(row['trades']):4d}  "
            f"avg_r={row['avg_r']:+.3f}  pf={row['pf']:.2f}  dd={row['max_dd_r']:+.2f}{marker}"
        )

    # Cost check summary
    if not cost_df.empty:
        n_pass = int(cost_df["fee_covers"].sum())
        n_net_pos = int(cost_df["net_r_after_fee_positive"].sum())
        avg_fee_r = float(cost_df["avg_fee_r"].mean())
        avg_net_after = float(cost_df["avg_net_r_after_fee"].mean())
        lines += [
            "",
            "COST CHECK (Unger §4.2 — avg trade must exceed fees)",
            "-" * 40,
            f"  Average fee cost per trade (in R): {avg_fee_r:.3f}R",
            f"  Avg net R after fee:               {avg_net_after:+.3f}R",
            f"  Symbols where avg_R > fee_R:       {n_pass}/{len(cost_df)}",
            f"  Symbols with positive net R after fee: {n_net_pos}/{len(cost_df)}",
        ]

    lines += [
        "",
        "INTERPRETATION RULES (Unger §2.1)",
        "-" * 40,
        "  PASS: stability_pct >= 67% in the zone AND canonical avg_R > 0.",
        "  WARN: stability_pct 40-67% — edge exists but parameter sensitivity is high.",
        "  FAIL: stability_pct < 40% — current parameters may be overfit.",
        "",
        "  If PASS: freeze canonical params, proceed to T16 robustness completion.",
        "  If WARN: consider widening Donchian N to the most stable value in the zone.",
        "  If FAIL: do NOT tune. Retire Donchian and evaluate §2.6 (T18) or §2.4 (ORB).",
        "",
        "  Do NOT update T9A based on this study alone.",
        "  T9A paper-live remains frozen until T16+T17+T18 are all reviewed.",
    ]

    (OUT / "phase_t15_entry_stability_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("PHASE T15 — ENTRY PARAMETER STABILITY STUDY")
    print("=" * 80)
    print(f"Output: {OUT}")
    print(f"Canonical: Donchian N={CANONICAL[0]}, filter=ema200_price")
    print()

    # Load OHLCV
    data = load_all_ohlcv()
    if not data:
        raise RuntimeError("No 1D OHLCV data loaded. Ensure ccxt is installed for fresh download.")

    # Grid sweep
    print("\nRunning parameter grid sweep...")
    t0 = time.time()
    grid, canonical_trades = run_grid(data)
    elapsed = time.time() - t0
    print(f"\nGrid sweep completed in {elapsed:.1f}s")

    # Sensitivity slices
    sensitivity = sensitivity_slices(grid)

    # Cost check
    print("\nRunning cost check (Unger §4.2)...")
    cost_df = cost_check(canonical_trades)

    # Stability analysis
    sa = stability_analysis(grid)
    print(f"\nStability zone ({sa['zone_combos']} combos): "
          f"{sa['zone_pf_gt1']} profitable ({sa['stability_pct']:.0f}%)")
    print(f"Canonical: trades={sa['canonical_trades']}  "
          f"avg_r={sa['canonical_avg_r']:+.4f}  pf={sa['canonical_pf']:.2f}")

    # Write outputs
    grid.to_csv(OUT / "phase_t15_entry_stability_grid.csv", index=False)
    sensitivity.to_csv(OUT / "phase_t15_entry_stability_sensitivity.csv", index=False)
    cost_df.to_csv(OUT / "phase_t15_entry_cost_check.csv", index=False)
    write_report(sa, grid, cost_df)

    # Write canonical trade list (input for T16)
    if canonical_trades:
        can_df = pd.DataFrame(canonical_trades)
        can_df["entry_time"] = pd.to_datetime(can_df["entry_time"], utc=True, errors="coerce")
        can_df["exit_time"]  = pd.to_datetime(can_df["exit_time"],  utc=True, errors="coerce")
        can_df = can_df.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
        can_df.to_csv(OUT / "phase_t15_entry_stability_trades.csv", index=False)
        print(f"\nCanonical trades saved: {len(can_df)}")
    else:
        print("\nWARNING: no canonical trades produced.")

    print("\n[OK] phase_t15_entry_stability_grid.csv")
    print("[OK] phase_t15_entry_stability_sensitivity.csv")
    print("[OK] phase_t15_entry_cost_check.csv")
    print("[OK] phase_t15_entry_stability_trades.csv")
    print("[OK] phase_t15_entry_stability_report.txt")
    print("\nNext step: run phase_t16_robustness_completion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
