#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE T1 — DUAL MA CROSSOVER CONCEPT DISCOVERY

Andrea Unger §2.1: test edge stability before committing to parameters.

Method:   DualMA
Entry:    Fast EMA crosses above Slow EMA on bar close (LONG only)
Exit:     Fast EMA crosses below Slow EMA (MA cross exit)
          OR initial ATR stop hit
          OR Chandelier trailing stop (activates at +4R MFE, trails 3×ATR)
Filter:   ema200_price (close > EMA200) or ema50_slope (EMA50 slope > 0)

PARAM GRID (15 valid pairs, fast < slow constraint):
  fast_ema:   [10, 20, 30, 50]
  slow_ema:   [50, 100, 150, 200]
  Valid:      (10/50, 10/100, 10/150, 10/200,
               20/50, 20/100, 20/150, 20/200,
               30/50, 30/100, 30/150, 30/200,
               50/100, 50/150, 50/200) = 15 combos

ATR_MULTS:    [2.0, 3.0]
FILTER_MODES: ["ema200_price", "ema50_slope"]

Total grid: 1 TF × 15 pairs × 2 filters × 2 ATR mults = 60 combos

Chandelier: same architecture as Donchian pipeline (ACT+4R, 3×ATR)
Exit sequencing: MA cross exit checked ONLY when Chandelier not yet active
(same as Donchian N//2 exit rule in T15)

OHLCV: reuses T15 1D cache → own cache → fresh Binance download
Timeframe: 1D, ~1500 bars (~4 years)

Benchmark: Donchian frozen config — avg_r=+0.179R, PF=1.27

NOTE on ema200_price filter: when slow_ema=200, the filter EMA and the slow_ema
are the same indicator. The crossover already implies fast > EMA200, making the
filter partially redundant for those 4 combos. See report for details.

Outputs: data/research_dualma_t1/
    phase_t1_dualma_stability_ranking.csv   all 60 combos with zone stability score
    phase_t1_dualma_n_sensitivity.csv       per-param sensitivity per group
    phase_t1_dualma_summary.txt             human-readable report + recommended config
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ccxt as _ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
OUT  = ROOT / "data" / "research_dualma_t1"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_OWN = OUT / "ohlcv_cache"
CACHE_OWN.mkdir(parents=True, exist_ok=True)

# Upstream: reuse T15 1D cache if available (same universe, same bars)
CACHE_T15 = ROOT / "data" / "research_trend_t15" / "ohlcv_cache"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GRID  (Section 13A)
# ─────────────────────────────────────────────────────────────────────────────
FAST_EMAS  = [10, 20, 30, 50]
SLOW_EMAS  = [50, 100, 150, 200]
EMA_PAIRS  = [(f, s) for f in FAST_EMAS for s in SLOW_EMAS if f < s]  # 15 pairs

FILTER_MODES = ["ema200_price", "ema50_slope"]
ATR_MULTS    = [2.0, 3.0]

STABILITY_PASS_PCT = 67.0   # Unger §2.1 threshold

# ─────────────────────────────────────────────────────────────────────────────
# EXIT / STOP PARAMS  (identical to Donchian pipeline for direct comparison)
# ─────────────────────────────────────────────────────────────────────────────
ACT_R         = 4.0    # Chandelier activates at +4R MFE
CH_MULT       = 3.0    # Chandelier trails at 3×ATR
ATR_N         = 14

EMA_FILTER_N  = 200    # Independent EMA200 for ema200_price filter
EMA_SLOPE_N   = 50     # EMA for ema50_slope filter
EMA_SLOPE_LB  = 10     # Slope lookback

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE & COST
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME      = "1d"
LIMIT_BARS     = 1500
UNIVERSE_SIZE  = 70
FEE_PER_SIDE   = 0.001     # 0.10% Binance taker per leg
LEVERAGE_PROXY = 1.0       # Binance Spot
ALLOW_SHORT    = False

COST_FLOOR_R    = 0.15     # Unger §4.2 minimum avg_r

BENCHMARK_AVG_R = 0.179    # Donchian T8 frozen config
BENCHMARK_PF    = 1.27

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
        return dict(trades=0, total_r=0.0, avg_r=0.0, pf=0.0,
                    max_dd_r=0.0, win_rate_pct=0.0)
    return dict(
        trades=int(len(r)),
        total_r=float(r.sum()),
        avg_r=float(r.mean()),
        pf=_pf(r),
        max_dd_r=_dd_r(r),
        win_rate_pct=float((r > 0).mean() * 100),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV LOADING  (same infrastructure as T15)
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
        now = pd.Timestamp.utcnow()
        df["close_time"] = df["time"] + pd.Timedelta(days=1)
        df = df[df["close_time"] <= now].drop(columns=["close_time"])
        return df if len(df) >= 100 else None
    except Exception:
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
        if any(x in sym for x in ["UP/", "DOWN/", "BULL/", "BEAR/", "3L/", "3S/"]):
            continue
        syms.append(sym)
    ordered = ([s for s in PREFERRED if s in syms]
               + [s for s in sorted(syms) if s not in PREFERRED])
    return ordered[:UNIVERSE_SIZE]


def _download_and_cache(ex, sym: str) -> pd.DataFrame | None:
    try:
        raw = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=LIMIT_BARS)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        fname = CACHE_OWN / f"{safe_sym(sym)}_{TIMEFRAME}.csv"
        df.to_csv(fname, index=False)
        return _load_csv(fname)
    except Exception as e:
        print(f"  Download failed {sym}: {e}")
        return None


def load_all_ohlcv() -> dict[str, pd.DataFrame]:
    # Prefer T15 cache (same 1D universe, already downloaded)
    t15_files = list(CACHE_T15.glob(f"*_{TIMEFRAME}.csv")) if CACHE_T15.exists() else []
    own_files  = list(CACHE_OWN.glob(f"*_{TIMEFRAME}.csv"))

    source_files = t15_files if t15_files else own_files
    if source_files:
        result: dict[str, pd.DataFrame] = {}
        for f in source_files:
            base_quote = f.stem[: -len(f"_{TIMEFRAME}")]
            if "_USDT" not in base_quote:
                continue
            sym = base_quote.replace("_USDT", "/USDT", 1)
            df = _load_csv(f)
            if df is not None:
                result[sym] = df
        src = "T15 1D cache" if t15_files else "DualMA own cache"
        print(f"Loaded {len(result)} symbols from {src}.")
        return result

    # Cold start — download from Binance
    if not HAS_CCXT:
        raise RuntimeError(
            "ccxt not available and no 1D OHLCV cache found. "
            "Run phase_t15_entry_parameter_stability.py first, or install ccxt."
        )
    print("No 1D cache found — downloading universe from Binance...")
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
    print(f"Downloaded and cached {len(result)} symbols.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame, fast_n: int, slow_n: int) -> pd.DataFrame:
    d = df.copy()
    # ATR
    pc = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()
    # Crossover EMAs
    d["fast_ema"] = d["close"].ewm(span=fast_n, adjust=False).mean()
    d["slow_ema"] = d["close"].ewm(span=slow_n, adjust=False).mean()
    # Independent EMA200 regime filter (same as Donchian pipeline)
    d["ema_filter"] = d["close"].ewm(span=EMA_FILTER_N, adjust=False).mean()
    # EMA50 slope filter
    d["ema50"] = d["close"].ewm(span=EMA_SLOPE_N, adjust=False).mean()
    d["ema50_prev"] = d["ema50"].shift(EMA_SLOPE_LB)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER  (single symbol, single combo)
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(sym: str, df: pd.DataFrame, fast_n: int, slow_n: int,
              filter_mode: str, atr_mult: float) -> list[dict]:
    d = _add_indicators(df, fast_n, slow_n)
    warmup = max(slow_n, EMA_FILTER_N, EMA_SLOPE_N + EMA_SLOPE_LB, ATR_N) + 5
    pos = None
    trades = []

    for i in range(warmup, len(d)):
        row   = d.iloc[i]
        row_p = d.iloc[i - 1]   # previous bar for crossover detection
        atr   = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        close     = float(row["close"])
        high      = float(row["high"])
        low       = float(row["low"])
        t         = row["time"]
        fast_cur  = float(row["fast_ema"])
        slow_cur  = float(row["slow_ema"])
        fast_prev = float(row_p["fast_ema"])
        slow_prev = float(row_p["slow_ema"])

        if not all(np.isfinite(x) for x in [fast_cur, slow_cur, fast_prev, slow_prev]):
            continue

        if pos is not None:
            pos["hh"] = max(pos["hh"], high)
            mfe = (high - pos["entry"]) / pos["risk"]
            pos["mfe"] = max(pos["mfe"], mfe)

            # 1. Stop check — uses stop from START of candle (before Chandelier update)
            if low <= pos["stop"]:
                exit_px = pos["stop"]
                net_r   = (exit_px - pos["entry"]) / pos["risk"]
                fee_r   = 2 * FEE_PER_SIDE * LEVERAGE_PROXY * pos["entry"] / pos["risk"]
                trades.append(_make_trade(sym, pos, t, exit_px, net_r, fee_r,
                                          fast_n, slow_n, filter_mode, atr_mult, "stop"))
                pos = None
                continue

            # 2. MA crossover exit on close — only while Chandelier not yet active
            #    (mirrors Donchian N//2 exit logic: disabled once Chandelier takes over)
            if not pos["trail"]:
                if fast_cur < slow_cur:
                    exit_px = close
                    net_r   = (exit_px - pos["entry"]) / pos["risk"]
                    fee_r   = 2 * FEE_PER_SIDE * LEVERAGE_PROXY * pos["entry"] / pos["risk"]
                    trades.append(_make_trade(sym, pos, t, exit_px, net_r, fee_r,
                                              fast_n, slow_n, filter_mode, atr_mult, "ma_cross"))
                    pos = None
                    continue

            # 3. Update Chandelier for NEXT candle — after all exit checks pass
            if pos["mfe"] >= ACT_R:
                new_stop = pos["hh"] - CH_MULT * atr
                pos["stop"] = max(pos["stop"], new_stop)
                pos["trail"] = True

        if pos is None:
            # Regime filter
            filter_ok = False
            if filter_mode == "ema200_price":
                ef = float(row["ema_filter"])
                filter_ok = np.isfinite(ef) and close > ef
            else:  # ema50_slope
                e50  = float(row["ema50"])
                e50p = float(row["ema50_prev"])
                filter_ok = np.isfinite(e50) and np.isfinite(e50p) and e50 > e50p

            if not filter_ok:
                continue

            # Entry: fast EMA crosses above slow EMA on this bar's close
            crossover = (fast_prev <= slow_prev) and (fast_cur > slow_cur)
            if crossover:
                risk = atr * atr_mult
                if risk > 0:
                    pos = dict(
                        side="LONG", entry_time=t, entry=close,
                        stop=close - risk, risk=risk,
                        hh=close, mfe=0.0, trail=False,
                    )

    return trades


def _make_trade(sym, pos, exit_time, exit_px, net_r, fee_r,
                fast_n, slow_n, filter_mode, atr_mult, reason):
    return {
        "symbol":            sym,
        "timeframe":         TIMEFRAME,
        "side":              pos["side"],
        "fast_ema":          fast_n,
        "slow_ema":          slow_n,
        "filter_mode":       filter_mode,
        "atr_mult":          atr_mult,
        "entry_time":        pos["entry_time"],
        "exit_time":         exit_time,
        "entry_price":       pos["entry"],
        "exit_price":        float(exit_px),
        "initial_stop":      pos["entry"] - pos["risk"],
        "initial_risk":      pos["risk"],
        "net_r":             float(net_r),
        "fee_r":             float(fee_r),
        "net_r_after_fee":   float(net_r - fee_r),
        "mfe_r":             pos["mfe"],
        "chandelier_active": pos["trail"],
        "exit_reason":       reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = len(EMA_PAIRS) * len(FILTER_MODES) * len(ATR_MULTS)
    print(f"Grid: {len(EMA_PAIRS)} pairs × {len(FILTER_MODES)} filters × "
          f"{len(ATR_MULTS)} ATR mults = {total} combos × {len(data)} symbols")

    rows = []
    done = 0
    for fast_n, slow_n in EMA_PAIRS:
        for fm in FILTER_MODES:
            for am in ATR_MULTS:
                combo_trades: list[dict] = []
                for sym, df in data.items():
                    combo_trades.extend(_backtest(sym, df, fast_n, slow_n, fm, am))

                r_ser = pd.Series([t["net_r"] for t in combo_trades])
                s = _stats(r_ser)
                row = {
                    "fast_ema":    fast_n,
                    "slow_ema":    slow_n,
                    "filter_mode": fm,
                    "atr_mult":    am,
                }
                row.update(s)
                rows.append(row)
                done += 1

                vs = f"{s['avg_r']:+.3f}  (Don={BENCHMARK_AVG_R:+.3f})"
                print(f"  [{done:2d}/{total}] EMA{fast_n:2d}/{slow_n:3d}  {fm:14s}  "
                      f"ATR×{am:.1f}  trades={s['trades']:4d}  avg_r={vs}  pf={s['pf']:.2f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY ANALYSIS  (2D parameter space: fast × slow)
# ─────────────────────────────────────────────────────────────────────────────

def _zone_pairs(cf: int, cs: int) -> set[tuple[int, int]]:
    """
    Zone for canonical (cf, cs): all valid (fast, slow) pairs where
    fast = cf or adjacent step, slow = cs or adjacent step, fast < slow.
    """
    fi = FAST_EMAS.index(cf)
    si = SLOW_EMAS.index(cs)
    zone: set[tuple[int, int]] = {(cf, cs)}
    # Fast neighbors
    if fi > 0 and FAST_EMAS[fi - 1] < cs:
        zone.add((FAST_EMAS[fi - 1], cs))
    if fi < len(FAST_EMAS) - 1 and FAST_EMAS[fi + 1] < cs:
        zone.add((FAST_EMAS[fi + 1], cs))
    # Slow neighbors
    if si > 0 and cf < SLOW_EMAS[si - 1]:
        zone.add((cf, SLOW_EMAS[si - 1]))
    if si < len(SLOW_EMAS) - 1:
        zone.add((cf, SLOW_EMAS[si + 1]))
    return zone


def stability_analysis(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, am), g in grid.groupby(["filter_mode", "atr_mult"]):
        for _, can_row in g.iterrows():
            cf = int(can_row["fast_ema"])
            cs = int(can_row["slow_ema"])
            zp = _zone_pairs(cf, cs)

            zone = g[g.apply(
                lambda r: (int(r["fast_ema"]), int(r["slow_ema"])) in zp, axis=1
            )]
            total_zone = len(zone)
            pf_gt1 = int((zone["pf"] > 1.0).sum())
            pct    = 100.0 * pf_gt1 / max(total_zone, 1)
            verdict = ("PASS" if pct >= STABILITY_PASS_PCT
                       else ("WARN" if pct >= 40 else "FAIL"))

            rows.append({
                "filter_mode":        fm,
                "atr_mult":           am,
                "fast_ema":           cf,
                "slow_ema":           cs,
                "zone_total":         total_zone,
                "zone_pf_gt1":        pf_gt1,
                "zone_stability_pct": round(pct, 1),
                "stability_verdict":  verdict,
                "trades":             int(can_row["trades"]),
                "avg_r":              round(float(can_row["avg_r"]), 4),
                "pf":                 round(float(can_row["pf"]), 3),
                "max_dd_r":           round(float(can_row["max_dd_r"]), 2),
                "win_rate_pct":       round(float(can_row["win_rate_pct"]), 1),
                "cost_floor_pass":    float(can_row["avg_r"]) > COST_FLOOR_R,
            })

    return pd.DataFrame(rows).sort_values(
        ["zone_stability_pct", "avg_r"], ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY SLICES
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_slices(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, am), g in grid.groupby(["filter_mode", "atr_mult"]):
        for _, row in g.iterrows():
            rows.append({
                "filter_mode":  fm,
                "atr_mult":     am,
                "fast_ema":     int(row["fast_ema"]),
                "slow_ema":     int(row["slow_ema"]),
                "trades":       int(row["trades"]),
                "total_r":      float(row["total_r"]),
                "avg_r":        float(row["avg_r"]),
                "pf":           float(row["pf"]),
                "max_dd_r":     float(row["max_dd_r"]),
                "win_rate_pct": float(row["win_rate_pct"]),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(grid: pd.DataFrame, sa: pd.DataFrame) -> None:
    total_combos = len(grid)
    pass_with_cost = sa[(sa["stability_verdict"] == "PASS") & sa["cost_floor_pass"]]
    pass_only      = sa[sa["stability_verdict"] == "PASS"]

    lines = [
        "PHASE T1 — DUAL MA CROSSOVER CONCEPT DISCOVERY",
        "=" * 80,
        "",
        "Method:    Fast EMA crosses above Slow EMA (LONG only, Binance Spot)",
        "Exit:      Fast EMA crosses below Slow EMA (while no Chandelier)",
        "           OR initial ATR stop hit",
        "           OR Chandelier trailing stop (activates at +4R, trails 3×ATR)",
        "Filter:    ema200_price (close > EMA200)  or  ema50_slope (EMA50 slope > 0)",
        "",
        f"Grid:      1 TF × {len(EMA_PAIRS)} pairs × {len(FILTER_MODES)} filters × "
        f"{len(ATR_MULTS)} ATR mults = {total_combos} combos",
        f"Timeframe: {TIMEFRAME}  Bars: ~{LIMIT_BARS} (~4 years)",
        "",
        f"Donchian benchmark:  avg_r={BENCHMARK_AVG_R:+.3f}R  PF={BENCHMARK_PF:.2f}",
        f"§4.2 cost floor:     avg_r > {COST_FLOOR_R:.2f}R  (Binance Spot taker fees)",
        f"Stability threshold: ≥{STABILITY_PASS_PCT:.0f}% of zone combos profitable",
        "",
        "NOTE — ema200_price filter circularity:",
        "  When slow_ema=200, the filter EMA and the slow_ema are the same indicator.",
        "  The crossover condition (fast > EMA200) already captures the uptrend regime.",
        "  The independent EMA200 filter is computed separately but will be near-identical.",
        "  Affected combos: 4 (10/200, 20/200, 30/200, 50/200). Not a disqualifier.",
        "",
        "═" * 80,
        "STABILITY RANKING — top 10 combos by zone stability + avg_r",
        "═" * 80,
    ]

    for _, row in sa.head(10).iterrows():
        v     = row["stability_verdict"]
        cf_ok = "§4.2✓" if row["cost_floor_pass"] else "§4.2✗"
        lines.append(
            f"  [{v:4s}] {cf_ok}  EMA{int(row['fast_ema']):2d}/{int(row['slow_ema']):3d}  "
            f"{row['filter_mode']:14s}  ATR×{row['atr_mult']:.1f}  "
            f"zone={row['zone_pf_gt1']}/{row['zone_total']} ({row['zone_stability_pct']:.0f}%)  "
            f"trades={row['trades']:4d}  avg_r={row['avg_r']:+.4f}  "
            f"pf={row['pf']:.2f}  dd={row['max_dd_r']:+.2f}  win%={row['win_rate_pct']:.1f}"
        )

    # Recommended config
    if not pass_with_cost.empty:
        b = pass_with_cost.iloc[0]
        lines += [
            "",
            "─" * 80,
            "RECOMMENDED CANONICAL CONFIGURATION",
            "─" * 80,
            f"  Timeframe:        {TIMEFRAME}",
            f"  Fast EMA:         {int(b['fast_ema'])}",
            f"  Slow EMA:         {int(b['slow_ema'])}",
            f"  Filter mode:      {b['filter_mode']}",
            f"  ATR mult:         {b['atr_mult']:.1f}",
            f"  Zone stability:   {b['zone_stability_pct']:.0f}%  "
            f"({b['zone_pf_gt1']}/{b['zone_total']} profitable)",
            f"  Trades:           {b['trades']}",
            f"  Avg R:            {b['avg_r']:+.4f}R  [Donchian: {BENCHMARK_AVG_R:+.3f}R]",
            f"  PF:               {b['pf']:.2f}  [Donchian: {BENCHMARK_PF:.2f}]",
            f"  Win rate:         {b['win_rate_pct']:.1f}%  [healthy: 30-45%]",
            f"  Max DD R:         {b['max_dd_r']:+.2f}R",
            "",
            "  T1 GATE: PASS — stability PASS + §4.2 cost floor PASS",
            "  Do NOT proceed to T2 until user confirms this config.",
        ]
    elif not pass_only.empty:
        b = pass_only.iloc[0]
        lines += [
            "",
            "─" * 80,
            "T1 GATE: FAIL — §4.2 COST FLOOR NOT MET",
            "─" * 80,
            f"  Best stable combo: EMA{int(b['fast_ema'])}/{int(b['slow_ema'])}  "
            f"{b['filter_mode']}  avg_r={b['avg_r']:+.4f}R",
            f"  Required avg_r > {COST_FLOOR_R:.2f}R — edge too thin for live trading.",
            "  DECISION: STOP. DualMA does not pass §4.2 on this universe.",
            "  Structural reason: MA crossover entry is lagged — smaller avg_r than breakout.",
        ]
    else:
        lines += [
            "",
            "─" * 80,
            "T1 GATE: FAIL — NO STABILITY PASS",
            "─" * 80,
            "  No combo passed ≥67% stability zone check.",
            "  DECISION: STOP. DualMA has no stable edge on this universe at 1D.",
        ]

    # Full grid per (filter, atr_mult) group
    lines += ["", "═" * 80, "FULL GRID — ALL 60 COMBOS BY (FILTER, ATR_MULT)", "═" * 80]

    for (fm, am), g in grid.groupby(["filter_mode", "atr_mult"]):
        g_sorted = g.sort_values(["slow_ema", "fast_ema"])
        lines += ["", f"  filter={fm}  ATR×{am:.1f}"]
        for _, row in g_sorted.iterrows():
            sign = "[+]" if row["pf"] > 1.0 and row["avg_r"] > 0 else "[-]"
            cf_ok = "§4.2✓" if row["avg_r"] > COST_FLOOR_R else "     "
            lines.append(
                f"    {sign} {cf_ok}  EMA{int(row['fast_ema']):2d}/{int(row['slow_ema']):3d}  "
                f"trades={int(row['trades']):4d}  avg_r={row['avg_r']:+.3f}  "
                f"pf={row['pf']:.2f}  dd={row['max_dd_r']:+.2f}  win={row['win_rate_pct']:.1f}%"
            )

    lines += [
        "",
        "═" * 80,
        "INTERPRETATION (Unger §2.1 + §4.2 + §4.1)",
        "═" * 80,
        "",
        f"  PASS (stability ≥{STABILITY_PASS_PCT:.0f}% AND avg_r > {COST_FLOOR_R:.2f}R):",
        "    → Proceed to T2 after user confirms canonical config.",
        "",
        "  PASS stability but §4.2 FAIL (avg_r ≤ 0.15R):",
        "    → STOP. Edge too thin. MA crossover lag is the structural cause.",
        "    → Do not proceed to T2. Archive and move to next method.",
        "",
        "  NO stability PASS:",
        "    → STOP. No stable zone found. DualMA does not work on this universe at 1D.",
        "    → Consider: different exit logic, relaxed stability zone, or abandon method.",
        "",
        "  §4.1 Win rate (healthy: 30-45%):",
        "    DualMA may show win rates above 45% due to lagged exit (smaller losers cut).",
        "    Win rate > 55% is suspicious — may indicate overfitting or short hold times.",
        "",
        "  Key risk vs Donchian:",
        "    MA crossover is lagged — enters after the move starts, exits after it reverses.",
        "    Expected lower avg_r than Donchian breakout. §4.2 is the most likely failure.",
        "",
        "  T9A paper-live remains FROZEN.",
        "  Do NOT proceed to T2 without user review of this report.",
    ]

    (OUT / "phase_t1_dualma_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("PHASE T1 — DUAL MA CROSSOVER CONCEPT DISCOVERY")
    print("=" * 80)
    print(f"EMA pairs ({len(EMA_PAIRS)}):  {EMA_PAIRS}")
    print(f"Filter modes:   {FILTER_MODES}")
    print(f"ATR mults:      {ATR_MULTS}")
    print(f"Chandelier:     activates at +{ACT_R}R, trails {CH_MULT}×ATR")
    print(f"Benchmark:      Donchian avg_r={BENCHMARK_AVG_R:+.3f}R  PF={BENCHMARK_PF:.2f}")
    print(f"Output:         {OUT}")
    print()

    data = load_all_ohlcv()
    if not data:
        raise RuntimeError("No 1D OHLCV data loaded.")

    print(f"\nRunning grid sweep ({len(EMA_PAIRS)} × {len(FILTER_MODES)} × "
          f"{len(ATR_MULTS)} = {len(EMA_PAIRS)*len(FILTER_MODES)*len(ATR_MULTS)} combos)...")
    t0 = time.time()
    grid = run_grid(data)
    print(f"\nGrid sweep completed in {time.time() - t0:.1f}s")

    print("\nRunning stability analysis...")
    sa = stability_analysis(grid)

    sensitivity = sensitivity_slices(grid)

    # Console summary
    print("\n=== TOP 10 COMBOS by zone_stability_pct + avg_r ===")
    cols = ["filter_mode", "atr_mult", "fast_ema", "slow_ema",
            "zone_stability_pct", "stability_verdict", "avg_r", "pf", "cost_floor_pass"]
    print(sa[cols].head(10).to_string(index=False))

    # Gate decision
    pass_with_cost = sa[(sa["stability_verdict"] == "PASS") & sa["cost_floor_pass"]]
    pass_only      = sa[sa["stability_verdict"] == "PASS"]

    print()
    if not pass_with_cost.empty:
        b = pass_with_cost.iloc[0]
        print(f"T1 GATE: PASS")
        print(f"  Recommended: EMA{int(b['fast_ema'])}/{int(b['slow_ema'])}  "
              f"{b['filter_mode']}  ATR×{b['atr_mult']:.1f}")
        print(f"  Stability: {b['zone_stability_pct']:.0f}%  "
              f"avg_r={b['avg_r']:+.4f}R  PF={b['pf']:.2f}  win%={b['win_rate_pct']:.1f}")
        print("  Do NOT proceed to T2 without user review.")
    elif not pass_only.empty:
        print("T1 GATE: FAIL — §4.2 cost floor not met.")
        print(f"  Best stable: avg_r={pass_only.iloc[0]['avg_r']:+.4f}R < {COST_FLOOR_R:.2f}R required.")
        print("  DualMA edge is too thin. Do not proceed to T2.")
    else:
        print("T1 GATE: FAIL — no stability PASS found.")
        print("  DualMA has no stable zone on this universe at 1D.")

    # Write outputs
    sa.to_csv(OUT / "phase_t1_dualma_stability_ranking.csv", index=False)
    sensitivity.to_csv(OUT / "phase_t1_dualma_n_sensitivity.csv", index=False)
    write_report(grid, sa)

    print("\n[OK] phase_t1_dualma_stability_ranking.csv")
    print("[OK] phase_t1_dualma_n_sensitivity.csv")
    print("[OK] phase_t1_dualma_summary.txt")
    print("\nNext step: review phase_t1_dualma_summary.txt — do NOT proceed to T2 without user confirmation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
