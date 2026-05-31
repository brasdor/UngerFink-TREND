import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG BLOCK -- all parameters here, none hardcoded below
# =============================================================================
CONFIG = {
    "method_name": "DualMA",
    "timeframes": ["1d"],
    "asset_universe": "70 USDT spot symbols on Binance",
    "data_root": r"C:\Users\Jean\UngerFink-TREND\data",
    "project_root": r"C:\Users\Jean\UngerFink-TREND",
    "exchange": "binance_spot",
    "leverage": 1.0,
    "risk_per_trade": 0.0025,
    "max_concurrent": [3, 5],
    "backtest_bars": 1500,
    "side": "LONG",
    "cost_floor_r": 0.15,
    "fast_ema": 30,
    "slow_ema": 100,
    "filter_mode": "ema200_price",
    "atr_mult": 2.0,
    "chandelier_activate_r": 4.0,
    "chandelier_atr_mult": 3.0,
    "atr_period": 14,
    "ema_regime_length": 200,
    "ema_slope_n": 50,
    "ema_slope_lookback": 10,
    "fee_per_side": 0.001,
    "allow_short": False,
    "exclude_bases": [
        "USDC", "BUSD", "TUSD", "USDP", "PAXG", "XAUT", "FDUSD",
        "USDD", "DAI", "USD1", "RLUSD", "EUR", "AEUR", "WBTC", "BTTC"
    ],
    "exclude_leveraged_keywords": ["UP/", "DOWN/", "BULL/", "BEAR/", "3L/", "3S/"],
    # T3B specific grid
    "chandelier_atr_mults_grid": [2.0, 3.0, 4.0],
    "chandelier_activate_r_grid": [2.0, 3.0, 4.0],
    # Benchmark from T2
    "t2_output_dir": "research_dualma_t2",
    "t3b_output_dir": "research_dualma_t3b",
    "ohlcv_cache_primary": "research_trend_t15/ohlcv_cache",
    "ohlcv_cache_fallback": "research_dualma_t1/ohlcv_cache",
    # Donchian benchmark for reference
    "donchian_benchmark_avg_r": 0.179,
    "donchian_benchmark_pf": 1.27,
    "donchian_benchmark_total_r": 82.0,
    "donchian_benchmark_max_dd_pct": -1.78,
}

# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT = Path(CONFIG["data_root"])
T2_DIR = DATA_ROOT / CONFIG["t2_output_dir"]
T3B_DIR = DATA_ROOT / CONFIG["t3b_output_dir"]
CACHE_PRIMARY = DATA_ROOT / CONFIG["ohlcv_cache_primary"]
CACHE_FALLBACK = DATA_ROOT / CONFIG["ohlcv_cache_fallback"]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol. Checks primary cache, fallback cache.
    CSV format: timestamp(ms), open, high, low, close, volume
    """
    filename = symbol.replace("/", "_") + f"_{timeframe}.csv"

    for cache_dir in [CACHE_PRIMARY, CACHE_FALLBACK]:
        fpath = cache_dir / filename
        if fpath.exists():
            df = pd.read_csv(fpath)
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
            elif df.index.dtype in [np.int64, np.float64]:
                df.index = pd.to_datetime(df.index, unit="ms", utc=True)
            else:
                df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

    raise FileNotFoundError(f"No OHLCV data found for {symbol} ({timeframe}) in either cache.")

def get_available_symbols(timeframe: str) -> list:
    """Return all symbols available in cache directories."""
    suffix = f"_{timeframe}.csv"
    symbols = set()
    for cache_dir in [CACHE_PRIMARY, CACHE_FALLBACK]:
        if cache_dir.exists():
            for f in cache_dir.glob(f"*{suffix}"):
                sym = f.stem.replace(suffix.replace(".csv", ""), "").replace("_", "/", 1)
                # Reconstruct symbol name: BTC_USDT_1d -> BTC/USDT
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    sym = "/".join(parts[:2])
                    symbols.add(sym)
    return sorted(symbols)

def filter_symbols(symbols: list) -> list:
    """Remove stablecoins, leveraged tokens, and non-USDT pairs."""
    exclude_bases = set(CONFIG["exclude_bases"])
    exclude_kw = CONFIG["exclude_leveraged_keywords"]
    result = []
    for sym in symbols:
        if not sym.endswith("/USDT"):
            continue
        base = sym.split("/")[0]
        if base in exclude_bases:
            continue
        skip = False
        for kw in exclude_kw:
            if kw in sym:
                skip = True
                break
        if not skip:
            result.append(sym)
    return result

# =============================================================================
# CORE BACKTEST ENGINE -- T3B COMBINED EXIT (EMA cross + Chandelier)
# =============================================================================

def backtest_symbol_t3b(
    df: pd.DataFrame,
    fast_ema_span: int,
    slow_ema_span: int,
    atr_period: int,
    atr_mult: float,
    fee_per_side: float,
    filter_mode: str,
    ema_regime_length: int,
    ema_slope_n: int,
    ema_slope_lookback: int,
    chandelier_atr_mult: float,
    chandelier_activate_r: float,
    backtest_bars: int,
) -> list:
    """
    Run DualMA backtest with T3B combined exit logic on a single symbol.

    EXIT LOGIC (T3B):
    At each bar:
      Step 1 -- check stop (initial or chandelier, from START of candle):
               if low <= stop -> exit at stop
      Step 2 -- if chandelier NOT yet active:
               if fast_ema < slow_ema on close -> exit at close
      Step 3 -- AFTER all exit checks:
               if mfe >= chandelier_activate_r, update chandelier stop
                 = highest_high - chandelier_atr_mult * atr

    CRITICAL: chandelier stop updated AFTER exit checks (never before).
    """
    df = df.tail(backtest_bars).copy()
    if len(df) < max(fast_ema_span, slow_ema_span, ema_regime_length) + 50:
        return []

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMAs
    fast_ema = close.ewm(span=fast_ema_span, adjust=False).mean()
    slow_ema_s = close.ewm(span=slow_ema_span, adjust=False).mean()
    ema_regime = close.ewm(span=ema_regime_length, adjust=False).mean()
    atr = compute_atr(df["high"], df["low"], df["close"], atr_period)

    # Filter: ema200_price -> close > EMA200 at entry
    # ema50_slope -> EMA_slope_n trending upward (slope > 0 over lookback)
    if filter_mode == "ema200_price":
        regime_ok = close > ema_regime
    elif filter_mode == "ema50_slope":
        slope_ema = close.ewm(span=ema_slope_n, adjust=False).mean()
        regime_ok = slope_ema > slope_ema.shift(ema_slope_lookback)
    else:
        regime_ok = pd.Series(True, index=df.index)

    trades = []
    in_trade = False
    entry_price = 0.0
    stop_price = 0.0
    entry_bar = 0
    highest_high = 0.0
    chandelier_stop = 0.0
    chandelier_active = False
    mfe_r = 0.0  # max favourable excursion in R

    bars = df.index
    n = len(bars)

    for i in range(1, n):
        bar = bars[i]
        prev_bar = bars[i - 1]

        o = df.loc[bar, "open"]
        h = df.loc[bar, "high"]
        l = df.loc[bar, "low"]
        c = df.loc[bar, "close"]

        fe_curr = fast_ema.iloc[i]
        se_curr = slow_ema_s.iloc[i]
        fe_prev = fast_ema.iloc[i - 1]
        se_prev = slow_ema_s.iloc[i - 1]
        atr_curr = atr.iloc[i]
        reg_ok = regime_ok.iloc[i]

        if not in_trade:
            # Entry: fast EMA crosses above slow EMA on bar close
            crossover = (fe_prev <= se_prev) and (fe_curr > se_curr)
            if crossover and reg_ok and atr_curr > 0:
                in_trade = True
                entry_price = c  # enter at close of crossover bar
                stop_price = entry_price - atr_mult * atr_curr
                entry_bar = i
                highest_high = h
                chandelier_stop = 0.0
                chandelier_active = False
                mfe_r = 0.0
        else:
            # -- Step 1: Check stop from START of candle ----------------------
            # Use stop_price as it was at the beginning of this bar
            stop_at_bar_start = stop_price  # saved before any update this bar

            risk = entry_price - stop_at_bar_start
            if risk <= 0:
                # Degenerate trade -- close at close
                net_r = (c - entry_price) / (entry_price * atr_mult / atr.iloc[entry_bar]) if entry_bar < n else 0.0
                fee_r = (2 * fee_per_side * entry_price) / (entry_price - stop_price) if (entry_price - stop_price) > 0 else 0.0
                trades.append({
                    "entry_bar": bars[entry_bar].isoformat(),
                    "exit_bar": bar.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": c,
                    "net_r": net_r - fee_r,
                    "exit_reason": "degen_close",
                    "chandelier_active": chandelier_active,
                })
                in_trade = False
                continue

            risk_at_entry = entry_price - (entry_price - atr_mult * atr.iloc[entry_bar])

            exit_price = None
            exit_reason = None

            # Step 1: stop hit
            if l <= stop_at_bar_start:
                exit_price = stop_at_bar_start
                exit_reason = "stop"

            # Step 2 (only if not yet stopped): if chandelier NOT active, check EMA cross
            if exit_price is None and not chandelier_active:
                if fe_curr < se_curr:
                    exit_price = c
                    exit_reason = "ema_cross"

            # -- If exited ----------------------------------------------------
            if exit_price is not None:
                raw_r = (exit_price - entry_price) / risk_at_entry if risk_at_entry > 0 else 0.0
                fee_r = (2 * fee_per_side * entry_price) / risk_at_entry if risk_at_entry > 0 else 0.0
                net_r = raw_r - fee_r
                trades.append({
                    "entry_bar": bars[entry_bar].isoformat(),
                    "exit_bar": bar.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_r": net_r,
                    "exit_reason": exit_reason,
                    "chandelier_active": chandelier_active,
                })
                in_trade = False
                continue

            # -- Still in trade: update MFE and chandelier AFTER exit checks -
            # Update highest high
            if h > highest_high:
                highest_high = h

            # Compute current MFE in R
            current_mfe_r = (highest_high - entry_price) / risk_at_entry if risk_at_entry > 0 else 0.0
            if current_mfe_r > mfe_r:
                mfe_r = current_mfe_r

            # Step 3: update chandelier AFTER exit checks
            if mfe_r >= chandelier_activate_r:
                new_chandelier = highest_high - chandelier_atr_mult * atr_curr
                if not chandelier_active:
                    chandelier_active = True
                    chandelier_stop = new_chandelier
                else:
                    # Chandelier only moves up (ratchet)
                    chandelier_stop = max(chandelier_stop, new_chandelier)
                # Update stop_price to chandelier (only moves up)
                stop_price = max(stop_price, chandelier_stop)

    return trades


def run_backtest_all_symbols(
    symbols: list,
    timeframe: str,
    fast_ema_span: int,
    slow_ema_span: int,
    chandelier_atr_mult: float,
    chandelier_activate_r: float,
    cfg: dict,
) -> pd.DataFrame:
    """Run T3B backtest across all symbols and return a trades DataFrame."""
    all_trades = []
    loaded = 0
    failed = 0

    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe)
        except FileNotFoundError:
            failed += 1
            continue

        if len(df) < 200:
            failed += 1
            continue

        trades = backtest_symbol_t3b(
            df=df,
            fast_ema_span=fast_ema_span,
            slow_ema_span=slow_ema_span,
            atr_period=cfg["atr_period"],
            atr_mult=cfg["atr_mult"],
            fee_per_side=cfg["fee_per_side"],
            filter_mode=cfg["filter_mode"],
            ema_regime_length=cfg["ema_regime_length"],
            ema_slope_n=cfg["ema_slope_n"],
            ema_slope_lookback=cfg["ema_slope_lookback"],
            chandelier_atr_mult=chandelier_atr_mult,
            chandelier_activate_r=chandelier_activate_r,
            backtest_bars=cfg["backtest_bars"],
        )

        for t in trades:
            t["symbol"] = sym

        all_trades.extend(trades)
        loaded += 1

    print(f"    Loaded {loaded} symbols, failed/skipped {failed}")

    if not all_trades:
        return pd.DataFrame()

    df_trades = pd.DataFrame(all_trades)
    df_trades["entry_bar"] = pd.to_datetime(df_trades["entry_bar"], utc=True)
    df_trades["exit_bar"] = pd.to_datetime(df_trades["exit_bar"], utc=True)
    return df_trades


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(df_trades: pd.DataFrame, label: str = "") -> dict:
    if df_trades is None or len(df_trades) == 0:
        return {
            "label": label, "n_trades": 0, "total_r": 0.0, "avg_r": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "max_dd_r": 0.0,
        }

    net_r = df_trades["net_r"].values
    n = len(net_r)
    total_r = net_r.sum()
    avg_r = net_r.mean()
    wins = (net_r > 0).sum()
    win_rate = wins / n if n > 0 else 0.0
    gross_profit = net_r[net_r > 0].sum() if wins > 0 else 0.0
    gross_loss = abs(net_r[net_r < 0].sum()) if (net_r < 0).any() else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max drawdown in R (running)
    cum = np.cumsum(net_r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd_r = dd.min() if len(dd) > 0 else 0.0

    return {
        "label": label,
        "n_trades": n,
        "total_r": round(total_r, 4),
        "avg_r": round(avg_r, 4),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(pf, 4),
        "max_dd_r": round(max_dd_r, 4),
    }


def compute_equity_curve(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades is None or len(df_trades) == 0:
        return pd.DataFrame(columns=["exit_bar", "cumulative_r"])
    df_sorted = df_trades.sort_values("exit_bar").copy()
    df_sorted["cumulative_r"] = df_sorted["net_r"].cumsum()
    return df_sorted[["exit_bar", "symbol", "net_r", "cumulative_r"]]


def compute_asset_summary(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades is None or len(df_trades) == 0:
        return pd.DataFrame()
    rows = []
    for sym, grp in df_trades.groupby("symbol"):
        m = compute_metrics(grp, label=sym)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("total_r", ascending=False)


# =============================================================================
# GATE CHECKS
# =============================================================================

def gate_cost_floor(metrics: dict, threshold: float = 0.15) -> bool:
    """§4.2: avg_r must exceed cost floor."""
    return metrics["avg_r"] > threshold

def gate_win_rate(metrics: dict, low: float = 30.0, high: float = 45.0) -> tuple:
    """§4.1: win rate should be 30-45%. Returns (pass, warn_message)."""
    wr = metrics["win_rate"]
    if wr < low:
        return False, f"WIN_RATE_LOW: {wr:.1f}% < {low}%"
    if wr > high:
        return False, f"WIN_RATE_HIGH: {wr:.1f}% > {high}% (suspect overfitting)"
    return True, ""

def gate_profit_factor(metrics: dict, threshold: float = 1.0) -> bool:
    """PF must be > 1.0."""
    return metrics["profit_factor"] > threshold

def gate_total_r(metrics: dict, baseline_total_r: float) -> bool:
    """T3B must beat T2 total_r."""
    return metrics["total_r"] > baseline_total_r

def gate_not_catastrophic_dd(metrics: dict, baseline_dd: float, tolerance: float = 2.0) -> bool:
    """Max DD must not be more than tolerance× worse than baseline."""
    if baseline_dd >= 0:
        return True
    limit = baseline_dd * tolerance
    return metrics["max_dd_r"] >= limit


# =============================================================================
# T2 BASELINE LOADER
# =============================================================================

def load_t2_baseline() -> dict:
    """Load T2 summary metrics as baseline."""
    summary_path = T2_DIR / "phase_t2_summary.csv"
    if not summary_path.exists():
        print(f"  WARNING: T2 summary not found at {summary_path}. Using defaults.")
        return {
            "n_trades": 0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "label": "T2_baseline_missing",
        }

    df = pd.read_csv(summary_path)
    # Expecting a single-row summary CSV
    if len(df) == 0:
        return {"total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_dd_r": 0.0, "n_trades": 0,
                "label": "T2_empty"}

    row = df.iloc[0].to_dict()
    # Normalise key names
    def get(d, *keys):
        for k in keys:
            if k in d:
                return float(d[k])
        return 0.0

    return {
        "label": "T2_baseline",
        "n_trades": int(get(row, "total_trades", "n_trades", "trades")),
        "total_r": get(row, "total_r"),
        "avg_r": get(row, "avg_r"),
        "win_rate": get(row, "win_rate_pct", "win_rate"),
        "profit_factor": get(row, "profit_factor"),
        "max_dd_r": get(row, "max_dd_r"),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE T3B -- EXIT ENGINEERING (DualMA, Binance Spot, LONG)")
    print("=" * 70)

    ensure_dir(T3B_DIR)

    # -- 1. Load T2 baseline --------------------------------------------------
    print("\n[1/6] Loading T2 baseline...")
    t2_baseline = load_t2_baseline()
    print(f"  T2 baseline: n={t2_baseline['n_trades']}, "
          f"total_r={t2_baseline['total_r']:.3f}R, "
          f"avg_r={t2_baseline['avg_r']:.4f}R, "
          f"win_rate={t2_baseline['win_rate']:.1f}%, "
          f"PF={t2_baseline['profit_factor']:.3f}, "
          f"max_dd={t2_baseline['max_dd_r']:.3f}R")

    # -- 2. Load symbols ------------------------------------------------------
    print("\n[2/6] Discovering available symbols...")
    timeframe = CONFIG["timeframes"][0]
    all_syms = get_available_symbols(timeframe)
    symbols = filter_symbols(all_syms)
    print(f"  Found {len(all_syms)} total symbols -> {len(symbols)} after filtering")

    if len(symbols) == 0:
        print("ERROR: No symbols found. Check cache directories:")
        print(f"  Primary: {CACHE_PRIMARY}")
        print(f"  Fallback: {CACHE_FALLBACK}")
        sys.exit(1)

    # -- 3. Grid search over chandelier params --------------------------------
    print("\n[3/6] Running T3B grid search...")
    print(f"  Canonical entry: fast_ema={CONFIG['fast_ema']}, slow_ema={CONFIG['slow_ema']}")
    print(f"  Chandelier ATR mults grid: {CONFIG['chandelier_atr_mults_grid']}")
    print(f"  Chandelier activation R grid: {CONFIG['chandelier_activate_r_grid']}")
    print(f"  Symbols: {len(symbols)}, Bars per symbol: {CONFIG['backtest_bars']}")
    print()

    grid_results = []
    all_variant_trades = {}

    combos = list(product(
        CONFIG["chandelier_atr_mults_grid"],
        CONFIG["chandelier_activate_r_grid"]
    ))
    total_combos = len(combos)

    for idx, (chan_mult, chan_act) in enumerate(combos, 1):
        label = f"chan_mult={chan_mult}_act={chan_act}R"
        print(f"  [{idx}/{total_combos}] {label}")

        df_trades = run_backtest_all_symbols(
            symbols=symbols,
            timeframe=timeframe,
            fast_ema_span=CONFIG["fast_ema"],
            slow_ema_span=CONFIG["slow_ema"],
            chandelier_atr_mult=chan_mult,
            chandelier_activate_r=chan_act,
            cfg=CONFIG,
        )

        metrics = compute_metrics(df_trades, label=label)
        metrics["chandelier_atr_mult"] = chan_mult
        metrics["chandelier_activate_r"] = chan_act

        # Gate checks
        beats_t2_total = gate_total_r(metrics, t2_baseline["total_r"])
        passes_cost_floor = gate_cost_floor(metrics, CONFIG["cost_floor_r"])
        passes_pf = gate_profit_factor(metrics)
        wr_pass, wr_msg = gate_win_rate(metrics)
        not_bad_dd = gate_not_catastrophic_dd(metrics, t2_baseline["max_dd_r"], tolerance=2.0)

        metrics["beats_t2_total_r"] = beats_t2_total
        metrics["passes_cost_floor"] = passes_cost_floor
        metrics["passes_pf"] = passes_pf
        metrics["passes_win_rate"] = wr_pass
        metrics["win_rate_msg"] = wr_msg
        metrics["not_catastrophic_dd"] = not_bad_dd
        # Overall pass: must beat T2 total_r AND pass cost floor AND PF > 1
        metrics["overall_pass"] = (
            beats_t2_total and passes_cost_floor and passes_pf and not_bad_dd
        )

        print(f"    -> total_r={metrics['total_r']:.3f}R  avg_r={metrics['avg_r']:.4f}R  "
              f"win%={metrics['win_rate']:.1f}%  PF={metrics['profit_factor']:.3f}  "
              f"max_dd={metrics['max_dd_r']:.3f}R  "
              f"beats_t2={beats_t2_total}  §4.2={passes_cost_floor}  PASS={metrics['overall_pass']}")
        if wr_msg:
            print(f"    !! Win rate note: {wr_msg}")

        grid_results.append(metrics)
        all_variant_trades[label] = df_trades

    # -- 4. Select best variant -----------------------------------------------
    print("\n[4/6] Selecting best T3B variant...")

    df_grid = pd.DataFrame(grid_results)

    # Sort by: overall_pass first, then total_r descending
    df_grid_sorted = df_grid.sort_values(
        ["overall_pass", "total_r"], ascending=[False, False]
    ).reset_index(drop=True)

    passing = df_grid_sorted[df_grid_sorted["overall_pass"]]
    
    if len(passing) == 0:
        print("\n  WARNING: No variant beats T2 on all gate checks.")
        print("  Selecting best variant by total_r (may still be useful diagnostic).")
        best_row = df_grid_sorted.iloc[0]
    else:
        best_row = passing.iloc[0]

    best_label = best_row["label"]
    best_chan_mult = best_row["chandelier_atr_mult"]
    best_chan_act = best_row["chandelier_activate_r"]

    print(f"\n  Best variant: {best_label}")
    print(f"    chandelier_atr_mult = {best_chan_mult}")
    print(f"    chandelier_activate_r = {best_chan_act}R")
    print(f"    total_r = {best_row['total_r']:.3f}R")
    print(f"    avg_r   = {best_row['avg_r']:.4f}R")
    print(f"    win%    = {best_row['win_rate']:.1f}%")
    print(f"    PF      = {best_row['profit_factor']:.3f}")
    print(f"    max_dd  = {best_row['max_dd_r']:.3f}R")
    print(f"    overall_pass = {best_row['overall_pass']}")

    best_trades = all_variant_trades[best_label]

    # -- 5. Fat-tail diagnostic -----------------------------------------------
    print("\n[5/6] Fat-tail diagnostic: T2 vs T3B best variant (top-5 symbols)...")

    t2_trades_path = T2_DIR / "phase_t2_trades.csv"
    if t2_trades_path.exists():
        df_t2_trades = pd.read_csv(t2_trades_path)
        t2_asset = df_t2_trades.groupby("symbol")["net_r"].sum().sort_values(ascending=False).head(5)
        print("  T2 top-5 symbols by R:")
        for sym, r in t2_asset.items():
            print(f"    {sym}: {r:.3f}R")
    else:
        print("  T2 trades not found -- skipping T2 side of fat-tail diagnostic")
        t2_asset = pd.Series(dtype=float)

    if best_trades is not None and len(best_trades) > 0:
        t3b_asset = best_trades.groupby("symbol")["net_r"].sum().sort_values(ascending=False).head(5)
        print(f"\n  T3B best ({best_label}) top-5 symbols by R:")
        for sym, r in t3b_asset.items():
            print(f"    {sym}: {r:.3f}R")

        # Compare overlap
        t3b_top5 = set(t3b_asset.index)
        if len(t2_asset) > 0:
            t2_top5 = set(t2_asset.index)
            overlap = t2_top5 & t3b_top5
            print(f"\n  Overlap in top-5: {len(overlap)}/5 symbols in common")
            # Check if chandelier extended winners
            for sym in overlap:
                t2_r = t2_asset.get(sym, 0.0)
                t3b_r = t3b_asset.get(sym, 0.0)
                delta = t3b_r - t2_r
                direction = "+extended" if delta > 0 else "-reduced"
                print(f"    {sym}: T2={t2_r:.3f}R -> T3B={t3b_r:.3f}R ({direction} {delta:+.3f}R)")
    else:
        print("  No T3B trades to analyse.")

    # Concentration check §4.7
    if best_trades is not None and len(best_trades) > 0:
        total_r_val = best_trades["net_r"].sum()
        if total_r_val != 0:
            asset_r = best_trades.groupby("symbol")["net_r"].sum()
            top1_r = asset_r.max()
            top1_pct = top1_r / total_r_val * 100
            top1_sym = asset_r.idxmax()
            if top1_pct > 50:
                print(f"\n  !! §4.7 CONCENTRATION WARNING: {top1_sym} = {top1_pct:.1f}% of total R")
            else:
                print(f"\n  §4.7 Concentration: top-1 {top1_sym} = {top1_pct:.1f}% (OK)")

    # -- 6. Write output files ------------------------------------------------
    print("\n[6/6] Writing output files...")

    # phase_t3b_wide_exit_trades.csv
    if best_trades is not None and len(best_trades) > 0:
        trades_out = best_trades.copy()
        trades_out["chandelier_atr_mult"] = best_chan_mult
        trades_out["chandelier_activate_r"] = best_chan_act
        out_path = T3B_DIR / "phase_t3b_wide_exit_trades.csv"
        trades_out.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(trades_out)} trades)")
    else:
        print("  WARNING: No trades to save for best variant.")

    # phase_t3b_wide_exit_asset_summary.csv
    asset_summary = compute_asset_summary(best_trades)
    if len(asset_summary) > 0:
        out_path = T3B_DIR / "phase_t3b_wide_exit_asset_summary.csv"
        asset_summary.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(asset_summary)} symbols)")

    # phase_t3b_wide_exit_equity.csv
    equity_curve = compute_equity_curve(best_trades)
    if len(equity_curve) > 0:
        out_path = T3B_DIR / "phase_t3b_wide_exit_equity.csv"
        equity_curve.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    # phase_t3b_wide_exit_summary.csv -- all variants + best flagged
    df_grid_sorted["is_best"] = df_grid_sorted["label"] == best_label
    out_path = T3B_DIR / "phase_t3b_wide_exit_summary.csv"
    # Re-order columns for readability
    col_order = [
        "label", "chandelier_atr_mult", "chandelier_activate_r",
        "n_trades", "total_r", "avg_r", "win_rate", "profit_factor", "max_dd_r",
        "beats_t2_total_r", "passes_cost_floor", "passes_pf",
        "passes_win_rate", "not_catastrophic_dd", "overall_pass", "is_best",
        "win_rate_msg",
    ]
    col_order = [c for c in col_order if c in df_grid_sorted.columns]
    df_grid_sorted[col_order].to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df_grid_sorted)} variants)")

    # -- Human-readable report ------------------------------------------------
    best_metrics = compute_metrics(best_trades, label=best_label)
    
    report_lines = [
        "=" * 70,
        "PHASE T3B -- EXIT ENGINEERING REPORT",
        f"Method: {CONFIG['method_name']} | Exchange: {CONFIG['exchange']} | Side: {CONFIG['side']}",
        "=" * 70,
        "",
        "CONFIGURATION",
        f"  Entry:  fast_ema={CONFIG['fast_ema']}, slow_ema={CONFIG['slow_ema']}",
        f"  Filter: {CONFIG['filter_mode']} (ema_regime_length={CONFIG['ema_regime_length']})",
        f"  ATR mult (initial stop): {CONFIG['atr_mult']}",
        f"  Fee per side: {CONFIG['fee_per_side']*100:.2f}%",
        f"  Backtest bars: {CONFIG['backtest_bars']}",
        "",
        "T2 BASELINE",
        f"  n_trades:       {t2_baseline['n_trades']}",
        f"  total_r:        {t2_baseline['total_r']:.3f}R",
        f"  avg_r:          {t2_baseline['avg_r']:.4f}R",
        f"  win_rate:       {t2_baseline['win_rate']:.1f}%",
        f"  profit_factor:  {t2_baseline['profit_factor']:.3f}",
        f"  max_dd_r:       {t2_baseline['max_dd_r']:.3f}R",
        "",
        "GRID RESULTS SUMMARY",
        f"  Total variants tested: {total_combos}",
        f"  Variants passing all gates: {len(passing)}",
        "",
    ]

    for _, row in df_grid_sorted.iterrows():
        flag = "* BEST" if row["label"] == best_label else ("  PASS" if row["overall_pass"] else "  FAIL")
        report_lines.append(
            f"  {flag} | {row['label']:35s} | "
            f"total_r={row['total_r']:7.3f}R | avg_r={row['avg_r']:.4f}R | "
            f"win%={row['win_rate']:.1f}% | PF={row['profit_factor']:.3f} | "
            f"dd={row['max_dd_r']:.3f}R"
        )

    report_lines += [
        "",
        "BEST VARIANT",
        f"  Label:                  {best_label}",
        f"  chandelier_atr_mult:    {best_chan_mult}",
        f"  chandelier_activate_r:  {best_chan_act}R",
        f"  n_trades:               {best_metrics['n_trades']}",
        f"  total_r:                {best_metrics['total_r']:.3f}R",
        f"  avg_r:                  {best_metrics['avg_r']:.4f}R",
        f"  win_rate:               {best_metrics['win_rate']:.1f}%",
        f"  profit_factor:          {best_metrics['profit_factor']:.3f}",
        f"  max_dd_r:               {best_metrics['max_dd_r']:.3f}R",
        "",
        "GATE CHECKS (best variant)",
        f"  §4.2 Cost floor (avg_r > {CONFIG['cost_floor_r']}R):  "
            f"{'PASS' if gate_cost_floor(best_metrics, CONFIG['cost_floor_r']) else 'FAIL'} "
            f"(avg_r={best_metrics['avg_r']:.4f}R)",
        f"  §4.1 Win rate (30-45%):                    "
            f"{'PASS' if gate_win_rate(best_metrics)[0] else 'WARN/FAIL'} "
            f"({best_metrics['win_rate']:.1f}%)",
        f"  PF > 1.0:                                  "
            f"{'PASS' if gate_profit_factor(best_metrics) else 'FAIL'} "
            f"(PF={best_metrics['profit_factor']:.3f})",
        f"  Beats T2 total_r ({t2_baseline['total_r']:.2f}R):          "
            f"{'PASS' if gate_total_r(best_metrics, t2_baseline['total_r']) else 'FAIL'} "
            f"(total_r={best_metrics['total_r']:.3f}R)",
        f"  DD not catastrophic (<=2x T2 dd):           "
            f"{'PASS' if gate_not_catastrophic_dd(best_metrics, t2_baseline['max_dd_r']) else 'FAIL'}",
        "",
        "FROZEN CANONICAL EXIT CONFIG (after T3B)",
        f"  CANONICAL_EXIT:           combined EMA_cross + Chandelier",
        f"  CHANDELIER_ATR_MULT:      {best_chan_mult}",
        f"  CHANDELIER_ACTIVATION:    {best_chan_act}R",
        "",
        "COMPARISON vs DONCHIAN BENCHMARK",
        f"  Donchian avg_r:    {CONFIG['donchian_benchmark_avg_r']:.3f}R  |  DualMA T3B avg_r:   {best_metrics['avg_r']:.4f}R",
        f"  Donchian PF:       {CONFIG['donchian_benchmark_pf']:.2f}    |  DualMA T3B PF:      {best_metrics['profit_factor']:.3f}",
        f"  Donchian total_r:  {CONFIG['donchian_benchmark_total_r']:.1f}R  |  DualMA T3B total_r: {best_metrics['total_r']:.3f}R",
        "",
        f"  STATUS: {'PASS -- proceed to T4' if best_row['overall_pass'] else 'WARN -- review before T4'}",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print()
    print(report_text)

    report_path = T3B_DIR / "phase_t3b_wide_exit_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Saved: {report_path}")

    # -- Final gate decision --------------------------------------------------
    print("\n" + "=" * 70)
    if best_row["overall_pass"]:
        print("T3B GATE: PASS")
        print(f"Best variant: chandelier_atr_mult={best_chan_mult}, "
              f"chandelier_activate_r={best_chan_act}R")
        print("Canonical exit config frozen. Proceed to T4.")
        print("=" * 70)
        sys.exit(0)
    else:
        # Check if any variant passes at least cost floor + PF (partial pass)
        partial = df_grid[df_grid["passes_cost_floor"] & df_grid["passes_pf"]]
        if len(partial) > 0:
            print("T3B GATE: PARTIAL PASS")
            print("  No variant beats T2 total_r, but cost floor and PF pass.")
            print("  Chandelier may not be improving over simple EMA exit.")
            print("  Recommended action: review T2 vs T3B fat-tail analysis,")
            print("  consider whether chandelier architecture is suited to this method.")
            print("  You may proceed to T4 with simple EMA cross exit if preferred.")
            print("=" * 70)
            sys.exit(0)
        else:
            print("T3B GATE: FAIL")
            print("  No variant passes §4.2 cost floor and PF > 1.0.")
            print("  Do NOT proceed to T4.")
            print("  Review entry signal quality -- avg_r may be structurally too low.")
            print("=" * 70)
            sys.exit(1)


if __name__ == "__main__":
    main()