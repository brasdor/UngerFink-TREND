#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T13 — TIMEFRAME ARCHETYPE VALIDATION

Offline only. T9 paper-live remains unchanged.

Compares the same trend archetype across:
4H / 6H / 8H / 1D

Core logic fixed:
- Donchian 20 breakout
- EMA50 slope filter
- ATR14 initial stop, 2 ATR
- Chandelier activation 4R
- Chandelier ATR multiplier 3
- closed candles only
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import ccxt

ROOT = Path.cwd()
OUT = ROOT / "data" / "research_trend_t13"
CACHE = OUT / "ohlcv_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = ["4h", "6h", "8h", "1d"]
LIMITS = {"4h": 1500, "6h": 1200, "8h": 1000, "1d": 800}
UNIVERSE_SIZE = 70

DONCHIAN_N = 20
EMA_N = 50
EMA_SLOPE_LOOKBACK = 10
ATR_N = 14
INIT_STOP_ATR = 2.0
CH_ACTIVATE_R = 4.0
CH_ATR_MULT = 3.0

EXCLUDE_BASES = {
    "USDC","BUSD","TUSD","USDP","PAXG","XAUT","FDUSD","USDD","DAI",
    "USD1","RLUSD","EUR","AEUR","WBTC","BTTC"
}
PREFERRED = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "DOGE/USDT","AVAX/USDT","LINK/USDT","TRX/USDT","DOT/USDT","BCH/USDT",
    "LTC/USDT","UNI/USDT","NEAR/USDT","APT/USDT","SUI/USDT","ICP/USDT"
]


def tf_hours(tf):
    return int(tf[:-1]) if tf.endswith("h") else int(tf[:-1]) * 24


def safe_symbol(s):
    return s.replace("/", "_").replace(":", "_")


def make_exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def get_universe(ex):
    markets = ex.load_markets()
    syms = []
    for sym, m in markets.items():
        if not m.get("spot", False):
            continue
        if not m.get("active", True):
            continue
        if m.get("quote") != "USDT":
            continue
        if ":" in sym:
            continue
        if m.get("base") in EXCLUDE_BASES:
            continue
        if any(x in sym for x in ["UP/","DOWN/","BULL/","BEAR/","3L/","3S/"]):
            continue
        syms.append(sym)
    syms = sorted(set(syms))
    return ([s for s in PREFERRED if s in syms] + [s for s in syms if s not in PREFERRED])[:UNIVERSE_SIZE]


def fetch_ohlcv(ex, sym, tf):
    path = CACHE / f"{safe_symbol(sym)}_{tf}.csv"
    last = None
    for attempt in range(3):
        try:
            raw = ex.fetch_ohlcv(sym, timeframe=tf, limit=LIMITS[tf])
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df.to_csv(path, index=False)
            return df
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    if path.exists():
        print(f"[CACHE] {sym} {tf}")
        return pd.read_csv(path)
    raise RuntimeError(f"{sym} {tf}: {last}")


def closed_candles(raw, tf):
    df = raw.copy()
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["close_time"] = df["time"] + pd.Timedelta(hours=tf_hours(tf))
    df = df[df["close_time"] <= pd.Timestamp.utcnow()].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["time","open","high","low","close"]).reset_index(drop=True)


def add_indicators(df):
    d = df.copy()
    d["ema"] = d["close"].ewm(span=EMA_N, adjust=False).mean()
    d["ema_slope"] = d["ema"] - d["ema"].shift(EMA_SLOPE_LOOKBACK)
    pc = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - pc).abs(),
        (d["low"] - pc).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()
    d["don_high"] = d["high"].shift(1).rolling(DONCHIAN_N).max()
    d["don_low"] = d["low"].shift(1).rolling(DONCHIAN_N).min()
    return d


def pf(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    g = r[r > 0].sum()
    l = -r[r < 0].sum()
    if l <= 1e-12:
        return np.inf if g > 0 else 0.0
    return float(g / l)


def max_dd_r(r):
    r = pd.to_numeric(r, errors="coerce").fillna(0).values
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def losing_streak(r):
    best = cur = 0
    for x in pd.to_numeric(r, errors="coerce").fillna(0):
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def base_pos(side, t, i, entry, stop, risk):
    return {
        "side": side,
        "entry_time": t,
        "entry_i": i,
        "entry": entry,
        "initial_stop": stop,
        "stop": stop,
        "risk": risk,
        "hh": entry,
        "ll": entry,
        "mfe": 0.0,
        "trail": False,
        "trail_time": pd.NaT,
        "trail_price": np.nan,
    }


def make_trade(sym, tf, pos, exit_time, exit_price, r, exit_i):
    return {
        "timeframe": tf,
        "variant": f"{tf.upper()}_DONCHIAN_ACT4_ATR3",
        "symbol": sym,
        "side": pos["side"],
        "entry_time": pos["entry_time"],
        "exit_time": exit_time,
        "entry_price": pos["entry"],
        "exit_price": exit_price,
        "initial_stop": pos["initial_stop"],
        "final_stop": pos["stop"],
        "initial_risk_per_unit": pos["risk"],
        "net_r": r,
        "bars_held": exit_i - pos["entry_i"],
        "max_favorable_r": pos["mfe"],
        "chandelier_active": pos["trail"],
        "chandelier_activation_time": pos["trail_time"],
        "chandelier_activation_price": pos["trail_price"],
        "ch_activate_r": CH_ACTIVATE_R,
        "ch_atr_mult": CH_ATR_MULT,
        "exit_reason": "STOP_OR_TRAILING",
    }


def simulate_symbol(sym, tf, df):
    d = add_indicators(df)
    warmup = max(DONCHIAN_N, EMA_N + EMA_SLOPE_LOOKBACK, ATR_N) + 5
    pos = None
    trades = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue

        t = row["time"]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])

        if pos is not None:
            if pos["side"] == "LONG":
                pos["hh"] = max(pos["hh"], high)
                pos["mfe"] = max(pos["mfe"], (high - pos["entry"]) / max(pos["risk"], 1e-12))
                if pos["mfe"] >= CH_ACTIVATE_R:
                    was = pos["trail"]
                    pos["trail"] = True
                    pos["stop"] = max(pos["stop"], pos["hh"] - CH_ATR_MULT * atr)
                    if not was:
                        pos["trail_time"] = t
                        pos["trail_price"] = close
                if low <= pos["stop"]:
                    exit_price = pos["stop"]
                    r = (exit_price - pos["entry"]) / max(pos["risk"], 1e-12)
                    trades.append(make_trade(sym, tf, pos, t, exit_price, r, i))
                    pos = None
                    continue
            else:
                pos["ll"] = min(pos["ll"], low)
                pos["mfe"] = max(pos["mfe"], (pos["entry"] - low) / max(pos["risk"], 1e-12))
                if pos["mfe"] >= CH_ACTIVATE_R:
                    was = pos["trail"]
                    pos["trail"] = True
                    pos["stop"] = min(pos["stop"], pos["ll"] + CH_ATR_MULT * atr)
                    if not was:
                        pos["trail_time"] = t
                        pos["trail_price"] = close
                if high >= pos["stop"]:
                    exit_price = pos["stop"]
                    r = (pos["entry"] - exit_price) / max(pos["risk"], 1e-12)
                    trades.append(make_trade(sym, tf, pos, t, exit_price, r, i))
                    pos = None
                    continue

        if pos is None:
            long_sig = close > float(row["don_high"]) and float(row["ema_slope"]) > 0
            short_sig = close < float(row["don_low"]) and float(row["ema_slope"]) < 0
            if long_sig:
                stop = close - INIT_STOP_ATR * atr
                risk = close - stop
                if risk > 0:
                    pos = base_pos("LONG", t, i, close, stop, risk)
            elif short_sig:
                stop = close + INIT_STOP_ATR * atr
                risk = stop - close
                if risk > 0:
                    pos = base_pos("SHORT", t, i, close, stop, risk)

    return trades


def summarize(df, groups):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(groups, keys)}
        r = pd.to_numeric(g["net_r"], errors="coerce").dropna()
        row.update({
            "trades": len(r),
            "total_r": float(r.sum()) if len(r) else 0.0,
            "avg_r": float(r.mean()) if len(r) else 0.0,
            "median_r": float(r.median()) if len(r) else 0.0,
            "pf": pf(r),
            "max_dd_r": max_dd_r(r),
            "win_rate_pct": float((r > 0).mean() * 100) if len(r) else 0.0,
            "best_r": float(r.max()) if len(r) else 0.0,
            "worst_r": float(r.min()) if len(r) else 0.0,
            "max_losing_streak": losing_streak(r),
            "avg_bars_held": float(pd.to_numeric(g["bars_held"], errors="coerce").mean()) if "bars_held" in g.columns else 0.0,
            "trail_activation_rate_pct": float(g["chandelier_active"].astype(bool).mean() * 100) if "chandelier_active" in g.columns else 0.0,
            "avg_mfe_r": float(pd.to_numeric(g["max_favorable_r"], errors="coerce").mean()) if "max_favorable_r" in g.columns else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_summary(df):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for tf, g in df.groupby("timeframe"):
        g = g.reset_index(drop=True)
        events = []
        row_by_id = {}
        for i, row in g.iterrows():
            row_by_id[i] = row
            events.append((row["entry_time"], 1, i))
            events.append((row["exit_time"], 0, i))
        events.sort(key=lambda x: (x[0], x[1]))

        open_ids = set()
        samples = []
        max_open = 0
        for _, typ, tid in events:
            if typ == 0:
                open_ids.discard(tid)
            else:
                open_ids.add(tid)
            max_open = max(max_open, len(open_ids))
            if open_ids:
                long_n = sum(1 for oid in open_ids if str(row_by_id[oid]["side"]).upper() == "LONG")
                short_n = sum(1 for oid in open_ids if str(row_by_id[oid]["side"]).upper() == "SHORT")
                samples.append({"open": len(open_ids), "max_same_side": max(long_n, short_n)})
        s = pd.DataFrame(samples)
        rows.append({
            "timeframe": tf,
            "raw_trades": len(g),
            "raw_max_simultaneous_open": max_open,
            "avg_raw_open": float(s["open"].mean()) if not s.empty else 0.0,
            "max_raw_same_side": int(s["max_same_side"].max()) if not s.empty else 0,
            "avg_raw_same_side": float(s["max_same_side"].mean()) if not s.empty else 0.0,
        })
    return pd.DataFrame(rows)


def write_report(summary, overlap):
    lines = [
        "PHASE T13 — TIMEFRAME ARCHETYPE VALIDATION REPORT",
        "=" * 80,
        "",
        "Offline only. T9 paper-live remains unchanged.",
        "Same Donchian/EMA/ATR/Chandelier archetype across 4H/6H/8H/1D.",
        "Goal: isolate timeframe effect, not optimize parameters.",
        "",
        "TIMEFRAME SUMMARY",
        "-" * 80,
    ]
    if summary.empty:
        lines.append("No results.")
    else:
        for _, r in summary.sort_values(["pf", "total_r"], ascending=False).iterrows():
            lines.append(
                f"{r['timeframe']}: trades={int(r['trades'])}, totalR={r['total_r']:.2f}, "
                f"avgR={r['avg_r']:.3f}, PF={r['pf']:.2f}, DD={r['max_dd_r']:.2f}, "
                f"win={r['win_rate_pct']:.1f}%, bestR={r['best_r']:.2f}, "
                f"trailRate={r['trail_activation_rate_pct']:.1f}%"
            )
    lines += ["", "OVERLAP SUMMARY", "-" * 80]
    if overlap.empty:
        lines.append("No overlap results.")
    else:
        for _, r in overlap.iterrows():
            lines.append(
                f"{r['timeframe']}: rawTrades={int(r['raw_trades'])}, "
                f"maxOpen={int(r['raw_max_simultaneous_open'])}, "
                f"avgOpen={r['avg_raw_open']:.2f}, maxSameSide={int(r['max_raw_same_side'])}"
            )
    lines += [
        "",
        "Interpretation:",
        "- Prefer structural stability, not the single highest PF.",
        "- Too many trades may mean noise; too few trades may mean weak sample.",
        "- A good timeframe should show acceptable PF, DD, sample size, and lower crowding.",
        "- Do not update T9 until the result is confirmed with portfolio replay.",
    ]
    (OUT / "phase_t13_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE T13 — TIMEFRAME ARCHETYPE VALIDATION")
    print("=" * 80)
    print(f"Output: {OUT}")

    ex = make_exchange()
    syms = get_universe(ex)
    print(f"Universe: {len(syms)} assets")

    all_trades = []
    for tf in TIMEFRAMES:
        print(f"\n===== TIMEFRAME {tf} =====")
        for idx, sym in enumerate(syms, 1):
            print(f"[{tf}] {idx:03d}/{len(syms)} {sym}")
            try:
                raw = fetch_ohlcv(ex, sym, tf)
                df = closed_candles(raw, tf)
                min_bars = max(DONCHIAN_N, EMA_N + EMA_SLOPE_LOOKBACK, ATR_N) + 50
                if len(df) < min_bars:
                    print(f"  skipped bars={len(df)}")
                    continue
                trades = simulate_symbol(sym, tf, df)
                all_trades.extend(trades)
                print(f"  bars={len(df)} trades={len(trades)}")
            except Exception as e:
                print(f"  ERROR: {e}")
            time.sleep(0.05)

    trades = pd.DataFrame(all_trades)
    trades_path = OUT / "phase_t13_timeframe_trades.csv"
    summary_path = OUT / "phase_t13_timeframe_summary.csv"
    asset_path = OUT / "phase_t13_timeframe_asset_summary.csv"
    month_path = OUT / "phase_t13_timeframe_month_summary.csv"
    overlap_path = OUT / "phase_t13_timeframe_overlap_summary.csv"

    if trades.empty:
        trades.to_csv(trades_path, index=False)
        write_report(pd.DataFrame(), pd.DataFrame())
        print("No trades generated.")
        return 0

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce", utc=True)
    trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")
    trades.to_csv(trades_path, index=False)

    summary = summarize(trades, ["timeframe"]).sort_values(["pf", "total_r"], ascending=False)
    summary.to_csv(summary_path, index=False)

    asset_summary = summarize(trades, ["timeframe", "symbol"])
    asset_summary.to_csv(asset_path, index=False)

    month_summary = summarize(trades, ["timeframe", "month"])
    month_summary.to_csv(month_path, index=False)

    overlap = overlap_summary(trades)
    overlap.to_csv(overlap_path, index=False)

    write_report(summary, overlap)

    print("\nTIMEFRAME SUMMARY")
    cols = ["timeframe", "trades", "total_r", "avg_r", "pf", "max_dd_r", "win_rate_pct", "best_r", "trail_activation_rate_pct"]
    print(summary[cols].to_string(index=False))
    print("\n[OK] phase_t13_timeframe_trades.csv")
    print("[OK] phase_t13_timeframe_summary.csv")
    print("[OK] phase_t13_timeframe_asset_summary.csv")
    print("[OK] phase_t13_timeframe_month_summary.csv")
    print("[OK] phase_t13_timeframe_overlap_summary.csv")
    print("[OK] phase_t13_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
