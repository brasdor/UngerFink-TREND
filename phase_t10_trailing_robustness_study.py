#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T10 — TREND TRAILING ROBUSTNESS STUDY

Offline research only. It does NOT change T9A/T9D paper-live.

Tests same 6H Donchian/EMA/ATR trend concept with different Chandelier exits:
- activation: 2R, 3R, 4R, 5R
- ATR multiplier: 3, 4, 5

Outputs:
data/research_trend_t10/
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import ccxt

ROOT = Path.cwd()
OUT = ROOT / "data" / "research_trend_t10"
CACHE = OUT / "ohlcv_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "6h"
LIMIT_BARS = 1000
UNIVERSE_SIZE = 70

DONCHIAN_N = 20
EMA_N = 50
EMA_SLOPE_LOOKBACK = 10
ATR_N = 14
INIT_STOP_ATR = 2.0

ACT_LEVELS = [2.0, 3.0, 4.0, 5.0]
CH_MULTS = [3.0, 4.0, 5.0]

EXCLUDE_BASES = {
    "USDC","BUSD","TUSD","USDP","PAXG","XAUT","FDUSD","USDD","DAI",
    "USD1","RLUSD","EUR","AEUR","WBTC","BTTC"
}
PREFERRED = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "DOGE/USDT","AVAX/USDT","LINK/USDT","TRX/USDT","DOT/USDT","BCH/USDT",
    "LTC/USDT","UNI/USDT","NEAR/USDT","APT/USDT","SUI/USDT","ICP/USDT"
]


def safe_symbol(s):
    return s.replace("/", "_").replace(":", "_")


def exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def universe(ex):
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
    ordered = [s for s in PREFERRED if s in syms] + [s for s in syms if s not in PREFERRED]
    return ordered[:UNIVERSE_SIZE]


def fetch(ex, sym):
    p = CACHE / f"{safe_symbol(sym)}_{TIMEFRAME}.csv"
    for attempt in range(3):
        try:
            raw = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=LIMIT_BARS)
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df.to_csv(p, index=False)
            return df
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    if p.exists():
        return pd.read_csv(p)
    raise RuntimeError(f"{sym}: {last}")


def closed_candles(raw):
    df = raw.copy()
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["close_time"] = df["time"] + pd.Timedelta(hours=6)
    df = df[df["close_time"] <= pd.Timestamp.utcnow()].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["time","open","high","low","close"]).reset_index(drop=True)


def indicators(df):
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


def dd_r(r):
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


def simulate(sym, df, act_r, mult):
    d = indicators(df)
    warmup = max(DONCHIAN_N, EMA_N + EMA_SLOPE_LOOKBACK, ATR_N) + 5
    pos = None
    trades = []

    for i in range(warmup, len(d)):
        row = d.iloc[i]
        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue

        t = row["time"]
        close, high, low, atr = map(float, [row["close"], row["high"], row["low"], row["atr"]])

        if pos:
            if pos["side"] == "LONG":
                pos["hh"] = max(pos["hh"], high)
                mfe = (high - pos["entry"]) / max(pos["risk"], 1e-12)
                pos["mfe"] = max(pos["mfe"], mfe)

                if pos["mfe"] >= act_r:
                    was = pos["trail"]
                    pos["trail"] = True
                    pos["stop"] = max(pos["stop"], pos["hh"] - mult * atr)
                    if not was:
                        pos["trail_time"] = t
                        pos["trail_price"] = close

                if low <= pos["stop"]:
                    exit_price = pos["stop"]
                    r = (exit_price - pos["entry"]) / max(pos["risk"], 1e-12)
                    trades.append(row_trade(sym, pos, t, exit_price, r, i, act_r, mult))
                    pos = None
                    continue

            else:
                pos["ll"] = min(pos["ll"], low)
                mfe = (pos["entry"] - low) / max(pos["risk"], 1e-12)
                pos["mfe"] = max(pos["mfe"], mfe)

                if pos["mfe"] >= act_r:
                    was = pos["trail"]
                    pos["trail"] = True
                    pos["stop"] = min(pos["stop"], pos["ll"] + mult * atr)
                    if not was:
                        pos["trail_time"] = t
                        pos["trail_price"] = close

                if high >= pos["stop"]:
                    exit_price = pos["stop"]
                    r = (pos["entry"] - exit_price) / max(pos["risk"], 1e-12)
                    trades.append(row_trade(sym, pos, t, exit_price, r, i, act_r, mult))
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


def row_trade(sym, pos, exit_time, exit_price, r, exit_i, act_r, mult):
    return {
        "variant": f"ACT{act_r:g}_ATR{mult:g}",
        "symbol": sym,
        "timeframe": TIMEFRAME,
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
        "ch_activate_r": act_r,
        "ch_atr_mult": mult,
        "exit_reason": "STOP_OR_TRAILING",
    }


def summarize(df, groups):
    rows = []
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
            "max_dd_r": dd_r(r),
            "win_rate_pct": float((r > 0).mean() * 100) if len(r) else 0.0,
            "best_r": float(r.max()) if len(r) else 0.0,
            "worst_r": float(r.min()) if len(r) else 0.0,
            "max_losing_streak": losing_streak(r),
            "avg_bars_held": float(pd.to_numeric(g["bars_held"], errors="coerce").mean()),
            "trail_activation_rate_pct": float(g["chandelier_active"].astype(bool).mean() * 100),
            "avg_mfe_r": float(pd.to_numeric(g["max_favorable_r"], errors="coerce").mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def report(summary):
    lines = []
    lines.append("PHASE T10 — TREND TRAILING ROBUSTNESS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Offline comparison only. Live T9A/T9D is unchanged.")
    lines.append("Core entry is unchanged; only Chandelier activation and width are compared.")
    lines.append("")
    if summary.empty:
        lines.append("No trades generated.")
    else:
        lines.append("MAIN SUMMARY")
        lines.append("-" * 70)
        for _, r in summary.sort_values(["pf", "total_r"], ascending=False).iterrows():
            lines.append(
                f"{r['variant']}: trades={int(r['trades'])}, "
                f"totalR={r['total_r']:.2f}, avgR={r['avg_r']:.3f}, "
                f"PF={r['pf']:.2f}, DD={r['max_dd_r']:.2f}, "
                f"win={r['win_rate_pct']:.1f}%, trailRate={r['trail_activation_rate_pct']:.1f}%, "
                f"bestR={r['best_r']:.2f}"
            )
    (OUT / "phase_t10_master_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE T10 — TREND TRAILING ROBUSTNESS STUDY")
    print("=" * 80)
    print(f"Output: {OUT}")

    ex = exchange()
    syms = universe(ex)
    print(f"Universe: {len(syms)} assets")

    all_trades = []

    for idx, sym in enumerate(syms, 1):
        print(f"[{idx:03d}/{len(syms)}] {sym}")
        try:
            raw = fetch(ex, sym)
            df = closed_candles(raw)
            if len(df) < 150:
                print(f"  skipped: only {len(df)} bars")
                continue

            count_before = len(all_trades)
            for act in ACT_LEVELS:
                for mult in CH_MULTS:
                    all_trades.extend(simulate(sym, df, act, mult))

            print(f"  bars={len(df)} trades_added={len(all_trades)-count_before}")

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(0.05)

    trades = pd.DataFrame(all_trades)
    trades_path = OUT / "phase_t10_trailing_trades.csv"
    summary_path = OUT / "phase_t10_trailing_summary.csv"
    asset_path = OUT / "phase_t10_trailing_asset_summary.csv"
    month_path = OUT / "phase_t10_trailing_month_summary.csv"

    if trades.empty:
        trades.to_csv(trades_path, index=False)
        report(pd.DataFrame())
        print("No trades generated.")
        return 0

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce", utc=True)
    trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")

    trades.to_csv(trades_path, index=False)

    summary = summarize(trades, ["variant", "ch_activate_r", "ch_atr_mult"]).sort_values(["pf", "total_r"], ascending=False)
    summary.to_csv(summary_path, index=False)

    asset_summary = summarize(trades, ["variant", "symbol"])
    asset_summary.to_csv(asset_path, index=False)

    month_summary = summarize(trades, ["variant", "month"])
    month_summary.to_csv(month_path, index=False)

    report(summary)

    print("")
    print("TOP RESULTS")
    print(summary[["variant","trades","total_r","avg_r","pf","max_dd_r","win_rate_pct","trail_activation_rate_pct","best_r"]].head(20).to_string(index=False))
    print("")
    print("[OK] phase_t10_trailing_trades.csv")
    print("[OK] phase_t10_trailing_summary.csv")
    print("[OK] phase_t10_trailing_asset_summary.csv")
    print("[OK] phase_t10_trailing_month_summary.csv")
    print("[OK] phase_t10_master_report.txt")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
