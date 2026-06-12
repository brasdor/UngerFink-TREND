"""Download 2H/6H/8H OHLCV for the 142 Binance Futures symbols with funding data.

Usage: python download_ohlcv_multi_tf.py <interval>   (interval: 2h, 6h, 8h)
Reuses the proven fapi.binance.com klines approach from download_ohlcv_4h.py
(local PC, no 451 restriction). Maximum available history per symbol.
"""
import sys, time, requests, pandas as pd
from pathlib import Path

INTERVAL = sys.argv[1].lower() if len(sys.argv) > 1 else None
if INTERVAL not in ("2h", "6h", "8h"):
    print("Usage: python download_ohlcv_multi_tf.py <2h|6h|8h>", flush=True)
    raise SystemExit(1)

OUT_DIR = Path(f"data/futures_universe/ohlcv_{INTERVAL}")
FUNDING_DIR = Path("data/futures_universe/funding_rates")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
START_MS = 1546300800000  # 2019-01-01

symbols = sorted([f.stem.replace("_funding", "") for f in FUNDING_DIR.glob("*_funding.csv")])
print(f"[{INTERVAL}] Downloading {len(symbols)} symbols with funding data...", flush=True)


def fetch_symbol(symbol):
    out_path = OUT_DIR / f"{symbol}_{INTERVAL}.csv"
    if out_path.exists():
        return "skip"
    rows = []
    start = START_MS
    while True:
        for attempt in range(3):
            try:
                r = requests.get(BASE_URL, params={
                    "symbol": symbol, "interval": INTERVAL,
                    "startTime": start, "limit": 1500
                }, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    return f"error: {e}"
                time.sleep(2.0 * (attempt + 1))
        if not data:
            break
        for d in data:
            rows.append({
                "timestamp": d[0],
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
            })
        if len(data) < 1500:
            break
        start = data[-1][0] + 1
        time.sleep(0.08)
    if not rows:
        return "no_data"
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    df.to_csv(out_path, index=False)
    return f"ok:{len(df)}"


errors = []
for i, sym in enumerate(symbols, 1):
    result = fetch_symbol(sym)
    if result.startswith("error") or result == "no_data":
        errors.append((sym, result))
    if i % 10 == 0 or result.startswith("error"):
        print(f"[{INTERVAL}] [{i}/{len(symbols)}] {sym}: {result}", flush=True)

# Coverage summary
print(f"\n[{INTERVAL}] ===== COVERAGE SUMMARY =====", flush=True)
files = sorted(OUT_DIR.glob(f"*_{INTERVAL}.csv"))
print(f"[{INTERVAL}] Files on disk: {len(files)}/{len(symbols)}", flush=True)
for f in files:
    df = pd.read_csv(f, usecols=["date"])
    sym = f.stem.replace(f"_{INTERVAL}", "")
    print(f"[{INTERVAL}]   {sym:<22} bars={len(df):6d}  {df['date'].iloc[0]} -> {df['date'].iloc[-1]}", flush=True)
if errors:
    print(f"[{INTERVAL}] ERRORS ({len(errors)}):", flush=True)
    for sym, msg in errors:
        print(f"[{INTERVAL}]   {sym}: {msg}", flush=True)
print(f"[{INTERVAL}] DONE", flush=True)
