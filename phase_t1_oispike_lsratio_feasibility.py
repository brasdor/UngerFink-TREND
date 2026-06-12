"""
BATCH 2 -- Tests 4 (OI Spike Short) and 5 (L/S Ratio Extreme Short): data feasibility.

Finding: Binance only exposes ~30 days of open interest history and global
long/short account ratio history via API (futures/data/openInterestHist and
futures/data/globalLongShortAccountRatio are limited to the last 30 days).
The local database (trading/data/trading.db) therefore only contains the
collected window. A 2019-2026 T1 backtest (year-by-year, 2022 gate) is not
possible with this data. This script quantifies exactly what exists and
writes the blocker report.

Output: data/research_oispike_funding_t1/  and  data/research_lsratio_funding_t1/
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB = "trading/data/trading.db"
OUT4 = Path("data/research_oispike_funding_t1"); OUT4.mkdir(parents=True, exist_ok=True)
OUT5 = Path("data/research_lsratio_funding_t1"); OUT5.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)
lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(s)

pr("=" * 78)
pr("BATCH 2 FEASIBILITY: OI Spike Short (Test 4) and L/S Ratio Short (Test 5)")
pr("=" * 78)

for table, label in [("open_interest", "Test 4 -- Open Interest"),
                     ("ls_ratio", "Test 5 -- Long/Short Ratio")]:
    df = pd.read_sql(f"SELECT * FROM {table}", con)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    n_sym = df["symbol"].nunique()
    pr(f"\n--- {label} ({table}) ---")
    pr(f"  rows: {len(df)}   symbols: {n_sym}")
    pr(f"  date range: {df['dt'].min().date()} -> {df['dt'].max().date()} "
       f"({(df['dt'].max() - df['dt'].min()).days} days)")
    per_sym = df.groupby("symbol").size()
    pr(f"  rows per symbol: median={per_sym.median():.0f}  max={per_sym.max()}")
    pr(f"  resolution: daily (1 row/day/symbol)")

pr("\n--- VERDICT ---")
pr("  BLOCKED -- INSUFFICIENT HISTORY for T1 validation.")
pr("  Available: ~1 month of daily OI and L/S ratio data (May-Jun 2026).")
pr("  Required:  2019-2026 with year-by-year breakdown; 2022 must be testable.")
pr("  Root cause: Binance API exposes only the last 30 days of OI history")
pr("  and global long/short account ratio. Historical depth cannot be")
pr("  downloaded retroactively from Binance.")
pr("")
pr("  OPTIONS:")
pr("  1. Keep collecting daily (CryptoTradingDailyPipeline already stores both")
pr("     tables) and revisit after 12+ months of accumulation -- still thin.")
pr("  2. Purchase historical OI/LS data from a vendor (Coinglass, Laevitas,")
pr("     Amberdata) -- only path to a 2019-2026 backtest.")
pr("  3. Drop Tests 4/5 from the sweep. Confluence (Test 7) runs with signals")
pr("     A (VolContraction), B (NHF), D (funding >= 0.02%) -- Signal C (OI)")
pr("     excluded for the same reason.")
pr("")
pr("  STATUS: Test 4 BLOCKED (data) -- Test 5 BLOCKED (data)")

report = "\n".join(lines)
for out in (OUT4, OUT5):
    with open(out / "t1_feasibility_report.txt", "w", encoding="ascii", errors="replace") as f:
        f.write(report)
print(f"\nSaved blocker report to {OUT4} and {OUT5}", flush=True)
con.close()
