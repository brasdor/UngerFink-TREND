"""
Regenerate S5 MomentumFactor trades from frozen config.
Covers 2020-01-01 through 2026-06-19 (IS + OOS).
1D bars, 290-symbol Futures universe, biweekly rebalance.

Frozen config (from engines/phase_t9b_momentum_factor_paper_engine.py):
  Lookback: 20 days
  Long basket: top 20 by 20-day return, price > EMA200
  Short basket: bottom 10 by 20-day return, no filter
  Rebalance: every 14+ days, on Fridays
  Allocation: 50% long / 50% short, equal weight
  Cost: 0.15% round-trip on turnover
  Capital: $8,000 starting

Each biweekly symbol position = one trade row.
R-multiple = position return / (equal-weight allocation fraction).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date as Date, timedelta

DATA_1D = Path("data/futures_universe/ohlcv_1d")
OUT_DIR = Path("data/research_momentum_factor_t8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = pd.Timestamp("2020-01-01")
FULL_END   = pd.Timestamp("2026-06-19")
IS_START   = pd.Timestamp("2020-01-01")
IS_END     = pd.Timestamp("2022-12-31")

LOOKBACK   = 20
LONG_K     = 20
SHORT_K    = 10
EMA_SPAN   = 200
REBAL_DAYS = 14
COST_RT    = 0.0015
STARTING_EQ = 8_000.0

DOCUMENTED_IS_CAGR  = 47.77
DOCUMENTED_IS_SHARPE = 1.755

# ── Load all 1D data into a close price DataFrame ────────────────────
files = sorted(DATA_1D.glob("*_1d.csv"))
symbols = [f.stem.replace("_1d", "") for f in files]
print(f"Universe: {len(symbols)} Futures symbols (1D)")

close_dict = {}
warmup = pd.Timedelta(days=300)
for sym in symbols:
    df = pd.read_csv(DATA_1D / f"{sym}_1d.csv")
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[(df["time"] >= FULL_START - warmup) & (df["time"] <= FULL_END)]
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    if len(df) < 50: continue
    close_dict[sym] = df["close"]

close = pd.DataFrame(close_dict)
close = close.sort_index()
print(f"Close matrix: {close.shape[0]} dates x {close.shape[1]} symbols")

# ── Compute EMA200 for all symbols ──────────────────────────────────
ema200 = close.ewm(span=EMA_SPAN, min_periods=20, adjust=False).mean()

# ── Generate biweekly rebalance dates ────────────────────────────────
def generate_rebal_dates(start, end):
    dates = []
    d = start.date() if hasattr(start, 'date') else start
    end_d = end.date() if hasattr(end, 'date') else end
    # Find first Friday on or after start
    while d.weekday() != 4:
        d += timedelta(days=1)
    last_rebal = None
    while d <= end_d:
        if last_rebal is None or (d - last_rebal).days >= REBAL_DAYS:
            dates.append(d)
            last_rebal = d
        d += timedelta(days=7)  # next Friday
    return dates

rebal_dates = generate_rebal_dates(FULL_START, FULL_END)
print(f"Rebalance dates: {len(rebal_dates)} ({rebal_dates[0]} to {rebal_dates[-1]})")

# ── Backtest ─────────────────────────────────────────────────────────
all_trades = []
equity = STARTING_EQ
equity_history = []

for idx in range(len(rebal_dates) - 1):
    entry_date = rebal_dates[idx]
    exit_date = rebal_dates[idx + 1]
    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date)

    if entry_ts not in close.index:
        avail = close.index[close.index <= entry_ts]
        if len(avail) == 0: continue
        entry_ts = avail[-1]
    if exit_ts not in close.index:
        avail = close.index[close.index <= exit_ts]
        if len(avail) == 0: continue
        exit_ts = avail[-1]

    # Lookback return
    lb_ts = close.index[close.index <= entry_ts]
    if len(lb_ts) < LOOKBACK + 1: continue
    t0 = lb_ts[-1]
    t_lb = lb_ts[-(LOOKBACK + 1)]

    close_t0 = close.loc[t0].dropna()
    close_tlb = close.loc[t_lb].dropna()
    common = close_t0.index.intersection(close_tlb.index)
    if len(common) < SHORT_K + 5: continue

    ret = (close_t0[common] / close_tlb[common] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    ema_t0 = ema200.loc[t0].dropna()

    # Long: price > EMA200
    long_eligible = [s for s in ret.index
                     if s in ema_t0.index and float(close_t0.get(s, 0)) > float(ema_t0.get(s, 0))]
    sig_long = ret.reindex(long_eligible).dropna()
    sig_short = ret.copy()

    actual_long_k = min(len(sig_long), LONG_K)
    longs = list(sig_long.nlargest(actual_long_k).index) if actual_long_k > 0 else []
    shorts = list(sig_short.nsmallest(SHORT_K).index) if len(sig_short) >= SHORT_K else []

    if not shorts: continue

    # Compute turnover cost (simplified)
    cost_pct = COST_RT
    eq_after_cost = equity * (1 - cost_pct)

    # Position sizing
    long_alloc_per = (eq_after_cost * 0.5 / len(longs)) if longs else 0
    short_alloc_per = (eq_after_cost * 0.5 / len(shorts)) if shorts else 0

    # Get exit prices
    close_exit = close.loc[exit_ts] if exit_ts in close.index else None
    if close_exit is None: continue
    close_entry = close.loc[t0]

    period_pnl = 0.0

    for sym in longs:
        ep = close_entry.get(sym)
        xp = close_exit.get(sym)
        if ep is None or xp is None or ep <= 0: continue
        ret_pct = (xp - ep) / ep
        trade_pnl = long_alloc_per * ret_pct
        period_pnl += trade_pnl
        r_mult = ret_pct / (1.0 / len(longs)) if len(longs) > 0 else 0
        all_trades.append({
            "symbol": sym,
            "entry_ts": int(entry_ts.timestamp() * 1000),
            "exit_ts": int(exit_ts.timestamp() * 1000),
            "r": round(r_mult, 6),
            "year": entry_date.year,
            "side": "long",
        })

    for sym in shorts:
        ep = close_entry.get(sym)
        xp = close_exit.get(sym)
        if ep is None or xp is None or ep <= 0: continue
        ret_pct = (ep - xp) / ep  # short: profit when price falls
        trade_pnl = short_alloc_per * ret_pct
        period_pnl += trade_pnl
        r_mult = ret_pct / (1.0 / len(shorts)) if len(shorts) > 0 else 0
        all_trades.append({
            "symbol": sym,
            "entry_ts": int(entry_ts.timestamp() * 1000),
            "exit_ts": int(exit_ts.timestamp() * 1000),
            "r": round(r_mult, 6),
            "year": entry_date.year,
            "side": "short",
        })

    equity += period_pnl
    equity_history.append({
        "date": str(exit_date),
        "equity": round(equity, 2),
        "n_long": len(longs),
        "n_short": len(shorts),
    })

trades_df = pd.DataFrame(all_trades)
trades_df.to_csv(OUT_DIR / "s5_regenerated_trades.csv", index=False)

print(f"\nTotal trades (full period): {len(trades_df)}")

# ── IS Verification ──────────────────────────────────────────────────
is_trades = trades_df[trades_df["year"].between(2020, 2022)].copy() if not trades_df.empty else pd.DataFrame()

# Compute IS equity curve for CAGR/Sharpe
is_eq_hist = [e for e in equity_history if int(e["date"][:4]) <= 2022]

if is_eq_hist:
    is_eq_values = [STARTING_EQ] + [e["equity"] for e in is_eq_hist]
    is_final = is_eq_values[-1]
    n_years = 3.0  # 2020-2022
    is_cagr = (is_final / STARTING_EQ) ** (1 / n_years) - 1
    is_cagr_pct = is_cagr * 100

    # Sharpe from biweekly returns
    biweekly_rets = []
    for i in range(1, len(is_eq_values)):
        biweekly_rets.append(is_eq_values[i] / is_eq_values[i-1] - 1)
    biweekly_rets = np.array(biweekly_rets)
    periods_per_year = 26  # ~26 biweekly periods per year
    if len(biweekly_rets) > 1 and biweekly_rets.std() > 0:
        is_sharpe = biweekly_rets.mean() / biweekly_rets.std() * np.sqrt(periods_per_year)
    else:
        is_sharpe = 0
else:
    is_cagr_pct = 0; is_sharpe = 0; is_final = STARTING_EQ

print("\n" + "="*60)
print("S5 REGENERATION — IS VERIFICATION")
print("="*60)
print(f"IS period: 2020-01-01 to 2022-12-31")
print(f"IS trades: {len(is_trades)}")
print(f"IS equity: ${STARTING_EQ:.0f} -> ${is_final:.0f}")
print(f"IS CAGR:   {is_cagr_pct:+.2f}%")
print(f"IS Sharpe: {is_sharpe:.3f}")
print(f"Documented IS CAGR:  +{DOCUMENTED_IS_CAGR}%")
print(f"Documented IS Sharpe: {DOCUMENTED_IS_SHARPE}")
print()

cagr_gate = abs(is_cagr_pct - DOCUMENTED_IS_CAGR) / DOCUMENTED_IS_CAGR <= 0.20
sharpe_gate = abs(is_sharpe - DOCUMENTED_IS_SHARPE) / DOCUMENTED_IS_SHARPE <= 0.20

print(f"CAGR within 20%:  {is_cagr_pct:+.2f}% vs +{DOCUMENTED_IS_CAGR}%  "
      f"[{'PASS' if cagr_gate else 'FAIL'}]")
print(f"Sharpe within 20%: {is_sharpe:.3f} vs {DOCUMENTED_IS_SHARPE}  "
      f"[{'PASS' if sharpe_gate else 'FAIL'}]")

if not (cagr_gate and sharpe_gate):
    print("\nSTOPPING — IS verification failed. Do not use for Step 20.")
    print("Investigate divergence before proceeding.")
else:
    print(f"\nPASS — S5 regeneration verified.")

print(f"\nYear breakdown:")
if not trades_df.empty:
    for yr in sorted(trades_df["year"].unique()):
        yr_t = trades_df[trades_df["year"] == yr]
        yr_long = yr_t[yr_t["side"] == "long"]
        yr_short = yr_t[yr_t["side"] == "short"]
        print(f"  {yr}: {len(yr_t)} trades ({len(yr_long)}L/{len(yr_short)}S)  "
              f"avg_r={yr_t['r'].mean():+.4f}")

eq_df = pd.DataFrame(equity_history)
eq_df.to_csv(OUT_DIR / "s5_equity_curve.csv", index=False)

print(f"\nSaved to: {OUT_DIR / 's5_regenerated_trades.csv'}")
print(f"Equity curve: {OUT_DIR / 's5_equity_curve.csv'}")
print(f"Columns: {list(trades_df.columns)}")
print(f"Rows: {len(trades_df)}")
print("="*60)
