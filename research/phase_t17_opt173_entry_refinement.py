#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T17 Opt-17.3 -- Entry Refinement (3 Variants)

Baseline: DonchianLong_UniverseV2 / N=20 / ema200_price / ATR×2.0 /
          Chandelier ACT4_ATR3 / max8
          avg_r=+1.101R  PF=3.072  CAGR=+14.3%  trades=461

Variant A -- Limit pullback entry
  Signal fires when close > Donchian upper. Place limit at
  breakout_level - 0.25×ATR(14). Fill if low reaches limit within 2 bars;
  else cancel. Entry at limit price; stop = entry - ATR×2.0.

Variant B -- Price-confirmation entry
  Signal fires when close > Donchian upper (bar i).
  Enter at close of bar i+1 only if close[i+1] > don_upper[i].
  (Next bar must confirm by closing above the original breakout level.)

Variant C -- Partial entry with add-on
  Enter 50% at breakout close. Add remaining 50% if price reaches +1R
  within 5 bars. net_r = 0.5×r1 [+ 0.5×r2 if add-on triggered].
  Stop applies to full position from the initial stop price.

All variants:
  Universe  : 24 symbols / filtered_symbols_v2_included_only.csv
  Stability : N=[15,20,25] all profitable
  T4        : MC 2000 runs, cost stress, period splits, remove-best
  Guard     : ≥80 trades
  T5/T6     : max8 portfolio + $10k capital execution
  Output    : data/research_donchian_entryV2_{A,B,C}/

Results printed as a single comparison table at the end.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT       = Path(__file__).resolve().parents[1]
OHLCV_DIR  = ROOT / "data" / "universe" / "ohlcv_1d"
IN_SYMBOLS = ROOT / "data" / "universe" / "filtered_symbols_v2_included_only.csv"

R_COL   = "net_r"
ENTRY_T = "entry_time"
EXIT_T  = "exit_time"

# Frozen base config
ATR_N      = 14
STOP_MULT  = 2.0
CHAN_ACT_R = 4.0
CHAN_TRAIL = 3.0

# Variant A
LIMIT_PULLBACK_ATR_FRAC = 0.25   # limit = breakout - 0.25×ATR
LIMIT_CANCEL_BARS       = 2      # bars to wait before cancelling

# Variant C
PARTIAL_SIZE   = 0.5    # initial position fraction
ADDON_R_LEVEL  = 1.0    # add-on at +1R
ADDON_WINDOW   = 5      # bars to wait for add-on

# Stability zone
STABILITY_N = [15, 20, 25]
CANONICAL_N = 20

# T4
MC_RUNS     = 2000
MC_BLOCKS   = [1, 3, 5, 10, 20]
MC_SEED     = 42
EXTRA_COSTS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# T6
START_CAP      = 10_000.0
RISK_PCT       = 0.0025
KILL_SWITCH_DD = -0.35
PORT_MAX_OPEN  = 8

BASELINE  = dict(avg_r=1.1011, pf=3.072, cagr_pct=14.3, trades=461)
MIN_TRADES = 80


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1.0)
    out[n - 1] = float(np.nanmean(close[:n]))
    for i in range(n, len(close)):
        if np.isfinite(close[i]) and np.isfinite(out[i - 1]):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    nb = len(close); tr = np.full(nb, np.nan)
    for i in range(1, nb):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.full(nb, np.nan)
    if nb > n:
        atr[n] = float(np.nanmean(tr[1:n+1]))
        for i in range(n+1, nb):
            if np.isfinite(tr[i]) and np.isfinite(atr[i-1]):
                atr[i] = (atr[i-1]*(n-1) + tr[i]) / n
    return atr


# =============================================================================
# DATA
# =============================================================================

def load_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "_")
    path  = OHLCV_DIR / f"{clean}_1d.csv"
    if not path.exists(): return None
    df = pd.read_csv(path)
    col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[col]):
        df["time"] = pd.to_datetime(df[col], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    return (df[["time","open","high","low","close"]]
            .dropna(subset=["time","close"]).sort_values("time").reset_index(drop=True))


# =============================================================================
# VARIANT A — LIMIT PULLBACK
# =============================================================================

def backtest_a(df: pd.DataFrame, symbol: str, entry_n: int = CANONICAL_N) -> List[dict]:
    """Enter at limit = breakout_level - 0.25×ATR, cancel after 2 bars."""
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(entry_n, ATR_N, 200) + 20: return []
    exit_n = entry_n // 2

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()
    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades: List[dict] = []
    pos: Optional[dict] = None
    pending: Optional[dict] = None   # pending limit order

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])): continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])): continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])): continue

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry"])/pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["entry"])/pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            exit_px = exit_rsn = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_rsn = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_rsn = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_rsn = cl[i], "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
                    exit_time=ts[i], entry_price=pos["entry"], exit_price=exit_px,
                    initial_stop=pos["stop"], initial_risk=pos["risk"],
                    exit_reason=exit_rsn, bars_held=pos["bars"],
                    net_r=net_r, mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
                pos = None
                pending = None
                continue
            if pos["mfe_r"] >= CHAN_ACT_R: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*CHAN_TRAIL)

        # ── TRY LIMIT FILL ───────────────────────────────────────────────────
        if pos is None and pending is not None:
            if lo[i] <= pending["limit_px"]:
                # Fill at limit price
                lx = pending["limit_px"]; risk = pending["risk"]
                stop_px = lx - risk
                pos = dict(entry=lx, stop=stop_px, risk=risk, entry_time=ts[i],
                           hh=hi[i], chan_active=False, chan_stop=stop_px,
                           mfe_r=0.0, mae_r=0.0, bars=1)
                pending = None
            else:
                pending["bars_left"] -= 1
                if pending["bars_left"] <= 0:
                    pending = None

        # ── CHECK FOR NEW SIGNAL ─────────────────────────────────────────────
        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            limit_px = don_upper[i] - LIMIT_PULLBACK_ATR_FRAC * atr14[i-1]
            risk = atr14[i-1] * STOP_MULT
            pending = dict(limit_px=limit_px, risk=risk,
                           bars_left=LIMIT_CANCEL_BARS)

    # Force-close
    if pos is not None:
        exit_px = cl[-1]; net_r = (exit_px-pos["entry"])/pos["risk"]
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1], entry_price=pos["entry"], exit_price=exit_px,
            initial_stop=pos["stop"], initial_risk=pos["risk"],
            exit_reason="end_of_data", bars_held=pos["bars"],
            net_r=net_r, mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
    return trades


# =============================================================================
# VARIANT B — PRICE-CONFIRMATION
# =============================================================================

def backtest_b(df: pd.DataFrame, symbol: str, entry_n: int = CANONICAL_N) -> List[dict]:
    """Enter at close of next bar only if it closes above the original breakout level."""
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(entry_n, ATR_N, 200) + 20: return []
    exit_n = entry_n // 2

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()
    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades: List[dict] = []
    pos:     Optional[dict] = None
    confirm: Optional[dict] = None   # pending confirmation

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])): continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])): continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])): continue

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry"])/pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["entry"])/pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            exit_px = exit_rsn = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_rsn = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_rsn = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_rsn = cl[i], "midline_exit"

            if exit_px is not None:
                net_r = (exit_px - pos["entry"]) / pos["risk"]
                trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
                    exit_time=ts[i], entry_price=pos["entry"], exit_price=exit_px,
                    initial_stop=pos["stop"], initial_risk=pos["risk"],
                    exit_reason=exit_rsn, bars_held=pos["bars"],
                    net_r=net_r, mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
                pos = None
                confirm = None
                continue
            if pos["mfe_r"] >= CHAN_ACT_R: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*CHAN_TRAIL)

        # ── CHECK CONFIRMATION ───────────────────────────────────────────────
        if pos is None and confirm is not None:
            if cl[i] > confirm["breakout_level"]:
                # Confirmed — enter at this bar's close
                risk = atr14[i-1] * STOP_MULT
                stop_px = cl[i] - risk
                pos = dict(entry=cl[i], stop=stop_px, risk=risk, entry_time=ts[i],
                           hh=hi[i], chan_active=False, chan_stop=stop_px,
                           mfe_r=0.0, mae_r=0.0, bars=1)
            confirm = None   # one-shot — consumed regardless of outcome

        # ── CHECK FOR NEW SIGNAL ─────────────────────────────────────────────
        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            # Signal: arm confirmation for next bar
            confirm = dict(breakout_level=don_upper[i])

    # Force-close
    if pos is not None:
        exit_px = cl[-1]; net_r = (exit_px-pos["entry"])/pos["risk"]
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1], entry_price=pos["entry"], exit_price=exit_px,
            initial_stop=pos["stop"], initial_risk=pos["risk"],
            exit_reason="end_of_data", bars_held=pos["bars"],
            net_r=net_r, mae_r=min(pos["mae_r"], net_r), mfe_r=pos["mfe_r"]))
    return trades


# =============================================================================
# VARIANT C — PARTIAL + ADD-ON
# =============================================================================

def backtest_c(df: pd.DataFrame, symbol: str, entry_n: int = CANONICAL_N) -> List[dict]:
    """50% position at breakout; add 50% if +1R within 5 bars.
    net_r = 0.5*r1 [+ 0.5*r2 if add-on triggered].
    Without add-on: only 0.5× exposure, so net_r = 0.5*r1.
    """
    df = df.sort_values("time").reset_index(drop=True)
    nb = len(df)
    if nb < max(entry_n, ATR_N, 200) + 20: return []
    exit_n = entry_n // 2

    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    ts = df["time"].to_numpy()
    ema200    = _ema(cl, 200)
    atr14     = _atr(hi, lo, cl, ATR_N)
    don_upper = pd.Series(hi).shift(1).rolling(entry_n).max().to_numpy()
    don_lower = pd.Series(lo).shift(1).rolling(exit_n).min().to_numpy()

    trades: List[dict] = []
    pos: Optional[dict] = None

    for i in range(1, nb):
        if not (np.isfinite(cl[i]) and np.isfinite(hi[i]) and np.isfinite(lo[i])): continue
        if not (np.isfinite(ema200[i-1]) and np.isfinite(atr14[i-1])): continue
        if not (np.isfinite(don_upper[i]) and np.isfinite(don_lower[i])): continue

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if pos is not None:
            pos["mfe_r"] = max(pos["mfe_r"], (hi[i]-pos["entry1"])/pos["risk"])
            pos["mae_r"] = min(pos["mae_r"], (lo[i]-pos["entry1"])/pos["risk"])
            pos["hh"]    = max(pos["hh"], hi[i])
            pos["bars"] += 1

            # Check add-on (before exit checks — add-on might hit then stop in same bar)
            if not pos["addon_done"] and pos["addon_bars_left"] > 0:
                if hi[i] >= pos["addon_target"]:
                    pos["addon_done"]  = True
                    pos["addon_price"] = pos["addon_target"]
                else:
                    pos["addon_bars_left"] -= 1

            exit_px = exit_rsn = None
            if lo[i] <= pos["stop"]:
                exit_px, exit_rsn = pos["stop"], "initial_stop"
            elif pos["chan_active"] and lo[i] <= pos["chan_stop"]:
                exit_px, exit_rsn = pos["chan_stop"], "chandelier_stop"
            elif not pos["chan_active"] and cl[i] < don_lower[i]:
                exit_px, exit_rsn = cl[i], "midline_exit"

            if exit_px is not None:
                r1 = (exit_px - pos["entry1"]) / pos["risk"]
                if pos["addon_done"]:
                    r2 = (exit_px - pos["addon_price"]) / pos["risk"]
                    net_r = PARTIAL_SIZE * r1 + PARTIAL_SIZE * r2
                else:
                    net_r = PARTIAL_SIZE * r1
                trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
                    exit_time=ts[i],
                    entry_price=(pos["entry1"]+pos["addon_price"])*0.5 if pos["addon_done"] else pos["entry1"],
                    exit_price=exit_px,
                    initial_stop=pos["stop"], initial_risk=pos["risk"],
                    exit_reason=exit_rsn, bars_held=pos["bars"],
                    net_r=net_r, mae_r=min(pos["mae_r"], r1), mfe_r=pos["mfe_r"],
                    addon_triggered=pos["addon_done"]))
                pos = None
                continue
            if pos["mfe_r"] >= CHAN_ACT_R: pos["chan_active"] = True
            if pos["chan_active"]:
                pos["chan_stop"] = max(pos["chan_stop"], pos["hh"] - atr14[i]*CHAN_TRAIL)

        # ── CHECK FOR NEW SIGNAL ─────────────────────────────────────────────
        if pos is None and cl[i] > ema200[i-1] and cl[i] > don_upper[i]:
            risk = atr14[i-1] * STOP_MULT
            stop_px = cl[i] - risk
            pos = dict(
                entry1=cl[i], stop=stop_px, risk=risk, entry_time=ts[i],
                hh=hi[i], chan_active=False, chan_stop=stop_px,
                mfe_r=0.0, mae_r=0.0, bars=1,
                addon_target=cl[i] + ADDON_R_LEVEL * risk,
                addon_done=False, addon_price=cl[i],  # default price (overwritten if triggered)
                addon_bars_left=ADDON_WINDOW,
            )

    # Force-close
    if pos is not None:
        exit_px = cl[-1]
        r1 = (exit_px-pos["entry1"])/pos["risk"]
        if pos["addon_done"]:
            r2 = (exit_px-pos["addon_price"])/pos["risk"]
            net_r = PARTIAL_SIZE*r1 + PARTIAL_SIZE*r2
        else:
            net_r = PARTIAL_SIZE*r1
        trades.append(dict(symbol=symbol, entry_time=pos["entry_time"],
            exit_time=ts[-1],
            entry_price=(pos["entry1"]+pos["addon_price"])*0.5 if pos["addon_done"] else pos["entry1"],
            exit_price=exit_px,
            initial_stop=pos["stop"], initial_risk=pos["risk"],
            exit_reason="end_of_data", bars_held=pos["bars"],
            net_r=net_r, mae_r=min(pos["mae_r"], r1), mfe_r=pos["mfe_r"],
            addon_triggered=pos["addon_done"]))
    return trades


# =============================================================================
# UNIVERSE RUNNER
# =============================================================================

def run_universe(backtest_fn, entry_n: int, symbols: List[str]) -> pd.DataFrame:
    all_trades = []
    for sym in symbols:
        df = load_ohlcv(sym)
        if df is None or len(df) < 250: continue
        all_trades.extend(backtest_fn(df, sym, entry_n))
    if not all_trades: return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df[EXIT_T]  = pd.to_datetime(df[EXIT_T],  utc=True, errors="coerce", format="mixed")
    df[ENTRY_T] = pd.to_datetime(df[ENTRY_T], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[EXIT_T, ENTRY_T, R_COL]).sort_values(EXIT_T).reset_index(drop=True)
    df["month"] = df[EXIT_T].dt.tz_convert(None).dt.to_period("M").astype(str)
    return df


# =============================================================================
# STATS / T4 / T5 / T6
# =============================================================================

def _pf(r):
    g=r[r>0].sum(); l=-r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.0)

def _mdd(r):
    if r.size==0: return 0.0
    eq=np.cumsum(r); pk=np.maximum.accumulate(eq); return float((eq-pk).min())

def summarize(r: np.ndarray) -> dict:
    if r.size==0:
        return dict(trades=0,total_r=0.,avg_r=0.,win_rate=0.,profit_factor=0.,
                    max_dd_r=0.,std_r=0.,t_score=0.)
    avg=float(r.mean()); std=float(r.std(ddof=1)) if r.size>1 else 0.
    t=avg/(std/math.sqrt(r.size)) if std>0 else 0.
    return dict(trades=int(r.size),total_r=float(r.sum()),avg_r=avg,
                win_rate=float((r>0).mean()),profit_factor=_pf(r),
                max_dd_r=_mdd(r),std_r=std,t_score=float(t))

def _bstrap(vals, bs, rng):
    n=len(vals); out=[]
    while len(out)<n:
        s=int(rng.integers(0,n)); e=min(s+bs,n); blk=vals[s:e]
        if len(blk)<bs: blk=np.concatenate([blk,vals[:bs-len(blk)]])
        out.extend(blk.tolist())
    return np.asarray(out[:n],dtype=float)

def t4_all(df: pd.DataFrame) -> dict:
    vals=df[R_COL].to_numpy(dtype=float)
    rng=np.random.default_rng(MC_SEED)
    # MC
    mc_rows=[]
    for bs in MC_BLOCKS:
        tots,dds,pfs=[],[],[]
        for _ in range(MC_RUNS):
            s=_bstrap(vals,bs,rng); tots.append(s.sum()); dds.append(_mdd(s)); pfs.append(_pf(s))
        tots=np.array(tots); dds=np.array(dds); pfs=np.array(pfs)
        mc_rows.append(dict(block_size=bs,
            total_r_p05=float(np.percentile(tots,5)),total_r_p50=float(np.percentile(tots,50)),
            total_r_p95=float(np.percentile(tots,95)),dd_p95=float(np.percentile(dds,95)),
            pf_p05=float(np.percentile(pfs,5)),pf_p50=float(np.percentile(pfs,50)),
            prob_positive=float((tots>0).mean())))
    mc=pd.DataFrame(mc_rows)
    # cost stress
    cost_rows=[]
    for ec in EXTRA_COSTS:
        s=summarize(vals-ec); s["extra_cost"]=ec; cost_rows.append(s)
    cost=pd.DataFrame(cost_rows)
    # period splits
    df2=df.sort_values(EXIT_T).reset_index(drop=True)
    mid=len(df2)//2; med_t=df2[EXIT_T].median()
    split_rows=[]
    for name,sub in [("first_half_by_trade",df2.iloc[:mid]),
                     ("second_half_by_trade",df2.iloc[mid:]),
                     ("last_100_trades",df2.tail(100)),
                     ("first_half_by_time",df2[df2[EXIT_T]<=med_t]),
                     ("second_half_by_time",df2[df2[EXIT_T]>med_t])]:
        s=summarize(sub[R_COL].to_numpy(dtype=float)); s["split"]=name; split_rows.append(s)
    splits=pd.DataFrame(split_rows)
    # remove best assets
    asset_r=df.groupby("symbol")[R_COL].sum().sort_values(ascending=False)
    ra_rows=[]
    for n in [0,1,3,5]:
        removed=asset_r.head(n).index.tolist(); sub=df[~df["symbol"].isin(removed)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(removed),assets_remaining=sub["symbol"].nunique())
        ra_rows.append(s)
    rem_assets=pd.DataFrame(ra_rows)
    # remove best months
    month_r=df.groupby("month")[R_COL].sum().sort_values(ascending=False)
    rm_rows=[]
    for n in [0,1,2,3]:
        removed=month_r.head(n).index.tolist(); sub=df[~df["month"].isin(removed)]
        s=summarize(sub[R_COL].to_numpy(dtype=float))
        s.update(removed_n=n,removed=",".join(removed),months_remaining=sub["month"].nunique())
        rm_rows.append(s)
    rem_months=pd.DataFrame(rm_rows)
    return dict(mc=mc, cost=cost, splits=splits, rem_assets=rem_assets, rem_months=rem_months)

def t5_replay(df: pd.DataFrame, max_open: int) -> Tuple[pd.DataFrame, dict]:
    df=df.sort_values(ENTRY_T).reset_index(drop=True); open_pos=[]; closed=[]
    def _flush(now):
        nonlocal open_pos
        still=[p for p in open_pos if p["exit_time"]>now]
        closed.extend(p for p in open_pos if p["exit_time"]<=now); open_pos[:]=still
    for _,row in df.iterrows():
        _flush(row[ENTRY_T])
        if any(p["symbol"]==row["symbol"] for p in open_pos): continue
        if len(open_pos)>=max_open: continue
        open_pos.append({"symbol":row["symbol"],"entry_time":row[ENTRY_T],
                          "exit_time":row[EXIT_T],R_COL:row[R_COL]})
    if open_pos:
        last=max(p["exit_time"] for p in open_pos); _flush(last+pd.Timedelta(seconds=1))
    acc=pd.DataFrame(closed)
    if acc.empty:
        return acc,dict(accepted=0,total_r=0.,avg_r=0.,profit_factor=0.,max_dd_r=0.,win_rate=0.)
    acc=acc.sort_values("exit_time").reset_index(drop=True); r=acc[R_COL].to_numpy(dtype=float)
    return acc,dict(accepted=int(len(r)),total_r=float(r.sum()),avg_r=float(r.mean()),
                    profit_factor=_pf(r),max_dd_r=_mdd(r),win_rate=float((r>0).mean()))

def t6_equity(acc: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if acc.empty: return pd.DataFrame(),{}
    equity=START_CAP; peak=START_CAP; kill=False; rows=[]
    for _,row in acc.sort_values("exit_time").iterrows():
        pnl=row[R_COL]*equity*RISK_PCT; equity+=pnl; peak=max(peak,equity)
        dd_pct=(equity-peak)/peak
        if dd_pct<=KILL_SWITCH_DD and not kill: kill=True
        rows.append(dict(exit_time=row["exit_time"],symbol=row["symbol"],net_r=row[R_COL],
                         equity=equity,peak=peak,dd_pct=float(dd_pct),kill_fired=kill))
    eq_df=pd.DataFrame(rows); final=float(eq_df["equity"].iloc[-1])
    start=acc[ENTRY_T].min(); end=acc["exit_time"].max()
    years=float((end-start).days/365.25) if pd.notnull(start) and pd.notnull(end) else 1.
    cagr=float((final/START_CAP)**(1/years)-1) if years>0 else 0.
    return eq_df,dict(end_capital=final,total_return_pct=float((final-START_CAP)/START_CAP*100),
                      cagr_pct=float(cagr*100),max_dd_pct=float(eq_df["dd_pct"].min()*100),
                      years=round(years,2),kill_switch_fired=kill)


# =============================================================================
# MASTER REPORT WRITER
# =============================================================================

def write_report(out_dir: Path, variant: str, desc: str, extra_note: str,
                 stab: pd.DataFrame, baseline: dict, t4: dict,
                 t5: dict, t6: dict, trade_guard: bool) -> None:
    def gate(cond, label): return f"  {'PASS' if cond else 'FAIL'}  {label}"

    mc10=t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10=t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec=t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1=t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1=t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]

    all_stab=all(r["avg_r"]>0 and r["total_r"]>0 for _,r in stab.iterrows())
    avg_r_ok=baseline["avg_r"]>0.15; pf_ok=baseline["profit_factor"]>1.
    mc_ok=float(mc10["total_r_p05"])>0; cost_ok=float(cost10["profit_factor"])>1.
    sec_ok=float(sec["avg_r"])>0; ra_ok=float(ra1["total_r"])>0
    rm_ok=float(rm1["total_r"])>0; kill_ok=not t6.get("kill_switch_fired",False)
    win_ok=0.30<=baseline["win_rate"]<=0.45
    all_gates=all([avg_r_ok,pf_ok,mc_ok,cost_ok,sec_ok,ra_ok,rm_ok,
                   kill_ok,win_ok,trade_guard,all_stab])

    lines=[
        f"PHASE T17 OPT-17.3 -- Entry Refinement Variant {variant}",
        "="*70,"",f"Variant  : {desc}",
        "Universe : 24 symbols / 1D / N=20 / ATR×2.0 / Chandelier ACT4","",
        f"Baseline : avg_r=+1.101R  PF=3.072  CAGR=+14.3%  trades=461",
    ]
    if extra_note: lines += ["", f"Note: {extra_note}"]
    lines += [
        "","="*70,"STABILITY ZONE","="*70,
        f"  {'N':>4s}  {'exit_n':>6s}  {'trades':>7s}  {'total_r':>8s}  {'avg_r':>8s}  {'PF':>5s}  PASS?",
        "  "+"-"*58,
    ]
    for _,r in stab.iterrows():
        ok=r["avg_r"]>0 and r["total_r"]>0
        lines.append(f"  {int(r['entry_n']):>4d}  {int(r['exit_n']):>6d}  {int(r['trades']):>7d}  "
                     f"{r['total_r']:>+8.2f}  {r['avg_r']:>+8.4f}  {r['profit_factor']:>5.3f}  "
                     f"{'YES' if ok else 'NO '}")
    lines+=[""  ,f"  Stability: {'ALL PASS' if all_stab else 'FAIL'}",
            "","="*70,"T4 -- BASELINE (N=20)","="*70,
        f"  Trades   : {baseline['trades']}  [guard >={MIN_TRADES}: {'PASS' if trade_guard else 'FAIL'}]",
        f"  Total R  : {baseline['total_r']:+.2f}R",
        f"  Avg R    : {baseline['avg_r']:+.4f}R  (ref: +1.101R)",
        f"  PF       : {baseline['profit_factor']:.3f}  (ref: 3.072)",
        f"  Win rate : {baseline['win_rate']:.1%}",
        f"  Max DD   : {baseline['max_dd_r']:+.2f}R",
        f"  t-score  : {baseline['t_score']:.2f}",
        f"  Period   : {baseline['start']} -> {baseline['end']}",
        "","="*70,"T4 -- MONTE CARLO (2000 runs, block=10)","="*70,
        f"  totalR p05/p50/p95 = {mc10['total_r_p05']:.1f} / {mc10['total_r_p50']:.1f} / {mc10['total_r_p95']:.1f}",
        f"  prob(>0)           = {mc10['prob_positive']:.1%}",
        f"  PF p05/p50         = {mc10['pf_p05']:.2f} / {mc10['pf_p50']:.2f}",
        f"  DD p95             = {mc10['dd_p95']:.2f}R",
        "","="*70,"T4 -- COST STRESS","="*70,
    ]
    for _,r in t4["cost"].iterrows():
        lines.append(f"  +{r['extra_cost']:.2f}R -> totalR={r['total_r']:+.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- PERIOD SPLITS","="*70]
    for _,r in t4["splits"].iterrows():
        lines.append(f"  {r['split']:30s}: t={int(r['trades']):3d}  totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  PF={r['profit_factor']:.3f}")
    lines+=["","="*70,"T4 -- REMOVE BEST ASSETS","="*70]
    for _,r in t4["rem_assets"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  [{r['removed']}]")
    lines+=["","="*70,"T4 -- REMOVE BEST MONTHS","="*70]
    for _,r in t4["rem_months"].iterrows():
        lines.append(f"  remove {int(r['removed_n'])}: totalR={r['total_r']:>+7.1f}  avgR={r['avg_r']:+.4f}  [{r['removed']}]")
    lines+=["","="*70,f"T5/T6 -- max{PORT_MAX_OPEN} PORTFOLIO","="*70,
            f"  Accepted : {t5.get('accepted',0)}",
            f"  avg_r    : {t5.get('avg_r',0):+.4f}R  PF: {t5.get('profit_factor',0):.3f}"]
    if t6:
        lines+=[f"  CAGR     : {t6.get('cagr_pct',0):+.1f}%  (ref: +14.3%)",
                f"  Max DD   : {t6.get('max_dd_pct',0):+.1f}%  (ref: -3.4%)",
                f"  Kill sw  : {'YES' if t6.get('kill_switch_fired') else 'no'}",
                f"  Duration : {t6.get('years',0):.1f} yr"]
    lines+=["","="*70,"FINAL SCORECARD","="*70,"",
        gate(trade_guard,f"Trade count >={MIN_TRADES} (got {baseline['trades']})"),
        gate(all_stab,"Stability zone all profitable"),
        gate(win_ok,f"§4.1 win rate 30-45% (got {baseline['win_rate']:.1%})"),
        gate(avg_r_ok,f"§4.2 avg_r > 0.15R (got {baseline['avg_r']:+.4f}R)"),
        gate(pf_ok,f"PF > 1.0 (got {baseline['profit_factor']:.3f})"),
        gate(mc_ok,f"MC p05 > 0 (got {float(mc10['total_r_p05']):+.1f}R)"),
        gate(cost_ok,f"Cost +0.10R PF > 1.0 (got {float(cost10['profit_factor']):.3f})"),
        gate(sec_ok,f"2nd-half avg_r > 0 (got {float(sec['avg_r']):+.4f}R)"),
        gate(ra_ok,f"Remove top-1 asset > 0 (got {float(ra1['total_r']):+.1f}R)"),
        gate(rm_ok,f"Remove top-1 month > 0 (got {float(rm1['total_r']):+.1f}R)"),
        gate(kill_ok,"Kill switch not fired"),"",
        "  vs BASELINE:",
        f"  avg_r : {baseline['avg_r']:+.4f}R vs +1.101R -> {'BETTER' if baseline['avg_r']>1.1011 else 'WORSE'}",
        f"  PF    : {baseline['profit_factor']:.3f} vs 3.072 -> {'BETTER' if baseline['profit_factor']>3.072 else 'WORSE'}",
        f"  CAGR  : {t6.get('cagr_pct',0):+.1f}% vs +14.3% -> {'BETTER' if t6.get('cagr_pct',0)>14.3 else 'WORSE'}",
        f"  trades: {baseline['trades']} vs 461 (delta: {baseline['trades']-461:+d})",
        "",f"  OVERALL: {'PASS' if all_gates else 'FAIL'} -- {'all gates cleared' if all_gates else 'one or more gates failed'}",
    ]
    rpt = out_dir / f"variant_{variant.lower()}_master_report.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# RUN ONE VARIANT END-TO-END
# =============================================================================

def run_variant(label: str, desc: str, extra_note: str,
                backtest_fn, symbols: List[str]) -> dict:
    """Run stability zone + T4 + T5 + T6 for one variant. Return key metrics dict."""
    out_dir = ROOT / "data" / f"research_donchian_entryV2_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stability zone
    stab_rows = []
    for n in STABILITY_N:
        df_n = run_universe(backtest_fn, n, symbols)
        r = df_n[R_COL].to_numpy(dtype=float) if not df_n.empty else np.array([])
        s = summarize(r)
        stab_rows.append(dict(entry_n=n, exit_n=n//2, trades=s["trades"],
                              total_r=s["total_r"], avg_r=s["avg_r"],
                              profit_factor=s["profit_factor"]))
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(out_dir / f"variant_{label.lower()}_stability.csv", index=False)

    # Canonical N=20
    trades = run_universe(backtest_fn, CANONICAL_N, symbols)
    if trades.empty:
        return dict(label=label, desc=desc, trades=0, avg_r=0., pf=0., cagr=0.,
                    mc_p05=0., t_score=0., win_rate=0., stab_pass=False,
                    trade_guard=False, all_gates=False, note="NO TRADES")
    trades.to_csv(out_dir / f"variant_{label.lower()}_trades_N20.csv", index=False)

    # Baseline stats
    r = trades[R_COL].to_numpy(dtype=float)
    base = summarize(r)
    base.update(assets=trades["symbol"].nunique(),
                start=str(trades[EXIT_T].min().date()),
                end=str(trades[EXIT_T].max().date()))

    trade_guard = base["trades"] >= MIN_TRADES

    # T4
    t4 = t4_all(trades)
    for key, df_obj in t4.items():
        df_obj.to_csv(out_dir / f"variant_{label.lower()}_t4_{key}.csv", index=False)

    # T5 / T6
    acc, t5 = t5_replay(trades, PORT_MAX_OPEN)
    t6 = {}
    if not acc.empty:
        eq_df, t6 = t6_equity(acc)
        eq_df.to_csv(out_dir / f"variant_{label.lower()}_t6_equity.csv", index=False)

    # Gate check
    mc10  = t4["mc"][t4["mc"]["block_size"]==10].iloc[0]
    cost10 = t4["cost"][t4["cost"]["extra_cost"]==0.10].iloc[0]
    sec   = t4["splits"][t4["splits"]["split"]=="second_half_by_trade"].iloc[0]
    ra1   = t4["rem_assets"][t4["rem_assets"]["removed_n"]==1].iloc[0]
    rm1   = t4["rem_months"][t4["rem_months"]["removed_n"]==1].iloc[0]
    all_stab = all(r2["avg_r"]>0 and r2["total_r"]>0 for _,r2 in stab.iterrows())
    all_gates = all([
        base["avg_r"]>0.15, base["profit_factor"]>1.,
        float(mc10["total_r_p05"])>0, float(cost10["profit_factor"])>1.,
        float(sec["avg_r"])>0, float(ra1["total_r"])>0, float(rm1["total_r"])>0,
        not t6.get("kill_switch_fired",False),
        0.30<=base["win_rate"]<=0.45,
        trade_guard, all_stab,
    ])

    # Write report
    write_report(out_dir, label, desc, extra_note,
                 stab, base, t4, t5, t6, trade_guard)

    return dict(
        label=label, desc=desc,
        trades=base["trades"], avg_r=base["avg_r"], pf=base["profit_factor"],
        win_rate=base["win_rate"], t_score=base["t_score"],
        mc_p05=float(mc10["total_r_p05"]),
        cagr=t6.get("cagr_pct", 0.), max_dd=t6.get("max_dd_pct", 0.),
        stab_pass=all_stab, trade_guard=trade_guard, all_gates=all_gates,
        all_stab=all_stab,
    )


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(results: List[dict]) -> None:
    SEP = "=" * 80
    print()
    print(SEP)
    print("OPT-17.3 ENTRY REFINEMENT - COMBINED COMPARISON TABLE")
    print(SEP)
    print()

    header = (f"  {'Variant':<12s}  {'Trades':>6s}  {'dTrades':>8s}  "
              f"{'avg_r':>8s}  {'PF':>5s}  {'CAGR':>6s}  {'DD':>6s}  "
              f"{'MC_p05':>8s}  {'t':>5s}  {'Win%':>5s}  {'Overall':>7s}")
    print(header)
    print("  " + "-"*76)

    baseline_row = (f"  {'Baseline':<12s}  {461:>6d}  {'—':>8s}  "
                    f"{'+1.101R':>8s}  {'3.072':>5s}  {'+14.3%':>6s}  {'-3.4%':>6s}  "
                    f"{'+347.7R':>8s}  {'7.33':>5s}  {'41.2%':>5s}  {'ref':>7s}")
    print(baseline_row)
    print("  " + "-"*76)

    for r in results:
        delta = r["trades"] - 461
        overall = "PASS" if r["all_gates"] else "FAIL"
        print(f"  {r['label']:<12s}  {r['trades']:>6d}  {delta:>+8d}  "
              f"{r['avg_r']:>+8.4f}  {r['pf']:>5.3f}  {r['cagr']:>+6.1f}%  "
              f"{r['max_dd']:>+6.1f}%  {r['mc_p05']:>+8.1f}R  "
              f"{r['t_score']:>5.2f}  {r['win_rate']:>5.1%}  {overall:>7s}")

    print()
    print("  Gate failures:")
    for r in results:
        fails = []
        if not r["stab_pass"]:   fails.append("stability zone")
        if not r["trade_guard"]: fails.append(f"trades<{MIN_TRADES}")
        if r["mc_p05"] <= 0:     fails.append("MC p05≤0")
        if r["avg_r"] <= 0.15:   fails.append("avg_r≤0.15R")
        if r["pf"] <= 1.0:       fails.append("PF≤1.0")
        if not (0.30 <= r["win_rate"] <= 0.45): fails.append("win%")
        fail_str = ", ".join(fails) if fails else "none"
        print(f"  Variant {r['label']}: {fail_str}")

    print()
    print("  Benchmark comparison vs baseline (+1.101R / PF 3.072 / +14.3% CAGR):")
    for r in results:
        r_flag  = "^" if r["avg_r"] > 1.1011 else "v"
        pf_flag = "^" if r["pf"] > 3.072 else "v"
        cg_flag = "^" if r["cagr"] > 14.3 else "v"
        print(f"  Variant {r['label']}: "
              f"avg_r {r_flag}{r['avg_r']:+.4f}R  "
              f"PF {pf_flag}{r['pf']:.3f}  "
              f"CAGR {cg_flag}{r['cagr']:+.1f}%")
    print()
    print(SEP)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    symbols = pd.read_csv(IN_SYMBOLS)["symbol"].tolist()

    results = []

    results.append(run_variant(
        label="A",
        desc="Limit pullback (breakout_level - 0.25×ATR, 2-bar cancel)",
        extra_note=(f"Limit = don_upper - {LIMIT_PULLBACK_ATR_FRAC}×ATR(14). "
                    f"Cancel if not filled within {LIMIT_CANCEL_BARS} bars. "
                    f"Stop = fill_price - ATR(14)×{STOP_MULT}."),
        backtest_fn=backtest_a,
        symbols=symbols,
    ))

    results.append(run_variant(
        label="B",
        desc="Price confirmation (next bar close > breakout level)",
        extra_note=("Signal at close[i] > don_upper[i]. Enter at close[i+1] only if "
                    "close[i+1] > don_upper[i]. One-bar confirmation window, no fill = skip."),
        backtest_fn=backtest_b,
        symbols=symbols,
    ))

    results.append(run_variant(
        label="C",
        desc="Partial + add-on (50% entry, +50% at +1R within 5 bars)",
        extra_note=(f"Initial 50% at breakout close. Add-on 50% if high >= entry+{ADDON_R_LEVEL}R "
                    f"within {ADDON_WINDOW} bars. "
                    f"net_r = 0.5×r1 [+ 0.5×r2 if add-on]. "
                    f"No add-on → only 50% exposure (net_r = 0.5×r1)."),
        backtest_fn=backtest_c,
        symbols=symbols,
    ))

    print_comparison_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
