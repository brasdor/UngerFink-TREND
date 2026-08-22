#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9B COMBINED SUMMARY -- ALL 6 SYSTEMS
======================================

Reads state.json + output CSVs for all six T9B paper engines and prints a
compact, phone-readable combined summary.

Systems:
  S1  DonchianLong_UniverseV2_ExitV2        (data/t9b_paper/)
  S2  MeanReversionRSI 1D                   (data/t9b_mr_paper/)
  S3  ConsecDownDaysMR 1D                   (data/t9b_consecdowndays_paper/)
  S6  MomentumFactor Lb20 L20/S10           (data/t9b_momentum_paper/)
  S7  VolContractionShort 4H                (data/t9b_volcontraction_paper/)
  S8  MACrossShort 4H                       (data/t9b_macross_paper/)

Total deployed capital: $60,000 (Scheme C regime-tilted allocation)

ASCII only, flush=True on all print statements.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent  # engines/ -> project root

# Scheme C regime-tilted allocation (confirmed 2026-06-17)
TARGET_ALLOC = {
    "S1":  8_000.0,   # Spot: Donchian Long
    "S2": 12_000.0,   # Spot: RSI MR Long
    "S3": 10_000.0,   # Spot: ConsecDownDays MR
    "S6":  8_000.0,   # Futures: Momentum Factor
    "S7": 11_000.0,   # Futures: VolContraction Short
    "S8": 11_000.0,   # Futures: MA Cross Short
}
TOTAL_CAPITAL = 60_000.0
SPOT_TARGET   = 30_000.0
FUT_TARGET    = 30_000.0

# Spot pool: S1+S2+S3  ($30k)  Futures pool: S6+S7+S8  ($30k)
SYSTEMS = [
    {
        "id":       "S1",
        "name":     "Donchian",
        "label":    "DonchianLong_UniverseV2_ExitV2",
        "data_dir": ROOT / "data" / "t9b_paper",
        # engine's own trade log -- "equity_curve.csv" in this dir is
        # mark_to_market.py's daily snapshot, a different schema.
        "equity_file": "engine_equity_curve.csv",
        "freeze":   date(2026, 5, 30),
        "type":     "spot_long",
        "pool":     "Spot",
    },
    {
        "id":       "S2",
        "name":     "RSI_MR",
        "label":    "MeanReversionRSI 1D",
        "data_dir": ROOT / "data" / "t9b_mr_paper",
        "freeze":   date(2026, 6, 1),
        "type":     "spot_long",
        "pool":     "Spot",
    },
    {
        "id":       "S3",
        "name":     "ConsecDown",
        "label":    "ConsecDownDaysMR 1D",
        "data_dir": ROOT / "data" / "t9b_consecdowndays_paper",
        "equity_file": "engine_equity_curve.csv",
        "freeze":   date(2026, 6, 2),
        "type":     "spot_long",
        "pool":     "Spot",
    },
    {
        "id":       "S6",
        "name":     "Momentum",
        "label":    "MomentumFactor Lb20 L20/S10 Biweekly",
        "data_dir": ROOT / "data" / "t9b_momentum_paper",
        "freeze":   date(2026, 6, 7),
        "type":     "momentum",
        "pool":     "Futures",
    },
    {
        "id":       "S7",
        "name":     "VolCont",
        "label":    "VolContractionShort 4H",
        "data_dir": ROOT / "data" / "t9b_volcontraction_paper",
        "freeze":   date(2026, 6, 10),
        "type":     "futures_short",
        "pool":     "Futures",
    },
    {
        "id":       "S8",
        "name":     "MACross",
        "label":    "MACrossShort 4H",
        "data_dir": ROOT / "data" / "t9b_macross_paper",
        "freeze":   date(2026, 6, 12),
        "type":     "futures_short",
        "pool":     "Futures",
    },
]

_BTC_PATHS = [
    ROOT / "data" / "universe"         / "ohlcv_1d" / "BTC_USDT_1d.csv",
    ROOT / "data" / "futures_universe" / "ohlcv_1d" / "BTCUSDT_1d.csv",
]


# ── helpers ────────────────────────────────────────────────────────────────

def p(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)


def load_state(data_dir: Path) -> dict:
    path = data_dir / "state.json"
    if not data_dir.exists():
        return {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def detect_regime(today: date) -> str:
    """BULL / BEAR based on BTC close vs EMA200. MIXED if data unavailable."""
    for path in _BTC_PATHS:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            if "close" not in df.columns:
                continue
            closes = df["close"].sort_index().dropna()
            if len(closes) < 50:
                continue
            ema200 = closes.ewm(span=200, min_periods=50, adjust=False).mean()
            cutoff = closes.index[closes.index <= pd.Timestamp(today)]
            if len(cutoff) == 0:
                continue
            last  = cutoff[-1]
            price = float(closes.loc[last])
            ema   = float(ema200.loc[last])
            return "BULL" if price > ema else "BEAR"
        except Exception:
            continue
    return "MIXED"


def system_snapshot(cfg: dict, today: date) -> dict:
    """Load state + CSVs for one engine and build a snapshot dict."""
    state   = load_state(cfg["data_dir"])
    eq_df   = load_csv(cfg["data_dir"] / cfg.get("equity_file", "equity_curve.csv"))
    sig_df  = load_csv(cfg["data_dir"] / "signals_today.csv")
    open_df = load_csv(cfg["data_dir"] / "open_positions.csv")

    sys_id     = cfg["id"]
    target_eq  = TARGET_ALLOC.get(sys_id, 10_000.0)
    not_started = not cfg["data_dir"].exists() or not state

    if not_started:
        return {
            **cfg,
            "not_started":  True,
            "equity":       target_eq,
            "target_eq":    target_eq,
            "peak":         target_eq,
            "ret_pct":      0.0,
            "dd_pct":       0.0,
            "drift_pct":    0.0,
            "kill_sw":      False,
            "n_open_long":  0,
            "n_open_short": 0,
            "n_open":       0,
            "n_closed":     0,
            "last_run":     "NOT STARTED",
            "days_running": 0,
            "review_date":  str(cfg["freeze"] + timedelta(days=90)),
            "days_to_rev":  (cfg["freeze"] + timedelta(days=90) - today).days,
            "signals":      pd.DataFrame(),
            "n_signals":    0,
            "open_pos":     pd.DataFrame(),
            "avg_r":        0.0,
            "total_r":      0.0,
            "win_rate":     0.0,
            "n_trades":     0,
            "n_rebal":      0,
            "last_rebal":   "n/a",
            "total_unreal": 0.0,
        }

    eq      = float(state.get("paper_equity_usdt", target_eq))
    peak    = float(state.get("peak_equity_usdt", target_eq))
    dd      = float(state.get("drawdown_pct", 0.0))
    ret_pct = (eq / target_eq - 1.0) * 100.0
    kill_sw = state.get("kill_switch_triggered", False)
    last_run = state.get("last_run_date", "n/a")

    t = cfg["type"]

    if t == "momentum":
        n_long  = len(state.get("long_positions", []))
        n_short = len(state.get("short_positions", []))
    elif t == "futures_short":
        n_long  = 0
        n_short = len(state.get("open_positions", []))
    else:
        n_long  = len(state.get("open_positions", []))
        n_short = 0

    n_open   = n_long + n_short
    n_closed = int(state.get("closed_trade_count", 0))

    freeze       = cfg["freeze"]
    days_running = max((today - freeze).days, 0)
    review_date  = freeze + timedelta(days=90)
    days_to_rev  = (review_date - today).days

    # Trade stats for non-momentum systems
    avg_r = total_r = win_rate = 0.0
    n_trades = 0
    if t not in ("momentum",) and not eq_df.empty and "gross_r" in eq_df.columns:
        r_vals   = pd.to_numeric(eq_df["gross_r"], errors="coerce").dropna()
        n_trades = len(r_vals)
        avg_r    = float(r_vals.mean()) if n_trades > 0 else 0.0
        total_r  = float(r_vals.sum())
        win_rate = float((r_vals > 0).mean() * 100) if n_trades > 0 else 0.0

    n_rebal      = int(state.get("total_rebal_count", 0)) if t == "momentum" else 0
    last_rebal   = state.get("last_rebal_date", "n/a") if t == "momentum" else None
    total_unreal = 0.0
    if t == "momentum" and not open_df.empty and "unrealized_pnl" in open_df.columns:
        total_unreal = float(pd.to_numeric(open_df["unrealized_pnl"], errors="coerce").sum())

    # Allocation drift: how far current equity is from target %
    total_eq_est = sum(
        float(load_state(c["data_dir"]).get("paper_equity_usdt",
              TARGET_ALLOC.get(c["id"], 10_000.0)))
        for c in SYSTEMS
    )
    current_pct = (eq / total_eq_est * 100) if total_eq_est > 0 else 0
    target_pct  = (target_eq / TOTAL_CAPITAL * 100)
    drift_pct   = current_pct - target_pct

    return {
        **cfg,
        "not_started":  False,
        "equity":       eq,
        "target_eq":    target_eq,
        "peak":         peak,
        "ret_pct":      ret_pct,
        "dd_pct":       dd,
        "drift_pct":    drift_pct,
        "kill_sw":      kill_sw,
        "n_open_long":  n_long,
        "n_open_short": n_short,
        "n_open":       n_open,
        "n_closed":     n_closed,
        "last_run":     last_run,
        "days_running": days_running,
        "review_date":  str(review_date),
        "days_to_rev":  days_to_rev,
        "signals":      sig_df,
        "n_signals":    len(sig_df) if not sig_df.empty else 0,
        "open_pos":     open_df,
        "avg_r":        avg_r,
        "total_r":      total_r,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
        "n_rebal":      n_rebal,
        "last_rebal":   last_rebal,
        "total_unreal": total_unreal,
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    today  = date.today()
    regime = detect_regime(today)

    p("=" * 75)
    p("T9B COMBINED SUMMARY -- ALL 6 SYSTEMS")
    p(f"Date: {today}  |  Regime: {regime}  |  Scheme C: $60k regime-tilted")
    p("=" * 75)

    snaps = [system_snapshot(cfg, today) for cfg in SYSTEMS]

    # ── Section 1: Equity overview + allocation drift ───────────────────
    p()
    p("EQUITY OVERVIEW (Scheme C allocation)")
    p(f"  {'ID':<3} {'System':<11} {'Pool':<7} {'Target':>8} {'Equity':>10}  "
      f"{'Return':>8}  {'DD':>7}  {'Drift':>6}  {'L':>2} {'S':>2}  {'Days':>4}")
    p(f"  {'-'*3} {'-'*11} {'-'*7} {'-'*8} {'-'*10}  "
      f"{'-'*8}  {'-'*7}  {'-'*6}  {'-'*2} {'-'*2}  {'-'*4}")

    spot_eq = fut_eq = 0.0

    for s in snaps:
        if s["not_started"]:
            tgt = s["target_eq"]
            p(f"  {s['id']:<3} {s['name']:<11} {s['pool']:<7} ${tgt:>6,.0f}  NOT STARTED")
            continue
        tgt = s["target_eq"]
        p(f"  {s['id']:<3} {s['name']:<11} {s['pool']:<7} ${tgt:>6,.0f} ${s['equity']:>9,.2f}  "
          f"{s['ret_pct']:>+7.2f}%  {s['dd_pct']:>+6.2f}%  "
          f"{s['drift_pct']:>+5.1f}%  "
          f"{s['n_open_long']:>2} {s['n_open_short']:>2}  "
          f"{s['days_running']:>4}")
        if s["pool"] == "Spot":
            spot_eq += s["equity"]
        else:
            fut_eq += s["equity"]

    total_eq  = spot_eq + fut_eq
    total_ret = (total_eq / TOTAL_CAPITAL - 1.0) * 100.0 if TOTAL_CAPITAL > 0 else 0.0
    spot_ret  = (spot_eq / SPOT_TARGET - 1.0) * 100.0 if SPOT_TARGET > 0 else 0.0
    fut_ret   = (fut_eq  / FUT_TARGET  - 1.0) * 100.0 if FUT_TARGET  > 0 else 0.0

    p(f"  {'':<3} {'Spot pool':<11} {'':>7} ${'30k':>6} ${spot_eq:>9,.2f}  "
      f"{spot_ret:>+7.2f}%")
    p(f"  {'':<3} {'Futures pool':<11} {'':>7} ${'30k':>6} ${fut_eq:>9,.2f}  "
      f"{fut_ret:>+7.2f}%")
    p(f"  {'':<3} {'PORTFOLIO':<11} {'':>7} ${'60k':>6} ${total_eq:>9,.2f}  "
      f"{total_ret:>+7.2f}%")

    # Long / short position summary
    total_longs  = sum(s["n_open_long"]  for s in snaps)
    total_shorts = sum(s["n_open_short"] for s in snaps)
    p(f"  Open positions: {total_longs} LONG  /  {total_shorts} SHORT  "
      f"(total {total_longs + total_shorts})")

    # ── Section 2: Trade performance ────────────────────────────────────
    p()
    p("TRADE PERFORMANCE (closed trades only)")
    p(f"  {'ID':<3} {'System':<11}  {'Trades':>7}  {'WinRate':>8}  "
      f"{'AvgR':>8}  {'TotalR':>8}")
    p(f"  {'-'*3} {'-'*11}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    for s in snaps:
        if s["not_started"]:
            p(f"  {s['id']:<3} {s['name']:<11}   NOT STARTED")
            continue
        if s["type"] == "momentum":
            unreal_str = (f"${s['total_unreal']:>+,.2f}" if s["total_unreal"] != 0.0
                          else "$0.00")
            p(f"  {s['id']:<3} {s['name']:<11}   "
              f"rebals={s['n_rebal']}  last={s['last_rebal']}  "
              f"unrealized={unreal_str}")
        elif s["n_trades"] == 0:
            p(f"  {s['id']:<3} {s['name']:<11}   (no closed trades yet)")
        else:
            p(f"  {s['id']:<3} {s['name']:<11}  {s['n_trades']:>7}  "
              f"{s['win_rate']:>7.1f}%  {s['avg_r']:>+7.3f}R  "
              f"{s['total_r']:>+7.2f}R")

    # ── Section 3: Signals today ────────────────────────────────────────
    total_signals = sum(s["n_signals"] for s in snaps)
    p()
    p(f"SIGNALS TODAY ({total_signals} total across all systems)")
    any_signal = False
    for s in snaps:
        sig_df = s["signals"]
        if sig_df.empty or s["n_signals"] == 0:
            continue
        any_signal = True
        p(f"\n  [{s['id']} {s['name']}] {s['n_signals']} signal(s):")
        if s["type"] == "momentum":
            longs  = sig_df[sig_df["side"] == "LONG"]  if "side" in sig_df.columns else sig_df
            shorts = sig_df[sig_df["side"] == "SHORT"] if "side" in sig_df.columns else pd.DataFrame()
            if not longs.empty:
                p(f"    LONGS ({len(longs)}):")
                for _, row in longs.iterrows():
                    sym = str(row.get("symbol", "?"))
                    px  = row.get("entry_price", 0)
                    rk  = int(row.get("rank", 0))
                    ret = row.get("period_return_pct", float("nan"))
                    rs  = f"{ret:>+6.1f}%" if ret == ret else "   n/a"
                    p(f"      #{rk:<3} {sym:<18}  px={px:<12.6g}  20d_ret={rs}")
            if not shorts.empty:
                p(f"    SHORTS ({len(shorts)}):")
                for _, row in shorts.iterrows():
                    sym = str(row.get("symbol", "?"))
                    px  = row.get("entry_price", 0)
                    rk  = int(row.get("rank", 0))
                    ret = row.get("period_return_pct", float("nan"))
                    rs  = f"{ret:>+6.1f}%" if ret == ret else "   n/a"
                    p(f"      #{rk:<3} {sym:<18}  px={px:<12.6g}  20d_ret={rs}")
        elif s["type"] == "futures_short":
            for _, row in sig_df.iterrows():
                sym     = str(row.get("symbol", "?"))
                close   = row.get("signal_close", row.get("close", "?"))
                funding = row.get("funding_rate", "?")
                stop    = row.get("stop_loss", "?")
                try:
                    funding_str = f"{float(funding)*100:.4f}%"
                except Exception:
                    funding_str = str(funding)
                p(f"    SHORT  {sym:<18}  close={close:<12.6g}  "
                  f"funding={funding_str}  stop={stop:.6g}" if isinstance(stop, float)
                  else f"    SHORT  {sym:<18}  close={close}")
        else:
            for _, row in sig_df.iterrows():
                sym  = str(row.get("symbol", "?"))
                cls  = row.get("close", "?")
                stop = row.get("stop_loss", "?")
                extra = ""
                if "rsi" in row:
                    extra = f"  RSI={row['rsi']:.1f}"
                elif "consec" in row:
                    extra = f"  consec={int(row['consec'])}"
                p(f"    LONG  {sym:<18}  close={cls:<12.6g}  stop={stop:<12.6g}{extra}")
    if not any_signal:
        p("  None today")

    # ── Section 4: Open positions (compact) ─────────────────────────────
    total_open = sum(s["n_open"] for s in snaps)
    p()
    p(f"OPEN POSITIONS ({total_open} total  |  "
      f"{total_longs} LONG  /  {total_shorts} SHORT)")

    for s in snaps:
        open_df = s["open_pos"]
        if s["not_started"] or s["n_open"] == 0:
            continue
        p(f"\n  [{s['id']} {s['name']}] {s['n_open']} position(s):")
        if s["type"] == "momentum":
            longs  = open_df[open_df["side"] == "LONG"]  if "side" in open_df.columns else open_df
            shorts = open_df[open_df["side"] == "SHORT"] if "side" in open_df.columns else pd.DataFrame()
            if not longs.empty:
                p(f"    LONGS ({len(longs)}):")
                for _, row in longs.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    notl = float(row.get("notional_usdt", 0))
                    upnl = float(row.get("unrealized_pnl", 0))
                    upct = float(row.get("unrealized_pct", 0))
                    held = int(row.get("days_held", 0))
                    p(f"      {sym:<18}  notional=${notl:>8,.2f}  "
                      f"pnl={upnl:>+8.2f}  ({upct:>+5.2f}%)  d={held}")
            if not shorts.empty:
                p(f"    SHORTS ({len(shorts)}):")
                for _, row in shorts.iterrows():
                    sym  = str(row.get("symbol", "?"))
                    notl = float(row.get("notional_usdt", 0))
                    upnl = float(row.get("unrealized_pnl", 0))
                    upct = float(row.get("unrealized_pct", 0))
                    held = int(row.get("days_held", 0))
                    p(f"      {sym:<18}  notional=${notl:>8,.2f}  "
                      f"pnl={upnl:>+8.2f}  ({upct:>+5.2f}%)  d={held}")
        elif s["type"] == "futures_short":
            for _, row in open_df.iterrows():
                sym   = str(row.get("symbol", "?"))
                edate = str(row.get("entry_date", "?"))
                held  = int(row.get("bars_held", 0))
                rem   = int(row.get("bars_remaining", 0))
                stop  = row.get("stop_loss", "?")
                try:
                    stop_str = f"{float(stop):.6g}"
                except (TypeError, ValueError):
                    stop_str = str(stop)
                p(f"    SHORT  {sym:<18}  entry={edate}  "
                  f"bars={held}/{held+rem}  stop={stop_str}")
        else:
            for _, row in open_df.iterrows():
                sym   = str(row.get("symbol", "?"))
                edate = str(row.get("entry_date", "?"))
                held  = int(row.get("bars_held", 0))
                rem   = int(row.get("bars_remaining", 0))
                raw_stop = row.get("stop_loss", row.get("current_stop", None))
                try:
                    stop_str = f"{float(raw_stop):.6g}"
                except (TypeError, ValueError):
                    stop_str = str(raw_stop) if raw_stop is not None else "?"
                p(f"    LONG   {sym:<18}  entry={edate}  "
                  f"held={held}  rem={rem}  stop={stop_str}")

    # ── Section 5: Review countdown ──────────────────────────────────────
    p()
    p("REVIEW COUNTDOWN (3-month T9B -> live eligibility)")
    for s in snaps:
        if s["not_started"]:
            p(f"  {s['id']} {s['name']:<11}  NOT STARTED -- {s['review_date']} target")
            continue
        if s["days_to_rev"] > 0:
            p(f"  {s['id']} {s['name']:<11}  {s['days_to_rev']:>3} days remaining  "
              f"(review {s['review_date']})")
        else:
            p(f"  {s['id']} {s['name']:<11}  REVIEW DUE  ({s['review_date']})")

    # ── Section 6: Regime + arbitration status ───────────────────────────
    p()
    arb_log = ROOT / "data" / "t9b_arbitration" / "daily_log.csv"
    if arb_log.exists():
        try:
            arb_df = pd.read_csv(arb_log, header=None)
            today_rows = arb_df[arb_df.iloc[:, 0] == str(today)]
            n_skipped = len(today_rows[today_rows.iloc[:, 2] == "SIGNAL_SKIPPED"])
            n_allowed = len(today_rows[today_rows.iloc[:, 2] == "SIGNAL_ALLOWED"])
            arb_str = f"Arb today: {n_allowed} allowed  {n_skipped} skipped"
        except Exception:
            arb_str = "Arb log: (parse error)"
    else:
        arb_str = "Arb log: (not found)"

    p(f"REGIME: {regime}  |  {arb_str}")
    p("NOTE: S7/S8 signals suppressed by Momentum 22-symbol footprint")
    p("      in early T9B -- September review must account for small S7/S8 sample")

    # ── Footer ────────────────────────────────────────────────────────────
    p()
    p("=" * 65)
    any_kill = any(s["kill_sw"] for s in snaps if not s["not_started"])
    any_warn = any(s["dd_pct"] < -20.0 for s in snaps if not s["not_started"])
    if any_kill:
        p("STATUS: KILL-SWITCH ACTIVE on one or more systems -- review required")
    elif any_warn:
        p("STATUS: WARN -- one or more systems have DD > 20%")
    else:
        p("STATUS: OK")
    p("[PAPER ONLY] No real orders were placed.")
    p("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
