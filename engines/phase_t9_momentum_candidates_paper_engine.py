#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9 -- shared paper-live engine for candidates 12 (cross-sectional rank momentum) and
19 (52-week-high proximity momentum), driven by their frozen T8 configs:
    data/research_candidate12_cross_sectional_momentum_t8/phase_t8_frozen_config.json
    data/research_candidate19_t8/phase_t8_frozen_config.json

One engine, two configs (per explicit instruction) -- shared code, separate state/
logs/output directories per candidate, selected via --candidate {12,19}.

Frozen from T1-T8 research, NOT re-derived here:
  - Entries: each candidate's pooled T1-survivor param combos (generate_universe_
    positions(panel, params) ranking interface), each on its OWN rebalance_n cadence.
  - Exit: flat/opposite-signal only -- NO protective stop is executed (T6/T7 already
    established this; a dedicated post-T7 stop-loss test found nothing that helped).
    A position closes when its own combo's rebalance drops it from the basket.
  - Portfolio caps (T5): max_open_positions, max_position_pct_of_equity,
    max_cluster_pct_of_equity -- enforced identically to tools/t5_portfolio_replay.py's
    replay_with_caps/_accepted_subset logic, just applied incrementally instead of as
    a full backtest replay.
  - Position sizing: risk_per_trade_pct of current equity / (ATR_MULT_R x ATR) stop
    distance, capped at max_position_pct_of_equity (T5/T6 convention).
  - Slippage: liquidity-tiered fill deduction (T6 convention), applied to both entry
    and exit fills.
  - Leverage: capped at leverage_max (<=2x for both candidates -- the only confirmed
    mitigation for their T6-flagged liquidation sensitivity).
  - Risk control: kill_switch_dd_pct=35%, equity_floor_pct=50% (matches every other
    T9/T9B engine in this repo).
  - Cluster map: modules/asset_clustering.py, corr_threshold=0.6 -- T8's blueprint
    explicitly flagged the RESEARCH map as static/full-window and NOT valid point-in-
    time; this engine fixes that by recomputing the cluster map monthly (on the 1st
    of each month) using only data available as of that recompute, not a fixed window.

Deliberately NOT wired into engines/t9b_shared.py's cross-system dedup or regime-
weight scaling -- both are hardcoded to the existing 7 S1-S8 engines
(_STATE_PATHS/_ENGINE_TO_SYSTEM/_N_SYSTEMS=7 in t9b_shared.py) and to SCHEME_C's
capital allocation. Integrating candidates 12/19 into that shared production
infrastructure is a distinct decision (how do two new systems fit into the existing
8-system capital split?) that hasn't been asked for and isn't made here -- these two
engines run as independent, self-contained $60k paper books. Uses t9b_shared.run_engine
only for its generic crash-logging wrapper, which is safe to reuse standalone.

PAPER ONLY -- NO REAL ORDERS. NO API KEYS REQUIRED for trading (ccxt used read-only
for OHLCV).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engines.t9b_shared as t9b_shared  # noqa: E402
from modules.asset_clustering import compute_static_clusters  # noqa: E402
from signal_generators import candidate_12_cross_sectional_momentum as c12  # noqa: E402
from signal_generators import candidate_19_proximity_high_momentum as c19  # noqa: E402

OHLCV_DIR = ROOT / "data" / "futures_universe" / "ohlcv_1d"
SYM_FILE = ROOT / "data" / "futures_universe" / "all_symbols.csv"

ATR_MULT_R = 2.0
ATR_WINDOW = 14
WARMUP_DAYS = 260
MIN_SYMBOL_BARS = 200
EPS = 1e-10

TIER_BPS = {"tier1": 2.0, "tier2": 8.0, "tier3": 25.0}
TIER1_ADV_USD = 50_000_000.0
TIER2_ADV_USD = 5_000_000.0
MIN_NOTIONAL_USD = 100.0

CANDIDATE_MODULES = {"12": c12, "19": c19}
CANDIDATE_T8_CONFIG = {
    "12": ROOT / "data" / "research_candidate12_cross_sectional_momentum_t8" / "phase_t8_frozen_config.json",
    "19": ROOT / "data" / "research_candidate19_t8" / "phase_t8_frozen_config.json",
}


# ============================================================
# Config / paths
# ============================================================

@dataclass
class EngineConfig:
    candidate_id: str
    entry_param_combos: list[dict]
    max_open_positions: int
    max_position_pct: float   # fraction, e.g. 0.10
    max_cluster_pct: float    # fraction, e.g. 0.30
    risk_per_trade_pct: float  # fraction, e.g. 0.0025
    initial_capital_usdt: float
    leverage_max: float
    max_margin_usage_pct: float
    kill_switch_dd_pct: float
    equity_floor_pct: float
    cluster_corr_threshold: float = 0.6
    cluster_min_size: int = 3


def load_engine_config(candidate_id: str) -> EngineConfig:
    path = CANDIDATE_T8_CONFIG[candidate_id]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EngineConfig(
        candidate_id=candidate_id,
        entry_param_combos=raw["frozen_entry_param_combos"],
        max_open_positions=int(raw["max_open_positions"]),
        max_position_pct=float(raw["max_position_pct_of_equity"]) / 100.0,
        max_cluster_pct=float(raw["max_cluster_pct_of_equity"]) / 100.0,
        risk_per_trade_pct=float(raw["risk_per_trade_pct"]) / 100.0,
        initial_capital_usdt=float(raw["initial_capital_usdt"]),
        leverage_max=float(raw["leverage_max"]),
        max_margin_usage_pct=float(raw["max_margin_usage_pct"]) / 100.0,
        kill_switch_dd_pct=float(raw["kill_switch_dd_pct"]),
        equity_floor_pct=float(raw["equity_floor_pct"]) / 100.0,
    )


def paths_for(candidate_id: str) -> dict:
    out_dir = ROOT / "data" / f"t9_candidate{candidate_id}_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        out_dir=out_dir,
        state=out_dir / "state.json",
        log=out_dir / "daily_log.csv",
        open_positions=out_dir / "open_positions.csv",
        signals=out_dir / "signals_today.csv",
        # NOT "equity_curve.csv" -- that filename is owned by .github/scripts/
        # mark_to_market.py (different schema: date,paper_equity,unrealized_pnl,
        # total_value,open_positions,total_cost,total_market_value), which the
        # Cloudflare status_worker.js /pnl handler reads. Two different schemas
        # writing to the same path corrupted the file on the first collision --
        # keep this engine's own richer risk-tracking curve under a separate name.
        equity_curve=out_dir / "engine_equity_curve.csv",
        health=out_dir / "system_health.json",
        cluster_cache=out_dir / "cluster_map_cache.json",
    )


# ============================================================
# Utilities
# ============================================================

def p(*a, **kw):
    kw.setdefault("flush", True)
    text = " ".join(str(x) for x in a)
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), **kw)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T9 momentum candidates (12/19) paper engine")
    ap.add_argument("--candidate", required=True, choices=["12", "19"])
    ap.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today)")
    ap.add_argument("--no-download", action="store_true", help="Skip live ccxt data update")
    ap.add_argument("--reset", action="store_true", help="Wipe state and restart from initial capital")
    ap.add_argument("--notify", action="store_true", help="Print compact summary at end")
    return ap.parse_args()


# ============================================================
# Data update (delta via ccxt) -- same pattern as the momentum_factor T9B engine
# ============================================================

def rebuild_symbol_file() -> int:
    syms = sorted(f.name[: -len("_1d.csv")] for f in OHLCV_DIR.glob("*_1d.csv"))
    if not syms:
        return 0
    with open(SYM_FILE, "w", encoding="utf-8") as fh:
        fh.write("symbol\n")
        for s in syms:
            fh.write(s + "\n")
    return len(syms)


def update_ohlcv_delta(symbols: list[str]) -> None:
    try:
        import ccxt
    except ImportError:
        p("  [WARN] ccxt not installed -- skipping data update, using cached CSVs")
        return

    p(f"  Fetching delta candles for {len(symbols)} symbols via ccxt binanceusdm...")
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30_000})
    try:
        markets = exchange.load_markets()
    except Exception as exc:
        p(f"  [WARN] load_markets failed: {exc} -- using cached data")
        return

    id_to_ccxt = {}
    for ccxt_sym, mkt in markets.items():
        if mkt.get("type") == "swap" and mkt.get("quote") == "USDT" and mkt.get("settle") == "USDT":
            id_to_ccxt[mkt["id"]] = ccxt_sym

    updated = errors = skipped = 0
    for sym_id in symbols:
        csv_path = OHLCV_DIR / f"{sym_id}_1d.csv"
        if not csv_path.exists():
            skipped += 1
            continue
        ccxt_sym = id_to_ccxt.get(sym_id)
        if not ccxt_sym:
            skipped += 1
            continue
        try:
            df_old = pd.read_csv(csv_path)
            last_ts = int(df_old["timestamp"].max())
            since = last_ts + 86_400_000
            batch = exchange.fetch_ohlcv(ccxt_sym, "1d", since=since, limit=10)
            if not batch:
                updated += 1
                continue
            df_new = pd.DataFrame(batch, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_new["date"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
            df_combined = (pd.concat([df_old, df_new]).drop_duplicates("timestamp")
                           .sort_values("timestamp").reset_index(drop=True))
            df_combined.to_csv(csv_path, index=False)
            updated += 1
        except Exception:
            errors += 1
        time.sleep(0.05)

    p(f"  Delta update: {updated} OK  |  {errors} errors  |  {skipped} skipped")


def load_panel() -> dict[str, pd.DataFrame]:
    panel = {}
    for f in sorted(OHLCV_DIR.glob("*_1d.csv")):
        symbol = f.stem.replace("_1d", "")
        df = pd.read_csv(f)
        if "date" not in df.columns or not {"open", "high", "low", "close"} <= set(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        panel[symbol] = df[["open", "high", "low", "close", "volume"]]
    return panel


def add_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / window, adjust=False).mean()
    return df


# ============================================================
# State
# ============================================================

def default_state(cfg: EngineConfig) -> dict:
    return {
        "system_name": f"T9_CANDIDATE{cfg.candidate_id}_MOMENTUM",
        "created_utc": utc_now_str(),
        "engine_start_date": None,   # anchor for per-combo rebalance-day math
        "last_run_date": None,
        "combo_last_rebalance": {},  # {combo_index(str): "YYYY-MM-DD"}
        "closed_equity_usdt": cfg.initial_capital_usdt,
        "peak_equity_usdt": cfg.initial_capital_usdt,
        "drawdown_pct": 0.0,
        "kill_switch_triggered": False,
        "open_positions": [],        # each tagged with combo_index, cluster, notional_pct
        "closed_trade_count": 0,
        "last_error": None,
    }


def load_state(paths: dict, cfg: EngineConfig, reset: bool) -> dict:
    if reset or not paths["state"].exists():
        return default_state(cfg)
    return json.loads(paths["state"].read_text(encoding="utf-8"))


def save_state(paths: dict, state: dict) -> None:
    tmp = paths["state"].with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(paths["state"])


def reserved_margin(open_positions: list[dict]) -> float:
    return sum(p_["margin_reserved_usdt"] for p_ in open_positions)


def open_risk_amount(open_positions: list[dict]) -> float:
    return sum(p_["risk_amount_usdt"] for p_ in open_positions)


# ============================================================
# Cluster map -- recomputed monthly (fixes the T8-flagged static/full-window issue)
# ============================================================

def get_cluster_map(panel: dict, cfg: EngineConfig, paths: dict, today: Date) -> dict[str, str]:
    cache = {}
    if paths["cluster_cache"].exists():
        cache = json.loads(paths["cluster_cache"].read_text(encoding="utf-8"))

    recompute_needed = (
        not cache
        or cache.get("computed_for_month") != f"{today.year:04d}-{today.month:02d}"
    )
    if not recompute_needed:
        return cache["symbol_to_cluster"]

    p("  Recomputing correlation cluster map (monthly refresh, point-in-time -- "
      "uses only data available as of today, not the research's static full-window map)...")
    end = pd.Timestamp(today)
    start = end - pd.Timedelta(days=365)  # trailing 1y correlation window for the live map
    clusters = compute_static_clusters(panel, corr_threshold=cfg.cluster_corr_threshold,
                                        min_cluster_size=cfg.cluster_min_size, start=start, end=end)
    symbol_to_cluster = {}
    for cluster_id, members in clusters.items():
        if cluster_id == "unclustered":
            for m in members:
                symbol_to_cluster[m] = f"solo_{m}"
        else:
            for m in members:
                symbol_to_cluster[m] = cluster_id

    paths["cluster_cache"].write_text(json.dumps(
        {"computed_for_month": f"{today.year:04d}-{today.month:02d}", "symbol_to_cluster": symbol_to_cluster},
        indent=2), encoding="utf-8")
    return symbol_to_cluster


# ============================================================
# Rebalance-day logic per combo
# ============================================================

def is_combo_rebal_day(state: dict, combo_idx: int, rebalance_n: int, today: Date) -> bool:
    key = str(combo_idx)
    last = state["combo_last_rebalance"].get(key)
    if last is None:
        return True  # first run: every combo rebalances immediately
    days_since = (today - Date.fromisoformat(last)).days
    return days_since >= rebalance_n


def get_liquidity_tier(panel: dict, symbol: str, as_of: pd.Timestamp) -> str:
    df = panel.get(symbol)
    if df is None:
        return "tier3"
    sub = df[df.index <= as_of].tail(20)
    if len(sub) < 5:
        return "tier3"
    adv = (sub["close"] * sub["volume"]).mean()
    if not np.isfinite(adv):
        return "tier3"
    if adv >= TIER1_ADV_USD:
        return "tier1"
    if adv >= TIER2_ADV_USD:
        return "tier2"
    return "tier3"


def slippage_fill(price: float, side: str, tier: str, closing: bool) -> float:
    """Apply liquidity-tiered slippage against the trade (T6 convention): buys/covers
    pay up, sells/shorts receive down."""
    bps = TIER_BPS[tier] / 10_000.0
    adverse = 1 if (side == "long") != closing else -1  # opening long or closing short = pay up
    return price * (1 + adverse * bps)


# ============================================================
# Main engine
# ============================================================

def main() -> int:
    args = parse_args()
    cid = args.candidate
    today = Date.fromisoformat(args.date) if args.date else Date.today()

    p("=" * 70)
    p(f"  T9 MOMENTUM CANDIDATE {cid} PAPER ENGINE  |  {today}")
    p("=" * 70)

    cfg = load_engine_config(cid)
    paths = paths_for(cid)
    module = CANDIDATE_MODULES[cid]

    state = load_state(paths, cfg, args.reset)
    if args.reset:
        p("  --reset: wiping state")

    if state["last_run_date"] == str(today):
        p("  Already ran today -- exiting (use --reset or --date to override)")
        return 0

    if state["engine_start_date"] is None:
        state["engine_start_date"] = str(today)

    if not SYM_FILE.exists():
        n = rebuild_symbol_file()
        p(f"  [WARN] {SYM_FILE.name} was missing -- rebuilt from ohlcv_1d cache ({n} symbols)")
    all_syms = pd.read_csv(SYM_FILE)["symbol"].tolist() if SYM_FILE.exists() else []

    if not args.no_download and all_syms:
        update_ohlcv_delta(all_syms)
    else:
        p("  --no-download: using cached CSVs")

    p("  Loading 290-symbol futures panel...")
    panel = load_panel()
    p(f"  Loaded {len(panel)} symbols")

    slice_start = pd.Timestamp(today) - pd.Timedelta(days=WARMUP_DAYS)
    prepared = {}
    for sym, df in panel.items():
        sub = df[df.index >= slice_start]
        if len(sub) >= MIN_SYMBOL_BARS:
            prepared[sym] = add_atr(sub)

    cluster_map = get_cluster_map(panel, cfg, paths, today)

    open_positions = state["open_positions"]
    equity = float(state["closed_equity_usdt"])

    # --------------------------------------------------------
    # Kill-switch / equity floor check (from prior state, before this run's activity)
    # --------------------------------------------------------
    if state["drawdown_pct"] <= -cfg.kill_switch_dd_pct:
        state["kill_switch_triggered"] = True
    if equity <= cfg.initial_capital_usdt * cfg.equity_floor_pct:
        state["kill_switch_triggered"] = True

    closed_rows = []
    signal_rows = []
    skipped_rows = []

    # --------------------------------------------------------
    # 1) For each combo whose rebalance day has arrived: compute target basket,
    #    close positions that fell out, propose new entries.
    # --------------------------------------------------------
    proposed_entries = []  # list of dicts: symbol, side, combo_idx before caps applied

    for combo_idx, params in enumerate(cfg.entry_param_combos):
        rebalance_n = int(params["rebalance_n"])
        if not is_combo_rebal_day(state, combo_idx, rebalance_n, today):
            continue

        p(f"\n  Combo {combo_idx} rebalance day: {params}")
        positions_by_symbol = module.generate_universe_positions(panel, params)

        # target side per symbol as of the latest available bar
        target = {}
        for sym, pos_series in positions_by_symbol.items():
            if len(pos_series) == 0:
                continue
            val = int(pos_series.iloc[-1])
            if val != 0:
                target[sym] = "long" if val == 1 else "short"

        # close this combo's positions that fell out of the new basket
        still_open = []
        for pos in open_positions:
            if pos["combo_idx"] != combo_idx:
                still_open.append(pos)
                continue
            new_side = target.get(pos["symbol"])
            if new_side == pos["side"]:
                still_open.append(pos)  # held
                continue
            # closed: flat or opposite-signal
            exit_px_raw = panel[pos["symbol"]]["close"].iloc[-1] if pos["symbol"] in panel else pos["entry_price"]
            tier = get_liquidity_tier(panel, pos["symbol"], pd.Timestamp(today))
            exit_px = slippage_fill(float(exit_px_raw), pos["side"], tier, closing=True)
            gross_r = ((exit_px - pos["entry_price"]) / pos["stop_distance_px"] if pos["side"] == "long"
                       else (pos["entry_price"] - exit_px) / pos["stop_distance_px"])
            pnl = pos["risk_amount_usdt"] * gross_r
            equity += pnl
            closed_rows.append({
                "timestamp_utc": utc_now_str(), "symbol": pos["symbol"], "side": pos["side"],
                "combo_idx": combo_idx, "entry_date": pos["entry_date"], "exit_date": str(today),
                "entry_price": pos["entry_price"], "exit_price": exit_px, "net_r": gross_r,
                "pnl_usdt": pnl, "risk_amount_usdt": pos["risk_amount_usdt"],
                "notional_usdt": pos["notional_usdt"], "exit_reason": "REBALANCE_DROPPED",
            })
            state["closed_trade_count"] += 1

        open_positions = still_open

        for sym, side in target.items():
            already_open = any(p_["symbol"] == sym and p_["combo_idx"] == combo_idx for p_ in open_positions)
            if not already_open:
                proposed_entries.append({"symbol": sym, "side": side, "combo_idx": combo_idx})

        state["combo_last_rebalance"][str(combo_idx)] = str(today)

    # peak/drawdown update from realized closes this run
    peak = max(float(state["peak_equity_usdt"]), equity)
    dd_pct = (equity - peak) / max(peak, EPS) * 100.0
    state["peak_equity_usdt"] = peak
    state["drawdown_pct"] = dd_pct
    if dd_pct <= -cfg.kill_switch_dd_pct:
        state["kill_switch_triggered"] = True

    # --------------------------------------------------------
    # 2) Apply portfolio-level caps to proposed entries (max_open, max_position_pct,
    #    max_cluster_pct), same acceptance logic as tools/t5_portfolio_replay.py
    # --------------------------------------------------------
    for entry in proposed_entries:
        sym, side = entry["symbol"], entry["side"]

        if state["kill_switch_triggered"]:
            skipped_rows.append({**entry, "reason": "KILL_SWITCH"})
            continue

        sub_atr = prepared.get(sym)
        if sub_atr is None or len(sub_atr) == 0:
            skipped_rows.append({**entry, "reason": "NO_PRICE_DATA"})
            continue

        atr = float(sub_atr["atr"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            skipped_rows.append({**entry, "reason": "INVALID_ATR"})
            continue

        entry_px_raw = float(sub_atr["close"].iloc[-1])
        stop_distance_px = ATR_MULT_R * atr
        stop_distance_pct = stop_distance_px / entry_px_raw

        cluster = cluster_map.get(sym, f"solo_{sym}")
        cluster_pct_open = sum(p_["notional_pct"] for p_ in open_positions if p_["cluster"] == cluster)

        if len(open_positions) >= cfg.max_open_positions:
            skipped_rows.append({**entry, "reason": "MAX_OPEN"})
            continue

        target_notional_pct = min(cfg.risk_per_trade_pct / stop_distance_pct, cfg.max_position_pct)
        if cluster_pct_open + target_notional_pct > cfg.max_cluster_pct:
            skipped_rows.append({**entry, "reason": "MAX_CLUSTER_PCT"})
            continue

        notional_usdt = target_notional_pct * equity
        if notional_usdt < MIN_NOTIONAL_USD:
            skipped_rows.append({**entry, "reason": "BELOW_MIN_NOTIONAL"})
            continue

        margin_usdt = notional_usdt / cfg.leverage_max
        if reserved_margin(open_positions) + margin_usdt > equity * cfg.max_margin_usage_pct:
            skipped_rows.append({**entry, "reason": "MAX_MARGIN_USAGE"})
            continue

        tier = get_liquidity_tier(panel, sym, pd.Timestamp(today))
        fill_px = slippage_fill(entry_px_raw, side, tier, closing=False)
        risk_amount_usdt = target_notional_pct * stop_distance_pct * equity

        new_pos = {
            "symbol": sym, "side": side, "combo_idx": entry["combo_idx"],
            "entry_date": str(today), "entry_price": fill_px,
            "stop_distance_px": stop_distance_px, "atr_at_entry": atr,
            "notional_usdt": notional_usdt, "notional_pct": target_notional_pct,
            "margin_reserved_usdt": margin_usdt, "risk_amount_usdt": risk_amount_usdt,
            "cluster": cluster, "liquidity_tier": tier,
        }
        open_positions.append(new_pos)
        signal_rows.append({**entry, "action": "OPEN", "entry_price": fill_px,
                             "notional_usdt": notional_usdt, "cluster": cluster, "tier": tier})

    # --------------------------------------------------------
    # 3) Finalize equity/state, write outputs
    # --------------------------------------------------------
    peak = max(float(state["peak_equity_usdt"]), equity)
    dd_pct = (equity - peak) / max(peak, EPS) * 100.0
    state["closed_equity_usdt"] = round(equity, 4)
    # Alias for .github/scripts/mark_to_market.py, which reads paper_equity_usdt
    # across every system -- kept in sync with closed_equity_usdt, not a separate
    # value, so this engine's own logic is unaffected.
    state["paper_equity_usdt"] = state["closed_equity_usdt"]
    state["peak_equity_usdt"] = round(peak, 4)
    state["drawdown_pct"] = round(dd_pct, 4)
    state["open_positions"] = open_positions
    state["last_run_date"] = str(today)

    _append_log(paths, today, "RUN",
                f"closed={len(closed_rows)} opened={len(signal_rows)} skipped={len(skipped_rows)} "
                f"open_now={len(open_positions)} equity=${equity:,.2f} dd={dd_pct:.2f}%", equity)
    _append_csv(paths["log"].parent / f"closed_trades_c{cid}.csv", closed_rows)
    _write_open_positions(paths, panel, open_positions, today)
    if signal_rows or skipped_rows:
        _write_signals(paths, signal_rows, skipped_rows)
    _append_equity_curve(paths, today, equity, peak, dd_pct, len(open_positions),
                          open_risk_amount(open_positions), reserved_margin(open_positions))

    health = {
        "system_name": state["system_name"], "timestamp_utc": utc_now_str(),
        "status": "KILL_SWITCH" if state["kill_switch_triggered"] else "OK",
        "candidate_id": cid, "open_positions": len(open_positions),
        "closed_equity_usdt": equity, "drawdown_pct": dd_pct,
        "portfolio_heat_pct": open_risk_amount(open_positions) / max(equity, EPS) * 100.0,
        "reserved_margin_pct": reserved_margin(open_positions) / max(equity, EPS) * 100.0,
        "entries_this_run": len(signal_rows), "closes_this_run": len(closed_rows),
        "skipped_this_run": len(skipped_rows), "paper_only": True,
    }
    paths["health"].write_text(json.dumps(health, indent=2), encoding="utf-8")

    save_state(paths, state)

    if args.notify:
        _print_notify(state, cid, today, len(closed_rows), len(signal_rows))
    else:
        p(f"\n  Done. equity=${equity:,.2f}  dd={dd_pct:.2f}%  "
          f"open={len(open_positions)}  closed_this_run={len(closed_rows)}  "
          f"opened_this_run={len(signal_rows)}  skipped={len(skipped_rows)}  "
          f"kill_switch={state['kill_switch_triggered']}")
    p()
    return 0


def _append_log(paths, run_date, event, detail, equity):
    exists = paths["log"].exists()
    with open(paths["log"], "a", encoding="utf-8", newline="") as f:
        if not exists:
            f.write("run_date,event,detail,equity\n")
        detail_clean = str(detail).replace('"', "'").replace("\n", " ")
        f.write(f'{run_date},{event},"{detail_clean}",{equity:.4f}\n')


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def _write_open_positions(paths, panel, open_positions, today):
    rows = []
    for pos in open_positions:
        price_series = panel.get(pos["symbol"], pd.DataFrame()).get("close")
        cur_price = float(price_series.iloc[-1]) if price_series is not None and len(price_series) else pos["entry_price"]
        gross_r = ((cur_price - pos["entry_price"]) / pos["stop_distance_px"] if pos["side"] == "long"
                   else (pos["entry_price"] - cur_price) / pos["stop_distance_px"])
        rows.append({
            "symbol": pos["symbol"], "side": pos["side"], "combo_idx": pos["combo_idx"],
            "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "current_price": cur_price, "unrealized_r": round(gross_r, 4),
            # qty in base-asset units, for compatibility with .github/scripts/
            # mark_to_market.py's shared cost/market_value formula (entry_price*qty,
            # current_price*qty) -- same convention the S1-S3 engines' open_positions.csv
            # already use, so mark_to_market.py needs no per-system special-casing.
            "qty": round(pos["notional_usdt"] / pos["entry_price"], 8),
            "notional_usdt": round(pos["notional_usdt"], 2), "cluster": pos["cluster"],
            "days_held": (today - Date.fromisoformat(pos["entry_date"])).days,
        })
    pd.DataFrame(rows).to_csv(paths["open_positions"], index=False)


def _write_signals(paths, signal_rows, skipped_rows):
    all_rows = [{**r, "status": "OPENED"} for r in signal_rows] + \
               [{**r, "status": "SKIPPED"} for r in skipped_rows]
    pd.DataFrame(all_rows).to_csv(paths["signals"], index=False)


def _append_equity_curve(paths, today, equity, peak, dd_pct, n_open, open_risk, reserved_margin_usdt):
    exists = paths["equity_curve"].exists()
    with open(paths["equity_curve"], "a", encoding="utf-8", newline="") as f:
        if not exists:
            f.write("date,equity,peak_equity,drawdown_pct,open_positions,open_risk_usdt,reserved_margin_usdt\n")
        f.write(f"{today},{equity:.4f},{peak:.4f},{dd_pct:.4f},{n_open},{open_risk:.4f},{reserved_margin_usdt:.4f}\n")


def _print_notify(state, cid, today, n_closed, n_opened):
    p()
    p("=" * 60)
    p(f"  T9 CANDIDATE {cid}  |  {today}  |  PAPER")
    p("=" * 60)
    p(f"  Equity:              ${state['closed_equity_usdt']:>10,.2f}")
    p(f"  Peak equity:         ${state['peak_equity_usdt']:>10,.2f}")
    p(f"  Drawdown:            {state['drawdown_pct']:>+9.2f}%")
    p(f"  Kill-switch:         {'TRIGGERED' if state['kill_switch_triggered'] else 'not triggered'}")
    p(f"  Open positions:      {len(state['open_positions']):>4}")
    p(f"  Closed this run:     {n_closed:>4}")
    p(f"  Opened this run:     {n_opened:>4}")
    p("=" * 60)


if __name__ == "__main__":
    t9b_shared.run_engine(f"t9_candidate_momentum", main)
