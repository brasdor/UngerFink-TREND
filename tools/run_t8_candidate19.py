#!/usr/bin/env python3
"""
T8 paper-live blueprint for candidate 19 (52-week-high / proximity-to-high momentum,
futures) -- the second candidate (alongside 12) to pass T7 strict robustness this
session, and the strongest overall result across T4-T7 (cleanest T4, best capped T5
Sharpe, T7 rolling-window profile matching 12's). 11u and 13u remain set aside per
candidate 12's T8 blueprint; 18/20 were eliminated earlier (18 at T0 redundancy-
adjacent overlap, 20 at T1 by the diversification gate).

Matches the existing repo's T8 convention and candidate 12's own T8 blueprint format
exactly: freezes the configuration, does NOT add filters or optimize, and produces the
same artifact set -- frozen_config.json, live_state.json, system_health.json,
paper_live_checklist.txt, closed_equity_template.csv, master_report.txt.

This is NOT a live trading engine (same disclaimer as candidate 12's T8 script and the
original repo T8 script). That is T9, not attempted here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "data" / "research_candidate19_t8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T1_ROBUSTNESS = ROOT / "data/research_candidate19_t1/t1_robustness.csv"


def frozen_entry_params() -> list[dict]:
    rob = pd.read_csv(T1_ROBUSTNESS)
    survivors = rob[rob["survives"]]
    return [dict(lookback=int(r["lookback"]), quantile=float(r["quantile"]),
                 rebalance_n=int(r["rebalance_n"]), side=r["side"])
            for _, r in survivors.iterrows()]


def main():
    entry_params = frozen_entry_params()

    frozen_config = {
        "system_name": "CANDIDATE19_PROXIMITY_HIGH_MOMENTUM_FUTURES",
        "version": "T8_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),

        # MARKET
        "exchange": "binance",
        "market_type": "futures_usdt_m",
        "universe_size": 290,
        "universe_note": "290-symbol Binance USDT-M futures universe as of T1 -- "
                          "listings/delistings will drift this over time; requires "
                          "periodic universe-file maintenance in T9.",

        # CORE LOGIC (T1->T7 confirmed, IS window 2020-01-01 to 2024-12-31)
        "timeframe": "1d",
        "entry_logic": "proximity_to_trailing_high_momentum",
        "entry_module": "signal_generators/candidate_19_proximity_high_momentum.py:generate_universe_positions",
        "frozen_entry_param_combos": entry_params,
        "entry_note": f"{len(entry_params)} T1-survivor combos (diversification gate "
                       f"active from T1 itself -- max_month_share=25%, max_asset_share="
                       f"20% -- 9/24 combos passed all gates, 8/9 survive drop-top-2) "
                       f"pooled and traded simultaneously (not a single hand-picked "
                       f"parameterization) -- all long_only; no short-side combo "
                       f"survived drop-top-2 robustness.",
        "exit_logic": "flat_opposite_signal",
        "exit_note": "Frozen at T3 -- breakeven/chandelier/partial-profit variants "
                      "all tested and none beat this baseline. A post-T7 exploratory "
                      "stop-loss test (fixed R-multiple, empirical-percentile-"
                      "calibrated, wider ATR-multiple, time-based, bar-close-confirmed) "
                      "found NONE meaningfully reduced liquidation exposure without "
                      "unacceptable edge cost -- no stop-loss variant adopted. "
                      "IMPORTANT: ATR x 2.0 is a POSITION-SIZING / R-normalization "
                      "unit only, NOT an executed protective stop order. Exits are "
                      "purely signal-based (rank/quantile membership at rebalance). "
                      "Intrabar adverse excursions are not capped by the strategy "
                      "itself -- this is the direct cause of the T6 liquidation "
                      "sensitivity below.",
        "initial_risk_atr_mult": 2.0,
        "closed_candles_only": True,
        "allow_short": False,

        # PORTFOLIO (T5-confirmed caps)
        "max_open_positions": 20,
        "max_position_pct_of_equity": 10.0,
        "max_cluster_pct_of_equity": 30.0,
        "cluster_method": "modules/asset_clustering.py:compute_static_clusters, corr_threshold=0.6, min_cluster_size=3",
        "cluster_note": "Research used a STATIC full-window (2020-2024) correlation "
                         "map -- not available point-in-time live. Seed-selection "
                         "non-determinism (Python set iteration order) was fixed prior "
                         "to this candidate's T7/T8 runs (sorted seed order), so this "
                         "trade set is reproducible run to run -- unlike candidate 12's "
                         "original T5/T6 runs, which predated the fix. T9 must still "
                         "implement a periodic rolling recompute (e.g. monthly, "
                         "trailing 180d correlations) for live point-in-time use -- "
                         "not yet built.",
        "risk_per_trade_pct": 0.25,
        "max_portfolio_heat_pct": 5.0,
        "max_portfolio_heat_note": "20 max_open x 0.25% risk-per-trade = 5.0% "
                                    "theoretical max concurrent open risk.",

        # EXECUTION (T6-confirmed)
        "slippage_model": "liquidity_tiered",
        "slippage_tiers_bps_per_fill": {"tier1_adv_ge_50m": 2.0, "tier2_adv_ge_5m": 8.0, "tier3_adv_lt_5m": 25.0},
        "fill_assumption": "market_at_next_bar_open",
        "fill_note": "T6 tested limit fills (15bps inside open) -- 97.6% fill rate but "
                      "SLIGHTLY worse avg_r among filled trades (adverse selection). "
                      "Market-at-open is confirmed as the correct convention.",

        # CAPITAL / MARGIN
        "initial_capital_usdt": 60000,
        "leverage_max": 2.0,
        "leverage_note": "T6 liquidation sweep: 18.3% of trades would be liquidated "
                          "before their modeled exit at 3x leverage (43.3% at 5x, "
                          "71.4% at 10x) -- nearly as leverage-sensitive as candidate "
                          "12 (21.6%/45.1%/71.8%), the highest of any candidate this "
                          "session besides 12. A dedicated post-T7 stop-loss test found "
                          "no exit-side mitigation that meaningfully helped (see "
                          "exit_note) -- <=2x leverage is the ONLY confirmed effective "
                          "mitigation, not an exit-engineering fix. <=2x also covers "
                          "the theoretical worst-case stacking of 20 positions at the "
                          "10% per-position cap.",
        "max_margin_usage_pct": 85.0,
        "min_notional_usd_floor": 100.0,
        "min_notional_note": "Position sizing check (T6): actual notional at 0.25% "
                              "risk ranged $184-$4,309 across the trade sample, 0.00% "
                              "of trades below the $100 floor.",

        # RISK CONTROL (matches candidate 12's T8 blueprint and existing live-engine
        # convention, e.g. engines/phase_t9b_momentum_factor_paper_engine.py KILL_SWITCH_DD_PCT)
        "kill_switch_dd_pct": 35.0,
        "equity_floor_pct": 50.0,

        # RESEARCH STATUS
        "status": "PAPER_ONLY",
        "live_trading_allowed": False,
        "research_phase": "T8",
        "related_candidates": {
            "12": "PASS T7, T8 frozen separately (data/research_candidate12_cross_sectional_momentum_t8/) "
                  "-- same $60k/max_open=20/leverage<=2x discipline, different entry construction "
                  "(raw rank momentum vs 19's proximity-to-high).",
            "11u": "T7 FAIL, set aside pending a stricter asset-diversification entry gate (future work).",
            "13u": "T7 FAIL (narrower), same future-work track as 11u.",
            "18": "T0 -- genuinely redundant with candidate 12 (avg corr 0.87), not pursued.",
            "20": "T1 FAIL -- every combo eliminated by the diversification gate.",
        },
    }

    live_state = {
        "last_update_utc": None,
        "system_status": "IDLE",
        "kill_switch_triggered": False,
        "open_positions": [],
        "closed_equity": frozen_config["initial_capital_usdt"],
        "peak_equity": frozen_config["initial_capital_usdt"],
        "drawdown_pct": 0.0,
        "portfolio_heat_pct": 0.0,
        "last_error": None,
    }

    system_health = {
        "system_name": frozen_config["system_name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_PAPER_LIVE",
        "research_complete": {"T1": True, "T2": True, "T3": True, "T4": True,
                               "T5": True, "T6": True, "T7": True, "T8": True},
        "warnings": [
            "T6: liquidation exposure rises sharply with leverage (18.3% of trades "
            "at 3x, nearly matching candidate 12's 21.6%) -- leverage MUST be capped "
            "at <=2x in T9. A dedicated stop-loss variant test post-T7 found no "
            "exit-side fix that meaningfully reduces this without giving up 26-43% "
            "of the edge -- leverage discipline is the only confirmed mitigation.",
            "No actual protective stop-loss is executed -- exits are signal-based "
            "only (see exit_note); do not assume ATRx2.0 caps single-trade loss.",
            "Correlation cluster map is static/full-window from research (though now "
            "deterministic) -- T9 needs a periodic rolling recompute, not yet built.",
            "290-symbol futures universe requires ongoing listing/delisting "
            "maintenance.",
            "T2 regime dependence flag=True (2022 net-negative year) -- same "
            "session-wide pattern seen in nearly every candidate, not a distinguishing "
            "weakness, but worth monitoring in live drawdown attribution.",
            "11u and 13u remain set aside; 18 and 20 were eliminated earlier in the "
            "pipeline -- none of these are deployed alongside 19.",
        ],
        "strengths": [
            "T4: cleanest result of any candidate this session -- clears 2x cost "
            "stress outright (avg_r=0.3896R), no fragility on any test, lowest "
            "best-asset concentration seen (BTCUSDT, 9.8% of total R).",
            "T5: strongest capped result of any candidate -- capped Sharpe 2.43-2.52 "
            "(vs 1.59-1.88 for 11u/12/13u), drawdown improved dramatically under "
            "capping (-28.5% to -14.1%).",
            "T7 PASSES all strict robustness tests on the locked, capped, "
            "slippage-adjusted trade set: cost stress to +0.15R, remove-top-5-"
            "assets, remove-top-2-months all clear the 0.25R floor.",
            "Only 44.5% of rolling 30-trade windows fall below the cost floor "
            "(vs 73-78% for 11u/13u, which failed T7; matches candidate 12's 44.1%) "
            "-- edge is comparatively well-distributed across time and assets.",
            "Position sizing at 0.25% risk never approaches the $100 min-notional "
            "floor (min observed $184).",
            "Slippage impact is small under the liquidity-tiered T6 model "
            "(-1.7% return, -0.01 Sharpe vs pre-slippage capped baseline).",
            "Market-at-open fills confirmed correct -- attempted limit fills show "
            "no edge improvement (slight adverse selection).",
        ],
    }

    checklist = [
        "Verify Binance USDT-M futures data integrity for the 290-symbol universe",
        "Use CLOSED candles only",
        "No intrabar execution assumptions beyond the modeled liquidity-tiered slippage",
        "No discretionary overrides",
        "Implement periodic (monthly) rolling correlation-cluster recompute -- "
        "static full-window map from research is NOT valid point-in-time",
        "Enforce leverage cap <=2x at the exchange/account level, not just in sizing logic",
        "Monitor skipped signals (MAX_OPEN / MAX_CLUSTER_PCT rejections)",
        "Monitor portfolio heat against the 5.0% theoretical ceiling",
        "Monitor equity only from CLOSED trades",
        "Do not optimise during observation",
        "Do not retrofit a stop-loss order based on the exploratory test in this "
        "research phase -- none was found to meaningfully help; re-open T3 formally "
        "if this is revisited, don't patch it into T9 informally",
        "Observe at least 100-200 additional live signals before considering real deployment",
        "Compare live equity trajectory to T7's rolling-window distribution "
        "(44.5% of historical 30-trade windows fell below the cost floor -- "
        "expect and tolerate stretches like this, do not kill-switch on that basis alone)",
    ]

    closed_equity_columns = ["timestamp_utc", "closed_equity", "peak_equity",
                              "drawdown_pct", "closed_trade_count"]
    closed_equity_df = pd.DataFrame(columns=closed_equity_columns)

    report_lines = [
        "PHASE T8 -- PAPER LIVE BLUEPRINT (Candidate 19)",
        "=" * 70, "",
        "SYSTEM STATUS", "-" * 70,
        "Research phase completed (T1-T7). System frozen for paper-live observation.",
        "Strongest overall result this session (T4/T5/T7 all best-or-tied-best); "
        "leverage sensitivity at T6 is the one caveat, mitigated via <=2x leverage cap "
        "-- confirmed as the only effective fix after a dedicated stop-loss test.",
        "",
        "FROZEN CORE", "-" * 70,
        f"Timeframe:           {frozen_config['timeframe']}",
        f"Entry:               {frozen_config['entry_logic']} ({len(entry_params)} pooled T1-survivor combos)",
        f"Entry params:        {entry_params}",
        f"Exit:                {frozen_config['exit_logic']} (no executed stop -- see exit_note)",
        f"Universe:            {frozen_config['universe_size']}-symbol Binance USDT-M futures",
        "",
        "PORTFOLIO", "-" * 70,
        f"Max open positions:  {frozen_config['max_open_positions']}",
        f"Max position size:   {frozen_config['max_position_pct_of_equity']}% of equity",
        f"Max cluster conc.:   {frozen_config['max_cluster_pct_of_equity']}% of equity",
        f"Risk per trade:      {frozen_config['risk_per_trade_pct']}%",
        f"Max portfolio heat:  {frozen_config['max_portfolio_heat_pct']}%",
        "",
        "EXECUTION", "-" * 70,
        f"Fill assumption:     {frozen_config['fill_assumption']}",
        f"Slippage model:      {frozen_config['slippage_model']} {frozen_config['slippage_tiers_bps_per_fill']}",
        "",
        "CAPITAL", "-" * 70,
        f"Initial capital:     ${frozen_config['initial_capital_usdt']:,} USDT",
        f"Max leverage:        {frozen_config['leverage_max']}x isolated margin",
        f"Max margin usage:    {frozen_config['max_margin_usage_pct']}% of equity",
        "",
        "RISK CONTROL", "-" * 70,
        f"Kill-switch DD:      -{frozen_config['kill_switch_dd_pct']}%",
        f"Equity floor:        {frozen_config['equity_floor_pct']}% of initial capital",
        "",
        "IMPORTANT RULE", "-" * 70,
        "NO NEW FILTERS", "NO OPTIMISATION", "NO LIVE CAPITAL YET",
        "NO INFORMAL STOP-LOSS RETROFIT (see exit_note -- tested, rejected)", "",
        "NEXT PHASE", "-" * 70,
        "T9 -- Binance Futures Paper Sim Engine",
        "- live Binance futures candles (closed-candle execution)",
        "- persistent state, CSV logging, dashboard monitoring",
        "- rolling correlation-cluster recompute (not yet built)",
        "- exchange-level leverage cap enforcement",
        "- can share T9 infrastructure with candidate 12's blueprint (same caps/",
        "  capital/leverage discipline), run as two independent frozen systems",
        "",
        "OBSERVATION OBJECTIVE", "-" * 70,
        "Observe at least 100-200 additional paper signals before considering real deployment.",
    ]

    (OUTPUT_DIR / "phase_t8_frozen_config.json").write_text(json.dumps(frozen_config, indent=4), encoding="utf-8")
    (OUTPUT_DIR / "phase_t8_live_state.json").write_text(json.dumps(live_state, indent=4), encoding="utf-8")
    (OUTPUT_DIR / "phase_t8_system_health.json").write_text(json.dumps(system_health, indent=4), encoding="utf-8")
    (OUTPUT_DIR / "phase_t8_paper_live_checklist.txt").write_text(
        "\n".join(f"- {c}" for c in checklist), encoding="utf-8")
    closed_equity_df.to_csv(OUTPUT_DIR / "phase_t8_closed_equity_template.csv", index=False)
    (OUTPUT_DIR / "phase_t8_master_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("PHASE T8 -- PAPER LIVE BLUEPRINT (Candidate 19)")
    print("=" * 80)
    print(f"Output dir: {OUTPUT_DIR}\n")
    for f in ["phase_t8_frozen_config.json", "phase_t8_live_state.json", "phase_t8_system_health.json",
              "phase_t8_paper_live_checklist.txt", "phase_t8_closed_equity_template.csv", "phase_t8_master_report.txt"]:
        print(f"[OK] {f}")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
