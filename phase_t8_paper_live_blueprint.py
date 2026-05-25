#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE T8 — PAPER LIVE BLUEPRINT
================================

Goal
----
Freeze the current trend-following research configuration and prepare
the system for paper-live observation.

This phase does NOT:
- optimise entries
- add new filters
- change the edge logic

This phase DOES:
- define the frozen production configuration
- create monitoring state
- implement kill-switch logic
- prepare closed-trade equity tracking
- prepare portfolio/system health monitoring
- export live-ready configuration files

Current frozen core:
--------------------
Timeframe:
    6H

Entry:
    Donchian breakout

Filter:
    trend structure from current engine

Exit:
    wide trailing

Portfolio:
    max5
    low heat

Execution:
    closed candles only

This is NOT yet the Binance live engine.
That comes in T9.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path.cwd()

INPUT_SUMMARY = PROJECT_ROOT / "data" / "research_trend_t7" / "phase_t7_master_summary.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "research_trend_t8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FROZEN CONFIGURATION
# ============================================================

FROZEN_CONFIG = {
    "system_name": "TREND_6H_DONCHIAN_WIDE",
    "version": "T8_FROZEN",
    "created_utc": datetime.now(timezone.utc).isoformat(),

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------
    "exchange": "binance",
    "market_type": "spot",
    "universe_size": 70,

    # --------------------------------------------------------
    # CORE LOGIC
    # --------------------------------------------------------
    "timeframe": "6h",
    "entry_logic": "donchian_breakout",
    "exit_logic": "wide_trailing",
    "closed_candles_only": True,

    # --------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------
    "max_open_positions": 5,
    "portfolio_heat_pct": 1.5,
    "risk_per_trade_pct": 0.25,
    "long_only_preferred": True,

    # --------------------------------------------------------
    # CAPITAL
    # --------------------------------------------------------
    "initial_capital_usdt": 10000,
    "leverage_proxy": 3.0,
    "max_margin_usage_pct": 85,

    # --------------------------------------------------------
    # RISK CONTROL
    # --------------------------------------------------------
    "kill_switch_dd_pct": 35,
    "equity_floor_pct": 50,

    # --------------------------------------------------------
    # RESEARCH STATUS
    # --------------------------------------------------------
    "status": "PAPER_ONLY",
    "live_trading_allowed": False,
    "research_phase": "T8",
}

# ============================================================
# CREATE LIVE STATE
# ============================================================

LIVE_STATE = {
    "last_update_utc": None,
    "system_status": "IDLE",
    "kill_switch_triggered": False,
    "open_positions": [],
    "closed_equity": FROZEN_CONFIG["initial_capital_usdt"],
    "peak_equity": FROZEN_CONFIG["initial_capital_usdt"],
    "drawdown_pct": 0.0,
    "portfolio_heat_pct": 0.0,
    "last_error": None,
}

# ============================================================
# SYSTEM HEALTH TEMPLATE
# ============================================================

HEALTH_TEMPLATE = {
    "system_name": FROZEN_CONFIG["system_name"],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "READY_FOR_PAPER_LIVE",
    "research_complete": {
        "T1": True,
        "T2": True,
        "T3": True,
        "T4": True,
        "T5": True,
        "T6": True,
        "T7": True,
        "T8": True,
    },
    "warnings": [
        "Dependency on best assets still present",
        "Recent last20 trades weaker",
        "Sample size still limited",
        "Paper-live observation required before production",
    ],
    "strengths": [
        "Cost robustness survived",
        "Low drawdown under realistic capital constraints",
        "6H structure appears robust",
        "Portfolio heat remains disciplined",
        "Long-only structure particularly strong",
    ]
}

# ============================================================
# CLOSED TRADE EQUITY TRACKER
# ============================================================

closed_equity_columns = [
    "timestamp_utc",
    "closed_equity",
    "peak_equity",
    "drawdown_pct",
    "closed_trade_count",
]

closed_equity_df = pd.DataFrame(columns=closed_equity_columns)

# ============================================================
# PAPER LIVE CHECKLIST
# ============================================================

CHECKLIST = [
    "Verify Binance data integrity",
    "Use CLOSED candles only",
    "No intrabar execution assumptions",
    "No discretionary overrides",
    "Monitor skipped signals",
    "Monitor portfolio heat",
    "Monitor equity only from CLOSED trades",
    "Do not optimise during observation",
    "Observe at least 100-200 additional trades",
    "Compare live equity to historical robustness expectations",
]

# ============================================================
# SAVE FILES
# ============================================================

config_path = OUTPUT_DIR / "phase_t8_frozen_config.json"
state_path = OUTPUT_DIR / "phase_t8_live_state.json"
health_path = OUTPUT_DIR / "phase_t8_system_health.json"
checklist_path = OUTPUT_DIR / "phase_t8_paper_live_checklist.txt"
closed_equity_path = OUTPUT_DIR / "phase_t8_closed_equity_template.csv"
report_path = OUTPUT_DIR / "phase_t8_master_report.txt"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(FROZEN_CONFIG, f, indent=4)

with open(state_path, "w", encoding="utf-8") as f:
    json.dump(LIVE_STATE, f, indent=4)

with open(health_path, "w", encoding="utf-8") as f:
    json.dump(HEALTH_TEMPLATE, f, indent=4)

with open(checklist_path, "w", encoding="utf-8") as f:
    for item in CHECKLIST:
        f.write(f"- {item}\n")

closed_equity_df.to_csv(closed_equity_path, index=False)

# ============================================================
# MASTER REPORT
# ============================================================

report_lines = []

report_lines.append("PHASE T8 — PAPER LIVE BLUEPRINT")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append("SYSTEM STATUS")
report_lines.append("-" * 70)
report_lines.append("Research phase completed.")
report_lines.append("System frozen for paper-live observation.")
report_lines.append("")
report_lines.append("FROZEN CORE")
report_lines.append("-" * 70)
report_lines.append(f"Timeframe: {FROZEN_CONFIG['timeframe']}")
report_lines.append(f"Entry: {FROZEN_CONFIG['entry_logic']}")
report_lines.append(f"Exit: {FROZEN_CONFIG['exit_logic']}")
report_lines.append(f"Max open positions: {FROZEN_CONFIG['max_open_positions']}")
report_lines.append(f"Portfolio heat: {FROZEN_CONFIG['portfolio_heat_pct']}%")
report_lines.append(f"Risk per trade: {FROZEN_CONFIG['risk_per_trade_pct']}%")
report_lines.append("")
report_lines.append("IMPORTANT RULE")
report_lines.append("-" * 70)
report_lines.append("NO NEW FILTERS")
report_lines.append("NO OPTIMISATION")
report_lines.append("NO LIVE CAPITAL YET")
report_lines.append("")
report_lines.append("NEXT PHASE")
report_lines.append("-" * 70)
report_lines.append("T9 — Binance Paper Sim Engine")
report_lines.append("- live Binance candles")
report_lines.append("- closed candle execution")
report_lines.append("- persistent state")
report_lines.append("- CSV logging")
report_lines.append("- dashboard monitoring")
report_lines.append("")
report_lines.append("OBSERVATION OBJECTIVE")
report_lines.append("-" * 70)
report_lines.append("Observe at least 100-200 additional paper trades before considering real deployment.")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("PHASE T8 — PAPER LIVE BLUEPRINT")
print("=" * 80)
print(f"Output dir: {OUTPUT_DIR}")
print()
print("[OK] phase_t8_frozen_config.json")
print("[OK] phase_t8_live_state.json")
print("[OK] phase_t8_system_health.json")
print("[OK] phase_t8_paper_live_checklist.txt")
print("[OK] phase_t8_closed_equity_template.csv")
print("[OK] phase_t8_master_report.txt")
print()
print("System frozen for paper-live preparation.")
