#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UngerFink Pipeline Agent v2.0
==============================
Claude-powered agent that runs the T1->T8 research pipeline autonomously.
Now supports Mean Reversion systems with adapted gate checks.

Usage
-----
    # Trend following (long-side, Spot)
    python pipeline_agent.py --method DualMA --start-phase T1

    # Trend following (short-side, Futures)
    python pipeline_agent.py --method DonchianShort --start-phase T1 --futures

    # Mean Reversion (different gate checks apply)
    python pipeline_agent.py --method MeanReversionRSI --start-phase T1 --mean-reversion

    # Resume from a specific phase
    python pipeline_agent.py --method MeanReversionRSI --start-phase T4 --mean-reversion

    # Fully autonomous (no human checkpoints)
    python pipeline_agent.py --method MeanReversionRSI --start-phase T1 --mean-reversion --auto

    # Use a custom config JSON
    python pipeline_agent.py --method MeanReversionRSI --config mr_config.json --mean-reversion

Requirements
------------
    pip install anthropic pandas python-dotenv

API key
-------
    Set ANTHROPIC_API_KEY in environment or .env file.

Author
------
    Jean / UngerFink -- pipeline_agent v2.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path.cwd()
MODEL        = "claude-sonnet-4-5"
MAX_TOKENS   = 8192

# ---------- Trend following gate thresholds (Unger methodology) ----------
TF_GATES = {
    "win_rate_min":       0.24,   # crypto-adjusted (structural below 30%)
    "win_rate_max":       0.55,   # above 55% = suspect overfitting
    "avg_r_min_spot":     0.15,   # §4.2 Binance Spot
    "avg_r_min_futures":  0.25,   # §4.2 Binance Futures (funding adjusted)
    "stability_min":      0.67,   # §2.1
    "mc_p05_min":         0.0,    # T16 p05 total R
    "cost_stress_pf_min": 1.0,    # T4 cost stress at 0.05R extra
    "min_trades":         80,     # T5 statistical floor
    "dec2024_conc_max":   0.50,   # T4 December 2024 concentration flag
}

# ---------- Mean Reversion gate thresholds (different from trend following) ----------
MR_GATES = {
    "win_rate_min":       0.50,   # MR typically wins >50% (many small winners)
    "win_rate_max":       0.80,   # above 80% = suspect overfitting or holding losers
    "avg_r_min_spot":     0.10,   # lower bar -- MR trades are shorter, smaller R
    "avg_r_min_futures":  0.15,   # futures adjusted
    "stability_min":      0.67,   # §2.1 -- same
    "mc_p05_min":         0.0,    # T16 p05 total R
    "cost_stress_pf_min": 1.0,    # T4 cost stress
    "min_trades":         100,    # MR needs more trades (shorter holding, more signals)
    "max_dd_r_max":       50.0,   # MR should have tighter DD than trend following
    "avg_hold_days_max":  10.0,   # MR trades should not run too long
}

# ---------- Phase sequences ----------
TF_PHASES = ["T1", "T2", "T3B", "T4", "T5", "T6", "T7", "T8", "T15", "T16", "T18"]
MR_PHASES = ["T1", "T2", "T3MR", "T4", "T5", "T6", "T7", "T8", "T15", "T16"]

# Human review checkpoints
TF_HUMAN_REVIEW = ["T1", "T4", "T8", "T16"]
MR_HUMAN_REVIEW = ["T1", "T4", "T8", "T16"]

# ---------- Script name templates ----------
TF_SCRIPTS = {
    "T1":   "phase_t1_{method}_concept_discovery.py",
    "T2":   "phase_t2_{method}_core_engine.py",
    "T3B":  "phase_t3b_{method}_wide_exit_engineering.py",
    "T4":   "phase_t4_{method}_robustness_engine.py",
    "T5":   "phase_t5_{method}_portfolio_filter.py",
    "T6":   "phase_t6_{method}_capital_execution_engine.py",
    "T7":   "phase_t7_{method}_asset_robustness.py",
    "T8":   "phase_t8_{method}_config_freeze.py",
    "T15":  "phase_t15_{method}_param_stability.py",
    "T16":  "phase_t16_{method}_montecarlo.py",
    "T18":  "phase_t18_{method}_volatility_gate.py",
}

MR_SCRIPTS = {
    "T1":   "phase_t1_{method}_concept_discovery.py",
    "T2":   "phase_t2_{method}_core_engine.py",
    "T3MR": "phase_t3mr_{method}_exit_engineering.py",   # MR exit: RSI neutral / time / profit target
    "T4":   "phase_t4_{method}_robustness_engine.py",
    "T5":   "phase_t5_{method}_portfolio_filter.py",
    "T6":   "phase_t6_{method}_capital_execution_engine.py",
    "T7":   "phase_t7_{method}_asset_robustness.py",
    "T8":   "phase_t8_{method}_config_freeze.py",
    "T15":  "phase_t15_{method}_param_stability.py",
    "T16":  "phase_t16_{method}_montecarlo.py",
}

# ---------- Output directories ----------
PHASE_DIRS = {
    "T1":   "data/research_{method}_t1",
    "T2":   "data/research_{method}_t2",
    "T3B":  "data/research_{method}_t3b",
    "T3MR": "data/research_{method}_t3mr",
    "T4":   "data/research_{method}_t4",
    "T5":   "data/research_{method}_t5",
    "T6":   "data/research_{method}_t6",
    "T7":   "data/research_{method}_t7",
    "T8":   "data/research_{method}_t8",
    "T15":  "data/research_{method}_t15",
    "T16":  "data/research_{method}_t16",
    "T18":  "data/research_{method}_t18",
}

# ---------- Key output files per phase ----------
KEY_FILES = {
    "T1":   ["phase_t1_stability_ranking.csv", "phase_t1_summary.txt"],
    "T2":   ["phase_t2_summary.csv", "phase_t2_asset_summary.csv"],
    "T3B":  ["phase_t3b_wide_exit_summary.csv", "phase_t3b_wide_exit_asset_summary.csv"],
    "T3MR": ["phase_t3mr_exit_summary.csv", "phase_t3mr_exit_asset_summary.csv"],
    "T4":   ["phase_t4_master_report.txt", "phase_t4_montecarlo_summary.csv",
             "phase_t4_remove_best_assets.csv", "phase_t4_cost_stress_summary.csv"],
    "T5":   ["phase_t5_portfolio_summary.csv"],
    "T6":   ["phase_t6_master_report.txt", "phase_t6_variant_summary.csv"],
    "T7":   ["phase_t7_remove_asset_summary.csv"],
    "T8":   ["phase_t8_final_scorecard.txt", "phase_t8_frozen_config.py"],
    "T15":  ["phase_t15_param_stability.csv"],
    "T16":  ["phase_t16_montecarlo_summary.csv"],
    "T18":  ["phase_t18_volatility_gate_summary.csv"],
}

# ---------- Default Mean Reversion config ----------
MR_DEFAULT_CONFIG = {
    "method_name":          "MeanReversionRSI",
    "entry_logic":          "RSI(N) crosses below oversold threshold -- enter long on oversold",
    "exit_logic":           "RSI crosses above exit_rsi_level OR ATR stop OR time exit",
    "rsi_n":                [2, 3, 5, 7, 10, 14],
    "oversold_threshold":   [10, 15, 20, 25, 30],
    "exit_rsi_level":       [50, 55, 60, 65],
    "atr_stop_mults":       [2.0, 3.0, 4.0],
    "time_exit_bars":       [5, 10, 15, 20],
    "filter_modes":         ["none", "ema200_price_above"],
    "timeframes":           ["1d", "4h", "6h"],
    "stability_zone":       "rsi_n +/-1 AND oversold_threshold +/-1 -- PASS if >=67%",
    "asset_universe":       "69-symbol crypto universe from raw_trend_t1 data cache",
    # DATA LOADING -- must match exactly:
    "data_raw_dir":         str(PROJECT_ROOT / "data" / "raw_trend_t1"),
    "data_root":            str(PROJECT_ROOT / "data" / "raw_trend_t1"),
    "filename_format":      "{symbol}_{tf}.csv  -- symbol uses BTC_USDT not BTC/USDT",
    "symbol_list":          [
        "AAVE_USDT","ADA_USDT","ALT_USDT","APT_USDT","ARB_USDT","ARKM_USDT",
        "ASTER_USDT","ATOM_USDT","AVAX_USDT","BCH_USDT","BNB_USDT","BTC_USDT",
        "CHZ_USDT","DASH_USDT","DOGE_USDT","DOT_USDT","EIGEN_USDT","ENA_USDT",
        "ETH_USDT","FET_USDT","FIL_USDT","GRT_USDT","HBAR_USDT","ICP_USDT",
        "INJ_USDT","LINK_USDT","LTC_USDT","NEAR_USDT","ONDO_USDT","ORDI_USDT",
        "PENDLE_USDT","PEPE_USDT","RENDER_USDT","SEI_USDT","SOL_USDT","SUI_USDT",
        "TAO_USDT","TIA_USDT","TON_USDT","TRX_USDT","UNI_USDT","WLD_USDT",
        "XRP_USDT","ZEC_USDT","ZEN_USDT","ATOM_USDT","AVAX_USDT","DOT_USDT",
        "HBAR_USDT","ICP_USDT","INJ_USDT","JTO_USDT","LPT_USDT","MORPHO_USDT",
        "NEAR_USDT","NIL_USDT","PENGU_USDT","SAGA_USDT","SPK_USDT","SUI_USDT",
    ],
    "project_root":         str(PROJECT_ROOT),
    "exchange":             "binance_spot",
    "leverage":             1.0,
    "risk_per_trade":       0.0025,
    "max_concurrent":       [3, 5, 10],
    "backtest_bars_1d":     1500,
    "backtest_bars_4h":     6000,
    "backtest_bars_6h":     4000,
    "side":                 "LONG",
    "cost_floor_r":         0.10,
    "note":                 "Mean reversion uses different gates than trend following -- see MR_GATES",
}


# =============================================================================
# MEAN REVERSION SYSTEM PROMPT
# =============================================================================

MR_SYSTEM_CONTEXT = """
MEAN REVERSION — KEY DIFFERENCES FROM TREND FOLLOWING:

1. Entry logic: buy weakness (oversold), not strength (breakout)
   - RSI < oversold_threshold = entry signal
   - Price has FALLEN recently -- we expect a bounce

2. Exit logic (multiple options, test all):
   - RSI returns to neutral (crosses above exit_rsi_level e.g. 50/55/60)
   - Fixed ATR stop below entry (prevents catastrophic loss on continued downtrend)
   - Time exit: exit after N bars regardless of price
   - Profit target: exit at fixed R (e.g. +1R, +2R)

3. Gate differences from trend following:
   - Win rate: expect 50-70% (many small winners, occasional large losers)
   - avg_r: lower bar (0.10R) -- shorter trades, smaller moves
   - CRITICAL: max drawdown must be tighter -- MR can blow up in sustained trends
   - §4.1 win rate 50-70% for mean reversion (NOT 30-45%)

4. Filter logic:
   - ema200_price_above: only take long MR signals when price is above EMA200
     (avoids catching falling knives in bear markets)
   - ema200_avoid_below: same -- reject signals when price < EMA200
   - none: take all signals regardless of trend (test this too)

5. Critical risk for MR on crypto:
   - Crypto can trend violently -- a mean reversion system that buys weakness
     will be destroyed in a sustained downtrend (2022 bear market)
   - The EMA200 filter is ESSENTIAL for crypto MR -- without it the system
     will have catastrophic drawdowns during bear markets
   - T4 period splits are the critical diagnostic -- check 2022 performance

6. T3MR exit engineering (different from T3B Chandelier):
   - Test: RSI exit only vs ATR stop only vs combined vs time exit
   - Do NOT use Chandelier -- MR trades are short, Chandelier is for fat tails
   - Profit target matters more for MR than for trend following

7. Universe: use the ORIGINAL 60-symbol universe, not the 24-symbol filtered one
   - Mean reversion may work on different symbols than trend following
   - Do not pre-filter by trend following performance
"""


# =============================================================================
# CLAUDE CLIENT
# =============================================================================

def get_claude_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Add to .env file: ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or set manually: set ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(client: anthropic.Anthropic, system: str, prompt: str,
                retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  API error (attempt {attempt+1}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


# =============================================================================
# FILE HELPERS
# =============================================================================

def read_output_files(phase: str, method: str) -> dict[str, str]:
    out_dir = PROJECT_ROOT / PHASE_DIRS[phase].format(method=method.lower())
    contents = {}
    for fname in KEY_FILES.get(phase, []):
        fpath = out_dir / fname
        if fpath.exists():
            try:
                contents[fname] = fpath.read_text(encoding="utf-8")
            except Exception as e:
                contents[fname] = f"[ERROR reading: {e}]"
        else:
            contents[fname] = "[FILE NOT FOUND]"
    return contents


def read_pipeline_doc() -> str:
    for name in ["TREND_FOLLOWING_RESEARCH_PIPELINE.md",
                 "TREND_FOLLOWING_RESEARCH_PIPELINE_2.md",
                 "TREND_FOLLOWING_RESEARCH_PIPELINE_3.md",
                 "TREND_FOLLOWING_RESEARCH_PIPELINE_4.md"]:
        p = PROJECT_ROOT / name
        if p.exists():
            print(f"  Pipeline doc: {name}")
            return p.read_text(encoding="utf-8")
    return "[Pipeline document not found]"


# =============================================================================
# SCRIPT GENERATION
# =============================================================================

def generate_script(client: anthropic.Anthropic, phase: str, method: str,
                    config: dict, pipeline_doc: str,
                    futures: bool, mean_reversion: bool) -> str:

    scripts = MR_SCRIPTS if mean_reversion else TF_SCRIPTS
    script_name = scripts[phase].format(method=method.lower())
    gates = MR_GATES if mean_reversion else TF_GATES
    cost_floor = (gates["avg_r_min_futures"] if futures
                  else gates["avg_r_min_spot"])

    mr_context = MR_SYSTEM_CONTEXT if mean_reversion else ""

    system = textwrap.dedent(f"""
        You are an expert algorithmic trading systems developer.
        You write clean, production-quality Python scripts for backtesting
        trading systems following the Andrea Unger methodology.
        You always follow the pipeline document exactly.
        You never add hardcoded values -- all parameters come from the config.
        You always write complete, runnable scripts with proper error handling.
        Use ASCII only -- no Unicode arrows or special characters (Windows cp1252).
        Output ONLY the Python script code, no explanations, no markdown fences.

        {mr_context}
    """).strip()

    prompt = textwrap.dedent(f"""
        Write the script for Phase {phase} of the UngerFink pipeline.

        Script filename: {script_name}
        Method: {method}
        System type: {"MEAN REVERSION" if mean_reversion else "TREND FOLLOWING"}
        Exchange: {"Binance Futures (USD-M)" if futures else "Binance Spot"}
        Side: {"SHORT" if futures and not mean_reversion else "LONG"}

        CONFIG:
        {json.dumps(config, indent=2)}

        GATE THRESHOLDS FOR THIS SYSTEM TYPE:
        {json.dumps(gates, indent=2)}

        PIPELINE DOCUMENT (follow exactly for phase {phase}):
        {pipeline_doc[:6000]}

        Requirements:
        - Follow the Phase {phase} spec in the pipeline document
        - Input: read from the correct upstream phase output directory
        - Output: write to {PHASE_DIRS[phase].format(method=method.lower())}/
        - Gate checks: implement all relevant gates from the thresholds above
        - §4.2 cost floor: {cost_floor}R
        - Use ASCII only in print statements (no Unicode -- Windows compatibility)
        - Print clear progress so the agent can monitor execution
        - Exit with code 0 on success, code 1 on gate failure
        {"- For T3MR: implement RSI exit, ATR stop, time exit, profit target variants" if phase == "T3MR" else ""}
        {"- For T1 MR: produce stability ranking AND heatmap of avg_r by (rsi_n, oversold_threshold)" if phase == "T1" and mean_reversion else ""}

        CRITICAL DATA LOADING INSTRUCTIONS (mean reversion):
        - RAW data directory: {config.get('data_raw_dir', str(PROJECT_ROOT / 'data' / 'raw_trend_t1'))}
        - Filename format: {{symbol}}_{{tf}}.csv  e.g. BTC_USDT_1d.csv or ETH_USDT_4h.csv
        - Symbols use underscore format: BTC_USDT (NOT BTC/USDT or BTCUSDT)
        - Available timeframes: 1d, 4h, 6h  (NO 1h data exists -- do not use 1h)
        - Use only the symbol_list from CONFIG -- do not scan directory or use ccxt
        - Load CSV with columns: timestamp, open, high, low, close, volume
        - RSI entry condition: use "RSI is BELOW threshold" (not just crossover) to generate more signals
          i.e. enter when rsi < oversold_threshold AND was not in position already (bar-by-bar re-entry allowed)
    """).strip()

    return call_claude(client, system, prompt)


# =============================================================================
# PHASE RUNNER
# =============================================================================

def run_script(script_path: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT, env=env,
        encoding="utf-8", errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


# =============================================================================
# GATE ANALYSIS
# =============================================================================

def analyze_phase(client: anthropic.Anthropic, phase: str, method: str,
                  config: dict, stdout: str, file_contents: dict[str, str],
                  futures: bool, mean_reversion: bool,
                  pipeline_doc: str) -> dict:

    gates = MR_GATES if mean_reversion else TF_GATES
    cost_floor = (gates["avg_r_min_futures"] if futures
                  else gates["avg_r_min_spot"])
    system_type = "MEAN REVERSION" if mean_reversion else "TREND FOLLOWING"

    files_text = "\n\n".join(
        f"=== {fname} ===\n{content}"
        for fname, content in file_contents.items()
    )

    mr_extra = ""
    if mean_reversion:
        mr_extra = f"""
MEAN REVERSION SPECIFIC CHECKS:
- Win rate must be {gates['win_rate_min']*100:.0f}-{gates['win_rate_max']*100:.0f}% (NOT 30-45%)
- avg_r floor is {gates['avg_r_min_spot']}R (lower than trend following)
- Check if EMA200 filter is essential (compare with/without filter performance)
- Check 2022 bear market performance -- MR on crypto without filter = catastrophic
- T4: flag if system loses badly in second half (2024-2026 period)
- T3MR: RSI exit vs time exit vs ATR stop -- which combination is best
"""

    system = textwrap.dedent(f"""
        You are an expert in the Andrea Unger systematic trading methodology.
        You analyze backtesting results and apply gate checks strictly.
        System type being analyzed: {system_type}
        You never waive a gate check. You always give a clear PASS or FAIL.
        You respond only in valid JSON.
        {mr_extra}
    """).strip()

    prompt = textwrap.dedent(f"""
        Analyze Phase {phase} results for method "{method}" ({system_type}).
        Return a JSON object with gate verdicts and recommendations.

        GATE CHECKS:
        - §2.1 Stability: >=67% of stability zone profitable
        - Win rate: {gates['win_rate_min']*100:.0f}-{gates['win_rate_max']*100:.0f}%
        - §4.2 Cost floor: avg_r > {cost_floor}R
        - §4.7 Concentration: flag if top-1 asset > 50% of total R
        {"- MR specific: check EMA200 filter necessity, 2022 bear period, avg hold duration" if mean_reversion else "- TF specific: check Dec 2024 concentration, second-half decay"}

        SCRIPT OUTPUT (last 3000 chars):
        {stdout[-3000:]}

        OUTPUT FILES:
        {files_text[:5000]}

        Return ONLY this JSON (no markdown, no explanation):
        {{
            "phase": "{phase}",
            "method": "{method}",
            "system_type": "{system_type}",
            "gate_pass": true or false,
            "gate_results": {{
                "stability": "PASS/FAIL/NA -- reason",
                "win_rate": "PASS/FAIL/NA -- value",
                "cost_floor": "PASS/FAIL/NA -- value",
                "concentration": "OK/FLAG -- top asset %",
                "system_specific": "any MR or TF specific findings"
            }},
            "canonical_config": {{
                "timeframe": "...",
                "filter_mode": "...",
                "param": "...",
                "atr_mult": "...",
                "notes": "..."
            }},
            "key_findings": ["finding 1", "finding 2", "finding 3"],
            "warnings": ["warning if any"],
            "decision": "PROCEED / HALT / HUMAN_REVIEW_REQUIRED",
            "decision_reason": "one sentence",
            "next_phase_inputs": {{"key": "value"}}
        }}
    """).strip()

    raw = call_claude(client, system, prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "phase": phase, "method": method,
            "gate_pass": False, "gate_results": {},
            "canonical_config": {},
            "key_findings": ["JSON parse error -- manual review needed"],
            "warnings": [raw[:300]],
            "decision": "HUMAN_REVIEW_REQUIRED",
            "decision_reason": "Could not parse Claude response as JSON",
            "next_phase_inputs": {},
        }


# =============================================================================
# HUMAN REVIEW CHECKPOINT
# =============================================================================

def human_review(phase: str, analysis: dict, mean_reversion: bool) -> bool:
    system_type = "MEAN REVERSION" if mean_reversion else "TREND FOLLOWING"
    print("\n" + "=" * 70)
    print(f"  HUMAN REVIEW -- Phase {phase} ({system_type})")
    print("=" * 70)
    print(f"\n  Gate pass : {'YES' if analysis.get('gate_pass') else 'NO'}")
    print(f"  Decision  : {analysis.get('decision', 'UNKNOWN')}")
    print(f"  Reason    : {analysis.get('decision_reason', '')}")

    print("\n  Gate results:")
    for gate, result in analysis.get("gate_results", {}).items():
        print(f"    {gate:25s}: {result}")

    print("\n  Key findings:")
    for f in analysis.get("key_findings", []):
        print(f"    - {f}")

    if analysis.get("warnings"):
        print("\n  Warnings:")
        for w in analysis.get("warnings", []):
            print(f"    ! {w}")

    if analysis.get("canonical_config"):
        cfg = analysis["canonical_config"]
        if any(v for v in cfg.values()):
            print("\n  Canonical config identified:")
            for k, v in cfg.items():
                if v:
                    print(f"    {k}: {v}")

    print("\n" + "-" * 70)
    try:
        response = input("  Proceed to next phase? [y/n]: ").strip().lower()
        return response == "y"
    except EOFError:
        print(f"\n  [NON-INTERACTIVE] Human review required at Phase {phase}.")
        print(f"  Review the results above, then re-run with:")
        print(f"    python pipeline_agent.py --method MeanReversionRSI --mean-reversion --start-phase T2")
        return False


# =============================================================================
# AGENT LOG
# =============================================================================

class AgentLog:
    def __init__(self, method: str):
        self.method = method
        self.log_path = PROJECT_ROOT / f"pipeline_agent_{method.lower()}_log.json"
        self.entries: list[dict] = []
        # Load existing log if resuming
        if self.log_path.exists():
            try:
                self.entries = json.loads(
                    self.log_path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []

    def record(self, phase: str, status: str, analysis: dict, stdout: str):
        self.entries.append({
            "phase": phase, "status": status,
            "analysis": analysis,
            "stdout_tail": stdout[-1500:],
        })
        self.log_path.write_text(
            json.dumps(self.entries, indent=2, default=str),
            encoding="utf-8")

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"  PIPELINE AGENT SUMMARY -- {self.method}")
        print("=" * 70)
        for entry in self.entries:
            icon = "PASS" if entry["status"] == "PASS" else "FAIL"
            print(f"  [{icon}] {entry['phase']:6s} -- "
                  f"{entry['analysis'].get('decision_reason', '')}")
        print(f"\n  Full log: {self.log_path}")


# =============================================================================
# MAIN AGENT LOOP
# =============================================================================

def run_agent(method: str, start_phase: str, config: dict,
              futures: bool, mean_reversion: bool, auto: bool):

    client       = get_claude_client()
    pipeline_doc = read_pipeline_doc()
    log          = AgentLog(method)

    phases         = MR_PHASES if mean_reversion else TF_PHASES
    scripts        = MR_SCRIPTS if mean_reversion else TF_SCRIPTS
    human_reviews  = MR_HUMAN_REVIEW if mean_reversion else TF_HUMAN_REVIEW
    system_type    = "MEAN REVERSION" if mean_reversion else "TREND FOLLOWING"

    print(f"\n{'='*70}")
    print(f"  UngerFink Pipeline Agent v2.0")
    print(f"  Method:      {method}")
    print(f"  System type: {system_type}")
    print(f"  Exchange:    {'Binance Futures' if futures else 'Binance Spot'}")
    print(f"  Start phase: {start_phase}")
    print(f"  Auto mode:   {'ON' if auto else 'OFF'}")
    print(f"{'='*70}\n")

    if start_phase not in phases:
        print(f"  ERROR: phase {start_phase} not in {phases}")
        sys.exit(1)

    phases_to_run    = phases[phases.index(start_phase):]
    canonical_config = config.copy()

    for phase in phases_to_run:
        print(f"\n{'--'*35}")
        print(f"  Phase {phase} -- {method} ({system_type})")
        print(f"{'--'*35}")

        script_name = scripts[phase].format(method=method.lower())
        script_path = PROJECT_ROOT / "research" / script_name

        # Step 1: generate script if missing
        if not script_path.exists():
            print(f"  Script not found: {script_name}")
            print(f"  Asking Claude to generate it...")
            code = generate_script(
                client, phase, method, canonical_config,
                pipeline_doc, futures, mean_reversion)
            # Strip markdown fences if Claude wrapped the code in backticks
            code = code.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                code = "\n".join(lines).strip()
            script_path.write_text(code, encoding="utf-8")
            print(f"  Script written: {script_name}")
        else:
            print(f"  Script exists: {script_name}")

        # Step 2: run the script
        print(f"  Running {script_name}...")
        returncode, stdout, stderr = run_script(script_path)

        if stderr:
            print(f"  stderr (first 500): {stderr[:500]}")
        if returncode != 0:
            print(f"  Script exited with code {returncode}")
            print(f"  stdout tail:\n{stdout[-800:]}")

        # Step 3: read output files
        file_contents = read_output_files(phase, method)
        missing = [f for f, c in file_contents.items()
                   if c == "[FILE NOT FOUND]"]
        if missing:
            print(f"  Missing output files: {missing}")

        # Step 4: Claude analyzes
        print(f"  Claude analyzing {phase} results...")
        analysis = analyze_phase(
            client, phase, method, canonical_config,
            stdout, file_contents, futures, mean_reversion, pipeline_doc)

        # Update canonical config from T1
        if phase == "T1" and analysis.get("canonical_config"):
            canonical_config.update(analysis["canonical_config"])
            print(f"  Canonical config from T1: {canonical_config}")

        # Step 5: human review if needed
        if phase in human_reviews and not auto:
            proceed = human_review(phase, analysis, mean_reversion)
            if not proceed:
                log.record(phase, "HALTED_BY_USER", analysis, stdout)
                print(f"\n  Pipeline halted by user at Phase {phase}.")
                log.print_summary()
                return

        # Step 6: gate decision
        gate_pass = analysis.get("gate_pass", False)
        decision  = analysis.get("decision", "HALT")

        if not gate_pass or decision == "HALT":
            log.record(phase, "FAIL", analysis, stdout)
            print(f"\n  Phase {phase} FAILED.")
            print(f"  Reason: {analysis.get('decision_reason', 'Unknown')}")
            print(f"  Findings:")
            for f in analysis.get("key_findings", []):
                print(f"    - {f}")
            print(f"\n  Pipeline halted. See log: {log.log_path}")
            log.print_summary()
            return

        log.record(phase, "PASS", analysis, stdout)
        print(f"  Phase {phase} PASSED -- {analysis.get('decision_reason', '')}")

    print(f"\n{'='*70}")
    print(f"  Pipeline complete -- {method} ({system_type}) all phases done")
    print(f"{'='*70}")
    log.print_summary()


# =============================================================================
# CLI
# =============================================================================

def build_config(args) -> dict:
    if args.mean_reversion:
        cfg = MR_DEFAULT_CONFIG.copy()
    else:
        cfg = {
            "method_name":    args.method,
            "timeframes":     ["1d"],
            "asset_universe": "60-symbol crypto universe",
            "data_root":      str(PROJECT_ROOT / "data"),
            "project_root":   str(PROJECT_ROOT),
            "exchange":       "binance_futures_usdm" if args.futures else "binance_spot",
            "leverage":       1.0,
            "risk_per_trade": 0.0025,
            "max_concurrent": [3, 5],
            "backtest_bars":  1500,
            "side":           "SHORT" if args.futures else "LONG",
            "cost_floor_r":   0.25 if args.futures else 0.15,
        }
    cfg["method_name"] = args.method
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="UngerFink Claude-powered pipeline agent v2.0")
    parser.add_argument("--method", required=True,
        help="Method name e.g. MeanReversionRSI, DualMA, KeltnerBreakout")
    parser.add_argument("--start-phase", default="T1",
        help="Phase to start from (default: T1)")
    parser.add_argument("--futures", action="store_true",
        help="Binance Futures config (short-side, higher cost floor)")
    parser.add_argument("--mean-reversion", action="store_true",
        help="Use Mean Reversion gate checks and phase sequence")
    parser.add_argument("--auto", action="store_true",
        help="Skip human review checkpoints (fully automated)")
    parser.add_argument("--config", default=None,
        help="Path to JSON config file (overrides defaults)")
    args = parser.parse_args()

    # Validate phase
    phases = MR_PHASES if args.mean_reversion else TF_PHASES
    if args.start_phase not in phases:
        print(f"ERROR: --start-phase must be one of {phases}")
        sys.exit(1)

    config = build_config(args)
    run_agent(
        method=args.method,
        start_phase=args.start_phase,
        config=config,
        futures=args.futures,
        mean_reversion=args.mean_reversion,
        auto=args.auto,
    )


if __name__ == "__main__":
    main()
