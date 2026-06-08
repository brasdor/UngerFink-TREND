# Memory Consolidation Skill

This skill consolidates the current session's work into the persistent memory system. Run at the end of any session where significant research, architecture changes, or new systems were built.

## When to invoke

- After completing a research phase (T1–T18)
- After adding or modifying a paper engine
- After major architecture changes (new modules, new rules, new pipelines)
- After the user says "consolidate to memory" or "update memory"

## What to consolidate

### Always update:

**`memory/project_trend_system.md`** — T9B Unger-style systems:
- Any new frozen system (T8 freeze date, key params, CAGR/DD/avg_r)
- Research results (T1 outcome: PASS/FAIL, key findings, awaiting review flag)
- Signal Arbitration Manager rule changes
- Paper engine script names and state file paths

**`memory/project_blueprint_m1m7.md`** — M1-M7 cross-sectional system:
- Paper trading rebalancing log (date, positions opened/closed, notable P&L)
- Module status changes
- API key status updates
- Phase 1 pass criteria tracking (0/4 weeks → update count)

**`memory/user_profile.md`** — if Jean expresses new preferences or constraints

**`memory/feedback_*.md`** — if Jean corrects an approach or validates an unusual choice

### Check before writing:

1. Read `MEMORY.md` index to avoid duplicate entries
2. Read the target memory file — update in-place, don't create duplicates
3. If creating a new memory file, add an entry to `MEMORY.md`

## Architecture summary (for context)

Two parallel systems in `C:\Users\Jean\UngerFink-TREND`:

**System A — T9B (Unger-style, GitHub Actions daily):**
- 4 paper engines: DonchianLong (trend), RSI MR Long (MR), ConsecDownDays (MR bull), Momentum Factor (trend futures)
- All coordinate via `signal_arbitrator.py` (5 rules: direction_conflict, duplicate_held, lower_priority_{regime}, heat_caps, momentum_short_conflict)
- Research pipeline: T1 (universe discovery) → T2 (parameter optimisation) → ... → T8 (freeze) → T9 (paper)
- Methodology: Unger Academy — stability zones, avg_r floors, no WFA

**System B — M1-M7 (Blueprint, CryptoTradingDailyPipeline):**
- Cross-sectional factor strategy: 5 factors → weekly ranking → Q5 long / Q1 short
- Modules: M1 data pipeline → M2 factor engine → M3 backtest → M4 risk → M5 execution → M6 monitor → M7 squeeze scanner
- Running via `run_daily.bat` / Windows Task Scheduler task `CryptoTradingDailyPipeline`
- Blueprint document: `data/systematic_crypto_trading_blueprint/Systematic_Crypto_Trading_Blueprint.docx`

## Standard memory file template

```markdown
---
name: <kebab-case-slug>
description: <one-line summary for MEMORY.md index>
metadata:
  type: <user|feedback|project|reference>
---

<body — for feedback: rule + **Why:** + **How to apply:**>
<body — for project: fact/decision + **Why:** + **How to apply:**>
```

## Consolidation checklist

- [ ] Read existing memory files before writing (never overwrite blind)
- [ ] Update `project_trend_system.md` with any new/changed T9B systems or research results
- [ ] Update `project_blueprint_m1m7.md` with latest paper trading log and module status
- [ ] Create new feedback memory if Jean corrected or confirmed an approach
- [ ] Verify `MEMORY.md` index is up to date (max ~200 lines)
- [ ] Commit new data files to git if not already done
