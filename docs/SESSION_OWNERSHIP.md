# Concurrent session ownership (as of 2026-10-11)

Two Claude Code sessions have been working this repo at the same time:
this one (health/monitoring/sizing-cap/fail-closed-stale-inputs work) and
another one (exchange-faithful tradability, gated behind
`data/tradability_config.json {"apply_state_fixes": false}`). Neither
session committed over the other's work as of this note, but several
files are modified by both, uncommitted, right now. **Don't commit any
file in the "Contested" list below without reading the live diff first**
-- a plain `git add`/`git commit` on any of them would silently include
the other session's in-progress, unreviewed changes.

## This session owns (committed, pushed, clean)

- `docs/HEALTH_CONTRACT.md`, `engines/health_emit.py`,
  `tools/health/run_checks.py` (as of commit `39b0bbc3` -- see Contested
  below for what's changed on top since)
- `tests/health/test_checks.py`, `tests/health/test_sizing_caps.py`
- `.github/workflows/health_check.yml`, `funding_refresh_runner.yml`
- `docs/FUNDING_HOST_SETUP.md`, this file
- The funding backfill commit and the 43-row `regime_history.csv` rebuild
  (commits `e88f2e17e`, `253b18bc1`)
- `data/t9b_momentum_paper/state.json`'s `kill_switch_triggered` field
- `data/universe/ohlcv_1d/*.csv` (the 2026-10-10 1970-row cleanup)

## Other session owns (uncommitted, in progress)

- `engines/tradability.py`, `tools/tradability_scan.py`,
  `tools/tradability_state_diff.py`, `data/tradability/`,
  `data/tradability_config.json`, `docs/TRADABILITY_STATE_DIFF.md`,
  `tests/health/test_tradability.py`
- `.github/scripts/check_missed_runs.py` (not touched by this session at
  any point)

## Contested -- modified by both, uncommitted right now

Per `git status` at the time of this note:

| File | This session's prior committed work | Other session's uncommitted additions on top |
|---|---|---|
| `engines/phase_t9b_donchian_universev2_paper_engine.py` (S1) | floor+cap, flat-window guard | tradability void/settlement wiring |
| `engines/phase_t9b_meanreversion_paper_engine.py` (S2) | floor+cap fix, RSI-freeze guard | tradability void/settlement wiring |
| `engines/phase_t9b_consecdowndays_paper_engine.py` (S3) | floor+cap, flat-window guard | tradability wiring |
| `engines/phase_t9b_momentum_factor_paper_engine.py` (S5) | momentum-ranking flat-window exclusion | tradability wiring |
| `engines/phase_t9b_rsi_mr_funding_paper_engine.py` (S8) | floor+cap, RSI-freeze guard, funding/OHLCV stale-gates | tradability wiring |
| `phase_t9b_volcontraction_short_paper_engine.py` (S6) | floor+cap, funding/OHLCV stale-gates | tradability wiring |
| `phase_t9b_macross_short_paper_engine.py` (S7) | floor+cap, funding/OHLCV stale-gates | tradability wiring |
| `engines/phase_t9_momentum_candidates_paper_engine.py` (C12/C19) | `position_pct_cap_bound` emit | tradability wiring |
| `tools/health/run_checks.py` | dead-runner check (committed `39b0bbc3`) | further additions on top, uncommitted |
| `tests/health/test_flat_series_guards.py` | 9 flat-series tests (committed `bffe48ddc`) | further additions on top, uncommitted |
| `tools/health/refresh_exchange_status_cache.py` | initial version (committed) | `onboard_date`/`contract_type` fields added |
| `data/exchange_status_cache.json` | initial fetch (committed) | regenerated with the richer schema above |
| `data/health/health_events.jsonl`, `health_report.json` | this session's `emit()` calls | the other session's engine `emit()` calls land in the same append-only file -- genuinely shared, not really "contested" so much as naturally interleaved |

**To reconcile**: once the other session commits its tradability work,
re-run `tools/health/run_checks.py` fresh (it's always safe to regenerate
`health_report.json`/`health_events.jsonl` -- that's its normal daily
behavior) and diff each contested engine file against its last commit
from this session to confirm this session's floor/cap/guard code and the
other session's tradability code didn't silently clobber each other
(they touch different sections of each file -- the sizing block vs. the
entry/exit control flow -- so a clean textual merge is expected, but
verify before trusting it).
