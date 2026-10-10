# Health contract

A stable, machine-readable description of what's wrong with the pipeline,
written by this repo and read by whatever delivers it to a human. This repo
makes no assumption about the consumer -- no Telegram call lives in
`tools/health/` or `engines/health_emit.py`. A separate consumer (currently
being built outside this repo) polls or watches the two files below and
decides how/whether to notify.

## Files

### `data/health/health_report.json`

The current state: what's open right now, plus a per-check summary. Rewritten
in full every run by `tools/health/run_checks.py` -- a consumer only ever
needs to read this one file for "what's wrong today."

```json
{
  "generated_at": "2026-10-10T15:32:00+00:00",
  "checks": {
    "ohlcv_staleness_spot":  {"status": "pass", "n_findings": 0},
    "funding_staleness":     {"status": "fail", "n_findings": 1},
    "stopout_rate_elevated": {"status": "fail", "n_findings": 1}
  },
  "findings": [
    {
      "id": "data/funding_rates:funding_staleness",
      "severity": "CRITICAL",
      "system": "data/funding_rates",
      "check": "funding_staleness",
      "message": "funding_rates: BTCUSDT is 43d behind (checked 290 symbols)",
      "first_seen": "2026-08-31",
      "last_seen": "2026-10-10",
      "status": "open",
      "evidence": {"worst_symbol": "BTCUSDT", "worst_days": 43, "n_checked": 290},
      "suggested_action": "Run the funding refresh on a non-US host; see Phase 2.",
      "run_date": "2026-10-10"
    }
  ]
}
```

### `data/health/health_events.jsonl`

Append-only. One JSON object per line, one of two `event` values:

```json
{"event": "open",    "id": "...", "severity": "...", "system": "...", "check": "...", "message": "...", "evidence": {}, "suggested_action": "...", "run_date": "2026-10-10", "ts": "2026-10-10T15:32:00+00:00"}
{"event": "resolve",  "id": "...", "run_date": "2026-10-10", "ts": "2026-10-10T15:32:00+00:00"}
```

This is the audit trail `health_report.json` is computed from (by replaying
every event for a given id, latest wins) and the thing a consumer can tail
for a live feed instead of diffing the report file. Never rewritten, never
pruned by this codebase.

## Finding schema

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable across runs -- see below. Primary key. |
| `severity` | `CRITICAL` \| `WARNING` \| `INFO` | CRITICAL = exit non-zero, something is actively wrong with live money/data. WARNING = needs attention, not urgent. INFO = informational (e.g. a suppressed symbol was re-verified). |
| `system` | string | What the finding is about -- a system id (`S1`..`S8`, `Candidate12`, `Candidate19`), a shared resource (`data/futures_universe/funding_rates`), or `runner` for a meta-finding about the health framework itself. |
| `check` | string | Stable slug identifying which check produced this. One of the names in `tools/health/run_checks.py`'s registry, or `<name>_crashed` if the check itself raised. |
| `message` | string | One line, human-readable, with the concrete numbers (not just "stale" -- "43d behind"). |
| `first_seen` | `YYYY-MM-DD` | `run_date` of the first "open" event in the current streak (resets to today if a finding with this id was previously resolved and has now reopened). |
| `last_seen` | `YYYY-MM-DD` | `run_date` of the most recent "open" event. Equal to `generated_at`'s date for anything still open as of the latest run. |
| `status` | `open` \| `resolved` | Only `open` findings appear in `health_report.json`'s `findings` list; `resolved` only exists as a historical event in the `.jsonl`. |
| `evidence` | object | Whatever numbers back up `message` -- varies per check, always JSON-serializable. |
| `suggested_action` | string | One line, concrete, pointing at a file/command/decision -- not "investigate further." |
| `run_date` | `YYYY-MM-DD` | The date this specific event pertains to (usually "today" for the runner; may be a past date for an engine-level `emit()` call during a backfill). |

## Stable ids

`id = "{system}:{check}"`, or `"{system}:{check}:{key}"` when one check can
raise more than one finding per system in a single run (e.g. OHLCV staleness
is one check across many symbols -- `key` is the worst symbol's ticker so a
second stale symbol doesn't collide with the first).

Consequences of this scheme, deliberately:
- The same real-world problem always gets the same id across runs, so
  `first_seen`/`last_seen` and open/resolve tracking work.
- An engine-level `emit()` call (see below) and a registry check that
  independently notices the same condition **must** agree on `system` and
  `check` to converge on one finding -- they are not reconciled by content,
  only by id. When adding a new `emit()` call site, check the registry for
  an existing `check` name that already means the same thing before minting
  a new one.
- Renaming a `check` slug breaks the history for every id built from it --
  treat check names as part of this contract, not as free-text.

## Two ways a finding gets opened

1. **The runner** (`tools/health/run_checks.py`), on its own daily schedule,
   independent of whether any engine ran today. Owns `health_report.json`
   end to end: it recomputes every registered check from scratch each run,
   and for any id that one of ITS OWN checks previously opened but did not
   re-open this run, it appends a `resolve` event automatically. This is
   the only place auto-resolution happens.

2. **`engines/health_emit.py`'s `emit()`**, called inline from an engine or
   a consumer script at the exact point it would otherwise have swallowed
   an exception or silently fallen back to a default. These findings are
   **not** auto-resolved by the runner -- there is no registry check re-
   deriving the same condition to compare against. They stay open until the
   same call site calls `resolve()` on its success path, or a human edits
   `health_events.jsonl` by hand. `emit()`/`resolve()` only ever append to
   the `.jsonl`; they never touch `health_report.json` directly, so a
   mid-run engine crash can never leave the report file half-written.

## Exit code

`run_checks.py` exits non-zero if any finding with severity `CRITICAL` is
open after this run's reconciliation. WARNING/INFO findings never fail the
job by themselves. Placement in CI follows from this: see the workflow
docs for why the health runner always comes **after** the state commit, not
before it.
