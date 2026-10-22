#!/usr/bin/env python3
"""Close findings whose producing check no longer exists or whose condition cleared, by APPENDING 'resolve'
events to data/health/health_events.jsonl (nothing is deleted or edited). Usage:
    python tools/health/resolve_retired_findings.py CHECK [CHECK ...]   # e.g. funding_stale_no_entry suppression_list_stale
Only findings currently open in data/health/health_report.json with one of those check names are resolved."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines"))
import health_emit  # noqa: E402

def main(checks):
    rep = json.loads((ROOT / "data/health/health_report.json").read_text(encoding="utf-8"))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = 0
    for f in rep["findings"]:
        if f["check"] in checks and f.get("status", "open") == "open":
            health_emit._append({"schema_version": health_emit.SCHEMA_VERSION, "event": "resolve", "id": f["id"],
                                 "run_date": today, "ts": datetime.now(timezone.utc).isoformat(),
                                 "note": "closed: producing check retired / condition cleared (resolve_retired_findings.py)"})
            n += 1
    print(f"resolved {n} open finding(s) for checks {sorted(checks)}")

if __name__ == "__main__":
    main(set(sys.argv[1:]))
