#!/usr/bin/env python3
"""DRY RUN. Exact live-state changes the exchange-tradability rules require, per engine. Writes nothing to state.
Output: docs/TRADABILITY_STATE_DIFF.md (also printed)."""
import json, sys
from datetime import date
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
import tradability as T

TODAY = date(2026, 10, 11)
SRC = [("S1","t9b_paper","spot",("open_positions",)),("S2","t9b_mr_paper","futures",("open_positions",)),
       ("S3","t9b_consecdowndays_paper","spot",("open_positions",)),("S5","t9b_momentum_paper","futures",("long_positions","short_positions")),
       ("S6","t9b_volcontraction_paper","futures",("open_positions",)),("S7","t9b_macross_paper","futures",("open_positions",)),
       ("S8","t9b_rsi_mr_funding_paper","futures",("open_positions",)),("C12","t9_candidate12_paper","futures",("open_positions",)),
       ("C19","t9_candidate19_paper","futures",("open_positions",))]
out = ["# Exchange-tradability: exact live-state diff (DRY RUN, nothing applied)", "",
       f"Computed {TODAY} against data/exchange_status_cache.json. Apply by setting `apply_state_fixes` to true in "
       "`data/tradability_config.json`; the engines then perform exactly these changes on their next run.", ""]
tot = 0
for eng, d, venue, keys in SRC:
    st = json.loads((ROOT/"data"/d/"state.json").read_text(encoding="utf-8"))
    rows = []
    for k in keys:
        for p in st.get(k, []):
            a = T.position_action(p["symbol"], p["entry_date"], TODAY, venue)
            if a.kind != "none":
                rows.append((k, p, a))
    n_open = sum(len(st.get(k, [])) for k in keys)
    out.append(f"## {eng} ({d}) - {n_open} open, {len(rows)} affected")
    if not rows:
        out += ["", "No change.", ""]; continue
    out += ["", "| list | symbol | entry | delivery | action | reason | settlement px | entry px | qty | P&L effect |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, p, a in rows:
        px = a.price
        pnl = "0 (void: no P&L booked)" if a.kind == "void" else "settle"
        out.append(f"| {k} | {p['symbol']} | {p['entry_date']} | {a.delivery} | {a.kind.upper()} | {a.reason} | {px if px else ''} | "
                   f"{p.get('entry_price')} | {p.get('qty', p.get('quantity'))} | {pnl} |")
    out += ["", f"Equity `{st.get('paper_equity_usdt', st.get('closed_equity_usdt'))}` unchanged by voids; `closed_trade_count` "
            f"`{st.get('closed_trade_count', 'n/a')}` unchanged (a void is not a trade). Positions removed from state:", "", "```json"]
    for k, p, a in rows:
        out.append(json.dumps({k: p}, indent=1))
    out += ["```", ""]
    tot += len(rows)
out.insert(3, f"**{tot} open positions affected in total.**\n")
(ROOT/"docs"/"TRADABILITY_STATE_DIFF.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[:80]))
