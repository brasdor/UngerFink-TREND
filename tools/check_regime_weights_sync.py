"""
Regime weight sync checker.

engines/regime_daily.py is the PRODUCTION source of truth for the Scheme C
regime weight table (it received the corrected S2/S8 MR-pool split, commit
cb1afe5). Research/backtest scripts carry their own copies of SCHEME_C and
have silently drifted before: the stale S8=0.00 / undivided S2 placeholders
survived in regime_allocator_v4.py and step20 for weeks after the fix.

This script AST-parses (no imports, no side effects) the SCHEME_C literal
and the shared hysteresis constants from every file that defines them and
fails loudly on any mismatch vs regime_daily.py. Run in CI before the
regime step, or standalone:  python tools/check_regime_weights_sync.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFERENCE = "engines/regime_daily.py"
CHECK_FILES = [
    "regime_allocator_v4.py",
    "step20_combined_oos_walkforward.py",
    "step23_combined_equity_replay.py",
]

SHARED_CONSTANTS = [
    "FUNDING_ENTER_HIGH", "FUNDING_EXIT_HIGH",
    "BREADTH_ENTER_BULL", "BREADTH_EXIT_BULL",
    "BREADTH_ENTER_BEAR", "BREADTH_EXIT_BEAR",
    "MIN_HOLD_DAYS",
]

def extract(path):
    """Return (SCHEME_C dict or None, {constant: value}) from module-level literals."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    scheme, consts = None, {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            name = getattr(t, "id", None)
            try:
                if name == "SCHEME_C":
                    scheme = ast.literal_eval(node.value)
                elif name in SHARED_CONSTANTS:
                    consts[name] = ast.literal_eval(node.value)
            except ValueError:
                pass  # non-literal assignment -- not a table we can compare
    return scheme, consts

def main():
    ref_scheme, ref_consts = extract(REFERENCE)
    errors = []

    if ref_scheme is None:
        print(f"FAIL: could not extract SCHEME_C from reference {REFERENCE}")
        return 1

    # Reference self-check (informational only): all consumers normalize the
    # table before use, so a sum != 1.0 is harmless. Known quirk: MIXED sums
    # to 0.875 (7 systems x 0.125) and normalizes to equal-weight 1/7 each.
    # Do NOT "fix" the table to sum to 1.0 -- the +/-0.05 funding tilts are
    # applied pre-normalization, so rescaling would change live behavior.
    for reg, weights in ref_scheme.items():
        total = sum(weights.values())
        if total <= 0:
            errors.append(f"{REFERENCE}: {reg} weights sum to {total} -- normalization would divide by zero")
        elif abs(total - 1.0) > 1e-6:
            print(f"  note: {REFERENCE} {reg} weights sum to {total:g} "
                  f"(normalized downstream -- known quirk, not an error)")

    for path in CHECK_FILES:
        if not (ROOT / path).exists():
            print(f"  note: {path} not found -- skipped")
            continue
        scheme, consts = extract(path)
        if scheme is None:
            print(f"  note: {path} defines no SCHEME_C -- skipped")
            continue
        for reg in ref_scheme:
            if reg not in scheme:
                errors.append(f"{path}: missing regime {reg}")
                continue
            for s, w in ref_scheme[reg].items():
                got = scheme[reg].get(s)
                if got is None or abs(got - w) > 1e-9:
                    errors.append(f"{path}: SCHEME_C[{reg}][{s}] = {got}, "
                                  f"reference ({REFERENCE}) = {w}")
        for name, val in ref_consts.items():
            if name in consts and abs(consts[name] - val) > 1e-12:
                errors.append(f"{path}: {name} = {consts[name]}, "
                              f"reference ({REFERENCE}) = {val}")
        if not any(e.startswith(path) for e in errors):
            print(f"  OK: {path} matches {REFERENCE}")

    if errors:
        print("\nREGIME WEIGHT SYNC FAIL -- stale copies detected:")
        for e in errors:
            print(f"  {e}")
        print(f"\nFix: update the listed files to match {REFERENCE} "
              f"(production source of truth).")
        return 1
    print("\nRegime weight sync: ALL FILES CONSISTENT")
    return 0

if __name__ == "__main__":
    sys.exit(main())
