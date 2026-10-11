"""Stage 0A sanity check only. Windows are fixed a priori from calendar knowledge; the rule is NOT adjusted."""
import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "riskoff")
lab = pd.read_csv(os.path.join(D, "labels_daily.csv"), index_col=0, parse_dates=True)
ts = pd.read_csv(os.path.join(D, "target_series.csv"), index_col=0, parse_dates=True)
btc = ts["btc_close"]; btc.index = btc.index.tz_localize(None).normalize()
KNOWN = [("May 2021 (China mining ban / 19-May crash)", "2021-05-10", "2021-05-25"),
         ("May 2022 (Terra/LUNA)", "2022-05-04", "2022-05-14"),
         ("Nov 2022 (FTX)", "2022-11-05", "2022-11-11"),
         ("Aug 2024 (yen carry unwind, 5-Aug)", "2024-07-30", "2024-08-06"),
         ("2025-26 decline (6-Oct-25 peak -> Feb-26)", "2025-10-06", "2026-02-06"),
         ("Sep 2026 dip (as seen in our data)", "2026-09-01", "2026-10-08")]
rows = []
for name, a, b in KNOWN:
    seg = btc[a:b]
    mdd = (seg / seg.cummax() - 1).min() * 100
    r = dict(episode=name, window=f"{a}..{b}", btc_max_drawdown_in_window_pct=round(mdd, 1))
    for col in ["BTC_10pct_10d", "EW_BASKET_10pct_10d", "BTC_7pct_10d", "BTC_15pct_10d"]:
        s = lab.loc[a:b, col]
        r[col] = f"{int(s.sum())}/{int(s.notna().sum())} days flagged"
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(D, "known_episodes_check.csv"), index=False)
print(out.to_string(index=False))
