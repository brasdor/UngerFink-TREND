"""Stage 0 power check: how many independent episodes does each data history window contain?"""
import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "riskoff")
ev = pd.read_csv(os.path.join(D, "events.csv"), parse_dates=["start"])
p = ev[ev.primary]
CUT = {"full history": "1900-01-01", "from 2020-09 (Binance vision OI/premium)": "2020-09-01",
       "from 2021-01": "2021-01-01", "from 2023-01": "2023-01-01",
       "from 2023-10-10 (FRED ICE BofA HY OAS)": "2023-10-10"}
rows = []
for k, c in CUT.items():
    for t in ["BTC", "EW_BASKET"]:
        q = p[(p.target == t) & (p.start >= c)]
        rows.append(dict(window=k, target=t, episodes=len(q), ge20pct=int((q.size_pct <= -20).sum()),
                         ge30pct=int((q.size_pct <= -30).sum()), ge40pct=int((q.size_pct <= -40).sum())))
out = pd.DataFrame(rows); out.to_csv(os.path.join(D, "episode_counts_by_window.csv"), index=False)
print(out.to_string(index=False))
# by year, BTC
b = p[p.target == "BTC"]; print(b.groupby(b.start.dt.year).size().to_dict())
# merge-gap sensitivity (reporting only): re-decluster BTC primary flags with 30d gap
lab = pd.read_csv(os.path.join(D, "labels_daily.csv"), index_col=0, parse_dates=True)
for t, col in [("BTC", "BTC_10pct_10d"), ("EW", "EW_BASKET_10pct_10d")]:
    days = lab.index[lab[col] == 1]
    for gap in (15, 30, 45):
        n = 1 + int(((pd.Series(days).diff().dt.days) > gap).sum())
        print(f"{t} merge-gap {gap}d -> {n} episodes")
