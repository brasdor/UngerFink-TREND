"""Risk-off Stage 0A: event catalogue. Read-only research; no feature, model or alert.

Rule (fixed, not tuned): day D is risk-off if min(close[D+1..D+h]) <= (1-thr)*close[D].
Forward-looking by construction -> LABEL ONLY. Never import this into a feature pipeline.
Targets: BTC futures daily close; equal-weight basket of the futures universe (daily-rebalanced
mean of simple returns over symbols with a valid close on both days).
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "riskoff")
FUT = os.path.join(ROOT, "data", "futures_universe", "ohlcv_1d")
THRS, HORIZONS, PRIMARY = (0.07, 0.10, 0.15), (5, 10, 20), (0.10, 10)
MERGE_GAP = 15  # flagged days <= 15 days apart belong to the same episode


def load_closes():
    s = {}
    for f in sorted(glob.glob(os.path.join(FUT, "*_1d.csv"))):
        d = pd.read_csv(f, usecols=["timestamp", "close"])
        d.index = pd.to_datetime(d["timestamp"], unit="ms", utc=True).dt.normalize()
        s[os.path.basename(f).replace("_1d.csv", "")] = d["close"][~d.index.duplicated()]
    return pd.DataFrame(s).sort_index()


def basket_index(px):
    r = px.pct_change(fill_method=None)
    r = r.where(px.shift(1).notna() & px.notna())
    # data-integrity guard: a single-day |return| > 80% is almost surely a bad print / redenomination
    bad = int((r.abs() > 0.8).sum().sum())
    r = r.where(r.abs() <= 0.8)
    ew = r.mean(axis=1, skipna=True)
    n = r.notna().sum(axis=1)
    ew = ew.where(n >= 5).dropna()  # need >=5 constituents
    return (1 + ew).cumprod() * 100, n, bad


def forward_min_ratio(c, h):
    """min(close[D+1..D+h]) / close[D]; NaN if the full window is not observed."""
    v = c.values
    out = np.full(len(v), np.nan)
    for i in range(len(v) - h):
        out[i] = v[i + 1:i + h + 1].min() / v[i]
    return pd.Series(out, c.index)


def episodes(flag, c, h):
    days = list(flag.index[flag])
    if not days:
        return []
    groups, cur = [], [days[0]]
    for d in days[1:]:
        if (d - cur[-1]).days <= MERGE_GAP:
            cur.append(d)
        else:
            groups.append(cur); cur = [d]
    groups.append(cur)
    rows = []
    for g in groups:
        s, e = g[0], g[-1]
        win = c.loc[s:e + pd.Timedelta(days=h)]
        tr = win.idxmin()
        rows.append(dict(start=s.date(), last_flag=e.date(), n_flag_days=len(g), trough=tr.date(),
                         size_pct=round((win.min() / c.loc[s] - 1) * 100, 1),
                         days_to_trough=(tr - s).days))
    return rows


def main():
    px = load_closes()
    btc = px["BTCUSDT"].dropna()
    ew, nsym, bad = basket_index(px)
    targets = {"BTC": btc, "EW_BASKET": ew}
    info = {"BTC": (btc.index[0].date(), btc.index[-1].date(), len(btc)),
            "EW_BASKET": (ew.index[0].date(), ew.index[-1].date(), len(ew))}
    labels = pd.DataFrame(index=pd.date_range(btc.index[0], btc.index[-1], freq="D", tz="UTC"))
    labels.index.name = "date"
    ev_rows, base_rows = [], []
    for name, c in targets.items():
        c = c.asfreq("D")  # calendar days; any gap stays NaN (reported)
        gaps = int(c.isna().sum()); c = c.ffill() if gaps == 0 else c.dropna()
        for h in HORIZONS:
            fm = forward_min_ratio(c, h)
            for t in THRS:
                valid = fm.notna()
                flag = (fm <= 1 - t) & valid
                prim = (t, h) == PRIMARY
                key = f"{name}_{int(t*100)}pct_{h}d"
                labels[key] = flag.astype("float").where(valid)  # NaN = window incomplete
                eps = episodes(flag, c, h)
                base_rows.append(dict(target=name, thr=t, horizon=h, primary=prim,
                                      series_start=c.index[0].date(), series_end=c.index[-1].date(),
                                      n_days_labeled=int(valid.sum()), n_flag_days=int(flag.sum()),
                                      base_rate_pct=round(flag.sum() / valid.sum() * 100, 1),
                                      n_episodes=len(eps)))
                for e in eps:
                    ev_rows.append(dict(target=name, thr=t, horizon=h, primary=prim, **e))
    ev = pd.DataFrame(ev_rows); base = pd.DataFrame(base_rows)
    os.makedirs(OUT, exist_ok=True)
    ev.to_csv(os.path.join(OUT, "events.csv"), index=False)
    base.to_csv(os.path.join(OUT, "base_rates.csv"), index=False)
    labels.index = labels.index.strftime("%Y-%m-%d")
    labels.to_csv(os.path.join(OUT, "labels_daily.csv"))
    pd.DataFrame({"btc_close": btc, "ew_index": ew, "n_symbols": nsym}).to_csv(os.path.join(OUT, "target_series.csv"))
    print("series:", info, "| bad-return cells dropped:", bad)
    print(base.to_string(index=False))
    print(ev[ev.primary].to_string(index=False))

if __name__ == "__main__":
    main()
