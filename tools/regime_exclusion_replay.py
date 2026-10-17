#!/usr/bin/env python3
"""Regime classifier replay, both ways: non-tradable symbols INCLUDED (today's behaviour) vs EXCLUDED (switch ON).
Read-only. Re-implements engines/regime_daily.py compute_regime() as a daily loop (same constants, hysteresis,
min-hold and smoothing), validated against data/regime_history.csv over the overlap window.
Tradability by DATE only (onboard/delivery from the exchange cache + the 3-bar flat-feed backstop). Survivor-biased:
the cache holds only currently listed / settling contracts.
Usage: python tools/regime_exclusion_replay.py [start_date]   Output: data/tradability/regime_replay_{include,exclude}.csv
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
import tradability as T  # noqa: E402
import regime_daily as R  # noqa: E402

START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
END = pd.Timestamp("2026-10-10")


def load():
    cl, flat = {}, {}
    for f in glob.glob(str(R.DATA_1D / "*_1d.csv")):
        s = os.path.basename(f).replace("_1d.csv", "")
        d = pd.read_csv(f, usecols=["timestamp", "high", "low", "close", "volume"])
        d.index = pd.to_datetime(d.timestamp, unit="ms").dt.normalize()
        d = d[~d.index.duplicated(keep="last")]
        cl[s] = d.close.astype(float)
        flat[s] = ((d.volume == 0) & (d.high == d.low)).astype(int).rolling(3).sum() == 3
    close = pd.DataFrame(cl).sort_index()
    flatdf = pd.DataFrame(flat).reindex(close.index).fillna(False)
    fund = {}
    for f in glob.glob(str(R.FUNDING_DIR / "*_funding.csv")):
        s = os.path.basename(f).replace("_funding.csv", "")
        d = pd.read_csv(f, usecols=["funding_time", "funding_rate"])
        d.index = pd.to_datetime(d.funding_time, unit="ms")
        fund[s] = d.funding_rate.groupby(level=0).last().sort_index()
    btc = pd.read_csv(R.DATA_1D / "BTCUSDT_1d.csv")
    btc.index = pd.to_datetime(btc.timestamp, unit="ms").dt.normalize()
    return close, flatdf, fund, btc[~btc.index.duplicated(keep="last")]


def run():
    close, flatdf, fund, btc = load()
    syms = list(close.columns)
    ema = close.ewm(span=200, adjust=False).mean()
    valid = close.notna().cumsum() >= 200
    above = (close > ema) & valid
    days = pd.date_range(START, END)
    tr = pd.DataFrame(True, index=days, columns=syms)
    for s in syms:
        i = T.info(s)
        col = pd.Series(True, index=days)
        if i.delivery is not None:
            col[col.index >= pd.Timestamp(i.delivery)] = False
        if i.onboard is not None:
            col[col.index < pd.Timestamp(i.onboard)] = False
        tr[s] = col & ~flatdf[s].reindex(days).fillna(False).astype(bool)
    bc = btc.close.astype(float)
    hi, lo = btc.high.astype(float), btc.low.astype(float)
    bema = bc.ewm(span=200, adjust=False).mean()
    out = {}
    for mode in ("include", "exclude"):
        trend, freg = "MIXED", "NEUTRAL"
        lts = lfs = pd.Timestamp("2020-01-01")
        bs = fs = None
        rows = []
        for D in days:
            mask = tr.loc[D] if mode == "exclude" else pd.Series(True, index=syms)
            m = mask & valid.loc[D]
            raw_b = (above.loc[D][m].sum() / m.sum() * 100) if m.sum() else 50
            cutoff = D + pd.Timedelta(days=1)
            fr = []
            for s in syms:
                if not mask[s] or s not in fund:
                    continue
                x = fund[s][fund[s].index < cutoff]
                if len(x):
                    fr.append(x.iloc[-1])
            raw_f = float(np.mean(fr)) if fr else 0.0001
            bs = raw_b if bs is None else bs * 0.5 + raw_b * 0.5
            fs = raw_f if fs is None else fs * (1 - 2 / 6) + raw_f * (2 / 6)
            above_btc = bool(bc.loc[D] > bema.loc[D]) if D in bc.index else False
            if (D - lts).days >= R.MIN_HOLD_DAYS:
                if trend != "BULL" and above_btc and bs > R.BREADTH_ENTER_BULL:
                    new = "BULL"
                elif trend == "BULL" and (not above_btc or bs < R.BREADTH_EXIT_BULL):
                    new = "MIXED"
                elif trend != "BEAR" and not above_btc and bs < R.BREADTH_ENTER_BEAR:
                    new = "BEAR"
                elif trend == "BEAR" and (above_btc or bs > R.BREADTH_EXIT_BEAR):
                    new = "MIXED"
                else:
                    new = trend
                if new != trend:
                    trend, lts = new, D
            if (D - lfs).days >= R.MIN_HOLD_DAYS:
                if freg != "HIGH" and fs > R.FUNDING_ENTER_HIGH:
                    nf = "HIGH"
                elif freg == "HIGH" and fs < R.FUNDING_EXIT_HIGH:
                    nf = "NEUTRAL"
                elif freg != "LOW" and fs < 0.00005:
                    nf = "LOW"
                elif freg == "LOW" and fs > R.FUNDING_EXIT_HIGH:
                    nf = "NEUTRAL"
                else:
                    nf = freg
                if nf != freg:
                    freg, lfs = nf, D
            base = dict(R.SCHEME_C[trend])
            if freg == "HIGH":
                for k in ("S6", "S7"):
                    base[k] += 0.05
                for k in ("S1", "S5"):
                    base[k] = max(0, base[k] - 0.05)
            elif freg == "LOW":
                for k in ("S2", "S3"):
                    base[k] += 0.05
                for k in ("S6", "S7"):
                    base[k] = max(0, base[k] - 0.05)
            tot = sum(base.values())
            base = {k: round(x / tot, 4) for k, x in base.items()}
            rows.append(dict(date=D, breadth_raw=raw_b, breadth_ema=bs, funding_raw=raw_f, funding_ema=fs, trend=trend,
                             funding=freg, n_breadth=int(m.sum()), n_untradable=int((~tr.loc[D]).sum()),
                             **{f"w_{k}": v for k, v in base.items()}))
        out[mode] = pd.DataFrame(rows).set_index("date")
    return out


if __name__ == "__main__":
    o = run()
    d = ROOT / "data" / "tradability"
    for k, v in o.items():
        v.to_csv(d / f"regime_replay_{k}.csv")
    a, b = o["include"], o["exclude"]
    h = pd.read_csv(ROOT / "data" / "regime_history.csv", parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")
    j = a.join(h, how="inner", rsuffix="_h")
    print("VALIDATION include-replay vs regime_history.csv:", len(j), "days | trend label match",
          round((j.trend == j.trend_h).mean(), 4), "| funding label match", round((j.funding == j.funding_h).mean(), 4),
          "| mean |breadth diff|", round((j.breadth_raw - j.breadth).abs().mean(), 2))
    ct, cf = (a.trend != b.trend), (a.funding != b.funding)
    wc = [c for c in a.columns if c.startswith("w_")]
    dw = (a[wc] - b[wc]).abs().max(axis=1)
    print("days", len(a), "| trend label differs", int(ct.sum()), "| funding label differs", int(cf.sum()), "| any weight differs", int((dw > 1e-9).sum()))
    print("breadth_raw diff (excl-incl): mean", round((b.breadth_raw - a.breadth_raw).mean(), 3), "max abs", round((b.breadth_raw - a.breadth_raw).abs().max(), 2))
    print("funding_raw diff mean", float((b.funding_raw - a.funding_raw).mean()), "| untradable symbols: min", int(b.n_untradable.min()), "max", int(b.n_untradable.max()))
    print("trend switches incl/excl:", int((a.trend != a.trend.shift()).sum() - 1), int((b.trend != b.trend.shift()).sum() - 1),
          "| funding switches:", int((a.funding != a.funding.shift()).sum() - 1), int((b.funding != b.funding.shift()).sum() - 1))
    print(pd.DataFrame({"trend_diff": ct, "fund_diff": cf, "weight_diff": dw > 1e-9}).groupby(a.index.year).sum())
    print("occupancy include:", a.trend.value_counts(normalize=True).round(3).to_dict(), a.funding.value_counts(normalize=True).round(3).to_dict())
    print("occupancy exclude:", b.trend.value_counts(normalize=True).round(3).to_dict(), b.funding.value_counts(normalize=True).round(3).to_dict())
    if ct.any() or cf.any():
        cols = ["trend", "funding", "breadth_ema", "funding_ema"]
        print(pd.concat([a[cols].add_suffix("_inc"), b[cols].add_suffix("_exc")], axis=1)[ct | cf].head(40).to_string())
