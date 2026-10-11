"""Risk-off Stage 1 - runs exactly the tests in docs/RISKOFF_PREREG.md (commit 12e10e2e). No variable combination, no flag.

Outputs (data/riskoff/): stage1_results.csv, stage1_loeo.csv, stage1_lead_times.csv, stage1_event_results.csv,
stage1_event_paths.csv, stage1_features_daily.csv (PIT features, for audit).
"""
import glob, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "riskoff", "raw")
OUT = os.path.join(ROOT, "data", "riskoff")
FUT = os.path.join(ROOT, "data", "futures_universe", "ohlcv_1d")
B_BOOT, B_PERM, SEED = 2000, 5000, 20261011
LAST_LABEL = pd.Timestamp("2026-09-28")
H7_END = pd.Timestamp("2026-08-28")
SPLIT = pd.Timestamp("2022-01-01")
MIN_TRAIN, M_SMOOTH, REFIT = 365, 20, 30


# ------------------------------------------------------------------ data
def day_index(ms):
    return pd.DatetimeIndex(pd.to_datetime(ms, unit="ms")).normalize()


def load_fut():
    cl, ohlc = {}, None
    for f in sorted(glob.glob(os.path.join(FUT, "*_1d.csv"))):
        d = pd.read_csv(f)
        d.index = day_index(d["timestamp"]); d = d[~d.index.duplicated(keep="last")].sort_index()
        sym = os.path.basename(f).replace("_1d.csv", "")
        cl[sym] = d["close"].astype(float)
        if sym == "BTCUSDT":
            ohlc = d[["high", "low", "close"]].astype(float)
    return pd.DataFrame(cl).sort_index(), ohlc


def raw(name):
    d = pd.read_csv(os.path.join(RAW, name)); d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["value"].sort_index()


def lag_to_cal(s, L, cal):
    """observation dated t becomes usable at t+L, forward-filled to calendar days."""
    s = s.dropna().copy(); s.index = s.index + pd.Timedelta(days=L)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(cal)).ffill().reindex(cal)


def build(ext=False):
    px, btc = load_fut()
    close = px["BTCUSDT"].dropna()
    cal = pd.date_range("2019-09-08", "2026-10-08", freq="D")
    close = close.reindex(cal)
    y_dn = pd.read_csv(os.path.join(OUT, "labels_daily.csv"), index_col=0, parse_dates=True)["BTC_10pct_10d"].reindex(cal)
    fwd_max = pd.Series([close.iloc[i + 1:i + 11].max() / close.iloc[i] if i + 10 < len(close) else np.nan for i in range(len(close))], cal)
    y_up = (fwd_max >= 1.10).astype(float).where(fwd_max.notna())
    y_dn[y_dn.index > LAST_LABEL] = np.nan; y_up[y_up.index > LAST_LABEL] = np.nan

    F, sign = {}, {}
    vix = raw("fred_VIXCLS.csv"); F["H4-VIX"] = lag_to_cal(np.log(vix) - np.log(vix.rolling(60).mean()), 1, cal); sign["H4-VIX"] = +1
    ndq = raw("fred_NASDAQCOM.csv"); F["H4-NDQ"] = lag_to_cal(ndq / ndq.rolling(20).max() - 1, 1, cal); sign["H4-NDQ"] = -1
    jpy = raw("ecb_USDJPY.csv"); F["H4-JPY"] = lag_to_cal(np.log(jpy / jpy.shift(10)), 1, cal); sign["H4-JPY"] = -1
    baa = raw("fred_BAA10Y.csv"); F["H4-BAA"] = lag_to_cal(baa - baa.rolling(60).mean(), 2, cal); sign["H4-BAA"] = +1

    m = pd.read_csv(os.path.join(RAW, "bv_btc_metrics_daily.csv"), parse_dates=["date"]).set_index("date")
    oi = m["oi_coin"].where(m["oi_coin"] > 0).reindex(pd.date_range(m.index[0], m.index[-1])).ffill(limit=3)
    F["H7-OI"] = lag_to_cal(np.log(oi) - np.log(oi.rolling(60).mean()), 1, cal); sign["H7-OI"] = +1
    pr = pd.read_csv(os.path.join(RAW, "bv_btc_premium_daily.csv"), parse_dates=["date"]).set_index("date")["close"]
    pr = pr.reindex(pd.date_range(pr.index[0], pr.index[-1])).ffill(limit=3)
    F["H7-BAS"] = lag_to_cal(pr.rolling(7).mean(), 1, cal); sign["H7-BAS"] = +1
    fd = pd.read_csv(os.path.join(ROOT, "data", "futures_universe", "funding_rates", "BTCUSDT_funding.csv"))
    fd["d"] = day_index(fd["funding_time"]).values; fdd = fd.groupby("d")["funding_rate"].agg(["sum", "count"])
    fdd = fdd[fdd["count"] >= 3]  # complete days only (3 settlements)
    fdd = fdd.reindex(pd.date_range(fdd.index[0], fdd.index[-1]))
    f3 = (fdd["sum"].rolling(3).sum() / fdd["count"].rolling(3).sum())
    F["H7-FUND"] = lag_to_cal(f3, 1, cal); sign["H7-FUND"] = +1

    # breadth: replicate regime_allocator_v4 on the 290 futures closes
    ema = px.ewm(span=200, min_periods=20, adjust=False).mean()
    above = (px > ema).sum(axis=1); tot = px.notna().sum(axis=1)
    rb = (above / tot * 100).fillna(50)
    b3 = rb.ewm(span=3, adjust=False).mean()
    b3 = b3.reindex(cal)
    nsym = tot.reindex(cal)
    brd = b3 - b3.shift(10)
    F["H8-BRD"] = brd.where(nsym >= 20); sign["H8-BRD"] = -1
    eb = np.log(px["ETHUSDT"] / px["BTCUSDT"]).reindex(cal)
    F["H8-ETHBTC"] = eb - eb.shift(10); sign["H8-ETHBTC"] = -1

    # baseline B1-B3 (own data, L=0)
    ema200 = btc["close"].ewm(span=200, adjust=False).mean()
    B1 = np.log(btc["close"] / ema200)
    tr = pd.concat([btc.high - btc.low, (btc.high - btc.close.shift()).abs(), (btc.low - btc.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    B3 = atr.rolling(252).apply(lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False)
    BASE = pd.DataFrame({"B1": B1, "B2": b3, "B3": B3}).reindex(cal)

    # event-study series
    calE = pd.date_range("2017-12-01", "2026-10-08", freq="D") if ext else cal
    hr = raw("bc_hash-rate.csv"); rev = raw("bc_miners-revenue.csv"); mp = raw("bc_market-price.csv")
    ix = hr.index.intersection(rev.index).intersection(mp.index)
    hr, rev, mp = hr[ix], rev[ix], mp[ix]
    ok = (hr > 0) & (rev > 0) & (mp > 0)
    hp = ((rev / mp) / hr).where(ok)
    E, esign = {}, {}
    hp7 = hp.rolling(7).mean(); E["H1-HP"] = lag_to_cal(np.log(hp7 / hp7.shift(30)), 1, calE); esign["H1-HP"] = -1
    hr7 = hr.where(ok).rolling(7).mean(); E["H1-HR"] = lag_to_cal(np.log(hr7 / hr7.shift(30)), 1, calE); esign["H1-HR"] = -1
    wa, tg, rr = raw("fred_WALCL.csv"), raw("fred_WTREGEN.csv"), raw("fred_RRPONTSYD.csv")
    rr_w = rr.reindex(rr.index.union(wa.index)).ffill().reindex(wa.index).fillna(0)
    nl = wa - tg.reindex(wa.index).ffill() - 1000 * rr_w
    E["H2-NL"] = lag_to_cal(np.log(nl / nl.shift(13)), 2, calE); esign["H2-NL"] = -1
    ry = raw("fred_DFII10.csv"); E["H2-RY"] = lag_to_cal(ry - ry.shift(10), 2, calE); esign["H2-RY"] = +1
    us = raw("fred_DTWEXBGS.csv"); E["H2-USD"] = lag_to_cal(us.pct_change(20), 8, calE); esign["H2-USD"] = +1
    st = raw("llama_total.csv"); E["H3-STB"] = lag_to_cal(st.pct_change(30), 1, calE); esign["H3-STB"] = -1
    ut = raw("llama_USDT.csv"); E["H3-USDT"] = lag_to_cal(ut.pct_change(30), 1, calE); esign["H3-USDT"] = -1
    return cal, close, y_dn, y_up, pd.DataFrame(F), sign, BASE, pd.DataFrame(E), esign


# ------------------------------------------------------------------ stats helpers
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def episodes(flag_dates, gap=30):
    """flag_dates: sorted DatetimeIndex of flagged days -> list of (start, last)"""
    out, cur = [], []
    for d in flag_dates:
        if cur and (d - cur[-1]).days > gap:
            out.append((cur[0], cur[-1])); cur = []
        cur.append(d)
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def trough_of(close, start, last):
    w = close.loc[start:last + pd.Timedelta(days=10)]
    return w.idxmin()


def clusters_for(days, y, gap=30):
    fl = days[(y == 1)]
    starts = [e[0] for e in episodes(fl, gap)]
    cid = np.searchsorted(np.array(starts, dtype="datetime64[ns]"), days.values, side="right") if starts else np.zeros(len(days), int)
    return cid, starts


def boot_stat(cid, fn, rng, B=B_BOOT):
    """cluster bootstrap of fn(index_array)->float"""
    ids = np.unique(cid); groups = [np.where(cid == c)[0] for c in ids]
    res = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        res[b] = fn(np.concatenate([groups[i] for i in pick]))
    return res


def bh(p, q=0.10):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, p[i] * n / rank); adj[i] = prev
    return adj


def walk_forward_brier(days, s, y):
    """returns p (NaN where not evaluable) and base (expanding base rate), per prereg 6.1"""
    n = len(days); p = np.full(n, np.nan); base = np.full(n, np.nan)
    sv = np.where(np.isnan(s), np.nan, s)
    for i in range(n):
        if np.isnan(sv[i]):
            continue
        past = sv[:i + 1]; past = past[~np.isnan(past)]
        tr_idx = np.arange(0, max(i - 10, 0))  # j <= i-11
        tr_idx = tr_idx[~np.isnan(sv[tr_idx]) & ~np.isnan(y[tr_idx])]
        if len(tr_idx) < MIN_TRAIN:
            continue
        edges = np.quantile(past, [0.2, 0.4, 0.6, 0.8])
        b_i = np.searchsorted(edges, sv[i], side="right"); b_tr = np.searchsorted(edges, sv[tr_idx], side="right")
        yt = y[tr_idx]; bs = yt.mean(); k = yt[b_tr == b_i].sum(); nn = (b_tr == b_i).sum()
        p[i] = (k + M_SMOOTH * bs) / (nn + M_SMOOTH); base[i] = bs
    return p, base


def brier_skill(idx, p, base, y):
    yy, pp, bb = y[idx], p[idx], base[idx]
    return 1 - np.mean((pp - yy) ** 2) / np.mean((bb - yy) ** 2)


def incremental(days, Bm, cand, y):
    """walk-forward logistic A (B1-B3) vs B (A + cand). returns pA, pB, base arrays (NaN if not evaluable)."""
    n = len(days); pA = np.full(n, np.nan); pB = np.full(n, np.nan); base = np.full(n, np.nan)
    XA = Bm.values; XB = np.column_stack([XA, cand])
    okA = ~np.isnan(XA).any(axis=1); okB = okA & ~np.isnan(cand)
    i = 0
    while i < n:
        tr = np.arange(0, max(i - 10, 0)); tr = tr[okB[tr] & ~np.isnan(y[tr])]
        te = np.arange(i, min(i + REFIT, n)); te = te[okB[te]]
        if len(tr) >= MIN_TRAIN and len(te) > 0 and y[tr].sum() > 5:
            for X, out in ((XA, pA), (XB, pB)):
                sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
                out[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
            base[te] = y[tr].mean()
        i += REFIT
    return pA, pB, base


def lead_time(days, s, ep_list, close):
    """prereg section 7"""
    n = len(days); alarm = np.zeros(n, bool); valid = np.zeros(n, bool)
    for i in range(n):
        if np.isnan(s[i]):
            continue
        past = s[:i + 1]; past = past[~np.isnan(past)]
        if len(past) < MIN_TRAIN:
            continue
        valid[i] = True; alarm[i] = s[i] >= np.quantile(past, 0.8)
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for st, tr, size in ep_list:
        if st not in pos or not valid[pos[st]]:
            continue
        i = pos[st]; lo = max(i - 10, 0)
        hits = [j for j in range(lo, i + 1) if valid[j] and alarm[j]]
        if not hits:
            rows.append(dict(start=st.date(), hit=0, lead_days=np.nan, size_pct=size)); continue
        j = hits[0]; onset = j
        while onset - 1 >= 0 and valid[onset - 1] and alarm[onset - 1] and i - (onset - 1) <= 60:
            onset -= 1
        rows.append(dict(start=st.date(), hit=1, lead_days=min(i - onset, 60), size_pct=size))
    # reference: share of valid days with an alarm in the trailing 11 days
    al = pd.Series(alarm & valid).astype(float).rolling(11, min_periods=1).max().values
    ref = al[valid].mean() if valid.any() else np.nan
    return pd.DataFrame(rows), ref


# ------------------------------------------------------------------ main
def event_study(E, esign, y_dn, y_up, rng, tag):
    # ---------------- event study (H1-H3)
    erows, paths = [], []
    for vid in E.columns:
        x = E[vid]
        mu = x.expanding(min_periods=MIN_TRAIN).mean(); sd = x.expanding(min_periods=MIN_TRAIN).std()
        z = ((x - mu) / sd).shift(0) * esign[vid]  # oriented for DOWN; expanding stats include day D value only (PIT-lagged series)
        for lab, ylab in (("DOWN", y_dn), ("UP", y_up)):
            zz = z if lab == "DOWN" else -z
            valid = zz.notna() & (zz.index <= LAST_LABEL)
            if valid.sum() < 500:
                erows.append(dict(var=vid, label=lab, note="insufficient")); continue
            w = zz[valid]; first = w.index[0]
            yy = ylab.reindex(w.index)
            starts = [e[0] for e in episodes(yy.index[yy == 1], 30)]
            starts = [s for s in starts if (s - first).days >= 10]
            if len(starts) < 3:
                erows.append(dict(var=vid, label=lab, n_episodes=len(starts), note="too few")); continue
            arr = w.values; N = len(arr); pos = np.array([w.index.get_loc(s) for s in starts])
            def T(ix):
                return np.mean([np.nanmean(arr[(i + np.arange(-10, 1)) % N]) for i in ix])
            t_obs = T(pos)
            null = np.array([T((pos + rng.integers(1, N)) % N) for _ in range(B_PERM)])
            p = (1 + np.sum(null >= t_obs)) / (B_PERM + 1)
            erows.append(dict(var=vid, label=lab, window_start=first.date(), n_days=N, n_episodes=len(starts), T_obs=round(t_obs, 4),
                              null_mean=round(null.mean(), 4), null_sd=round(null.std(), 4), p_one_sided=round(p, 4)))
            for off in range(-30, 11):
                v = [arr[(i + off) % N] for i in pos if 0 <= i + off < N]
                paths.append(dict(var=vid, label=lab, offset=off, mean_oriented_z=round(float(np.nanmean(v)), 4), n=len(v)))
    ER = pd.DataFrame(erows)
    for lab in ("DOWN", "UP"):
        m = (ER.label == lab) & ER.p_one_sided.notna()
        ER.loc[m, "q_bh"] = bh(ER.loc[m, "p_one_sided"].values).round(4)
    ER.to_csv(os.path.join(OUT, f"stage1_event_results{tag}.csv"), index=False)
    pd.DataFrame(paths).to_csv(os.path.join(OUT, f"stage1_event_paths{tag}.csv"), index=False)
    print(ER.to_string())



def main():
    rng = np.random.default_rng(SEED)
    cal, close, y_dn, y_up, F, sign, BASE, E, esign = build()
    F.to_csv(os.path.join(OUT, "stage1_features_daily.csv"))
    ev = pd.read_csv(os.path.join(OUT, "events.csv"), parse_dates=["start"])
    prim = ev[(ev.target == "BTC") & ev.primary]
    dn15 = [(r.start.tz_localize(None).normalize() if r.start.tzinfo else r.start, None, r.size_pct) for r in prim.itertuples()]
    # up episodes at 15d from y_up
    up_fl = y_up.index[y_up == 1]
    up15 = []
    for st, la in episodes(up_fl, 15):
        w = close.loc[st:la + pd.Timedelta(days=10)]
        up15.append((st, None, round((w.max() / close.loc[st] - 1) * 100, 1)))

    windows = {"H4": (cal[0], LAST_LABEL), "H8": (cal[0], LAST_LABEL), "H7": (pd.Timestamp("2020-09-01"), H7_END)}
    rows, loeo_rows, lead_rows = [], [], []
    for vid in F.columns:
        grp = vid[:2]; w0, w1 = windows[grp]
        for lab, ylab, eps15 in (("DOWN", y_dn, dn15), ("UP", y_up, up15)):
            sgn = sign[vid] * (1 if lab == "DOWN" else -1)  # UP: opposite of DOWN expected sign
            sc = (sign[vid] * F[vid]).reindex(cal)  # score oriented for the DOWN hypothesis
            s_or = sc if lab == "DOWN" else -sc  # oriented for this label's expected sign
            msk = (cal >= w0) & (cal <= w1)
            days = cal[msk]; s = s_or[msk].values; y = ylab[msk].values
            have = ~np.isnan(s) & ~np.isnan(y)
            d_, s_, y_ = days[have], s[have], y[have]
            if len(d_) < 400 or y_.sum() < 10:
                rows.append(dict(var=vid, label=lab, n_days=len(d_), note="insufficient sample")); continue
            a = auc(s_, y_)
            cid, starts30 = clusters_for(d_, y_, 30)
            # AUC cluster bootstrap
            bA = boot_stat(cid, lambda ix: auc(s_[ix], y_[ix]), rng)
            pval = (1 + np.sum(bA <= 0.5)) / (B_BOOT + 1)
            # walk-forward Brier (on full window arrays so that day offsets are right)
            p_all, base_all = walk_forward_brier(days, np.where(have, s, np.nan), np.where(have, y, np.nan))
            ev_ok = ~np.isnan(p_all) & have
            sub = np.where(ev_ok[have])[0]
            if len(sub) > 100:
                p_ = p_all[have]; b_ = base_all[have]
                bs = brier_skill(sub, p_, b_, y_)
                cid_s = cid[sub]
                bB = boot_stat(cid_s, lambda ix: brier_skill(sub[ix], p_, b_, y_), rng)
                bs_lo, bs_hi = np.percentile(bB, [2.5, 97.5])
                a_wf = auc(s_[sub], y_[sub])
            else:
                bs = bs_lo = bs_hi = a_wf = np.nan
            # subperiods
            pre = d_ < SPLIT
            a_pre = auc(s_[pre], y_[pre]); a_post = auc(s_[~pre], y_[~pre])
            n_pre = sum(1 for st in starts30 if st < SPLIT); n_post = len(starts30) - n_pre
            # LOEO
            eps30 = episodes(d_[y_ == 1], 30)
            lo = []
            for st, la in eps30:
                tr_ = trough_of(close, st, la)
                keep = ~((d_ >= st - pd.Timedelta(days=10)) & (d_ <= tr_))
                lo.append(auc(s_[keep], y_[keep]))
                loeo_rows.append(dict(var=vid, label=lab, episode_start=st.date(), trough=tr_.date(), auc_without=round(lo[-1], 4)))
            lo = np.array(lo)
            # incremental
            Bm = BASE.reindex(cal)[msk]
            pA, pB, bse = incremental(days, Bm, np.where(have, s, np.nan), np.where(~np.isnan(y), y, np.nan))
            okI = ~np.isnan(pA) & ~np.isnan(pB) & have
            subI = np.where(okI[have])[0]
            if len(subI) > 100:
                yI = y_[subI]; pAi, pBi, bi = pA[have][subI], pB[have][subI], bse[have][subI]
                dA = auc(pBi, yI) - auc(pAi, yI)
                dBS = (1 - np.mean((pBi - yI) ** 2) / np.mean((bi - yI) ** 2)) - (1 - np.mean((pAi - yI) ** 2) / np.mean((bi - yI) ** 2))
                cidI = cid[subI]
                bd = boot_stat(cidI, lambda ix: auc(pBi[ix], yI[ix]) - auc(pAi[ix], yI[ix]), rng)
                p_inc = (1 + np.sum(bd <= 0)) / (B_BOOT + 1)
                aA = auc(pAi, yI); aB = auc(pBi, yI); inc_lo, inc_hi = np.percentile(bd, [2.5, 97.5]); n_inc = len(subI)
            else:
                dA = dBS = p_inc = aA = aB = inc_lo = inc_hi = np.nan; n_inc = 0
            # lead time
            lt, ref = lead_time(days, np.where(have, s, np.nan), [e for e in eps15 if w0 <= e[0] <= w1], close)
            lt.insert(0, "label", lab); lt.insert(0, "var", vid); lead_rows.append(lt)
            hit = lt.hit.mean() if len(lt) else np.nan
            rows.append(dict(var=vid, label=lab, sign_dn=sign[vid], n_days=len(d_), start=d_[0].date(), end=d_[-1].date(), n_pos=int(y_.sum()),
                             base_rate=round(y_.mean(), 4), n_ep30=len(starts30), auc=round(a, 4), auc_ci_lo=round(np.percentile(bA, 2.5), 4), auc_ci_hi=round(np.percentile(bA, 97.5), 4),
                             p_auc=round(pval, 4), brier_skill=round(bs, 4) if bs == bs else np.nan, bs_ci_lo=round(bs_lo, 4) if bs == bs else np.nan, bs_ci_hi=round(bs_hi, 4) if bs == bs else np.nan,
                             auc_walkfwd=round(a_wf, 4) if a_wf == a_wf else np.nan,
                             auc_pre2022=round(a_pre, 4), auc_2022on=round(a_post, 4), ep30_pre=n_pre, ep30_post=n_post,
                             loeo_min=round(np.nanmin(lo), 4) if len(lo) else np.nan, loeo_max=round(np.nanmax(lo), 4) if len(lo) else np.nan,
                             loeo_below_half=int((lo < 0.5).sum()) if len(lo) else np.nan, n_loeo=len(lo),
                             inc_n=n_inc, auc_base=round(aA, 4) if aA == aA else np.nan, auc_base_plus=round(aB, 4) if aB == aB else np.nan,
                             d_auc=round(dA, 4) if dA == dA else np.nan, d_auc_ci_lo=round(inc_lo, 4) if inc_lo == inc_lo else np.nan,
                             d_auc_ci_hi=round(inc_hi, 4) if inc_hi == inc_hi else np.nan, p_inc=round(p_inc, 4) if p_inc == p_inc else np.nan,
                             d_brier_skill=round(dBS, 4) if dBS == dBS else np.nan,
                             lead_hit_rate=round(hit, 3), lead_ref_rate=round(ref, 3), lead_lift=round(hit / ref, 2) if ref else np.nan,
                             lead_median=float(np.nanmedian(lt.lead_days)) if lt.hit.sum() else np.nan, n_lead_eps=len(lt)))
            print(vid, lab, "AUC", round(a, 3), "p", round(pval, 3), "BS", None if bs != bs else round(bs, 3), "dAUC", None if dA != dA else round(dA, 3), flush=True)
    R = pd.DataFrame(rows)
    for lab in ("DOWN", "UP"):
        m = (R.label == lab) & R.p_auc.notna()
        R.loc[m, "q_auc"] = bh(R.loc[m, "p_auc"].values).round(4)
        m2 = (R.label == lab) & R.p_inc.notna()
        R.loc[m2, "q_inc"] = bh(R.loc[m2, "p_inc"].values).round(4)
    R.to_csv(os.path.join(OUT, "stage1_results.csv"), index=False)
    pd.DataFrame(loeo_rows).to_csv(os.path.join(OUT, "stage1_loeo.csv"), index=False)
    pd.concat(lead_rows).to_csv(os.path.join(OUT, "stage1_lead_times.csv"), index=False)

    event_study(E, esign, y_dn, y_up, rng, "")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ext":  # post-hoc sensitivity A1: z-scores from full vendor history
        rng = np.random.default_rng(SEED)
        cal, close, y_dn, y_up, F, sign, BASE, E, esign = build(ext=True)
        event_study(E, esign, y_dn, y_up, rng, "_ext")
    else:
        main()
