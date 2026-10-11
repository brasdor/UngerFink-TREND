"""Stage 1 bulk fetch (free sources only). Writes small daily CSVs to data/riskoff/raw/. Read-only w.r.t. the rest of the repo."""
import io, os, sys, time, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "riskoff", "raw")
os.makedirs(RAW, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (research)"}
S = requests.Session(); S.headers.update(H)


def get(url, tries=4, **kw):
    for i in range(tries):
        try:
            r = S.get(url, timeout=40, **kw)
            if r.status_code in (200, 404):
                return r
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def fred():
    for sid in ["VIXCLS", "NASDAQCOM", "BAA10Y", "WALCL", "WTREGEN", "RRPONTSYD", "DFII10", "DTWEXBGS"]:
        r = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
        d = pd.read_csv(io.StringIO(r.text)); d.columns = ["date", "value"]
        d["value"] = pd.to_numeric(d["value"], errors="coerce"); d = d.dropna()
        d.to_csv(os.path.join(RAW, f"fred_{sid}.csv"), index=False); print("fred", sid, len(d), d.date.iloc[0], d.date.iloc[-1])


def frank():
    rows = []; a = dt.date(2019, 6, 1); end = dt.date(2026, 10, 10)
    while a <= end:
        b = min(a + dt.timedelta(days=79), end)
        r = get(f"https://api.frankfurter.app/{a}..{b}?from=USD&to=JPY")
        for k, v in r.json()["rates"].items():
            rows.append((k, v["JPY"]))
        a = b + dt.timedelta(days=1)
    d = pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates("date").sort_values("date")
    d.to_csv(os.path.join(RAW, "ecb_USDJPY.csv"), index=False); print("usdjpy", len(d), d.date.iloc[0], d.date.iloc[-1])


def chain():
    for n in ["hash-rate", "miners-revenue", "market-price"]:
        r = get(f"https://api.blockchain.info/charts/{n}?timespan=9years&sampled=false&format=json")
        v = r.json()["values"]
        d = pd.DataFrame(v); d["date"] = pd.to_datetime(d.x, unit="s").dt.strftime("%Y-%m-%d")
        d = d.groupby("date").y.last().reset_index(); d.columns = ["date", "value"]
        d.to_csv(os.path.join(RAW, f"bc_{n}.csv"), index=False); print("blockchain", n, len(d), d.date.iloc[0], d.date.iloc[-1])


def llama():
    for name, q in [("total", ""), ("USDT", "?stablecoin=1"), ("USDC", "?stablecoin=2")]:
        r = get("https://stablecoins.llama.fi/stablecoincharts/all" + q)
        d = pd.DataFrame([(dt.datetime.fromtimestamp(int(x["date"]), dt.timezone.utc).strftime("%Y-%m-%d"),
                           x["totalCirculatingUSD"]["peggedUSD"]) for x in r.json()], columns=["date", "value"])
        d.to_csv(os.path.join(RAW, f"llama_{name}.csv"), index=False); print("llama", name, len(d), d.date.iloc[0], d.date.iloc[-1])


def one_metrics(day):
    u = f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{day}.zip"
    r = get(u)
    if r is None or r.status_code != 200:
        return (day, None)
    z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open(z.namelist()[0]))
    x = x.sort_values("create_time").iloc[-1]
    return (day, (float(x.sum_open_interest), float(x.sum_open_interest_value), float(x.count_long_short_ratio), float(x.sum_taker_long_short_vol_ratio)))


def metrics():
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-09-01", "2026-08-31")]
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(one_metrics, days))
    ok = [(d, *v) for d, v in res if v]; miss = [d for d, v in res if not v]
    pd.DataFrame(ok, columns=["date", "oi_coin", "oi_usd", "ls_ratio", "taker_ratio"]).to_csv(os.path.join(RAW, "bv_btc_metrics_daily.csv"), index=False)
    print("metrics ok", len(ok), "missing", len(miss), miss[:10])


def premium():
    out = []
    for m in pd.period_range("2020-01", "2026-07", freq="M"):
        u = f"https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1d/BTCUSDT-1d-{m}.zip"
        r = get(u)
        if r is None or r.status_code != 200:
            print("premium missing month", m); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open(z.namelist()[0]), header=None if not z.read(z.namelist()[0])[:4].isalpha() else 0)
        x = x.iloc[:, :7]; x.columns = ["open_time", "open", "high", "low", "close", "v", "close_time"]
        out.append(x[["open_time", "close"]])
    def dayfile(day):
        u = f"https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/BTCUSDT/1d/BTCUSDT-1d-{day}.zip"
        r = get(u)
        if r is None or r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open(z.namelist()[0]), header=None)
        x = x[pd.to_numeric(x[0], errors="coerce").notna()].iloc[:, :7]; x.columns = ["open_time", "open", "high", "low", "close", "v", "close_time"]
        return x[["open_time", "close"]]
    with ThreadPoolExecutor(8) as ex:
        for x in ex.map(dayfile, [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-08-01", "2026-08-31")]):
            if x is not None: out.append(x)
    d = pd.concat(out); d = d[pd.to_numeric(d.open_time, errors="coerce").notna()]
    d["date"] = pd.to_datetime(d.open_time.astype("int64"), unit="ms").dt.strftime("%Y-%m-%d")
    d["close"] = d.close.astype(float)
    d = d.drop_duplicates("date").sort_values("date")[["date", "close"]]
    d.to_csv(os.path.join(RAW, "bv_btc_premium_daily.csv"), index=False); print("premium", len(d), d.date.iloc[0], d.date.iloc[-1])


if __name__ == "__main__":
    for f in (fred, frank, chain, llama, premium, metrics):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, type(e).__name__, e)
