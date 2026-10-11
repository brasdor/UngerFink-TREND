#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily append-only snapshots of series that are RESTATED by their vendors (no vintage history exists).

Every month this does not run is vintage history lost for good (docs/RISKOFF_STAGE1.md, H3/H6/H8).
Each snapshot file is named by FETCH date and stores the fetch timestamp, so a later backtest can ask
"what did this series say on day D?" instead of using today's restated history. Existing files are never
rewritten (a second run on the same day is a no-op).

  data/riskoff/forward/stablecoins/stablecoins_<fetchdate>.csv   DefiLlama total / USDT / USDC, last 45 rows each
  data/riskoff/forward/epu/epu_daily_<fetchdate>.csv             Baker-Bloom-Davis daily EPU, last 150 rows + file sha256
  data/riskoff/forward/epu/epu_full_<YYYY-MM>.csv.gz             full EPU file, once per month
  data/riskoff/forward/gpr/gpr_daily_<fetchdate>.csv             Caldara-Iacoviello GPR daily, last 150 rows (needs xlrd) or hash log
  data/riskoff/forward/gpr/gpr_raw_<fetchdate>.xls.gz            raw file, only if the parse fails (weekly cap)
  data/riskoff/forward/dominance/btc_dominance.csv               CoinGecko /global: BTC/ETH dominance, total mcap (append rows)
  data/riskoff/forward/snapshot_heartbeat.json                   last success per source (read by the health check)

Run alone: python tools/forward_snapshots.py . Also called by tools/refresh_funding_local.py (the funding-host job).
Free keyless sources only. A failing source is logged and skipped; it never stops the others or the funding refresh.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "riskoff" / "forward"
H = {"User-Agent": "Mozilla/5.0 (research snapshot)"}
HEARTBEAT = OUT / "snapshot_heartbeat.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=H, timeout=60, **kw)
    r.raise_for_status()
    return r


def _beat(source: str, detail: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except Exception:
        hb = {}
    hb[source] = {"last_success_utc": _now().isoformat(timespec="seconds"), "detail": detail}
    HEARTBEAT.write_text(json.dumps(hb, indent=2), encoding="utf-8")


def stablecoins() -> str:
    d = OUT / "stablecoins"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"stablecoins_{_now():%Y-%m-%d}.csv"
    if f.exists():
        return "exists"
    rows = []
    for name, q in (("total", ""), ("USDT", "?stablecoin=1"), ("USDC", "?stablecoin=2")):
        js = _get("https://stablecoins.llama.fi/stablecoincharts/all" + q).json()
        for x in js[-45:]:
            rows.append((name, datetime.fromtimestamp(int(x["date"]), timezone.utc).strftime("%Y-%m-%d"),
                         x["totalCirculatingUSD"]["peggedUSD"]))
    df = pd.DataFrame(rows, columns=["series", "date", "circulating_usd"])
    df.insert(0, "fetched_utc", _now().isoformat(timespec="seconds"))
    df.to_csv(f, index=False)
    return f"{len(df)} rows"


def epu() -> str:
    d = OUT / "epu"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"epu_daily_{_now():%Y-%m-%d}.csv"
    if f.exists():
        return "exists"
    raw = _get("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv").content
    sha = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day)).dt.strftime("%Y-%m-%d")
    tail = df.tail(150)[["date", "daily_policy_index"]].copy()
    tail.insert(0, "file_sha256", sha); tail.insert(0, "fetched_utc", _now().isoformat(timespec="seconds"))
    tail["file_rows"] = len(df)
    tail.to_csv(f, index=False)
    full = d / f"epu_full_{_now():%Y-%m}.csv.gz"
    if not full.exists():
        full.write_bytes(gzip.compress(raw))
    return f"{len(df)} rows total, sha {sha[:10]}"


def gpr() -> str:
    d = OUT / "gpr"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"gpr_daily_{_now():%Y-%m-%d}.csv"
    if f.exists():
        return "exists"
    r = _get("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls")
    raw = r.content
    sha = hashlib.sha256(raw).hexdigest()
    last_mod = r.headers.get("last-modified", "")
    try:
        df = pd.read_excel(io.BytesIO(raw))  # needs xlrd (see requirements-datahost.txt)
        tail = df.tail(150).copy()
        tail.insert(0, "file_sha256", sha); tail.insert(0, "fetched_utc", _now().isoformat(timespec="seconds"))
        tail["last_modified_header"] = last_mod
        tail.to_csv(f, index=False)
        return f"{len(df)} rows, sha {sha[:10]}"
    except Exception as exc:
        # parse failed (xlrd missing / format change): keep the raw bytes so nothing is lost, at most weekly
        prev = sorted(d.glob("gpr_raw_*.xls.gz"))
        if not prev or (_now() - datetime.strptime(prev[-1].name[8:18], "%Y-%m-%d").replace(tzinfo=timezone.utc)).days >= 6:
            (d / f"gpr_raw_{_now():%Y-%m-%d}.xls.gz").write_bytes(gzip.compress(raw))
        pd.DataFrame([{"fetched_utc": _now().isoformat(timespec="seconds"), "file_sha256": sha, "bytes": len(raw),
                       "last_modified_header": last_mod, "parse_error": f"{type(exc).__name__}: {exc}"[:200]}]).to_csv(f, index=False)
        return f"hash only (xlrd missing?): {type(exc).__name__}"


def dominance() -> str:
    d = OUT / "dominance"; d.mkdir(parents=True, exist_ok=True)
    f = d / "btc_dominance.csv"
    today = f"{_now():%Y-%m-%d}"
    if f.exists() and today in pd.read_csv(f)["fetched_utc"].str[:10].values:
        return "exists"
    js = _get("https://api.coingecko.com/api/v3/global").json()["data"]
    row = {"fetched_utc": _now().isoformat(timespec="seconds"), "btc_dominance_pct": js["market_cap_percentage"].get("btc"),
           "eth_dominance_pct": js["market_cap_percentage"].get("eth"), "total_mcap_usd": js["total_market_cap"].get("usd"),
           "total_volume_usd": js["total_volume"].get("usd"), "coingecko_updated_at": js.get("updated_at")}
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    return f"btc {row['btc_dominance_pct']:.2f}%"


def run() -> int:
    bad = 0
    for name, fn in (("stablecoins", stablecoins), ("epu", epu), ("gpr", gpr), ("dominance", dominance)):
        try:
            detail = fn()
            if detail != "exists":
                _beat(name, detail)
            print(f"[SNAPSHOT] {name}: {detail}")
        except Exception as exc:
            bad += 1
            print(f"[SNAPSHOT] {name} FAILED: {type(exc).__name__}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
