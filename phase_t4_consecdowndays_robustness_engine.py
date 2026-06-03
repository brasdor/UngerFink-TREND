#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T4 -- ConsecDownDaysMR Robustness Engine
UngerFink Pipeline / Andrea Unger Methodology

Reads best T3MR variant trades and runs full robustness battery.
Enhanced vs RSI MR T4: includes year-removal test (remove 2021, remove 2024).

Input : data/research_consecdowndays_mr_t3mr/phase_t3mr_trades_{best}.csv
Output: data/research_consecdowndays_mr_t4/

T2 baseline: avg_r=+0.470R  PF=2.31
MR gates: win_rate 50-70%, avg_r > 0.10R, PF > 1.0
"""

from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

def p(*args, **kwargs):
    kwargs.pop("flush", None)
    print(*args, flush=True, **kwargs)

ROOT     = Path(__file__).parent
T3MR_DIR = ROOT / "data" / "research_consecdowndays_mr_t3mr"
OUT_DIR  = ROOT / "data" / "research_consecdowndays_mr_t4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = {"avg_r": 0.4696, "pf": 2.31, "total_r": 84.05}
MR_GATES  = {"win_rate_min": 0.50, "win_rate_max": 0.70,
             "avg_r_min": 0.10, "pf_min": 1.0}

MC_RUNS   = 2000
MC_BLOCKS = [1, 5, 10, 20]
np.random.seed(42)


def metrics(rs: np.ndarray) -> dict:
    if len(rs)==0:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,
                "max_dd_r":0.0,"total_r":0.0,"std_r":0.0,"t_score":0.0}
    w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum=np.cumsum(rs); peak=np.maximum.accumulate(cum)
    dd=float(np.max(peak-cum))
    std=float(np.std(rs,ddof=1)) if len(rs)>1 else 0.0
    t=float(np.mean(rs)/(std/np.sqrt(len(rs)))) if std>0 and len(rs)>1 else 0.0
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":dd,"total_r":float(np.sum(rs)),"std_r":std,"t_score":t}


def gate_str(m: dict) -> str:
    ok = (MR_GATES["win_rate_min"]<=m["win_rate"]<=MR_GATES["win_rate_max"] and
          m["avg_r"]>=MR_GATES["avg_r_min"] and m["pf"]>=MR_GATES["pf_min"])
    return "PASS" if ok else "FAIL"


def block_bootstrap(rs: np.ndarray, block_size: int, n_runs: int) -> dict:
    n = len(rs)
    totals, pfs = [], []
    for _ in range(n_runs):
        starts = np.random.randint(0, n, size=max(1, n//block_size)+5)
        sample = np.concatenate([rs[s:min(s+block_size,n)] for s in starts])[:n]
        if not len(sample): continue
        m = metrics(sample)
        totals.append(m["total_r"]); pfs.append(m["pf"])
    totals=np.array(totals); pfs=np.array(pfs)
    mc_pass = np.percentile(totals,5)>0 and np.percentile(pfs,5)>=MR_GATES["pf_min"]
    return {"block_size":block_size,
            "p05_total_r":float(np.percentile(totals,5)),
            "p50_total_r":float(np.percentile(totals,50)),
            "p95_total_r":float(np.percentile(totals,95)),
            "prob_positive":float(np.mean(totals>0)),
            "pf_p05":float(np.percentile(pfs,5)),
            "mc_pass":mc_pass}


def main() -> None:
    # Determine best variant
    best_file = T3MR_DIR / "best_variant.txt"
    best_var  = best_file.read_text(encoding="utf-8").strip() if best_file.exists() else "E"
    trades_path = T3MR_DIR / f"phase_t3mr_trades_{best_var}.csv"
    if not trades_path.exists():
        p(f"  ERROR: {trades_path} not found. Run T3MR first.")
        sys.exit(1)

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    if "year" not in df.columns:
        df["year"] = df["entry_time"].dt.year
    rs_all = df["net_r"].values

    p("=" * 70)
    p(f"  Phase T4 -- ConsecDownDaysMR Robustness (Variant {best_var})")
    p(f"  Config: consec5/ema200/hold20+/atr2  [{trades_path.name}]")
    p(f"  Trades: {len(df)}  Benchmark: avg_r={BENCHMARK['avg_r']}R  PF={BENCHMARK['pf']}")
    p("=" * 70)

    report_lines = []
    def rp(line=""):
        p(line)
        report_lines.append(line)

    # 1. Baseline
    rp(); rp("="*70); rp("  1. BASELINE SUMMARY"); rp("="*70)
    bm = metrics(rs_all)
    rp(f"  Trades    : {bm['n']}")
    rp(f"  Win rate  : {bm['win_rate']*100:.1f}%  (gate 50-70%: {'OK' if MR_GATES['win_rate_min']<=bm['win_rate']<=MR_GATES['win_rate_max'] else 'FAIL'})")
    rp(f"  Avg R     : {bm['avg_r']:+.4f}R  (gate >0.10R: {'OK' if bm['avg_r']>=MR_GATES['avg_r_min'] else 'FAIL'})")
    rp(f"  Total R   : {bm['total_r']:+.2f}R")
    rp(f"  Prof Fac  : {bm['pf']:.2f}")
    rp(f"  Max DD R  : {bm['max_dd_r']:.2f}R")
    rp(f"  t-score   : {bm['t_score']:.2f}  ({'significant >=2.0' if bm['t_score']>=2.0 else 'weak <2.0'})")
    rp(f"  Gate      : {gate_str(bm)}")

    # 2. Year-by-year (DETAILED -- bull specialist framing)
    rp(); rp("="*70); rp("  2. YEAR-BY-YEAR (bull specialist framing)"); rp("="*70)
    rp(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>8}  {'TotalR':>8}  {'DD':>7}  {'PF':>5}  Note")
    year_rows = []
    for yr in sorted(df["year"].unique()):
        ydf = df[df["year"]==yr]
        ym  = metrics(ydf["net_r"].values)
        note = "<<< BEAR" if yr==2022 else ("<<< BULL YEAR" if yr in (2021,2024) else "")
        neg  = "  <<< WEAK" if ym["avg_r"]<0 else ""
        rp(f"  {yr:>5}  {ym['n']:>5}  {ym['win_rate']*100:>5.1f}%  "
           f"{ym['avg_r']:>+7.3f}R  {ym['total_r']:>+7.2f}R  "
           f"{ym['max_dd_r']:>6.2f}R  {ym['pf']:>5.2f}  {note}{neg}")
        year_rows.append({"year":yr,**ym,"note":note})
    pd.DataFrame(year_rows).to_csv(OUT_DIR/"phase_t4_yearly.csv",index=False)

    # 3. REMOVE BEST YEAR (enhanced test)
    rp(); rp("="*70); rp("  3. REMOVE BEST YEAR (key test for bull-specialist system)"); rp("="*70)
    year_total_r = {r["year"]: r["total_r"] for r in year_rows}
    best_year = max(year_total_r, key=year_total_r.get)
    second_best = sorted(year_total_r, key=year_total_r.get, reverse=True)[1] if len(year_total_r)>1 else None

    rm_year_rows = []
    for yr_remove in sorted([2021, 2024] + ([best_year] if best_year not in [2021,2024] else [])):
        sub = df[df["year"] != yr_remove]
        m_sub = metrics(sub["net_r"].values)
        inverts = m_sub["avg_r"] < 0
        rp(f"  Remove {yr_remove} ({year_total_r.get(yr_remove,0):+.1f}R removed):  "
           f"n={m_sub['n']}  avg_r={m_sub['avg_r']:+.4f}R  PF={m_sub['pf']:.2f}  "
           f"total={m_sub['total_r']:+.2f}R  "
           f"{'<<< INVERTS SYSTEM' if inverts else gate_str(m_sub)}")
        rm_year_rows.append({"removed_year":yr_remove,"inverts":inverts,**m_sub})
    pd.DataFrame(rm_year_rows).to_csv(OUT_DIR/"phase_t4_remove_best_years.csv",index=False)

    # 4. Period splits
    rp(); rp("="*70); rp("  4. PERIOD SPLITS"); rp("="*70)
    mid_n = len(df)//2
    h1_n=df.iloc[:mid_n]; h2_n=df.iloc[mid_n:]
    m1n=metrics(h1_n["net_r"].values); m2n=metrics(h2_n["net_r"].values)
    rp(f"  Count-half 1 ({h1_n['entry_time'].iloc[0].date()} - {h1_n['entry_time'].iloc[-1].date()}):")
    rp(f"    n={m1n['n']}  avg_r={m1n['avg_r']:+.4f}R  PF={m1n['pf']:.2f}  total={m1n['total_r']:+.2f}R  {gate_str(m1n)}")
    rp(f"  Count-half 2 ({h2_n['entry_time'].iloc[0].date()} - {h2_n['entry_time'].iloc[-1].date()}):")
    rp(f"    n={m2n['n']}  avg_r={m2n['avg_r']:+.4f}R  PF={m2n['pf']:.2f}  total={m2n['total_r']:+.2f}R  "
       f"{gate_str(m2n)}  {'<<< NEG' if m2n['avg_r']<0 else ''}")
    mid_t=df["entry_time"].quantile(0.5)
    h1_t=df[df["entry_time"]<=mid_t]; h2_t=df[df["entry_time"]>mid_t]
    mt1=metrics(h1_t["net_r"].values); mt2=metrics(h2_t["net_r"].values)
    rp(f"  Time-half 1 ({h1_t['entry_time'].iloc[0].date()} - {h1_t['entry_time'].iloc[-1].date()}):")
    rp(f"    n={mt1['n']}  avg_r={mt1['avg_r']:+.4f}R  PF={mt1['pf']:.2f}  {gate_str(mt1)}")
    rp(f"  Time-half 2 ({h2_t['entry_time'].iloc[0].date()} - {h2_t['entry_time'].iloc[-1].date()}):")
    rp(f"    n={mt2['n']}  avg_r={mt2['avg_r']:+.4f}R  PF={mt2['pf']:.2f}  "
       f"{gate_str(mt2)}  {'<<< NEG' if mt2['avg_r']<0 else ''}")
    pd.DataFrame([{"split":"count_h1",**m1n},{"split":"count_h2",**m2n},
                  {"split":"time_h1",**mt1},{"split":"time_h2",**mt2}]).to_csv(
        OUT_DIR/"phase_t4_splits.csv",index=False)

    # 5. Remove best assets
    rp(); rp("="*70); rp("  5. REMOVE BEST ASSETS"); rp("="*70)
    asset_totals = df.groupby("symbol")["net_r"].sum().sort_values(ascending=False)
    rm_asset_rows = []
    for k in [1,3,5]:
        top_syms = asset_totals.head(k).index.tolist()
        sub = df[~df["symbol"].isin(top_syms)]
        m_sub = metrics(sub["net_r"].values)
        rp(f"  Remove top-{k} ({', '.join(top_syms[:3])}):")
        rp(f"    n={m_sub['n']}  avg_r={m_sub['avg_r']:+.4f}R  PF={m_sub['pf']:.2f}  {gate_str(m_sub)}")
        rm_asset_rows.append({"removed_k":k,"removed":str(top_syms),**m_sub})
    pd.DataFrame(rm_asset_rows).to_csv(OUT_DIR/"phase_t4_remove_best_assets.csv",index=False)

    # 6. Remove best months
    rp(); rp("="*70); rp("  6. REMOVE BEST MONTHS"); rp("="*70)
    df["ym"] = df["entry_time"].dt.to_period("M")
    month_totals = df.groupby("ym")["net_r"].sum().sort_values(ascending=False)
    rm_month_rows = []
    for k in [1,2,3]:
        top_m = month_totals.head(k).index.tolist()
        sub = df[~df["ym"].isin(top_m)]
        m_sub = metrics(sub["net_r"].values)
        months_str = ", ".join(str(m) for m in top_m[:3])
        rp(f"  Remove top-{k} months ({months_str}):")
        rp(f"    n={m_sub['n']}  avg_r={m_sub['avg_r']:+.4f}R  PF={m_sub['pf']:.2f}  {gate_str(m_sub)}")
        rm_month_rows.append({"removed_k":k,"months":months_str,**m_sub})
    pd.DataFrame(rm_month_rows).to_csv(OUT_DIR/"phase_t4_remove_best_months.csv",index=False)

    # 7. Block Bootstrap MC
    rp(); rp("="*70); rp(f"  7. BLOCK BOOTSTRAP MC ({MC_RUNS} runs)"); rp("="*70)
    rp(f"  {'BlkSz':>6}  {'p05':>9}  {'p50':>9}  {'p95':>9}  {'Prob+':>7}  {'PF_p05':>7}  MC")
    mc_rows = []
    for bs in MC_BLOCKS:
        r = block_bootstrap(rs_all, bs, MC_RUNS)
        rp(f"  {bs:>6}  {r['p05_total_r']:>+8.2f}R  {r['p50_total_r']:>+8.2f}R  "
           f"{r['p95_total_r']:>+8.2f}R  {r['prob_positive']:>6.1%}  "
           f"{r['pf_p05']:>7.2f}  {'OK' if r['mc_pass'] else 'FAIL'}")
        mc_rows.append(r)
    pd.DataFrame(mc_rows).to_csv(OUT_DIR/"phase_t4_montecarlo.csv",index=False)

    # 8. Critical flags
    rp(); rp("="*70); rp("  8. CRITICAL FLAGS"); rp("="*70)

    mc_p05_ok     = all(r["p05_total_r"]>0 for r in mc_rows)
    h2_ok         = m2n["avg_r"] >= 0
    rm1_df        = pd.read_csv(OUT_DIR/"phase_t4_remove_best_assets.csv")
    rm1_pass      = rm1_df[rm1_df["removed_k"]==1].iloc[0]["avg_r"] >= MR_GATES["avg_r_min"]
    rm_mon1_df    = pd.read_csv(OUT_DIR/"phase_t4_remove_best_months.csv")
    rm_mon1_pass  = rm_mon1_df[rm_mon1_df["removed_k"]==1].iloc[0]["avg_r"] >= MR_GATES["avg_r_min"]
    y2022_row     = next((r for r in year_rows if r["year"]==2022), None)
    bear_ok       = y2022_row is None or y2022_row["total_r"] > -20.0

    # Year removal flags
    rm_yr_df = pd.read_csv(OUT_DIR/"phase_t4_remove_best_years.csv")
    rm2021 = rm_yr_df[rm_yr_df["removed_year"]==2021].iloc[0] if 2021 in rm_yr_df["removed_year"].values else None
    rm2024 = rm_yr_df[rm_yr_df["removed_year"]==2024].iloc[0] if 2024 in rm_yr_df["removed_year"].values else None
    rm2021_inverts = bool(rm2021["inverts"]) if rm2021 is not None else False
    rm2024_inverts = bool(rm2024["inverts"]) if rm2024 is not None else False

    def chk(label, ok, detail=""):
        rp(f"  [{'PASS' if ok else 'FAIL'}] {label:<45} {detail}")

    chk("MC p05 > 0 (all block sizes)", mc_p05_ok,
        f"p05={mc_rows[0]['p05_total_r']:+.2f}R (bs=1)")
    chk("Remove top-1 asset: avg_r > 0.10R", rm1_pass,
        f"avg_r={rm1_df[rm1_df['removed_k']==1].iloc[0]['avg_r']:+.4f}R")
    chk("Remove top-1 month: avg_r > 0.10R", rm_mon1_pass,
        f"avg_r={rm_mon1_df[rm_mon1_df['removed_k']==1].iloc[0]['avg_r']:+.4f}R")
    chk("Second half avg_r >= 0", h2_ok, f"avg_r={m2n['avg_r']:+.4f}R")
    chk("2022 bear year total R > -20R", bear_ok,
        f"total_r={y2022_row['total_r']:+.2f}R" if y2022_row else "N/A")
    chk("Remove 2021: system does NOT invert", not rm2021_inverts,
        f"avg_r={float(rm2021['avg_r']):+.4f}R" if rm2021 is not None else "N/A")
    chk("Remove 2024: system does NOT invert", not rm2024_inverts,
        f"avg_r={float(rm2024['avg_r']):+.4f}R" if rm2024 is not None else "N/A")

    all_critical = all([mc_p05_ok, rm1_pass, rm_mon1_pass, h2_ok, bear_ok,
                        not rm2021_inverts, not rm2024_inverts])
    rp()
    rp(f"  Critical checks: {'ALL PASS' if all_critical else 'SOME FAIL -- review required'}")

    # Bull-specialist framing summary
    rp()
    rp("  --- Bull-Specialist Context ---")
    yr_data = {r["year"]: r["total_r"] for r in year_rows}
    bull_R  = sum(v for yr,v in yr_data.items() if v > 0)
    bear_R  = sum(v for yr,v in yr_data.items() if v <= 0)
    rp(f"  Positive years total R : {bull_R:+.2f}R")
    rp(f"  Negative years total R : {bear_R:+.2f}R")
    rp(f"  Bull/Bear R ratio      : {bull_R/abs(bear_R):.1f}x" if bear_R<0 else "  No negative years")
    rp(f"  2021+2024 combined R   : {yr_data.get(2021,0)+yr_data.get(2024,0):+.2f}R  "
       f"({(yr_data.get(2021,0)+yr_data.get(2024,0))/bm['total_r']*100:.0f}% of total)")

    with open(OUT_DIR/"phase_t4_master_report.txt","w",encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    p()
    p("[OK] phase_t4_master_report.txt")
    p("[OK] phase_t4_yearly.csv")
    p("[OK] phase_t4_splits.csv")
    p("[OK] phase_t4_remove_best_years.csv")
    p("[OK] phase_t4_remove_best_assets.csv")
    p("[OK] phase_t4_remove_best_months.csv")
    p("[OK] phase_t4_montecarlo.csv")
    p()
    p(f"T4 GATE: {'PASS' if all_critical else 'FAIL -- review required'}")
    sys.exit(0 if all_critical else 1)


if __name__ == "__main__":
    main()
