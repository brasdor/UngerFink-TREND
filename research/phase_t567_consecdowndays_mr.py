#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase T5 + T6 + T7 -- ConsecDownDaysMR Portfolio Filter + Capital Engine + Asset Robustness
UngerFink Pipeline / Andrea Unger Methodology

Input   : data/research_consecdowndays_mr_t3mr/phase_t3mr_trades_E.csv
T5 out  : data/research_consecdowndays_mr_t5/
T6 out  : data/research_consecdowndays_mr_t6/
T7 out  : data/research_consecdowndays_mr_t7/

PAUSES before T8 -- prints combined scorecard for review.

Capital config: $10,000 / 0.25% risk / leverage 1.0 / kill-switch -35%
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

ROOT     = Path(__file__).resolve().parents[1]
T3MR_DIR = ROOT / "data" / "research_consecdowndays_mr_t3mr"
T5_DIR   = ROOT / "data" / "research_consecdowndays_mr_t5"
T6_DIR   = ROOT / "data" / "research_consecdowndays_mr_t6"
T7_DIR   = ROOT / "data" / "research_consecdowndays_mr_t7"
for d in [T5_DIR, T6_DIR, T7_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CAPS       = [None, 5, 10, 15, 20]
CAP_LABELS = ["uncapped", "max5", "max10", "max15", "max20"]

INITIAL_CAPITAL  = 10_000.0
RISK_PCT         = 0.0025
KILL_SWITCH_DD   = 0.35

MR_GATES = {"min_trades":100,"win_rate_min":0.50,"win_rate_max":0.70,
             "avg_r_min":0.10,"pf_min":1.0}
EPS = 1e-12


# =============================================================================
# SHARED UTILS
# =============================================================================

def metrics(rs: np.ndarray) -> dict:
    if len(rs)==0:
        return {"n":0,"win_rate":0.0,"avg_r":0.0,"pf":0.0,"max_dd_r":0.0,"total_r":0.0}
    w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    cum=np.cumsum(rs); peak=np.maximum.accumulate(cum)
    dd=float(np.max(peak-cum))
    return {"n":len(rs),"win_rate":float(len(w)/len(rs)),"avg_r":float(np.mean(rs)),
            "pf":float(pf),"max_dd_r":dd,"total_r":float(np.sum(rs))}


def gate_str(m: dict) -> str:
    ok=(MR_GATES["min_trades"]<=m["n"] and
        MR_GATES["win_rate_min"]<=m["win_rate"]<=MR_GATES["win_rate_max"] and
        m["avg_r"]>=MR_GATES["avg_r_min"] and m["pf"]>=MR_GATES["pf_min"])
    return "PASS" if ok else "FAIL"


def apply_cap(df: pd.DataFrame, max_concurrent) -> pd.DataFrame:
    if max_concurrent is None:
        return df.copy()
    df_s = df.sort_values("entry_time").reset_index(drop=True)
    open_exits, accepted = [], []
    for i, row in df_s.iterrows():
        open_exits = [et for et in open_exits if et > row["entry_time"]]
        if len(open_exits) < max_concurrent:
            open_exits.append(row["exit_time"])
            accepted.append(i)
    return df_s.loc[accepted].reset_index(drop=True)


def year_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(df["year"].unique()):
        yd = df[df["year"]==yr]
        m  = metrics(yd["net_r"].values)
        rows.append({"year":int(yr),**m})
    return pd.DataFrame(rows)


# =============================================================================
# T5 -- PORTFOLIO FILTER
# =============================================================================

def run_t5(df_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p("\n" + "="*70)
    p("  T5 -- ConsecDownDaysMR Portfolio Filter")
    p(f"  Input trades: {len(df_full)}  Caps: {CAP_LABELS}")
    p("="*70)

    total = len(df_full)
    summary_rows = []
    best_trades  = None
    best_label   = "uncapped"

    for cap, label in zip(CAPS, CAP_LABELS):
        acc = apply_cap(df_full, cap)
        m   = metrics(acc["net_r"].values)
        g   = gate_str(m)

        p(f"\n  {label}: accepted={m['n']}/{total} ({m['n']/total*100:.1f}%)  "
          f"avg_r={m['avg_r']:+.4f}R  PF={m['pf']:.2f}  DD={m['max_dd_r']:.2f}R  {g}")

        # Year-by-year for this cap
        if not acc.empty:
            yb = year_breakdown(acc)
            p(f"  {'Year':>5}  {'N':>4}  {'AvgR':>8}  {'TotR':>8}  {'DD':>6}")
            for _, yr in yb.iterrows():
                bull = " *** BULL" if int(yr["year"]) in (2021,2024) else ""
                p(f"  {int(yr['year']):>5}  {int(yr['n']):>4}  "
                  f"{yr['avg_r']:>+7.3f}R  {yr['total_r']:>+7.2f}R  "
                  f"{yr['max_dd_r']:>5.2f}R{bull}")
            acc.to_csv(T5_DIR/f"phase_t5_trades_{label}.csv", index=False)

        summary_rows.append({"cap":label,"accepted":m["n"],"accept_pct":round(m["n"]/total*100,1),
            "win_rate":round(m["win_rate"],4),"avg_r":round(m["avg_r"],4),
            "total_r":round(m["total_r"],2),"pf":round(m["pf"],2),
            "max_dd_r":round(m["max_dd_r"],2),"gate":g})

        if best_trades is None and g=="PASS":
            best_trades = acc.copy()
            best_label  = label

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(T5_DIR/"phase_t5_portfolio_summary.csv", index=False)

    p("\n  T5 Comparison Table:")
    p(f"  {'Cap':<10} {'Acc':>5} {'Acc%':>6} {'WR%':>6} {'AvgR':>8} {'TotR':>8} {'PF':>5} {'DD':>7} Gate")
    p("  "+"-"*70)
    for r in summary_rows:
        p(f"  {r['cap']:<10} {r['accepted']:>5} {r['accept_pct']:>5.1f}% "
          f"{r['win_rate']*100:>5.1f}% {r['avg_r']:>+7.4f}R "
          f"{r['total_r']:>+7.2f}R {r['pf']:>5.2f} {r['max_dd_r']:>6.2f}R {r['gate']}")

    if best_trades is None:
        best_trades = df_full.copy()
        best_label  = "uncapped"
    p(f"\n  Best cap for T6: {best_label}")
    return summary_df, best_trades


# =============================================================================
# T6 -- CAPITAL ENGINE
# =============================================================================

def simulate_capital(df: pd.DataFrame, label: str) -> dict:
    df = df.sort_values("entry_time").reset_index(drop=True)
    equity = INITIAL_CAPITAL; peak = INITIAL_CAPITAL
    eq_curve = [INITIAL_CAPITAL]
    kill_fired = False; kill_at = None
    trade_data = []

    for _, row in df.iterrows():
        if kill_fired: break
        risk_usd = equity * RISK_PCT
        pnl      = row["net_r"] * risk_usd
        equity  += pnl
        peak     = max(peak, equity)
        dd_pct   = (peak - equity) / max(peak, EPS)
        if dd_pct >= KILL_SWITCH_DD:
            kill_fired = True; kill_at = equity
        eq_curve.append(equity)
        trade_data.append({"symbol":row["symbol"],"entry_time":row["entry_time"],
            "exit_time":row["exit_time"],"net_r":row["net_r"],
            "pnl_usd":round(pnl,2),"equity_after":round(equity,2)})

    rs = df["net_r"].values[:len(trade_data)]
    if len(rs)==0:
        return {}
    w=rs[rs>0]; l=np.abs(rs[rs<0])
    pf=w.sum()/l.sum() if l.sum()>0 else (99.0 if w.sum()>0 else 0.0)
    eq_arr=np.array(eq_curve); pk_arr=np.maximum.accumulate(eq_arr)
    max_dd_pct=float(np.max((pk_arr-eq_arr)/np.maximum(pk_arr,EPS))*100)
    max_dd_usd=float(np.max(pk_arr-eq_arr))
    first=df["entry_time"].iloc[0]; last=df["exit_time"].iloc[len(trade_data)-1]
    years=max((last-first).days/365.25,0.01)

    # Year P&L
    tdf=pd.DataFrame(trade_data)
    tdf["entry_time"]=pd.to_datetime(tdf["entry_time"],utc=True)
    tdf["year"]=tdf["entry_time"].dt.year
    year_pnl=tdf.groupby("year").agg(n=("pnl_usd","count"),pnl_usd=("pnl_usd","sum"),
        avg_r=("net_r","mean"),win_rate=("net_r",lambda x:(x>0).mean())).reset_index()

    return {"label":label,"n":len(rs),"kill_fired":kill_fired,"kill_equity":kill_at,
        "final_equity":round(equity,2),"total_return":round((equity/INITIAL_CAPITAL-1)*100,2),
        "cagr":round(((equity/INITIAL_CAPITAL)**(1/years)-1)*100,2),"years":round(years,2),
        "max_dd_pct":round(max_dd_pct,2),"max_dd_usd":round(max_dd_usd,2),
        "avg_r":round(float(np.mean(rs)),4),"win_rate":round(float(len(w)/len(rs)),4),
        "pf":round(float(pf),2),"year_pnl":year_pnl,"trade_df":tdf}


def print_t6_scorecard(r: dict, out_dir: Path) -> bool:
    p(f"\n  T6 Scorecard -- {r['label']}")
    p(f"  Trades     : {r['n']}")
    if r["kill_fired"]:
        p(f"  Kill-switch: FIRED  (equity=${r['kill_equity']:,.0f})")
    else:
        p(f"  Kill-switch: NOT fired")
    p(f"  Init cap   : ${INITIAL_CAPITAL:,.2f}")
    p(f"  Final eq   : ${r['final_equity']:,.2f}")
    p(f"  Total ret  : {r['total_return']:+.1f}%")
    p(f"  CAGR       : {r['cagr']:+.2f}%  (over {r['years']:.1f} years)")
    p(f"  Max DD %   : {r['max_dd_pct']:.2f}%")
    p(f"  Max DD $   : ${r['max_dd_usd']:,.2f}")
    p(f"  Avg R      : {r['avg_r']:+.4f}R")
    p(f"  PF         : {r['pf']:.2f}")

    yp = r["year_pnl"]
    p(f"\n  Year-by-Year P&L:")
    p(f"  {'Year':>5}  {'N':>4}  {'WR%':>6}  {'AvgR':>7}  {'P&L($)':>10}  Note")
    cumul = INITIAL_CAPITAL
    for _, row in yp.iterrows():
        cumul += row["pnl_usd"]
        yr = int(row["year"])
        note = "<<< BULL YEAR" if yr in (2021,2024) else ("<<< BEAR" if yr==2022 else "")
        neg  = "  <<< WEAK" if row["pnl_usd"]<0 else ""
        p(f"  {yr:>5}  {int(row['n']):>4}  {row['win_rate']*100:>5.1f}%  "
          f"{row['avg_r']:>+6.3f}R  ${row['pnl_usd']:>+9,.2f}  {note}{neg}")

    gate_ok = (not r["kill_fired"] and r["avg_r"]>=MR_GATES["avg_r_min"]
               and r["pf"]>=MR_GATES["pf_min"])
    p(f"\n  T6 Gate: {'PASS' if gate_ok else 'FAIL'}")

    # Save
    r["trade_df"].to_csv(out_dir/"phase_t6_equity_trades.csv", index=False)
    yp.to_csv(out_dir/"phase_t6_year_pnl.csv", index=False)
    with open(out_dir/"phase_t6_scorecard.txt","w",encoding="utf-8") as f:
        f.write(f"T6 Scorecard -- {r['label']}\n")
        f.write(f"Config: consec5/ema200/hold20/atr2/1D\n\n")
        f.write(f"Final equity : ${r['final_equity']:,.2f}\n")
        f.write(f"Total return : {r['total_return']:+.1f}%\n")
        f.write(f"CAGR         : {r['cagr']:+.2f}%\n")
        f.write(f"Max DD %     : {r['max_dd_pct']:.2f}%\n")
        f.write(f"Kill-switch  : {'FIRED' if r['kill_fired'] else 'not fired'}\n")
        f.write(f"Avg R        : {r['avg_r']:+.4f}R\n")
        f.write(f"PF           : {r['pf']:.2f}\n\n")
        f.write(f"T6 Gate: {'PASS' if gate_ok else 'FAIL'}\n")
    return gate_ok


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # Load T3MR Variant E trades
    trades_path = T3MR_DIR / "phase_t3mr_trades_E.csv"
    if not trades_path.exists():
        p(f"ERROR: {trades_path} not found"); sys.exit(1)

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    if "year" not in df.columns:
        df["year"] = df["entry_time"].dt.year

    p("="*70)
    p("  T5+T6+T7 -- ConsecDownDaysMR Pipeline")
    p(f"  Input: {len(df)} trades from T3MR Variant E")
    p("="*70)

    # T5
    t5_summary, best_trades = run_t5(df)

    # T6 (full universe, best cap)
    p("\n" + "="*70)
    p("  T6 -- Capital Execution Engine")
    p("="*70)
    r_t6 = simulate_capital(best_trades, f"ConsecDownDays (cap={t5_summary[t5_summary['gate']=='PASS'].iloc[0]['cap'] if any(t5_summary['gate']=='PASS') else 'uncapped'})")
    t6_pass = print_t6_scorecard(r_t6, T6_DIR)

    # T7 -- remove ZEC
    p("\n" + "="*70)
    p("  T7 -- Asset Robustness: Remove ZEC_USDT (top-1)")
    p("="*70)

    # Re-apply best cap without ZEC from original trades
    best_cap_label = t5_summary[t5_summary["gate"]=="PASS"].iloc[0]["cap"] if any(t5_summary["gate"]=="PASS") else "uncapped"
    best_cap_n     = next((c for c,l in zip(CAPS,CAP_LABELS) if l==best_cap_label), None)
    df_no_zec      = df[df["symbol"] != "ZEC/USDT"].copy()
    # Also try underscore format
    df_no_zec      = df_no_zec[df_no_zec["symbol"].str.upper() != "ZEC_USDT"]
    p(f"  Trades after removing ZEC: {len(df_no_zec)} (from {len(df)})")
    t7_trades = apply_cap(df_no_zec, best_cap_n)
    p(f"  After re-cap ({best_cap_label}): {len(t7_trades)} trades")
    r_t7 = simulate_capital(t7_trades, "ConsecDownDays (no ZEC)")
    t7_pass = print_t6_scorecard(r_t7, T7_DIR)

    # Combined side-by-side
    p("\n" + "="*70)
    p("  T6 vs T7 SIDE-BY-SIDE")
    p("="*70)
    p(f"  {'Metric':<22} {'T6 (full)':>15} {'T7 (no ZEC)':>15} {'Delta':>10}")
    p("  "+"-"*65)
    def row(label, v6, v7, fmt=".2f", suffix=""):
        p(f"  {label:<22} {f'{v6:{fmt}}{suffix}':>15} {f'{v7:{fmt}}{suffix}':>15}  {f'{v7-v6:+{fmt}}{suffix}':>10}")
    if r_t6 and r_t7:
        row("Trades",         r_t6["n"],            r_t7["n"],            fmt="d")
        row("Final equity($)",r_t6["final_equity"],  r_t7["final_equity"],  fmt=",.2f")
        row("Total return(%)",r_t6["total_return"],  r_t7["total_return"],  fmt=".1f",suffix="%")
        row("CAGR (%)",       r_t6["cagr"],          r_t7["cagr"],          fmt=".2f",suffix="%")
        row("Max DD (%)",     r_t6["max_dd_pct"],    r_t7["max_dd_pct"],    fmt=".2f",suffix="%")
        row("Max DD ($)",     r_t6["max_dd_usd"],    r_t7["max_dd_usd"],    fmt=",.2f")
        row("Avg R",          r_t6["avg_r"],         r_t7["avg_r"],         fmt=".4f",suffix="R")
        row("PF",             r_t6["pf"],            r_t7["pf"],            fmt=".2f")
    p("  "+"-"*65)
    p(f"  T6 gate: {'PASS' if t6_pass else 'FAIL'}    T7 gate: {'PASS' if t7_pass else 'FAIL'}")
    p()
    if t6_pass and t7_pass:
        p("  Both T6 and T7 PASS.")
        p("  PAUSING before T8. Review scorecard above, then run:")
        p("  python phase_t8_consecdowndays_mr_config_freeze.py")
    else:
        p("  One or more gates FAILED -- do not proceed to T8.")

    sys.exit(0 if (t6_pass and t7_pass) else 1)


if __name__ == "__main__":
    main()
