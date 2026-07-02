"""
Daily regime state computation for T9B CI pipeline.
Lightweight version of regime_allocator_v4 — uses CSV funding files
instead of futures.db (which is too large for GitHub Actions).

Writes regime_state.json to data/regime_state.json.
All T9B engines read this file via t9b_shared.get_regime_weight().

Three-axis detection with anti-oscillation (v4):
  Axis 1: Trend (EMA200 + breadth with hysteresis)
  Axis 2: Funding (smoothed + hysteresis)
  Axis 3: Vol (ATR percentile)
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from datetime import date as Date

ROOT = Path(__file__).resolve().parents[1]
DATA_1D = ROOT / "data" / "futures_universe" / "ohlcv_1d"
FUNDING_DIR = ROOT / "data" / "futures_universe" / "funding_rates"
STATE_OUT = ROOT / "data" / "regime_state.json"
HISTORY_OUT = ROOT / "data" / "regime_history.csv"

SCHEME_C = {
    "BULL":  {"S1": 0.25, "S2": 0.15, "S3": 0.15, "S5": 0.25,
              "S6": 0.10, "S7": 0.10, "S8": 0.00},
    "BEAR":  {"S1": 0.05, "S2": 0.25, "S3": 0.05, "S5": 0.15,
              "S6": 0.25, "S7": 0.25, "S8": 0.00},
    "MIXED": {"S1": 0.125, "S2": 0.125, "S3": 0.125, "S5": 0.125,
              "S6": 0.125, "S7": 0.125, "S8": 0.125},
}

FUNDING_ENTER_HIGH = 0.00025
FUNDING_EXIT_HIGH  = 0.00010
BREADTH_ENTER_BULL = 58
BREADTH_EXIT_BULL  = 48
BREADTH_ENTER_BEAR = 42
BREADTH_EXIT_BEAR  = 48
MIN_HOLD_DAYS      = 5


def load_previous_state():
    if STATE_OUT.exists():
        with open(STATE_OUT) as f:
            return json.load(f)
    return None


def compute_regime(run_date=None):
    if run_date is None:
        run_date = Date.today()

    prev = load_previous_state()

    # BTC close + EMA200
    btc_path = DATA_1D / "BTCUSDT_1d.csv"
    if not btc_path.exists():
        print("[regime] BTCUSDT_1d.csv not found — defaulting to MIXED")
        return _default_state(run_date)

    btc = pd.read_csv(btc_path)
    btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms")
    btc = btc.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    btc_close = btc["close"].astype(float)
    btc_ema200 = btc_close.ewm(span=200, adjust=False).mean()

    # Breadth (% of symbols above their EMA200)
    above_count = 0
    total_count = 0
    for f in sorted(DATA_1D.glob("*_1d.csv")):
        try:
            df = pd.read_csv(f, usecols=["timestamp", "close"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
            cl = df["close"].astype(float)
            if len(cl) < 200:
                continue
            ema = cl.ewm(span=200, adjust=False).mean()
            latest_idx = cl.index[-1]
            if cl.loc[latest_idx] > ema.loc[latest_idx]:
                above_count += 1
            total_count += 1
        except:
            continue

    raw_breadth = (above_count / total_count * 100) if total_count > 0 else 50

    # Funding rate — average across all funded symbols (latest available date)
    funding_rates = []
    for f in sorted(FUNDING_DIR.glob("*_funding.csv")):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                funding_rates.append(df["funding_rate"].iloc[-1])
        except:
            continue

    raw_funding = np.mean(funding_rates) if funding_rates else 0.0001

    # Apply EMA smoothing using previous state if available
    if prev and "breadth_ema3" in prev and "funding_ema5" in prev:
        alpha_br = 2.0 / (3 + 1)
        alpha_fr = 2.0 / (5 + 1)
        breadth_smooth = prev["breadth_ema3"] * (1 - alpha_br) + raw_breadth * alpha_br
        funding_smooth = prev["funding_ema5"] * (1 - alpha_fr) + raw_funding * alpha_fr
    else:
        breadth_smooth = raw_breadth
        funding_smooth = raw_funding

    # BTC above EMA200?
    btc_above = False
    if len(btc_close) > 0 and len(btc_ema200) > 0:
        btc_above = float(btc_close.iloc[-1]) > float(btc_ema200.iloc[-1])

    # Trend axis with hysteresis + min hold
    current_trend = prev.get("trend_regime", "MIXED") if prev else "MIXED"
    last_trend_switch = prev.get("last_trend_switch", "2020-01-01") if prev else "2020-01-01"
    days_since_trend = (run_date - Date.fromisoformat(last_trend_switch)).days

    new_trend = current_trend
    if days_since_trend >= MIN_HOLD_DAYS:
        if current_trend != "BULL" and btc_above and breadth_smooth > BREADTH_ENTER_BULL:
            new_trend = "BULL"
        elif current_trend == "BULL" and (not btc_above or breadth_smooth < BREADTH_EXIT_BULL):
            new_trend = "MIXED"
        elif current_trend != "BEAR" and not btc_above and breadth_smooth < BREADTH_ENTER_BEAR:
            new_trend = "BEAR"
        elif current_trend == "BEAR" and (btc_above or breadth_smooth > BREADTH_EXIT_BEAR):
            new_trend = "MIXED"

    if new_trend != current_trend:
        last_trend_switch = str(run_date)
        current_trend = new_trend

    # Funding axis with hysteresis + min hold
    current_funding = prev.get("funding_regime", "NEUTRAL") if prev else "NEUTRAL"
    last_funding_switch = prev.get("last_funding_switch", "2020-01-01") if prev else "2020-01-01"
    days_since_funding = (run_date - Date.fromisoformat(last_funding_switch)).days

    new_funding = current_funding
    if days_since_funding >= MIN_HOLD_DAYS:
        if current_funding != "HIGH" and funding_smooth > FUNDING_ENTER_HIGH:
            new_funding = "HIGH"
        elif current_funding == "HIGH" and funding_smooth < FUNDING_EXIT_HIGH:
            new_funding = "NEUTRAL"
        elif current_funding != "LOW" and funding_smooth < 0.00005:
            new_funding = "LOW"
        elif current_funding == "LOW" and funding_smooth > FUNDING_EXIT_HIGH:
            new_funding = "NEUTRAL"

    if new_funding != current_funding:
        last_funding_switch = str(run_date)
        current_funding = new_funding

    # Vol axis
    if len(btc) > 252:
        hi = btc["high"].astype(float)
        lo = btc["low"].astype(float)
        tr = pd.concat([hi - lo, (hi - btc_close.shift()).abs(),
                        (lo - btc_close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        if len(atr14.dropna()) > 252:
            current_atr = atr14.iloc[-1]
            window = atr14.iloc[-252:]
            vol_pct = (window <= current_atr).mean() * 100
        else:
            vol_pct = 50
    else:
        vol_pct = 50

    if vol_pct < 30:   vol_mult = 0.75
    elif vol_pct > 70: vol_mult = 1.25
    else:              vol_mult = 1.00

    # Compute weights
    base = dict(SCHEME_C[current_trend])
    if current_funding == "HIGH":
        for s in ["S6", "S7"]:  base[s] = base.get(s, 0) + 0.05
        for s in ["S1", "S5"]:  base[s] = max(0, base.get(s, 0) - 0.05)
    elif current_funding == "LOW":
        for s in ["S2", "S3"]:  base[s] = base.get(s, 0) + 0.05
        for s in ["S6", "S7"]:  base[s] = max(0, base.get(s, 0) - 0.05)
    total = sum(base.values())
    if total > 0:
        base = {k: round(v / total, 4) for k, v in base.items()}

    state = {
        "date": str(run_date),
        "trend_regime": current_trend,
        "funding_regime": current_funding,
        "vol_multiplier": vol_mult,
        "weights": base,
        "allocator_version": "v4_ci",
        "breadth_raw": round(raw_breadth, 2),
        "breadth_ema3": round(breadth_smooth, 4),
        "funding_raw": round(raw_funding, 8),
        "funding_ema5": round(funding_smooth, 8),
        "last_trend_switch": last_trend_switch,
        "last_funding_switch": last_funding_switch,
        "btc_above_ema200": btc_above,
        "vol_percentile": round(vol_pct, 1),
    }

    # Save state
    with open(STATE_OUT, "w") as f:
        json.dump(state, f, indent=2)

    # Append to history
    hist_row = f"{run_date},{current_trend},{current_funding},{vol_mult},"
    hist_row += ",".join(f"{base.get(s,0):.4f}" for s in ["S1","S2","S3","S5","S6","S7","S8"])
    hist_row += f",{raw_breadth:.1f},{funding_smooth:.8f}\n"

    if not HISTORY_OUT.exists():
        header = "date,trend,funding,vol_mult,w_S1,w_S2,w_S3,w_S5,w_S6,w_S7,w_S8,breadth,funding_avg\n"
        HISTORY_OUT.write_text(header)
    with open(HISTORY_OUT, "a") as f:
        f.write(hist_row)

    return state


def _default_state(run_date):
    state = {
        "date": str(run_date),
        "trend_regime": "MIXED",
        "funding_regime": "NEUTRAL",
        "vol_multiplier": 1.0,
        "weights": SCHEME_C["MIXED"],
        "allocator_version": "v4_ci_default",
    }
    with open(STATE_OUT, "w") as f:
        json.dump(state, f, indent=2)
    return state


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()

    if args.date:
        d = Date.fromisoformat(args.date)
    else:
        from datetime import timedelta
        d = Date.today() - timedelta(days=1)

    state = compute_regime(d)
    print(f"[REGIME] {state['date']}: {state['trend_regime']} / {state['funding_regime']} "
          f"/ vol={state['vol_multiplier']}")
    print(f"  Weights: {json.dumps(state['weights'])}")
    print(f"  Breadth: {state.get('breadth_raw', '?')}% (ema3: {state.get('breadth_ema3', '?')}%)")
    print(f"  Funding: {state.get('funding_raw', '?')} (ema5: {state.get('funding_ema5', '?')})")
    print(f"  Saved to: {STATE_OUT}")
