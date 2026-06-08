"""
Walk-forward validation for M3 Backtest.

Periods are fixed as specified in the blueprint.  Given the current DB has
OHLCV history from 2024-06-07 onward, the Train period will be data-empty
and Validate partial.  Both cases are handled gracefully.

Overfitting flag: if out-of-sample (Test) Sharpe < 50% of Train Sharpe,
a warning is printed.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from loguru import logger
from m3_backtest.engine import run_backtest
from m3_backtest.metrics import sharpe, max_drawdown, cagr, calmar

PERIODS = [
    ("Train",    "2022-01-03", "2023-12-31"),
    ("Validate", "2024-01-01", "2024-12-31"),
    ("Test",     "2025-01-01", "2026-06-02"),
]

LEVERAGE = 3


def run_walk_forward() -> dict[str, dict]:
    results = {}
    for name, start, end in PERIODS:
        logger.info(f"Running {name} period: {start} -> {end}")
        r = run_backtest(start, end, leverage=LEVERAGE, long_only=False)
        if not r:
            logger.warning(f"{name}: no data / no trades")
            results[name] = {}
        else:
            results[name] = r

    _print_wf_table(results)
    _check_overfit(results)
    return results


def _period_stats(r: dict) -> dict:
    if not r:
        return {
            "sharpe": float("nan"), "mdd": float("nan"),
            "cagr": float("nan"), "calmar": float("nan"),
            "n_periods": 0, "n_trades": 0,
        }
    s = r["summary_stats"]
    return {
        "sharpe":    s["sharpe"],
        "mdd":       s["max_drawdown"],
        "cagr":      s["cagr"],
        "calmar":    s["calmar"],
        "n_periods": s["periods_run"],
        "n_trades":  s["total_trades"],
    }


def _fmt(v, fmt=".3f", suffix="") -> str:
    if v != v:  # nan
        return "  n/a  "
    return f"{v:{fmt}}{suffix}"


def _print_wf_table(results: dict) -> None:
    print()
    print("=" * 72)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 72)
    hdr = f"  {'Period':<12} {'Dates':<28} {'Sharpe':>8} {'MDD%':>8} {'CAGR%':>8} {'Calmar':>8} {'N wks':>6}"
    print(hdr)
    print("  " + "-" * 68)

    for name, start, end in PERIODS:
        r = results.get(name, {})
        ps = _period_stats(r)
        date_str = f"{start} -> {end}"
        print(
            f"  {name:<12} {date_str:<28} "
            f"{_fmt(ps['sharpe']):>8} "
            f"{_fmt(ps['mdd'], '.2f', ''):>8} "
            f"{_fmt(ps['cagr'], '.2f', ''):>8} "
            f"{_fmt(ps['calmar']):>8} "
            f"{ps['n_periods']:>6}"
        )

    print("=" * 72)

    # Data availability note
    print()
    print("  NOTE: OHLCV history starts 2024-06-07; M2 factor window needs")
    print("  ~60 days warmup.  Train period has no data; Validate is partial.")
    print()


def _check_overfit(results: dict) -> None:
    train_s = _period_stats(results.get("Train", {}))["sharpe"]
    test_s  = _period_stats(results.get("Test",  {}))["sharpe"]

    if train_s != train_s or test_s != test_s:
        return  # can't compare nan

    if train_s > 0 and test_s < 0.5 * train_s:
        print(
            f"  WARNING: Out-of-sample Sharpe ({test_s:.3f}) is below 50% of "
            f"Train Sharpe ({train_s:.3f}) -- strategy may be overfit."
        )
    elif test_s >= 0 and train_s > 0:
        ratio = test_s / train_s
        print(
            f"  OOS retention: Test Sharpe / Train Sharpe = {ratio:.1%}"
        )


if __name__ == "__main__":
    run_walk_forward()
