import argparse
import os
import sys
from dotenv import load_dotenv
from loguru import logger

# resolve paths before any relative imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
load_dotenv(os.path.join(_ROOT, ".env"))

from m1_data_pipeline.database import init_db, log_pipeline_run
from m1_data_pipeline.universe import fetch_universe
from m1_data_pipeline.ohlcv import fetch_ohlcv
from m1_data_pipeline.funding import fetch_funding
from m1_data_pipeline.macro import fetch_macro
from m1_data_pipeline.onchain import fetch_onchain

_LOG_PATH = os.path.join(_ROOT, "data", "pipeline.log")


def _setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(_LOG_PATH, level="DEBUG", rotation="10 MB", retention="30 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def run_full():
    _setup_logging()
    logger.info("=== M1 FULL RUN ===")
    init_db()

    logger.info("Step 1/5 — Universe")
    universe_df = fetch_universe(save=True)
    symbols = universe_df["symbol"].tolist()
    logger.info(f"Universe: {len(symbols)} symbols")

    logger.info("Step 2/5 — OHLCV (full lookback)")
    ohlcv_res = fetch_ohlcv(symbols, incremental=False)
    _log_ohlcv(ohlcv_res)

    logger.info("Step 3/5 — Funding / OI / L-S ratio (full)")
    fund_res = fetch_funding(symbols, incremental=False)
    _log_funding(fund_res)

    logger.info("Step 4/5 — Macro (FRED)")
    macro_res = fetch_macro()
    for k, v in macro_res.items():
        logger.info(f"  FRED {k}: {v} rows")

    logger.info("Step 5/5 — On-chain (Glassnode)")
    chain_res = fetch_onchain()
    for k, v in chain_res.items():
        logger.info(f"  Glassnode {k}: {v} rows")

    ok, err, rows = _run_stats(ohlcv_res, fund_res)
    rows += sum(macro_res.values()) + sum(chain_res.values())
    run_id = log_pipeline_run("full", symbols_ok=ok, symbols_err=err, rows_inserted=rows)
    logger.info(f"=== FULL RUN COMPLETE — run_id={run_id} "
                f"ok={ok} err={err} rows={rows} ===")


def run_incremental():
    _setup_logging()
    logger.info("=== M1 INCREMENTAL RUN ===")
    init_db()

    logger.info("Step 1/5 — Universe")
    universe_df = fetch_universe(save=True)
    symbols = universe_df["symbol"].tolist()
    logger.info(f"Universe: {len(symbols)} symbols")

    logger.info("Step 2/5 — OHLCV (incremental)")
    ohlcv_res = fetch_ohlcv(symbols, incremental=True)
    _log_ohlcv(ohlcv_res)

    logger.info("Step 3/5 — Funding / OI / L-S ratio (incremental)")
    fund_res = fetch_funding(symbols, incremental=True)
    _log_funding(fund_res)

    logger.info("Step 4/5 — Macro (FRED)")
    macro_res = fetch_macro()
    for k, v in macro_res.items():
        logger.info(f"  FRED {k}: {v} rows")

    logger.info("Step 5/5 — On-chain (Glassnode)")
    chain_res = fetch_onchain()
    for k, v in chain_res.items():
        logger.info(f"  Glassnode {k}: {v} rows")

    ok, err, rows = _run_stats(ohlcv_res, fund_res)
    rows += sum(macro_res.values()) + sum(chain_res.values())
    run_id = log_pipeline_run("incremental", symbols_ok=ok, symbols_err=err, rows_inserted=rows)
    logger.info(f"=== INCREMENTAL RUN COMPLETE — run_id={run_id} "
                f"ok={ok} err={err} rows={rows} ===")


def _run_stats(ohlcv_res: dict, fund_res: dict) -> tuple[int, int, int]:
    """Return (symbols_ok, symbols_err, total_rows) from ohlcv + funding results."""
    err_syms = {s for s, v in fund_res.items() if "error" in v}
    ok = sum(1 for s in ohlcv_res if s not in err_syms)
    err = len(err_syms)
    ohlcv_rows = sum(ohlcv_res.values())
    fund_rows = sum(
        v.get("funding", 0) + v.get("oi", 0) + v.get("ls", 0)
        for v in fund_res.values()
        if isinstance(v, dict) and "error" not in v
    )
    return ok, err, ohlcv_rows + fund_rows


def _log_ohlcv(res: dict):
    updated = sum(1 for v in res.values() if v > 0)
    skipped = sum(1 for v in res.values() if v == 0)
    total_rows = sum(res.values())
    logger.info(f"  OHLCV: {updated} symbols updated, {skipped} skipped, {total_rows} rows inserted")


def _log_funding(res: dict):
    ok = [s for s, v in res.items() if "error" not in v]
    err = [s for s, v in res.items() if "error" in v]
    total_funding = sum(v.get("funding", 0) for v in res.values() if isinstance(v, dict))
    logger.info(f"  Funding: {len(ok)} ok / {len(err)} errors / {total_funding} funding rows")
    for s in err:
        logger.warning(f"  Funding error {s}: {res[s]['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Data Pipeline")
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    args = parser.parse_args()

    if args.mode == "full":
        run_full()
    else:
        run_incremental()
