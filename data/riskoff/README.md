# data/riskoff

Outputs of the risk-off research (Stage 0 catalogue and data inventory, Stage 1 univariate tests). Reports: `docs/RISKOFF_STAGE0.md`, `docs/RISKOFF_STAGE1.md`. Pre-registrations: `docs/RISKOFF_PREREG.md`, `docs/RISKOFF_OI_FORWARD_PREREG.md`. Closed specs: `docs/RISKOFF_DO_NOT_RETEST.md`. Code: `tools/riskoff/`.

**The FRED data is NOT vintaged.** Every FRED series (VIX, Nasdaq, Baa spread, WALCL, TGA, reverse repo, real yield, broad dollar) was downloaded on 2026-10-11 as its latest-revised values, not as first-release values. Revisions are immaterial for market series (VIX, Nasdaq, Baa, real yields) and small for H.4.1 balance-sheet series, but nothing here was tested on real-time vintages (ALFRED was not used). The DefiLlama stablecoin, blockchain.com hashrate and revenue series are likewise as published on the download date, and DefiLlama restates its history, so H3 is not point-in-time at all.

Not committed: `raw/` (vendor downloads, regenerate with `python tools/riskoff/stage1_fetch.py`; it is git-ignored). No API keys are used by any script in this folder; all sources are keyless public endpoints.

Files: `events.csv`, `labels_daily.csv`, `base_rates.csv`, `known_episodes_check.csv`, `episode_counts_by_window.csv`, `target_series.csv` (Stage 0); `data_inventory.csv` (Stage 0 inventory); `stage1_*.csv` (Stage 1 results, features, leave-one-out, lead times, event studies, noise calibration). `forward/` holds append-only forward logs written by the daily data job.
