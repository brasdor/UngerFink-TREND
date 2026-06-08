@echo off
cd C:\Users\Jean\UngerFink-TREND\trading
echo %date% %time% -- run_daily.bat started >> data\daily_run.log

:: Step 1 -- catch up any missed data first
python -m m1_data_pipeline.check_missed_runs >> data\daily_run.log 2>&1

:: Step 2 -- regular incremental data update
python -m m1_data_pipeline.pipeline --mode incremental >> data\daily_run.log 2>&1
if %errorlevel% neq 0 (
    echo %date% %time% -- PIPELINE ERROR code %errorlevel% >> data\daily_run.log
    python -m m6_monitor.alerts --message "Daily pipeline FAILED" --level critical
)

:: Step 3 -- check and recover any missed rebalancing
python -m m5_execution.missed_rebalancing >> data\daily_run.log 2>&1

:: Step 4 -- squeeze scanner (M7 -- separate $2,000 budget)
python -m m7_squeeze.scanner >> data\daily_run.log 2>&1

echo %date% %time% -- run_daily.bat complete >> data\daily_run.log
