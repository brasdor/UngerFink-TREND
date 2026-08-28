@echo off
REM ---------------------------------------------------------------------------
REM Daily local funding-rate refresh.
REM
REM fapi.binance.com returns HTTP 451 to GitHub's US runners, and binance.vision
REM publishes funding only monthly, so the daily workflow cannot refresh funding
REM at all -- it just stales. This runs the refresh from THIS machine (which can
REM reach Binance) and pushes the result, so the regime funding axis and the
REM S6/S7/S8 funding gates read current carry.
REM
REM Register with Task Scheduler via setup_task_scheduler.ps1, or run by hand.
REM Log: logs\funding_refresh.log
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
echo. >> "logs\funding_refresh.log"
echo ==== %DATE% %TIME% ==== >> "logs\funding_refresh.log"
python tools\refresh_funding_local.py >> "logs\funding_refresh.log" 2>&1
echo exit=%ERRORLEVEL% >> "logs\funding_refresh.log"
