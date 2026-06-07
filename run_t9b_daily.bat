@echo off
setlocal enabledelayedexpansion

cd /d C:\Users\Jean\UngerFink-TREND

:: ISO date via PowerShell (locale-independent)
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd'"') do set ISODATE=%%i

echo ============================================================
echo UngerFink T9B Daily Run -- %ISODATE% %time%
echo ============================================================

echo.
echo --- Donchian ---
python phase_t9b_donchian_universev2_paper_engine.py --notify
if errorlevel 1 echo [WARN] Donchian engine exited with error

echo.
echo --- RSI Mean Reversion ---
python phase_t9b_meanreversion_paper_engine.py --notify
if errorlevel 1 echo [WARN] MR engine exited with error

echo.
echo --- ConsecDownDays ---
python phase_t9b_consecdowndays_paper_engine.py --notify
if errorlevel 1 echo [WARN] ConsecDownDays engine exited with error

echo.
echo --- Momentum Factor ---
python phase_t9b_momentum_factor_paper_engine.py --notify
if errorlevel 1 echo [WARN] Momentum engine exited with error

echo.
echo --- Combined Summary ---
python t9b_combined_summary.py

echo.
echo --- Committing to GitHub ---
git add data/t9b_paper/ data/t9b_mr_paper/ data/t9b_consecdowndays_paper/ data/t9b_momentum_paper/ data/universe/ohlcv_1d/ data/futures_universe/ohlcv_1d/
git commit -m "T9B daily update %ISODATE% [skip ci]"
if errorlevel 1 (
    echo No changes to commit or commit failed
) else (
    git pull --rebase origin master
    git push origin HEAD:master
    if errorlevel 1 echo [WARN] Git push failed
)

echo.
echo ============================================================
echo Done -- %time%
echo ============================================================

endlocal
