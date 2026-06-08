@echo off
setlocal

set TASK_NAME=CryptoTradingDailyPipeline
set BAT_PATH=C:\Users\Jean\UngerFink-TREND\trading\run_daily.bat
set WORK_DIR=C:\Users\Jean\UngerFink-TREND\trading

echo ============================================================
echo  setup_scheduler.bat
echo  Task : %TASK_NAME%
echo  Script: %BAT_PATH%
echo  Requires: run as Administrator
echo ============================================================
echo.

echo --- Equivalent schtasks command (daily trigger only, for reference) ---
echo schtasks /create /tn "%TASK_NAME%" /tr "%BAT_PATH%" /sc DAILY /st 06:00 /f /rl HIGHEST
echo.

echo --- Full PowerShell command that will be executed: ---
echo $a  = New-ScheduledTaskAction -Execute '%BAT_PATH%' -WorkingDirectory '%WORK_DIR%'
echo $t1 = New-ScheduledTaskTrigger -Daily -At '06:00'
echo $t2 = New-ScheduledTaskTrigger -AtStartup
echo $t2.Delay = 'PT2M'
echo $s  = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew -RunOnlyIfNetworkAvailable:$false
echo $p  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
echo Register-ScheduledTask -TaskName '%TASK_NAME%' -Action $a -Trigger $t1,$t2 -Settings $s -Principal $p -Force
echo.
echo --------------------------------------------------------
echo.

:: Execute via PowerShell.
:: $env:USERNAME is a PowerShell expression -- not expanded by CMD ($ is not special in CMD).
:: %BAT_PATH% and %WORK_DIR% are expanded by CMD before PowerShell sees them.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$a=New-ScheduledTaskAction -Execute '%BAT_PATH%' -WorkingDirectory '%WORK_DIR%';" ^
 "$t1=New-ScheduledTaskTrigger -Daily -At '06:00';" ^
 "$t2=New-ScheduledTaskTrigger -AtStartup; $t2.Delay='PT2M';" ^
 "$s=New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew -RunOnlyIfNetworkAvailable:$false;" ^
 "$p=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest;" ^
 "Register-ScheduledTask -TaskName '%TASK_NAME%' -Action $a -Trigger $t1,$t2 -Settings $s -Principal $p -Force"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Task Scheduler registration FAILED.
    echo Make sure this batch file is running as Administrator.
) else (
    echo.
    echo Task registered successfully.
    echo.
    schtasks /query /tn "%TASK_NAME%" /fo LIST
)

endlocal
