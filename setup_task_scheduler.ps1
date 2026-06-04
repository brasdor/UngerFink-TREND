# setup_task_scheduler.ps1
# Registers UngerFink T9B Daily as a Windows Task Scheduler task.
# Run once as Administrator: powershell -ExecutionPolicy Bypass -File setup_task_scheduler.ps1

$TaskName    = "UngerFink T9B Daily"
$WorkDir     = "C:\Users\Jean\UngerFink-TREND"
$BatFile     = "$WorkDir\run_t9b_daily.bat"
$RunTime     = "09:00"   # 09:00 local time (Binance candle closes 00:00 UTC = 02:00 CET)
$LogFile     = "$WorkDir\logs\t9b_task_scheduler.log"

# Create logs directory
New-Item -ItemType Directory -Force -Path "$WorkDir\logs" | Out-Null

# Verify bat file exists
if (-not (Test-Path $BatFile)) {
    Write-Error "Batch file not found: $BatFile"
    exit 1
}

# Remove existing task if present (clean re-register)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Action: cmd.exe runs the bat file, stdout+stderr redirected to log
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`" >> `"$LogFile`" 2>&1" `
    -WorkingDirectory $WorkDir

# Trigger: daily at 09:00
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime

# Settings: run whether logged on or not, restart on failure, no time limit
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# Principal: run as current user, highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Highest

# Register
$Task = Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "Runs all three UngerFink T9B paper trading engines daily and commits output to GitHub."

if ($Task) {
    Write-Host ""
    Write-Host "Task registered successfully."
    Write-Host "  Name     : $($Task.TaskName)"
    Write-Host "  Trigger  : Daily at $RunTime"
    Write-Host "  Action   : $BatFile"
    Write-Host "  Log file : $LogFile"
    Write-Host "  Run as   : $env:USERNAME (no login required)"
    Write-Host ""
    Write-Host "To run manually right now:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "To check last run status:"
    Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
} else {
    Write-Error "Failed to register task."
    exit 1
}
