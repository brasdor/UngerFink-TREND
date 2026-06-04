# setup_task_scheduler.ps1
# Registers UngerFink T9B Daily in Windows Task Scheduler.
# No admin required -- runs as the current logged-in user.
# Usage: powershell -ExecutionPolicy Bypass -File setup_task_scheduler.ps1

$TaskName = "UngerFink T9B Daily"
$WorkDir  = "C:\Users\Jean\UngerFink-TREND"
$BatFile  = "$WorkDir\run_t9b_daily.bat"
$RunTime  = "09:00"

New-Item -ItemType Directory -Force -Path "$WorkDir\logs" | Out-Null

if (-not (Test-Path $BatFile)) {
    Write-Error "Batch file not found: $BatFile"; exit 1
}

# Step 1: register via schtasks.exe (works without admin for own-user tasks)
schtasks /delete /tn $TaskName /f 2>$null | Out-Null
$out = schtasks /create /tn "$TaskName" /tr "`"$BatFile`"" /sc DAILY /st $RunTime /f 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host $out; Write-Error "schtasks failed"; exit 1 }

# Step 2: patch battery settings (Get/Set-ScheduledTask works without admin)
try {
    $task = Get-ScheduledTask -TaskName $TaskName
    $task.Settings.DisallowStartIfOnBatteries = $false
    $task.Settings.StopIfGoingOnBatteries     = $false
    Set-ScheduledTask -InputObject $task | Out-Null
    Write-Host "  Battery : runs on battery power (patched)"
} catch {
    Write-Host "  Battery : manual fix needed -- open Task Scheduler GUI, edit task, Conditions tab, uncheck battery options"
}

Write-Host ""
Write-Host "Task registered: $TaskName"
Write-Host "  Next run : tomorrow at $RunTime"
Write-Host "  Runs as  : $env:USERNAME (interactive -- must be logged in)"
Write-Host "  Log      : $WorkDir\logs\t9b_task_scheduler.log"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Run now   : schtasks /run /tn `"$TaskName`""
Write-Host "  Status    : schtasks /query /tn `"$TaskName`" /fo list"
Write-Host "  Delete    : schtasks /delete /tn `"$TaskName`" /f"
