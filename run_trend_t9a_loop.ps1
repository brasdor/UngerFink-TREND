cd C:\Users\Jean\Desktop\UngerFink_TREND

while ($true) {

    Clear-Host

    Write-Host "==================================================="
    Write-Host "TREND T9A V2 PAPER LOOP"
    Write-Host "$(Get-Date)"
    Write-Host "==================================================="

    python phase_t9a_binance_paper_sim_engine_V2.py

    Write-Host ""
    Write-Host "Sleeping 15 minutes..."
    Start-Sleep -Seconds 900
}