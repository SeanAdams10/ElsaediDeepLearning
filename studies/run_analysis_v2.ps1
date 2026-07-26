param(
    [string]$StudyName = "Optuna_batchsize_v2",
    [string]$StudyDb = "studies/Optuna_batchsize_v2.db",
    [switch]$ConvergenceOnly = $false
)

if (-not $ConvergenceOnly) {
    Write-Host "Running full analysis..." -ForegroundColor Cyan
    uv run python studies/optuna_analysis.py --study-name $StudyName --study-db $StudyDb
}

Write-Host "`nRunning convergence check..." -ForegroundColor Cyan
uv run python studies/optuna_convergence_check.py --study-name $StudyName --study-db $StudyDb
