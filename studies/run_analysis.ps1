param(
    [string]$StudyName = "Optuna_batchsize",
    [string]$StudyDb = "studies/Optuna_batchsize.db"
)

uv run python studies/optuna_analysis.py --study-name $StudyName --study-db $StudyDb