$grid = @(
    @{n=100; d=3; lr=0.05},
    @{n=100; d=5; lr=0.1 },
    @{n=200; d=5; lr=0.1 },
    @{n=200; d=7; lr=0.05},
    @{n=300; d=7; lr=0.1 }
)
foreach ($g in $grid) {
    $name = "exp_$($g.n)_$($g.d)_$($g.lr)"
    dvc exp run --name $name `
        --set-param "train.n_estimators=[$($g.n)]" `
        --set-param "train.max_depth=[$($g.d)]" `
        --set-param "train.learning_rate=[$($g.lr)]"
}
dvc exp show --no-pager --csv > artifacts/experiments.csv
Write-Host "experiments -> artifacts/experiments.csv" -ForegroundColor Green
