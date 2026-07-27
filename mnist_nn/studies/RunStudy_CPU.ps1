# Usage: .\studies\RunStudy_CPU.ps1 -MaxCPU 55
param(
    [ValidateRange(1, 100)]
    [int]$MaxCPU = 75
)

$total = 100

$i = 1
while ($i -le $total) {

    $cpuUsage = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue

    if ($cpuUsage -lt $MaxCPU) {
        Write-Host "CPU is below $MaxCPU%: $cpuUsage"

        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "`$env:MNIST_DEVICE='cpu'; d:; cd D:\SeanDevLocal\ElsaediDeepLearning\; uv run .\studies\Optuna_batchsize.py"

        Write-Host "Launched instance $i"
    }
    else {
        Write-Host "CPU is too high (threshold $MaxCPU%): $cpuUsage"
        $i--
    }


    if ($i -lt $total) {
        Start-Sleep -Seconds 90
    }

    $i++
}

Write-Host "All $total instances launched."