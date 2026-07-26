# Usage: .\studies\RunStudy_CUDA.ps1 -MaxCPU 55
param(
    [ValidateRange(1, 100)]
    [int]$MaxCPU = 75,

    [ValidateRange(1, 100)]
    [int]$MaxGPU = 75
)

$total = 100

function Get-MaxGpuUsagePercent {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw "nvidia-smi was not found on PATH; cannot read GPU utilization."
    }

    $lines = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
    if (-not $lines) {
        throw "nvidia-smi returned no GPU utilization values."
    }

    $values = $lines | ForEach-Object {
        $trimmed = $_.ToString().Trim()
        if ($trimmed -match '^\d+$') { [int]$trimmed }
    }

    if (-not $values -or $values.Count -eq 0) {
        throw "Unable to parse GPU utilization from nvidia-smi output."
    }

    return ($values | Measure-Object -Maximum).Maximum
}

$i = 1
while ($i -le $total) {

    $cpuUsage = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    try {
        $gpuUsage = Get-MaxGpuUsagePercent
    }
    catch {
        Write-Host "GPU usage check failed: $($_.Exception.Message)"
        break
    }

    if ($cpuUsage -lt $MaxCPU -and $gpuUsage -lt $MaxGPU) {
        Write-Host "CPU is below $MaxCPU%: $cpuUsage"
        Write-Host "GPU is below $MaxGPU%: $gpuUsage"

        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "`$env:MNIST_DEVICE='cuda'; d:; cd D:\SeanDevLocal\ElsaediDeepLearning\; uv run .\studies\Optuna_batchsize.py"

        Write-Host "Launched instance $i"
    }
    else {
        if ($cpuUsage -ge $MaxCPU) {
            Write-Host "CPU is too high (threshold $MaxCPU%): $cpuUsage"
        }
        if ($gpuUsage -ge $MaxGPU) {
            Write-Host "GPU is too high (threshold $MaxGPU%): $gpuUsage"
        }
        $i--
    }


    if ($i -lt $total) {
        Start-Sleep -Seconds 90
    }

    $i++
}

Write-Host "All $total instances launched."