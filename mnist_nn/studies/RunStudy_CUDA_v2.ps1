# Usage: .\studies\RunStudy_CUDA_v2.ps1 -MaxInstances 4 -MaxCPU 80
# Study v2: refined hyperparameter ranges based on Study 1 analysis
# With 32 cores, running 3-4 instances in parallel is optimal
param(
    [ValidateRange(1, 100)]
    [int]$MaxInstances = 4,

    [ValidateRange(1, 100)]
    [int]$MaxCPU = 80,

    [ValidateRange(1, 100)]
    [int]$MaxGPU = 75
)

$total = $MaxInstances

function Get-MaxGpuUsagePercent {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Host "nvidia-smi not found; skipping GPU check"
        return 0
    }

    try {
        $output = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1
        $lines = @($output) | Where-Object { $_ }

        if (-not $lines) {
            Write-Host "nvidia-smi returned empty output"
            return 0
        }

        $values = @()
        foreach ($line in $lines) {
            $trimmed = $line.ToString().Trim()
            if ($trimmed -match '^\d+') {
                $num = [int]($trimmed -replace '\D.*', '')
                $values += $num
            }
        }

        if ($values.Count -eq 0) {
            Write-Host "Could not parse GPU values from: $output"
            return 0
        }

        return ($values | Measure-Object -Maximum).Maximum
    }
    catch {
        Write-Host "GPU check exception: $($_.Exception.Message)"
        return 0
    }
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
            "`$env:MNIST_DEVICE='cuda'; d:; cd D:\SeanDevLocal\ElsaediDeepLearning\; uv run .\studies\Optuna_batchsize_v2.py"

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
