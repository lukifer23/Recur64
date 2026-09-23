# Recur64 HP search-budget comparison (H1.1).
#
# Runs the SAME model/scheduling at several simulations-per-move values and
# reports move latency, games/hour, leaf evaluations/sec, and batch behaviour.
# The GPU is monitored in-process so peak VRAM/util/power/temp are recorded per
# budget.
#
# This chooses the learning/search budget. The chosen value is frozen before any
# training; changing it later is a new experiment identity.
#
# Usage (invoke with &, not `powershell -File`, so arrays bind correctly):
#   & .\scripts\hp-search-budget.ps1 -BaseConfig configs/hp/f15-selfplay.toml `
#       -SimValues 64,128,256 -ActiveGames 32 -CpuWorkers 8

param(
    [string]$BaseConfig = "configs/hp/f15-selfplay.toml",
    [string]$Bin = "target\release\recur64.exe",
    [string]$OutDir = "runs/hp-search-budget",
    [int[]]$SimValues = @(64, 128, 256),
    [int]$ActiveGames = 32,
    [int]$CpuWorkers = 8,
    [int]$MaxBatch = 64,
    [int]$TimeoutUs = 2000,
    [int]$PlyCap = 120,
    [int]$Repeat = 1
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bin)) { throw "binary not found: $Bin" }
if (-not (Test-Path $BaseConfig)) { throw "base config not found: $BaseConfig" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

$rows = New-Object System.Collections.Generic.List[object]

foreach ($sims in $SimValues) {
    for ($rep = 1; $rep -le $Repeat; $rep++) {
        $name = "sims$sims-r$rep"
        $dir = Join-Path $OutDir $name
        $monCsv = Join-Path $OutDir "gpu-$name.csv"
        Remove-Item $monCsv -ErrorAction SilentlyContinue

        # In-process GPU sampler job.
        $job = Start-Job -ScriptBlock {
            param($csv)
            for ($i = 0; $i -lt 6000; $i++) {
                $s = & nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits
                Add-Content -Path $csv -Value $s
                Start-Sleep -Milliseconds 500
            }
        } -ArgumentList $monCsv

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $raw = & $Bin selfplay --config $BaseConfig --output $dir `
                --active-games $ActiveGames --cpu-workers $CpuWorkers `
                --max-inference-batch $MaxBatch --batch-timeout-us $TimeoutUs `
                --simulations-per-move $sims --ply-cap $PlyCap 2>&1 | Out-String
        } finally {
            $sw.Stop()
            Stop-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force -ErrorAction SilentlyContinue
        }

        $i = $raw.IndexOf('{')
        if ($i -lt 0) { Write-Warning "no metrics JSON for $name"; continue }
        $m = $raw.Substring($i) | ConvertFrom-Json
        $inf = $m.inference
        $wall = $sw.Elapsed.TotalSeconds

        $gpu = $null
        if (Test-Path $monCsv) {
            $samples = Get-Content $monCsv | Where-Object { $_ -match '\d' } | ForEach-Object {
                $p = $_ -split ',' | ForEach-Object { $_.Trim() }
                [pscustomobject]@{ mem=[int]$p[0]; util=[int]$p[1]; power=[double]$p[2]; temp=[int]$p[3]; clk=[int]$p[4] }
            }
            if ($samples) {
                $gpu = [pscustomobject]@{
                    peak_vram_mib = ($samples | Measure-Object mem -Maximum).Maximum
                    peak_util     = ($samples | Measure-Object util -Maximum).Maximum
                    mean_util     = [math]::Round((($samples | Where-Object { $_.util -gt 0 }) | Measure-Object util -Average).Average, 1)
                    peak_power_w  = ($samples | Measure-Object power -Maximum).Maximum
                    peak_temp_c   = ($samples | Measure-Object temp -Maximum).Maximum
                }
            }
        }

        $plies = [int]$m.plies
        $rows.Add([pscustomobject]@{
            name            = $name
            simulations     = $sims
            repeat          = $rep
            games           = $m.games
            plies           = $plies
            mean_game_plies = [math]::Round($m.mean_game_plies, 1)
            wall_secs       = [math]::Round($wall, 2)
            moves_per_sec   = if ($wall -gt 0) { [math]::Round($plies / $wall, 3) } else { 0 }
            ms_per_move     = if ($plies -gt 0) { [math]::Round($wall * 1000.0 / $plies, 1) } else { 0 }
            games_per_hour  = if ($wall -gt 0) { [math]::Round($m.games * 3600.0 / $wall, 1) } else { 0 }
            evals_per_sec   = if ($wall -gt 0) { [math]::Round($inf.submitted / $wall, 1) } else { 0 }
            batch_mean      = [math]::Round($inf.batch_size_mean, 3)
            batch_p50       = $inf.batch_size_p50
            batch_p95       = $inf.batch_size_p95
            peak_in_flight  = $m.peak_in_flight_evaluations
            errors          = $inf.errors
            peak_vram_mib   = if ($gpu) { $gpu.peak_vram_mib } else { $null }
            peak_util       = if ($gpu) { $gpu.peak_util } else { $null }
            mean_util       = if ($gpu) { $gpu.mean_util } else { $null }
            peak_power_w    = if ($gpu) { $gpu.peak_power_w } else { $null }
            peak_temp_c     = if ($gpu) { $gpu.peak_temp_c } else { $null }
        })

        Write-Output ("sims={0} wall={1}s ms/move={2} moves/s={3} evals/s={4} batch p50/p95={5}/{6} peak_in_flight={7}" -f `
            $sims, [math]::Round($wall, 1), $rows[-1].ms_per_move, $rows[-1].moves_per_sec, `
            $rows[-1].evals_per_sec, $inf.batch_size_p50, $inf.batch_size_p95, $m.peak_in_flight_evaluations)
    }
}

$rows | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $OutDir "search-budget.json")
$rows | Export-Csv -Path (Join-Path $OutDir "search-budget.csv") -NoTypeInformation
Write-Output ""
Write-Output "wrote $OutDir\search-budget.json and .csv ($($rows.Count) runs)"
