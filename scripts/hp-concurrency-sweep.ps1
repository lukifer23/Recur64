# Recur64 HP concurrency sweep (H0.7).
#
# Drives `recur64 selfplay` over a grid of hardware-scheduling parameters and
# records the ACTUAL concurrency evidence, not the configured values:
#   - peak_in_flight  : simultaneous evaluator calls (must be > 1)
#   - batch p50/p95/max : coalescing (p50 == 1 means batches are not forming)
#   - games / plies / elapsed : throughput
#
# If batch p50 stays at 1 across the grid, STOP AND DIAGNOSE; do not trust the
# sweep. See docs/HP_EXPERIMENT.md (Section: concurrency).
#
# Usage:
#   powershell -NoProfile -File scripts/hp-concurrency-sweep.ps1 `
#       -BaseConfig configs/smoke-cuda.toml -OutDir runs/hp-concurrency-sweep

param(
    [string]$BaseConfig = "configs/smoke-cuda.toml",
    [string]$Bin = "target\release\recur64.exe",
    [string]$OutDir = "runs/hp-concurrency-sweep",
    [int[]]$ActiveGames = @(8, 16, 32),
    [int[]]$CpuWorkers = @(2, 4, 6),
    [int[]]$MaxBatch = @(16, 32, 64),
    [int[]]$TimeoutUs = @(500, 2000),
    [int]$SimulationsPerMove = 0,   # 0 = use the base config value
    [int]$PlyCap = 0,               # 0 = use the base config value
    [int]$Repeat = 1
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Bin)) { throw "binary not found: $Bin (build first)" }
if (-not (Test-Path $BaseConfig)) { throw "base config not found: $BaseConfig" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$rows = New-Object System.Collections.Generic.List[object]
$runIndex = 0

foreach ($ag in $ActiveGames) {
    foreach ($cw in $CpuWorkers) {
        if ($cw -gt $ag) { continue }          # more workers than games is pointless
        if ($ag % $cw -ne 0) { continue }      # keep the work split even
        foreach ($mb in $MaxBatch) {
            foreach ($tus in $TimeoutUs) {
                for ($rep = 1; $rep -le $Repeat; $rep++) {
                    $runIndex++
                    $name = "ag$ag-cw$cw-mb$mb-to$tus-r$rep"
                    $dir = Join-Path $OutDir $name
                    $flags = @(
                        "--active-games", "$ag",
                        "--cpu-workers", "$cw",
                        "--max-inference-batch", "$mb",
                        "--batch-timeout-us", "$tus"
                    )
                    if ($SimulationsPerMove -gt 0) { $flags += @("--simulations-per-move", "$SimulationsPerMove") }
                    if ($PlyCap -gt 0) { $flags += @("--ply-cap", "$PlyCap") }

                    $sw = [System.Diagnostics.Stopwatch]::StartNew()
                    $raw = & $Bin selfplay --config $BaseConfig --output $dir @flags 2>&1 | Out-String
                    $sw.Stop()

                    $i = $raw.IndexOf('{')
                    if ($i -lt 0) {
                        Write-Warning "run $name produced no metrics JSON; raw: $raw"
                        continue
                    }
                    $m = $raw.Substring($i) | ConvertFrom-Json
                    $inf = $m.inference
                    $rows.Add([pscustomobject]@{
                        name                   = $name
                        active_games           = $ag
                        cpu_workers            = $cw
                        max_inference_batch    = $mb
                        batch_timeout_us       = $tus
                        repeat                 = $rep
                        games                  = $m.games
                        plies                  = $m.plies
                        peak_in_flight         = $m.peak_in_flight_evaluations
                        batch_mean             = [math]::Round($inf.batch_size_mean, 3)
                        batch_p50              = $inf.batch_size_p50
                        batch_p95              = $inf.batch_size_p95
                        batch_max              = $inf.batch_size_max
                        submitted              = $inf.submitted
                        errors                 = $inf.errors
                        queue_wait_p95_us      = $inf.queue_wait_us_p95
                        forward_us_mean        = [math]::Round($inf.forward_us_mean, 1)
                        wall_secs              = [math]::Round($sw.Elapsed.TotalSeconds, 2)
                    })
                    Write-Output ("[{0}] {1}: peak_in_flight={2} batch p50/p95/max={3}/{4}/{5} games={6} wall={7}s" -f `
                        $runIndex, $name, $m.peak_in_flight_evaluations, $inf.batch_size_p50, `
                        $inf.batch_size_p95, $inf.batch_size_max, $m.games, [math]::Round($sw.Elapsed.TotalSeconds, 1))
                }
            }
        }
    }
}

$jsonPath = Join-Path $OutDir "sweep.json"
$csvPath = Join-Path $OutDir "sweep.csv"
$rows | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath
$rows | Export-Csv -Path $csvPath -NoTypeInformation

Write-Output ""
Write-Output "wrote $jsonPath and $csvPath ($($rows.Count) runs)"

# Hard-stop check.
$valid = $rows | Where-Object { $_.games -gt 0 }
$maxPeak = ($valid | Measure-Object -Property peak_in_flight -Maximum).Maximum
$maxP50 = ($valid | Measure-Object -Property batch_p50 -Maximum).Maximum
Write-Output "best peak_in_flight=$maxPeak  best batch_p50=$maxP50"
if ($maxPeak -le 1) {
    Write-Output "STOP AND DIAGNOSE: no run achieved real concurrency (peak_in_flight <= 1)."
    exit 2
}
if ($maxP50 -le 1) {
    Write-Output "STOP AND DIAGNOSE: batch p50 never exceeded 1; batches are not forming."
    exit 3
}
Write-Output "OK: real concurrency and batch coalescing both observed."
