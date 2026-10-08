# Run a V69 command under hard wall-time, host-memory and device-memory limits.
#   run_limited.ps1 -Exe <path> -CmdArgs <string[]> -WallSecs 7200 -MemGiB 8 -GpuMiB 3072 -Log <path>
# - Arguments containing spaces are quoted; stdout -> <Log>.out, stderr -> <Log>.
# - Captures the native exit code (<Log>.exit, and returned). Kills return -1.
# - On wall/host-memory/device-memory breach, kills the whole owned process tree
#   (taskkill /T /F) and then verifies that no owned descendant survived.
# - Records peak host memory (main process: max of working set / private bytes, 500 ms poll),
#   sampled peak whole-GPU used memory (nvidia-smi, ~2 s poll) and orphan count.
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [string[]]$CmdArgs = @(),
  [int]$WallSecs = 7200,
  [double]$MemGiB = 8,
  [int]$GpuMiB = 0,
  [Parameter(Mandatory = $true)][string]$Log
)
$ErrorActionPreference = 'Stop'
$memLimit = [int64]($MemGiB * 1GB)
$CmdArgs = @($CmdArgs | ForEach-Object { if ($_.Contains(' ')) { "`"$_`"" } else { $_ } })
$p = Start-Process -FilePath $Exe -ArgumentList $CmdArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput "$Log.out" -RedirectStandardError $Log
$null = $p.Handle  # cache handle so ExitCode is readable
$sw = [Diagnostics.Stopwatch]::StartNew()
$peak = 0L
$gpuPeak = 0
$owned = New-Object System.Collections.Generic.HashSet[int]
[void]$owned.Add($p.Id)
function Get-Descendants([int]$rootPid) {
  $all = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Select-Object ProcessId, ParentProcessId
  $found = New-Object System.Collections.Generic.List[int]
  $queue = New-Object System.Collections.Generic.Queue[int]
  $queue.Enqueue($rootPid)
  while ($queue.Count -gt 0) {
    $cur = $queue.Dequeue()
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $cur })) { $found.Add($c.ProcessId); $queue.Enqueue($c.ProcessId) }
  }
  return $found
}
$reason = 'completed'
$tick = 0
while (-not $p.HasExited) {
  Start-Sleep -Milliseconds 500
  $tick++
  try { $p.Refresh(); $ws = $p.WorkingSet64; $pm = $p.PrivateMemorySize64 } catch { break }
  $m = [Math]::Max($ws, $pm)
  if ($m -gt $peak) { $peak = $m }
  if ($m -gt $memLimit) { $reason = 'killed_memory_limit'; break }
  if ($sw.Elapsed.TotalSeconds -gt $WallSecs) { $reason = 'killed_wall_limit'; break }
  if ($tick % 4 -eq 0) {
    foreach ($d in (Get-Descendants $p.Id)) { [void]$owned.Add($d) }
    if ($GpuMiB -gt 0 -or $tick % 20 -eq 0) {
      $g = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
      if ($g -match '^\s*\d+') {
        $gi = [int]$g.Trim()
        if ($gi -gt $gpuPeak) { $gpuPeak = $gi }
        if ($GpuMiB -gt 0 -and $gi -gt $GpuMiB) { $reason = 'killed_device_memory_limit'; break }
      }
    }
  }
}
if ($reason -ne 'completed') {
  foreach ($d in (Get-Descendants $p.Id)) { [void]$owned.Add($d) }
  & taskkill.exe /PID $p.Id /T /F | Out-Null
  $p.WaitForExit()
}
$p.WaitForExit()
Start-Sleep -Milliseconds 300
$orphans = @($owned | Where-Object { $_ -ne $p.Id -or $reason -ne 'completed' } | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count
$code = if ($reason -eq 'completed') { $p.ExitCode } else { -1 }
"peak_host_mib=$([int]($peak / 1MB)) peak_gpu_mib_sampled=$gpuPeak" | Out-File "$Log.peak" -Encoding ascii
"reason=$reason exit_code=$code wall_secs=$([int]$sw.Elapsed.TotalSeconds) orphans=$orphans owned_pids=$($owned.Count)" | Out-File "$Log.exit" -Encoding ascii
Write-Output "reason=$reason exit_code=$code wall_secs=$([int]$sw.Elapsed.TotalSeconds) peak_host_mib=$([int]($peak / 1MB)) peak_gpu_mib_sampled=$gpuPeak orphans=$orphans"
exit $code
