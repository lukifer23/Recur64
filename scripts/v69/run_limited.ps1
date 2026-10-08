# Run a V69 command under hard wall-time and memory limits.
#   run_limited.ps1 -Exe <path> -Args <string[]> -WallSecs 7200 -MemGiB 8 -Log <path>
# - Captures the native exit code (written to <Log>.exit and returned).
# - On wall/memory breach, kills the whole owned process tree (taskkill /T /F).
# - Records peak working set (MiB) to <Log>.peak.
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [string[]]$CmdArgs = @(),
  [int]$WallSecs = 7200,
  [double]$MemGiB = 8,
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
$reason = 'completed'
while (-not $p.HasExited) {
  Start-Sleep -Milliseconds 500
  try { $p.Refresh(); $ws = $p.WorkingSet64; $pm = $p.PrivateMemorySize64 } catch { break }
  $m = [Math]::Max($ws, $pm)
  if ($m -gt $peak) { $peak = $m }
  if ($m -gt $memLimit) { $reason = 'killed_memory_limit'; break }
  if ($sw.Elapsed.TotalSeconds -gt $WallSecs) { $reason = 'killed_wall_limit'; break }
}
if ($reason -ne 'completed') {
  & taskkill.exe /PID $p.Id /T /F | Out-Null
  $p.WaitForExit()
}
$p.WaitForExit()
$code = if ($reason -eq 'completed') { $p.ExitCode } else { -1 }
"peak_mib=$([int]($peak / 1MB))" | Out-File "$Log.peak" -Encoding ascii
"reason=$reason exit_code=$code wall_secs=$([int]$sw.Elapsed.TotalSeconds)" | Out-File "$Log.exit" -Encoding ascii
Write-Output "reason=$reason exit_code=$code wall_secs=$([int]$sw.Elapsed.TotalSeconds) peak_mib=$([int]($peak / 1MB))"
exit $code
