# Self-test of run_limited.ps1: exit-code capture, argument handling (spaces),
# logging, wall-limit kill with owned-child cleanup, and memory-limit kill.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmp = Join-Path $env:TEMP "v69-launcher-test-$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
$ps = (Get-Command powershell.exe).Source
$results = @()
function Run([string]$name, [string[]]$a, [int]$wall, [double]$mem) {
  $log = Join-Path $tmp "$name.log"
  $out = & "$here\run_limited.ps1" -Exe $ps -CmdArgs $a -WallSecs $wall -MemGiB $mem -Log $log
  [pscustomobject]@{ name = $name; summary = ($out | Select-Object -Last 1); code = $LASTEXITCODE; log = $log }
}
# 1. exit code propagation
$r = Run 'exit7' @('-NoProfile', '-Command', 'exit 7') 30 8
$results += $r; if ($r.code -ne 7) { throw "exit code not captured: $($r.code)" }
# 2. argument handling with spaces and logging of stdout
$r = Run 'args' @('-NoProfile', '-Command', "Write-Output 'hello world with spaces'") 30 8
$results += $r; if ((Get-Content "$($r.log).out" -Raw) -notmatch 'hello world with spaces') { throw 'stdout/arg handling failed' }
# 3. wall-limit kill cleans up the owned child tree (parent spawns a grandchild)
$r = Run 'wall' @('-NoProfile', '-Command', "Start-Process -NoNewWindow -FilePath powershell.exe -ArgumentList '-NoProfile','-Command','Start-Sleep 120' ; Start-Sleep 120") 4 8
$results += $r
if ($r.summary -notmatch 'killed_wall_limit' -or $r.summary -notmatch 'orphans=0') { throw "wall kill/cleanup failed: $($r.summary)" }
# 4. host-memory-limit kill (allocate ~300 MiB with a 0.1 GiB limit)
$r = Run 'mem' @('-NoProfile', '-Command', '$a = New-Object byte[] 300000000; Start-Sleep 60') 60 0.1
$results += $r
if ($r.summary -notmatch 'killed_memory_limit' -or $r.summary -notmatch 'orphans=0') { throw "memory kill failed: $($r.summary)" }
$results | ForEach-Object { "$($_.name): $($_.summary)" }
"LAUNCHER SELF-TEST PASS"
Remove-Item -Recurse -Force $tmp
exit 0
