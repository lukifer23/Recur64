# Pre-registered T3 runs (T3_CONTRACT section 1), sequential, each under the bounded launcher.
$root = (Resolve-Path "$PSScriptRoot\..\..").Path; $art = "$root\artifacts\v69"
$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"; $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$exe = "$root\target-v69-cuda2\release\v69-t3.exe"
New-Item -ItemType Directory -Force "$art\t3\grid_logs" | Out-Null
foreach ($recipe in 'S','D') { foreach ($fold in 'KQQvK','KQRvK','KRRvK') {
  & "$root\scripts\v69\run_limited.ps1" -Exe $exe -CmdArgs @('run','--artifacts',$art,'--fold',$fold,'--recipe',$recipe) -WallSecs 3000 -MemGiB 8 -GpuMiB 3072 -Log "$art\t3\grid_logs\${fold}_${recipe}.log"
} }
"GRID DONE"
