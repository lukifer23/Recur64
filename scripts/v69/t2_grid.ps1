# Pre-registered T2 grid (T2_CONTRACT §2), sequential, each run under the bounded launcher.
$root = (Resolve-Path "$PSScriptRoot\..\..").Path; $art = "$root\artifacts\v69"
$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"; $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$exe = "$root\target-v69-cuda2\release\v69-t2-train.exe"; $runs = @()
foreach ($k in 2000,6000) { foreach ($mode in 'd8','sym') { foreach ($lr in '3e-3','1e-2') { $runs += ,@('M',$k,$mode,$lr,30) } } }
$runs += ,@('M',6000,'d8','3e-3',60); $runs += ,@('M',6000,'sym','3e-3',60)
$runs += ,@('A',6000,'d8','5e-4',18); $runs += ,@('A',6000,'d8','1e-3',18); $runs += ,@('A',2000,'d8','5e-4',54)
New-Item -ItemType Directory -Force "$art\t2\grid_logs" | Out-Null
foreach ($r in $runs) {
  $name = "$($r[0])_k$($r[1])_$($r[2])_lr$($r[3])_e$($r[4])"
  & "$root\scripts\v69\run_limited.ps1" -Exe $exe -CmdArgs @('--artifacts',$art,'--model',$r[0],'--k',"$($r[1])",'--mode',$r[2],'--lr',$r[3],'--epochs',"$($r[4])") -WallSecs 4000 -MemGiB 8 -GpuMiB 3072 -Log "$art\t2\grid_logs\$name.log"
}
"GRID DONE"
