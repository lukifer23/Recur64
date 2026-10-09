# Pre-registered T1 grid (T1_CONTRACT §2), sequential, each run under the bounded launcher.
$root = (Resolve-Path "$PSScriptRoot\..\..").Path; $art = "$root\artifacts\v69"
$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"; $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$exe = "$root\target-v69-cuda\release\v69-t1-train.exe"; $runs = @()
foreach ($k in 250,1000,2000) { foreach ($aug in 'off','d8') { foreach ($lr in '3e-4','1e-3','3e-3') { $runs += ,@('M',$k,$aug,$lr,30) } } }
$runs += ,@('A',2000,'d8','5e-4',12); $runs += ,@('A',2000,'off','5e-4',12); $runs += ,@('A',1000,'d8','5e-4',20); $runs += ,@('A',250,'d8','5e-4',40)
New-Item -ItemType Directory -Force "$art\t1\grid_logs" | Out-Null
foreach ($r in $runs) {
  $name = "$($r[0])_k$($r[1])_$($r[2])_lr$($r[3])"
  & "$root\scripts\v69\run_limited.ps1" -Exe $exe -CmdArgs @('--artifacts',$art,'--model',$r[0],'--k',"$($r[1])",'--aug',$r[2],'--lr',$r[3],'--epochs',"$($r[4])") -WallSecs 3000 -MemGiB 8 -GpuMiB 3072 -Log "$art\t1\grid_logs\$name.log"
}
"GRID DONE"
