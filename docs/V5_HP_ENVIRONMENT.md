# V5 HP environment

Recorded 2026-10-03. DETECTED is not TESTED.

Current TESTED functional fixture graph: source
`64c4dd4008a9b5bc8d715515279ec67e6e1173a1`, FP32, microbatch 2, CPU and the
intended RTX 2050 CUDA backend. Both-stream Q2/Q4/Q8 × R1/R2/R4 forward/backward/
AdamW, 50 resident Q8/R4 updates, exact null/baseline/reference/resume checks pass.
CUDA device-wide sampled baseline/peak 138/1,068 MiB; every resident update
sample 364 MiB. Preferred 3,072 MiB envelope met in these fixture measurements;
this is not a process-only/continuous peak or a full-data training qualification.
Warm qualifier intervals: CPU maximum 0.7248367 s, CUDA maximum 0.4127371 s;
CUDA resident first/last 0.2029461/0.1985860 s. See V5_RESULTS for timing limits.
Physical microbatch 2 resolved, no fallback. The exact FIT drill remains NOT RUN.

The current process environment MUST set both CUDA_PATH to the existing pinned
12.9.1 root and prepend its bin to PATH. A PATH-only runtime attempt failed to
find cuda_runtime.h; the correct setup is in V5_RESUME. Native serial MSVC CUDA
build passes. Feature inventory contains no tf32, autotune or fusion; V5 rejects
TF32-enabled binaries. No drivers, system settings or unrelated processes changed.

Re-detection during the depth amendment: CPU remains Ryzen 5 7535HS, 6/12;
Windows 11 Home build 26300; visible RAM 31.20835 GiB, available approximately
19.25930 GiB; C: free approximately 56.265 GiB, D: free approximately 69.879 GiB.
RTX 2050 reports total 4,096 MiB, used 9 MiB, free 3,954 MiB, driver 616.92.
NVIDIA's free/used fields exclude some reserved memory and are reported directly.
ChatGPT is an existing GPU client; no unrelated process was stopped. These are
DETECTED snapshots, not amended-model CUDA measurements.

| Item | Status |
|---|---|
| OS | DETECTED: Windows 11 Home, build 26300 |
| CPU | DETECTED: Ryzen 5 7535HS, 6 cores / 12 threads |
| RAM | DETECTED: 31.21 GiB visible, 19.41 GiB free at inspection |
| GPU | DETECTED: NVIDIA GeForce RTX 2050, 4,096 MiB; 3,947 MiB free |
| Driver | DETECTED: 616.92 |
| Other GPU processes | DETECTED: LM Studio and ChatGPT; neither may be killed |
| Disk C: | DETECTED: 76.06 GiB free |
| Rust/Cargo | DETECTED: 1.97.1 MSVC, matching repository pin |
| MSVC | DETECTED: VS 2022 Build Tools with x64 C++ component; not loaded in ordinary shell |
| CUDA | DETECTED: user-space 12.9.1 under `%LOCALAPPDATA%\Recur64\cuda\12.9.1` |
| CUDA DLLs | DETECTED: cudart, NVRTC canonical DLL and alias, nvJitLink |
| V5 graph | TESTED current CPU/RTX 2050 functional fixture qualification PASS at 64c4dd4; exact FIT drill/data training NOT RUN |

V5 commands use a process-local environment only. They load VS Build Tools and
prepend the pinned CUDA `bin`; they do not modify PATH, registry, drivers,
security settings or Windows features.

The preferred resident peak is <=3,072 MiB. The actual paired graph passed at
physical batch 2, Q8/R4, both streams and full backward on the historical
network fixtures. Full engineering qualification is
now requires requalification after the depth and execution amendments in
`V5_ROOT_CAUSE.md` and `V5_EXECUTION_PARITY.md`. The microbatch-1 fallback was
not used.

The HISTORICAL pre-amendment CUDA qualification ran through the pinned path and the
intended RTX 2050. NVIDIA device-wide used memory rose from 144 MiB before the
run to a measured peak of 338 MiB (194 MiB delta). Because WDDM did not expose
reliable process-resident memory, this is reported as device-wide rather than
mislabelled as a process peak. Other GPU workloads remained active. The worst
warm historical-source qualification update was 0.4107268 s on CUDA and 0.7699340 s
on CPU.
