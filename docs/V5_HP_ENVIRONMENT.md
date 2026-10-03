# V5 HP environment

Recorded 2026-10-03. DETECTED is not TESTED.

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
| V5 graph | TESTED: release CPU and intended RTX 2050 CUDA, FP32, physical microbatch 2 |

V5 commands use a process-local environment only. They load VS Build Tools and
prepend the pinned CUDA `bin`; they do not modify PATH, registry, drivers,
security settings or Windows features.

The preferred resident peak is <=3,072 MiB. The actual paired graph passed at
physical batch 2, Q8/R4, both streams and full backward; that layout is now
frozen. The microbatch-1 fallback was not used.

The CUDA qualification ran through the pinned user-space CUDA path and the
intended RTX 2050. NVIDIA device-wide used memory rose from 144 MiB before the
run to a measured peak of 338 MiB (194 MiB delta). Because WDDM did not expose
reliable process-resident memory, this is reported as device-wide rather than
mislabelled as a process peak. Other GPU workloads remained active. The worst
warm qualification update was 0.4693646 s on CUDA and 0.7318495 s on CPU.
