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
| V5 graph | NOT TESTED |

V5 commands use a process-local environment only. They load VS Build Tools and
prepend the pinned CUDA `bin`; they do not modify PATH, registry, drivers,
security settings or Windows features.

The preferred resident peak is <=3,072 MiB. Reader qualification starts at
physical batch 2, Q8/R4, both streams and full backward. Batch 1 with equivalent
accumulation is the only authorized reader fallback.

