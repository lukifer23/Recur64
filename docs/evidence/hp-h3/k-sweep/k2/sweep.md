# Recur64 runtime sweep (Stage A)

- warmup: 2.91s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 16 | 16 | 32 | 32 | 1000 | 64 | 497.9 | 188.0 | 7.8 | 7.8 | 18.20/18/32 | 10413 | 298 |
