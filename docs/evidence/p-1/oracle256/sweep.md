# Recur64 runtime sweep (Stage A)

- warmup: 3.36s | games/cell: 8

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 256 | 431.4 | 45.9 | 1.7 | 1.7 | 8.66/10/16 | 9801 | 289 |
