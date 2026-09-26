# Recur64 runtime sweep (Stage A)

- warmup: 3.09s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 64 | 479.1 | 180.9 | 7.5 | 7.5 | 11.36/15/16 | 10106 | 298 |
