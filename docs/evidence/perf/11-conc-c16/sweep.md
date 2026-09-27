# Recur64 runtime sweep (Stage A)

- warmup: 5.79s | games/cell: 16

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 16 | 16 | 32 | 32 | 1000 | 32 | 655.8 | 269.3 | 20.6 | 13.1 | 20.61/22/32 | 10623 | 545 |
