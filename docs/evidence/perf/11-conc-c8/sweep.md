# Recur64 runtime sweep (Stage A)

- warmup: 6.42s | games/cell: 16

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 535.2 | 219.8 | 16.8 | 10.7 | 10.61/13/16 | 10298 | 289 |
