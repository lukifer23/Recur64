# Recur64 runtime sweep (Stage A)

- warmup: 2.79s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 327.7 | 209.0 | 10.3 | 8.8 | 11.73/14/16 | 10250 | 289 |
