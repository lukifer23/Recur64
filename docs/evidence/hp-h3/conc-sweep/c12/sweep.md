# Recur64 runtime sweep (Stage A)

- warmup: 2.54s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 12 | 12 | 24 | 24 | 1000 | 64 | 474.9 | 179.3 | 7.5 | 7.5 | 15.34/22/24 | 10537 | 298 |
