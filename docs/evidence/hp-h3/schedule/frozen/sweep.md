# Recur64 runtime sweep (Stage A)

- warmup: 2.40s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 64 | 457.0 | 172.6 | 7.2 | 7.2 | 11.36/15/16 | 10188 | 298 |
