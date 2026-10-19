# raid6-gfni

| op | block | kernel, best of avx512x1/x2/x4 (ns) | gfni (ns) | speedup |
|---|---|---:|---:|---:|
| gen | 4 KiB | 426 | 191 | 2.2x |
| gen | 16 KiB | 1695 | 744 | 2.3x |
| gen | 64 KiB | 6796 | 2274 | 3.0x |
| xor, start=4 stop=7 | 4 KiB | 376 | 87 | 4.3x |
| xor, start=4 stop=7 | 16 KiB | 1509 | 460 | 3.3x |
| xor, start=4 stop=7 | 64 KiB | 6044 | 1610 | 3.8x |
| recov 2data | 4 KiB | 650 | 176 | 3.7x |
| recov 2data | 64 KiB | 11189 | 3155 | 3.5x |
| recov datap | 4 KiB | 631 | 171 | 3.7x |
| recov datap | 64 KiB | 10193 | 3349 | 3.0x |
