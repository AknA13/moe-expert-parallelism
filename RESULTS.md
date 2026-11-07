# Recorded validation and measurements

The single-job worker passed reference checks at 1, 2, and 4 ranks on ARM64 macOS with Python 3.13 and PyTorch 2.9.1. `verify.py` ran separate learned, balanced, skewed, local, and 30-epoch training jobs at each rank count. Linux was not tested.

| Ranks | Max reference error | Training loss, first → last |
|---:|---:|---:|
| 1 | 0 | 0.095015 → 0.080596 |
| 2 | 2.22e-16 | 0.095015 → 0.080596 |
| 4 | 1.67e-16 | 0.095015 → 0.080596 |

## Fixed global batch (256 tokens), learned routing

| Ranks | Parameter bytes/rank | Median ms | p95 ms | Forward / backward activation bytes (cluster) |
|---:|---:|---:|---:|---:|
| 1 | 278,848 | 2.713 | 3.280 | 0 / 0 |
| 2 | 144,704 | 3.551 | 3.760 | 133,018 / 133,018 |
| 4 | 77,632 | 6.880 | 7.414 | 193,894 / 193,894 |

Expert parameter storage falls from 268,288 to 67,072 bytes/rank while shared storage stays 10,560 bytes/rank. These are parameter tensor bytes, not peak process memory. CPU latency increases with rank count for this small model.

Each `results/verify/*.json` or `results/scale/*.json` file contains one training job: its config, every rank's per-epoch loss, timing, gradient norms, routing load, communication estimates, and optional reference errors. `verify.py` evaluates correctness. `results/summary.txt` covers both scaling regimes. These CPU measurements do not establish a GPU speedup.
