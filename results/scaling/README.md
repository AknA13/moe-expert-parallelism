# CPU scaling results

## Question and method

How do rank count, parameter storage, communication, and step time change when the same eight-expert MoE is split across ranks? These one-block CPU/Gloo jobs use float64, learned top-2 routing, Adam at learning rate 0.001, and one CPU thread per worker. Each job ran 25 steps: 5 warmup and 20 measured. A barrier starts each timed step, and its duration is the slowest rank's time. Startup, logging, and correctness comparisons are outside the timed step.

We compare **fixed total work** (256 tokens over 1, 2, or 4 ranks) and **fixed local work** (256 tokens per rank, so 256, 512, or 1,024 total). The one-rank job is shared by both comparisons. Expert count, top-2 routing, and model dimensions stay fixed.

## Measurements

| Ranks | Total tokens | Tokens/rank | Model KiB/rank | Median ms | p95 ms | Tokens/s at median | Expert imbalance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 256 | 272.3 | 2.485 | 2.526 | 103,022 | 1.36 |
| 2 | 256 | 128 | 141.3 | 3.694 | 3.812 | 69,306 | 1.36 |
| 4 | 256 | 64 | 75.8 | 6.907 | 7.209 | 37,065 | 1.36 |
| 2 | 512 | 256 | 141.3 | 4.320 | 4.532 | 118,521 | 1.45 |
| 4 | 1,024 | 256 | 75.8 | 7.956 | 8.345 | 128,710 | 1.49 |

For fixed total work, model storage falls from **272.3 to 75.8 KiB per rank**, while median step time rises from **2.485 to 6.907 ms**. Expert weights are sharded; the shared projection and router still occupy 10.3 KiB on every rank. This small CPU workload pays more for communication and coordination than it gains from dividing expert computation.

For fixed local work, total throughput rises from **103,022 to 128,710 tokens/s** as total batch size grows. The median step also becomes longer: **2.485 to 7.956 ms**. This describes larger total work, not a fixed-batch speedup.

![CPU scaling measurements](scaling.svg)

## Communication and interpretation

The 4-rank, 256-token job averages 375.1 remote token-expert assignments per measured step. The corresponding logical payload is 192,077 bytes in the forward activation phase and the same amount in backward, plus 3,097 bytes of routing metadata. Across all jobs, the expert imbalance ratio ranges from 1.36 to 1.49; this is the most-loaded expert's average assignment count divided by the mean.

The payload estimates count tensor elements: for `R` remote assignments, dimension `d`, element size `b`, and `W` ranks, either activation phase moves `2Rdb` bytes (dispatch and return), while expert IDs and count exchange add `8R + 8W(W-1)` bytes. They are not physical network measurements. Parameter bytes exclude gradients, Adam state, activations, and process overhead. Twenty measured steps describe this local run; they do not establish tail latency or GPU performance.

## Files

- [summary.json](summary.json): one derived metrics row for each of the five jobs.
- [scaling.svg](scaling.svg): fixed-total timing, fixed-local timing, and model storage plots.
- `ranks{R}-tokens{N}.json`: per-rank configuration, loss, timing, routing counts, and parameter storage.
- `ranks{R}-tokens{N}.environment.json`: Python, PyTorch, host, dtype, and backend details for each job.
