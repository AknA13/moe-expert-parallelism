# Recorded correctness and scaling results

The final suite ran on ARM64 macOS with Python 3.13 and PyTorch 2.9.1. The one-rank baseline used `MoEModel(parallel=False)`; 2- and 4-rank runs used `MoEModel(parallel=True)`. All used learned top-2 routing, identical data and global expert initialization, float64, and Adam at learning rate 0.001.

| Blocks | Total tokens | Epochs | Parallel ranks | Largest recorded reference difference |
|---:|---:|---:|---:|---:|
| 1 | 65 | 30 | 2 | 1.56e-16 |
| 1 | 65 | 30 | 4 | 1.39e-16 |
| 2 | 17 | 2 | 2 | 9.71e-17 |
| 2 | 17 | 2 | 4 | 2.62e-15 |

The first step compares token outputs, input gradients, every owned or replicated parameter gradient, and parameters after Adam. Every epoch compares global loss; final parameters are compared after training. The one-block 65-token loss fell from **0.095015 to 0.014079**. `results/verify/comparison-summary.json` has differences by category. These tests check observed learned routes; they do not force every expert to receive work or an owner to receive zero assignments.

## One-block CPU scaling

Each job ran 25 epochs, with 5 warmup and 20 measured epochs. The slowest rank defines step latency. Payload columns are average logical bytes per measured step across the cluster. The 256-token one-rank job is the baseline for both fixed-total and growing-total comparisons.

| Ranks | Total tokens | Parameter bytes/rank | Median ms | p95 ms | Forward / backward activation bytes | Routing metadata bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 278,848 | 2.684 | 3.397 | 0 / 0 | 0 |
| 2 | 256 | 144,704 | 3.798 | 3.945 | 132,966 / 132,966 | 2,094 |
| 4 | 256 | 77,632 | 7.130 | 7.718 | 192,077 / 192,077 | 3,097 |
| 2 | 512 | 144,704 | 4.973 | 18.832 | 250,573 / 250,573 | 3,931 |
| 4 | 1,024 | 77,632 | 8.380 | 9.430 | 789,581 / 789,581 | 12,433 |

Expert parameter storage falls from 268,288 to 67,072 bytes/rank at four ranks. Shared storage stays 10,560 bytes/rank. The single-rank baseline avoids distributed exchange entirely; multi-rank runs perform it inside each parallel block. These are parameter tensor bytes; Adam's moment tensors, gradients, activations, and process memory are excluded. Logical payload estimates exclude transport overhead. The 2-rank, 512-token job has a high p95 relative to its median; 20 measured steps are descriptive rather than a reliable tail estimate. The small CPU model gets slower with more ranks; no GPU speedup is claimed.

Saved scaling jobs record their training and model configuration, environment, per-epoch loss and latency, and expert assignment counts. The verification suite uses full tensors temporarily, then saves only `results/verify/comparison-summary.json` and `results/verify/demo.json`. `show_results.py` derives logical activation and metadata payload from remote assignment counts. See `results/summary.txt` for the measured table and token trace. Linux and GPU executions were not tested.
