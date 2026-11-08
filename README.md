# Expert-parallel MoE, from one model to many ranks

A private CPU/Gloo teaching demo with learned top-2 routing, eight experts, and a configurable number of MoE blocks. One `MoEModel` selects local or expert-parallel blocks with its `parallel` constructor argument.

## Reproduce

Use Python 3.13. The recorded environment was ARM64 macOS with PyTorch 2.9.1; pins are in `requirements.txt`.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python experiments/correctness.py && python experiments/scaling.py
```

`experiments/correctness.py` prints named pass/fail checks and saves a loss plot and [short results report](results/correctness/README.md). `experiments/scaling.py` runs the timing jobs and saves a scaling plot and [short results report](results/scaling/README.md). Both launch experiments only through `run.py`. You can also run one job directly:

```sh
python run.py --ranks 2 --tokens 65 --epochs 30 --lr 0.001 --blocks 1
python run.py --ranks 4 --tokens 1024 --epochs 25 --warmup 5
```

`--tokens` is always the **total** batch size; `torch.tensor_split` divides it as evenly as possible. The target is deterministic regression to `0.3*tanh(x)+0.2`. Each epoch reuses that batch. Every run uses learned routing and Adam with its default betas and no weight decay. The default learning rate is 0.001.

## Read the code

1. `moe/model.py`: `MoEModel` builds `StandardMoEBlock` or `ParallelMoEBlock` instances. `ModelConfig.num_blocks` controls the stack depth. The parallel block uses PyTorch's autograd-aware variable-size all-to-all for activations.
2. `moe/training.py`: the training step, step reporting, shared-gradient synchronization, and one training loop that calls `model(x)` in either mode.
3. `run.py`: one generic worker that initializes the model and data shard; the two files in `experiments/` launch it with different inputs.

Each block contains its own token projection (`Linear` + `Tanh`), router, and experts. In parallel mode, projection and router parameters are replicated, while each rank owns `8/world_size` experts per block. In standard mode, one process owns all experts. Both modes use the same output dimension, gate calculation, and global expert initialization. No test-only routing mode enters the model.

## Two-rank token flow

Rank 0 owns experts 0–3 in each block; rank 1 owns 4–7. For token 2 in the saved learned-routing trace, the router selects experts **7 and 0**: one remote and one local.

```text
rank 0 token 2 → block projection → block router → [expert 7, expert 0]
                                              ├─ to rank 1 → expert 7 ─┐
                                              └─ local expert 0 ────────┤
rank 0 receives both outputs ← inverse packing ←──────────────────────────┘
rank 0 combines them with the selected router weights → token output/loss
```

Each `ParallelMoEBlock.forward` sorts assignments by owner, exchanges counts and expert IDs, sends activation rows, runs its owned experts, returns their outputs, restores token/slot order, and combines them. Each block can do this independently, so a model can contain multiple blocks. PyTorch's differentiable all-to-all reverses split sizes in backward to return activation gradients to the original rows. Empty owners still enter the same collective sequence and retain zero-valued graph paths.

The local `StandardMoEBlock` executes the same selected experts and gate combination without distributed calls. It is the independent one-rank baseline for numerical checks.

## Global loss and gradient ownership

For `N` total tokens, source rank `r` contributes

```text
L_r = (1/N) · Σ_{tokens i on rank r} mean_j (prediction_ij - target_ij)²
L_global = Σ_r L_r
```

| Parameters | Placement | Action before Adam |
|---|---|---|
| Token projection in every block | Replicated | SUM all-reduce gradients |
| Router in every block | Replicated | SUM all-reduce gradients |
| Experts in every block | Unique owner | Use accumulated local gradient; no expert-gradient all-reduce |

An expert owner receives assignments from every source, so its backward pass already accumulates the complete gradient for that expert. Shared replicas initially see only their source tokens, so they need the SUM. The global denominator handles unequal shards such as `[33, 32]` for 65 tokens on two ranks. Top-k indices are discrete; gradients reach selected router logits through the two selected softmax weights. Every expert runs even when it receives zero rows, producing zero gradients for unused experts; Adam can still move them through momentum from earlier steps.

## Correctness runs

`experiments/correctness.py` launches one-block jobs with **65 total tokens** at 1, 2, and 4 ranks. One-epoch jobs check one-step outputs, input gradients, every owned/shared parameter gradient and post-step value. They also check that the projection, router, and all eight routed experts get nonzero gradients and change after the step. Thirty-epoch jobs compare global loss at every epoch and final parameters. The one-rank job builds `MoEModel(parallel=False)`; the others build `MoEModel(parallel=True)`. Numerical comparisons use float64, `atol=1e-8`, and `rtol=1e-6`.

It also launches a short **two-block** job at 1, 2, and 4 ranks to check that stacked blocks preserve those equivalences. `--capture-state` captures full tensors for external comparison in temporary files; it does not execute a reference model inside parallel workers. Compact checks, parameter-update evidence, a two-token trace, and [the loss plot](results/correctness/loss.svg) are saved in `results/correctness`. Each worker initializes its model and data shard, trains, and records its result.

The 65-token one-step fixture reaches and updates all eight experts. These tests do not force an empty destination. The code supports an empty owner through graph-preserving empty expert calls and matching collectives, but that edge case is not forced by the submitted suite.

## Scaling runs

`experiments/scaling.py` launches nine one-block jobs: every combination of 1, 2, or 4 ranks and 256, 512, or 1,024 total tokens. This supports separate trend plots for increasing ranks at a fixed batch size and increasing data at a fixed rank count. It also retains the 256-token fixed-total comparison and the 256-tokens-per-rank diagonal. Each job runs 25 epochs, with 5 warmup epochs and 20 measured epochs. Workers use one CPU thread each; a barrier precedes timing, and an all-reduced maximum makes the slowest rank define step latency. Startup, result serialization, and correctness comparisons are excluded.

Scaling logs contain per-rank parameter bytes and per-epoch loss, latency, expert assignments, and remote assignment counts. `experiments/scaling.py` derives load imbalance, token throughput, and logical payload from those counts, then saves [rank trends](results/scaling/rank-trends.svg), [data trends](results/scaling/data-trends.svg), and the [original bar chart](results/scaling/scaling.svg). For `R` remote assignments, dimension `d`, element size `b`, and `W` ranks, each forward or backward activation phase moves `2Rdb` bytes across the cluster; routing IDs and count exchange add `8R + 8W(W-1)` metadata bytes. These are logical tensor payloads, not measured physical network traffic. Parameter storage excludes gradients, optimizer state, activations, and process memory. The ring all-reduce traffic formula, when discussed, is analytical rather than an observation of Gloo's chosen algorithm.

See the [correctness report](results/correctness/README.md) and [scaling report](results/scaling/README.md) for the measured results and their limits. This small single-host CPU experiment does not establish a GPU speedup.
