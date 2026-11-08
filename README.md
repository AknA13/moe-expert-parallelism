# Expert-parallel MoE, from one model to many ranks

A private CPU/Gloo teaching demo with learned top-2 routing, eight experts, and a configurable number of MoE blocks. One `MoEModel` selects local or expert-parallel blocks with its `parallel` constructor argument.

## Reproduce

Use Python 3.13. The recorded environment was ARM64 macOS with PyTorch 2.9.1; pins are in `requirements.txt`.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` launches the correctness and scaling jobs, then writes `results/summary.txt`. You can run one job directly:

```sh
python run.py --ranks 2 --tokens 65 --epochs 30 --lr 0.15 --blocks 1
python run.py --ranks 4 --tokens 1024 --epochs 25 --warmup 5
python show_results.py
```

`--tokens` is always the **total** batch size; `torch.tensor_split` divides it as evenly as possible. The target is deterministic regression to `0.3*tanh(x)+0.2`. Each epoch reuses that batch. Every run uses learned routing and SGD without momentum or weight decay. The default learning rate is 0.15.

## Read the code

1. `moe/model.py`: `MoEModel` builds `StandardMoEBlock` or `ParallelMoEBlock` instances. `ModelConfig.num_blocks` controls the stack depth. The parallel block uses PyTorch's autograd-aware variable-size all-to-all for activations.
2. `moe/training.py`: deterministic model initialization and data shards, shared-gradient synchronization, and one training loop that calls `model(x)` in either mode.
3. `run.py`: one generic worker; `verify.py` and `scale.py` launch it with different inputs.

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

| Parameters | Placement | Action before SGD |
|---|---|---|
| Token projection in every block | Replicated | SUM all-reduce gradients |
| Router in every block | Replicated | SUM all-reduce gradients |
| Experts in every block | Unique owner | Use accumulated local gradient; no expert-gradient all-reduce |

An expert owner receives assignments from every source, so its backward pass already accumulates the complete gradient for that expert. Shared replicas initially see only their source tokens, so they need the SUM. The global denominator handles unequal shards such as `[33, 32]` for 65 tokens on two ranks. Top-k indices are discrete; gradients reach selected router logits through the two selected softmax weights.

## Correctness runs

`verify.py` launches three one-block jobs with **65 total tokens and 30 epochs** at 1, 2, and 4 ranks. The one-rank job builds `MoEModel(parallel=False)`; the others build `MoEModel(parallel=True)` with their rank and world size. It compares every epoch's global loss, first-step token outputs and input gradients, every owned/shared parameter gradient and post-step value, each block's selected IDs and returned values, and final parameters. The first epoch supplies the one-step check. Shared layers must update and loss must decrease. Numerical comparisons use float64, `atol=1e-8`, and `rtol=1e-6`.

It also launches a short **two-block** job at 1, 2, and 4 ranks to check that stacked blocks preserve those equivalences. `--capture-state` saves tensors for this external comparison; it does not execute a reference model inside parallel workers. The normal worker only trains and records its result.

The learned router may leave an expert unused in a particular run. These tests verify the observed routes; they do not guarantee an empty destination or all-expert update case. The code supports an empty owner through graph-preserving empty expert calls and matching collectives, but that edge case is not forced by the submitted suite.

## Scaling runs

`scale.py` launches five one-block jobs. For fixed total work it uses `(ranks, total tokens) = (1,256), (2,256), (4,256)`. For growing total work it adds `(2,512), (4,1024)` and reuses the one-rank 256-token baseline. Each job runs 25 epochs, with 5 warmup epochs and 20 measured epochs. Workers use one CPU thread each; a barrier precedes timing, and an all-reduced maximum makes the slowest rank define step latency. Startup, result serialization, and verification comparisons are excluded.

Logs contain per-rank parameter bytes, per-expert assignment counts, load imbalance, and logical inter-rank payload estimates. For `R` remote assignments, dimension `d`, and element size `b`, each forward or backward activation phase moves `2Rdb` bytes across the cluster. Metadata includes expert IDs and count exchange; this is logical tensor payload, not measured physical network traffic. Parameter storage excludes gradients, optimizer state, activations, and process memory. The ring all-reduce traffic formula, when discussed, is analytical rather than an observation of Gloo's chosen algorithm.

See [RESULTS.md](RESULTS.md) and [results/summary.txt](results/summary.txt) for saved measurements. This small single-host CPU experiment does not establish a GPU speedup.

## Onsite walkthrough (~30 minutes plus discussion)

1. **4 min:** why experts are sharded and what stays replicated.
2. **7 min:** follow token 2 through the two-rank trace and one MoE block.
3. **7 min:** reverse the exchanges in backward and derive the global-loss normalization.
4. **5 min:** show the standard-versus-parallel checks and the two-block run.
5. **7 min:** interpret storage, payload, learned expert loads, and CPU latency.

For production discussion: expert-data-parallel replica groups need separate expert-gradient synchronization; GPU/NCCL execution needs device timing and transport validation; grouped expert GEMMs and overlap may reduce overhead; capacity limits and balancing objectives change model semantics; Adam state and checkpoint resharding add memory and state-management work. None of these extensions is implemented here.

## Study guides

- [Visual code walkthrough](study-guides/01-code-walkthrough.html)
- [Deeper review workbook](study-guides/02-deeper-review.html)

GitHub displays HTML source rather than running these guides. Clone or download the private repository and open them locally.
