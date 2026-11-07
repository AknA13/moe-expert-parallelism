# Expert parallelism, one token journey at a time

A private, CPU-only teaching demo: eight experts distributed over 1, 2, or 4 Gloo workers. It includes top-2 routing, a differentiable variable-size token exchange, a full-model reference, training, and resource measurements. No transformer framework is required.

## Reproduce

Use Python 3.13; the recorded validation environment is Python 3.13 on ARM64 macOS. Dependency pins are in `requirements.txt`. Linux x86/ARM64 use the same CPU/Gloo code but have not been tested here.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py all
```

Run from this directory. This overwrites JSON results in `results/`; use `--output fresh-results` to preserve the submitted evidence. Each command launches and joins fresh workers; failures propagate to the parent. Collectives have a 90-second timeout. No server, GPU, account, or network service is needed after installation.

```sh
python run.py verify                 # reference and communication checks
python run.py train                  # 30 matching SGD steps
python run.py scale                  # two scaling regimes, three routing patterns
python run.py trace --ranks 2        # eight tokens with source/owner/output values
python show_results.py               # terminal tables and token trace
```

`all` runs all four commands at 1, 2, and 4 ranks. Seeds, dimensions, dtype, optimizer settings, platform, versions, and measurement settings are recorded in the source and JSON. The actual task is deterministic regression to `0.3*tanh(x)+0.2`, not language modeling.

## Read the code in this order

1. `moe/model.py`: a small model, its top-2 gating, and a reference with no distributed operations. `moe/experiments.py` handles reproducible initialization and diagnostic fixtures separately.
2. `moe/parallel.py`: packing, token exchange, expert execution, inverse exchange, and shared-gradient synchronization.
3. `moe/experiments.py`: assertions against the reference, finite differences, and benchmark boundaries. `run.py` supplies process lifecycle and JSON output.

## The forward journey

Every rank has the same shared token encoder and router, different tokens, and a unique contiguous subset of experts. Each token selects two experts. The gate is the softmax over the two selected logits, equivalent to renormalizing the selected full-softmax probabilities. The selected indices are not differentiated; gate values are.

For two ranks, rank 0 owns experts 0–3 and rank 1 owns experts 4–7:

```text
rank 0 tokens -> token encoder/router -> pack assignments by owner
                                      | local experts 0–3
                                      +------ activations ------> rank 1 experts 4–7
                                      <------ expert outputs ----+
             <- restore order <- weighted sum of two expert outputs

rank 1 simultaneously sends its own token assignments to rank 0.
```

Tokens remain logically owned by their source rank. Experts temporarily receive activations, not the source rank's loss or targets. We exchange assignment counts, expert IDs, and then activation rows. The return exchange reverses the split sizes. Restoring the sorting permutation reconnects each returned row to its token and routing weight.

The trace uses deliberately balanced expert IDs so the journey is easy to inspect. It records both returned expert values and the combined value. Training uses learned top-2 selection. Balanced, skewed, and local routes are explicit diagnostic fixtures: their selected experts are prescribed, but their gate weights still come from the learned router.

## Backward and the normalization contract

With global token count N, rank r contributes:

```text
L_r = (1/N) * sum(tokens on rank r) mean(output dimensions) squared_error
L_global = sum_r L_r
```

Unequal token-shard sizes therefore need no extra weighting. Each source computes only its own contribution. Backward reverses each activation exchange: output gradients travel to expert owners, then input-activation gradients travel back to sources. For an exchange viewed as a permutation P, backward applies its transpose, Pᵀ.

| Parameters | Placement | Gradient action before SGD |
|---|---|---|
| Shared token encoder | Replicated | SUM all-reduce across all ranks |
| Router | Replicated | SUM all-reduce across all ranks |
| Experts | Unique owner | No parameter-gradient all-reduce within this group |

Each expert sees assignments from all sources, so its local backward already accumulates all contributions to that expert. Summing different experts' gradients would be incorrect. Using local mean loss and blindly averaging all shared gradients would also be incorrect for unequal shards. The global normalization makes the contract explicit. DDP's usual averaging semantics would require adapting this normalization; this demo uses manual sums.

The router receives gradients from weighted combination on the source. The token encoder receives both the router path and expert-input paths. SGD uses learning rate 0.15 without momentum or weight decay. Unused gradients become zeros before synchronization and stepping.

**Empty destinations matter:** every rank must execute matching collectives in matching order, even if no assignments arrive. `computed = received * 0` retains the dispatch's graph dependency. Empty expert calls also retain zero gradients. There are no data-dependent skipped collective calls.

## What the checks prove

`verify` compares full outputs, global loss, input gradients, all owned/shared parameter gradients, and updated parameters to an independently executed full-model reference. Tolerances are `atol=1e-8`, `rtol=1e-6`, using float64. Every replica is explicitly compared with rank 0. The balanced fixture requires every parameter tensor to change.

Cases cover learned routing, balanced expert coverage, skew to experts 0 and 1 (unused experts and zero-receive ranks), entirely local routes, and unequal input shards (65 tokens). The communication check uses asymmetric split sizes and checks both the inverse mapping and finite differences of a global objective. The 30-step training trace checks equivalence on every step and requires final loss below initial loss. Recorded maximum error covers outputs, parameter gradients, and updated parameters; other assertions are pass/fail.

See [RESULTS.md](RESULTS.md) for recorded tables, [results/summary.txt](results/summary.txt) for all timing rows, and [ONSITE.md](ONSITE.md) for rehearsal questions.

## Scaling: what is measured and what is modeled

The model stays fixed at eight experts. Strong-scaling experiments use 256 global tokens; weak-scaling experiments use 256 tokens per rank. Each runs learned, balanced, and skewed routing. Each worker uses one CPU thread. Five warmup steps precede 20 measured steps. A barrier precedes timing; the slowest worker's duration defines the step. Startup, validation, and result aggregation are excluded. Forward metadata handling and optimizer work are included. p95 is the nearest lower order statistic at index `floor(.95*(samples-1))`.

Per-rank parameter storage is `shared_bytes + total_expert_bytes/world_size`. This is tensor storage, not RSS or peak memory: gradients, activations, communication buffers, interpreter overhead, and the reference are separate. The benchmark does not construct a reference. Verification does construct one on every worker for simplicity and is unsuitable for memory measurements.

For R remote assignments, dimension d, and element size b:

- Forward activation payload = `2*R*d*b` for dispatch plus return.
- Backward activation payload = `2*R*d*b` for the two inverse exchanges.
- Routing metadata = `8*R + 8*world_size*(world_size-1)` cluster bytes (expert IDs plus pairwise counts).
- Shared-gradient logical input = shared parameter bytes per rank.
- A ring all-reduce would send `2*(world_size-1)*shared_bytes` cluster bytes; this is a model, **not a claim that Gloo chose a ring**.

Self-directed assignments are excluded from inter-rank payload totals. Protocol framing, transport copies, synchronization, and control messages are excluded. Measured payload totals are averaged over the measured steps; learned routes can change. Load imbalance is maximum expert assignment count divided by the mean. No padding is used on the validated Gloo build.

CPU processes share one machine, so latency is not evidence of GPU speedup. Small matrix operations and communication overhead can outweigh saved compute. The useful results are correctness, expert-storage reduction, payload accounting, and sensitivity to hot experts. Increasing model size or hardware bandwidth changes the tradeoff; do not extrapolate these milliseconds into GPU claims.

## Onsite walkthrough (~30 minutes + ~30 minutes discussion)

1. **4 min:** explain sparse compute, model capacity, expert ownership, and why replication costs memory.
2. **7 min:** show the two-rank trace; follow one cross-rank token through sorting, dispatch, expert, return, and combination.
3. **7 min:** derive global loss normalization; walk through `Exchange.backward` and the synchronization table.
4. **5 min:** run `verify --ranks 2`, show the recorded 30-step trajectory, and explain what each assertion catches.
5. **7 min:** compare rank counts and balanced/skewed loads; distinguish measured results from resource models.

Keep the submitted JSON and generated tables available if live execution fails. Rehearse explaining why an empty rank must still participate and why a router can learn despite discrete top-k indices.

## Extensions to discuss, not implemented

- **Expert data parallelism:** replicate the same expert across separate expert-parallel groups; synchronize that expert's gradients only across its replica group. Shared parameters need a synchronization group and normalization matching the complete logical batch.
- **GPU/NCCL:** move parameters, payloads, and metadata to GPUs, validate collective compatibility, and synchronize GPU timing. Grouped expert GEMMs can reduce small-kernel overhead.
- **Overlap:** asynchronous communication and chunking may hide transfers behind expert work, but require correct buffer lifetimes and collective ordering.
- **Balancing/capacity:** auxiliary objectives and router regularization address hot experts. Capacity limits introduce dropping or rerouting and change model semantics; reference tests must match those semantics.
- **Optimizer memory:** Adam adds moments; quantify parameters, gradients, optimizer states, and transient activations separately. Expert sharding partitions expert states, while shared states remain replicated here.
- **Checkpointing:** use stable global expert IDs and save router/token encoder, optimizer state, RNG, and training position. Reshard expert state when changing topology.

No capacity dropping, mixed precision, distributed checkpointing, transformer, or production framework integration is implemented. The implementation and assignment remain private.

## Study guides

- [Visual code walkthrough](study-guides/01-code-walkthrough.html): token flow, gradients, normalization calculator, and rehearsal.
- [Deeper review workbook](study-guides/02-deeper-review.html): evidence limits, concrete review findings, and proposed tests.

GitHub displays HTML source rather than executing these interactive guides. Clone or download the private repository and open the HTML files locally to use them.
