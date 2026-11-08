# Correctness results

## Question and method

Does expert-parallel training compute the same outputs, gradients, and Adam updates as one process owning all eight experts? We ran the same deterministic global batch and global expert initialization at 1, 2, and 4 ranks. The one-rank reference uses `StandardMoEBlock`; multi-rank jobs use `ParallelMoEBlock`. Every run used float64, learned top-2 routing, and Adam with learning rate 0.001. Comparisons use `atol=1e-08` and `rtol=1e-06`.

The primary fixture has one MoE block and 65 total tokens, split unevenly across ranks. One-epoch jobs test one complete optimizer step; 30-epoch jobs test the loss trajectory and final parameters. A second fixture has two blocks, 17 tokens, and two epochs.

## One-step equivalence

Maximum absolute difference from the one-rank reference, over every token or owned parameter:

| Parallel ranks | Output | Input gradient | Parameter gradient | Parameter after Adam |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 8.67e-19 | 5.55e-17 |
| 4 | 0 | 0 | 1.08e-18 | 3.47e-17 |

The checks also compare selected expert IDs, returned expert outputs, and global loss. The 30-epoch runs reached final-parameter differences of 1.56e-16 (2 ranks) and 1.39e-16 (4 ranks). The two-block fixture's largest recorded differences were 9.71e-17 and 2.62e-15.

## Did the parameters actually update?

The 4-rank, one-step run routed assignments to every expert: counts by expert ID were 12, 5, 19, 9, 5, 35, 26, 19. Each row below reports the **smallest**, across its parameter tensors, of that tensor's largest absolute gradient and largest absolute change. A positive value therefore means every tensor in that row's group received a gradient and changed. Projection and router values are checked against the reference on every replica; each expert is checked on its owning rank.

| Parameter group | Owner rank | Assignments | Gradient | One-step change |
|---|---:|---:|---:|---:|
| token_projection | replicated | — | 0.00248 | 0.001 |
| router | replicated | — | 0.000871 | 0.001 |
| experts.0 | 0 | 12 | 0.000735 | 0.001 |
| experts.1 | 0 | 5 | 0.000439 | 0.001 |
| experts.2 | 1 | 19 | 0.000852 | 0.001 |
| experts.3 | 1 | 9 | 0.000346 | 0.001 |
| experts.4 | 2 | 5 | 0.000382 | 0.001 |
| experts.5 | 2 | 35 | 0.00161 | 0.001 |
| experts.6 | 3 | 26 | 0.000791 | 0.001 |
| experts.7 | 3 | 19 | 0.000776 | 0.001 |

The router and token projection are replicated, so their gradients are SUM all-reduced before Adam. Each expert has one owner; its backward pass accumulates contributions from every source rank through the activation exchanges, so expert gradients are not all-reduced within this group.

## Training trajectory

The one-block global loss fell from **0.095015** to **0.014079** in 30 epochs. The 1-, 2-, and 4-rank curves overlap:

![Global training loss](loss.svg)

## Files and limits

- [comparison-summary.json](comparison-summary.json): maximum numerical errors for all six parallel comparisons.
- [parameter-updates.json](parameter-updates.json): first-step gradients, changes, assignment counts, and owners.
- [demo.json](demo.json): a two-rank token trace and the first/final loss.
- [loss.svg](loss.svg): global loss by epoch.

Full tensors are used during comparison and discarded; these compact files retain the evidence. The two execution paths share their router and expert definitions, so agreement cannot detect a bug common to both. Learned routing reaches every expert in this fixture but does not force an owner to receive zero assignments.
