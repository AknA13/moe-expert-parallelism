# Correctness Results

Does our multi-rank expert-parallel training compute the same:
- outputs
- gradients
- Adam updates
  
as single-rank naive MoE training?

We ran deterministic experiments with the same training dataset and weight initializations, comparing one-rank training using `StandardMoEBlock` (no communication code; meant to run on a single device) with 2- and 4-rank training using `ParallelMoEBlock`, which uses expert parallelism. We also keep the learning hyperparams the same in this comparison. 

## One-step equivalence

Maximum absolute difference from the single-rank trained model (non-parallel), over every token or owned parameter:


| # of ranks | Output | Token/Input gradient | Parameter gradient | Parameter after Adam |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 8.67e-19 | 5.55e-17 |
| 4 | 0 | 0 | 1.08e-18 | 3.47e-17 |


## Did the parameters actually update?

The 4-rank, one-step run routed assignments to every expert: counts by expert ID were 12, 5, 19, 9, 5, 35, 26, 19. Each row below reports the **smallest**, across its parameter tensors, of that tensor's largest absolute gradient and largest absolute change. A positive value therefore means every tensor in that row's group received a gradient and changed. 
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

The router and token projection are replicated across ranks, so their gradients are SUM all-reduced before Adam. Each expert has a single owning rank.

## Training trajectory

The global loss fell from **0.095015** to **0.014079** in 30 epochs. The 1-, 2-, and 4-rank curves overlap:

![Global training loss](loss.svg)

