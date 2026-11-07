"""Print saved demo results without starting distributed workers."""
import json
from pathlib import Path

root = Path(__file__).parent / 'results'
print('world regime         routing   params/rank  median_ms  p95_ms imbalance')
for path in sorted(root.glob('demo-*.json')):
    rank_zero = json.loads(path.read_text())[0]
    world = rank_zero['world_size']
    for result in rank_zero['benchmarks']:
        print(f"{world:5} {result['regime']:14} {result['routing']:8} "
              f"{result['parameter_bytes_per_rank']:11} "
              f"{result['median_ms']:10.3f} {result['p95_ms']:7.3f} "
              f"{result['max_to_mean_load']:9.2f}")
    losses = rank_zero['trajectory']['loss']
    error = rank_zero['trajectory']['max_abs_error']
    print(f"Training ranks={world}: {losses[0]:.6f} -> {losses[-1]:.6f}; "
          f"max error {error:.3g}")

path = root / 'demo-2.json'
if path.exists():
    print('\nTwo-rank trace (first output coordinate):')
    for rank_result in json.loads(path.read_text()):
        for token in rank_result['tokens']:
            print(token)
