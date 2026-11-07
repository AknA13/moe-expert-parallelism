"""Summarize saved training and scaling logs without starting workers."""
import json
import statistics
from pathlib import Path

root = Path(__file__).parent / 'results'
print('world regime       routing   params/rank median_ms  p95_ms imbalance')
for path in sorted((root / 'scale').glob('*.json')):
    ranks = json.loads(path.read_text())
    if not isinstance(ranks, list) or not ranks or 'steps' not in ranks[0]:
        continue
    world = len(ranks)
    config = ranks[0]['config']
    regime = 'fixed-local' if config['tokens_per_rank'] else 'fixed-global'
    measured = [step for step in ranks[0]['steps'] if not step['warmup']]
    durations = sorted(step['step_ms'] for step in measured)
    expert_load = [
        sum(step['routing']['assignments'][expert]
            for rank in ranks for step in rank['steps'] if not step['warmup'])
        for expert in range(8)
    ]
    imbalance = max(expert_load) / (sum(expert_load) / 8)
    print(f'{world:5} {regime:12} {config["routing"]:9} '
          f'{ranks[0]["parameter_bytes_per_rank"]:11} '
          f'{statistics.median(durations):9.3f} '
          f'{durations[int(.95 * (len(durations)-1))]:7.3f} '
          f'{imbalance:9.2f}')

for world in (1, 2, 4):
    path = root / 'verify' / f'{world}-training.json'
    if path.exists():
        steps = json.loads(path.read_text())[0]['steps']
        print(f'Training ranks={world}: '
              f'{steps[0]["global_loss"]:.6f} -> {steps[-1]["global_loss"]:.6f}')

trace_path = root / 'verify' / '2-balanced.json'
if trace_path.exists():
    print('\nTwo-rank token trace:')
    for rank in json.loads(trace_path.read_text()):
        for token in rank['trace'][:4]:
            print(token)
