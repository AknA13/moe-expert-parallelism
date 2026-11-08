"""Summarize saved correctness and scaling runs without starting workers."""
import json
import statistics
from pathlib import Path

root = Path(__file__).parent / 'results'
print('ranks total_tokens params/rank median_ms p95_ms imbalance forward_B backward_B metadata_B')
for world, tokens in ((1, 256), (2, 256), (4, 256), (2, 512), (4, 1024)):
    path = root / 'scale' / f'ranks{world}-tokens{tokens}.json'
    if not path.exists():
        continue
    ranks = json.loads(path.read_text())
    measured = [row for row in ranks[0]['epochs'] if not row['warmup']]
    durations = sorted(row['step_ms'] for row in measured)
    expert_load = [
        sum(row['blocks'][0]['assignments'][expert]
            for rank in ranks for row in rank['epochs'] if not row['warmup'])
        for expert in range(ranks[0]['model_config']['experts'])
    ]
    imbalance = max(expert_load) / (sum(expert_load) / len(expert_load))
    remote = sum(row['blocks'][0]['remote_assignments']
                 for rank in ranks for row in rank['epochs']
                 if not row['warmup']) / len(measured)
    row_bytes = ranks[0]['model_config']['dimension'] * 8  # float64
    activation_bytes = 2 * remote * row_bytes
    metadata_bytes = 8 * remote + 8 * world * (world - 1)
    print(f'{world:5} {tokens:12} {ranks[0]["parameter_bytes_per_rank"]:11} '
          f'{statistics.median(durations):9.3f} '
          f'{durations[int(.95*(len(durations)-1))]:7.3f} '
          f'{imbalance:9.2f} {activation_bytes:9.0f} '
          f'{activation_bytes:10.0f} {metadata_bytes:10.0f}')

path = root / 'verify' / 'demo.json'
if path.exists():
    demo = json.loads(path.read_text())
    print(f'Training loss: {demo["initial_loss"]:.6f} -> '
          f'{demo["final_loss"]:.6f}')
    print('\nTwo-rank mixed-owner token trace:')
    for token in demo['mixed_owner_tokens']:
        print(token)
