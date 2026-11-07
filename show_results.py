"""Summarize saved correctness and scaling runs without starting workers."""
import json
import statistics
from pathlib import Path

root = Path(__file__).parent / 'results'
print('ranks total_tokens params/rank median_ms p95_ms imbalance forward_B backward_B')
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
    payload = {}
    for name in ('activation_forward_bytes', 'activation_backward_bytes'):
        payload[name] = sum(row['blocks'][0][name]
                            for rank in ranks for row in rank['epochs']
                            if not row['warmup']) / len(measured)
    print(f'{world:5} {tokens:12} {ranks[0]["parameter_bytes_per_rank"]:11} '
          f'{statistics.median(durations):9.3f} '
          f'{durations[int(.95*(len(durations)-1))]:7.3f} '
          f'{imbalance:9.2f} {payload["activation_forward_bytes"]:9.0f} '
          f'{payload["activation_backward_bytes"]:10.0f}')

path = root / 'verify' / 'blocks1-ranks1.json'
if path.exists():
    steps = json.loads(path.read_text())[0]['epochs']
    print(f'Training loss: {steps[0]["global_loss"]:.6f} -> '
          f'{steps[-1]["global_loss"]:.6f}')

path = root / 'verify' / 'blocks1-ranks2.json'
if path.exists():
    print('\nTwo-rank mixed-owner token trace:')
    for rank in json.loads(path.read_text()):
        tokens = rank['trace'][0]['tokens']
        mixed = next((token for token in tokens
                      if len(set(token['owners'])) == 2), tokens[0])
        print(mixed)
