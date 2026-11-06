"""Print saved evidence; does not launch workers."""
import json
from pathlib import Path
root = Path(__file__).parent/'results'
print('world regime         routing   params/rank  median_ms  p95_ms imbalance')
for path in sorted(root.glob('scale-*.json')):
    r = json.loads(path.read_text())[0]
    for b in r['benchmarks']:
        print(f"{r['world_size']:5} {b['regime']:14} {b['routing']:8} {b['parameter_bytes_per_rank']:11} {b['median_ms']:10.3f} {b['p95_ms']:7.3f} {b['max_to_mean_load']:9.2f}")
for path in sorted(root.glob('train-*.json')):
    r = json.loads(path.read_text())[0]; t = r['trajectory']
    print(f"Training ranks={r['world_size']}: {t['loss'][0]:.6f} -> {t['loss'][-1]:.6f}; max error {t['max_abs_error']:.3g}")
path = root/'trace-2.json'
if path.exists():
    print('\nTwo-rank trace (first output coordinate):')
    for r in json.loads(path.read_text()):
        for t in r['tokens']: print(t)
