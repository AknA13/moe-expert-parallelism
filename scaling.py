#!/usr/bin/env python3
"""Run CPU scaling experiments through run.py and summarize their measurements."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


JOBS = ((1, 256), (2, 256), (4, 256), (2, 512), (4, 1024))


def summarize(ranks):
    """Derive one cluster-level row from per-rank run.py logs."""
    world = len(ranks)
    tokens = ranks[0]['train_config']['total_tokens']
    measured = [step for step in ranks[0]['epochs'] if not step['warmup']]
    durations = sorted(step['step_ms'] for step in measured)
    expert_load = [
        sum(step['blocks'][0]['assignments'][expert]
            for rank in ranks for step in rank['epochs'] if not step['warmup'])
        for expert in range(ranks[0]['model_config']['experts'])
    ]
    remote = sum(step['blocks'][0]['remote_assignments']
                 for rank in ranks for step in rank['epochs']
                 if not step['warmup']) / len(measured)
    row_bytes = ranks[0]['model_config']['dimension'] * 8  # float64
    activation_bytes = 2 * remote * row_bytes  # dispatch + return, per phase
    return {
        'ranks': world,
        'total_tokens': tokens,
        'local_tokens': tokens // world,
        'parameter_bytes_per_rank': ranks[0]['parameter_bytes_per_rank'],
        'shared_parameter_bytes_per_rank': ranks[0]['shared_parameter_bytes_per_rank'],
        'median_step_ms': statistics.median(durations),
        'p95_step_ms': durations[int(.95 * (len(durations) - 1))],
        'tokens_per_second_at_median': tokens * 1000 / statistics.median(durations),
        'expert_load_imbalance': max(expert_load) / (sum(expert_load) / len(expert_load)),
        'remote_assignments_per_step': remote,
        'forward_activation_bytes_per_step': activation_bytes,
        'backward_activation_bytes_per_step': activation_bytes,
        'routing_metadata_bytes_per_step': 8 * remote + 8 * world * (world - 1),
    }


def save_plot(path, rows):
    """Create three small SVG bar charts without adding a plotting dependency."""
    by_job = {(row['ranks'], row['total_tokens']): row for row in rows}
    panels = (
        ('Fixed 256 total tokens', 'median_step_ms',
         [by_job[world, 256] for world in (1, 2, 4)], 'ms', '#315a9b'),
        ('Fixed 256 tokens/rank', 'median_step_ms',
         [by_job[world, 256 * world] for world in (1, 2, 4)], 'ms', '#16866a'),
        ('Model bytes per rank', 'parameter_bytes_per_rank',
         [by_job[world, 256] for world in (1, 2, 4)], 'KiB', '#c8713e'),
    )
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 420">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="35" y="36" font-family="sans-serif" font-size="23">CPU expert-parallel scaling</text>']
    for panel, (title, key, data, unit, color) in enumerate(panels):
        x0 = 35 + panel * 325
        parts += [f'<text x="{x0}" y="78" font-family="sans-serif" font-size="17">{title}</text>',
                  f'<text x="{x0}" y="101" font-family="sans-serif" font-size="12" fill="#555">{unit}</text>',
                  f'<line x1="{x0 + 18}" x2="{x0 + 300}" y1="350" y2="350" stroke="#777"/>']
        values = [row[key] / 1024 if unit == 'KiB' else row[key] for row in data]
        scale = max(values) * 1.18
        for index, (row, value) in enumerate(zip(data, values)):
            x = x0 + 32 + index * 94
            height = 225 * value / scale
            y = 350 - height
            parts += [f'<rect x="{x}" y="{y:.1f}" width="53" height="{height:.1f}" fill="{color}"/>',
                      f'<text x="{x + 26.5:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.2f}</text>',
                      f'<text x="{x + 26.5:.1f}" y="374" text-anchor="middle" font-family="sans-serif" font-size="13">{row["ranks"]} rank</text>']
    parts += ['<text x="35" y="405" font-family="sans-serif" font-size="12" fill="#555">Median of 20 measured steps after 5 warmup steps; one CPU thread per worker.</text>',
              '</svg>']
    path.write_text('\n'.join(parts) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/scaling'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for world, total_tokens in JOBS:
        path = args.output / f'ranks{world}-tokens{total_tokens}.json'
        subprocess.run([
            sys.executable, 'run.py', '--ranks', str(world),
            '--tokens', str(total_tokens), '--epochs', '25',
            '--warmup', '5', '--output', str(path),
        ], check=True)
        row = summarize(json.loads(path.read_text()))
        rows.append(row)
        print(f'MEASURED {world} ranks, {total_tokens} tokens: '
              f'{row["median_step_ms"]:.3f} ms median, '
              f'{row["parameter_bytes_per_rank"]:,} parameter bytes/rank',
              flush=True)
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    save_plot(args.output / 'scaling.svg', rows)
    print('Saved scaling results; see results/scaling/summary.json and scaling.svg')


if __name__ == '__main__':
    main()
