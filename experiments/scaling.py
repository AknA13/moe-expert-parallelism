#!/usr/bin/env python3
"""Run CPU scaling experiments through run.py and summarize their measurements."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


JOBS = ((1, 256), (2, 256), (4, 256), (2, 512), (4, 1024))
ROOT = Path(__file__).resolve().parents[1]


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


def save_readme(path, rows):
    """Write the interpretation beside the plot and raw logs."""
    fixed_total = [row for row in rows if row['total_tokens'] == 256]
    fixed_local = [row for row in rows if row['local_tokens'] == 256]
    table = [
        f'| {row["ranks"]} | {row["total_tokens"]:,} | {row["local_tokens"]} | '
        f'{row["parameter_bytes_per_rank"] / 1024:.1f} | '
        f'{row["median_step_ms"]:.3f} | {row["p95_step_ms"]:.3f} | '
        f'{row["tokens_per_second_at_median"]:,.0f} | '
        f'{row["expert_load_imbalance"]:.2f} |'
        for row in rows
    ]
    path.write_text(f'''# CPU scaling results

## Question and method

How do rank count, parameter storage, communication, and step time change when the same eight-expert MoE is split across ranks? These one-block CPU/Gloo jobs use float64, learned top-2 routing, Adam at learning rate 0.001, and one CPU thread per worker. Each job ran 25 steps: 5 warmup and 20 measured. A barrier starts each timed step, and its duration is the slowest rank's time. Startup, logging, and correctness comparisons are outside the timed step.

We compare **fixed total work** (256 tokens over 1, 2, or 4 ranks) and **fixed local work** (256 tokens per rank, so 256, 512, or 1,024 total). The one-rank job is shared by both comparisons. Expert count, top-2 routing, and model dimensions stay fixed.

## Measurements

| Ranks | Total tokens | Tokens/rank | Model KiB/rank | Median ms | p95 ms | Tokens/s at median | Expert imbalance |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

For fixed total work, model storage falls from **{fixed_total[0]['parameter_bytes_per_rank'] / 1024:.1f} to {fixed_total[-1]['parameter_bytes_per_rank'] / 1024:.1f} KiB per rank**, while median step time rises from **{fixed_total[0]['median_step_ms']:.3f} to {fixed_total[-1]['median_step_ms']:.3f} ms**. Expert weights are sharded; the shared projection and router still occupy {fixed_total[0]['shared_parameter_bytes_per_rank'] / 1024:.1f} KiB on every rank. This small CPU workload pays more for communication and coordination than it gains from dividing expert computation.

For fixed local work, total throughput rises from **{fixed_local[0]['tokens_per_second_at_median']:,.0f} to {fixed_local[-1]['tokens_per_second_at_median']:,.0f} tokens/s** as total batch size grows. The median step also becomes longer: **{fixed_local[0]['median_step_ms']:.3f} to {fixed_local[-1]['median_step_ms']:.3f} ms**. This describes larger total work, not a fixed-batch speedup.

![CPU scaling measurements](scaling.svg)

## Communication and interpretation

The 4-rank, 256-token job averages {fixed_total[-1]['remote_assignments_per_step']:.1f} remote token-expert assignments per measured step. The corresponding logical payload is {fixed_total[-1]['forward_activation_bytes_per_step']:,.0f} bytes in the forward activation phase and the same amount in backward, plus {fixed_total[-1]['routing_metadata_bytes_per_step']:,.0f} bytes of routing metadata. Across all jobs, the expert imbalance ratio ranges from {min(row['expert_load_imbalance'] for row in rows):.2f} to {max(row['expert_load_imbalance'] for row in rows):.2f}; this is the most-loaded expert's average assignment count divided by the mean.

The payload estimates count tensor elements: for `R` remote assignments, dimension `d`, element size `b`, and `W` ranks, either activation phase moves `2Rdb` bytes (dispatch and return), while expert IDs and count exchange add `8R + 8W(W-1)` bytes. They are not physical network measurements. Parameter bytes exclude gradients, Adam state, activations, and process overhead. Twenty measured steps describe this local run; they do not establish tail latency or GPU performance.

## Files

- [summary.json](summary.json): one derived metrics row for each of the five jobs.
- [scaling.svg](scaling.svg): fixed-total timing, fixed-local timing, and model storage plots.
- `ranks{{R}}-tokens{{N}}.json`: per-rank configuration, loss, timing, routing counts, and parameter storage.
- `ranks{{R}}-tokens{{N}}.environment.json`: Python, PyTorch, host, dtype, and backend details for each job.
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/scaling')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for world, total_tokens in JOBS:
        path = args.output / f'ranks{world}-tokens{total_tokens}.json'
        subprocess.run([
            sys.executable, str(ROOT / 'run.py'), '--ranks', str(world),
            '--tokens', str(total_tokens), '--epochs', '25',
            '--warmup', '5', '--output', str(path),
        ], check=True, cwd=ROOT)
        row = summarize(json.loads(path.read_text()))
        rows.append(row)
        print(f'MEASURED {world} ranks, {total_tokens} tokens: '
              f'{row["median_step_ms"]:.3f} ms median, '
              f'{row["parameter_bytes_per_rank"]:,} parameter bytes/rank',
              flush=True)
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    save_plot(args.output / 'scaling.svg', rows)
    save_readme(args.output / 'README.md', rows)
    print(f'Saved scaling results; see {args.output / "README.md"}')


if __name__ == '__main__':
    main()
