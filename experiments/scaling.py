#!/usr/bin/env python3
"""Run CPU scaling experiments through run.py and summarize their measurements."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


RANKS = (1, 2, 4)
EXTENDED_RANKS = (1, 2, 4, 8)
SMALL_TOKENS = (256, 512, 1024)
LARGER_TOKENS = (1024, 2048, 4096, 8192)
EXTENDED_TOKENS = (4096, 8192, 16384)
BLOCK_COMPARISON_TOKENS = (16384,)
SMALL_EXPERT_HIDDEN = 64
LARGER_EXPERT_HIDDEN = 512
EXTENDED_EXPERT_HIDDEN = 1024
ROOT = Path(__file__).resolve().parents[1]


def summarize(ranks):
    """Derive one cluster-level row from per-rank run.py logs."""
    world = len(ranks)
    tokens = ranks[0]['train_config']['total_tokens']
    measured = [step for step in ranks[0]['epochs'] if not step['warmup']]
    durations = sorted(step['step_ms'] for step in measured)
    blocks = ranks[0]['model_config']['num_blocks']
    expert_load = [
        sum(block['assignments'][expert]
            for rank in ranks for step in rank['epochs'] if not step['warmup']
            for block in step['blocks'])
        for expert in range(ranks[0]['model_config']['experts'])
    ]
    remote = sum(block['remote_assignments']
                 for rank in ranks for step in rank['epochs']
                 if not step['warmup'] for block in step['blocks']) / len(measured)
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
        'routing_metadata_bytes_per_step': 8 * remote + 8 * world * (world - 1) * blocks,
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


def save_trend_plot(path, title, x_label, x_values, series):
    """Plot step latency while varying one experiment axis at a time."""
    width, height = 760, 490
    left, right, top, bottom = 75, 28, 145, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    highest = max(value for _, values in series for value in values)
    ceiling = highest * 1.15
    colors = ('#315a9b', '#16866a', '#c8713e', '#8057a5')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="35" font-family="sans-serif" font-size="22">{title}</text>',
             '<text x="75" y="126" font-family="sans-serif" font-size="13">Median step time (ms)</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        parts += [f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e5eb"/>',
                  f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{ceiling * fraction:.1f}</text>']
    positions = [left + (value - x_values[0]) * plot_width / (x_values[-1] - x_values[0])
                 for value in x_values]
    for value, x in zip(x_values, positions):
        parts += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top + plot_height}" y2="{top + plot_height + 6}" stroke="#777"/>',
                  f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:,}</text>']
    for index, (label, values) in enumerate(series):
        color = colors[index]
        points = ' '.join(f'{x:.1f},{top + plot_height * (1 - value / ceiling):.1f}'
                          for x, value in zip(positions, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, value in zip(positions, values):
            y = top + plot_height * (1 - value / ceiling)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        legend_x = left + (index % 2) * 310
        legend_y = 68 + (index // 2) * 23
        parts += [f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{legend_x + 29}" y="{legend_y + 4}" font-family="sans-serif" font-size="13">{label}</text>']
    parts += [f'<text x="{left + plot_width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
              '</svg>']
    path.write_text('\n'.join(parts) + '\n')


def save_trends(output, rows, token_counts, ranks):
    by_job = {(row['ranks'], row['total_tokens']): row for row in rows}
    if len(ranks) > 1:
        save_trend_plot(
            output / 'rank-trends.svg', 'Step time as ranks increase', 'Ranks', ranks,
            [(f'{tokens:,} tokens', [by_job[world, tokens]['median_step_ms']
                                    for world in ranks]) for tokens in token_counts],
        )
    if len(token_counts) > 1:
        save_trend_plot(
            output / 'data-trends.svg', 'Step time as data grows', 'Total tokens', token_counts,
            [(f'{world} rank' + ('s' if world > 1 else ''),
              [by_job[world, tokens]['median_step_ms'] for tokens in token_counts])
             for world in ranks],
        )


def save_readme(path, rows, larger_rows, extended_rows, two_block_rows):
    """Write the interpretation beside the plot and raw logs."""
    fixed_total = [row for row in rows if row['total_tokens'] == 256]
    fixed_local = [row for row in rows if row['local_tokens'] == 256]
    larger = {(row['ranks'], row['total_tokens']): row for row in larger_rows}
    extended = {(row['ranks'], row['total_tokens']): row for row in extended_rows}
    two_blocks = {(row['ranks'], row['total_tokens']): row for row in two_block_rows}
    table = [
        f'| {row["ranks"]} | {row["total_tokens"]:,} | {row["local_tokens"]:,} | '
        f'{row["parameter_bytes_per_rank"] / 1024:.1f} | '
        f'{row["median_step_ms"]:.3f} | {row["p95_step_ms"]:.3f} | '
        f'{row["tokens_per_second_at_median"]:,.0f} | '
        f'{row["expert_load_imbalance"]:.2f} |'
        for row in rows
    ]
    larger_table = [
        f'| {tokens:,} | '
        + ' | '.join(f'{larger[world, tokens]["median_step_ms"]:.3f}' for world in RANKS)
        + f' | {larger[1, tokens]["median_step_ms"] / larger[4, tokens]["median_step_ms"]:.2f}× |'
        for tokens in LARGER_TOKENS
    ]
    extended_table = [
        f'| {tokens:,} | '
        + ' | '.join(f'{extended[world, tokens]["median_step_ms"]:.3f}'
                     for world in EXTENDED_RANKS)
        + f' | {extended[1, tokens]["median_step_ms"] / extended[8, tokens]["median_step_ms"]:.2f}× |'
        for tokens in EXTENDED_TOKENS
    ]
    block_table = [
        f'| {world} | {extended[world, 16384]["median_step_ms"]:.3f} | '
        f'{two_blocks[world, 16384]["median_step_ms"]:.3f} | '
        f'{two_blocks[world, 16384]["median_step_ms"] / extended[world, 16384]["median_step_ms"]:.2f}× |'
        for world in EXTENDED_RANKS
    ]
    path.write_text(f'''# CPU scaling results

## Question and method

How do rank count, parameter storage, communication, and step time change when an MoE is split across ranks? These CPU/Gloo jobs use float64, learned top-2 routing, Adam at learning rate 0.001, and one CPU thread per worker. Each job ran 25 steps: 5 warmup and 20 measured. A barrier starts each timed step, and its duration is the slowest rank's time. Startup, logging, and correctness comparisons are outside the timed step.

We run three one-block sweeps. The first two use eight experts and hidden widths **{SMALL_EXPERT_HIDDEN}** and **{LARGER_EXPERT_HIDDEN}** across 1, 2, and 4 ranks. The third uses **16 experts**, hidden width **{EXTENDED_EXPERT_HIDDEN}**, and 1, 2, 4, and 8 ranks. All use top-2 routing and token dimension 32. Within each sweep, model size is fixed as ranks and total tokens change. A follow-up compares one and two blocks at the largest 16-expert batch.

## Small expert: communication dominates

| Ranks | Total tokens | Tokens/rank | Model KiB/rank | Median ms | p95 ms | Tokens/s at median | Expert imbalance |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

For fixed total work, model storage falls from **{fixed_total[0]['parameter_bytes_per_rank'] / 1024:.1f} to {fixed_total[-1]['parameter_bytes_per_rank'] / 1024:.1f} KiB per rank**, while median step time rises from **{fixed_total[0]['median_step_ms']:.3f} to {fixed_total[-1]['median_step_ms']:.3f} ms**. Expert weights are sharded; the shared projection and router still occupy {fixed_total[0]['shared_parameter_bytes_per_rank'] / 1024:.1f} KiB on every rank. This small CPU workload pays more for communication and coordination than it gains from dividing expert computation.

For fixed local work, total throughput is **{fixed_local[0]['tokens_per_second_at_median']:,.0f}, {fixed_local[1]['tokens_per_second_at_median']:,.0f}, and {fixed_local[2]['tokens_per_second_at_median']:,.0f} tokens/s** at 1, 2, and 4 ranks. Median step time grows from **{fixed_local[0]['median_step_ms']:.3f} to {fixed_local[-1]['median_step_ms']:.3f} ms**. This describes larger total work, not a fixed-batch speedup.

### Trends by rank and data size

Each point is a separate run's median step time. Lines in the first graph hold total tokens constant while ranks change; lines in the second hold ranks constant while total tokens change.

![Median step time versus rank count for fixed total token counts](rank-trends.svg)

![Median step time versus total tokens for fixed rank counts](data-trends.svg)

The original bar chart gives a compact view of the 256-token fixed-total comparison, the 256-tokens-per-rank diagonal, and parameter storage:

![CPU scaling bar chart](scaling.svg)

## Larger expert: more compute per exchanged token

With hidden width {LARGER_EXPERT_HIDDEN}, the following table shows median step latency for the same global token count across ranks. The last column is `1-rank time / 4-rank time`; values above 1 indicate a measured speedup.

| Total tokens | 1 rank ms | 2 ranks ms | 4 ranks ms | 4-rank speedup |
|---:|---:|---:|---:|---:|
{chr(10).join(larger_table)}

At {LARGER_TOKENS[-1]:,} tokens, the 4-rank run stores {larger[4, LARGER_TOKENS[-1]]['parameter_bytes_per_rank'] / 1024:.1f} KiB of model parameters per rank, versus {larger[1, LARGER_TOKENS[-1]]['parameter_bytes_per_rank'] / 1024:.1f} KiB on one rank. The trend graphs show where the larger expert workload begins to overcome dispatch and synchronization overhead:

![Larger expert latency versus ranks](larger/rank-trends.svg)

![Larger expert latency versus token count](larger/data-trends.svg)

The 1,024-token row is shared across the two model-size sweeps; only expert hidden width changes there. The remaining rows show the effect of increasing data within each model. These single-host CPU measurements are descriptive and are not a GPU speedup claim.

At a fixed **1,024 tokens per rank**, the larger workload processes 1,024, 2,048, and 4,096 total tokens at 1, 2, and 4 ranks. Median cluster throughput is {larger[1, 1024]['tokens_per_second_at_median']:,.0f}, {larger[2, 2048]['tokens_per_second_at_median']:,.0f}, and {larger[4, 4096]['tokens_per_second_at_median']:,.0f} tokens/s, respectively. This is a separate fixed-local-work comparison; the table above holds total work fixed.

## Extended sweep: 16 experts, 1,024 hidden units

The model has the same 16 experts at every rank count. At eight ranks, each worker owns two experts. Each row holds global token count fixed; the final column is 1-rank time divided by 8-rank time.

| Total tokens | 1 rank ms | 2 ranks ms | 4 ranks ms | 8 ranks ms | 8-rank speedup |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(extended_table)}

![Extended sweep latency versus ranks](extended/rank-trends.svg)

![Extended sweep latency versus token count](extended/data-trends.svg)

At 16,384 tokens, model parameter storage is {extended[1, 16384]['parameter_bytes_per_rank'] / 1024:.1f} KiB on one rank and {extended[8, 16384]['parameter_bytes_per_rank'] / 1024:.1f} KiB per rank on eight ranks. As with the other sweeps, these are single-host CPU timings and do not measure GPU performance.

Four ranks beat eight at 4,096 tokens; at 16,384 tokens, eight ranks are only {extended[4, 16384]['median_step_ms'] / extended[8, 16384]['median_step_ms']:.2f}× faster than four. The 16-expert run has an expert-load imbalance ratio of {extended[8, 16384]['expert_load_imbalance']:.2f} at the largest batch. Communication, synchronization, and uneven expert work can all limit eight-rank scaling; this experiment does not isolate their individual contributions.

### Does adding a block change scaling?

Holding 16 experts, hidden width 1,024, and 16,384 total tokens fixed, we compare one and two MoE blocks. Each block has its own projection, router, experts, and dispatch/return collectives. The last column is two-block time divided by one-block time at the same rank count.

| Ranks | One block ms | Two blocks ms | Time ratio |
|---:|---:|---:|---:|
{chr(10).join(block_table)}

![One versus two blocks across ranks](block-trends.svg)

The one-block 8-rank speedup is {extended[1, 16384]['median_step_ms'] / extended[8, 16384]['median_step_ms']:.2f}×, versus {two_blocks[1, 16384]['median_step_ms'] / two_blocks[8, 16384]['median_step_ms']:.2f}× with two blocks. Adding depth changes the model and doubles the number of routing/communication stages; it does not isolate expert-parallel efficiency as cleanly as changing ranks at fixed depth.

## Communication and interpretation

The 4-rank, 256-token job averages {fixed_total[-1]['remote_assignments_per_step']:.1f} remote token-expert assignments per measured step. The corresponding logical payload is {fixed_total[-1]['forward_activation_bytes_per_step']:,.0f} bytes in the forward activation phase and the same amount in backward, plus {fixed_total[-1]['routing_metadata_bytes_per_step']:,.0f} bytes of routing metadata. Across the one-block sweeps, the expert imbalance ratio ranges from {min(row['expert_load_imbalance'] for row in rows + larger_rows + extended_rows):.2f} to {max(row['expert_load_imbalance'] for row in rows + larger_rows + extended_rows):.2f}; this is the most-loaded expert's average assignment count divided by the mean.

The payload estimates count tensor elements: for `R` remote assignments summed across `B` blocks, dimension `d`, element size `b`, and `W` ranks, either activation phase moves `2Rdb` bytes (dispatch and return), while expert IDs and count exchange add `8R + 8BW(W-1)` bytes. They are not physical network measurements. Parameter bytes exclude gradients, Adam state, activations, and process overhead. Twenty measured steps describe this local run; they do not establish tail latency or GPU performance.

## Files

- [summary.json](summary.json): one derived metrics row for each of the nine small-expert jobs.
- [larger/summary.json](larger/summary.json): one derived metrics row for each of the twelve larger-expert jobs.
- [extended/summary.json](extended/summary.json): one derived metrics row for each of the twelve 16-expert jobs.
- [two-blocks/summary.json](two-blocks/summary.json): four jobs isolating block count at 16,384 tokens.
- [rank-trends.svg](rank-trends.svg): median latency against ranks at three fixed total token counts.
- [data-trends.svg](data-trends.svg): median latency against total tokens at three fixed rank counts.
- [scaling.svg](scaling.svg): the original fixed-total, fixed-local, and storage bar chart.
- [larger/rank-trends.svg](larger/rank-trends.svg) and [larger/data-trends.svg](larger/data-trends.svg): larger-expert latency trends.
- [extended/rank-trends.svg](extended/rank-trends.svg) and [extended/data-trends.svg](extended/data-trends.svg): 16-expert latency trends.
- [block-trends.svg](block-trends.svg): one versus two blocks at fixed model width and total tokens.
- `ranks{{R}}-tokens{{N}}.json` (in the sweep directories): per-rank configuration, loss, timing, routing counts, and parameter storage.
- `ranks{{R}}-tokens{{N}}.environment.json` (in the sweep directories): Python, PyTorch, host, dtype, and backend details for each job.
''')


def run_grid(output, expert_hidden, token_counts, *, experts=8, blocks=1,
             ranks=RANKS):
    """Launch a full ranks-by-tokens grid through run.py."""
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for total_tokens in token_counts:
        for world in ranks:
            path = output / f'ranks{world}-tokens{total_tokens}.json'
            subprocess.run([
                sys.executable, str(ROOT / 'run.py'), '--ranks', str(world),
                '--tokens', str(total_tokens), '--expert-hidden', str(expert_hidden),
                '--experts', str(experts), '--blocks', str(blocks),
                '--epochs', '25', '--warmup', '5', '--output', str(path),
            ], check=True, cwd=ROOT)
            row = summarize(json.loads(path.read_text()))
            rows.append(row)
            print(f'MEASURED experts={experts}, hidden={expert_hidden}, '
                  f'blocks={blocks}, {world} ranks, '
                  f'{total_tokens:,} tokens: {row["median_step_ms"]:.3f} ms median',
                  flush=True)
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    save_trends(output, rows, token_counts, ranks)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/scaling')
    args = parser.parse_args()
    args.output = args.output.resolve()
    small = run_grid(args.output, SMALL_EXPERT_HIDDEN, SMALL_TOKENS)
    larger = run_grid(args.output / 'larger', LARGER_EXPERT_HIDDEN, LARGER_TOKENS)
    extended = run_grid(args.output / 'extended', EXTENDED_EXPERT_HIDDEN,
                        EXTENDED_TOKENS, experts=16, ranks=EXTENDED_RANKS)
    two_blocks = run_grid(args.output / 'two-blocks', EXTENDED_EXPERT_HIDDEN,
                          BLOCK_COMPARISON_TOKENS, experts=16, blocks=2,
                          ranks=EXTENDED_RANKS)
    save_plot(args.output / 'scaling.svg', small)
    save_trend_plot(
        args.output / 'block-trends.svg', 'One versus two blocks as ranks increase',
        'Ranks', EXTENDED_RANKS,
        [('One block', [row['median_step_ms'] for row in extended
                        if row['total_tokens'] == 16384]),
         ('Two blocks', [row['median_step_ms'] for row in two_blocks])],
    )
    save_readme(args.output / 'README.md', small, larger, extended, two_blocks)
    print(f'Saved scaling results; see {args.output / "README.md"}')


if __name__ == '__main__':
    main()
