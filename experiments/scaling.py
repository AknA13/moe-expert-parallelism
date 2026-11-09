#!/usr/bin/env python3
"""Run CPU scaling experiments through run.py and summarize their measurements."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


RANKS = (1, 2, 4)
LARGER_RANKS = (1, 2, 4, 8)
EXTENDED_RANKS = (1, 2, 4, 8)
SMALL_TOKENS = (256, 512, 1024)
LARGER_TOKENS = (1024, 2048, 4096, 8192)
EXTENDED_TOKENS = (4096, 8192, 16384)
BLOCK_COMPARISON_TOKENS = (16384,)
SMALL_EXPERT_HIDDEN = 64
LARGER_EXPERT_HIDDEN = 512
EXTENDED_EXPERT_HIDDEN = 1024
ADDITIONAL_EXPERT_WIDTHS = (2048, 4096)
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
    parameter_bytes = ranks[0]['parameter_bytes_per_rank']
    # Adam keeps two float64 moments and one float32 step scalar per tensor.
    parameter_tensors = blocks * (
        4 + 4 * (ranks[0]['model_config']['experts'] // world))
    return {
        'ranks': world,
        'total_tokens': tokens,
        'local_tokens': tokens // world,
        'parameter_bytes_per_rank': parameter_bytes,
        'adam_state_bytes_per_rank': 2 * parameter_bytes + 4 * parameter_tensors,
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


def save_trend_plot(path, title, x_label, x_values, series,
                    y_label='Median step time (ms)', *, categorical_x=False,
                    reference_value=None, last_value_format=None,
                    tick_format='.1f', tick_round_to=None, y_limit=None):
    """Plot measured values against one experiment axis."""
    width, height = 760, 490
    left, right, top, bottom = 75, 28, 145, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    highest = max(value for _, values in series for value in values)
    if reference_value is not None:
        highest = max(highest, reference_value)
    ceiling = highest * 1.15 if y_limit is None else y_limit
    colors = ('#315a9b', '#16866a', '#c8713e', '#8057a5')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="35" font-family="sans-serif" font-size="22">{title}</text>',
             f'<text x="75" y="126" font-family="sans-serif" font-size="13">{y_label}</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        tick = ceiling * fraction
        if tick_round_to is not None:
            tick = round(tick / tick_round_to) * tick_round_to
        parts += [f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e5eb"/>',
                  f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:{tick_format}}</text>']
    if categorical_x:
        positions = [left + index * plot_width / (len(x_values) - 1)
                     for index in range(len(x_values))]
    else:
        positions = [left + (value - x_values[0]) * plot_width / (x_values[-1] - x_values[0])
                     for value in x_values]
    for value, x in zip(x_values, positions):
        parts += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top + plot_height}" y2="{top + plot_height + 6}" stroke="#777"/>',
                  f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:,}</text>']
    if reference_value is not None:
        y = top + plot_height * (1 - reference_value / ceiling)
        parts += [f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#777" stroke-dasharray="5 5"/>',
                  f'<text x="{left + plot_width - 7}" y="{y + 20:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#555">{reference_value:g}× one-rank baseline</text>']
    for index, (label, values) in enumerate(series):
        color = colors[index]
        points = ' '.join(f'{x:.1f},{top + plot_height * (1 - value / ceiling):.1f}'
                          for x, value in zip(positions, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, value in zip(positions, values):
            y = top + plot_height * (1 - value / ceiling)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        if last_value_format is not None:
            y = top + plot_height * (1 - values[-1] / ceiling)
            label_y = y + 20 if len(series) > 1 and index == 0 else y - 10
            parts.append(f'<text x="{positions[-1] - 10:.1f}" y="{label_y:.1f}" text-anchor="end" font-family="sans-serif" font-size="13" font-weight="bold" fill="{color}">{values[-1]:{last_value_format}}</text>')
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


def save_story_plots(output, larger_rows, extended_rows, width_rows):
    """Build the three presentation figures and one fixed-local backup."""
    larger = {(row['ranks'], row['total_tokens']): row for row in larger_rows}
    by_width = {
        width: {(row['ranks'], row['total_tokens']): row for row in sweep}
        for width, sweep in {1024: extended_rows, **width_rows}.items()
    }
    widths = (1024, *ADDITIONAL_EXPERT_WIDTHS)

    save_trend_plot(
        output / 'batch-crossover.svg', 'More tokens improve rank scaling',
        'Total tokens (doubling)', LARGER_TOKENS,
        [(f'{world} ranks', [larger[1, tokens]['median_step_ms'] /
                            larger[world, tokens]['median_step_ms']
                            for tokens in LARGER_TOKENS])
         for world in (2, 4, 8)],
        y_label='Speedup over one rank', categorical_x=True,
        reference_value=1, y_limit=2,
    )
    save_trend_plot(
        output / 'rank-speedup-by-width.svg',
        'Wider experts improve rank scaling', 'Ranks (doubling)', EXTENDED_RANKS,
        [(f'Expert width {width:,}', [
            by_width[width][1, 16384]['median_step_ms'] /
            by_width[width][world, 16384]['median_step_ms']
            for world in EXTENDED_RANKS]) for width in widths],
        y_label='Speedup over one rank', categorical_x=True,
        reference_value=1, last_value_format='.2f',
    )
    save_trend_plot(
        output / 'parameter-storage.svg', 'Sharding reduces per-rank storage',
        'Ranks (doubling)', EXTENDED_RANKS,
        [('Model parameters', [
            by_width[4096][world, 16384]['parameter_bytes_per_rank'] / 1024**2
            for world in EXTENDED_RANKS]),
         ('Adam state', [
            by_width[4096][world, 16384]['adam_state_bytes_per_rank'] / 1024**2
            for world in EXTENDED_RANKS]),
         ('Parameters + Adam state', [
            (by_width[4096][world, 16384]['parameter_bytes_per_rank'] +
             by_width[4096][world, 16384]['adam_state_bytes_per_rank']) / 1024**2
            for world in EXTENDED_RANKS])],
        y_label='Storage per rank (MiB)', categorical_x=True,
        tick_format='.0f', y_limit=120,
    )
    save_trend_plot(
        output / 'fixed-local-throughput.svg',
        'Throughput with 2,048 tokens per rank', 'Ranks (doubling)', (2, 4, 8),
        [('Expert width 4,096', [
            by_width[4096][world, 2048 * world]['tokens_per_second_at_median']
            for world in (2, 4, 8)])],
        y_label='Cluster throughput (tokens/s)', categorical_x=True,
        last_value_format=',.0f', tick_format=',.0f', tick_round_to=1000,
    )


def save_readme(path, rows, larger_rows, extended_rows, two_block_rows,
                width_rows):
    """Explain the scaling evidence through figures and short captions."""
    small = {(row['ranks'], row['total_tokens']): row for row in rows}
    larger = {(row['ranks'], row['total_tokens']): row for row in larger_rows}
    extended = {(row['ranks'], row['total_tokens']): row for row in extended_rows}
    two_blocks = {(row['ranks'], row['total_tokens']): row for row in two_block_rows}
    middle = {(row['ranks'], row['total_tokens']): row
              for row in width_rows[2048]}
    widest = {(row['ranks'], row['total_tokens']): row
              for row in width_rows[4096]}
    two_rank_speedups = [
        larger[1, tokens]['median_step_ms'] / larger[2, tokens]['median_step_ms']
        for tokens in LARGER_TOKENS]
    path.write_text(f"""# Scaling Results (on CPU)

How do training step time, parameter storage, and communication overhead change as we split the MoE across more ranks? These CPU jobs use float64, learned top-2 routing, Adam at learning rate 0.001, and one CPU thread per worker/rank. Each job ran 25 steps: 5 warmup and 20 measured. The step time is measured as the duration of the slowest rank.

## How to read the figures

Each point represents the median of the 20 measured steps. **Speedup means one-rank step time divided by multi-rank step time using the same data and model configuration.**

## 1. More data per-training step makes parallelism more effective

![Two-rank, four-rank, and eight-rank speedup versus total tokens for eight experts with hidden width 512](batch-crossover.svg)

This plot holds the eight-expert, 512-wide expert MLP fixed. The two-rank line ranges from **{min(two_rank_speedups):.2f}× to {max(two_rank_speedups):.2f}×**. At 4-rank, the speedup rises from **{larger[1, 1024]['median_step_ms'] / larger[4, 1024]['median_step_ms']:.2f}× at 1,024 tokens** to **{larger[1, 8192]['median_step_ms'] / larger[4, 8192]['median_step_ms']:.2f}× at 8,192 tokens**. At 8-rank, it rises from **{larger[1, 1024]['median_step_ms'] / larger[8, 1024]['median_step_ms']:.2f}× to {larger[1, 8192]['median_step_ms'] / larger[8, 8192]['median_step_ms']:.2f}×** over the same range, but remains below the four-rank line. The dashed 1× line marks the baseline one-rank result. A smaller width-64 expert MLP is slower with four ranks even at 1,024 tokens (**{small[1, 1024]['median_step_ms']:.1f} versus {small[4, 1024]['median_step_ms']:.1f} ms**); its [rank-latency plot](rank-trends.svg) shows the overhead-dominated case.

## 2. Wider experts improve rank scaling

![One-to-eight-rank speedup at 16,384 total tokens for three expert widths](rank-speedup-by-width.svg)

This plot fixes **16 experts, 16,384 total tokens, top-2 routing, token dimension 32, and one MoE block**. Each line represents one width for the expert MLP. Using rank-8 speedup reaches **{extended[1, 16384]['median_step_ms'] / extended[8, 16384]['median_step_ms']:.2f}×**, **{middle[1, 16384]['median_step_ms'] / middle[8, 16384]['median_step_ms']:.2f}×**, and **{widest[1, 16384]['median_step_ms'] / widest[8, 16384]['median_step_ms']:.2f}×** at widths 1,024, 2,048, and 4,096. At width 4,096, the step time falls from **{widest[1, 16384]['median_step_ms']:.1f} to {widest[8, 16384]['median_step_ms']:.1f} ms**.

## 3. Sharding reduces model and optimizer storage per rank

![Model parameters, Adam state, and their total per rank for 16 experts with hidden width 4096](parameter-storage.svg)

With 16 experts at width 4,096, measured model parameters fall from **{widest[1, 16384]['parameter_bytes_per_rank'] / 1024**2:.1f} MiB on one rank** to **{widest[8, 16384]['parameter_bytes_per_rank'] / 1024**2:.1f} MiB per rank on eight ranks**. Adam state falls from **{widest[1, 16384]['adam_state_bytes_per_rank'] / 1024**2:.1f} to {widest[8, 16384]['adam_state_bytes_per_rank'] / 1024**2:.1f} MiB per rank**. Together, they fall from **{(widest[1, 16384]['parameter_bytes_per_rank'] + widest[1, 16384]['adam_state_bytes_per_rank']) / 1024**2:.1f} to {(widest[8, 16384]['parameter_bytes_per_rank'] + widest[8, 16384]['adam_state_bytes_per_rank']) / 1024**2:.1f} MiB per rank**. Each rank retains its own copy of the projection and router layers. Experts are divided and owned among different ranks. Adam state is calculated from two float64 moment tensors and one 4-byte step scalar per parameter tensor. The plot excludes gradients, activations, and process memory.

## Supporting views

### Increasing total work with ranks

![Cluster throughput with 2048 tokens per rank at two four and eight ranks](fixed-local-throughput.svg)

This is a **fixed-local-batch** comparison: 2,048 tokens per rank, so total tokens rise from 4,096 to 16,384 as ranks rise from two to eight. For the width-4,096 model, throughput grows from **{widest[2, 4096]['tokens_per_second_at_median']:,.0f} to {widest[8, 16384]['tokens_per_second_at_median']:,.0f} tokens/s**, while step time grows from **{widest[2, 4096]['median_step_ms']:.1f} to {widest[8, 16384]['median_step_ms']:.1f} ms**. This is different from the fixed-global-batch speedup in Figure 2. The saved grid has no one-rank, 2,048-token point for this particular comparison.

### Additional experiments

- The [small-expert plot](rank-trends.svg) and [larger-expert plot](larger/rank-trends.svg) show absolute latency as ranks change at fixed total tokens.
- The [width-versus-speedup plot](width-speedup.svg) shows how batch size and expert width affect eight-rank speedup. Each width also has its own rank and data trend plots in `extended/`, `width-2048/`, and `width-4096/`.
- The [one-versus-two-block plot](block-trends.svg) shows the depth check at 16,384 tokens and width 1,024. Eight-rank speedup is **{extended[1, 16384]['median_step_ms'] / extended[8, 16384]['median_step_ms']:.2f}× with one block** and **{two_blocks[1, 16384]['median_step_ms'] / two_blocks[8, 16384]['median_step_ms']:.2f}× with two**. A second block adds another routing and communication stage, so this is a separate model-depth question.

## Interpretation and limits

The width-4,096, eight-rank, 16,384-token job averages **{widest[8, 16384]['remote_assignments_per_step']:,.0f} remote token-expert assignments** per measured step. Its logical forward activation payload is **{widest[8, 16384]['forward_activation_bytes_per_step'] / 1024**2:.1f} MiB**, and its backward activation payload is the same. These count tensor elements, not physical network traffic or time spent in collectives. At this configuration, the most-loaded expert receives **{widest[8, 16384]['expert_load_imbalance']:.2f}×** the average assignment count. We have not separately timed communication and expert computation, so the exact sources of the eight-rank plateau remain unresolved. These single-host CPU results do not establish GPU speedup.

## Reproduce and inspect

Run `python experiments/scaling.py` from the repository root. It reproduces all **65 jobs** and regenerates the plots, summaries, and this report. The job grids and their derived `summary.json` files are in this directory (width 64), `larger/` (width 512), `extended/` (16 experts, width 1,024), `width-2048/`, `width-4096/`, and `two-blocks/`. Each `ranks{{R}}-tokens{{N}}.json` contains per-rank configuration, loss, timing, routing counts, and parameter storage; its companion `.environment.json` records the runtime and host details.
""")



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
    larger = run_grid(args.output / 'larger', LARGER_EXPERT_HIDDEN,
                      LARGER_TOKENS, ranks=LARGER_RANKS)
    extended = run_grid(args.output / 'extended', EXTENDED_EXPERT_HIDDEN,
                        EXTENDED_TOKENS, experts=16, ranks=EXTENDED_RANKS)
    width_runs = {
        width: run_grid(args.output / f'width-{width}', width,
                        EXTENDED_TOKENS, experts=16, ranks=EXTENDED_RANKS)
        for width in ADDITIONAL_EXPERT_WIDTHS
    }
    two_blocks = run_grid(args.output / 'two-blocks', EXTENDED_EXPERT_HIDDEN,
                          BLOCK_COMPARISON_TOKENS, experts=16, blocks=2,
                          ranks=EXTENDED_RANKS)
    save_plot(args.output / 'scaling.svg', small)
    save_story_plots(args.output, larger, extended, width_runs)
    save_trend_plot(
        args.output / 'block-trends.svg', 'One versus two blocks as ranks increase',
        'Ranks', EXTENDED_RANKS,
        [('One block', [row['median_step_ms'] for row in extended
                        if row['total_tokens'] == 16384]),
         ('Two blocks', [row['median_step_ms'] for row in two_blocks])],
    )
    width_by_job = {
        width: {(row['ranks'], row['total_tokens']): row for row in sweep}
        for width, sweep in {1024: extended, **width_runs}.items()
    }
    save_trend_plot(
        args.output / 'width-speedup.svg', 'Eight-rank speedup versus expert width',
        'Expert hidden width', (1024, *ADDITIONAL_EXPERT_WIDTHS),
        [(f'{tokens:,} tokens', [
            width_by_job[width][1, tokens]['median_step_ms'] /
            width_by_job[width][8, tokens]['median_step_ms']
            for width in (1024, *ADDITIONAL_EXPERT_WIDTHS)])
         for tokens in EXTENDED_TOKENS],
        y_label='One-rank time / eight-rank time',
    )
    save_readme(args.output / 'README.md', small, larger, extended,
                two_blocks, width_runs)
    print(f'Saved scaling results; see {args.output / "README.md"}')


if __name__ == '__main__':
    main()
