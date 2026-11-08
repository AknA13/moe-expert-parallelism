#!/usr/bin/env python3
"""Check standard and expert-parallel training through run.py."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ATOL = 1e-8
RTOL = 1e-6


def close(actual, expected):
    actual = torch.as_tensor(actual, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return float((actual - expected).abs().max()) if actual.numel() else 0.0


def checked_close(label, actual, expected):
    try:
        return close(actual, expected)
    except AssertionError:
        print(f'FAIL {label}', flush=True)
        raise


def max_abs(values):
    return float(torch.as_tensor(values, dtype=torch.float64).abs().max())


def run(output, world, blocks, tokens, epochs):
    path = output / f'blocks{blocks}-epochs{epochs}-ranks{world}.json'
    subprocess.run([
        sys.executable, 'run.py', '--ranks', str(world),
        '--blocks', str(blocks), '--tokens', str(tokens),
        '--epochs', str(epochs), '--capture-state', '--output', str(path),
    ], check=True)
    ranks = json.loads(path.read_text())
    assert len(ranks) == world
    return ranks


def compare(standard, parallel):
    reference = standard[0]
    world = len(parallel)
    total_tokens = reference['train_config']['total_tokens']
    blocks = reference['model_config']['num_blocks']
    assert reference['parallel'] is False
    assert sum(len(rank['first_step']['output']) for rank in parallel) == total_tokens
    errors = {name: 0.0 for name in (
        'loss', 'output', 'input_gradient', 'parameter_gradient',
        'parameter_after_step', 'final_parameter', 'expert_output')}

    for rank_id, rank in enumerate(parallel):
        assert rank['rank'] == rank_id
        assert rank['parallel'] is True
        assert rank['train_config'] == reference['train_config']
        assert rank['model_config'] == reference['model_config']
        local_tokens = total_tokens // world + (rank_id < total_tokens % world)
        assert len(rank['trace']) == blocks
        for epoch, (actual, expected) in enumerate(zip(rank['epochs'],
                                                       reference['epochs'])):
            assert actual['epoch'] == epoch
            errors['loss'] = max(errors['loss'], checked_close('global loss',
                actual['global_loss'], expected['global_loss']))
            assert len(actual['blocks']) == blocks
            for report in actual['blocks']:
                assert sum(report['assignments']) == 2 * local_tokens
        for block_id, block_trace in enumerate(rank['trace']):
            assert block_trace['block'] == block_id
            assert len(block_trace['tokens']) == local_tokens
            expected_by_token = {row['token']: row for row in
                                 reference['trace'][block_id]['tokens']}
            for row in block_trace['tokens']:
                expected_row = expected_by_token[row['token']]
                assert row['experts'] == expected_row['experts']
                errors['expert_output'] = max(errors['expert_output'], checked_close('expert output',
                    row['returned_first_coordinate'],
                    expected_row['returned_first_coordinate']))
        for name in ('parameter_gradient', 'parameter_after_step'):
            for parameter, value in rank['first_step'][name].items():
                errors[name] = max(errors[name], checked_close(name,
                    value, reference['first_step'][name][parameter]))
        for parameter, value in rank['final_parameters'].items():
            errors['final_parameter'] = max(errors['final_parameter'], checked_close('final parameter',
                value, reference['final_parameters'][parameter]))
        changes = rank['parameter_changes']
        assert all(change > 0 for name, change in changes.items()
                   if '.token_projection.' in name or '.router.' in name)

    for name in ('output', 'input_gradient'):
        global_values = [row for rank in parallel
                         for row in rank['first_step'][name]]
        errors[name] = checked_close(name, global_values, reference['first_step'][name])
    if world > 1:
        assert sum(rank['epochs'][0]['blocks'][0]['remote_assignments']
                   for rank in parallel) > 0
    if blocks == 1 and len(reference['epochs']) > 1:
        assert reference['epochs'][-1]['global_loss'] < reference['epochs'][0]['global_loss']
    return errors


def update_checks(ranks):
    """Summarize one-step gradients and changes by global parameter owner."""
    world = len(ranks)
    experts = ranks[0]['model_config']['experts']
    assignments = [sum(rank['epochs'][0]['blocks'][0]['assignments'][expert]
                       for rank in ranks) for expert in range(experts)]
    rows = []
    for name, owner in [('token_projection', None), ('router', None)] + [
        (f'experts.{expert}', expert // (experts // world))
        for expert in range(experts)
    ]:
        selected = ranks if owner is None else [ranks[owner]]
        gradients, changes = [], []
        for rank in selected:
            names = [key for key in rank['parameter_changes']
                     if f'.{name}.' in key]
            assert names, f'no parameters found for {name}'
            gradients.extend(max_abs(rank['first_step']['parameter_gradient'][key])
                             for key in names)
            changes.extend(rank['parameter_changes'][key] for key in names)
        row = {'parameter': name, 'owner': owner,
               'assignments': None if owner is None else assignments[int(name.split('.')[1])],
               'minimum_tensor_gradient': min(gradients),
               'minimum_tensor_change': min(changes)}
        try:
            assert row['minimum_tensor_gradient'] > 0, f'{name} did not receive a gradient'
            assert row['minimum_tensor_change'] > 0, f'{name} did not update'
            if owner is not None:
                assert row['assignments'] > 0, f'{name} received no assignments'
        except AssertionError:
            print(f'FAIL {world} ranks: {name} gradient or update', flush=True)
            raise
        rows.append(row)
        print(f'PASS {world} ranks: {name} has a gradient and updates', flush=True)
    return rows


def save_loss_plot(path, standard, two, four):
    """Plot the three global loss curves as a dependency-free SVG."""
    curves = [([step['global_loss'] for step in run[0]['epochs']], color, dash)
              for run, color, dash in ((standard, '#253c84', ''),
                                        (two, '#dc6945', '8 5'),
                                        (four, '#16866a', '3 4'))]
    width, height = 760, 430
    left, right, top, bottom = 78, 28, 62, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(max(values) for values, _, _ in curves) * 1.08
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="78" y="35" font-family="sans-serif" font-size="21">Global training loss</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        parts += [f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e4e7ed"/>',
                  f'<text x="68" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum * fraction:.3f}</text>']
    for (values, color, dash), label in zip(curves, ('1 rank', '2 ranks', '4 ranks')):
        points = ' '.join(f'{left + i * plot_width / (len(values) - 1):.1f},{top + plot_height * (1 - value / maximum):.1f}'
                          for i, value in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        position = {'1 rank': 210, '2 ranks': 330, '4 ranks': 460}[label]
        parts += [f'<line x1="{position}" x2="{position + 22}" y1="400" y2="400" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>',
                  f'<text x="{position + 28}" y="404" font-family="sans-serif" font-size="13">{label}</text>']
    parts += [f'<text x="{left + plot_width / 2:.1f}" y="378" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch (0–{len(curves[0][0]) - 1})</text>',
              '</svg>']
    path.write_text('\n'.join(parts) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/correctness'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, updates = [], []
    with tempfile.TemporaryDirectory() as directory:
        raw_output = Path(directory)
        for blocks, tokens, epochs in ((1, 65, 1), (1, 65, 30), (2, 17, 2)):
            standard = run(raw_output, 1, blocks, tokens, epochs)
            parallel_runs = {}
            for world in (2, 4):
                parallel = run(raw_output, world, blocks, tokens, epochs)
                parallel_runs[world] = parallel
                try:
                    errors = compare(standard, parallel)
                except Exception:
                    print(f'FAIL {blocks} blocks, {epochs} epochs, {world} ranks', flush=True)
                    raise
                summary.append({'blocks': blocks, 'ranks': world,
                                'total_tokens': tokens, 'epochs': epochs,
                                'max_errors': errors})
                for category, error in errors.items():
                    print(f'PASS {blocks} blocks, {epochs} epochs, {world} ranks: '
                          f'{category} (max error {error:.3g})', flush=True)
                print(f'PASS {blocks} blocks, {epochs} epochs, {world} ranks: '
                      'routing IDs, assignment counts, shared replicas, and remote traffic',
                      flush=True)
                if blocks == 1 and epochs == 1:
                    updates.append({'ranks': world, 'parameters': update_checks(parallel)})
                if blocks == 1 and epochs == 30 and world == 2:
                    mixed_tokens = []
                    for rank in parallel:
                        rows = rank['trace'][0]['tokens']
                        mixed_tokens.append(next(
                            (row for row in rows if len(set(row['owners'])) == 2),
                            rows[0],
                        ))
                    demo = {
                        'initial_loss': standard[0]['epochs'][0]['global_loss'],
                        'final_loss': standard[0]['epochs'][-1]['global_loss'],
                        'mixed_owner_tokens': mixed_tokens,
                    }
            if blocks == 1 and epochs == 30:
                save_loss_plot(args.output / 'loss.svg', standard,
                               parallel_runs[2], parallel_runs[4])
    (args.output / 'comparison-summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    (args.output / 'demo.json').write_text(json.dumps(demo, indent=2) + '\n')
    (args.output / 'parameter-updates.json').write_text(
        json.dumps(updates, indent=2) + '\n')
    print('PASS all correctness checks; see results/correctness/loss.svg')


if __name__ == '__main__':
    main()
