#!/usr/bin/env python3
"""Compare independently launched standard and parallel training runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ATOL = 1e-8
RTOL = 1e-6


def close(actual, expected):
    actual = torch.as_tensor(actual, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return float((actual - expected).abs().max()) if actual.numel() else 0.0


def run(output, world, blocks, tokens, epochs):
    path = output / f'blocks{blocks}-ranks{world}.json'
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
    assert reference['model_type'] == 'StandardMoEModel'
    assert sum(len(rank['first_step']['output']) for rank in parallel) == total_tokens
    errors = {name: 0.0 for name in (
        'loss', 'output', 'input_gradient', 'parameter_gradient',
        'parameter_after_step', 'final_parameter', 'expert_output')}

    for rank_id, rank in enumerate(parallel):
        assert rank['rank'] == rank_id
        assert rank['model_type'] == 'ParallelMoEModel'
        assert rank['train_config'] == reference['train_config']
        assert rank['model_config'] == reference['model_config']
        local_tokens = total_tokens // world + (rank_id < total_tokens % world)
        assert len(rank['trace']) == blocks
        for epoch, (actual, expected) in enumerate(zip(rank['epochs'],
                                                       reference['epochs'])):
            assert actual['epoch'] == epoch
            errors['loss'] = max(errors['loss'], close(
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
                errors['expert_output'] = max(errors['expert_output'], close(
                    row['returned_first_coordinate'],
                    expected_row['returned_first_coordinate']))
        for name in ('parameter_gradient', 'parameter_after_step'):
            for parameter, value in rank['first_step'][name].items():
                errors[name] = max(errors[name], close(
                    value, reference['first_step'][name][parameter]))
        for parameter, value in rank['final_parameters'].items():
            errors['final_parameter'] = max(errors['final_parameter'], close(
                value, reference['final_parameters'][parameter]))
        changes = rank['parameter_changes']
        assert all(change > 0 for name, change in changes.items()
                   if name.startswith('token_encoder.') or '.router.' in name)

    for name in ('output', 'input_gradient'):
        global_values = [row for rank in parallel
                         for row in rank['first_step'][name]]
        errors[name] = close(global_values, reference['first_step'][name])
    if world > 1:
        assert sum(rank['epochs'][0]['blocks'][0]['remote_assignments']
                   for rank in parallel) > 0
    if blocks == 1:
        assert reference['epochs'][-1]['global_loss'] < reference['epochs'][0]['global_loss']
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/verify'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for blocks, tokens, epochs in ((1, 65, 30), (2, 17, 2)):
        standard = run(args.output, 1, blocks, tokens, epochs)
        for world in (2, 4):
            parallel = run(args.output, world, blocks, tokens, epochs)
            errors = compare(standard, parallel)
            summary.append({'blocks': blocks, 'ranks': world,
                            'total_tokens': tokens, 'epochs': epochs,
                            'max_errors': errors})
            print(f'PASS {blocks} blocks, {world} ranks: '
                  f'max error {max(errors.values()):.3g}', flush=True)
    (args.output / 'comparison-summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print('PASS standard versus expert-parallel training')


if __name__ == '__main__':
    main()
