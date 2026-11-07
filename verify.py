#!/usr/bin/env python3
"""Run correctness checks by varying inputs to the ordinary training command."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def check(path, world, *, every_parameter=False, loss_reduction=False):
    ranks = json.loads(path.read_text())
    assert len(ranks) == world
    for rank, result in enumerate(ranks):
        assert result['rank'] == rank
        config = result['config']
        token_count = config['tokens'] * world if config['tokens_per_rank'] else config['tokens']
        local_count = token_count // world + (rank < token_count % world)
        assert len(result['steps']) == config['epochs']
        assert len(result['trace']) == local_count
        for step in result['steps']:
            assert sum(step['routing']['assignments']) == 2 * local_count
            errors = step['reference_error']
            assert set(errors) == {'output', 'loss', 'input_gradient',
                                   'parameter_gradient', 'parameter_update'}
            assert max(errors.values()) < 1e-8
        if every_parameter:
            assert all(change > 0 for change in result['parameter_changes'].values())
        if loss_reduction:
            assert result['steps'][-1]['global_loss'] < result['steps'][0]['global_loss']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/verify'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for world in (1, 2, 4):
        for route in ('learned', 'balanced', 'skewed', 'local'):
            path = args.output / f'{world}-{route}.json'
            subprocess.run([sys.executable, 'run.py', '--ranks', str(world),
                            '--tokens', '65', '--epochs', '1', '--routing', route,
                            '--validate', '--output', str(path)], check=True)
            check(path, world, every_parameter=route == 'balanced')
        path = args.output / f'{world}-training.json'
        subprocess.run([sys.executable, 'run.py', '--ranks', str(world),
                        '--validate', '--output', str(path)], check=True)
        check(path, world, loss_reduction=True)
    print('PASS reference, routing, and training checks')


if __name__ == '__main__':
    main()
