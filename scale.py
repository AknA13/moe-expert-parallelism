#!/usr/bin/env python3
"""Measure the unchanged training command at different rank counts and loads."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/scale'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for world in (1, 2, 4):
        for per_rank in (False, True):
            for route in ('learned', 'balanced', 'skewed'):
                regime = 'local' if per_rank else 'global'
                path = args.output / f'{world}-{regime}-{route}.json'
                command = [sys.executable, 'run.py', '--ranks', str(world),
                           '--tokens', '256', '--epochs', '25', '--warmup', '5',
                           '--routing', route, '--output', str(path)]
                if per_rank:
                    command.append('--tokens-per-rank')
                subprocess.run(command, check=True)
    print('PASS fixed-global and fixed-local scaling measurements')


if __name__ == '__main__':
    main()
