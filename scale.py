#!/usr/bin/env python3
"""Measure the same trainer as token count and rank count change."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/scale'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # The one-rank 256-token job is shared by both comparisons.
    for world, total_tokens in ((1, 256), (2, 256), (4, 256),
                                (2, 512), (4, 1024)):
        path = args.output / f'ranks{world}-tokens{total_tokens}.json'
        subprocess.run([
            sys.executable, 'run.py', '--ranks', str(world),
            '--tokens', str(total_tokens), '--epochs', '25',
            '--warmup', '5', '--output', str(path),
        ], check=True)
    print('PASS fixed-total and growing-batch scaling measurements')


if __name__ == '__main__':
    main()
