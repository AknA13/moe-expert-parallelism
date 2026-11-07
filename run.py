#!/usr/bin/env python3
"""Train one expert-parallel model on a deterministic regression dataset."""
import argparse
import datetime
import json
import platform
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe.training import TrainConfig, run_training


def worker(rank, world, init_url, config, output):
    """Run the same training job on every rank and collect its log."""
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=init_url, rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        result = {'rank': rank, **run_training(rank, world, config)}
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(output).write_text(json.dumps(gathered, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--tokens', type=int, default=65)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--routing', choices=('learned', 'balanced', 'skewed', 'local'),
                        default='learned')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--tokens-per-rank', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/training.json'))
    args = parser.parse_args()
    if args.tokens < 1 or args.epochs < 1 or args.lr <= 0:
        parser.error('tokens, epochs, and learning rate must be positive')
    if args.warmup < 0 or args.warmup >= args.epochs:
        parser.error('warmup must be between zero and epochs - 1')
    if not args.tokens_per_rank and args.tokens < args.ranks:
        parser.error('global token count must be at least the number of ranks')

    config = TrainConfig(
        tokens=args.tokens, epochs=args.epochs, learning_rate=args.lr,
        routing=args.routing, warmup=args.warmup,
        tokens_per_rank=args.tokens_per_rank, validate=args.validate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'platform': platform.platform(), 'machine': platform.machine(),
        'threads_per_worker': 1, 'dtype': 'float64',
        'dimensions': [32, 64, 8], 'backend': 'gloo',
        'transport': 'variable-split all_to_all_single',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    environment_path = args.output.with_suffix('.environment.json')
    environment_path.write_text(json.dumps(environment, indent=2) + '\n')
    with tempfile.TemporaryDirectory() as directory:
        init_url = Path(directory, 'process-group').as_uri()
        mp.spawn(worker, args=(args.ranks, init_url, config, str(args.output.resolve())),
                 nprocs=args.ranks, join=True)
    print(f'PASS {args.ranks} ranks, {config.epochs} epochs: {args.output}', flush=True)


if __name__ == '__main__':
    main()
