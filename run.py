#!/usr/bin/env python3
"""Train one standard or expert-parallel MoE on synthetic regression data."""
import argparse
import datetime
import json
import platform
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe.model import ModelConfig
from moe.training import TrainConfig, run_training


def worker(rank, world, init_url, train_config, model_config, capture_state, output):
    """Run one training job on this rank and collect the rank logs."""
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=init_url, rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        result = {'rank': rank, **run_training(
            rank, world, train_config, model_config, capture_state=capture_state)}
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(output).write_text(json.dumps(gathered, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--tokens', type=int, default=65,
                        help='total tokens across all ranks')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--capture-state', action='store_true',
                        help='save first-step tensors and final parameters for verification')
    parser.add_argument('--output', type=Path, default=Path('results/training.json'))
    args = parser.parse_args()
    if args.tokens < args.ranks or args.epochs < 1 or args.lr <= 0:
        parser.error('tokens must cover every rank; epochs and learning rate must be positive')
    if args.warmup < 0 or args.warmup >= args.epochs:
        parser.error('warmup must be between zero and epochs - 1')
    if args.blocks < 1:
        parser.error('blocks must be positive')

    train_config = TrainConfig(args.tokens, args.epochs, args.lr, args.warmup)
    model_config = ModelConfig(num_blocks=args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'platform': platform.platform(), 'machine': platform.machine(),
        'threads_per_worker': 1, 'dtype': 'float64', 'backend': 'gloo',
        'transport': 'variable-split all_to_all_single',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.output.with_suffix('.environment.json').write_text(
        json.dumps(environment, indent=2) + '\n')
    with tempfile.TemporaryDirectory() as directory:
        init_url = Path(directory, 'process-group').as_uri()
        mp.spawn(worker, args=(args.ranks, init_url, train_config, model_config,
                               args.capture_state, str(args.output.resolve())),
                 nprocs=args.ranks, join=True)
    print(f'PASS {args.ranks} ranks, {args.epochs} epochs, '
          f'{args.blocks} blocks: {args.output}', flush=True)


if __name__ == '__main__':
    main()
