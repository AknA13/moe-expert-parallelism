#!/usr/bin/env python3
"""Spawn isolated Gloo workers; reference validation never enters benchmark timing."""
import argparse
import datetime
import json
import platform
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from moe.experiments import run_verify, run_train, run_scale, run_trace

TASKS = {
    'verify': run_verify,
    'train': run_train,
    'scale': run_scale,
    'trace': run_trace,
}


def worker(rank, world, rendezvous, task, output):
    """Join the process group, run one injected task, and collect its result."""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=world,
                            timeout=datetime.timedelta(seconds=90))
    try:
        result = {'world_size': world, **task(rank, world)}
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(output).write_text(json.dumps(gathered, indent=2) + '\n')
    finally:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=[*TASKS, 'all'])
    parser.add_argument('--ranks',nargs='+',type=int,choices=[1,2,4],default=[1,2,4])
    parser.add_argument('--output',type=Path,default=Path('results'))
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    commands = list(TASKS) if args.command == 'all' else [args.command]
    environment = {'python':platform.python_version(),'torch':torch.__version__,
                   'platform':platform.platform(),'machine':platform.machine(),
                   'threads_per_worker':1,'dtype':'float64','dimensions':[32,64,8],
                   'learning_rate':.15,'backend':'gloo','transport':'variable-split all_to_all_single',
                   'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    for command in commands:
        for world in args.ranks:
            path = args.output/f'{command}-{world}.json'
            with tempfile.TemporaryDirectory() as tmp:
                mp.spawn(worker,args=(world,Path(tmp,'rendezvous').as_uri(),TASKS[command],str(path.resolve())),nprocs=world,join=True)
            print(f'PASS {command} ranks={world}: {path}',flush=True)

if __name__ == '__main__': main()
