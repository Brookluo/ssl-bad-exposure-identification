# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# YL: This script is copied from
# https://github.com/Brookluo/vicreg-sage/tree/main/distributed.py
# commit a7eeeda4d7e9fa6bf1af9cdef99093918a5496c7

import torch
import os
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.backend is None:
        rank, n_ranks = 0, 1
    elif args.backend == 'mpi':
        rank, n_ranks = init_workers_mpi()
    elif args.backend == 'nccl':
        rank, n_ranks = init_workers_nccl_slurm()
        #rank, n_ranks = init_workers_nccl_file()
    elif args.backend == 'gloo':
        rank, n_ranks = init_workers_gloo_file()
    else:
        print('Not using distributed mode')
        return
    args.rank = rank
    args.world_size = n_ranks
    if "SLURM_LOCALID" in os.environ and torch.cuda.device_count() == 1:
        # this assumes we won't use single gpu with distributed
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
        args.gpu = int(os.environ["SLURM_LOCALID"])
    else:
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    print('{} | distributed init (rank {}): {}, gpu {}, world size {}'.format(
        args.backend, args.rank, args.dist_url, args.gpu, args.world_size), flush=True)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# from nersc_distributed.py
def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_workers_gloo_file():
    """Initialize workers with GLOO backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    dist.init_process_group(backend='gloo', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_file():
    """Initialize workers with NCCL backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    print('Setting up with sync file', sync_file)
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_slurm():
    """Initialize workers with NCCL backend and SLURM env variables.

    You must set the master address and port in your slurm script:
        export MASTER_ADDR=$(hostname)
        export MASTER_PORT=29500
        srun ...
    """
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank)
    return rank, n_ranks

def init_workers_mpi():
    """Initialize workers with MPI backend"""
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    return rank, n_ranks

def init_workers(backend=None):
    """Initialize workers for specified backend.

    Note that only a few modes are currently supported:
    - MPI backend
    - NCCL backend with ranks determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    - GLOO backend with rank determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    """
    if backend is None:
        rank, n_ranks = 0, 1
    elif backend == 'mpi':
        rank, n_ranks = init_workers_mpi()
    elif backend == 'nccl':
        rank, n_ranks = init_workers_nccl_slurm()
        #rank, n_ranks = init_workers_nccl_file()
    elif backend == 'gloo':
        rank, n_ranks = init_workers_gloo_file()
    return rank, n_ranks

def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass