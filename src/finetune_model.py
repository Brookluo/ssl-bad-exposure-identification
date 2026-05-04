# This file is incomplete — a skeleton for fine-tuning DINOv2 on DECam data.
# Imports below are unused until the training loop is implemented.

from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import h5py
from astropy.table import Table
import fitsio
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms

from . import decam_info


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=Path, help="Path to config file")
    parser.add_argument("--dset-path", type=Path, default="/path/to/imagenet",
                            help='Path to the *.csv file')
    parser.add_argument('--exp-dir', type=Path, 
                        help='Experiment directory')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # dist params
    parser.add_argument('--backend', choices=["mpi", "nccl", "gloo"], default="nccl",
                        help='backend to use')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    