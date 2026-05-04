from __future__ import annotations

import subprocess
from pathlib import Path
from argparse import Namespace, ArgumentParser
import h5py
import sys
import pandas as pd
import torch
from .parallel_eval import split_dset
from .config import load_config, PipelineConfig


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-dir", "--output-dir", type=str, required=True,
                            help='full directory path')
    parser.add_argument("--dset-path", type=str, required=True,
                        help='Path to the dataset file')
    parser.add_argument("--dr", choices=['dr10', 'dr11'], required=True,
                       help='which data release to use')
    parser.add_argument("--image-dir", type=str, default=None,
                        help='Path to DECam image directory (overrides dr-based default)')
    parser.add_argument("--cont", action="store_true",
                        help='Whether to continue last inferencing')
    parser.add_argument("--config", type=str, default=None,
                        help='Path to YAML config file')
    return parser


def main(args: Namespace) -> None:
    pipe_config, _ = load_config(args.config)
    num_gpu = torch.cuda.device_count()
    output_dir = Path(args.output_dir)
    worker_dset_dir = output_dir / "tmp"
    embeds_dir = output_dir / "embeds_out"
    if args.image_dir:
        imdir = args.image_dir
    elif args.dr == 'dr10':
        imdir = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/images"
    elif args.dr == "dr11":
        imdir = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/images'
    else:
        raise ValueError(f"{args.dr} not available")
    keep_index = False
    if not embeds_dir.exists():
        embeds_dir.mkdir(parents=True)
        keep_index = True
    elif args.cont:
        idx = []
        for fpath in embeds_dir.glob("*.h5"):
            with h5py.File(fpath, 'r') as h5f:
                idx += [it.split("_")[1] for it in h5f["images"]]
        df = pd.read_csv(args.dset_path)
        print(f"Resuming from {len(idx)}/{len(df)}")
        df = df.drop(idx)
        df.index.name = "original_df_idx"
        df.to_csv(worker_dset_dir / "remaining_sample.csv", index=True)
        args.dset_path = worker_dset_dir / "remaining_sample.csv"
    split_dset(args.dset_path, worker_dset_dir, num_gpu, keep_index)
    project_root = Path(__file__).resolve().parent.parent
    parallel_bin = project_root / "bin" / "parallel_eval"
    procs = []
    for i in range(num_gpu):
        print("Running on GPU", i)
        proc = subprocess.Popen(
            [
                sys.executable, str(parallel_bin),
                f'--dset-path={str(worker_dset_dir)}/{i}_worker_sample.csv',
                f'--exp-dir={str(output_dir)}',
                f'--imdir={imdir}',
                '--model-size=base',
                '--batch-size=1',
                '--crop-size', '2352', '1176',
                '--num-workers=2',
                f'--gpu-idx={i}'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
    errors = []
    for i, proc in enumerate(procs):
        proc.wait()
        stdout, stderr = proc.communicate()
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))
        if proc.returncode != 0:
            errors.append(f"GPU {i} failed with code {proc.returncode}")
    if errors:
        raise RuntimeError("; ".join(errors))
    print("Done.")

        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    