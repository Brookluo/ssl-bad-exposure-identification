"""Orchestration for distributed embedding inference across GPUs."""
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Set
import h5py
import pandas as pd
import torch


def _split_dset(dset_path: str, dst_dir: Path, num_parts: int, keep_index: bool = True) -> None:
    """Split a CSV dataset into per-worker files."""
    dst_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(dset_path)
    idx_length = len(df) // num_parts
    for i in range(num_parts):
        if i == num_parts - 1:
            tmp_df = df.iloc[i * idx_length:]
        else:
            tmp_df = df.iloc[i * idx_length:(i + 1) * idx_length]
        if keep_index:
            tmp_df.insert(0, "original_df_idx", tmp_df.index)
        tmp_df.to_csv(dst_dir / f"{i}_worker_sample.csv", index=False)


def _get_completed_indices(embeds_dir: str) -> Set[int]:
    """Return set of original DataFrame indices already processed."""
    idx: Set[int] = set()
    embeds_path = Path(embeds_dir)
    if not embeds_path.exists():
        return idx
    for fpath in embeds_path.glob("*.h5"):
        with h5py.File(fpath, 'r') as h5f:
            if "images" not in h5f:
                continue
            for name in h5f["images"]:
                idx.add(int(name.split("_")[1]))
    return idx


_IMAGE_ROOTS = {
    "dr10": "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr10/images",
    "dr11": "/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/images",
}


class ParallelEvaluator:
    """Orchestrates distributed embedding inference across GPUs."""

    def __init__(
        self,
        config: Dict[str, Any],
        dataset_path: str,
        dr: str,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.dataset_path = dataset_path
        self.dr = dr
        self.resume = resume
        self.scratch_dir = Path(config["scratch_dir"])
        self.worker_dset_dir = self.scratch_dir / "tmp"
        self.embeds_dir = self.scratch_dir / "embeds_out"
        self.num_gpu = torch.cuda.device_count()

    def run(self) -> List[Path]:
        """Run embedding inference across all GPUs.

        Returns
        -------
        List[Path]
            Paths to the generated HDF5 embedding files.
        """
        if self.dr not in _IMAGE_ROOTS:
            raise ValueError(f"Unknown data release: {self.dr}")
        imdir = _IMAGE_ROOTS[self.dr]

        keep_index = True
        dset_path = self.dataset_path

        if self.resume:
            completed = _get_completed_indices(str(self.embeds_dir))
            if completed:
                df = pd.read_csv(self.dataset_path)
                df = df.drop(list(completed), errors="ignore")
                df.index.name = "original_df_idx"
                remaining_path = self.worker_dset_dir / "remaining_sample.csv"
                df.to_csv(remaining_path, index=True)
                dset_path = str(remaining_path)
                keep_index = False

        self.embeds_dir.mkdir(parents=True, exist_ok=True)
        self.worker_dset_dir.mkdir(parents=True, exist_ok=True)

        _split_dset(dset_path, self.worker_dset_dir, self.num_gpu, keep_index)

        num_data_workers = self.config.get("data", {}).get("num_workers", 10)
        procs = []
        log_files = []

        try:
            for i in range(self.num_gpu):
                log_f = open(self.worker_dset_dir / f"{i}_worker.log", "wb")
                log_files.append(log_f)
                proc = subprocess.Popen(
                    [
                        "python", "-m", "decam_qa.worker",
                        f"--dset-path={self.worker_dset_dir}/{i}_worker_sample.csv",
                        f"--exp-dir={self.scratch_dir}",
                        f"--imdir={imdir}",
                        "--model-size=base",
                        "--batch-size=1",
                        "--crop-size", "2352", "1176",
                        f"--num-workers={num_data_workers}",
                        f"--gpu-idx={i}",
                    ],
                    stdout=log_f,
                    stderr=log_f,
                )
                procs.append(proc)
        except Exception:
            for f in log_files:
                f.close()
            raise

        failures = []
        for proc in procs:
            proc.wait()
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                msg = stderr.decode("utf-8") if stderr else f"exit code {proc.returncode}"
                failures.append(msg)

        for f in log_files:
            f.close()

        if failures:
            raise RuntimeError("Worker failures:\n" + "\n".join(failures))

        return sorted(self.embeds_dir.glob("*.h5"))
