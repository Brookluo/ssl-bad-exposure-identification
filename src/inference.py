from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


def read_embeds(target_dir: str | Path, num: int = 4
                ) -> tuple[list[np.ndarray], list[str], list[str]]:
    target_dir = Path(target_dir)
    data: list[np.ndarray] = []
    idx: list[str] = []
    label: list[str] = []
    for i in range(num):
        with h5py.File(target_dir / f"{i}_worker_embeds.h5", 'r') as h5f:
            dset = h5f["images"]
            for it in dset:
                data.append(np.array(dset[it]))
                names = it.split("_")
                idx.append(names[1])
                label.append(names[-1])
    return data, idx, label
