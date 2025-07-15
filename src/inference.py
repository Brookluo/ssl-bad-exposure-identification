from pathlib import Path
import h5py
import numpy as np

def read_embeds(target_dir: str, num=4):
    target_dir = Path(target_dir)
    data = []
    idx = []
    label = []
    for i in range(num):
        with h5py.File(target_dir / f"{i}_worker_embeds.h5", 'r') as h5f:
            dset = h5f["images"]
            for it in dset:
                data.append(np.array(dset[it]))
                names = it.split("_")
                idx.append(names[1])
                label.append(names[-1])
    return data, idx, label