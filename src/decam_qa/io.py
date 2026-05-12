"""I/O utilities: HDF5 embedding read/write and FITS image loading."""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from astropy.io import fits


def read_embeddings(h5_dir: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Read all worker HDF5 embedding files from a directory.

    Parameters
    ----------
    h5_dir : str
        Directory containing *_worker_embeds.h5 files.

    Returns
    -------
    data : List[np.ndarray]
        List of embedding arrays (one per entry).
    idx : List[str]
        Original dataframe indices as strings.
    label : List[str]
        Labels as strings.

    Raises
    ------
    FileNotFoundError
        If no *_worker_embeds.h5 files found.
    """
    target_dir = Path(h5_dir)
    h5_files = sorted(target_dir.glob("*_worker_embeds.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No *_worker_embeds.h5 files found in {target_dir}")
    import h5py
    data, idx, label = [], [], []
    for fpath in h5_files:
        with h5py.File(fpath, 'r') as h5f:
            dset = h5f["images"]
            for it in dset:
                data.append(np.array(dset[it]))
                names = it.split("_")
                idx.append(names[1])
                label.append(names[-1])
    return data, idx, label


def write_embeddings(
    embeds: List[np.ndarray],
    indices: List[int],
    labels: List[int],
    output_dir: str,
    worker_id: int = 0,
) -> List[Path]:
    """Write embedding vectors to an HDF5 file in standard worker format.

    Parameters
    ----------
    embeds : List[np.ndarray]
        Embedding arrays, each shape (1, D).
    indices : List[int]
        Original dataframe indices.
    labels : List[int]
        Integer labels.
    output_dir : str
        Output directory (created if missing).
    worker_id : int
        Worker GPU ID for naming.

    Returns
    -------
    List[Path]
        Paths to written HDF5 files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5path = output_dir / f"{worker_id}_worker_embeds.h5"
    import h5py
    with h5py.File(h5path, 'a') as h5f:
        if "images" not in h5f:
            h5f.create_group("images")
        for embed, idx, lab in zip(embeds, indices, labels):
            name = f"idx_{idx:d}_label_{lab:d}"
            if name not in h5f["images"]:
                h5f["images"].create_dataset(name, data=embed, dtype='float')
    return [h5path]


def read_fits_image(image_path: str, hdu: int = 0) -> np.ndarray:
    """Read a DECam FITS image with native byte order.

    Parameters
    ----------
    image_path : str
        Path to the FITS file.
    hdu : int
        HDU index to read.

    Returns
    -------
    np.ndarray
        Image data with native byte ordering.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"FITS file not found: {image_path}")
    with fits.open(image_path) as hdul:
        img_data = hdul[hdu].data
    return np.asarray(img_data).astype(img_data.dtype.newbyteorder('='))


def read_exposure_embeddings(h5_dir_or_path):
    """Read multi-scale exposure embeddings from HDF5 directory or file.

    Accepts either a directory (globs ``*.h5``, merges across all
    workers — matching read_embeddings behavior) or a single file path.

    Returns dict: {expnum: {"global": array, "locals": [arrays], "metadata": dict}}
    """
    import json
    import h5py

    target = Path(h5_dir_or_path)
    if target.is_dir():
        h5_files = sorted(target.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No *.h5 files found in {target}")
    else:
        h5_files = [target]

    result = {}
    for fpath in h5_files:
        with h5py.File(fpath, 'r') as f:
            if "exposures" not in f:
                continue
            for exp_name in f["exposures"]:
                expnum = int(exp_name.split("_")[1])
                grp = f["exposures"][exp_name]
                local_keys = sorted([k for k in grp.keys() if k.startswith("local_")])
                result[expnum] = {
                    "global": np.array(grp["global"]),
                    "locals": [np.array(grp[k]) for k in local_keys],
                    "metadata": json.loads(grp["metadata"][()]) if "metadata" in grp else {},
                }
    return result
