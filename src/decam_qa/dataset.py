"""PyTorch Dataset for DECam single-CCD images with bad-exposure reason labelling."""
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.io import fits
from decam_qa.info import reason_num_dict, decode_reason
from decam_qa.focalplane import build_focalplane_stamp
from decam_qa.selection import compute_anomaly_scores, select_top_k_ccds


class DECamImageDataset(Dataset):
    """PyTorch Dataset yielding single-CCD DECam images and their bad-exposure label.

    Each item is ``(image, label)`` where ``image`` is a float32 (1, 2048, 4096)
    array in native byte order and ``label`` is 0 for good or >= 1 for a bad-
    exposure category (bit index + 1 from the reason bitmask).

    Parameters
    ----------
    dataset_path : pathlib.Path or str
        CSV with columns ``image_filename``, ``expnum``, ``ccdnum``,
        ``image_hdu``, ``filter``, ``reasons``, ``vi_source``.
    image_dir : pathlib.Path or str
        Directory containing the FITS images.
    transform : callable, optional
        Transform applied to the image array before returning.
    seed : int, optional
        Base random seed used by ``shuffle``.
    reason_dict : dict, optional
        Mapping from reason string to integer index. Defaults to
        ``decam_qa.info.reason_num_dict``.
    """

    def __init__(self, dataset_path, image_dir, transform=None, seed=0, reason_dict=None):
        self.df_data = pd.read_csv(dataset_path)
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        if reason_dict is None:
            self.reason_dict = reason_num_dict
        else:
            self.reason_dict = reason_dict
        self.reason_list = list(self.reason_dict.keys())
        self.access_idx = np.arange(len(self.df_data))
        unique_exp, recon_idx = np.unique(self.df_data['expnum'], return_inverse=True)
        idx_arr = [np.where(recon_idx == i)[0] for i in range(len(unique_exp))]
        longest = 0
        for arr in idx_arr:
            if len(arr) > longest:
                longest = len(arr)
        matrix = np.full((len(idx_arr), longest), fill_value=np.nan)
        for i, arr in enumerate(idx_arr):
            matrix[i, :len(arr)] = arr
        self.idx_mat = matrix
        self.seed = seed

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        idx = self.access_idx[idx]
        data_row = self.df_data.iloc[idx]
        with fits.open(self.image_dir / data_row['image_filename']) as hdul:
            img_data = hdul[data_row['image_hdu']].data
        img_data = np.expand_dims(img_data, axis=0)
        reason_orig_str = [
            r_str
            for r_str in decode_reason(
                data_row['reasons'],
                return_num=False)
        ]
        reasons = np.array([
            self.reason_dict[r_str]
            for r_str in reason_orig_str
        ])
        out_reason = reasons[0] + 1 if len(reasons) > 0 else 0
        if self.transform is not None:
            img_data = self.transform(img_data)
        return img_data.astype(img_data.dtype.newbyteorder('=')), out_reason

    def shuffle(self, seed):
        """Shuffle the access order in an exposure-grouped fashion.

        Permutes the order of exposures and the order of CCDs within each
        exposure independently, producing a deterministic but randomized
        iteration sequence.

        Parameters
        ----------
        seed : int
            Added to the base ``self.seed`` to form the RNG key.
        """
        rng = np.random.default_rng(self.seed + seed)
        new_mat = rng.permuted(rng.permutation(self.idx_mat), axis=1)
        new_acc_idx = new_mat.ravel()
        new_acc_idx = new_acc_idx[~np.isnan(new_acc_idx)].astype(int)
        self.access_idx = new_acc_idx


class DECamExposureDataset:
    """Dataset yielding exposure-grouped multi-scale DECam data.

    Groups CCD rows by ``expnum``. Each item is a dict with the global focal-plane
    stamp, top-K anomalous CCD images, and metadata.

    Parameters
    ----------
    dataset_path : pathlib.Path or str
        CSV with columns ``image_filename``, ``expnum``, ``ccdnum``,
        ``image_hdu``, ``filter``, ``reasons``, ``vi_source``.
    image_dir : pathlib.Path or str
        Directory containing the FITS images.
    binsize : int
        Downsampling factor for the focal-plane stamp (default 120).
    top_k : int
        Max number of anomalous CCDs to select (default 8).
    include_center_fallback : bool
        Always include one central CCD in selection (default True).
    include_edge_fallback : bool
        Always include one edge CCD in selection (default True).
    transform : callable, optional
        Transform applied to the returned dict before returning.
    """

    def __init__(self, dataset_path, image_dir, binsize=120,
                 top_k=8, include_center_fallback=True,
                 include_edge_fallback=True, transform=None):
        self.df_data = pd.read_csv(dataset_path)
        self.image_dir = Path(image_dir)
        self.binsize = binsize
        self.top_k = top_k
        self.include_center_fallback = include_center_fallback
        self.include_edge_fallback = include_edge_fallback
        self.transform = transform
        self.exposure_groups = self.df_data.groupby("expnum")
        self.expnums = list(self.exposure_groups.groups.keys())

    def __len__(self):
        return len(self.expnums)

    def __getitem__(self, idx):
        expnum = self.expnums[idx]
        rows = self.exposure_groups.get_group(expnum)

        image_filename = rows["image_filename"].iloc[0]
        filt = rows["filter"].iloc[0]
        reason_bitmask = int(np.bitwise_or.reduce(rows["reasons"].values))

        with fits.open(self.image_dir / Path(image_filename)) as hdul:
            ccd_images = []
            ccd_rows = []
            num_readable_ccds = 0
            for _, row in rows.iterrows():
                try:
                    img = np.asarray(hdul[row["image_hdu"]].data, dtype=np.float32)
                    img = np.expand_dims(img, axis=0)
                    ccd_images.append(img)
                    ccd_rows.append(row.to_dict())
                    num_readable_ccds += 1
                except (IndexError, KeyError, OSError):
                    continue

            if num_readable_ccds == 0:
                raise RuntimeError(f"No readable CCDs for exposure {expnum}")

            global_stamp = build_focalplane_stamp(
                hdul, None, binsize=self.binsize,
            )

            scores = compute_anomaly_scores(ccd_images, ccd_rows)
            selected = select_top_k_ccds(
                scores, ccd_rows, k=self.top_k,
                include_center_fallback=self.include_center_fallback,
                include_edge_fallback=self.include_edge_fallback,
            )

            ccd_images_by_hdu = {
                row["image_hdu"]: img
                for row, img in zip(ccd_rows, ccd_images)
            }
            selected_images = [
                ccd_images_by_hdu[sel_entry["image_hdu"]]
                for sel_entry in selected
            ]

        result = {
            "expnum": expnum,
            "image_filename": image_filename,
            "filter": filt,
            "reason_bitmask": reason_bitmask,
            "global_stamp": global_stamp,
            "selected_ccds": selected,
            "selected_images": selected_images,
            "num_readable_ccds": num_readable_ccds,
        }

        if self.transform is not None:
            result = self.transform(result)

        return result
