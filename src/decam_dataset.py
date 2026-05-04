from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pandas as pd
from . import decam_info
from .read_image import read_image
from typing import Union
import logging

logger = logging.getLogger(__name__)


class DECamImageDataset(Dataset):
    def __init__(self, dataset_path: "Path | str", image_dir: "Path | str", transform=None, seed=0, reason_dict=None):
        self.df_data = pd.read_csv(dataset_path)
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        if reason_dict is None:
            self.reason_dict = decam_info.reason_num_dict
        else:
            self.reason_dict = reason_dict
        self.reason_list = list(self.reason_dict.keys())
        self.access_idx = np.arange(len(self.df_data))
        # use 2D array + permutation to speed up shuffling
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
        with fits.open(self.image_dir / data_row['image_filename'], 'r') as hdul:
            img_data = hdul[data_row['image_hdu']].data
        # single channel image
        img_data = np.expand_dims(img_data, axis=0)
        # first translate number to str with original dictionary
        reason_orig_str = [
            r_str
            for r_str in decam_info.decode_reason(
                            data_row['reasons'],
                            return_num=False)
        ]
        # then translate the string into number with the updated
        # training dictionary
        reasons = np.array([
            self.reason_dict[r_str]
            for r_str in reason_orig_str
        ])
        
        # sources = np.array([decam_info.reason_source[s] for s in decam_info.decode_vi_source(data_row['vi_source'])], dtype=int)
        if self.transform is not None:
            img_data = self.transform(img_data)
        # arr_reasons = np.zeros(len(self.reason_list), dtype=int)
        # arr_reasons[reasons] = 1
        # uniformly choose reasons if multiple exists
        # label follows the class dist.
        # 0 means good images, num > 0 indicate the category
        # out_reason = np.random.choice(reasons, size=1)[0] + 1 if len(reasons) > 0 else 0
        # to ensure deterministic, use one number, lower number has higher
        # precedence
        out_reason = reasons[0] + 1 if len(reasons) > 0 else 0
        
        # binary clas
        # out_reason = 1 if len(reasons) > 0 else 0
        # arr_sources = np.zeros(len(decam_info.reason_source.keys()), dtype=int)
        # vi_source identifier starts at 1
        # arr_sources[sources-1] = 1
        return img_data, out_reason
        
    def shuffle(self, seed):
        rng = np.random.default_rng(self.seed + seed)
        new_mat = rng.permuted(rng.permutation(self.idx_mat), axis=1)
        new_acc_idx = new_mat.ravel()
        new_acc_idx = new_acc_idx[~np.isnan(new_acc_idx)].astype(int)
        self.access_idx = new_acc_idx


class DECamExposureDataset(Dataset):
    """
    A dataset class for DECam exposure data.

    Args:
        bad_exp_table_path (Path): Path to the bad exposure table file.
        imgdir (Path): Path to the directory containing the image files.
        padding (bool): if true, adding a row/column of empty pixel to each side to expand
        the image to (4096, 2048). Else, trimming the image -> removing a row from bottom and top

    Attributes:
        expid (ndarray): Array of exposure IDs.
        fnames (ndarray): Array of image filenames.
        imdir (Path): Path to the directory containing the image files.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the half CCD image data and CCD names for the given index.
    
    Raises:
        ValueError: If the bad exposure table format is unknown.
    """

    def __init__(self, bad_exp_table_path: Union[Path, str], imdir: Path, padding=False):
        if isinstance(bad_exp_table_path, str):
            bad_exp_table_path = Path(bad_exp_table_path)
        if bad_exp_table_path.suffix == ".csv":
            bad_exp_tab = pd.read_csv(bad_exp_table_path)
        elif bad_exp_table_path.suffix == ".fits":
            bad_exp_tab = Table.read(bad_exp_table_path)
        else:
            raise ValueError("Unknown bad exposure table format.")
        self.expid = np.array(bad_exp_tab["expnum"])
        self.fnames = np.array(bad_exp_tab["image_filename"])
        self.imdir = imdir
        self.padding = padding

    def __len__(self):
        return len(self.expid)

    def __getitem__(self, idx: int):
        logger.info("Reading exposure: %s @ %s" % (self.expid[idx], self.fnames[idx]))
        half_ccd_image_data, ccdnames = read_image(self.imdir / self.fnames[idx], self.padding)
        return half_ccd_image_data, ccdnames
