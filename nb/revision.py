# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pytorch-2.6.0
#     language: python
#     name: pytorch-2.6.0
# ---

# %%
import sys
sys.path.append("../src")
sys.path.append("/global/homes/b/brookluo/.local/perlmutter/pytorch2.6.0/lib/python3.12/site-packages")
import decam_info
from decam_dataset import DECamImageDataset

sys.path.append("../../img-spec-ml/src/")
from plot_utils import plot_zscale_image
from inference import read_embeds

import matplotlib.pyplot as plt
from pathlib import Path

import h5py
import numpy as np

# %%
from astropy.table import Table

# %%
# root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dino_v2")
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure")
exp_name = "revision"
exp_dir = root_dir / exp_name
eval_dir = exp_dir / "eval"
# data_dir = root_dir / "data"
ckpt_dir = exp_dir / "checkpoint"
# test_dir = data_dir / "test"

if not eval_dir.exists():
    eval_dir.mkdir()

embeds_dir = eval_dir / "embeds_out"
if not embeds_dir.exists():
    embeds_dir.mkdir()

# model = "base_resize"
# model_dir = embeds_dir / model
train_dir = exp_dir / "train"
test_dir = exp_dir / "test"

plot_dir = eval_dir / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
dr8_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr8')
dr9_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')

# %%
# meta_tab = Table.read('/global/cfs/cdirs/desi/users/rongpu/useful/survey-ccds-decam-dr9-trim.fits')
dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")

# %%
np.sum(dr10_tab['ccd_cuts'] > 0) / len(dr10_tab)

# %%
bad_ccd = dr10_tab['ccd_cuts'] > 0

# %%
np.unique( dr10_tab[bad_ccd]["expnum"]).shape[0] / np.unique(dr10_tab['expnum']).shape[0]

# %%
