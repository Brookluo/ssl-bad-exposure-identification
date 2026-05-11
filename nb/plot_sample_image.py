# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pytorch-2.0.1
#     language: python
#     name: pytorch-2.0.1
# ---

# %%
import sys
sys.path.append("../src")
import decam_info

sys.path.append("../../img-spec-ml/src/")
from plot_utils import plot_zscale_image

# %%
import matplotlib.pyplot as plt
from pathlib import Path

# %%
import h5py
import numpy as np

# %%
from astropy.table import Table
import fitsio

# %%
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
dr10_imdir = dr10_dir / "images"
dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")

# %%
from pprint import pprint


# %%
def plot_image(expnum, ccdname):
    imgrow = dr10_tab[(dr10_tab['expnum'] == expnum) & (dr10_tab['ccdname'] == ccdname)]
    print("ccd_cuts:", imgrow['ccd_cuts'])
    data = fitsio.read(dr10_imdir / imgrow['image_filename'][0], ext=imgrow['image_hdu'][0])
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_zscale_image(data, ax, 'gray')
    return imgrow


# %%
plot_image(412036, "S29")

# %%
plot_image(797871, "S15")

# %%
imgrow = dr10_tab[(dr10_tab['expnum'] == 797871) & (dr10_tab['ccdname'] == "S15")]
imgrow

# %%
decam_info.ccdnum2name[14]

# %%
