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
# astropy imports
from astropy.table import Table
from astropy.io import fits
import fitsio

from astropy.coordinates import SkyCoord
import astropy.units as u

# %%
from pathlib import Path
import os

# %%
import numpy as np

# %%
dr9_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
dr11_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr11')

# %% [markdown]
# # external sources
#
# One bad image in [LS pipe 725](https://github.com/legacysurvey/legacypipe/issues/725)
#
#
# Several bad exposures in [LS pipe 363](https://github.com/legacysurvey/legacypipe/issues/363):  
# CP/V4.0/CP20161012/c4d_161012_044657_ood_i_v1.fits.fz  
# CP/V4.0/CP20161012/c4d_161012_045325_ood_i_v1.fits.fz  
# CP/V4.0/CP20161012/c4d_161012_051251_ood_i_v1.fits.fz  
# CP/V4.0/CP20161012/c4d_161012_050622_ood_i_v1.fits.fz  
# CP/V4.0/CP20161012/c4d_161012_045952_ood_i_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_020326_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_023156_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_010503_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_024304_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_060107_ood_i_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_055438_ood_i_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_014812_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_050143_ood_i_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_011341_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_060734_ood_i_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_011008_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_010736_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_024103_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_011539_ood_Y_v1.fits.fz  
# CP/V4.0/CP20160915/c4d_160915_024915_ood_Y_v1.fits.fz  

# %%
os.listdir(dr11_dir)

# %%
dr11a = Table.read(dr11_dir / "survey-ccds-decam-dr11-a.kd.fits")
dr11b = Table.read(dr11_dir / "survey-ccds-decam-dr11-b.kd.fits")
delve = Table.read(dr11_dir / "survey-ccds-decam-delve.kd.fits")
dr10 = Table.read(dr11_dir / "survey-ccds-decam-dr10.kd.fits")

# %%
from astropy.table import hstack, vstack

# %%
# Check exposure overlapping to prevent double counting
# from functools import reduce
# import numpy as np

# reduce(np.intersect1d, [dr11a['expnum'], dr11b['expnum'], delve['expnum'], dr10['expnum']])

# %%
dr11_tab = vstack([dr10, delve, dr11a, dr11b])

# %%
ind = np.lexsort((dr11_tab['image_hdu'], dr11_tab['expnum']))

# %%
dr11_sorted = dr11_tab[ind]

# %%
for name, tab in zip(["dr10", "delve", "dr11a", "dr11b"], [dr10, delve, dr11a, dr11b]):
    print(name, len(tab[tab['expnum'] == 0]))

# %%
# dr11_sorted.write("/pscratch/sd/b/brookluo/decam-exposure/dr11/dr11-suvey-ccds.fits")

# %% [markdown]
# # Sample images for inference
#
# - select 20 images per exposure
# - inference and then do classification
# - do a ridge line plot on the top 10 bad ccds

# %%
num_ccd_exp = 20

# %%
dr11_imdir = dr11_dir / "images"

# %%
(dr11_imdir / dr11_sorted[dr11_sorted['expnum'] == 0][0]["image_filename"]).exists()

# %%
dr11_sorted_good = dr11_sorted[dr11_sorted['expnum'] > 0]

# %%
uni_hdu, hdu_counts = np.unique(dr11_sorted_good['image_hdu'], return_counts=True)
uni_exp, exp_counts = np.unique(dr11_sorted_good['expnum'], return_counts=True)

# %%
np.unique(exp_counts, return_counts=True)

# %%
reproc_exp = uni_exp[exp_counts > 110]

# %%
dr11_sorted_good['plver'].data

# %%
np.logical_or(dr11_sorted_good['plver'] == "V4.8.2a", dr11_sorted_good['plver'] == "V4.8.2")

# %%
any(dr11_sorted_good['plver'] == "V4.8.2")

# %%
drop_rows = np.logical_or.reduce([
    (dr11_sorted_good['expnum'] == exp)
             & np.logical_or.reduce(
                 (dr11_sorted_good['plver'] == "V4.8.2a",
                  dr11_sorted_good['plver'] == "V4.8.2",
                dr11_sorted_good['plver'] == "V5.4", # exp 1070328
                 )
             )
             for exp in reproc_exp])

# %%
dr11_sorted_good_drop = dr11_sorted_good[~drop_rows]
uni_exp, exp_counts = np.unique(dr11_sorted_good_drop['expnum'], return_counts=True)

# %%
np.unique(exp_counts, return_counts=True)

# %%
# dr11_sorted_good_drop.write("/pscratch/sd/b/brookluo/decam-exposure/dr11/good-exp-dr11-suvey-ccds.fits")

# %%
dr11_sorted_good_drop = Table.read("/pscratch/sd/b/brookluo/decam-exposure/dr11/good-exp-dr11-suvey-ccds.fits")

# %% [markdown]
# # Generate sample for exposures with 60 or 61 CCDs
#
# All other exposures are ignored for this step, as they have been selected through CP.
# We can include those exposures in future analysis

# %%
import pandas as pd

# %%
data_dir = Path("../data")
bad_exp = pd.read_csv(data_dir / "decam_dr10_bad_exp_all.csv")

# %%
# reject exposures identified in the past
good_dr11_expnum = np.setdiff1d(uni_exp, bad_exp['expnum'])

# %%
good_exp_counts = exp_counts[[np.where(exp == uni_exp)[0][0] for exp in good_dr11_expnum]]

# %%
len(good_dr11_expnum)

# %%
miss1 = good_exp_counts == 61
miss2 = good_exp_counts == 60

# %%
selected_dr11_exp = good_dr11_expnum[miss1 | miss2]
selected_dr11_exp_counts = good_exp_counts[miss1 | miss2]

# %%
selected_miss1 = selected_dr11_exp_counts == 61
selected_miss2 = selected_dr11_exp_counts == 60

# %%
len(selected_dr11_exp)

# %%
tab_idx = np.full((len(selected_dr11_exp), 20), fill_value=-1)

# %%
rng = np.random.default_rng(36)
# unique index in each row
tab_idx[selected_miss1] = np.vstack([rng.choice(np.arange(1, 61+1), (num_ccd_exp), replace=False)
                           for row in range(sum(selected_miss1))])
tab_idx[selected_miss2] = np.vstack([rng.choice(np.arange(1, 60+1), (num_ccd_exp), replace=False)
                           for row in range(sum(selected_miss2))])
# -1 to convert the number of hdu to table index
tab_idx -= 1

# %%
uni_exp, uni_idx, uni_counts = np.unique(dr11_sorted_good_drop['expnum'], return_index=True, return_counts=True)

# %%
import sys
sys.path.append("../src")
import decam_info

# %%
from tqdm import tqdm

# %%
filter_num = []
fnames = []
missing_two = []
uni_tab = dr11_sorted_good_drop[uni_idx]
ccdname_list = []
samp_good_hdu = []
samp_good_exp = []
for i, exp in enumerate(selected_dr11_exp):
    this_idx = np.where(uni_tab['expnum'] == exp)[0][0]
    row = uni_tab[this_idx]
    if row['filter'] not in decam_info.filter_dict.keys():
        continue
    samp_good_exp.append(exp)
    filter_num.append(decam_info.filter_dict[row['filter']])
    fnames.append(row['image_filename'])
    all_idx = np.arange(0, uni_counts[this_idx]) + uni_idx[this_idx]
    this_exp_tab = dr11_sorted_good_drop[all_idx][['image_hdu', 'ccdname']]
    # Table is already sorted in image hdu / ccdnum order
    # sort_idx = this_exp_tab.argsort("image_hdu")
    # this_exp_tab = this_exp_tab[sort_idx]
    pick_exp_idx = tab_idx[i] #np.sort(rng.choice(len(this_exp_tab), num_ccd_exp, replace=False))
    # if exp == 978264:
    #     tab_idx = [np.where(x == this_exp_tab['image_hdu'])[0][0] for x in ori_samp_good_hdu[i]]
    samp_good_hdu.append(this_exp_tab[pick_exp_idx]['image_hdu'])
    # ccdname_list.append(this_exp_tab['ccdname'][samp_good_hdu[i]-1])
    ccdname_list.append(this_exp_tab[pick_exp_idx]['ccdname'])


# %%
def flatten(xss):
    return [x for xs in xss for x in xs]

expand_repeat = lambda arr, n: flatten([[it]*n for it in arr])

# %%
ccd_li = [decam_info.decam_fp_ccdname2num[ccdname] for ccdname in flatten(ccdname_list)]

# %%
n_samp = len(samp_good_exp)

# %%
df_good = pd.DataFrame(zip(*[
    expand_repeat(fnames, num_ccd_exp), # fname
    expand_repeat(samp_good_exp, num_ccd_exp), # expnum
    ccd_li,
    flatten(samp_good_hdu),
    expand_repeat(filter_num, num_ccd_exp),
    [0] * (num_ccd_exp * n_samp),
    [0] * (num_ccd_exp * n_samp)
]), columns=['image_filename'] + list(bad_exp.columns))

# %%
df_good = df_good.sort_values(by=['expnum', 'ccdnum'])

# %%
df_good.to_csv("/pscratch/sd/b/brookluo/decam-exposure/dr11/good_exp_20ccd_decam_dr11.csv", index=False)

# %%
np.unique(df_good['filter'], return_counts=True)

# %%
num_nodes = 16

# %%
split_size = len(df_good) // num_nodes

# %%
split_size / 159618

# %%
for i in range(num_nodes):
    df_good.iloc[i*split_size:(i+1)*split_size].to_csv(f"/pscratch/sd/b/brookluo/decam-exposure/dr11/proc-data/node{i}_dr11_sample.csv",
                                                  index=False)

# %%
df_good.iloc[(num_nodes-1)*split_size:].to_csv(f"/pscratch/sd/b/brookluo/decam-exposure/dr11/proc-data/node{num_nodes-1}_dr11_sample.csv",
                                                  index=False)

# %%
# # check the difference of expousres in dr11 used in inferece
# root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11")

# data_dir = root_dir / "data"

# dr11_tab = Table.read(data_dir / "dr11-suvey-ccds.fits")

# dr11_infer = pd.read_csv(data_dir / "good_exp_20ccd_decam_dr11.csv")

# dr11_tab = dr11_tab[dr11_tab["expnum"] > 0]

# ori_exp = np.unique(dr11_tab['expnum']).data

# infer_exp = np.unique(dr11_infer['expnum']).data

# used_infer = np.zeros(len(ori_exp), dtype=bool)

# _, ori_idx, use_idx = np.intersect1d(ori_exp, infer_exp, return_indices=True)

# used_infer[ori_idx] = True

# df = pd.DataFrame({"expnum": ori_exp, "used_in_inference": used_infer})

# df.sort_values(by='expnum').to_csv(data_dir / "dr11-all-exposures-in-inference.csv",
#                                   index=False)
