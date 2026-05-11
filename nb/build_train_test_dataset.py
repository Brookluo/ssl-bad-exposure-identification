# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# import torch
# from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# %%
import pandas as pd
import os

# %%
# astropy imports
from astropy.table import Table
from astropy.io import fits
# import fitsio

from astropy.coordinates import SkyCoord
import astropy.units as u

# %%
import sys
sys.path.append("../src")
import decam_info

# %%
data_dir = Path("../data")

# %%
bad_exp = pd.read_csv(data_dir / "decam_dr10_bad_exp_all.csv")
good_expnum = np.loadtxt(data_dir / "decam_dr10_good_exp.csv", dtype=int)

# %%
np.unique(bad_exp.expnum)

# %%
# to change readout from 15 to 9 merging with Noise category
# import importlib
# importlib.reload(decam_info)

# for i, row in bad_exp.iterrows():
#     rea = decam_info.decode_reason(row['reasons'])
#     row['reasons'] = sum([2**decam_info.reason_num_dict[r] for r in rea])


# bad_exp.to_csv(data_dir / "decam_dr10_bad_exp_all.csv", index=False)

# %%
bad_exp.head()

# %%
dr8_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr8')
dr9_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')

# %%
dr9_imdir = dr9_dir / "images"
dr10_imdir = dr10_dir / "images"

# %%
# meta_tab = Table.read('/global/cfs/cdirs/desi/users/rongpu/useful/survey-ccds-decam-dr9-trim.fits')
# dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")
ccd_meta = Table.read(dr10_dir / "ccds-annotated-decam-dr10.fits.gz")
dr10_tab = ccd_meta

# %%
# bad exps
# 349  769209
# 350  769213
# 351  769215
# 352  769222
# 353  768938
# 354  768948
# 355  768952
# 356  768960
# 357  768976
# 358  768980
# 359  768981
# 360  768987
# 361  768988
# 362  768990
# 363  769010
# 364  769013
# 365  769015
# 366  769020
# 367  769038
# 368  769045
# 369  769046

# %%
# all(ccd_meta['ccdname'] == dr10_tab['ccdname']) # True

# %%
# Galactic latitude cut
# img_coord = SkyCoord(ra=dr10_tab['ra'], dec=dr10_tab['dec'], unit="deg")
# gal_cut = np.abs(img_coord.galactic.b) > 30 * u.deg
img_coord = SkyCoord(ra=ccd_meta['ra'], dec=ccd_meta['dec'], unit="deg")
gal_cut = np.abs(img_coord.galactic.b) > 30 * u.deg

# %%
np.quantile(ccd_meta['ebv'][gal_cut], q=0.9), np.mean(ccd_meta['ebv'][gal_cut])

# %%
# E(B-V) extinction cut
ebv_cut = ccd_meta['ebv'] < 0.04

# %%
ccd_cut = ebv_cut

# %%
bad_exp

# %%
# find exposures with the same reason
for rea in decam_info.reason_li:
    cond = ((bad_exp['reasons'] & 2**decam_info.reason_num_dict[rea]) & (bad_exp['vi_source'] & 2**decam_info.reason_source_dict['Rongpu']))
    cond = np.array(cond, dtype=bool)
    print(rea, np.sum(cond > 0), len(np.unique(bad_exp[cond]['expnum'])))
# np.where(bad_exp['reasons'] & 2**reason_num[reason_li[1]])

# %%
data = np.load("../data/decam_dr10_RZ_bad_exp.npy")

# %%
np.unique(data[:, 0]).shape

# %%
np.unique(bad_exp['vi_source'] & 2**decam_info.reason_source_dict['Rongpu'])

# %%
set().union(*[set(decam_info.decode_reason(i)) for i in data[:, -2]])

# %% [markdown]
# # Include cuts implemented in DR10
#
# https://www.legacysurvey.org/dr10/bitmasks/#ccd-cuts
#
# This should also increase the number of bad exposures.

# %%
res = [np.where(dr10_tab['ccd_cuts'] & 2**i)[0] for i in range(19)]

all_cut_idx = np.hstack(res)
pick_idx = np.ones(len(dr10_tab), dtype=bool)
pick_idx[all_cut_idx] = False

dr10_after_cut_expnum = dr10_tab["expnum"][pick_idx]
# ccd_cut &= pick_idx

# %%
from functools import reduce

# # ! now use dr10 gal cut

# ooi and bad exposures with file names
ooi_oki_exp = pd.read_csv(data_dir / "decam_dr10_ooi_oki_images_exist.csv")
# oki has much more exposures than ooi, so use it instead
expnum_bad_oki = reduce(np.intersect1d, [ooi_oki_exp.query("oki_exist")['expnum'], bad_exp['expnum'], ccd_meta['expnum'][ccd_cut]])
expnum_bad_ooi = reduce(np.intersect1d, [ooi_oki_exp.query("ooi_exist")['expnum'], bad_exp['expnum'], ccd_meta['expnum'][ccd_cut]])
expnum_bad_oki_idx = np.where(np.isin(bad_exp['expnum'], expnum_bad_oki))[0]
expnum_bad_ooi_idx = np.where(np.isin(bad_exp['expnum'], expnum_bad_ooi))[0]
print("Bad exp CCD oki exists:", expnum_bad_oki_idx.shape,
      "\nBad exp CCD ooi exists:", expnum_bad_ooi_idx.shape)

# %%
expnum_good_oki = reduce(np.intersect1d, [ooi_oki_exp.query("oki_exist")['expnum'], good_expnum, ccd_meta['expnum'][ccd_cut]])
expnum_good_ooi = reduce(np.intersect1d, [ooi_oki_exp.query("ooi_exist")['expnum'], good_expnum, ccd_meta['expnum'][ccd_cut]])
# exp * 60 for total number of single CCD images
print("Good exp oki exists:", expnum_good_oki.shape[0], ", total CCD images:", expnum_good_oki.shape[0]*60)
print("Good exp ooi exists:", expnum_good_ooi.shape[0], ", total CCD images:", expnum_good_ooi.shape[0]*60)

# %%
# All exposures in DR10 (images *= 60)
np.sum(ooi_oki_exp['oki_exist']), np.sum(ooi_oki_exp['ooi_exist'])

# %%
bad_exp_with_fnames = pd.read_csv(data_dir / "decam_dr10_bad_exp_all_with_fname.csv")

# save the good images with ooi and oki into the file
bad_exp_with_fnames.iloc[expnum_bad_ooi_idx].to_csv(data_dir / "decam_dr10_bad_exp_ooi.csv", index=False)

# %%
# try to train using all reasons from one exposure with other ones.
# try to pick the same filter.
# expnum is the key!
bad_reason = bad_exp['reasons'].apply(decam_info.decode_reason)
bad_reason_num = bad_reason.apply(lambda x: [decam_info.reason_num_dict[i] for i in x])

# %%
# find exposures with the same reason
print("Counts for bad oki exposures")
oki_counts = []
for i, rea in enumerate(decam_info.reason_li):
    oki_counts.append(np.sum(bad_exp['reasons'][expnum_bad_oki_idx] & 2**decam_info.reason_num_dict[rea] > 0))
    print(i, rea, np.sum(bad_exp['reasons'][expnum_bad_oki_idx] & 2**decam_info.reason_num_dict[rea] > 0))
print("Total images:", expnum_bad_oki_idx.shape[0])
plt.bar(np.arange(len(decam_info.reason_li)), oki_counts)
plt.xticks(np.arange(len(decam_info.reason_li)))
plt.xlabel("Category number")
plt.show()


# %%
def plot_category_counts(df, reason_dict=None):
    if reason_dict is None:
        reason_dict = decam_info.reason_num_dict
        reason_li = decam_info.reason_li 
    else:
        reason_li = list(reason_dict.keys())
    ooi_counts = []
    for i, rea in enumerate(reason_li):
        ooi_counts.append(np.sum(df['reasons'] & 2**reason_dict[rea] > 0))
        print(i, rea, np.sum(df['reasons'] & 2**reason_dict[rea] > 0))
    print("Total images:", len(df))
    fig, ax1 = plt.subplots()
    ax1.bar(np.arange(len(reason_li)), ooi_counts)
    ax2 = ax1.twinx()
    ax1.set_xticks(np.arange(len(reason_li)))
    ax1.set_xlabel("Category number")
    ax1.set_ylabel("counts")
    ax2.bar(np.arange(len(reason_li)), ooi_counts / np.sum(ooi_counts))
    ax2.set_ylabel("percentage")
    plt.show()


# %% [markdown]
# ## Analysis used in the paper
# Table 3 in the bad exposure papera

# %%
# find exposures with the same reason
print("Counts for bad ooi exposures")
ooi_counts = []
ooi_train = []
bad_exp_ooi = bad_exp.iloc[expnum_bad_ooi_idx]
print("label", "reason", "DECaLS", "DES+DELVE", "both", "total")
for i, rea in enumerate(decam_info.reason_li):
    sel_rea = (bad_exp_ooi['reasons'] & 2**decam_info.reason_num_dict[rea]) > 0
    if i not in [0, 3, 19, 11]:
        ooi_train.append(np.sum(sel_rea))
    ooi_counts.append(np.sum(sel_rea))
    print(i, rea, np.sum((bad_exp_ooi[sel_rea]['vi_source'] == 2**decam_info.reason_source_dict['Rongpu'])),
          np.sum((bad_exp_ooi[sel_rea]['vi_source'] == 2**decam_info.reason_source_dict['Alex'])) , 
          np.sum((bad_exp_ooi[sel_rea]['vi_source'] == (2**decam_info.reason_source_dict['Rongpu']+2**decam_info.reason_source_dict['Alex']))),
          np.sum(sel_rea))
print("Total images with OOI:", expnum_bad_ooi_idx.shape[0])
print("Total images in training:", np.sum(ooi_train))
plt.bar(np.arange(len(decam_info.reason_li)), ooi_counts)
plt.xticks(np.arange(len(decam_info.reason_li)))
plt.xlabel("Category number")
plt.show()
plt.hist(bad_exp_ooi['filter'], bins=np.arange(1, 7))
plt.xticks(np.arange(1, 7))
plt.xlabel("filter number")
plt.show()
print(decam_info.filter_dict)

# %%
df_bad_exp_ooi = bad_exp.iloc[expnum_bad_ooi_idx]

# %% [markdown]
# ## Build the dataset
# Build the training dataset with many bad exposures and then adding good
# exopures. Using `ooi` images for supervised learning, and `oki` for self-supervised learning. This is because `oki` has better background uniformity in per CCD level, while `ooi` images are matched to large-scale background and used in downstream tasks. The number of bad `ooi` images is more balanced in each category. 

# %% [markdown]
# ### First build `ooi` image dataset
#
# Use all the bad `ooi` exposure CCD images and add 60k random good exposure CCD images to the dataset. This would be equivalent to take 3k exposures with 20 CCD images per exposure. In total, we have 48k bad images + 60k good images bofore applying any cut.
#
# Filter is another consideration for this issue.
#
# ### Galactic latitude cut
#
# Adding the extinctio cut (instead of galactic latitude) cut to remove the many cases where the dust in milky way was misclassified as bad exposures. The number of bad exposures drops to 31k. After adding the galactic latitude cut, we drop these categories due to the low occurence:
#
# - 3 Bad_seeing -> can be detected with zpt
# - 10 Fringing -> none after ebv cut
# - 11 Canopus -> bright star should be easily detected with sky position
#
# There are gaps in label, so we need to remake the label class using a second dictionary to map the original label to training labels.
# One may consider canopus as a general class that the image is taken around a bright star.
# Adjusting the number of good exposures to 20k to balance the dataset. This is equivalent to take 2k exposures with 10 CCD images per exposure.
#
# - Rev. 1: due to many good exposures will lead the model to cheat, reduce the number of good exposures to 5000 images. 1k exposures with 5 CCD images each.
# - Rev. 2: for binary class, increase number of good images to 24k, which is 2k expsoures with 12 CCD images each.
# - Rev. 3: after incorprating the new bad images in, change the good images to 12k = 2k exp x 6 CCD. This is to ensure bad exposures have chances to be processed in the training.
# - Rev. 4: after applying dr10 cuts, the categories with bad exposures dropped drastically.

# %%
drop_class_set = {0, 3, 10, 11} 

# %%
drop = [len(drop_class_set & set(li)) > 0 for li in df_bad_exp_ooi['reasons'].apply(lambda x: decam_info.decode_reason(x, True))]
df_bad_exp_ooi_cut = df_bad_exp_ooi[~np.array(drop)]

# %%
# find exposures with the same reason
print("Training dataset for bad ooi exposures")
plot_category_counts(df_bad_exp_ooi_cut)

# %% [markdown]
# # Updates
#
# From the plot above, the dataset is very imbalanced. To balanece it, we downsample some classes and upsample their weights in the classification task:
#
# - class 2: reduce to 1/3, weight x3
# - class 6: reduce to 1/2, weight x2
# - class 7: reduce to 0.75, weight x1.333

# %%
rng = np.random.default_rng(42)

class_2_idx = df_bad_exp_ooi_cut[(df_bad_exp_ooi_cut["reasons"] & 2**2) > 0].index
class_6_idx = df_bad_exp_ooi_cut[(df_bad_exp_ooi_cut["reasons"] & 2**6) > 0].index
class_7_idx = df_bad_exp_ooi_cut[(df_bad_exp_ooi_cut["reasons"] & 2**7) > 0].index


# remember this is the drop idx fraction, so complement the number above
drop_idx = np.hstack(
    [
        rng.choice(class_2_idx, replace=False, size=int(0.667*len(class_2_idx))),
        rng.choice(class_6_idx, replace=False, size=int(0.5*len(class_6_idx))),
        rng.choice(class_7_idx, replace=False, size=int(0.25*len(class_7_idx)))
    ]
)

df_bad_exp_ooi_cut_bal = df_bad_exp_ooi_cut.drop(drop_idx)

# find exposures with the same reason
print("Training dataset for bad ooi exposures")
plot_category_counts(df_bad_exp_ooi_cut_bal)
# df_bad_exp_ooi_cut_bal = df_bad_exp_ooi_cut

# %%
rng = np.random.default_rng(42)

n_samp = 10000
# # ! changed 20 to 50 for verification, as recommended by the referee.
num_ccd_exp = 50
# to ensure good ooi are in the dr10 after the dr10 cut expnum
# see above
samp_good_expnum = rng.choice(np.intersect1d(expnum_good_ooi, dr10_after_cut_expnum),
                              n_samp, replace=False)    
# note that the number 61 (CCD 62) will not be included in the analysis
samp_good_hdu = np.vstack([rng.choice(np.arange(1, 60), num_ccd_exp, replace=False) for _ in range(n_samp)])
samp_good_hdu = np.sort(samp_good_hdu, axis=1)
samp_good_exp_hdu = np.hstack([samp_good_expnum.reshape(-1, 1), samp_good_hdu])
# np.savetxt("../data/")

# %%
uni_exp, uni_idx, uni_counts = np.unique(dr10_tab['expnum'], return_index=True, return_counts=True)

# %%
picked_image_hdu = samp_good_hdu #ori_good['image_hdu']
ori_samp_good_hdu = picked_image_hdu.reshape(n_samp, num_ccd_exp)

# %%
samp_good_hdu.shape

# %%
filter_num = []
fnames = []
missing_two = []
uni_tab = dr10_tab[uni_idx]
ccdname_list = []
samp_good_hdu = []
for i, exp in enumerate(samp_good_expnum):
    this_idx = np.where(uni_tab['expnum'] == exp)[0][0]
    row = uni_tab[this_idx]
    filter_num.append(decam_info.filter_dict[row['filter']])
    fnames.append(row['image_filename'])
    all_idx = np.arange(0, uni_counts[this_idx]) + uni_idx[this_idx]
    this_exp_tab = dr10_tab[all_idx][['image_hdu', 'ccdname']]
    # Table is already sorted in image hdu / ccdnum order
    # sort_idx = this_exp_tab.argsort("image_hdu")
    # this_exp_tab = this_exp_tab[sort_idx]
    tab_idx = ori_samp_good_hdu[i]-1 #np.sort(rng.choice(len(this_exp_tab), num_ccd_exp, replace=False))
    # if exp == 978264:
    #     tab_idx = [np.where(x == this_exp_tab['image_hdu'])[0][0] for x in ori_samp_good_hdu[i]]
    samp_good_hdu.append(this_exp_tab[tab_idx]['image_hdu'])
    # ccdname_list.append(this_exp_tab['ccdname'][samp_good_hdu[i]-1])
    ccdname_list.append(this_exp_tab[tab_idx]['ccdname'])


# %%
def flatten(xss):
    return [x for xs in xss for x in xs]

expand_repeat = lambda arr, n: flatten([[it]*n for it in arr])

# %%
samp_good_exp_hdu

# %%
# get_all = lambda to_get, li: [li[it] for it in to_get]
# ccd_li = flatten([get_all(row, decam_info.ccdnum_li_m2) if miss_2 
#                   else get_all(row, decam_info.ccdnum_li_m1)
#                   for miss_2, row in zip(missing_two, samp_good_hdu)])
ccd_li = [decam_info.ccdname2num[ccdname] for ccdname in flatten(ccdname_list)]

# %%
df_good = pd.DataFrame(zip(*[
    expand_repeat(fnames, num_ccd_exp), # fname
    expand_repeat(samp_good_expnum, num_ccd_exp), # expnum
    ccd_li,
    flatten(samp_good_hdu),
    expand_repeat(filter_num, num_ccd_exp),
    [0] * (num_ccd_exp * n_samp),
    [0] * (num_ccd_exp * n_samp)
]), columns=['image_filename'] + list(bad_exp.columns))
    

# %%
df_good.to_csv("../data/samples/all_sample_decam_dr10_good_exp_ooi.csv", index=False)

# %%
# the expnum are sorted
all(df_bad_exp_ooi_cut_bal.expnum.unique() == np.unique(df_bad_exp_ooi_cut_bal.expnum))

# %%
uni_exp, counts = np.unique(df_bad_exp_ooi_cut_bal.expnum, return_counts=True)
fnames = flatten([[uni_tab[uni_tab['expnum'] == exp]['image_filename'][0]]*co for exp, co in zip(uni_exp, counts)])

# %%
df_bad_exp_ooi_cut_bal.insert(0, "image_filename", fnames)

# %%
df_bad_exp_ooi_cut_bal.to_csv("../data/samples/sample_decam_dr10_bad_exp_ooi.csv", index=False)

# %% [markdown]
# ## Now split the dataset
#
# Use 70-30 split for training and testing
#
# ! Should do the split for category rather than use dataset as a whole.
# This will ensure there are both training and testing dataset for each category.

# %%
from sklearn.model_selection import train_test_split

# %%
all_category = set(decam_info.reason_num_dict.values()) - drop_class_set
ooi_labels = []
for i in all_category:
    ooi_labels.append(list(df_bad_exp_ooi_cut_bal[df_bad_exp_ooi_cut_bal['reasons'] & 2**i > 0].index))

# %%
all_category

# %%
bad_train_idx = []
bad_test_idx = []
for l in ooi_labels:
    print(len(l))
    train_bad_cat_idx, test_bad_cat_idx = train_test_split(l, test_size=0.3, random_state=42, shuffle=True)
    bad_train_idx += train_bad_cat_idx
    bad_test_idx += test_bad_cat_idx

# %%
good_train_idx, good_test_idx = train_test_split(np.arange(len(df_good)), test_size=0.3, random_state=42, shuffle=True)
# bad_train_idx, bad_test_idx = train_test_split(np.arange(len(df_bad_exp_ooi_cut_bal)), test_size=0.3, random_state=42, shuffle=True)

# %%
df_all = pd.concat([df_good, df_bad_exp_ooi_cut_bal], ignore_index=True, axis=0) \
            .sort_values(by=['expnum'])

# %%
# this line will create a difference between all samples vs. train + test
# because we dropped some exposures and some categories
df_all.to_csv("../data/samples/all_samples_ooi_dataset.csv", index=False)

# %%
train_df = pd.concat([df_good.iloc[good_train_idx],
                      df_bad_exp_ooi_cut_bal.loc[bad_train_idx]],
                     ignore_index=True, axis=0).sort_values(by=['expnum'])
test_df = pd.concat([df_good.iloc[good_test_idx],
                      df_bad_exp_ooi_cut_bal.loc[bad_test_idx]],
                     ignore_index=True, axis=0).sort_values(by=['expnum'])

# %%
# decam/CP/V3.7/CP20130418/c4d_130419_082748_ooi_r_v1.fits.fz only has 45 HDUs
# expnum 198978
# decam/CP/V3.7/CP20130418/c4d_130419_081306_ooi_r_v1.fits.fz only has 45 HDUs
# expnum 198970
train_df = train_df.drop(train_df.query('expnum == 198978 | expnum == 198970').index)
test_df = test_df.drop(test_df.query('expnum == 198978 | expnum == 198970').index)

# %%
train_df.to_csv("../data/samples/train_supervised_ooi_dataset.csv", index=False)
test_df.to_csv("../data/samples/test_supervised_ooi_dataset.csv", index=False)

# %%
# check_exist = [(dr10_imdir / fn).exists() for fn in df_good['image_filename'].unique()]
# all(check_exist)

# %%
# find exposures with the same reason
print("Training dataset for bad ooi exposures")
plot_category_counts(train_df)

# %%
# find exposures with the same reason
print("Test dataset for bad ooi exposures")
plot_category_counts(test_df)

# %%
# good_path_check = [(dr10_imdir / p).exists() for p in df_good['image_filename'].unique()]
# all(good_path_check)

# bad_path_check = [(dr10_imdir / p).exists() for p in df_bad_exp_ooi_cut_bal['image_filename'].unique()]
# all(bad_path_check)

# df_good.query('expnum == 198978').index

# %%
ooi_counts = []
for i, rea in enumerate(decam_info.reason_li):
    ooi_counts.append(np.unique(train_df[train_df['reasons'] & 2**decam_info.reason_num_dict[rea] > 0].index))
    print(i, rea, np.sum(train_df['reasons'] & 2**decam_info.reason_num_dict[rea] > 0))

# %%
# check more than one category
multi = []
for idx, row in test_df.iterrows():
    rea = decam_info.decode_reason(row['reasons'], True)
    if len(rea) > 1:
        multi.append((idx, rea))

# %%
len(multi)

# %%
# train_reason_list = [it for i, it in enumerate(decam_info.reason_li)
#                      if i not in (0, 3, 7, 9, 10, 11)]

# train_reason_dict = dict(zip(train_reason_list, range(len(train_reason_list))))

# translate_table = {v: decam_info.reason_num_dict[k] for k, v in train_reason_dict.items()}

# translate_table

# plot_category_counts(train_df, translate_table)

# %%
df_all = pd.read_csv('/global/u1/b/brookluo/ssl-exposure-identification-paper/data/samples/all_samples_ooi_dataset.csv')
num_nodes = 1 # 4
split_size = len(df_all) // num_nodes
for i in range(num_nodes):
    df_all.iloc[i*split_size:(i+1)*split_size].to_csv(f"/pscratch/sd/b/brookluo/decam-exposure/revision/proc-data/node{i}_dr10_sample.csv",
                                                  index=False)

# %%
df_all.reasons.unique()

# %%
