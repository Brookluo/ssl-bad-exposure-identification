# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: DESI main
#     language: python
#     name: desi-main
# ---

# %%
# import torch
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# %%
# astropy imports
from astropy.table import Table
from astropy.io import fits
import fitsio

# %%
import sys
sys.path.append("../src")
import decam_info

# %%
from tqdm import tqdm
from joblib import Parallel, delayed

import pandas as pd

# %%
dr8_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr8')
dr9_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')

# %%
dr9_imdir = dr9_dir / "images"
dr10_imdir = dr10_dir / "images"

# %%
# meta_tab = Table.read('/global/cfs/cdirs/desi/users/rongpu/useful/survey-ccds-decam-dr9-trim.fits')
dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")

# %% [markdown]
# # Generally divide the exposure problems into several main catagories
#
#
# need some insights here, we cannot go into too much details
# otherwise we lose generalizability.
#
# 0. WCSCAL bad or not
# 1. saturated
# 2. clouds or bad transparency
# 3. bad seeing
# 4. psf
# 5. nonoptimal-exp
# 6. Ghost/Scatter
# 7. NObjects
# 8. Bad CCD
# 9. Noise
# 10. Fringing
# 11. Canopus
# 12. Wonky
# 13. Telescope Moving
# 14. out of focus
#
# Clouds -> 2
#
# Ghosting -> 6
#
# Telescope Tracking -> 13
#
#
# WCSCAL successful but with additional problems.
#
# Use bitmap to indicate the true or false for specific parameters.

# %% [markdown]
# # Make a table with exposure and ccd and reasons for it's bad
#
# Criteria:
# 1. If it's in Decam_list or Rongpu's list, then all CCDs are bad unless otherwise noted. All those exposures have 60 CCDs
# 2. Alex's list has the exposure numbers

# %% [markdown]
# # Decam or Rongpu's list

# %%
rp_decam_expid = []
rp_decam_reason = []
with open('../data/psf_rongpu_decam-bad_expid.txt', 'r') as fp:
    for l in fp:
        if l.startswith('#') or len(l.strip()) == 0:
            continue
        expnum, *exprea = l.split()
        if "-" in expnum:
            expnum, *subs = expnum.split("-")
            subs = " ".join(subs)
        else:
            subs = "all"
        rp_decam_expid.append((int(expnum), subs))
        if len(exprea) == 0:
            rp_decam_reason.append("Reason not specified")
        else:
            rp_decam_reason.append(" ".join(exprea))

# %%
rp_decam_expid, rp_decam_ccds = list(zip(*rp_decam_expid))
rp_decam_expid = np.array(rp_decam_expid)
rp_decam_ccds = np.array(rp_decam_ccds)


# %%
def extract_seeing_num(string):
    import re
    seeing_pattern = re.compile("seeing\s*[>=]\s*\d+.?\d*", re.IGNORECASE)
    extract_num = re.compile("\d+.?\d*", re.IGNORECASE)
    li_out = seeing_pattern.findall(string) 
    if len(li_out) == 0:
        print(f"No matched seeing string in \"{string}\"")
        return -1
    if len(li_out) > 1:
        print(f"Multiple seeing in \"{string}\" choose the first one")
    return float(extract_num.findall(li_out[0].strip('\"'))[0])


# %%
# first check all Rongpu's list
dset = []
missing_exp_rz = []
for expnum, ccd, reason in tqdm(zip(rp_decam_expid, rp_decam_ccds, rp_decam_reason), total=len(rp_decam_expid)):
    # 0 expnum, 1 ccdnum, 2 fits hdunum , 3 filter, 4 reasons, 5 source
    # reason is in binary bit map, since binary operation is fast
    # also takes much less space to store that info
    # have a hard cut on num of CCDs, limiting to first 60 CCDs
    rows = dr10_tab[dr10_tab['expnum'] == expnum]
    if len(rows) == 0:
        missing_exp_rz.append(expnum)
        continue
    if ccd == "all":
        params = np.zeros((len(rows), 6), dtype=int)
        # 2 CCDs were not functional all the time
        # num of CCD can be 60, 61
        # S30/2 and N30/61 might not be functional
        # DECam focal plane starts with 1
        # https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        params[:, 1] = np.array([decam_info.ccdname2num[ccdname] for ccdname in rows['ccdname']])
        params[:, 2] = rows['image_hdu'].data
        # they should have the same filter for a single exposure
        params[:, 3] = decam_info.filter_dict[rows['filter'].data.astype(str)[0]]
    else:
        # should have only 1 CCD
        # unsure about the DECam CCD layout number
        # assert len(ccd) == 1, f"Should have only 1 bad CCD for this exposure: {expnum}, but has {len(ccd)}"
        pick_row = rows[rows['ccdname'] == ccd]
        assert len(pick_row) > 0, f"Cannot find {expnum}, {ccd} in DR10 table"
        assert len(pick_row) == 1, f"More than one {expnum}, {ccd} entry in DR10 table"
        params = np.zeros((1, 6), dtype=int)
        params[:, 1] = decam_info.ccdname2num[ccd]
        params[:, 2] = pick_row['image_hdu'].data
        params[:, 3] = decam_info.filter_dict[pick_row['filter'].data.astype(str)[0]]
        
    params[:, 0] = expnum
    reason = reason.strip().lower()
    if 'expfactor' in reason:
        params[:, 4] |= 2**decam_info.reason_num_dict['Nonoptimal_exp']
    if reason == '1':
        params[:, 4] |= 2**decam_info.reason_num_dict['Saturated']
    if ('transparency' in reason) or ('cloud' in reason) or (reason == '2'):
        params[:, 4] |= 2**decam_info.reason_num_dict['Clouds_transparency']
    if 'seeing' in reason:
        if ('bad seeing' in reason) or (extract_seeing_num(reason) >= 2.5):
            params[:, 4] |= 2**decam_info.reason_num_dict['Bad_seeing']
    if ("trail" in reason) or ("psf" in reason):
        params[:, 4] |= 2**decam_info.reason_num_dict['PSF']
    if "focus" in reason:
        params[:, 4] |= 2**decam_info.reason_num_dict['Out_of_focus']
    if "double image" in reason:
        # double image caused by telescope moving
        params[:, 4] |= 2**decam_info.reason_num_dict['Telescope_Moving']
    params[:, 5] |= 2**decam_info.reason_source_dict['Rongpu']
    dset.append(params)
arr_dset = np.vstack(dset)
np.save("../data/decam_dr10_RZ_bad_exp.npy", arr_dset)

# %%
np.unique(arr_dset[:, 0]).shape

# %%
# arr_dset[np.where(arr_dset[:, 2] & 2**decam_info.reason_num_dict['bad_seeing'])[0]]

# %% [markdown]
# # Alex's list

# %%
alex_bad_new = Table.read('../ext-data/des_exclude_y6a2.fits')

# %%
alex_bad_old = Table.read("../ext-data/delve_exclude_20230725.fits")

# %%
print("Number of exposures:", len(np.unique(alex_bad_old["EXPNUM"])), "number of category:", len(np.unique(alex_bad_old["REASON"])))
all_rea, counts = np.unique(alex_bad_old["REASON"], return_counts=True)
ii = np.argsort(counts)[::-1]
for (r, c) in zip(all_rea[ii], counts[ii]):
    print(f"{r}: {c} images")

# %%
print("Number of exposures:", len(np.unique(alex_bad_new["EXPNUM"])), "number of category:", len(np.unique(alex_bad_new["REASON"])))
all_rea, counts = np.unique(alex_bad_new["REASON"], return_counts=True)
ii = np.argsort(counts)[::-1]
for (r, c) in zip(all_rea[ii], counts[ii]):
    print(f"{r}: {c} images")

# %%
# original reasons
# reason_li = (
#     "Bad_WCSCAL",
#     "Saturated",
#     "Clouds_transparency",
#     "Bad_seeing",
#     "PSF",
#     "Nonoptimal_exp",
#     "Ghost_Scatter",
#     "NObjects",
#     "Bad_CCD",
#     "Noise",
#     "Fringing",
#     "Canopus",
#     "Wonky",
#     "Telescope_Moving",
#     "Out_of_focus"
# )

# new reasons:
reason_new2old_dict = {
               "HeavyClouds": "Clouds_transparency", 
               "Skysub+NOBJ": "NObjects",
             "Ghost/Scatter": "Ghost_Scatter",
               "BadPSFModel": "PSF",
          "double/elongated": "PSF",
                "double": "PSF",
              "Out of focus": "Out_of_focus",
                      # "DOME": None,
                    "Clouds": "Clouds_transparency",
              "Noise bkpl 1": "Noise",
         # "ExtremeBackground": "Noise",
           "TelescopeMoving": "Telescope_Moving",
          "Telescope moving": "Telescope_Moving",
                "HeavyCloud": "Clouds_transparency",
         # "Dome (from Wyatt)": None
                  # "All zero": # empty image
                     "Noise": "Noise",
       "CloudPlusBackground": "Clouds_transparency",
            "telescope moved": "Telescope_Moving",
                    "readout": "Readout",
             "Weird readout": "Readout",
            "Heavy Clouds": "Clouds_transparency",
    "No guiding... drifts": "Telescope_Moving",
            "Telescope moved": "Telescope_Moving",
            "telescope moved": "Telescope_Moving",
            "Guide fail": "Telescope_Moving",
            "Saturated/Shutter": "Saturated",
            "Bad readout": "Readout",
            "BAD_READOUT": "Readout",
# Y6A1_Astromety_substandard
               # NoVsubFlush
}

# %%
# Map images in the new dataset based the dictionary, 
# and then merge it with the old dataset
pick_rows_new = []
for i, row in enumerate(alex_bad_new):
    if row['REASON'] in reason_new2old_dict.keys():
        row['REASON'] = reason_new2old_dict[row['REASON']]
        pick_rows_new.append(i)
    elif row['REASON'] in decam_info.reason_num_dict.keys():
        pick_rows_new.append(i)

# %%
from astropy.table import vstack

# %%
pick_rows_old = []
for i, row in enumerate(alex_bad_old):
    rea = row['REASON'].replace(' ', '_').replace('/', '_')
    if rea in decam_info.reason_num_dict.keys():
        row['REASON'] = rea
        pick_rows_old.append(i)

# %%
alex_bad_exp_tab = vstack([alex_bad_new[pick_rows_new], alex_bad_old[pick_rows_old]])

# %%
reason, counts = np.unique(alex_bad_exp_tab['REASON'], return_counts=True)
idx = np.argsort(-counts)
print(*zip([r for r in reason[idx]], counts[idx]))

# %%
dr10_alex_bad_exp_tab = alex_bad_exp_tab[alex_bad_exp_tab['EXPNUM'] <= max(dr10_tab['expnum'])]

# %%
len(dr10_alex_bad_exp_tab), len(alex_bad_exp_tab)

# %%
# # Take top 10 reasons from here
# reason, counts = np.unique(dr10_alex_bad_exp_tab['REASON'], return_counts=True)
# idx = np.argsort(-counts)

# print(reason[idx][:20])
# print(counts[idx][:20])

# pick_top = np.where(np.logical_or.reduce([dr10_alex_bad_exp_tab['REASON'] == r for r in reason[idx][:10]]))[0]
# top_alex_bad_tab = dr10_alex_bad_exp_tab[pick_top]

# treat top catagory the same as the dr10 catagory
top_alex_bad_tab = dr10_alex_bad_exp_tab

# %%
# all_expnum = set(dr10_alex_bad_exp_tab[pick_top]['EXPNUM']) | set(decam_expid)
# top_alex_bad = set(dr10_alex_bad_exp_tab[pick_top]['EXPNUM'])
# set(decam_expid)

# %%
# # then check Alex's dataset

# # serial code
# dset = []
# missing_exp_alex = []
# arr_dset = None
# restart = pick_top.copy()
# if Path("../data/decam_dr10_Alex_bad_exp.npy").exists():
#     arr_dset = np.load("../data/decam_dr10_Alex_bad_exp.npy")
#     restart = pick_top[arr_dset.shape[0]:]
# for row in tqdm(dr10_alex_bad_exp_tab[pick_top]):
#     # bad_match = np.where((arr_dset[:, 0] == row['EXPNUM']) & (arr_dset[:, 1] == row['CCDNUM']))[0]
#     dr10_match = np.where((dr10_tab['expnum'] == row['EXPNUM']) & (dr10_tab['ccdname'] == ccdnum2name[row['CCDNUM']]))[0]
#     if len(dr10_match) == 0:
#         missing_exp_alex.append(row['EXPNUM'])
#         continue
#     pick_dr10_row = dr10_tab[dr10_match]    
#     params = np.zeros(6, dtype=int)
#     params[:4] = (row['EXPNUM'], row['CCDNUM'], pick_dr10_row['image_hdu'].data[0],
#                   filter_dict[pick_dr10_row['filter'].data.astype(str)[0]])
#     params[4] |= 2**decam_info.reason_num_dict[row["REASON"].replace(' ', '_')]
#     params[5] |= 2**decam_info.reason_source_dict['Alex']
#     dset.append(params)
#     if len(dset) == 5000:
#         if arr_dset is None:
#             arr_dset = np.vstack(dset)
#         else:
#             arr_dset = np.vstack([arr_dset, dset])
#         np.save("../data/decam_dr10_Alex_bad_exp.npy", arr_dset)
#         dset = []
# np.save("../data/decam_dr10_Alex_bad_exp.npy", arr_dset)

# %%

# Force DR10 arrays into memory and simple types
col_expnum  = np.array(dr10_tab['expnum'])
col_ccdname = np.array(dr10_tab['ccdname'], dtype=str)
col_hdu     = np.array(dr10_tab['image_hdu'])
col_filter  = np.array(dr10_tab['filter'], dtype=str)

# Build dictionary keyed by (expnum, ccdname)
dr10_lookup = {
    (int(exp), str(ccd)): (int(hdu), str(filt))
    for exp, ccd, hdu, filt in zip(col_expnum, col_ccdname, col_hdu, col_filter)
}

dset = []
missing_exp_alex = []

for row in tqdm(top_alex_bad_tab):
    expnum = int(row['EXPNUM'])
    ccdnum = int(row['CCDNUM'])
    ccdname = decam_info.ccdnum2name[ccdnum]

    key = (expnum, ccdname)
    if key not in dr10_lookup:
        missing_exp_alex.append(expnum)
        continue

    hdu, filt = dr10_lookup[key]

    params = (
        expnum,
        ccdnum,
        hdu,
        decam_info.filter_dict[filt],
        2**decam_info.reason_num_dict[row["REASON"]],
        2**decam_info.reason_source_dict['Alex'],
    )
    dset.append(params)

dset = np.array(dset, dtype=int)

# %%
np.save("../data/decam_dr10_alex_bad_exp.npy", dset)
np.save("../data/decam_alex_expnum_not_in_dr10.npy", np.unique(missing_exp_alex))

# %%
np.unique(dset[:, 0]).shape

# %%
dset.shape


# %% [markdown]
# # WCSCAL bad images

# %%
# lastly, check WCSCAL
# bad_exp = []
def find_bad(imdir, file):
    fpath = Path(imdir, file)
    if not fpath.exists():
        # is bad wcscal, bad file, expnum
        return False, file, -1
    meta = fitsio.read_header(fpath)
    if meta['WCSCAL'].strip().lower() != 'successful':
        return True, None, meta['EXPNUM']
    return False, None, -1
# find_bad = lambda path: f, None if Path(path).exists() and fitsio.read_header(path)['WCSCAL'].strip() != 'Successful' else 

dr10_uni_fnames = np.unique(dr10_tab['image_filename'])
with Parallel(n_jobs=10) as parallel:
    bad_exp_files = parallel(delayed(find_bad)(dr10_imdir, f) for f in tqdm(dr10_uni_fnames))
    # for f in tqdm(files):
        # header = fitsio.read_header(dr10_imdir / f)
        # if header['WCSCAL'].strip() != 'Successful':
        #     bad_exp.append(f)
        # print(f)
# with open("../data/bad_exp_dr10_decam.txt", 'w') as fp:
#     fp.write('\n'.join(bad_exp))
bad_exp, bad_files, expnum = list(zip(*bad_exp_files))
arr_bad_exp = np.array(bad_exp)
arr_expnum = np.array(expnum)
wcscal_bad_fnames = dr10_uni_fnames[arr_bad_exp]
expnum_wcscal = arr_expnum[arr_bad_exp]
wcscal_fname_expnum = list(zip(wcscal_bad_fnames, expnum_wcscal))
with open("../data/decam_dr10_bad_WCSCAL_fname_expnum.txt", 'w') as fp:
    fp.write("\n".join([",".join([str(x) for x in it]) for it in wcscal_fname_expnum]))
bad_files_clean = [f for f in bad_files if f is not None]
with open("../data/missing_files_exp_dr10_decam.txt", 'w') as fp:
    fp.write('\n'.join(bad_files_clean))

# fnames, idx_fnames = np.unique(col_fname, return_index=True)
# uni_col_expnum = col_expnum[idx_fnames]
# expnum_wcscal = [uni_col_expnum[fnames == fn] for fn in wcscal_bad]

# wcscal_fname_expnum = list(zip(wcscal_bad, expnum_wcscal))
# with open("../data/decam_dr10_bad_WCSCAL_fname_expnum.txt", 'w') as fp:
#     fp.write("\n".join([",".join([str(x) for x in it]) for it in wcscal_fname_expnum]))

# %%
# Now merge three files together
arr_rz = np.load("../data/decam_dr10_RZ_bad_exp.npy")
arr_alex = np.load("../data/decam_dr10_alex_bad_exp.npy")
with open("../data/decam_dr10_bad_WCSCAL_fname_expnum.txt", 'r') as fp:
    wcscal_fname_expnum = [l.strip().split(",") for l in fp]
wcscal_fname, wcscal_expnum = list(zip(*wcscal_fname_expnum))
expnum_wcscal = np.array(wcscal_expnum, dtype=int)

# %%
arr_all = arr_rz.copy()

# first find common exposures
exp_both = np.intersect1d(arr_rz[:, 0], arr_alex[:, 0], assume_unique=False)
# Use RZ output as base, copy Alex data in
for expnum in exp_both:
    exp_alex = arr_alex[arr_alex[:, 0] == expnum]
    idx_rz = np.where(arr_rz[:, 0] == expnum)[0]
    exp_rz = arr_rz[idx_rz]
    for row in exp_alex:
        one_rz_idx = np.where(exp_rz[:, 1] == row[1])[0]
        if len(one_rz_idx) == 0:
            # unique to alex list, not included in RZ's list
            arr_all = np.append(arr_all, row.reshape(1, -1), axis=0)
            continue
        # update 4 reason, 5 source
        arr_all[idx_rz[one_rz_idx], 4] |= row[4]
        arr_all[idx_rz[one_rz_idx], 5] |= row[5]

# then find the unique exps in Alex list
unique_exp_alex = np.setdiff1d(arr_alex[:, 0], arr_rz[:, 0], assume_unique=False)
for expnum in unique_exp_alex:
    exp_alex = arr_alex[arr_alex[:, 0] == expnum]
    arr_all = np.append(arr_all, exp_alex, axis=0)

# lastly, check WCSCAL bad
exp_both_wcscal = np.intersect1d(arr_all[0], expnum_wcscal, assume_unique=False)
for expnum in exp_both_wcscal:
    exp_comm = arr_all[arr_all[:, 0] == expnum]
    exp_comm[:, 4] |= 2**decam_info.reason_num_dict['Bad_WCSCAL']

# for other cases with bad WCSCAL, just ignore them for now


# %%
exp_both_wcscal

# %%
pd.DataFrame(arr_all, columns=['expnum', 'ccdnum', 'image_hdu' , 'filter', 'reasons', 'vi_source'], dtype=int) \
    .sort_values(by=['expnum', 'ccdnum'], inplace=False) \
    .to_csv("../data/decam_dr10_bad_exp_all.csv", index=False)

# %%
arr_all.shape

# %%
arr_good = np.setdiff1d(dr10_tab['expnum'], arr_all[:, 0])

# %%
arr_good.tofile('../data/decam_dr10_good_exp.csv', sep='\n')

# %%
# dr10 ooi images availability
col_exp, idx = np.unique(dr10_tab['expnum'], return_index=True)
col_fname = dr10_tab['image_filename'][idx]
# dr10_ooi_exist = np.zeros(len(col_fname), dtype=bool)
# dr10_oki_exist = np.zeros(len(col_fname), dtype=bool)
# for i in tqdm(range(len(col_fname))):
#     # exp_fname = col_fname[col_exp == exp][0]
#     if (dr10_imdir / col_fname[i]).exists():
#         dr10_ooi_exist[i] = True
#     oki_fname = col_fname[i].replace("ooi", "oki")
#     if (dr10_imdir / oki_fname).exists():
#         dr10_oki_exist[i] = True
# with open("../data/dr10_ooi_images.csv", 'w') as fp:
#     fp.write("\n".join([",".join([str(x) for x in it]) for it in dr10_ooi_exist]))

# parallel
def check_exist(ooi_fname):
    oki_fname = ooi_fname.replace("ooi", "oki")
    return ((dr10_imdir / ooi_fname).exists(),
            (dr10_imdir / oki_fname).exists())
with Parallel(n_jobs=10) as parallel:
    ooi_oki = parallel(delayed(check_exist)(fn) for fn in tqdm(col_fname))

# %%
# all(['ooi' in fn for fn in col_fname])

# %%
ooi, oki = list(zip(*ooi_oki))

# %%
df = pd.DataFrame(zip(col_fname, col_exp, ooi, oki), columns=["dr10_image_filename", "expnum", "ooi_exist", "oki_exist"])

# %%
df.to_csv("../data/decam_dr10_ooi_oki_images_exist.csv", index=False)

# %%
df = pd.read_csv("../data/decam_dr10_ooi_oki_images_exist.csv")
bad_exp = pd.read_csv("../data/decam_dr10_bad_exp_all.csv")
np.setdiff1d(df.expnum[df.oki_exist], bad_exp.expnum)

# %% [markdown]
# ## Check DR9 and 10 LS flag

# %%
np.unique(dr10_tab['expnum'][dr10_tab['ccd_cuts'] > 0])

# %%
