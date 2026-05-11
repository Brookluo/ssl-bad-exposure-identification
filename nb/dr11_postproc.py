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
import numpy as np
import pandas as pd

# %%
# import fitsio
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# %%
from pathlib import Path

# %%
from decam_qa import reason_li, ccdnum2name

# %%
dr11_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr11')
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11")
data_dir = root_dir / "proc-data"
path_tmpl = "/pscratch/sd/b/brookluo/decam-exposure/dr11/node{}/embeds_out"
outdir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11/inference")

# %%
df_merge = pd.read_csv(outdir / "merged_sample.csv")
# remember the -1 has done in the processing step, so no need to -1

# %%
df_merge

# %% [markdown]
# # Select high probability bad exposures

# %%
p_thresh = 0.9

# %%
pick_thresh = df_merge['ml_prob'] > p_thresh
pick_bad = df_merge['ml_label'] > 0
# pick_class = df_merge['ml_label'] == 14
pick = pick_thresh & pick_bad #& pick_class

# %%
selected = df_merge[pick]
uni_exp, idx, counts = np.unique(selected['expnum'], return_counts=True, return_index=True)

# %%
sel_exp = uni_exp[counts > 2]

# %% [markdown]
# # Cross check with the DR10 bad and the bad exposure list

# %%
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")

# %%
dr9tab = Table.read('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-ccds-decam-dr9.fits.gz')

# %%
[np.unique(dr9tab['expnum']).data, np.unique(dr10_tab['expnum'])]

# %%
[np.unique(dr9tab['expnum']).data, np.unique(dr10_tab['expnum']).data]

# %%
len(sel_exp)

# %%
np.intersect1d(sel_exp, np.hstack([np.unique(dr9tab['expnum']).data, np.unique(dr10_tab['expnum']).data])).shape

# %%
res = [np.where(dr10_tab['ccd_cuts'] & 2**i)[0] for i in range(19)]

all_cut_idx = np.unique(np.hstack(res))
pick_idx = np.ones(len(dr10_tab), dtype=bool)
pick_idx[all_cut_idx] = False

dr10_after_cut_expnum = dr10_tab["expnum"][pick_idx]

# %%
min(np.unique(dr10_tab["ccd_cuts"][~pick_idx]))

# %%
dr10_bad_in_selected = np.intersect1d(sel_exp, np.unique(dr10_tab["expnum"][~pick_idx]))

# %%
len(dr10_bad_in_selected)

# %%
data_dir = Path("../data")
bad_exp = pd.read_csv(data_dir / "decam_dr10_bad_exp_all.csv")

# %%
np.intersect1d(sel_exp, bad_exp["expnum"])
# None means all unique to the three list of bad exposures

# %% [markdown]
# # Check low and high radec to determine the crowd fields

# %%
dr11_sorted_good_drop = Table.read(root_dir / "data/good-exp-dr11-suvey-ccds.fits")

# %%
_, dr11idx = np.unique(dr11_sorted_good_drop['expnum'], return_index=True)
dr11_uni_tab = dr11_sorted_good_drop[dr11idx]

# %%
# DR11 check ra and dec first to find low and high sources
# create boresight ra, dec
exp_radec_idx = []
for exp in sel_exp:
    exp_radec_idx.append(np.where(dr11_uni_tab['expnum'] == exp)[0][0])

# %%
exp_radec = SkyCoord.guess_from_table(dr11_uni_tab[exp_radec_idx]["ra_bore", "dec_bore"], unit='deg')

# %%
gal_cut = np.abs(exp_radec.galactic.b) > 30 * u.deg

# %%
sum(gal_cut)

# %%
# 56 exposures are not available

# divide the exposures into high and low galactic latitude angle
hi_lat_expnum = dr11_uni_tab[exp_radec_idx]["expnum"][gal_cut]
hi_lat_radec = exp_radec[gal_cut]
lo_lat_expnum = dr11_uni_tab[exp_radec_idx]["expnum"][~gal_cut]
lo_lat_radec = exp_radec[~gal_cut]

# %%
print("Total num of exposures:", sel_exp.shape[0])
print("High galactic latitude count:", len(hi_lat_expnum))
print("Low galactic latitude count:", len(lo_lat_expnum))


# %% [markdown]
# # Divide the exposure VI task
#
# Each packet contains 400 exposures. 7 high galactic, 4 low galactic.

# %%
def flatten(xss):
    return [x for xs in xss for x in xs]


# %%
def get_info_from_html(html_path):
    # parse the html webpage from Frank
    # html is in a table format
    # meta = []
    # img_src = []
    entry = []
    with open(html_path, 'r') as f:
        # first_entry = [] # meta
        # second_entry = [] # img_src
        count = 0
        for l in f:
            l = l.strip()
            # print(l)
            if l.startswith("<table>") or l.startswith("</table>"):
                continue
            meta, img_src = l.rsplit("<td>", maxsplit=1)
            fname, expnum, *other = meta.split("<br>")
            img_path_fmt = '<td><img src="./images/{}.jpg"></tr>'
            img_src = img_path_fmt.format(fname.split(">")[-1])
            entry.append([fname, expnum, *other, img_src])
    return entry


# %% [markdown]
# # Sort the index based on its expnum

# %%
entry = get_info_from_html("/pscratch/sd/b/brookluo/decam-exposure/dr11/data/bad_candidates/index.html")

entry_expnum = [et[1] for et in entry]

sorted_idx = np.argsort(np.array(entry_expnum, dtype=int))

entry[:] = [entry[i] for i in sorted_idx]

# %% [markdown]
# # Remove redaundant exposures and make another webpage to display them

# %%
_, dr10_uniexp_idx = np.unique(dr10_tab["expnum"], return_index=True)
dr10_uni_tab = dr10_tab[dr10_uniexp_idx]

# %%
dr10_ccd_cuts_reasons = (
    "ERR_LEGACYZPTS",
    "NOT_GRZ",
    "NOT_THIRD_PIX",
    "EXPTIME_LT_30",
    "CCDNMATCH_LT_20",
    "ZPT_DIFF_AVG",
    "ZPT_SMALL",
    "ZPT_LARGE",
    "SKY_IS_BRIGHT",
    "BADEXP_FILE",
    "PHRMS",
    "RADECRMS",
    "SEEING_BAD",
    "EARLY_DECAM",
    "DEPTH_CUT",
    "TOO_MANY_BAD_CCDS",
    "FLAGGED_IN_DES",
    "PHRMS_S7",
    "DEPTH_CUT_2",
)

# %%
dr10_tab_bad = dr10_tab[dr10_tab["ccd_cuts"] > 0]

# %%
import copy

# %%
# Add DR10 reason to those images
mod_entry = copy.deepcopy(entry)
dupl_bad_idx = []
for i, et in enumerate(mod_entry):
    expnum = int(et[1])
    if not (expnum in dr10_bad_in_selected):
        continue
    rows = dr10_tab_bad[dr10_tab_bad["expnum"] == expnum]
    res = []
    for onerow in rows[rows["ccd_cuts"] > 0]:
        res += [dr10_ccd_cuts_reasons[i] for i in
            np.where([onerow['ccd_cuts'] & 2**i for i in range(19)])[0]
        ]
    newfname = et[0].split("<td>")[-1]
    oldfname = onerow['image_filename'].rsplit("/")[-1].split(".")[0]
    et.insert(-1, f"DR10 ccd_cuts: {', '.join(np.unique(res))}")
    if newfname == oldfname:
        dupl_bad_idx.append(i)

# Now remove the duplicate bad exposures from the CCD list
# dupl_bad = [entry[i] for i in dupl_bad_idx]
# for i in sorted(dupl_bad_idx, reverse=True):
#     del entry[i]

# %%
entry[:] = [et for et in mod_entry]


# %%
def make_webpage(master_list, pack_idx, root_dir, base_name, num_element=400):
    count = 0
    base_tmpl = "<table>{}\n</table>"
    content = []
    start_exp = -1
    for i, idx in enumerate(pack_idx):
        if start_exp == -1:
            start_exp = master_list[idx][1]
        content.append("<br>".join(master_list[idx]))
        if num_element > 0 and i and i % num_element == 0:
            end_exp = master_list[idx][1]
            with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
                web.write(base_tmpl.format("\n".join(content)))
            count += 1
            start_exp = -1
            content = []
    if len(content):
        end_exp = master_list[idx][1]
        with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
            web.write(base_tmpl.format("\n".join(content))) 


# %%
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11/data/bad_candidates")

# %%
# # Remove the duplicate DR10 exposure cuts from the bad exposure list
# for i in sorted(dupl_bad_idx, reverse=True):
#     del mod_entry[i]
mod_entry = copy.deepcopy(entry)

# %% [markdown]
# # Now split into two big category, low and high and subsplit in into 400 images per packet
#
# Add ra, dec, and ml labels with CCD number and probablity

# %%
hi_lat_pack = []
lo_lat_pack = []
entry_rea = []
entry_expnum = []
for i, et in enumerate(mod_entry):
    exp = int(et[1])
    rows = selected.query("expnum==@exp")
    reasons = [reason_li[l] if l > 0 else "good" for l in rows["ml_label"]]
    entry_rea.append(list(np.unique(reasons)))
    entry_expnum.append(exp)
    probs = rows["ml_prob"]
    ccdnames = [ccdnum2name[num] for num in rows["ccdnum"]]
    idxs = np.where(lo_lat_expnum == exp)[0]
    if len(idxs):
        idx = idxs[0]
        exp_radec = lo_lat_radec[idx]
        is_low = True
        lo_lat_pack.append(i)
    else:
        idx = np.where(hi_lat_expnum == exp)[0][0]
        exp_radec = hi_lat_radec[idx]
        is_low = False
        hi_lat_pack.append(i)
    # need RA Dec, ml identification
    et[-1:-1] = [
        f"boresight radec: {exp_radec.ra.deg:.4f}, {exp_radec.dec.deg:.4f}",
        f"ml identify:<br>" + "<br>".join(map(str, list(zip(ccdnames, reasons, probs)))) # can use str()
    ]

# %%
# Bad exposures
make_webpage(mod_entry, dupl_bad_idx, root_dir, "duplicate_dr10_bad_same_fname", -1)

# %%
# remove duplicate index from hi and low index
hi_lat_pack = np.setdiff1d(hi_lat_pack, dupl_bad_idx)
lo_lat_pack = np.setdiff1d(lo_lat_pack, dupl_bad_idx)

# %%
make_webpage(mod_entry, hi_lat_pack, root_dir, "high_gal_lat_exp", 400)

# %%
make_webpage(mod_entry, lo_lat_pack, root_dir, "low_gal_lat_exp", 400)


# %%
def write_to_csv(master_list, pack_idx, root_dir, base_name, num_element=400):
    csv_list = []
    onelist = []
    for i, idx in enumerate(pack_idx):
        et = master_list[idx]
        exp = int(et[1])
        fname = et[0].split("<td>")[-1]
        rea = ",".join(np.unique([it.split(",")[1].strip().strip("'") for it in et[-2].split("<br>")[1:]]))
        onelist.append((exp, fname, rea))
        if i and i % num_element == 0:
            csv_list.append(onelist)
            onelist = []
    if len(onelist):
        csv_list.append(onelist)
    for i, li in enumerate(csv_list):
        pd.DataFrame(li, columns=["expnum", "image_fname", "ML_reason"]).to_csv(root_dir / f"{i}_{base_name}.csv", index=False)


# %%
write_to_csv(mod_entry, hi_lat_pack, root_dir, "high_gal_lat_exp")

# %%
write_to_csv(mod_entry, lo_lat_pack, root_dir, "low_gal_lat_exp")

# %%
len(entry)

# %%
len(dr10_bad_in_selected) - len(dupl_bad_idx)

# %%
len(dupl_bad_idx)

# %%
len(entry) - len(dupl_bad_idx)

# %%
