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
import pandas as pd
from pathlib import Path
from typing import List

# %%
import numpy as np

# %%
from astropy.table import Table

# %%
# %%
from decam_qa.utils import get_info_from_html, make_webpage

# %%
vidir = Path("/global/u1/b/brookluo/decam-exposure-quality/data/vi")

# %%
alexdir = vidir / "alex"
frankdir = vidir / "frank"
yldir = vidir / "yl"

# %%
datadir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11/data/bad_candidates")
mydr11 = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11/data")

# %%
dr9_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')
dr8_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr8')
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
# dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")
dr11_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr11')

# %%
dr11tab = Table.read("/pscratch/sd/b/brookluo/decam-exposure/dr11/data/dr11-suvey-ccds.fits")

# %%
alldfs = []
for csv in datadir.glob("*.csv"):
    alldfs.append(pd.read_csv(csv))
alldf = pd.concat(alldfs, ignore_index=True)

# %%
alexdfs = []
for fp in alexdir.glob("*.xlsx"):
    alexdfs.append(pd.read_excel(fp))
alexdf = pd.concat(alexdfs, ignore_index=True).rename(columns={"Is_bad?": "Is_bad"})

# %%
frankdfs = []
for fp in frankdir.glob("*.xlsx"):
    df = pd.read_excel(fp)
    if df.columns[0] != "expnum":
        df = pd.read_excel(fp, header=1)
    frankdfs.append(df)
    # print(df.columns)
    # print(len(df.Reason[~df["Is_bad?"].isna()]))
frankdf = pd.concat(frankdfs, ignore_index=True).rename(columns={"Is_bad?": "Is_bad"})

# %%
yldfs = []
for fp in yldir.glob("*.xlsx"):
    df = pd.read_excel(fp)
    if df.columns[0] != "expnum":
        df = pd.read_excel(fp, header=1)
    yldfs.append(df)
yldf = pd.concat(yldfs, ignore_index=True).rename(columns={"Is_bad?": "Is_bad"})

# %%
yldf_not_nan = yldf[~yldf["Is_bad"].isna()]

# %%
sum(yldf_not_nan["Is_bad"].isin(['y', 'y?', 'yes', 'yes?']))

# %%
find_uni = lambda onedf: np.unique(onedf[~onedf["Is_bad"].isna()]["Is_bad"])
print("YL response:", find_uni(yldf))
print("Frank response:", find_uni(frankdf))
print("Alex response:", find_uni(alexdf))


# %%
def merge_vi_info(masterdf, vidf, viname, bad_symbols: List[str]):
    # if reason is None, that means the expnum is not VIed by that person
    badexp = np.zeros(len(masterdf), dtype=bool)
    badrea = np.full(len(masterdf), dtype=object, fill_value=None)
    reason = []
    vidf_not_nan = vidf[~vidf["Is_bad"].isna()]
    # now use response dictonary to parse the output where the bad is true
    # bad_symbol = resp_dict["bad_symbol"]
    vidf_bad = vidf_not_nan[vidf_not_nan["Is_bad"].isin(bad_symbols)]
    _, idx1, idx2 = np.intersect1d(masterdf.expnum, vidf_bad.expnum,
                                   return_indices=True)
    badexp[idx1] = True
    badrea[idx1] = vidf_bad.iloc[idx2].Reason
    masterdf[f"{viname}_badexp"] = badexp
    masterdf[f"{viname}_reason"] = badrea


# %%
merge_vi_info(alldf, alexdf, "alex", ['maybe', "yes", 'yes '])
merge_vi_info(alldf, frankdf, "frank", [1.])
merge_vi_info(alldf, yldf, "yufeng", ['y', 'y?', 'yes', 'yes?'])

# %%
at_least_one = alldf["alex_badexp"] | alldf["frank_badexp"] | alldf["yufeng_badexp"]
two_agree = (alldf["alex_badexp"] & alldf["frank_badexp"]) \
            | (alldf["alex_badexp"] & alldf["yufeng_badexp"]) \
            | (alldf["frank_badexp"] & alldf["yufeng_badexp"])

all_agree = alldf["alex_badexp"] & alldf["frank_badexp"] & alldf["yufeng_badexp"]

# %%
len(alldf[at_least_one]) / len(alldf)

# %%
len(alldf)

# %%
sum(at_least_one)

# %%
badexp = alldf[at_least_one]

# %%
badexp = badexp.sort_values(by=['expnum'])

# %%
matched_exp, badexp_idx, dr11_idx = np.intersect1d(badexp["expnum"], dr11tab["expnum"], return_indices=True)

# %%
np.unique(dr11tab[dr11_idx]["plver"])

# %%
plver = np.full(len(badexp), fill_value=None)
plver[badexp_idx] = dr11tab[dr11_idx]["plver"]
plver = plver.astype(str)

# %%
badexp.insert(2, "plver", plver)

# %%
sum(plver == "V4.8.1")

# %%
html_dir = Path("/global/cfs/cdirs/desicollab/users/brookluo/decam-exposure-data/dr11")

# %%
alldf.columns

# %%
final_bad_exp_html = []
for html in html_dir.glob(r"?_[low,high]*.html"):
    entry = get_info_from_html(html)
    for et in entry:
        idx = np.where(badexp["expnum"] == int(et[1]))[0]
        if len(idx) > 0:
            # print("here")
            onerow = badexp.iloc[idx[0]]
            et.insert(-1, f"plver: {onerow['plver']}")
            # print(onerow)
            vi = [name for name in ("alex", "frank", "yufeng") if onerow[f"{name}_badexp"]]
            et.insert(-1, f"confirmed by: {(',').join(vi)}")
            final_bad_exp_html.append(et)
        # break

expnum = [int(et[1]) for et in final_bad_exp_html]
sort_idx = np.argsort(expnum)

final_bad_exp_html = [final_bad_exp_html[i] for i in sort_idx]

# %%
# def make_webpage(master_list, pack_idx, root_dir, base_name, num_element=400):
#     count = 0
#     base_tmpl = "<table>{}\n</table>"
#     content = []
#     start_exp = -1
#     for i, idx in enumerate(pack_idx):
#         if start_exp == -1:
#             start_exp = master_list[idx][1]
#         content.append("<br>".join(master_list[idx]))
#         if num_element > 0 and i and i % num_element == 0:
#             end_exp = master_list[idx][1]
#             with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
#                 web.write(base_tmpl.format("\n".join(content)))
#             count += 1
#             start_exp = -1
#             content = []
#     if len(content):
#         end_exp = master_list[idx][1]
#         with open(root_dir / f"{count}_{base_name}_{start_exp}_{end_exp}.html", "w") as web:
#             web.write(base_tmpl.format("\n".join(content))) 

# make_webpage(final_bad_exp_html, np.arange(len(final_bad_exp_html)), html_dir, "vi_checked", 1e4)

# %%
len(final_bad_exp_html)

# %%
filters = np.array([fn.split("_")[-2] for fn in alldf["image_fname"]])

# %%
sum(filters == "Y")

# %%
from astropy.table import Table

# %%
tab = Table.from_pandas(badexp)

# %%
mydr11

# %%
badexp.to_csv(mydr11 / "dr11_final_selected_bad_exposures.csv", index=False)

# %%
alldf

# %%
badexp

# %%
