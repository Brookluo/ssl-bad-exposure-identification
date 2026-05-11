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
import numpy as np
import pandas as pd
from pathlib import Path

# %%
import sys
sys.path.append("../src")
from decam_info import decode_reason, reason_li

# %%
datadir = Path("../data/samples/")
df_all = pd.read_csv(datadir / "all_samples_ooi_dataset.csv")

# %%
embeds_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/revision/eval/embeds_out")
allem_dir = embeds_dir / "all"

# %%
seed = 9
n_samp = 5 #20
newdir = embeds_dir / f"samp_{seed}"
newdir.mkdir(parents=True, exist_ok=True)

# %%
# sample from the good exposures with five sets of 20 random CCDs
df_all['reasons'] == 0

# %%
sel_good = df_all.groupby('expnum').apply(
    lambda x: (x['reasons'] == 0).all()
)
expnums = sel_good[sel_good].index.tolist()
df_good = df_all.query('expnum in @expnums').groupby('expnum').filter(lambda x: x.size > 50)

# %%
df_good_sampled = df_good.groupby('expnum').apply(
    lambda x: x.sample(n=n_samp, random_state=seed, replace=False)
).droplevel(0)

# %%
df_bad = df_all[df_all['reasons'] != 0]
df_new = pd.concat([df_good_sampled, df_bad], axis=0)

# %%
idx, num = np.unique(df_bad['reasons'], return_counts=True)

# %%
# use the lowest number as the default case


# %%
df_new.to_csv(newdir / "sampled_ooi_dataset.csv", index=True)

# %%
data = np.load(allem_dir / "original_embeds.npy")
idx =  np.load(allem_dir / "original_idx.npy").astype(int)
label = np.load(allem_dir / "label.npy")


# %%
# idx, num = np.unique(label, return_counts=True)

# sum(num[1:])

# idx

# for i in range(len(idx)):
#     if i == 0:
#         print("good", num[i])
#     print(int(idx[i-1]))
#     print(reason_li[int(idx[i-1])], num[i])

# %%
len(df_bad)

# %%
sort_idx = np.argsort(idx)

# %%
np.save(newdir / "embeds.npy", data[sort_idx][df_new.index])
np.save(newdir / "idx.npy", df_new.index)
np.save(newdir / "label.npy", label[sort_idx][df_new.index])

# %%

for i in {51..54}; do python do_analysis_viz.py -edir $SCRATCH/decam-exposure/revision/eval/embeds_out/samp_$i -ppath ~/decam-exposure-quality/postproc/knn_pipe.pkl -rdir $SCRATCH/decam-exposure/revision/eval/embeds_out/samp_$i -figdir $SCRATCH/decam-exposure/revision/eval/plots/samp_$i --stage reduce plot classify --reduce_tsne -clsdir $SCRATCH/decam-exposure/revision/eval/embeds_out/samp_$i; done
