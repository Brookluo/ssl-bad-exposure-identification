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
import torch

# %%
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import joblib

# %%
from astropy.table import Table
import fitsio

# %%
sys.path.append("../src")
import decam_info

# %%
from inference import read_embeds

# %% [markdown]
# # Check the model performance

# %%
# load train data from last train
train_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dino_v2/base_resize_dr10cut/train")
train_data, train_idx, train_label = read_embeds(train_dir / "eval/embeds_out")
train_embeds = np.vstack(train_data) #[np.mean(it, axis=0) for it in train_data]) for five crop instead of resize

# %%
train_label = np.array(train_label, dtype=int)

# %%
knnpl = joblib.load('/global/u1/b/brookluo/decam-exposure-quality/postproc/knn_pipe.pkl')

# %%
knnpl.score(train_embeds, train_label)

# %%
y_pred_prob = knnpl.predict_proba(train_embeds)

pred_label = np.argmax(y_pred_prob, axis=1)
classes = knnpl.classes_
pred_label = np.array([classes[i] for i in pred_label], dtype=int)
y_prob = np.max(y_pred_prob, axis=1)

# %%
pick_thresh = y_prob > 0.8

# %%
train_label_cut = train_label[pick_thresh]
pred_label_cut = pred_label[pick_thresh]


# %%
def calculate_accuracy_per_class(y_true, y_pred):
    pick = y_true == 0
    score = sum(y_pred[pick] == y_true[pick]) / len(y_true[pick])
    num = 0
    label_name = ["0_good"]
    print(num, "good", score)
    for i, rea in enumerate(decam_info.reason_li, start=1):
        if sum(y_true==i) == 0:
            continue
        num += 1
        pick = y_true == i
        # score = knnpl.score(train_embeds[pick], train_label[pick])
        score = sum(y_pred[pick] == y_true[pick]) / len(y_true[pick])
        print(num, rea, score)
        label_name.append(f"{num}_{rea}")
    return label_name


# %%
label_name = calculate_accuracy_per_class(train_label_cut, pred_label_cut)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

cm = confusion_matrix(y_true=train_label_cut, y_pred=pred_label_cut)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# plt.savefig(plot_dir / f"vit_base_confusion_pretrain.png")
print("vit_base with resizing images")
print(classification_report(train_label_cut, pred_label_cut, target_names=label_name))

# %% [markdown]
# # Do inference on the DR11 images

# %%
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11")

# %%
dr11_imdir = Path("/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/images")

# %%
dr11_tab = pd.read_csv(root_dir / "data/good_exp_20ccd_decam_dr11.csv")

# %%
data_dir = root_dir / "proc-data"
path_tmpl = "/pscratch/sd/b/brookluo/decam-exposure/dr11/node{}/embeds_out"
outdir = Path("/pscratch/sd/b/brookluo/decam-exposure/dr11/inference")


# %%
def get_label_proba(model, X):
    y_pred_prob = model.predict_proba(X)
    y_pred = np.argmax(y_pred_prob, axis=1)
    pred_prob = np.max(y_pred_prob, axis=1)
    # convert from model class to data class
    pred_label = np.array([model.classes_[i] for i in y_pred], dtype=int)
    return pred_label, pred_prob


# %%
get_label_proba(knnpl, train_embeds)

# %%
# first check if all embeds have been generated
# name = f'idx_{orig_idx:d}_label_{onelab.item():d}'
for i in range(16):
    df = pd.read_csv(data_dir / f"node{i}_dr11_sample.csv")
    pred_class = np.full(len(df), fill_value=-1, dtype=int)
    pred_prob = np.full(len(df), fill_value=-1, dtype=float)
    # data, idx, label = read_embeds(path_tmpl.format(i))
    print(i)
    data, idx, label = read_embeds(path_tmpl.format(i))
    data = np.vstack(data)
    idx = np.array(idx, dtype=int)
    #[np.mean(it, axis=0) for it in train_data]) for five crop instead of resize
    y_pred, y_proba = get_label_proba(knnpl, data)
    pred_class[idx] = y_pred
    pred_prob[idx] = y_proba
    df["ml_label"] = pred_class
    df['ml_prob'] = pred_prob
    df.to_csv(outdir / f"node{i}_output.csv", index=False)
# miss some due to leftovers from splitting, now fixed
# remember to regenerate embeddings for the leftover exposures

# %%
# target_dir = Path(target_dir)
# data = []
# idx = []
# label = []
# for i in range(num):
#     with h5py.File(target_dir / f"{i}_worker_embeds.h5", 'r') as h5f:
#         dset = h5f["images"]
#         for it in dset:
#             data.append(np.array(dset[it]))
#             names = it.split("_")
#             if names[1] in idx:
#                 print("reptition:", num, num-1, "on idx:", idx)
#             idx.append(names[1])
#             label.append(names[-1])

# %%
df_list = []
for fp in outdir.glob("*.csv"):
    df_list.append(pd.read_csv(fp, index_col=0))

# %%
df_merge = pd.concat(df_list, ignore_index=True)

# %%
df_merge['ml_label']

# %%
df_merge.loc[df_merge['ml_label'] > 0, 'ml_label'] -= 1

# %%
np.unique(df_merge['ml_label'])

# %%
df_merge.to_csv(outdir / "merged_sample.csv", index=False)

# %%
df_merge = pd.read_csv(outdir / "merged_sample.csv")

# %%
pick_thresh = df_merge['ml_prob'] > 0.9
pick_bad = df_merge['ml_label'] > 0
# pick_class = df_merge['ml_label'] == 14
pick = pick_thresh & pick_bad #& pick_class

# %%
# first add a probablity cut
# then select top 200 bad exposures in each category
# this will overestimate the performance of the model

# %%
selected = df_merge[pick]

# %%
uni_exp, idx, counts = np.unique(selected['expnum'], return_counts=True, return_index=True)

# %%
exp_fnames = selected['image_filename'].iloc[idx]

# %%
sum(counts > 2)

# %%
# sort_idx = np.argsort(counts)[::-1]
sort_idx = np.argsort(uni_exp)
uni_exp = uni_exp[sort_idx]
counts = counts[sort_idx]
exp_fnames = exp_fnames.to_numpy()[sort_idx]

# %%
with open(root_dir / "fnames_bad_exp_prob_gt_09_occur_gt_2.csv", 'w') as fh:
    for i in np.where(counts > 2)[0]:
        exp = uni_exp[i]
        fname = exp_fnames[i]
        fh.write(f"{exp},{fname}\n")

# %%
plt.hist(selected['ml_label'], bins=np.arange(1, 16, 1))
plt.xticks(np.arange(1, 16, 1))
plt.show()

# %%
import fitsio
sys.path.append("../../img-spec-ml/src/")
from plot_utils import plot_zscale_image

# %%
df_exp = selected.query("expnum == 1214278")
# plt.subplots()
for i, row in df_exp.iterrows():
    print(row)
    data = fitsio.read(dr11_imdir / row["image_filename"], ext=row['image_hdu'])
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_zscale_image(data, ax, 'gray')
    plt.show()

# %%
np.unique(df_merge['expnum']).shape

# %%
uni_exp.shape[0] / np.unique(df_merge['expnum']).shape[0]

# %%
