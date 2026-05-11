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
import decam_info
from decam_dataset import DECamImageDataset

# %%
sys.path.append("../../img-spec-ml/src/")
from plot_utils import plot_zscale_image

# %%
import matplotlib.pyplot as plt
from pathlib import Path

# %%
import h5py
import numpy as np

# %%
import os

# %%
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure")
exp_name = "dino_v2"
exp_dir = root_dir / exp_name
eval_dir = exp_dir / "eval"
# data_dir = root_dir / "data"
ckpt_dir = exp_dir / "checkpoint"
# test_dir = data_dir / "test"

embeds_dir = eval_dir / "embeds_out"

model = "base"
model_dir = embeds_dir / model
train_dir = model_dir / "train"
test_dir = model_dir / "test"
    
plot_dir = eval_dir / "plots"


# %%
def read_embeds(outdir, num=4):
    data = []
    idx = []
    label = []
    for i in range(num):
        with h5py.File(outdir / f"{i}_worker_embeds.h5", 'r') as h5f:
            dset = h5f["images"]
            for it in dset:
                data.append(np.array(dset[it]))
                names = it.split("_")
                idx.append(names[1])
                label.append(names[-1])
    return data, idx, label


# %%
# train_data, train_idx, train_label = read_embeds(train_dir)
# train_embeds = np.vstack([np.mean(it, axis=0) for it in train_data])

# idx = np.array(train_idx, dtype=int)
# label = np.array(train_label, dtype=int)

sampdir = Path("/pscratch/sd/b/brookluo/decam-exposure/revision/eval/embeds_out/samp_55")
idx = np.load(sampdir / "idx.npy")
label = np.load(sampdir / "label.npy")
train_embeds = np.load(sampdir / "embeds.npy")
tsne_all = np.load(sampdir / "tsne_2D_reduction_train.npy")

# %%
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

# %%
single_arr = StandardScaler().fit_transform(train_embeds)

# %%
pca_components = 100
n_clusters = 20
# percent = prop * 100

# %%
# # %%time
# pca = PCA(n_components='mle', svd_solver='full')
#pca = PCA(n_components=0.99999, svd_solver='full')
# question: using one PCA for all or each image has its separate PCA?
pca = PCA(n_components=pca_components, svd_solver='auto')
# pca.fit(arr_img)

# %%
trans_all = pca.fit_transform(single_arr)

# %%
from sklearn.cluster import HDBSCAN, DBSCAN

# %%
hdb = HDBSCAN(min_cluster_size=5, max_cluster_size=500, n_jobs=20)

# %%
hdb.fit(trans_all)

# %%
# tsne_all = np.load(train_dir / f"tsne_samp_{int(percent)}per_pca{pca_components}.npy")

# %%
plt.figure(figsize=(15, 15))
for i in np.unique(hdb.labels_):
    plt.scatter(tsne_all[hdb.labels_==i, 0], tsne_all[hdb.labels_==i, 1], s=2, label=i)
plt.gca().set_aspect("equal")
plt.xlabel("tSNE axis-1")
plt.ylabel("tSNE axis-2")
# plt.legend(loc="upper right")
# lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
# for hdl in lgnd.legendHandles:
#     hdl._sizes = [30]
# plt.savefig(plot_dir / f"train_pca{pca_components}_cluster{n_clusters}_percent{percent}_clustering.png", bbox_inches='tight')

# %% [markdown]
# ## Do analysis with Silhouette score

# %%
from permetrics import ClusteringMetric

# %%
cm = ClusteringMetric(X=trans_all, y_pred=hdb.labels_)

# %%
cm.density_based_clustering_validation_index()

# %%
cm.silhouette_index()

# %%
from sklearn import metrics

# %%
metrics.silhouette_score(trans_all, hdb.labels_)
