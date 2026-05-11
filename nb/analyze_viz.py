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
from decam_qa import reason_li, reason_num_dict, ccdnum2name, decode_reason, DECamImageDataset, read_embeddings

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
train_data, train_idx, train_label = read_embeddings(exp_dir / "node0/embeds_out")
train_embeds = np.vstack([np.mean(it, axis=0) for it in train_data])

# %%
len(train_idx)

# %%
21319 + len(train_idx)

# %%
train_idx = np.array(train_idx, dtype=int)
train_label = np.array(train_label, dtype=int)

# %%
np.unique(train_label, return_counts=True, )

# %%
# test_data, test_idx, test_label = read_embeddings(test_dir)
# test_embeds = np.vstack([np.mean(it, axis=0) for it in test_data])

# %%
# test_idx = np.array(test_idx, dtype=int)
# test_label = np.array(test_label, dtype=int)

# %%
train_embeds.shape

# %%
# test_embeds.shape

# %%
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

# %%
from sklearn.pipeline import Pipeline
import joblib

pipepath = Path("/global/u1/b/brookluo/decam-exposure-quality/postproc/knn_pipe.pkl")
pipe = joblib.load(pipepath)

# %%
pipe.steps

# %%
# trans_all = np.vstack([train_embeds, test_embeds])
trans_all = train_embeds
# fit everything but the classifier
for i in range(3):
    trans_all = pipe.steps[i][1].transform(trans_all)

# %%
trans_train = trans_all[:len(train_embeds)]
# trans_test = trans_all[len(train_embeds):]

# %%
# # %%time
# # scale data
# # this will do a column-wise standardization
# # # ! One might consider RobustScaler if outliers are too many
# # arr_rgb = StandardScaler().fit_transform(arr_rgb)
# # arr_ir = StandardScaler().fit_transform(arr_ir)
# single_arr = StandardScaler().fit_transform(train_embeds)

# prop = 1

# pca_components = 50
# n_clusters = 20
# percent = prop * 100

# # # %%time
# # pca = PCA(n_components='mle', svd_solver='full')
# #pca = PCA(n_components=0.99999, svd_solver='full')
# # question: using one PCA for all or each image has its separate PCA?
# pca = PCA(n_components=pca_components, svd_solver='auto')
# # pca.fit(arr_img)

# # there might be some convergence problem
# trans_all = pca.fit_transform(single_arr)
# # np.save(embeds_dir / f"dimred_pca_samp_{int(percent)}per_pca{pca_components}", trans_all)

# %%
# %%time
# with joblib.parallel_backend(backend="threads", n_jobs=10):
    # tsne_img = TSNE(n_componembeds_dirts=2, learning_rate='auto', init='pca', perplexity=50).fit_transform(trans_img)
    # tsne_spec = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50).fit_transform(trans_spec)
# 2D
tsne_all = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50, n_jobs=-1).fit_transform(trans_train)
np.save(embeds_dir / "tsne_2D_reduction_train.npy", tsne_all)
# 3D
# tsne_all = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=50, n_jobs=15).fit_transform(trans_train)
# np.save(embeds_dir / "tsne_3D_reduction_train.npy", tsne_all)
# np.save(embeds_dir / f"tsne_samp_{int(percent)}per_pca{pca_components}", tsne_all)

# %%
# Load the data from last step
tsne_all = np.load(embeds_dir / "tsne_2D_reduction_train.npy")

# %%
# idx = np.array(train_idx, dtype=int)
# label = np.array(train_label, dtype=int)

sampdir = Path("/pscratch/sd/b/brookluo/decam-exposure/revision/eval/embeds_out/samp_51")
idx = np.load(sampdir / "idx.npy")
label = np.load(sampdir / "label.npy").astype(int)
train_embeds = np.load(sampdir / "embeds.npy")
tsne_all = np.load(sampdir / "tsne_2D_reduction_train.npy")
eval_dir = sampdir
plot_dir = eval_dir / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
# dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
# dr10_imdir = dr10_dir / "images"

# dataset = DECamImageDataset("/global/u1/b/brookluo/decam-exposure-quality/data/samples/train_supervised_ooi_dataset.csv",
#                             image_dir=dr10_imdir, seed=0)

# all_labels = []
# for i in idx:
#     reasons = decode_reason(dataset.df_data['reasons'].iloc[i], return_num=True)
#     if len(reasons) == 0:
#         lab = 0
#     else:
#         lab = reasons[0] + 1
#     all_labels.append(lab)
# # all_labels = np.array([decode_reason(dataset.df_data['reasons'].iloc[i], return_num=True)[0] for i in idx])
# all_labels = np.array(all_labels)

# %%
np.unique(label)

# %% [markdown]
# ## Make interactive plot

# %%
import plotly.express as px
import plotly.graph_objects as go

# import pandas as pd

# df = pd.DataFrame(tsne_all, columns=["good", *reason_li[1:]])

# fig = px.scatter_3d(x=tsne_all[label==0, 0], y=tsne_all[label==0, 1], z=tsne_all[label==0, 2],
#         marker=dict(
#         size=2,
#         # color=z,                # set color to an array/list of desired values
#         # colorscale='Viridis',   # choose a colorscale
#         # opacity=0.8
#     ))
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=tsne_all[label==0, 0],
    y=tsne_all[label==0, 1],
    z=tsne_all[label==0, 2],
    mode='markers',
    marker=dict(
        size=2,
        # color="#1f77b4",
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.9
    ),
    name='good'
))

for i, rea in enumerate(reason_li, start=1):
    if sum(label==i) == 0:
        continue
    # ax[num].scatter(tsne_all[label!=i, 0], tsne_all[label!=i, 1], s=2, c="gray", alpha=0.5)
    # ax[num].scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], s=2, label="good", c='tab:blue')
    # ax[num].scatter(tsne_all[label==i, 0], tsne_all[label==i, 1], s=2, label=rea, c='tab:orange')
    # ax[num].set_aspect("equal")
    # ax[num].set_xlabel("tSNE axis-1")
    # ax[num].set_ylabel("tSNE axis-2")
    fig.add_trace(go.Scatter3d(
        x=tsne_all[label==i, 0],
        y=tsne_all[label==i, 1],
        z=tsne_all[label==i, 2],
        mode='markers',
        marker=dict(
            size=2,
            # color="#ff7f0e",
            # color=z,                # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.9
        ),
        name=rea
    ))
# i = 5
# rea = reason_li[6]
# fig.add_trace(go.Scatter3d(
#     x=tsne_all[label==i, 0],
#     y=tsne_all[label==i, 1],
#     z=tsne_all[label==i, 2],
#     mode='markers',
#     marker=dict(
#         size=2,
#         color="#ff7f0e",
#         # color=z,                # set color to an array/list of desired values
#         # colorscale='Viridis',   # choose a colorscale
#         opacity=0.9
#     ),
#     name=rea
# ))
# fig.update_layout(
#     xaxis=dict(visible=False),
#     yaxis=dict(visible=False),
#     # plot_bgcolor="rgba(0, 0, 0, 0)",
#     # paper_bgcolor="rgba(0, 0, 0, 0)",
# )

fig.update_layout(
    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )

# fig.write_html("all_3D_interactive.html")
# fig.show()

# %%
reason_li

# %% [markdown]
# ## Make 3D plot

# %%
# plt.figure(figsize=(15, 15))
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
# ax.scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], tsne_all[label==0, 2], s=2, label="good")
ax.scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], tsne_all[label==0, 2], s=2, label="good")
# for i, rea in enumerate(reason_li, start=1):
    # if sum(label==i) == 0:
        # continue
i = 6
rea = reason_li[i]
ax.scatter(tsne_all[label==i, 0], tsne_all[label==i, 1], tsne_all[label==i, 2], s=2, label=rea)
ax.set_aspect("equal")
ax.set_xlabel("tSNE axis-1")
ax.set_ylabel("tSNE axis-2")
ax.set_zlabel("tSNE axis-3")

ax.set_xlim([-20, 25])
ax.set_ylim([-20, 25])
ax.set_zlim([-20, 20])
ax.set_aspect("equal")

plt.legend(loc="upper right")
    # lgnd = fig.legend(loc="upper right", scatterpoints=1, fontsize=10)
    # for hdl in lgnd.legendHandles:
    #     hdl._sizes = [30]
plt.savefig(plot_dir / f"3D_train_clustering.pdf", bbox_inches='tight')

# %%
# fig, ax = plt.subplots(2, 2, figsize=(15, 15))
# ax = ax.ravel()
# i = 5
# # front
# ax[0].scatter(tsne_all[label==0, 0], tsne_all[label==0, 2], s=2, label="good")
# ax[0].scatter(tsne_all[label==i, 0], tsne_all[label==i, 2], s=2, label="Ghost_Scatter")
# ax[0].set_xlabel("tSNE axis-1")
# # ax[0].set_ylabel("tSNE axis-2")
# ax[0].set_ylabel("tSNE axis-3")
# # right
# ax[1].scatter(tsne_all[label==0, 1], tsne_all[label==0, 2], s=2, label="good")
# ax[1].scatter(tsne_all[label==i, 1], tsne_all[label==i, 2], s=2, label="Ghost_Scatter")
# # ax[0].set_xlabel("tSNE axis-1")
# ax[1].set_xlabel("tSNE axis-2")
# ax[1].set_ylabel("tSNE axis-3")
# # top
# ax[2].scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], s=2, label="good")
# ax[2].scatter(tsne_all[label==i, 0], tsne_all[label==i, 1], s=2, label="Ghost_Scatter") 
# ax[2].set_xlabel("tSNE axis-1")
# ax[2].set_ylabel("tSNE axis-2")

# ax[3].set_visible(False)
# plt.savefig(plot_dir / f"subplots_train_clustering.pdf", bbox_inches='tight')

# %% [markdown]
# ## Make 2D plot

# %%
# for revision: randomly select 20 per good image and then replot


# %%
fig, ax = plt.subplots(4, 3, figsize=(18, 25))
ax = ax.ravel()
num = 0
i = 0
ax[num].scatter(tsne_all[label!=i, 0], tsne_all[label!=i, 1], s=2, c="gray", alpha=0.5)
ax[num].scatter(tsne_all[label==i, 0], tsne_all[label==i, 1], s=2, label="good", c='tab:blue')
ax[num].set_aspect("equal")
# ax[num].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# ax[num].set_xlabel("tSNE axis-1")
# ax[num].set_ylabel("tSNE axis-2")
# lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
# for hdl in lgnd.legend_handles:
#     hdl._sizes = [30]
ax[num].set_title("Good", fontsize=20)
ax[num].tick_params(axis='both', labelsize=16)
num += 1
for i, rea in enumerate(reason_li, start=1):
    if sum(label==i) == 0:
        continue
    ax[num].scatter(tsne_all[label!=i, 0], tsne_all[label!=i, 1], s=2, c="gray", alpha=0.5)
    ax[num].scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], s=2, label="good", c='tab:blue')
    ax[num].scatter(tsne_all[label==i, 0], tsne_all[label==i, 1], s=2, label=rea, c='tab:orange')
    ax[num].set_aspect("equal")
    # ax[num].set_xlabel("tSNE axis-1")
    # ax[num].set_ylabel("tSNE axis-2")
    # plt.legend(loc="upper right")
    # lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
    # for hdl in lgnd.legend_handles:
    #     hdl._sizes = [30]
    ax[num].set_title(rea, fontsize=20)
    ax[num].tick_params(axis='both', labelsize=16)
    num += 1
    # plt.show()
# ax[-1].axis("off")
# ax[-2].axis("off")
plt.savefig(plot_dir / f"subplots_clustering.png", bbox_inches='tight')

# %%
plt.figure(figsize=(15, 15))
plt.scatter(tsne_all[label==0, 0], tsne_all[label==0, 1], s=2, label="good", zorder=100)
plt.scatter(tsne_all[label!=0, 0], tsne_all[label!=0, 1], s=2, label="bad")
plt.gca().set_aspect("equal")
plt.xlabel("tSNE axis-1")
plt.ylabel("tSNE axis-2")
# plt.legend(loc="upper right")
lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
for hdl in lgnd.legend_handles:
    hdl._sizes = [30]
plt.savefig(plot_dir / f"train_binary_clustering.png", bbox_inches='tight')

# %% [markdown]
# ## Classification evaluation

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, classification_report

# %%
# X_train, X_test, y_train, y_test = train_test_split(trans_all,
#                                                     label,
#                                                     test_size=0.33, random_state=42)

# %%
single_arr = np.vstack([train_embeds, test_embeds])

# %%
# knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
# pca = PCA(n_components=pca_components, svd_solver='auto')
# trans_all = pca.fit_transform(StandardScaler().fit_transform(single_arr))
# tsne_all = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50, n_jobs=-1).fit_transform(trans_all)

# %%
X_train = single_arr[:len(train_embeds)]
X_test = single_arr[len(train_embeds):]
y_train = np.array(train_label, dtype=int)
y_test = np.array(test_label, dtype=int)

# %%
# knn.fit(X_train, y_train)
pipe.fit(X_train, y_train)

# %%
idx = np.array(test_idx, dtype=int)
label = np.array(test_label, dtype=int)

# %%
# y_pred_prob = knn.predict_proba(X_test)
# y_pred_label = knn.predict(X_test)
y_pred_prob = pipe.predict_proba(X_test)
y_pred_label = pipe.predict(X_test)

# %%
# y_pred_label = y_pred_label.astype(int)

# %%
prob_thresh = 0.9

# %%
np.sum(y_pred_prob > prob_thresh)

# %%
y_cut = np.zeros(y_pred_label.shape, dtype=bool)

map_back = dict(zip(np.unique(y_pred_label), np.arange(0, len(np.unique(y_pred_label))+1)))

for i, idx in enumerate(y_pred_label):
    idx = map_back[idx]
    if y_pred_prob[i, idx] > prob_thresh:
        y_cut[i] = True

# %%
X_test_cut = X_test[y_cut]
y_test_cut = y_test[y_cut]
y_pred_cut = y_pred_label[y_cut]

# %%
reason_li = ["good"] + list(reason_li[1:])
# reason_li = np.array(reason_li)[np.unique(label)]
# ooi_counts = []
# for i, rea in enumerate(reason_li):
#     ooi_counts.append(np.sum(df['reasons'] & 2**reason_dict[rea] > 0))
#     print(i, rea, np.sum(df['reasons'] & 2**reason_dict[rea] > 0))
# print("Total images:", len(df))
fig, ax1 = plt.subplots()
y_mod = y_test-1
y_mod[y_mod < 0] = 0
ax1.hist(y_mod, bins=np.arange(0, len(reason_li)+1), label='accepted', align='left')
ax2 = ax1.twinx()
ax1.set_xticks(np.arange(0, len(reason_li)))
ax1.set_xticklabels(reason_li, rotation=90)
# plt.xticks(rotation=90)
ax1.set_xlabel("Category number")
ax1.set_ylabel("counts")
num, counts = np.unique(y_mod, return_counts=True)
ax2.bar(num, counts / len(y_test), align='edge', alpha=0)
ax2.set_ylabel("percentage")

ax1.hist(y_mod[~y_cut], bins=np.arange(0, len(reason_li)+1), label='rejected',
         histtype='stepfilled', align='left')

ax1.legend()
plt.show()

print("Remaining ratio:", sum(y_cut) / len(y_cut))

# %%
pick = y_test_cut == 0
# score = knn.score(X_test_cut[pick], y_test_cut[pick])
score = pipe.score(X_test_cut[pick], y_test_cut[pick])
num = 0
label_name = ["0_good"]
print(num, "good", score)

# %%
sum(label == i)

# %%
pick

# %%
pick = y_test_cut == 0
# score = knn.score(X_test_cut[pick], y_test_cut[pick])
score = pipe.score(X_test_cut[pick], y_test_cut[pick])
num = 0
label_name = ["0_good"]
print(num, "good", score)
for i, rea in enumerate(reason_li, start=1):
    if sum(label==i) == 0:
        continue
    num += 1
    pick = y_test_cut == i
    # score = knn.score(X_test_cut[pick], y_test_cut[pick])
    score = pipe.score(X_test_cut[pick], y_test_cut[pick])
    print(num, rea, score)
    label_name.append(f"{num}_{rea}")

# %%
class_name = [l.split("_", 1)[-1] for l in label_name]

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_true=y_test_cut, y_pred=y_pred_cut)
cm_norm = np.round(cm/cm.sum(axis=1)[:, np.newaxis], 2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                              display_labels=[l.split("_", 1)[-1] for l in label_name],
                             )
disp.plot(ax=ax, cmap="rocket")
plt.xticks(rotation=90)
plt.xlabel('Predicted label', fontsize=18)
plt.ylabel('True label', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.savefig(plot_dir / f"{prob_thresh}_pcut_vit_base_confusion_pretrain.png", bbox_inches='tight')
print("vit_base with resizing images")
print(classification_report(y_test_cut, y_pred_cut, target_names=label_name))

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true=y_test_cut, y_pred=y_pred_cut)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[l.split("_", 1)[-1] for l in label_name])
disp.plot()
plt.xticks(rotation=90)
# plt.savefig(plot_dir / f"vit_base_confusion_pretrain.png")
print("vit_base with resizing images")
print(classification_report(y_test_cut, y_pred_cut, target_names=label_name))

# %%
np.unique(y_pred_cut)

# %%
test_idx

# %%
import pandas as pd

# %%
df_test = pd.read_csv("/global/u1/b/brookluo/decam-exposure-quality/data/samples/test_supervised_ooi_dataset.csv")

# %%
df_train = pd.read_csv("/global/u1/b/brookluo/decam-exposure-quality/data/samples/train_supervised_ooi_dataset.csv")

# %%
all_reasons = df_test.iloc[test_idx[y_cut][y_pred_cut == 15]]['reasons'].apply(decode_reason)

# %%
df_test.iloc[test_idx[y_cut][y_pred_cut == 15]].query("expnum == 468244")

# %%
np.sum(y_test_cut == 0)

# %%
np.sum(pick_fp)

# %%
oneidx = 468244
df_train.query(f"expnum == {oneidx}"), df_test.query(f"expnum == {oneidx}"), 

# %%
df_train.query("expnum == 468232"), df_test.query("expnum == 468232"), 

# %%
all_reasons

# %%
# single_arr = np.vstack([train_embeds, test_embeds])
# pca = PCA(n_components=pca_components, svd_solver='auto')
# trans_all = pca.fit_transform(StandardScaler().fit_transform(single_arr))
# tsne_all = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50, n_jobs=-1).fit_transform(trans_all)

# %%
idx_all = np.hstack([train_idx, test_idx], dtype=int)
label_all = np.hstack([train_label, test_label], dtype=int)

# %%
tsne_train = tsne_all[:len(train_embeds)]
tsne_test = tsne_all[len(train_embeds):]

# %%
plot_dir

# %%
tsne_all.shape

# %%
label_all.shape

# %%
idx_all.shape

# %%
fig, ax = plt.subplots(4, 3, figsize=(25, 35))
ax = ax.ravel()
num = 0
i = 0
ax[num].scatter(tsne_all[label_all!=i, 0], tsne_all[label_all!=i, 1], s=2, c="gray", alpha=0.5)
ax[num].scatter(tsne_all[label_all==i, 0], tsne_all[label_all==i, 1], s=2, label="train", c='tab:orange', alpha=0.7)
ax[num].scatter(tsne_test[(test_label==i), 0], tsne_test[(test_label==i), 1], s=2, label="test", c='tab:blue')
ax[num].scatter(tsne_test[(test_label==i) & y_cut, 0], tsne_test[(test_label==i) & y_cut, 1], s=2, label="test & prob cut", c='tab:green')
ax[num].set_aspect("equal")
ax[num].set_xlabel("tSNE axis-1")
ax[num].set_ylabel("tSNE axis-2")
ax[num].set_title("good", fontsize=15)
lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
for hdl in lgnd.legendHandles:
    hdl._sizes = [30]
num += 1
for i, rea in enumerate(reason_li, start=1):
    if sum(label==i) == 0:
        continue
    ax[num].scatter(tsne_all[label_all!=i, 0], tsne_all[label_all!=i, 1], s=2, c="gray", alpha=0.5)
    ax[num].scatter(tsne_all[label_all==i, 0], tsne_all[label_all==i, 1], s=2, label="train", c='tab:orange', alpha=0.7)
    ax[num].scatter(tsne_test[(test_label==i), 0], tsne_test[(test_label==i), 1], s=2, label="test", c='tab:blue')
    ax[num].scatter(tsne_test[(test_label==i) & y_cut, 0], tsne_test[(test_label==i) & y_cut, 1], s=2, label="test & prob cut", c='tab:green')
    ax[num].set_title(rea, fontsize=15)
    ax[num].set_aspect("equal")
    ax[num].set_xlabel("tSNE axis-1")
    ax[num].set_ylabel("tSNE axis-2")
    # plt.legend(loc="upper right")
    lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
    for hdl in lgnd.legendHandles:
        hdl._sizes = [30]
    num += 1
    # plt.show()
# ax[-1].axis("off")
# ax[-2].axis("off")
# plt.savefig(plot_dir / f"train_test_prob_subplots_pca{pca_components}_cluster{n_clusters}_percent{percent}_clustering.png", bbox_inches='tight')

# %%
# # now check miss classified 0 vs. 5 (7) (good vs. ghost_scatter case)
# # reason_num_dict['Ghost_Scatter'] == 6 (+1)
# miss_class_0 = (y_test == 7) & (y_pred == 0)
# miss_class_5 = (y_test == 0) & (y_pred == 7)
# idx_miss_0 = np.array(test_idx, dtype=int)[miss_class_0]
# idx_miss_5 = np.array(test_idx, dtype=int)[miss_class_5]

import pandas as pd

dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
dr10_imdir = dr10_dir / "images"

df_test = pd.read_csv("/global/u1/b/brookluo/decam-exposure-quality/data/samples/test_supervised_ooi_dataset.csv")

# %%
pick_fp = (y_test_cut == 0) & (y_pred_cut > 0)
pick_match = (y_test_cut == y_pred_cut)
# pick_fp = pick_match

# %%
sum(pick_fp)

# %%
len(y_test_cut)

# %%
import fitsio

# %%
# i = df_test.query("expnum == 453898").iloc[0].index

idx = np.array(test_idx, dtype=int)
label = np.array(test_label, dtype=int)

# row = df_test.iloc[idx[y_cut][pick_fp][i]]
row = df_test.query("expnum == 453898").iloc[0]
exp = row.expnum
ccdnum = row.ccdnum
ccdname = ccdnum2name[ccdnum]
expheader = fitsio.read_header(dr10_imdir / row["image_filename"])
img, imgheader = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"], header=True)
fig, ax = plt.subplots(figsize=(6, 8))
plot_zscale_image(img, ax, 'gray')
ax.set_title(f"{exp} {ccdname}\n"
    f"Exptime: {expheader['EXPTIME']}, Filter: {expheader['FILTER'][:1]}\n"
            f"Gain: ({imgheader['GAINA']:.4f}, {imgheader['GAINB']:.4f})\n"
            f"Read Noise: ({imgheader['RDNOISEA']:.4f}, {imgheader['RDNOISEB']:.4f})")
print(imgheader['RDNOISEA'], imgheader['RDNOISEB'])
print(imgheader['GAINA'], imgheader['GAINB'])
print(expheader['EXPTIME'], expheader['FILTER'][:1])
# print(reason_li[y_pred_cut[pick_fp][i]-1])
print(decode_reason(row['reasons']))

# %%
idx_cut_match = idx_cut[pick_match]

# %%
y_cut_match = y_test_cut[pick_match]

# %%
all_match_pick = [np.where(y_cut_match == i)[0]  for i in np.unique(y_cut_match)]

# %%
rng = np.random.default_rng(42)
all_match_pick_100 = [rng.choice(arr, size=100, replace=False) if len(arr) > 100 else arr for arr in all_match_pick]

# %%
# idx = np.array(test_idx, dtype=int)
# label = np.array(test_label, dtype=int)
# for arr in all_match_pick_100:
#     for i in arr:
#         row = df_test.iloc[idx_cut_match[i]]
#         exp = row.expnum
#         ccdnum = row.ccdnum
#         ccdname = ccdnum2name[ccdnum]
#         all_reason = " & ".join(decode_reason(row.reasons))
#         all_source = " & ".join(decode_vi_source(row.vi_source))
#         if y_pred_cut[pick_fp][i] == 0:
#             reason = 'good'
#             all_reason_source = 'good'
#         else:
#             reason = reason_li[y_cut_match[i]-1]
#             all_reason_source = f"{all_reason} by {all_source}"
#         dir_rea = Path("/global/u1/b/brookluo/decam-exposure-quality/matched_images", reason)
#         dir_rea.mkdir(exist_ok=True, parents=True)
#         # print(row.vi_source, all_source)
#         expheader = fitsio.read_header(dr10_imdir / row["image_filename"])
#         img, imgheader = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"], header=True)
#         fig, ax = plt.subplots(figsize=(6, 8))
#         plot_zscale_image(img, ax, 'gray')
#         ax.set_title(f"{exp} {ccdname}\n"
#                      f"Exptime: {expheader['EXPTIME']}, Filter: {expheader['FILTER'][:1]}\n"
#                     f"Gain: ({imgheader['GAINA']:.4f}, {imgheader['GAINB']:.4f})\n"
#                     f"Read Noise: ({imgheader['RDNOISEA']:.4f}, {imgheader['RDNOISEB']:.4f})")
#         # img = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"])
#         # fig, ax = plt.subplots(figsize=(6, 8))
#         # plot_zscale_image(img, ax, 'gray')
#         # plt.title(f"pred class: {reason}, prob: {max(y_pred_prob[y_cut][pick_fp][i])}\n"
#         #          f"{exp} {ccdname} {all_reason_source}")
#         fname = f"{exp}_{ccdname}.png"
#         plt.savefig(dir_rea / fname)
#         plt.close()


# %%
dir_path = Path("/global/u1/b/brookluo/decam-exposure-quality/false_negative_dr10cut")

# %%
idx = np.array(test_idx, dtype=int)
label = np.array(test_label, dtype=int)
idx_cut = idx[y_cut]
for i in range(len(idx_cut[pick_fp])):
    row = df_test.iloc[idx_cut[pick_fp][i]]
    exp = row.expnum
    ccdnum = row.ccdnum
    ccdname = ccdnum2name[ccdnum]
    # all_reason = " & ".join(decode_reason(row.reasons))
    # all_source = " & ".join(decode_vi_source(row.vi_source))
    reason = reason_li[y_pred_cut[pick_fp][i]-1]
    # all_reason_source = f"{all_reason} by {all_source}"
    dir_rea = Path(dir_path, reason)
    dir_rea.mkdir(exist_ok=True, parents=True)
    # print(row.vi_source, all_source)
    expheader = fitsio.read_header(dr10_imdir / row["image_filename"])
    img, imgheader = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"], header=True)
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_zscale_image(img, ax, 'gray')
    ax.set_title(f"{exp} {ccdname}\n"
                 f"pred class: {reason}, prob: {max(y_pred_prob[y_cut][pick_fp][i])}\n"
                 f"Exptime: {expheader['EXPTIME']}, Filter: {expheader['FILTER'][:1]}\n"
                # f"Gain: ({imgheader['GAINA']:.4f}, {imgheader['GAINB']:.4f})\n"
                f"Read Noise: ({imgheader['RDNOISEA']:.4f}, {imgheader['RDNOISEB']:.4f})")
    fname = f"{exp}_{ccdname}.png"
    plt.savefig(dir_rea / fname)
    plt.close()


# %%
import os

# %%
html = [
    "<!DOCTYPE html>",
    "<html>",
        "<head>",
            # "<title>Expert Model matched images</title>",
            "<title>Images missed by experts</title>",
        "</head>",
        "<body>"
]

end = ["</body>", "</html>"]
# remember to include </body> </html> in the end!

dirpath = Path("/global/u1/b/brookluo/decam-exposure-quality/false_negative_dr10")

for onedir in dirpath.glob("*"):
    body = ["<table>"]
    if not onedir.is_dir():
        continue
    for i, im_name in enumerate(onedir.glob("*")):
        if i % 5 == 0:
            body.append("<tr>")
        body.append(f"<td><img src=\"{im_name.parent.name}/{im_name.name}\"></td>")
    body.append("</table>")
    with open(dirpath / f"{onedir.name}.html", "w") as fp:
        fp.write("\n".join(html + body + end))

# for i, (idx, row) in enumerate(df_test.iloc[idx[y_cut][pick_fp]].iterrows()):
#     exp = row.expnum
#     ccdnum = row.ccdnum
#     ccdname = ccdnum2name[ccdnum]
#     fname = f"{exp}_{ccdname}.png"
#     if i % 5 == 0:
#         body.append("<tr>")
#     body.append(f"<td><img src=\"{dirpath.name}/{fname}\"></td>")
# body.append("</table>")
# with open(dirpath.parent / f"{dirpath.name}.html", "w") as fp:
#     fp.write("\n".join(html + body + end))


# This list of images are matched between human expert's label and ML model's label.
index = '''<!DOCTYPE html>
<html>
    <body>
        This is the list of images considered as "good" and used in DR10 but flagged as bad by ML model.
        <ol>
            {list}
        </ol>
    </body>
</html>
'''

all_reason = [f'<li><a href="./{r}.html">{r}</a></li>' for r in (["good"] + list(reason_li)) if r in os.listdir(dirpath)]
with open(dirpath / "index.html", "w") as fp:
    fp.write(index.format(list="\n".join(all_reason)))

# %%
# import fitsio

# miss_0as5_dir = plot_dir / "misclass-good-as-ghost_scatter"
# miss_0as5_dir.mkdir(exist_ok=True)
# miss_5as0_dir = plot_dir / "misclass-ghost_scatter-as-good"
# miss_5as0_dir.mkdir(exist_ok=True)

# prob_test = knn.predict_proba(X_test)

# for i in range(len(idx_miss_0)):
#     row = df_test.loc[idx_miss_0[i]]
#     exp = row.expnum
#     ccdnum = row.ccdnum
#     ccdname = ccdnum2name[ccdnum]
#     all_reason = " & ".join(decode_reason(row.reasons))
#     all_source = " & ".join(decode_vi_source(row.vi_source))
#     img = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"])
#     fig, ax = plt.subplots(figsize=(6, 8))
#     plot_zscale_image(img, ax, 'gray')
#     plt.title(f"true class: Ghost_scatter, {prob_test[miss_class_0][i, 5]}\n"
#               f"pred class: good, {prob_test[miss_class_0][i, 0]}\n"
#              f"{exp} {ccdname} {all_reason} by {all_source}")
#     fname = f"{exp}_{ccdname}.png"
#     plt.savefig(miss_5as0_dir / fname)
#     plt.close()

# for i in range(len(idx_miss_5)):
#     row = df_test.loc[idx_miss_5[i]]
#     exp = row.expnum
#     ccdnum = row.ccdnum
#     ccdname = ccdnum2name[ccdnum]
#     all_reason = " & ".join(decode_reason(row.reasons))
#     all_source = " & ".join(decode_vi_source(row.vi_source))
#     img = fitsio.read(dr10_imdir / row["image_filename"], ext=row["image_hdu"])
#     fig, ax = plt.subplots(figsize=(6, 8))
#     plot_zscale_image(img, ax, 'gray')
#     plt.title(
#         f"true class: good, {prob_test[miss_class_5][i, 0]}\n"
#         f"pred class: Ghost_scatter, {prob_test[miss_class_5][i, 5]}\n"
#         f"{exp} {ccdname}")
#     fname = f"{exp}_{ccdname}.png"
#     plt.savefig(miss_0as5_dir / fname)
#     plt.close()

# html = [
#     "<!DOCTYPE html>",
#     "<html>",
#         "<head>",
#             "<title>Misclassify {truth} as {pred}</title>",
#         "</head>",
#         "<body>"
# ]

# end = ["</body>", "</html>"]
# # remember to include </body> </html> in the end!

# def write_website(df, dirpath, truth, pred):
#     body = ["<table>"]
#     for i, (idx, row) in enumerate(df.iterrows()):
#         exp = row.expnum
#         ccdnum = row.ccdnum
#         ccdname = ccdnum2name[ccdnum]
#         fname = f"{exp}_{ccdname}.png"
#         if i % 5 == 0:
#             body.append("<tr>")
#         body.append(f"<td><img src=\"{dirpath.name}/{fname}\"></td>")
#     body.append("</table>")
#     with open(plot_dir / f"{dirpath.name}.html", "w") as fp:
#         fp.write("\n".join(html + body + end).format(truth=truth, pred=pred))

# write_website(df_test.loc[idx_miss_5].sort_values("expnum"), miss_0as5_dir, "good", "ghost_scatter")
# write_website(df_test.loc[idx_miss_0].sort_values("expnum"), miss_5as0_dir, "ghost_scatter", "good")

# index = '''<!DOCTYPE html>
# <html>
#     <body>
#         This is the list of misclassified images.
#         <ol>
#             {list}
#         </ol>
#     </body>
# </html>
# '''

# all_reason = [f'<li><a href="./{r}.html">{r}</a></li>' for r in reason_expnum_fp.keys()]
# with open(f"../images/index.html", "w") as fp:
#     fp.write(index.format(list="\n".join(all_reason)))

# %%
