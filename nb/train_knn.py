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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
from decam_qa import decode_ml_label, reason_li, ccdnum2name, decode_reason, decode_vi_source, read_embeddings

# %%
from astropy.table import Table
import fitsio

# %%
import h5py

# %%
root_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dino_v2")
exp_name = "base_test_good"
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
test_data, test_idx, test_label = read_embeddings(test_dir / "eval/embeds_out")
test_embeds = np.vstack(test_data) #[np.mean(it, axis=0) for it in test_data])

test_idx = np.array(test_idx, dtype=int)
test_label = np.array(test_label, dtype=int)

# %%
# load train data from last train
train_dir = Path("/pscratch/sd/b/brookluo/decam-exposure/dino_v2/base_resize_dr10cut/train")
train_data, train_idx, train_label = read_embeddings(train_dir / "eval/embeds_out")
train_embeds = np.vstack(train_data) #[np.mean(it, axis=0) for it in train_data])

# %%
# match with dr10 reject and dr10 accept
df_all = pd.read_csv("/global/u1/b/brookluo/decam-exposure-quality/data/samples/all_sample_decam_dr10_good_exp_ooi.csv")

# %%
dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
dr10_imdir = dr10_dir / "images"
dr10_tab = Table.read(dr10_dir / "survey-ccds-decam-dr10.fits.gz")

# %%
df_df10tab = dr10_tab.to_pandas()

# %%
merged = pd.merge(df_all, df_df10tab[['expnum', 'image_hdu', "ccd_cuts"]], on=['expnum', 'image_hdu'])

# %%
tested = merged.iloc[test_idx]

# %%
rejected = tested[tested['ccd_cuts'] > 0]
accepted = tested[tested['ccd_cuts'] == 0]

# %%
bad_ccd = pd.read_csv("/global/u1/b/brookluo/decam-exposure-quality/data/decam_dr10_bad_exp_all.csv")

# %% [markdown]
# # Do training

# %%
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, StandardScaler,\
                            MinMaxScaler, PowerTransformer, MaxAbsScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
from decam_qa.classifier import build_pipeline, train

# %%
train_embeds.shape

# %%
len(train_label)

# %%
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
# param_grid = {'leaf_size': np.arange(1, 100, 1),
#               'weights': ['uniform', 'distance']}
# base_estimator = KNeighborsClassifier()
# sh = HalvingRandomSearchCV(base_estimator, param_grid, cv=5,
#                          factor=2, resource='n_neighbors',
#                           n_jobs=10,
#                           max_resources=30,
#                           random_state=0).fit(train_embeds, train_label)
# sh.best_estimator_

# %%
pipe = build_pipeline()

parameters = {'scaler': [StandardScaler(), MinMaxScaler(),
	Normalizer(), MaxAbsScaler(), RobustScaler(), PowerTransformer()],
    'preprocessor__n_components': [5, 10, 15, 25, 30, 50, 70, 100],
	'selector__threshold': [0, 0.001, 0.01],
	'classifier__n_neighbors': [1, 3, 5, 7, 10, 20, 30],
	'classifier__p': [1, 2, 3, 5],
	'classifier__leaf_size': [1, 5, 10, 15, 30, 35]
}

pipe_best = train(pipe, train_embeds, train_label,
                  search_params=parameters, n_jobs=10)

# %%
print(pipe_best)

# %%
import joblib
joblib.dump(pipe_best, '/global/u1/b/brookluo/decam-exposure-quality/postproc/knn_pipe.pkl', compress=1)

# %%
# pipe_best = grid.best_estimator_  # preserved

# %%
scaler = StandardScaler().fit(train_embeds)
train_scale = scaler.transform(train_embeds)
test_scale = scaler.transform(test_embeds)

# %%
prop = 1

pca_components = 50
n_clusters = 20
percent = prop * 100

# %%
pca = PCA(n_components=pca_components, svd_solver='auto').fit(train_scale)

# %%
train_trans = pca.transform(train_scale)
test_trans = pca.transform(test_scale)

# %%
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)

# %%
np.unique(train_label)

# %%
knn.fit(train_trans, train_label)

# %%
y_pred_prob = knn.predict_proba(test_trans)

# %%
y_pred_prob = pipe_best.predict_proba(test_embeds)

# %%
labels = np.argmax(y_pred_prob, axis=1)
classes = pipe_best.classes_
labels = np.array([classes[i] for i in labels], dtype=int)
y_prob = np.max(y_pred_prob, axis=1)

# %%
pick_top = y_prob > 0.85

# %%
tp = np.array((labels > 0) & (tested["ccd_cuts"] > 0) & pick_top)
tn = np.array((labels == 0) & (tested["ccd_cuts"] == 0) & pick_top)
fp = np.array((labels > 0) & (tested["ccd_cuts"] == 0) & pick_top)
fn = np.array((labels == 0) & (tested["ccd_cuts"] > 0) & pick_top)

# %%
len(labels[fp])

# %%
len(labels[fp])

# %%
len(labels[fp]) / len(tested)

# %%
sum(tp)

# %%
print([sum(x) for x in [tp, tn, fp, fn]])

# %%
labels.shape

# %%
tested['ml_label'] = labels
tested['ml_proba'] = y_prob

# %%
# wite to results
tested.to_csv(eval_dir / "dr10cut_ml_inference_output.csv", index=False)

# %%
tested = pd.read_csv(eval_dir / "dr10cut_ml_inference_output.csv")

# %%
np.unique(decode_ml_label(tested["ml_label"]))

# %%
prob_thres = 0.7
top_num = 200

# %%
plot_idx = np.hstack([tested.query("ml_label == @i & ml_proba > @prob_thres & ccd_cuts == 0")
            .sort_values("ml_proba", ascending=False)
            .index[:top_num]
           for i in tested['ml_label'].unique()
           ])

# %%
# %%
from plot_utils import plot_zscale_image

def plot_ccd_images(outdir: Path, imdir: Path, df_img, idx):
    count = 0
    for _, row in df_img.loc[idx].iterrows():
        # row = df_img.loc[i]
        exp = row.expnum
        ccdnum = row.ccdnum
        ccdname = ccdnum2name[ccdnum]
        # all_reason = " & ".join(decode_reason(row.reasons))
        # all_source = " & ".join(decode_vi_source(row.vi_source))
        reason = reason_li[row["ml_label"]-1]
        # all_reason_source = f"{all_reason} by {all_source}"
        dir_rea = Path(outdir, reason)
        dir_rea.mkdir(exist_ok=True, parents=True)
        # print(row.vi_source, all_source)
        expheader = fitsio.read_header(imdir / row["image_filename"])
        img, imgheader = fitsio.read(imdir / row["image_filename"], ext=row["image_hdu"], header=True)
        fig, ax = plt.subplots(figsize=(6, 8))
        plot_zscale_image(img, ax, 'gray')
        ax.set_title(f"{exp} {ccdname}\n"
                     f"pred class: {reason}, prob: {row['ml_proba']}\n"
                     f"Exptime: {expheader['EXPTIME']}, Filter: {expheader['FILTER'][:1]}\n"
                    # f"Gain: ({imgheader['GAINA']:.4f}, {imgheader['GAINB']:.4f})\n"
                    f"Read Noise: ({imgheader['RDNOISEA']:.4f}, {imgheader['RDNOISEB']:.4f})")
        fname = f"{exp}_{ccdname}.png"
        plt.savefig(dir_rea / fname)
        plt.close()
        del img
        count += 1
        if count % 100 == 0:
            print(count)


# %%
outdir = Path("/pscratch/sd/b/brookluo/decam-exposure/dino_v2/base_test_good/plot")

# %%
plot_ccd_images(outdir,
               dr10_imdir,
               tested,
               plot_idx[1600:])

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

dirpath = outdir

for onedir in dirpath.glob("*"):
    body = ["<table>"]
    if not onedir.is_dir():
        continue
    all_files = list(onedir.glob("*"))
    all_files = list(onedir.glob("*"))
    expnum, ccdname = list(zip(*[f.name.split("_") for f in all_files]))
    expnum = np.array(expnum, dtype=int)
    exp_sort_idx = np.argsort(expnum)
    for i, idx in enumerate(exp_sort_idx):
        if i % 5 == 0:
            body.append("<tr>")
        im_name = f"{expnum[idx]}_{ccdname[idx]}"
        body.append(f"<td><img src=\"{onedir.name}/{im_name}\"></td>")
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

# %%
