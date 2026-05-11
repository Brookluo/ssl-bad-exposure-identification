import sys
from os import pipe
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sympy import reduced

import decam_info
from decam_dataset import DECamImageDataset
from inference import read_embeds

# sys.path.append("/global/homes/b/brookluo/.local/perlmutter/pytorch2.6.0/lib/python3.12/site-packages")


def combine_data(h5embeds_dir, output_dir):
    data, idx, label = read_embeds(h5embeds_dir)
    embeds = np.vstack([np.mean(it, axis=0) for it in data])
    np.save(output_dir / "label.npy", label)
    np.save(output_dir / "original_idx.npy", idx)
    np.save(output_dir / "original_embeds.npy", embeds)
    return embeds, idx, label


def reduce_dim(np_embeds_dir, pipeline, reduced_embeds_path, reduce_tsne):
    embeds = np.load(np_embeds_dir / "embeds.npy")
    label = np.load(np_embeds_dir / "label.npy")
    # trans_all = np.vstack([train_embeds, test_embeds])
    trans_all = embeds
    # fit everything but the classifier
    for i in range(3):
        trans_all = pipeline.steps[i][1].transform(trans_all)
    np.save(reduced_embeds_path / "reduced_embeds.npy", trans_all)
    if reduce_tsne:
        tsne_arr = TSNE(
            n_components=2, learning_rate="auto", init="pca", perplexity=50, n_jobs=-1
        ).fit_transform(trans_all)
        np.save(reduced_embeds_path / "tsne_2D_reduction_train.npy", tsne_arr)


def plot_tsne(tsne_arr, label, fig_output_dir):
    fig, ax = plt.subplots(4, 3, figsize=(18, 25))
    ax = ax.ravel()
    num = 0
    i = 0
    ax[num].scatter(
        tsne_arr[label != i, 0], tsne_arr[label != i, 1], s=2, c="gray", alpha=0.5
    )
    ax[num].scatter(
        tsne_arr[label == i, 0],
        tsne_arr[label == i, 1],
        s=2,
        label="good",
        c="tab:blue",
    )
    ax[num].set_aspect("equal")
    # ax[num].tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    # ax[num].set_xlabel("tSNE axis-1")
    # ax[num].set_ylabel("tSNE axis-2")
    # lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
    # for hdl in lgnd.legend_handles:
    #     hdl._sizes = [30]
    ax[num].set_title("Good", fontsize=20)
    num += 1
    for i, rea in enumerate(decam_info.reason_li, start=1):
        if sum(label == i) == 0:
            continue
        ax[num].scatter(
            tsne_arr[label != i, 0], tsne_arr[label != i, 1], s=2, c="gray", alpha=0.5
        )
        ax[num].scatter(
            tsne_arr[label == 0, 0],
            tsne_arr[label == 0, 1],
            s=2,
            label="good",
            c="tab:blue",
        )
        ax[num].scatter(
            tsne_arr[label == i, 0],
            tsne_arr[label == i, 1],
            s=2,
            label=rea,
            c="tab:orange",
        )
        ax[num].set_aspect("equal")
        # ax[num].set_xlabel("tSNE axis-1")
        # ax[num].set_ylabel("tSNE axis-2")
        # plt.legend(loc="upper right")
        # lgnd = ax[num].legend(loc="upper right", scatterpoints=1, fontsize=10)
        # for hdl in lgnd.legend_handles:
        #     hdl._sizes = [30]
        ax[num].set_title(rea, fontsize=20)
        num += 1
        # plt.show()
    # ax[-1].axis("off")
    # ax[-2].axis("off")
    plt.savefig(fig_output_dir / "subplots_clustering.png", bbox_inches="tight")


def classify_knn(
    embeds, label, pipeline, prob_thresh=0.5, fig_output_dir: Path = None, seed=0,
    output_dir: Path = None
):
    # knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    # knn.fit(embeds, label)
    # pred = knn.predict(embeds)
    
    pred = pipeline.predict(embeds)
    prob = pipeline.predict_proba(embeds)
    y_cut = np.zeros(pred.shape, dtype=bool)
    map_back = dict(zip(np.unique(pred), np.arange(0, len(np.unique(pred)) + 1)))
    

    for i, idx in enumerate(pred):
        idx = map_back[idx]
        if prob[i, idx] > prob_thresh:
            y_cut[i] = True
    pred_cut = pred[y_cut]
    label_cut = label[y_cut]
    embeds_cut = embeds[y_cut]

    report = classification_report(label_cut, pred_cut, zero_division=0)
    print(report)
    reason_li = ["Good"] + list(decam_info.reason_li[1:])
    pick = label_cut == str(0)
    # score = knn.score(reduced_embeds[pick], label[pick])
    score = pipeline.score(embeds_cut[pick], label_cut[pick])
    num = 0
    label_name = ["0_Good"]
    print(num, "good", score)
    for i, rea in enumerate(decam_info.reason_li, start=1):
        if sum(label_cut == str(i)) == 0:
            continue
        num += 1
        pick = label_cut == str(i)
        # score = knn.score(embeds[pick], label_cut[pick])
        score = pipeline.score(embeds_cut[pick], label_cut[pick])
        print(num, rea, score)
        label_name.append(f"{num}_{rea}")

    # make histogram
    fig, ax1 = plt.subplots()
    y_mod = label.astype(int) - 1
    y_mod[y_mod < 0] = 0
    ax1.hist(
        y_mod, bins=np.arange(0, len(reason_li) + 1), label="accepted", align="left"
    )
    ax2 = ax1.twinx()
    ax1.set_xticks(np.arange(0, len(reason_li)))
    ax1.set_xticklabels(reason_li, rotation=90)
    # plt.xticks(rotation=90)
    ax1.set_xlabel("Category number")
    ax1.set_ylabel("counts")
    num, counts = np.unique(y_mod, return_counts=True)
    ax2.bar(num, counts / len(label), align="edge", alpha=0)
    ax2.set_ylabel("percentage")
    
    ax1.hist(
        y_mod[~y_cut],
        bins=np.arange(0, len(reason_li) + 1),
        label="rejected",
        histtype="stepfilled",
        align="left",
    )

    ax1.legend()
    fig.savefig(fig_output_dir / "knn_classification_hist.png", bbox_inches="tight")
    print("Remaining ratio:", sum(y_cut) / len(y_cut))

    # make confusion matrix plot

    # plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots(figsize=(8, 6))
    # cm = confusion_matrix(label_cut, pred_cut)
    cm_norm = ConfusionMatrixDisplay.from_predictions(
       label_cut, pred_cut, normalize='true')
    cm_norm = np.round(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=[l.split("_", 1)[-1] for l in label_name],
    )
    disp.plot(ax=ax, cmap="rocket")
    plt.xticks(rotation=90)
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.colorbar(img, label='Accuracy')
    plt.savefig(
        fig_output_dir / f"{prob_thresh}_pcut_vit_base_confusion_pretrain.png",
        bbox_inches="tight",
    )
    print("vit_base with resizing images")
    report = classification_report(label_cut, pred_cut, labels=np.unique(label_cut.astype(int)), target_names=label_name)
    if output_dir is not None:
        with open(output_dir / f"classification_report_knn_{prob_thresh}.txt", "w") as f:
            f.write(report)
    return report, cm_norm


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="analyze and visualize the embedding results"
    )
    parser.add_argument(
        "--embeds_dir",
        "-edir",
        type=Path,
        required=True,
        help="Directory containing the embedding files.",
    )
    parser.add_argument(
        "--pipeline_path",
        "-ppath",
        type=Path,
        required=True,
        help="Path to the load the trained sklearn pipeline.",
    )
    parser.add_argument(
        "--reduced_embeds_dir",
        "-rdir",
        type=Path,
        required=True,
        help="Directory to save reduced embeddings.",
    )
    parser.add_argument(
        "--fig_output_dir",
        "-figdir",
        type=Path,
        required=True,
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--classify_output_dir",
        "-clsdir",
        type=Path,
        default=None,
        help="Directory to save classification reports.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["read", "reduce", "plot", "classify"],
        help="Stages to run: reduce_dim, plot_tsne, classify_knn",
    )
    parser.add_argument(
        "--reduce_tsne",
        action="store_true",
        help="Whether to perform tSNE reduction when reducing dimensions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.reduced_embeds_dir.mkdir(parents=True, exist_ok=True)
    args.fig_output_dir.mkdir(parents=True, exist_ok=True)
    if "read" in args.stage:
        print("Combining data...")
        combine_data(args.embeds_dir, args.reduced_embeds_dir)

    if "reduce" in args.stage:
        print("Reducing dimension...")
        reduce_dim(
            args.embeds_dir,
            joblib.load(args.pipeline_path),
            args.reduced_embeds_dir,
            reduce_tsne=args.reduce_tsne,
        )
    ori_embeds = np.load(args.reduced_embeds_dir / "embeds.npy")
    # reduced_embeds = np.load(args.reduced_embeds_dir / "reduced_embeds.npy")
    label = np.load(args.reduced_embeds_dir / "label.npy")
    if "plot" in args.stage:
        print("Plotting tSNE...")
        tsne_arr = np.load(args.reduced_embeds_dir / "tsne_2D_reduction_train.npy")
        plot_tsne(tsne_arr, label.astype(int), args.fig_output_dir)

    if "classify" in args.stage:
        print("Classifying with KNN...")
        classify_knn(
        ori_embeds,
        label,
        joblib.load(args.pipeline_path),
        prob_thresh=0.9,
        fig_output_dir=args.fig_output_dir,
        output_dir=args.classify_output_dir,
    )
