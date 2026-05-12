# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       notebook_metadata_filter: all
#   kernelspec:
#     display_name: desi-main
#     language: python
#     name: desi-main
# ---

# %% [markdown]
# # Multi-Scale Exposure Identification — Evaluation

# %%
import numpy as np
import h5py
import json
from pathlib import Path
from collections import defaultdict

from decam_qa.io import read_exposure_embeddings
from decam_qa.classifier import (
    aggregate_exposure_embeddings,
    train_logistic_binary,
    predict_binary,
    train_multilabel_reason,
    predict_multilabel,
)
from decam_qa.evaluation import (
    compute_binary_metrics,
    compute_reason_metrics,
    exposure_grouped_cross_validate,
)
from decam_qa.info import decode_reason

# %% [markdown]
# ## Load multi-scale embeddings

# %%
# EMBEDS_DIR = "/pscratch/sd/b/brookluo/decam-exposure/multiscale/embeds_out"
# result = read_exposure_embeddings(EMBEDS_DIR)
# print(f"Loaded {len(result)} exposures")

# %% [markdown]
# ## Aggregate features and build label arrays

# %%
# X_list = []
# y_list = []
# y_reason_list = []
# groups_list = []
# 
# for expnum, data in result.items():
#     features = aggregate_exposure_embeddings(
#         data["global"], data["locals"],
#         scores=[], k=8)
#     X_list.append(features)
#     
#     # Binary label: any non-zero reason bitmask
#     reason_bm = data.get("metadata", {}).get("reason_bitmask", 0)
#     y_list.append(1 if reason_bm > 0 else 0)
#     y_reason_list.append(reason_bm)
#     groups_list.append(expnum)
# 
# X = np.array(X_list)
# y = np.array(y_list, dtype=int)
# y_reason = np.array(y_reason_list, dtype=int)
# groups = np.array(groups_list)
# 
# print(f"{X.shape=}, n_good={np.sum(y==0)}, n_bad={np.sum(y==1)}")
# print(f"Unique groups: {len(np.unique(groups))}")

# %% [markdown]
# ## Binary classification with GroupKFold

# %%
# fold_metrics = exposure_grouped_cross_validate(X, y, groups, n_splits=5, random_state=42)
# 
# for fm in fold_metrics:
#     print(f"Fold {fm['fold']}: AP={fm['avg_precision']:.4f}, "
#           f"ROC_AUC={fm['roc_auc']:.4f}, "
#           f"n_train_pos={fm['n_train_pos']}, n_test_pos={fm['n_test_pos']}")
# 
# # Aggregate
# avg_ap = np.mean([fm["avg_precision"] for fm in fold_metrics])
# avg_roc = np.mean([fm["roc_auc"] for fm in fold_metrics])
# print(f"\nMean AP: {avg_ap:.4f}, Mean ROC AUC: {avg_roc:.4f}")

# %% [markdown]
# ## Multi-label reason classification

# %%
# model = train_multilabel_reason(X, y_reason, n_reasons=15, class_balanced=True, random_state=42)
# y_prob_reason = predict_multilabel(model, X)
# reason_metrics = compute_reason_metrics(y_reason, y_prob_reason)
# 
# for name, ap in sorted(reason_metrics.items(), key=lambda x: -x[1]):
#     print(f"  {name}: AP={ap:.4f}")
