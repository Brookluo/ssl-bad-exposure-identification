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
from pathlib import Path

# %%
rootdir = Path("/pscratch/sd/b/brookluo/decam-exposure/revision/eval/embeds_out")

# %%
case =  list(range(6))
case.append(42)

# %%
for i in case:
    name = f"samp_{i}"
    pd.read_csv(rootdir / name / "classification_report_knn_0.9.txt")

# %%
i = 52
name = f"samp_{i}"
pd.read_csv(rootdir / name / "classification_report_knn_0.9.txt", header=0)

# %%
import pandas as pd
import re

def read_classification_report(path):
    rows = []
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "accuracy" in line:
                el = line.split()
                rows.append([el[0], None, None, el[1], el[2]])
                continue
            # Match lines like:
            # class_name   precision  recall  f1  support
            m = re.match(
                r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)$",
                line
            )
            if m:
                label = m.group(1).strip()
                precision = float(m.group(2))
                recall = float(m.group(3))
                f1 = float(m.group(4))
                support = int(m.group(5))
                
                rows.append([label, precision, recall, f1, support])

    df = pd.DataFrame(rows, columns=["label", "precision", "recall", "f1", "support"])
    return df


# %%
name = f"samp_{i}"
read_classification_report(rootdir / name / "classification_report_knn_0.9.txt")

# %%
with open(rootdir / "samp_50" / "classification_report_knn_0.9.txt", "r") as f:
    report = f.read()
print(report)


# %%
def combine_and_mean(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine equally shaped DataFrames and take the mean value at each row/column position."""
    if not dfs:
        raise ValueError("Provide at least one DataFrame.")
    ref = dfs[0]
    for idx, df in enumerate(dfs[1:], start=1):
        if df.shape != ref.shape or not df.columns.equals(ref.columns) or not df.index.equals(ref.index):
            raise ValueError(f"DataFrame at position {idx} is not aligned with the first one.")
    stacked = np.stack([df.to_numpy()[:-3, 1:].astype(float) for df in dfs], axis=0)
    mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0)
    mean_df = pd.DataFrame(mean_values, index=ref.index[:-3], columns=ref.columns[1:])
    std_df = pd.DataFrame(std_values, index=ref.index[:-3], columns=ref.columns[1:])
    mean_df.insert(0, "label", ref["label"][:-3].values)
    std_df.insert(0, "label", ref["label"][:-3].values)
    return mean_df, std_df  


# %%
case = list(range(50, 60))
dfs = []
for i in case:
    name = f"samp_{i}"
    df = read_classification_report(rootdir / name / "classification_report_knn_0.9.txt")
    dfs.append(df)
mean_df, std_df = combine_and_mean(dfs)

# %%
mean_df

# %%
for i in range(mean_df.shape[0]):
    label = mean_df.iloc[i, 0]
    mean_vals = mean_df.iloc[i, 1:]
    std_vals = std_df.iloc[i, 1:]
    mean_str = ", ".join([f"{val:.4f} ± {std:.4f}" for val, std in zip(mean_vals, std_vals)])
    print(f"{label}: {mean_str}")

# %%
