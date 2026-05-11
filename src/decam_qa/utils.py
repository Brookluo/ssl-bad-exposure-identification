"""Utility functions: HTML webpage generation, parsing, and dimension reduction helpers."""
from pathlib import Path
from typing import List, Tuple
import numpy as np


def get_info_from_html(html_path: str) -> List[List[str]]:
    """Parse a table-format HTML page into structured entries."""
    entry = []
    with open(html_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("<table>") or line.startswith("</table>"):
                if "<td>" not in line:
                    continue
                line = line.strip("<table>").strip("</table>")
            meta, img_src = line.rsplit("<td>", maxsplit=1)
            fname, expnum, *other = meta.split("<br>")
            img_path_fmt = '<td><img src="./images/{}.jpg"></tr>'
            img_src_data = img_path_fmt.format(fname.split(">")[-1])
            entry.append([fname, expnum, *other, img_src_data])
    return entry


def make_webpage(master_list, pack_idx, root_dir, base_name, num_element=400):
    """Generate paginated HTML tables from exposure data."""
    root_dir = Path(root_dir)
    base_tmpl = "<table>\n{}\n</table>\n"
    content, start_exp, count = [], -1, 0
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


def combine_embeds(h5embeds_dir: str, output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read HDF5 embeddings and combine into numpy arrays."""
    from decam_qa.io import read_embeddings
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, idx, label = read_embeddings(h5embeds_dir)
    embeds = np.vstack([np.mean(it, axis=0) for it in data])
    np.save(output_dir / "label.npy", label)
    np.save(output_dir / "original_idx.npy", idx)
    np.save(output_dir / "original_embeds.npy", embeds)
    return embeds, idx, label


def reduce_dim(embeds, pipeline, output_dir, do_tsne=False):
    """Reduce embedding dimensionality through a fitted pipeline and optionally t-SNE."""
    from sklearn.manifold import TSNE
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reduced = embeds.copy()
    n_steps = len(pipeline.steps) - 1
    for i in range(n_steps):
        step = pipeline.steps[i][1]
        if hasattr(step, "transform"):
            reduced = step.transform(reduced)
    np.save(output_dir / "reduced_embeds.npy", reduced)
    if do_tsne:
        tsne_arr = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=50, n_jobs=-1
                        ).fit_transform(reduced)
        np.save(output_dir / "tsne_2D_reduction.npy", tsne_arr)
    return reduced
