# AGENTS.md — ssl-exposure-identification-paper

## What this repo is

Research code for "Semi-Supervised Learning for Bad Exposures in Large Imaging Surveys." Uses DINOv2 (ViT) embeddings + kNN to classify DECam exposures as good/bad. Input data lives at NERSC on CFS; code runs on Perlmutter GPU nodes.

## Notebooks are jupytext-tracked

- `*.ipynb` is in `.gitignore`. The truth is `*.py` files in `nb/`, authored with jupytext (percent format).
- To open: `jupytext --to notebook nb/foo.py` or open the `.py` directly in Jupyter with the jupytext extension.
- Do not edit `.ipynb` files directly; paired `.py` files are the source of record.

## Architecture

```
src/           — Python modules + entrypoint scripts
  master_eval.py     → entrypoint: spawns one subprocess per GPU
  parallel_eval.py   → per-GPU worker: loads DINOv2, generates embeddings → HDF5
  decam_dataset.py   → PyTorch Dataset classes for DECam images
  decam_info.py      → static lookup tables (CCD names↔nums, reason bitmask dicts, filters)
  inference.py       → utility to read generated HDF5 embedding files
  finetune_model.py  → fine-tuning script (unused in main workflow)
  do_analysis_viz.py → analysis/visualization
  util.py            → HTML page generation, postage-stamp layouts
nb/           — jupytext notebooks for data prep + analysis
  build_dataset.py              → builds bad/good exposure CSVs from DES+DELVE expert labels
  build_train_test_dataset.py   → train/test split with per-category stratification
  train_knn.py                  → PCA + kNN classifier training on DINOv2 embeddings
  dr11_proc.py / dr11_postproc.py → DR11 inference + postprocessing
  clustering.py / analyze_viz.py  → t-SNE plots, distribution analysis
data/         — intermediate CSVs, exposure lists; read-only inputs from CFS
ext-data/     — DES/DELVE expert label FITS files (gitignored, large binary)
```

## Main workflow (in order)

1. **Build datasets (notebooks):** `build_dataset.py` → `build_train_test_dataset.py`
   Produces CSVs in `data/samples/` with `image_filename`, `expnum`, `ccdnum`, `image_hdu`, `filter`, `reasons` (bitmask), `vi_source`.
2. **Generate DINOv2 embeddings:** Run `master_eval.py` via SLURM (see `desi_template_perlmutter_module.slurm` + `run-dist-job.sh`).
   Spawns one `parallel_eval.py` per GPU. Outputs go to `embeds_out/` as per-worker `.h5` files keyed by `idx_<orig_idx>_label_<label>`.
3. **Train kNN classifier:** `train_knn.py` notebook — reads embeddings, runs PCA + KNeighborsClassifier with HalvingRandomSearchCV over StandardScaler/PCA/VarianceThreshold/kNN pipeline. Saves pipeline as joblib pickle.
4. **DR11 inference:** `dr11_proc.py` + `dr11_postproc.py` — apply trained model to new data release.
5. **Visual inspection (VI):** `read_vi_results.py` — compare ML predictions against human expert labels.

## Commands

### Generate embeddings (only GPU command)
```bash
# Via SLURM:
sbatch src/desi_template_perlmutter_module.slurm
# Direct:
python src/master_eval.py -dir $OUTPUT_DIR --dset-path sample.csv --dr dr10 --cont
```
`--cont` resumes from partially completed runs by skipping already-processed indices.

### Run on Perlmutter
- Module required: `pytorch/2.6.0`
- GPU nodes, 4 GPUs per node, `--gpu-bind=none` is needed
- SLURM account: `desi_g` (GPU allocation)
- Image data paths are hard-coded to CFS: `/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/images/` etc.
- Scratch output: `/pscratch/sd/b/brookluo/decam-exposure/`

## Hard-coded paths (review before running)

These are absolute NERSC paths you must change for your system:
- `src/master_eval.py:32-34` — image directories for dr10/dr11
- `src/run-dist-job.sh:7,14` — output/scratch dirs
- `nb/build_dataset.py:41-48` — DR8/9/10 CFS paths + `survey-ccds-decam-dr10.fits.gz`
- `nb/build_train_test_dataset.py:66-68,560-565` — same CFS paths + scratch output paths
- `nb/train_knn.py:40,72,78,81,179,322` — train/test data directories, output paths
- `src/desi_template_perlmutter_module.slurm:10-11` — SLURM log paths

## Reason classification scheme

- 15 bad-exposure categories stored as bitmask integers (see `decam_info.py:102-123`)
- `reason_num_dict` maps reason name → bit index (0 = Bad_WCSCAL, 1 = Saturated, …)
- `decode_reason(bit_reason)` converts bitmask → list of reason strings
- Labels 0 = good, ≥1 = bad category index (bit index + 1)
- Training drops classes 0 (Bad_WCSCAL), 3 (Bad_seeing), 10 (Fringing), 11 (Canopus) — stored in `config.yaml:11`

## Data constraints

- DECam images are ~4096×2048 pixels. Must be center-cropped to multiples of 14 for ViT (e.g., 4088×2044 before resize).
- `parallel_eval.py` expands single-channel images to 3 channels for DINOv2.
- Exposure metadata table: `survey-ccds-decam-dr10.fits.gz` on CFS (read by `astropy.table.Table`).
- `ext-data/` contains `des_exclude_y6a2.fits` and `delve_exclude_20230725.fits` — human expert labels from DES and DELVE surveys. These are gitignored.

## No test/lint/typecheck infrastructure

There is no test suite, linter config, CI, or type checking in this repo. All code is research-grade exploratory analysis.

## Jupyter kernel

Notebooks use `desi-main` kernel (see jupytext headers). Activate with DESI environment on Perlmutter.
