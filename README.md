# Semi-Supervised Learning for Bad Exposures in Large Imaging Surveys

This repository contains the code for the paper with the same title.
The link to the paper is [arXiv](), [OJAP]()
## Directories

- `data/`: input data from DECaLS (bad_expid.txt) and generated data at each processing stage
- `ext-data/`: external bad-exposure labels from DES and DELVE surveys
- `nb/`: Jupyter notebooks for processing and analysis
- `src/`: source code (Python package)
- `bin/`: entry-point scripts for running the pipeline

## Setup

```bash
pip install -r requirements.txt
```

To use notebooks, install the package in editable mode or ensure the project root is on `sys.path` (set via the first notebook cell).

## Usage

Generate DINOv2 embeddings for a dataset of DECam exposures:

```bash
# Run from the project root
bin/master_eval --output-dir /path/to/output \
                --dset-path /path/to/dataset.csv \
                --dr dr11 \
                --image-dir /path/to/decam/images
```

Key options:
- `--output-dir` / `-dir`: directory for embedding output
- `--dset-path`: CSV dataset (columns: `expnum`, `image_filename`, `image_hdu`, `reasons`, `vi_source`)
- `--dr`: data release (`dr10` or `dr11`)
- `--image-dir`: path to DECam FITS images (optional; overrides dr-based default)
- `--cont`: resume a partial run

On NERSC Perlmutter, use the SLURM template:

```bash
export PROJECT_ROOT=/path/to/this/repo
export OUTPUT_ROOT=/path/to/output
export DSET_ROOT=/path/to/datasets
sbatch src/desi_template_perlmutter_module.slurm
```

## Workflow

1. **Data preparation** (notebooks): `build_dataset.ipynb` → `build_train_test_dataset.ipynb`
2. **Embedding generation**: `bin/master_eval` (splits CSV, launches one worker per GPU)
3. **Analysis** (notebooks): `train_knn.ipynb` → `clustering.ipynb` → `analyze_viz.ipynb`
4. **Inference on new data**: `dr11_proc.ipynb` → `bin/master_eval` → `dr11_postproc.ipynb` → `read_vi_results.ipynb`

## Configurable paths

Paths to DECam images and survey catalogs are configurable via environment variables for the `decam_postage_stamps` module:

- `DECAM_IMAGE_DIR` — image staging directory
- `DECAM_BLOB_DIR` — CCD blob mask directory
- `SURVEYCCD_PATH` — survey-ccds catalog (dr9)
- `SURVEYCCD_PATH_DR8` — survey-ccds catalog (dr8)
