# AGENTS.md — ssl-bad-exposure-identification

## Project summary

Research code for "Semi-Supervised Learning for Bad Exposures in Large Imaging Surveys." Uses a pre-trained DINOv2 ViT to generate embeddings from DECam survey FITS images, then applies kNN classification to identify bad exposures.

## Environment

- **Target machine**: NERSC Perlmutter (SLURM, GPU nodes, module system)
- **Python module**: `pytorch/2.0.1` (see SLURM script)
- **No `requirements.txt` or `pyproject.toml` exists.** Required packages (inferred from imports): `torch`, `torchvision`, `h5py`, `pandas`, `numpy`, `astropy`, `fitsio`, `matplotlib`, `scipy`
- DINOv2 is loaded at runtime via `torch.hub.load(repo_or_dir="facebookresearch/dinov2", ...)` — no separate install needed (but requires internet on first run to download weights)
- **No tests, no CI, no linter/formatter config.**

## Architecture

```
src/
  master_eval.py        # Entry point: splits dataset across GPUs, spawns parallel_eval.py subprocesses
  parallel_eval.py      # Worker: loads DINOv2 model, generates HDF5 embeddings per GPU
  decam_dataset.py      # Torch Dataset classes: DECamImageDataset (labeled images), DECamExposureDataset (exposures)
  decam_info.py         # Static reference data: CCD name↔number mappings, bad-exposure reason codes, bitmask decoders
  decam_focalplane.py   # Focal plane layout: CCD RA/Dec/pixel positions, plotting, array assembly
  decam_postage_stamps.py # Postage stamp + full-focal-plane visualization (legacy, duplicates some decam_focalplane.py)
  finetune_model.py     # Fine-tuning script (incomplete/partial, imports distributed)
  distributed.py        # Distributed training utilities (copied from Meta vicreg-sage repo)
  inference.py          # Helper to read embeddings back from HDF5 files
  read_image.py         # FITS↔NPZ image I/O; also a CLI for batch FITS→NPZ conversion
  util.py               # HTML scraping helpers for generating inspection webpages
  config.yaml           # Placeholder/template config (paths are nonsensical, not used at runtime)
  desi_template_perlmutter_module.slurm  # SLURM batch script template
```

## Hardcoded paths (must change for any new deployment)

- `src/master_eval.py` L30-32: `imdir` paths (`/global/cfs/cdirs/cosmo/work/legacysurvey/dr*/images`)
- `src/decam_postage_stamps.py`: `image_dir`, `blob_dir`, `surveyccd_path` (GLOBUS paths)
- `src/desi_template_perlmutter_module.slurm`: `output_dir`, `src_dir`, `dset-path`, email

## Entry points

| What | Command |
|------|---------|
| Generate embeddings | `python src/master_eval.py -dir <out> --dset-path <csv> --dr dr11 --cont` |
| Resume partial run | Same as above + `--cont` flag (skips already-embedded images by reading existing `.h5` files) |
| Convert FITS to NPZ | `python src/read_image.py <image_dir>` |

## Workflow order

1. **Notebooks** (data preparation): `build_dataset.ipynb` → `build_train_test_dataset.ipynb`
2. **Embedding generation**: `master_eval.py` (splits CSV, launches one `parallel_eval.py` per GPU)
3. **Notebooks** (analysis): `train_knn.ipynb` → `clustering.ipynb` → `analyze_viz.ipynb`
4. **Inference on new data**: `dr11_proc.ipynb` → `master_eval.py` → `dr11_postproc.ipynb` → `read_vi_results.ipynb`

## Data conventions

- **Images**: DECam FITS files (`.fits.fz`), 62 CCDs per exposure. Each CCD is 4094×2046 pixels. The code splits each CCD into two square halves (2046×2046 or 2048×2048 with padding).
- **Datasets**: CSV files with columns `expnum`, `image_filename`, `image_hdu`, `reasons`, `vi_source`, (optionally) `original_df_idx`
- **Embeddings**: HDF5 (`.h5`), one file per GPU worker (`{i}_worker_embeds.h5`), stored under `embeds_out/`. Groups named `images/idx_{orig_idx}_label_{label}` containing DINOv2 embeddings.
- **Labels**: Bitmask-encoded integers decoded via `decam_info.decode_reason()`. Label 0 = good. 14 distinct bad-exposure reason categories.

## Things agents commonly get wrong

1. **There is no `requirements.txt`** — don't try to install dependencies automatically; list what's needed from import inspection.
2. **`config.yaml` is a dead placeholder** — all actual config is passed via CLI args to `master_eval.py` and `parallel_eval.py`. Don't read it expecting runtime values.
3. **Hardcoded NERSC paths will fail anywhere else.** Any code move requires updating at minimum `master_eval.py` image directories.
4. **Image dimensions are hardcoded** — `(4094, 2046)` for full CCD, `(2046, 2046)` or `(2048, 2048)` for halves. Changing input data may require updating `decam_info.py`, `read_image.py`, and `decam_focalplane.py`.
5. **DINOv2 models download on first run** — torch.hub needs internet access. On NERSC compute nodes this may fail; pre-download on a login node or cache the weights.
6. **SCP exposure IDs and filenames** — the notebook `dr11_proc.ipynb` scrapes raw SCP html files to build the image table. This is fragile and site-specific.
7. **`distributed.py`** is copied from an external repo and uses NERSC-specific SLURM env vars (`SLURM_PROCID`, `SLURM_NTASKS`, `SCRATCH`). Don't assume it works elsewhere.
