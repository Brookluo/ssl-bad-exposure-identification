# Semi-Supervised Learning for Bad Exposures in Large Imaging Surveys

ML pipeline for classifying DECam exposures as good/bad using DINOv2 embeddings + kNN.

## Quick start
```bash
pip install -r requirements.txt
python -m decam_qa.cli embed --config configs/embed.yaml --dset data.csv --dr dr10
python -m decam_qa.cli train --config configs/train.yaml
python -m decam_qa.cli classify --config configs/inference.yaml
python -m pytest tests/ -v
```

## NERSC/Perlmutter
```bash
module load pytorch/2.6.0
export PYTHONPATH=$HOME/ssl-exposure-identification-paper/src:$PYTHONPATH
sbatch src/desi_template_perlmutter_module.slurm
```

## Configuration
All params in `configs/`: embed.yaml, train.yaml, inference.yaml.
Paths are NERSC-specific — change for other systems.

## Tests
```bash
python -m pytest tests/ -v
```
Tests use synthetic data + mocked DINOv2 — no GPU or CFS required.

## Notebooks
Jupytext (percent format). `.py` files are source of record; `.ipynb` is gitignored.
To open: `jupytext --to notebook nb/train_knn.py`

## Repository structure
```
src/decam_qa/     — Python package: embeddings, classifier, pipeline, CLI, utils
configs/          — Per-stage YAML configuration files
nb/               — Jupytext-tracked analysis notebooks
tests/            — pytest test suite
data/             — Intermediate CSVs and exposure lists
```

## Reason classification scheme
15 bad-exposure categories stored as bitmask integers. See `src/decam_qa/info.py`
for the full scheme. `decode_reason(bitmask)` converts to human-readable strings.
