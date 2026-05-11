# Refactor: Clean Package Structure for ssl-exposure-identification-paper

**Date:** 2026-05-10
**Status:** Design updated — testing strategy added, awaiting user spec review

## Motivation

The current codebase has tangled concerns: embedding generation, kNN training, and inference logic are spread across scripts and notebooks with hard-coded paths, no module boundaries, and minimal documentation. The goal is to make the codebase clearer, more reusable, and easier to adapt to new data releases — without abstracting away DECam-specific domain knowledge.

## Scope

- **In scope:** Restructure `src/` into an importable package with clear module boundaries, extract callable functions from notebooks into library code, add comprehensive docstrings, create per-stage YAML config files to replace scattered hard-coded paths/params, add a unified CLI entrypoint, update README, add extensive unit and integration tests.
- **Out of scope:** Multi-instrument support (DECam only), new ML models, CI/CD, removing jupytext, adding type checking.

## Package structure

New importable package `src/decam_qa/` with clear module boundaries:

```
src/decam_qa/
  __init__.py          # re-exports key public symbols
  info.py              # (was decam_info.py) — CCD maps, reason dicts, filters
  config.py            # YAML config loader + dataclass defaults
  dataset.py           # DECamImageDataset (kept, cleaned up)
  embeddings.py        # DINOv2 model loading + embed generation (extracted from parallel_eval.py)
  classifier.py        # kNN pipeline building, training, prediction
  io.py                # HDF5 embed I/O, FITS image reading
  pipeline.py          # orchestration: embed → train → classify
  cli.py               # unified entrypoint: decam-qa {embed|train|classify|evaluate}
  utils.py             # HTML generation, plotting helpers (from current util.py, do_analysis_viz.py)
```

Old `src/*.py` files (master_eval, parallel_eval, etc.) become deprecated re-exports or are deleted once notebooks are updated.

New directory for stage-specific configuration:
```
configs/
  embed.yaml           # DINOv2 model, batch size, crop params, image root paths
  train.yaml           # PCA components, kNN hyperparam grid, output paths
  inference.yaml       # thresholds, output paths, data release selection
```

New test directory (pytest):
```
tests/
  conftest.py           # shared fixtures: synthetic images, mock configs, small embeddings
  test_info.py          # CCD name↔num consistency, bitmask encode/decode round-trip
  test_config.py        # YAML load, validation, missing-key errors, defaults
  test_io.py            # HDF5 read/write round-trip, FITS image I/O, embedding glob
  test_dataset.py       # DECamImageDataset indexing, label encoding, shuffle
  test_embeddings.py    # create_model (architecture check), generate_embeddings on synthetic data
  test_classifier.py    # build_pipeline, train on synthetic embeddings, predict, score
  test_pipeline.py      # ParallelEvaluator subprocess spawning (mocked GPU)
  test_cli.py           # argparse subcommand routing, --help output
```

Notebooks in `nb/` become thin wrappers that import from `decam_qa`:
```python
from decam_qa.pipeline import build_dataset, train, classify
build_dataset(dr="dr10")
train(config="configs/train.yaml")
classify(config="configs/inference.yaml")
```

## Public API

Each module exposes a small, documented surface with numpy-style docstrings:

### `decam_qa.embeddings`
```python
def create_model(size="base", use_register=True) -> torch.nn.Module
def generate_embeddings(dataset, model, device, output_dir) -> None
```

### `decam_qa.classifier`
```python
def build_pipeline(param_grid: dict | None = None) -> Pipeline
def train(pipeline, X, y, search_params: dict) -> Pipeline
def predict(pipeline, X) -> tuple[np.ndarray, np.ndarray]
```

### `decam_qa.io`
```python
def read_embeddings(h5_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]
def read_fits_image(path, hdu) -> np.ndarray
def write_embeddings(embeds, indices, labels, output_dir) -> list[Path]
```

### `decam_qa.config`
```python
class EmbedConfig(TypedDict): ...
class TrainConfig(TypedDict): ...
class InferenceConfig(TypedDict): ...

def load_config(path, stage: str) -> Config
def get_default_config(stage: str) -> Config
```

### `decam_qa.pipeline`
```python
class ParallelEvaluator:
    """Spawns per-GPU worker subprocesses for embed generation."""
    def __init__(self, config: EmbedConfig, dataset_path, dr, resume=False): ...
    def run(self) -> list[Path]:  # returns paths to HDF5 output
```

### `decam_qa.cli`
```bash
decam-qa embed --config configs/embed.yaml --dset dataset.csv --dr dr10
decam-qa train --config configs/train.yaml
decam-qa classify --config configs/inference.yaml
decam-qa evaluate --predictions results.csv --labels ground_truth.csv
```

## Testing strategy

All tests use **pytest**. Shared fixtures live in `tests/conftest.py`. Tests must run without GPUs or real CFS data — they use synthetic data and mocked external dependencies.

### Test framework

- **Runner:** `pytest` (no extra plugins required beyond stdlib `unittest.mock`)
- **Dependencies:** All test dependencies are already in `requirements.txt` (numpy, pandas, h5py, torch, scipy, matplotlib, astropy, fitsio, jupytext). Add `pytest` to `requirements.txt`.
- **Run command:** `python -m pytest tests/ -v`
- **Coverage target:** ≥80% line coverage on `decam_qa/` modules

### Module-by-module test plan

#### `tests/test_info.py`
Test all pure functions in `decam_qa/info.py`:

| Test | What it verifies |
|------|-----------------|
| `test_ccdname2num_all_keys` | All 62 CCD names map to distinct integers 1-62 |
| `test_ccdnum2name_roundtrip` | `ccdnum2name[ccdname2num[n]] == n` for all names |
| `test_reason_num_dict_consistency` | Dict values are 0-22, no duplicate values |
| `test_decode_reason_zero` | `decode_reason(0)` returns `[]` |
| `test_decode_reason_single_bit` | `decode_reason(2**5)` returns `['Nonoptimal_exp']` |
| `test_decode_reason_multi_bit` | `decode_reason(2**1 | 2**2)` returns `['Saturated', 'Clouds_transparency']` |
| `test_decode_reason_return_num` | `decode_reason(2**3, return_num=True)` returns `[3]` |
| `test_decode_vi_source_single` | `decode_vi_source(1)` returns `['Rongpu']` |
| `test_decode_vi_source_combined` | `decode_vi_source(3)` returns `['Rongpu', 'Alex']` |
| `test_decode_ml_label_good` | `decode_ml_label([0])` returns `['good']` |
| `test_decode_ml_label_bad` | `decode_ml_label([2])` returns `['Saturated']` |
| `test_encode_decode_roundtrip` | Encode reasons → decode → encode: same bitmask |
| `test_filter_dict_coverage` | All g/r/i/z/Y filters present, values 1-5 |
| `test_is_miss_2ccd_known_dates` | Before 2014 returns False, 2014-2016 returns True, after 2017 returns False |

#### `tests/test_config.py`
Test `decam_qa/config.py`:

| Test | What it verifies |
|------|-----------------|
| `test_load_valid_embed_config` | Loads valid YAML, returns EmbedConfig with correct types |
| `test_load_valid_train_config` | Loads valid YAML, returns TrainConfig with correct types |
| `test_load_valid_inference_config` | Loads valid YAML, returns InferenceConfig with correct types |
| `test_load_missing_required_key` | Raises ValueError with key name and file path |
| `test_load_invalid_stage_name` | Raises ValueError for unknown stage string |
| `test_get_default_config` | Each stage has sensible defaults for all optional fields |
| `test_embed_config_defaults` | Missing optional fields (e.g., `pin_memory`) get default values |
| `test_config_field_types` | `crop_size` is a list of ints, `model.size` is a string, etc. |

#### `tests/test_io.py`
Test `decam_qa/io.py`:

| Test | What it verifies |
|------|-----------------|
| `test_write_read_embeddings_roundtrip` | Write embeddings to HDF5 → read back: identical data |
| `test_read_embeddings_empty_dir` | Raises FileNotFoundError or returns empty |
| `test_read_embeddings_multi_worker` | Reads 4 HDF5 files, concatenates correctly |
| `test_hdf5_key_format` | Keys match pattern `idx_<int>_label_<int>` |
| `test_write_embeddings_creates_dir` | Auto-creates output directory if missing |
| `test_read_fits_image_valid` | Reads a valid FITS file, returns numpy array |
| `test_read_fits_image_bad_hdu` | Raises error on non-existent HDU index |
| `test_read_fits_image_missing_file` | Raises FileNotFoundError with path in message |
| `test_read_fits_image_byte_order` | Output is native byte order (not big-endian from FITS) |
| `test_write_embeddings_overwrite` | Writing to existing HDF5 appends rather than overwrites |

#### `tests/test_dataset.py`
Test `decam_qa/dataset.py` using a small in-memory DataFrame:

| Test | What it verifies |
|------|-----------------|
| `test_dataset_len` | `len(dataset)` matches CSV row count |
| `test_dataset_getitem_returns_tuple` | Returns `(image, label)` tuple |
| `test_dataset_image_shape` | Image is `(1, H, W)` with correct dimensions |
| `test_dataset_image_dtype` | Image dtype is native byte order |
| `test_dataset_label_good` | Good image (reason=0) → label 0 |
| `test_dataset_label_single_bad` | Single reason → correct positive label |
| `test_dataset_label_deterministic` | Same index always returns same label |
| `test_dataset_shuffle_changes_order` | After shuffle, access order differs from original |
| `test_dataset_shuffle_seed_reproducible` | Same seed → same order |
| `test_dataset_missing_image_file` | Raises FileNotFoundError |

#### `tests/test_embeddings.py`
Test `decam_qa/embeddings.py`:

| Test | What it verifies |
|------|-----------------|
| `test_create_model_base` | Returns torch.nn.Module, correct backbone string |
| `test_create_model_small` | `size="small"` returns vits14 variant |
| `test_create_model_no_register` | `use_register=False` returns no-reg variant |
| `test_create_model_invalid_size` | Raises KeyError for unknown size |
| `test_generate_embeddings_output_shape` | On synthetic images, output has expected dim (DINOv2 base=768) |
| `test_generate_embeddings_output_count` | N input images → N embedding entries in HDF5 |
| `test_generate_embeddings_single_channel_expansion` | 1-channel input → expanded to 3-channel before model |
| `test_generate_embeddings_resume_skips_existing` | With existing HDF5 entries, only new indices processed |

#### `tests/test_classifier.py`
Test `decam_qa/classifier.py` with synthetic embeddings:

| Test | What it verifies |
|------|-----------------|
| `test_build_pipeline_returns_sklearn_pipeline` | Output is a `sklearn.pipeline.Pipeline` |
| `test_build_pipeline_steps_order` | Steps: scaler → PCA → VarianceThreshold → KNeighborsClassifier |
| `test_build_pipeline_custom_grid` | Custom param_grid is accepted |
| `test_train_on_synthetic_data` | Fits without error, `train_score` > `chance` |
| `test_train_reproducible` | Same seed and data → same pipeline parameters |
| `test_predict_returns_labels_and_probs` | Returns `(labels, probabilities)` tuple of correct length |
| `test_predict_probabilities_sum_to_one` | For binary: probs sum to 1 per sample |
| `test_predict_on_untrained_raises` | Calling predict before fit raises error |
| `test_pipeline_serialization_roundtrip` | joblib.dump/load preserves predictions |
| `test_train_with_halving_random_search` | `search_params` activates HalvingRandomSearchCV path |

#### `tests/test_pipeline.py`
Test `decam_qa/pipeline.py` orchestration:

| Test | What it verifies |
|------|-----------------|
| `test_parallel_evaluator_splits_dataset` | CSV of N rows split across K workers evenly |
| `test_parallel_evaluator_resume_skips_done` | With `resume=True`, already-processed indices excluded |
| `test_parallel_evaluator_worker_failure` | Non-zero subprocess exit → RuntimeError raised |
| `test_parallel_evaluator_creates_output_dirs` | `tmp/` and `embeds_out/` created automatically |
| `test_full_pipeline_synthetic` | End-to-end: synthetic images → embeds → train → classify |

#### `tests/test_cli.py`
Test `decam_qa/cli.py` argparse routing:

| Test | What it verifies |
|------|-----------------|
| `test_cli_embed_requires_config` | Missing `--config` → SystemExit |
| `test_cli_embed_requires_dset` | Missing `--dset` → SystemExit |
| `test_cli_embed_all_args_parsed` | All args populate correctly in namespace |
| `test_cli_train_all_args_parsed` | All training args parse correctly |
| `test_cli_classify_all_args_parsed` | All inference args parse correctly |
| `test_cli_help` | `--help` exits 0 and prints usage |
| `test_cli_subcommand_help` | `embed --help`, `train --help`, `classify --help` all work |

### Test fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def synthetic_image():
    """A (1, 2048, 4096) float32 array simulating a DECam CCD image."""

@pytest.fixture
def sample_dataframe():
    """A small pd.DataFrame with columns matching DECamImageDataset expectations,
    pointing to temporary real FITS files."""

@pytest.fixture
def small_embeddings():
    """(100, 768) synthetic embeddings with known class structure."""

@pytest.fixture
def small_labels():
    """100 labels with 3 classes for classifier testing."""

@pytest.fixture
def temp_h5_dir(tmp_path):
    """Temporary directory with mock HDF5 embedding files."""

@pytest.fixture
def temp_config(tmp_path):
    """Write minimal valid YAML configs to temporary files."""

@pytest.fixture
def mock_model(mocker):
    """Mocked DINOv2 model returning fixed-size embeddings, to avoid
    downloading actual weights (~1GB) during tests."""
```

### What is NOT tested (by design)

- **Actual DINOv2 inference on real images:** Too slow and requires GPU + model download. Covered by `mock_model` fixture.
- **Full SLURM distributed execution:** Untestable without scheduler. Covered by `ParallelEvaluator` unit tests with mocked `subprocess.Popen`.
- **Visual output correctness (plots, HTML):** Test that files are created with expected names/paths, not pixel-level output.
- **GPU-specific code paths:** All GPU logic is tested via mocked `torch.cuda` and mocked model forwarding.

## Configuration files

Three per-stage YAML configs replace scattered hard-coded values. Each is loaded via `decam_qa.config.load_config()` which validates required keys and fills defaults for optional ones.

### `configs/embed.yaml`
```yaml
model:
  size: base
  use_register: true

data:
  batch_size: 1
  crop_size: [2352, 1176]
  num_workers: 10
  pin_memory: false

image_roots:
  dr10: /global/cfs/cdirs/cosmo/work/legacysurvey/dr10/images
  dr11: /global/cfs/cdirs/cosmo/work/legacysurvey/dr11/images

distributed:
  backend: nccl
  gpu_bind: none

scratch_dir: /pscratch/sd/b/brookluo/decam-exposure/revision
```

### `configs/train.yaml`
```yaml
embeddings:
  train_dir: /pscratch/sd/b/brookluo/decam-exposure/dino_v2/base_resize_dr10cut/train/eval/embeds_out
  test_dir: /pscratch/sd/b/brookluo/decam-exposure/dino_v2/base_test_good/eval/embeds_out

pipeline:
  scalers: [StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, PowerTransformer]
  pca_components: [5, 10, 15, 25, 30, 50, 70, 100]
  variance_threshold: [0, 0.001, 0.01]
  knn:
    n_neighbors: [1, 3, 5, 7, 10, 20, 30]
    p: [1, 2, 3, 5]
    leaf_size: [1, 5, 10, 15, 30, 35]

cv:
  n_splits: 3
  test_size: 0.3
  random_state: 42

model_path: /pscratch/sd/b/brookluo/decam-exposure/postproc/knn_pipe.pkl
```

### `configs/inference.yaml`
```yaml
pipeline_path: /pscratch/sd/b/brookluo/decam-exposure/postproc/knn_pipe.pkl

embeddings_dir: /pscratch/sd/b/brookluo/decam-exposure/revision/node0/embeds_out
dataset_csv: /pscratch/sd/b/brookluo/decam-exposure/revision/proc-data/node0_dr10_sample.csv

probability_threshold: 0.7
top_n_per_class: 200

metadata:
  dr10_tab: /global/cfs/cdirs/cosmo/work/legacysurvey/dr10/survey-ccds-decam-dr10.fits.gz
  ccd_cuts_enabled: true

results_dir: /pscratch/sd/b/brookluo/decam-exposure/revision/results
plot_dir: /pscratch/sd/b/brookluo/decam-exposure/revision/plots
```

## Migration plan

### Phase 0: Setup testing infrastructure
0. Add `pytest` to `requirements.txt`, create `tests/` directory with `conftest.py`

### Phase 1: Create new package + tests (no breaking changes)
1. Create `src/decam_qa/` directory with `__init__.py`
2. Port `decam_info.py` → `decam_qa/info.py` (no logic changes)
3. Write `tests/test_info.py` (all pure functions, run immediately to verify)
4. Extract I/O functions → `decam_qa/io.py`
5. Write `tests/test_io.py`
6. Build `decam_qa/config.py` with dataclasses + YAML loader
7. Write `tests/test_config.py`
8. Extract embed logic from `parallel_eval.py` → `decam_qa/embeddings.py`
9. Write `tests/test_embeddings.py` (with mocked model)
10. Write `decam_qa/utils.py` by extracting from `util.py` and `do_analysis_viz.py`

### Phase 2: New pipeline modules + tests
11. Extract classifier logic from `train_knn.py` notebook → `decam_qa/classifier.py`
12. Write `tests/test_classifier.py`
13. Build `decam_qa/pipeline.py` with `ParallelEvaluator` class
14. Write `tests/test_pipeline.py`
15. Build `decam_qa/cli.py` with argparse-based CLI
16. Write `tests/test_cli.py`
17. Write `tests/test_dataset.py`
18. Create `configs/` directory with three YAML files

### Phase 3: Update notebooks
19. Refactor `nb/train_knn.py` to call `decam_qa.classifier` functions
20. Refactor `nb/dr11_inference.py` and `nb/dr11_postproc.py` to call `decam_qa.classifier`
21. Refactor remaining notebooks similarly

### Phase 4: Clean up (once verified by test suite + manual run)
22. Make old `src/*.py` files (master_eval, parallel_eval, inference, util) into deprecated thin wrappers
23. Update SLURM scripts to use CLI
24. Update README with new commands and structure
25. Verify full pipeline produces identical results to pre-refactor output

## Error handling conventions

- All file I/O in `decam_qa.io` raises `FileNotFoundError` with clear messages when CFS paths are missing
- Config loading raises `ValueError` with the missing key name and file path
- `ParallelEvaluator.run()` raises `RuntimeError` if any worker subprocess returns non-zero exit code
- kNN training catches `sklearn` exceptions and re-raises with the failing hyperparameter context

## Backward compatibility

The old `src/` scripts and notebook imports (`from decam_info import ...`) continue to work via thin wrapper stubs that re-export from `decam_qa`. No existing workflow is broken during migration.

## Verification

After migration, the full workflow should produce identical results:
1. Embeddings from `decam-qa embed` match those from `master_eval.py`
2. Trained kNN pipeline from `decam-qa train` matches the existing joblib pickle (same hyperparameter search space, same random seed)
3. Classification output from `decam-qa classify` produces identical labels and probabilities
