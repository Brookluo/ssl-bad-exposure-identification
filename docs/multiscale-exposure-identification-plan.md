# Multi-Scale Exposure Identification Pipeline Plan

## 1. Goal

Improve the bad-exposure identification pipeline by moving from independent
full-resolution CCD inference to an exposure-grouped, multi-scale pipeline.
The primary optimization target is model quality, with efficiency improved by
avoiding unnecessary full-resolution DINO inference on all 61/62 CCDs.

The current pipeline processes one CCD row at a time, expands the single
channel image to three channels, resizes it, embeds it with frozen DINOv2, and
classifies the embedding with kNN. This is simple and useful as a baseline, but
it is inefficient for full DECam exposures and does not naturally model
exposure-level defects.

The new pipeline should:

- Open each FITS exposure once and process all available CCDs together.
- Use a low-resolution focal-plane representation for global exposure context.
- Use selective high-resolution CCD/tile embeddings for localized defects.
- Keep CCD/tile-level localization information.
- Produce exposure-level good/bad and reason probabilities.
- Preserve the existing CCD-only pipeline as a reproducible baseline.

## 2. Current Constraints and Observations

DECam exposures are large:

- One exposure contains up to 62 CCDs.
- Each CCD is about `4094 x 2046` pixels.
- A native stitched focal-plane image would be about `29590 x 26787` pixels.
- The stitched native image is mostly empty gaps between CCDs and is too large
  to be a useful ViT input.

Therefore, the efficient representation is not a native full-resolution
focal-plane image. The correct direction is exposure-grouped processing with
two resolutions:

1. A compact low-resolution focal-plane stamp for global context.
2. A selected set of higher-resolution CCD/tile crops for local defects.

The existing repository already contains useful pieces:

- `src/decam_qa/dataset.py` reads one CCD HDU per sample.
- `src/decam_qa/embeddings.py` generates DINOv2 embeddings.
- `src/parallel_eval.py` contains the older embedding-generation path.
- `src/decam_focalplane.py` contains focal-plane geometry and native assembly.
- `src/decam_postage_stamps.py` contains low-resolution postage-stamp logic.
- `src/read_image.py` contains FITS-to-half-CCD utilities.

## 3. Target Architecture

### 3.1 Data Flow

The new inference/training data flow should be:

```text
CSV rows / survey CCD table
        |
        v
group rows by exposure: expnum + image_filename
        |
        v
open one FITS exposure once
        |
        +--> build low-res focal-plane stamp
        |        |
        |        v
        |   global DINO embedding
        |
        +--> compute cheap per-CCD/tile anomaly scores
                 |
                 v
            select top-K CCDs/tiles
                 |
                 v
            high-res DINO embeddings
        |
        v
aggregate global + selected local embeddings
        |
        v
exposure-level classifier + optional CCD/tile localization output
```

### 3.2 Representations

Use three explicit representations:

1. `ccd`
   - Existing baseline mode.
   - One embedding per CCD row.
   - Kept for reproducibility and ablations.

2. `focalplane_stamp`
   - One low-resolution image per exposure.
   - CCDs are downsampled and placed in focal-plane layout.
   - Used for global artifacts: clouds, scattered light, focus, transparency,
     telescope motion, large-scale sky/background issues.

3. `exposure_multiscale`
   - One low-resolution focal-plane stamp embedding per exposure.
   - Top-K selected high-resolution CCD/tile embeddings per exposure.
   - Final model operates at exposure level.
   - This should become the recommended path.

## 4. Implementation Plan

### 4.1 Add Exposure-Level Dataset

Add a new dataset class, tentatively `DECamExposureDataset`, in
`src/decam_qa/dataset.py` or a new `src/decam_qa/exposure_dataset.py`.

Input:

- A CSV with the existing columns:
  - `image_filename`
  - `expnum`
  - `ccdnum`
  - `image_hdu`
  - `filter`
  - `reasons`
  - `vi_source`
- `image_dir`
- representation config

Behavior:

- Group rows by `expnum` and `image_filename`.
- Each `__getitem__` returns one exposure-level sample.
- Open the FITS file once with memory mapping where possible.
- Read all CCD HDUs referenced by the grouped rows.
- Return:
  - exposure identifier
  - exposure-level label bitmask
  - optional per-CCD labels
  - low-resolution focal-plane stamp tensor
  - selected high-resolution tile tensors
  - metadata for selected tiles

Exposure-level label:

- Preserve the original reason bitmask instead of collapsing to the first
  decoded reason.
- For binary good/bad, use `label_bad = reasons > 0`.
- For reason classification, use one-vs-rest multi-label targets over the
  reason bits.

Important behavior:

- Missing or unreadable CCD HDUs should not crash the whole exposure by
  default. Record the missing CCD in metadata and continue if at least one CCD
  is readable.
- If no CCD is readable, raise an explicit error for that exposure.
- The dataset should not apply random augmentation in the first version.

### 4.2 Build Low-Resolution Focal-Plane Stamp

Create a reusable function, tentatively:

```python
build_focalplane_stamp(
    hdul,
    exposure_rows,
    binsize=120,
    subtract_median_sky=True,
    reducer="median",
    fill_value=0.0,
)
```

Implementation details:

- Use the focal-plane positions from `decam_focalplane.py` /
  `decam_postage_stamps.py`.
- For each CCD:
  - read the image from the open HDU list
  - optionally subtract a robust median sky estimate
  - trim to a multiple of `binsize`
  - downsample using median by default
  - place the result into the low-resolution focal-plane canvas
- Replace non-finite values with `fill_value`.
- Return a single-channel tensor with shape `(1, H, W)`.

Default:

- `binsize = 120`
- `reducer = "median"`
- `subtract_median_sky = True`

Rationale:

- `binsize=120` turns the native focal plane into a few hundred pixels across,
  which is appropriate for one ViT pass.
- Median downsampling is slower than mean but more robust to stars, trails,
  and hot pixels. Because this is one pass per exposure, the robustness is
  worth it for the first version.

### 4.3 Add Cheap Candidate Selection

Before running high-resolution DINO on local crops, compute cheap anomaly
scores from the CCD data or downsampled CCD stamps.

For each CCD, compute:

- robust median background
- robust scatter, e.g. normalized MAD
- high-percentile value, e.g. p99 or p99.5
- low-percentile value, e.g. p0.5 or p1
- saturated or extreme-pixel fraction
- non-finite fraction
- local contrast score on the downsampled CCD

Create a scalar candidate score:

```text
score =
    z(abs(background - exposure_median_background))
  + z(scatter)
  + z(extreme_pixel_fraction)
  + z(local_contrast)
  + z(nonfinite_fraction)
```

Use robust z-scores within the exposure. Clip each component to a reasonable
range, for example `[-5, 5]`, before summing.

Default selection:

- Select top `K=8` CCDs or tiles per exposure.
- Always include at least:
  - the highest-scoring CCD
  - one central CCD
  - one edge CCD
- If fewer than `K` CCDs are available, use all readable CCDs.

The first implementation should select CCD-level crops, not fine-grained tiles,
to keep the system simple. Tile-level selection can be added after the
exposure-level path is working.

### 4.4 Add High-Resolution Local Embeddings

For selected CCDs, create a high-resolution DINO input.

Default local view:

- Preserve the CCD aspect ratio.
- Resize to a ViT-compatible shape such as `(2352, 1176)` or the current config
  value.
- Ensure both dimensions are multiples of 14 for DINOv2 ViT patching.
- Use single-channel input by converting the DINO patch projection from
  3-channel to 1-channel.

Preprocessing:

- Convert to `float32`.
- Apply robust percentile clipping.
- Apply either robust z-score or `asinh` scaling.
- Use the same preprocessing for train and inference.

Recommended first default:

```text
clip: p0.5 to p99.5 per CCD
scale: robust z-score using median and MAD
fill non-finite: 0 after scaling
```

Output metadata per local embedding:

- `expnum`
- `image_filename`
- `ccdnum`
- `image_hdu`
- `selection_score`
- `view_type = "ccd"`

### 4.5 Convert DINO to Single-Channel Input

Add a utility function in `src/decam_qa/embeddings.py`:

```python
def convert_patch_embed_to_single_channel(model):
    ...
```

Behavior:

- Replace `model.patch_embed.proj`, a 3-channel convolution, with a 1-channel
  convolution.
- Initialize the new weight as:

```python
new_weight = old_weight.sum(dim=1, keepdim=True)
```

- Copy the bias unchanged.

This is mathematically equivalent to feeding the same unnormalized image into
all three RGB channels. It avoids unnecessary channel handling and makes the
input representation honest.

Important caveat:

- If ImageNet per-channel RGB normalization is added later, the folding must
  include the normalization constants. The simple sum is exact only for the
  current replicated grayscale path without separate RGB normalization.

### 4.6 Generate and Store Multi-Scale Embeddings

Add a new embedding generation function, tentatively:

```python
generate_exposure_multiscale_embeddings(
    dataset,
    model,
    device,
    output_dir,
    batch_size=1,
    num_workers=2,
    top_k=8,
)
```

Storage format:

- Continue using HDF5 for compatibility.
- Store one group per exposure:

```text
exposures/
  exp_<expnum>/
    global
    local_000
    local_001
    ...
    metadata
```

Dataset contents:

- `global`: focal-plane stamp embedding, shape `(D,)`
- `local_N`: selected CCD/tile embedding, shape `(D,)`
- `metadata`: structured metadata or JSON string with selected CCDs and scores
- group attributes:
  - `image_filename`
  - `filter`
  - `reason_bitmask`
  - `num_readable_ccds`
  - `num_selected_views`

Do not overwrite existing embeddings unless an explicit `--overwrite` option is
set. For resume behavior, skip exposure groups that already contain a `global`
embedding and the expected number of local embeddings.

### 4.7 Aggregate Embeddings for Classification

Implement a deterministic first-version aggregation:

```text
exposure_feature =
  concat(
    global_embedding,
    mean(local_embeddings),
    max(local_embeddings),
    std(local_embeddings),
    top_selection_scores
  )
```

Details:

- If an exposure has no local embeddings, fill local aggregate vectors with
  zeros and set `num_selected_views = 0`.
- Pad or truncate `top_selection_scores` to length `K=8`.
- Standardize the final feature vector before classification.

Classifiers to compare:

1. Binary good/bad classifier:
   - logistic regression with class balancing
   - calibrated linear SVM as a secondary option

2. Multi-label reason classifier:
   - one-vs-rest logistic regression
   - one probability per reason bit

3. Baseline:
   - existing PCA + kNN pipeline

The first production candidate should be:

```text
frozen DINOv2 embeddings
+ deterministic exposure aggregation
+ class-balanced logistic regression
+ optional probability calibration
```

This is easier to interpret and validate than immediately fine-tuning a neural
aggregation head.

### 4.8 Add Configuration

Extend `configs/embed.yaml` or add `configs/exposure_multiscale.yaml`.

Recommended new config:

```yaml
representation: exposure_multiscale

model:
  size: base
  use_register: true
  single_channel: true

data:
  batch_size: 1
  num_workers: 4
  pin_memory: false

preprocessing:
  clip_percentiles: [0.5, 99.5]
  scale: robust_zscore
  fill_value: 0.0

focalplane_stamp:
  binsize: 120
  reducer: median
  subtract_median_sky: true

local_views:
  selection: ccd_anomaly_score
  top_k: 8
  crop_size: [2352, 1176]
  include_center_fallback: true
  include_edge_fallback: true

output:
  format: hdf5
  resume: true
  overwrite: false
```

### 4.9 CLI and Scripts

Add or extend a CLI command so the user can run:

```bash
python -m decam_qa.cli embed \
  --config configs/exposure_multiscale.yaml \
  --dset-path data/samples/test_supervised_ooi_dataset.csv \
  --dr dr10 \
  --output-dir /pscratch/sd/b/brookluo/decam-exposure/multiscale
```

SLURM behavior:

- Keep GPU execution on Perlmutter GPU nodes.
- Use `desi_g` for GPU jobs.
- Do not run full embedding jobs on login nodes.
- Keep one process per GPU if retaining the current multiprocessing pattern.
- Split by exposure, not by individual CCD row.

## 5. Evaluation Plan

### 5.1 Data Splits

Use exposure-grouped splits only:

- No CCDs from the same `expnum` may appear in both train and test.
- Prefer `StratifiedGroupKFold` when class balance allows it.
- Fall back to `GroupKFold` plus explicit reporting of class balance.

Recommended split keys:

- primary group: `expnum`
- optional stress tests:
  - split by observing night
  - split by filter
  - train on DR10, test on DR11 VI labels if available

### 5.2 Metrics

Report exposure-level metrics first:

- binary good/bad precision-recall
- average precision
- recall at fixed false-positive rate
- number of exposures sent to VI at a fixed threshold
- confusion matrix at selected operating points

Report reason-level metrics:

- one-vs-rest average precision per reason
- per-reason recall at fixed VI workload
- macro and weighted averages

Report localization diagnostics:

- whether selected top-K CCDs overlap human-flagged or visually suspicious CCDs
- HTML pages showing:
  - focal-plane stamp
  - selected CCDs
  - model probabilities
  - nearest training examples or top contributing selected views

### 5.3 Ablations

Run these ablations before deciding whether the new pipeline is better:

1. Existing CCD-only DINOv2 + PCA/kNN.
2. Existing CCD-only DINOv2 + logistic regression.
3. Low-resolution focal-plane stamp only.
4. Selected high-resolution CCD embeddings only.
5. Combined global + local multi-scale embeddings.
6. Combined embeddings with and without single-channel DINO patch projection.
7. Different candidate counts: `K = 4, 8, 16`.
8. Different focal-plane binsizes: `80, 120, 160`.

## 6. Testing Plan

Add focused tests rather than broad end-to-end GPU tests.

Dataset tests:

- Exposure dataset groups rows by `expnum` and `image_filename`.
- One item returns one exposure, not one CCD.
- Labels preserve the full reason bitmask.
- Missing optional CCDs are recorded and do not crash the exposure.
- Completely unreadable exposures raise a clear error.

Focal-plane stamp tests:

- Stamp shape is deterministic for a given binsize.
- CCDs are placed at expected relative positions.
- Non-finite values are replaced after downsampling.
- Median and mean reducers both work.

Candidate selection tests:

- Top-K selection is deterministic.
- Fallback center/edge CCDs are included when configured.
- Selection handles fewer than K readable CCDs.

Embedding tests:

- Single-channel patch projection accepts `(B, 1, H, W)`.
- Converted projection produces numerically equivalent output to replicated
  grayscale input for the patch-projection layer.
- HDF5 output stores one group per exposure.
- Resume skips completed exposure groups.

Classifier tests:

- Aggregation produces fixed-length features for variable numbers of selected
  local embeddings.
- Binary classifier trains on synthetic grouped exposure features.
- Multi-label reason classifier returns one probability per reason.

## 7. Acceptance Criteria

The implementation is complete when:

- The new pipeline can generate exposure-level multi-scale embeddings for a
  small CSV on a GPU node.
- HDF5 outputs are grouped by exposure and include global/local embeddings plus
  metadata.
- The classifier notebook or script can train on the new embedding format.
- Evaluation uses exposure-grouped splits.
- The new pipeline is compared against the current CCD-only baseline.
- HTML/plot diagnostics can show focal-plane stamps and selected local views.
- Documentation explains how to run the new path and how to reproduce the old
  baseline.

## 8. Risks and Mitigations

Risk: low-resolution stamps may miss small local defects.

- Mitigation: always include selected high-resolution CCD views and tune `K`.

Risk: cheap anomaly scoring may select the wrong CCDs.

- Mitigation: include center/edge fallback CCDs and evaluate selection overlap
  with VI examples.

Risk: median downsampling may be CPU-heavy.

- Mitigation: keep binsize configurable, cache stamps, and benchmark mean vs
  median.

Risk: exposure labels may be noisy at CCD level.

- Mitigation: train exposure-level classifiers first and treat CCD selection as
  localization/interpretability, not ground truth.

Risk: HDF5 schema changes could break existing notebooks.

- Mitigation: keep old CCD embedding reader unchanged and add a separate
  reader for exposure-level embeddings.

## 9. Recommended Implementation Order

1. Add single-channel DINO patch projection utility.
2. Add low-resolution focal-plane stamp builder with tests.
3. Add exposure-level dataset grouping and FITS-open-once behavior.
4. Add deterministic candidate scoring and top-K CCD selection.
5. Add multi-scale embedding generation and HDF5 schema.
6. Add exposure embedding reader and deterministic aggregation.
7. Add logistic-regression baseline classifier.
8. Add grouped evaluation metrics and ablation notebook/script.
9. Add SLURM config and run a small DR10 sample.
10. Compare against the current CCD-only DINOv2 + kNN baseline.

## 10. First Experiment Defaults

Use these defaults for the first serious run:

- model: `dinov2_vitb14_reg`
- model input: single-channel patch projection
- global stamp binsize: `120`
- global stamp reducer: median
- local selected views: top `K=8` CCDs
- local crop size: `[2352, 1176]`
- preprocessing: p0.5-p99.5 clipping plus robust z-score
- classifier: class-balanced logistic regression
- split: grouped by `expnum`
- baseline: current CCD-only PCA + kNN

