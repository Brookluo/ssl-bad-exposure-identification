# Multi-Scale Exposure Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-CCD DINOv2 + kNN pipeline with an exposure-grouped multi-scale pipeline using low-resolution focal-plane stamps, top-K selected high-resolution CCD embeddings, and class-balanced logistic regression for exposure-level classification.

**Architecture:** New `DECamExposureDataset` groups rows by exposure, opens FITS once, builds a low-res focal-plane stamp and computes anomaly scores to select top-K CCDs for high-res DINOv2. Single-channel DINO patch projection avoids 3-channel expansion. Deterministic aggregation (global + mean/max/std of locals + scores) feeds logistic regression with exposure-grouped cross-validation.

**Spec:** `docs/multiscale-exposure-identification-plan.md`

**Issues found in the design doc (fixed in this plan):**
1. `decam_postage_stamps.py:decam_postage_stamp()` calls `create_image()` which does not exist.
2. `read_image.py:read_image()` docstring says returns `ndarray` but actually returns `tuple(ndarray, list)`.
3. `DECamImageDataset` has no `original_df_idx` attribute (referenced by `embeddings.py`).
4. `decam_postage_stamps.py` duplicates `ccdname2num` from `decam_qa.info`.
5. Design proposes `StratifiedGroupKFold` which doesn't exist in sklearn.
6. No canonical focal-plane stamp function exists as a clean, testable unit — the logic is entangled in `decam_postage_stamp()` with plotting code.

**Design corrections applied in this plan:**
- Extract `build_focalplane_stamp()` as a standalone, testable function using coordinates from `decam_focalplane.py` and `ccdname2num` from `decam_qa.info`.
- Use `GroupKFold` + manual class-balance reporting instead of `StratifiedGroupKFold`.
- Fix `read_image()` docstring.
- Build the stamp function from first principles rather than depending on `decam_postage_stamps.py:create_image()`.

---

## Pre-existing issues to fix before starting

### Fix 1: `read_image.py` docstring mismatch

`read_image()` docstring says returns `numpy.ndarray` but actually returns `(half_imdata, ccdnames)`. Fix the docstring and add a return type annotation.

### Fix 2: `embeddings.py` per-sample write bug

In `generate_embeddings()`, the inner loop writes `embeds` (full batch) instead of `embeds[i]` (single sample). Also references `dataset.df_data.original_df_idx` which doesn't exist. Fix both.

### Fix 3: `decam_postage_stamps.py` missing `create_image`

The function `create_image()` is called but never defined. It should be implemented or the call removed (the plan extracts stamp building into a new standalone function anyway, so the old file won't be used — just note this as a preexisting issue in comments).

These fixes are committed as a separate preliminary commit before the main work begins.

---

### Task 1: Single-channel DINO patch projection utility

**Files:**
- Modify: `src/decam_qa/embeddings.py`
- Create: `tests/test_embeddings.py` (extend)

- [ ] **Step 1: Add fake_dino_model fixture to tests/conftest.py**

```python
@pytest.fixture
def fake_dino_model():
    """A minimal mock DINOv2 model with a real nn.Conv2d patch_embed.proj.

    Returns a real nn.Sequential as patch_embed so tests can verify
    weight manipulation. Does NOT mock torch.hub.load — use mock_torch_hub
    for tests that need to intercept model loading.
    """
    import torch
    import torch.nn as nn
    class FakePatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14, bias=True)
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = FakePatchEmbed()
            self.embed_dim = 768
    return FakeModel()
```

### Step 2: Write test for convert_patch_embed_to_single_channel

Extend `tests/test_embeddings.py` with:
```python
class TestSingleChannel:
    def test_convert_preserves_numerics(self, fake_dino_model):
        """Replicated grayscale via channel sum must match single-channel path."""
        import torch
        from decam_qa.embeddings import convert_patch_embed_to_single_channel

        model = fake_dino_model
        original_proj = model.patch_embed.proj
        convert_patch_embed_to_single_channel(model)

        # Create dummy 1-channel input
        x1 = torch.randn(1, 1, 224, 224)

        # Single-channel result
        out1 = model.patch_embed.proj(x1)

        # Manually compute what 3-channel would give: sum channels then project
        summed_weight = original_proj.weight.sum(dim=1, keepdim=True)
        expected = torch.nn.functional.conv2d(x1, summed_weight, original_proj.bias,
                                              stride=original_proj.stride,
                                              padding=original_proj.padding)
        assert torch.allclose(out1, expected, atol=1e-5)

    def test_convert_accepts_single_channel(self, fake_dino_model):
        """After conversion, model accepts (1, 1, H, W) input."""
        import torch
        from decam_qa.embeddings import convert_patch_embed_to_single_channel

        model = fake_dino_model
        convert_patch_embed_to_single_channel(model)

        x = torch.randn(2, 1, 224, 224)
        out = model.patch_embed.proj(x)
        assert out.shape[1] == model.embed_dim
        assert out.shape[1] == model.embed_dim
```

- [ ] **Step 2: Implement convert_patch_embed_to_single_channel**

```python
def convert_patch_embed_to_single_channel(model):
    """Replace 3-channel DINOv2 patch projection with 1-channel equivalent.

    Sums the existing 3-channel conv weights so a single-channel grayscale
    input produces identical patch embeddings. Copy bias unchanged.

    This is exact only when no per-channel RGB normalization is applied.

    Parameters
    ----------
    model : torch.nn.Module
        DINOv2 model with a ``patch_embed.proj`` Conv2d attribute.
    """
    old_conv = model.patch_embed.proj
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.patch_embed.proj = new_conv
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_embeddings.py::TestSingleChannel -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py src/decam_qa/embeddings.py tests/test_embeddings.py
git commit -m "feat: add convert_patch_embed_to_single_channel for 1-channel DINOv2 input"
```

---

### Task 2: Build low-resolution focal-plane stamp function

**Files:**
- Create: `src/decam_qa/focalplane.py`
- Create: `tests/test_focalplane.py`

- [ ] **Step 1: Write test_focalplane.py**

```python
"""Tests for decam_qa.focalplane — focal-plane stamp building."""
import numpy as np
import pytest
from astropy.io import fits
from decam_qa.focalplane import build_focalplane_stamp


@pytest.fixture
def multi_hdu_fits(tmp_path):
    """Create a FITS file with 3 CCD HDUs named N1, N2, N3."""
    rng = np.random.default_rng(42)
    fpath = tmp_path / "exposure.fits.fz"
    hdus = [fits.PrimaryHDU()]
    for name in ["N1", "N2", "N3"]:
        data = rng.normal(100, 10, (2046, 4094)).astype(np.float32)
        hdu = fits.ImageHDU(data, name=name)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(fpath)
    return fpath


class TestBuildFocalplaneStamp:
    def test_output_is_single_channel(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},  # N1
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 33, "image_hdu": 2},  # N2
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 34, "image_hdu": 3},  # N3
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120)
        assert stamp.ndim == 3
        assert stamp.shape[0] == 1
        hdul.close()

    def test_stamp_shape_deterministic(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        s1 = build_focalplane_stamp(hdul, rows, binsize=120)
        s2 = build_focalplane_stamp(hdul, rows, binsize=120)
        assert s1.shape == s2.shape
        np.testing.assert_array_equal(s1, s2)
        hdul.close()

    def test_binsize_changes_output_size(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        s120 = build_focalplane_stamp(hdul, rows, binsize=120)
        s60 = build_focalplane_stamp(hdul, rows, binsize=60)
        assert s120.shape[1] < s60.shape[1]
        hdul.close()

    def test_no_finite_values(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120, fill_value=0.0)
        assert np.all(np.isfinite(stamp))
        hdul.close()

    def test_mean_reducer_works(self, multi_hdu_fits):
        rows = [
            {"expnum": 1, "image_filename": str(multi_hdu_fits),
             "ccdnum": 32, "image_hdu": 1},
        ]
        hdul = fits.open(multi_hdu_fits)
        stamp = build_focalplane_stamp(hdul, rows, binsize=120, reducer="mean")
        assert np.all(np.isfinite(stamp))
        hdul.close()
```

- [ ] **Step 2: Implement build_focalplane_stamp in src/decam_qa/focalplane.py**

```python
"""DECam focal-plane stamp builder for multi-scale exposure analysis.

Builds low-resolution single-channel stamps from exposure HDUs by downsampling
CCDs and placing them in focal-plane layout.
"""
import numpy as np
from pathlib import Path
from decam_qa.info import ccdname2num


# Focal-plane layout: x_pix and y_pix are CCD center positions in native
# pixel grid (29590 x 26787). Indices are ordered by ccdnum_list:
# [1,2,3,...,60,62] (missing 61).
_CCD_NUM_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62]

_X_PIX = np.array([8498., 12747., 16996., 6373.5, 10622.5, 14871.5, 19120.5,
                   4249., 8498., 12747., 16996., 21245., 2124.5, 6373.5,
                   10622.5, 14871.5, 19120.5, 23369.5, 2124.5, 6373.5,
                   10622.5, 14871.5, 19120.5, 23369.5, 0., 4249., 8498.,
                   12747., 16996., 21245., 25494., 0., 4249., 8498., 12747.,
                   16996., 21245., 25494., 2124.5, 6373.5, 10622.5, 14871.5,
                   19120.5, 23369.5, 2124.5, 6373.5, 10622.5, 14871.5,
                   19120.5, 23369.5, 4249., 8498., 12747., 16996., 21245.,
                   6373.5, 10622.5, 14871.5, 19120.5, 8498., 16996.])

_Y_PIX = np.array([0, 0, 0, 2249, 2249, 2249, 2249, 4498, 4498, 4498, 4498,
                   4498, 6747, 6747, 6747, 6747, 6747, 6747, 8996, 8996,
                   8996, 8996, 8996, 8996, 11245, 11245, 11245, 11245,
                   11245, 11245, 11245, 13494, 13494, 13494, 13494, 13494,
                   13494, 13494, 15743, 15743, 15743, 15743, 15743, 15743,
                   17992, 17992, 17992, 17992, 17992, 17992, 20241, 20241,
                   20241, 20241, 20241, 22490, 22490, 22490, 22490, 24739,
                   24739.])

_NATIVE_X = 29590
_NATIVE_Y = 26787
_CCD_WIDTH = 4094
_CCD_HEIGHT = 2046

# Pre-build mapping: ccdnum -> focal-plane row index
_CCDNUM_TO_IDX = {num: i for i, num in enumerate(_CCD_NUM_LIST)}


def build_focalplane_stamp(hdul, exposure_rows, binsize=120,
                           subtract_median_sky=True, reducer="median",
                           fill_value=0.0):
    """Build a low-resolution focal-plane stamp from exposure HDUs.

    For each CCD referenced in exposure_rows: read from the open HDUList,
    subtract sky, trim to a multiple of binsize, downsample, and place
    into a focal-plane canvas.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS HDUList for the exposure (caller manages open/close).
    exposure_rows : list of dict or pd.DataFrame
        Must have keys: ccdnum, image_hdu.
    binsize : int
        Downsampling factor. Native CCD -> ~(CCD_WIDTH/binsize, CCD_HEIGHT/binsize).
    subtract_median_sky : bool
        If True, subtract median sky from each CCD before downsampling.
    reducer : str
        'median' or 'mean'. Downsampling reducer.
    fill_value : float
        Value for empty regions and non-finite pixels.

    Returns
    -------
    np.ndarray
        Single-channel stamp with shape (1, out_h, out_w).
    """
    stamp_h = _NATIVE_Y // binsize + 1
    stamp_w = _NATIVE_X // binsize + 1
    stamp = np.full((stamp_h, stamp_w), fill_value, dtype=np.float32)

    # Normalize input: DataFrame -> list of dicts
    if hasattr(exposure_rows, "columns"):  # pd.DataFrame
        exposure_rows = exposure_rows.to_dict("records")

    for row in exposure_rows:
        ccdnum = row["ccdnum"]
        hdu_idx = row["image_hdu"]

        if ccdnum not in _CCDNUM_TO_IDX:
            continue

        try:
            img = np.asarray(hdul[hdu_idx].data, dtype=np.float32)
        except (IndexError, KeyError, OSError):
            continue

        if subtract_median_sky:
            finite = img[np.isfinite(img)]
            if len(finite) > 0:
                img = img - np.median(finite)

        # Trim to multiple of binsize
        trim_h = (img.shape[0] // binsize) * binsize
        trim_w = (img.shape[1] // binsize) * binsize
        img = img[:trim_h, :trim_w]

        # Downsample via reshaping
        if reducer == "median":
            reshaped = img.reshape(trim_h // binsize, binsize,
                                   trim_w // binsize, binsize)
            downsampled = np.median(reshaped, axis=(1, 3))
        else:  # mean
            reshaped = img.reshape(trim_h // binsize, binsize,
                                   trim_w // binsize, binsize)
            downsampled = np.mean(reshaped, axis=(1, 3))

        # Place into canvas
        fp_idx = _CCDNUM_TO_IDX[ccdnum]
        x0 = int(_X_PIX[fp_idx] // binsize)
        y0 = int(_Y_PIX[fp_idx] // binsize)
        cc_h = downsampled.shape[0]
        cc_w = downsampled.shape[1]

        # Clip to canvas boundaries
        y_end = min(y0 + cc_h, stamp_h)
        x_end = min(x0 + cc_w, stamp_w)
        cc_h = y_end - y0
        cc_w = x_end - x0

        stamp[y0:y_end, x0:x_end] = downsampled[:cc_h, :cc_w]

    # Replace non-finite values
    stamp[~np.isfinite(stamp)] = fill_value

    return stamp[np.newaxis, :, :]
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_focalplane.py -v
```
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/focalplane.py tests/test_focalplane.py
git commit -m "feat: add build_focalplane_stamp for low-res exposure stamps"
```

---

### Task 3: CCD anomaly scoring and top-K selection

**Files:**
- Create: `src/decam_qa/selection.py`
- Create: `tests/test_selection.py`

- [ ] **Step 1: Write test_selection.py**

```python
"""Tests for decam_qa.selection — CCD anomaly scoring and top-K selection."""
import numpy as np
import pytest
from decam_qa.selection import compute_anomaly_scores, select_top_k_ccds


@pytest.fixture
def sample_ccds():
    """3 CCDs: one normal, one with high scatter, one with extreme pixels."""
    rng = np.random.default_rng(42)
    return [
        rng.normal(100, 5, (2046, 4094)).astype(np.float32),   # normal
        rng.normal(100, 30, (2046, 4094)).astype(np.float32),  # high scatter
        np.full((2046, 4094), 1e5, dtype=np.float32),           # extreme
    ]


@pytest.fixture
def sample_rows():
    return [
        {"ccdnum": 25, "image_hdu": 1},
        {"ccdnum": 26, "image_hdu": 2},
        {"ccdnum": 27, "image_hdu": 3},
    ]


class TestAnomalyScores:
    def test_returns_array_of_scores(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_extreme_ccd_gets_high_score(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        assert scores[2] > scores[1]
        assert scores[2] > scores[0]

    def test_high_scatter_above_normal(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        assert scores[1] > scores[0]

    def test_all_nan_returns_zero(self, sample_rows):
        ccds = [np.full((100, 100), np.nan, dtype=np.float32)]
        scores = compute_anomaly_scores(ccds, [sample_rows[0]])
        assert scores[0] == 0.0


class TestSelectTopK:
    def test_selects_k_when_enough(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        selected = select_top_k_ccds(scores, sample_rows, k=2)
        assert len(selected) == 2

    def test_selects_all_when_fewer_than_k(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        selected = select_top_k_ccds(scores, sample_rows, k=10)
        assert len(selected) == 3

    def test_includes_highest_scoring(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        selected = select_top_k_ccds(scores, sample_rows, k=1)
        assert selected[0]["ccdnum"] == 27

    def test_deterministic(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        s1 = select_top_k_ccds(scores, sample_rows, k=2)
        s2 = select_top_k_ccds(scores, sample_rows, k=2)
        assert [r["ccdnum"] for r in s1] == [r["ccdnum"] for r in s2]

    def test_include_fallbacks_adds_center_and_edge(self, sample_ccds, sample_rows):
        scores = compute_anomaly_scores(sample_ccds, sample_rows)
        selected = select_top_k_ccds(
            scores, sample_rows, k=5,
            include_center_fallback=True,
            include_edge_fallback=True,
        )
        # Never exceeds k. Only 3 CCDs available total.
        assert len(selected) == min(5, len(sample_rows))
        ccdnums = [r["ccdnum"] for r in selected]
        assert 27 in ccdnums  # top-scoring is always included

    def test_handles_empty_input(self):
        selected = select_top_k_ccds(np.array([]), [], k=8)
        assert selected == []
```

- [ ] **Step 2: Implement src/decam_qa/selection.py**

```python
"""CCD anomaly scoring and top-K selection for multi-scale pipeline."""
import numpy as np
from decam_qa.info import ccdname2num

_CENTER_CCD = ccdname2num.get("S1", 25)
_EDGE_CCDS = {ccdname2num.get("S31", 3), ccdname2num.get("N31", 62)}


def compute_anomaly_scores(ccd_images, ccd_rows):
    """Compute cheap anomaly scores for each CCD from image statistics.

    Scores combine: background deviation, scatter, extreme pixel fraction,
    and non-finite fraction. Higher score = more anomalous.

    Parameters
    ----------
    ccd_images : list of np.ndarray
        List of CCD image arrays.
    ccd_rows : list of dict
        Metadata rows matching ccd_images. Must have 'ccdnum' key.

    Returns
    -------
    np.ndarray
        Score per CCD, shape (n_ccds,). Higher = more anomalous.
    """
    n = len(ccd_images)
    backgrounds = np.zeros(n)
    scatters = np.zeros(n)
    extreme_fracs = np.zeros(n)
    nonfinite_fracs = np.zeros(n)

    for i, img in enumerate(ccd_images):
        finite = img[np.isfinite(img)]
        if len(finite) == 0:
            backgrounds[i] = 0
            scatters[i] = 0
            extreme_fracs[i] = 0
            nonfinite_fracs[i] = 1.0
            continue

        backgrounds[i] = np.median(finite)
        scatters[i] = 1.4826 * np.median(np.abs(finite - backgrounds[i]))
        p99 = np.percentile(finite, 99.5)
        p01 = np.percentile(finite, 0.5)
        extreme_fracs[i] = np.mean((finite > p99) | (finite < p01))
        nonfinite_fracs[i] = 1.0 - len(finite) / img.size

    # Compute robust z-scores within the exposure
    def robust_z(values):
        med = np.median(values)
        mad = 1.4826 * np.median(np.abs(values - med))
        if mad == 0:
            return np.zeros_like(values)
        return (values - med) / mad

    scores = np.zeros(n)
    # Use abs(z-score) for all components so anomalies always add and never cancel.
    # scatter, extreme_frac, nonfinite_frac are always "higher is worse".
    # For background, deviation from the exposure median in either direction is anomalous.
    scores += np.clip(np.abs(robust_z(backgrounds)), 0, 5)
    scores += np.clip(np.abs(robust_z(scatters)), 0, 5)
    scores += np.clip(np.abs(robust_z(extreme_fracs)), 0, 5)
    scores += np.clip(np.abs(robust_z(nonfinite_fracs)), 0, 5)

    return scores


def select_top_k_ccds(scores, ccd_rows, k=8,
                      include_center_fallback=True,
                      include_edge_fallback=True):
    """Select top-K CCDs by anomaly score with optional fallback CCDs.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores per CCD.
    ccd_rows : list of dict
        Metadata rows. Must have 'ccdnum' key.
    k : int
        Maximum number of CCDs to select.
    include_center_fallback : bool
        If True, always include one central CCD if not already selected.
    include_edge_fallback : bool
        If True, always include one edge CCD if not already selected.

    Returns
    -------
    list of dict
        Selected CCD rows, sorted by descending score.
    """
    if len(scores) == 0:
        return []

    # Sort by descending score
    order = np.argsort(scores)[::-1]
    effective_k = min(k, len(order))

    # Build a priority queue: top-K by score first, then fallbacks.
    # Fallbacks can only displace lower-scoring items within k, never exceed k.
    # Never evict the highest-scoring CCD.
    selected_indices = set(order[:effective_k].tolist())

    def _lowest_selected():
        """Return index of lowest-scoring selected item, excluding the top scorer."""
        scored = [(scores[i], i) for i in selected_indices]
        scored.sort()
        # scored[0] is lowest, scored[-1] is highest. Don't evict the highest.
        if len(scored) <= 1:
            return None
        return scored[0][1]

    # Add center fallback: if missing, displace the lowest-scoring (not top)
    if include_center_fallback and len(selected_indices) > 0:
        has_center = any(ccd_rows[i]["ccdnum"] == _CENTER_CCD for i in selected_indices)
        if not has_center:
            for i, row in enumerate(ccd_rows):
                if row["ccdnum"] == _CENTER_CCD and i not in selected_indices:
                    victim = _lowest_selected()
                    if victim is not None:
                        selected_indices.discard(victim)
                        selected_indices.add(i)
                    break

    # Add edge fallback: same logic, recompute lowest after center change
    if include_edge_fallback and len(selected_indices) > 0:
        has_edge = any(ccd_rows[i]["ccdnum"] in _EDGE_CCDS for i in selected_indices)
        if not has_edge:
            for i, row in enumerate(ccd_rows):
                if row["ccdnum"] in _EDGE_CCDS and i not in selected_indices:
                    victim = _lowest_selected()
                    if victim is not None:
                        selected_indices.discard(victim)
                        selected_indices.add(i)
                    break

    result = []
    for i in order:
        if i in selected_indices:
            entry = dict(ccd_rows[i])
            entry["selection_score"] = float(scores[i])
            result.append(entry)
            selected_indices.discard(i)

    return result
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_selection.py -v
```
Expected: 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/selection.py tests/test_selection.py
git commit -m "feat: add CCD anomaly scoring and top-K selection"
```

---

### Task 4: DECamExposureDataset

**Files:**
- Modify: `src/decam_qa/dataset.py` (add new class)
- Create: `tests/test_exposure_dataset.py`

- [ ] **Step 1: Write test_exposure_dataset.py**

```python
"""Tests for decam_qa.dataset — DECamExposureDataset."""
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from pathlib import Path
from decam_qa.dataset import DECamExposureDataset


@pytest.fixture
def exposure_fits(tmp_path):
    """FITS file with 2 named CCD HDUs."""
    rng = np.random.default_rng(42)
    fpath = tmp_path / "exp.fits.fz"
    hdus = [fits.PrimaryHDU()]
    for name in ["N1", "N2"]:
        data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
        hdus.append(fits.ImageHDU(data, name=name))
    fits.HDUList(hdus).writeto(fpath)
    return fpath


@pytest.fixture
def exposure_csv(tmp_path, exposure_fits):
    """CSV with 2 CCD rows from same exposure."""
    df = pd.DataFrame({
        "image_filename": [str(exposure_fits), str(exposure_fits)],
        "expnum": [1, 1],
        "ccdnum": [32, 33],
        "image_hdu": [1, 2],
        "filter": [1, 1],
        "reasons": [1, 4],
        "vi_source": [0, 0],
    })
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    return path


class TestExposureDataset:
    def test_len_equals_exposure_count(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                   binsize=120)
        assert len(ds) == 1

    def test_getitem_returns_dict(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                   binsize=120)
        item = ds[0]
        assert isinstance(item, dict)
        assert "expnum" in item
        assert item["expnum"] == 1

    def test_label_is_exposure_reason_bitmask(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                   binsize=120)
        item = ds[0]
        # reasons 1 (bit 0) | 4 (bit 2) = bitmask 5
        assert item["reason_bitmask"] == 5

    def test_global_stamp_present(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                   binsize=120)
        item = ds[0]
        assert item["global_stamp"] is not None
        assert item["global_stamp"].ndim == 3
        assert item["global_stamp"].shape[0] == 1

    def test_selected_ccds_present(self, exposure_csv, exposure_fits):
        ds = DECamExposureDataset(str(exposure_csv), str(exposure_fits.parent),
                                   binsize=120, top_k=8)
        item = ds[0]
        assert "selected_ccds" in item
        assert "selected_images" in item
        assert len(item["selected_ccds"]) == 2  # only 2 CCDs available

    def test_two_exposures(self, tmp_path):
        """Two distinct exposures, each with one CCD."""
        rng = np.random.default_rng(42)
        dfs = []
        for eid, hdu_name in enumerate(["N1", "N2"], 1):
            fpath = tmp_path / f"exp{eid}.fits.fz"
            data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
            fits.HDUList([fits.PrimaryHDU(),
                          fits.ImageHDU(data, name=hdu_name)]).writeto(fpath)
            df = pd.DataFrame({
                "image_filename": [str(fpath)],
                "expnum": [eid],
                "ccdnum": [32],
                "image_hdu": [1],
                "filter": [1],
                "reasons": [0],
                "vi_source": [0],
            })
            dfs.append(df)
        csv_path = tmp_path / "two_exp.csv"
        pd.concat(dfs).to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        assert len(ds) == 2
        e1 = ds[0]
        e2 = ds[1]
        assert e1["expnum"] == 1
        assert e2["expnum"] == 2

    def test_missing_ccd_does_not_crash(self, tmp_path):
        """CCD with bad HDU index should be skipped, not crash."""
        rng = np.random.default_rng(42)
        fpath = tmp_path / "exp.fits.fz"
        data = rng.normal(100, 5, (2046, 4094)).astype(np.float32)
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(data, name="N1")]).writeto(fpath)

        df = pd.DataFrame({
            "image_filename": [str(fpath), str(fpath)],
            "expnum": [1, 1],
            "ccdnum": [32, 99],
            "image_hdu": [1, 999],  # second HDU doesn't exist
            "filter": [1, 1],
            "reasons": [0, 0],
            "vi_source": [0, 0],
        })
        csv_path = tmp_path / "missing_ccd.csv"
        df.to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        item = ds[0]
        assert item["num_readable_ccds"] == 1

    def test_completely_unreadable_exposure_raises(self, tmp_path):
        """All HDUs invalid -> raise error."""
        fpath = tmp_path / "bad.fits.fz"
        fits.HDUList([fits.PrimaryHDU()]).writeto(fpath)

        df = pd.DataFrame({
            "image_filename": [str(fpath)],
            "expnum": [1],
            "ccdnum": [32],
            "image_hdu": [999],
            "filter": [1],
            "reasons": [0],
            "vi_source": [0],
        })
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        ds = DECamExposureDataset(str(csv_path), str(tmp_path), binsize=120)
        with pytest.raises(RuntimeError, match="No readable CCDs"):
            ds[0]
```

- [ ] **Step 2: Implement DECamExposureDataset**

Add to `src/decam_qa/dataset.py` (keep existing `DECamImageDataset`). The new class groups rows by exposure, opens FITS once, builds the stamp, computes scores, selects top-K, and returns a dict:

```python
class DECamExposureDataset:
    """Group CCD rows by exposure for multi-scale analysis.

    Each __getitem__ returns a dict with:
    - expnum, image_filename, filter, reason_bitmask
    - global_stamp: low-res focal-plane stamp (1, H, W)
    - selected_ccds: list of row dicts for top-K CCDs with scores
    - selected_images: list of CCD image arrays
    - num_readable_ccds
    """
    def __init__(self, dataset_path, image_dir, binsize=120,
                 top_k=8, include_center_fallback=True,
                 include_edge_fallback=True, transform=None):
        ...
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_exposure_dataset.py -v
```
Expected: 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/dataset.py tests/test_exposure_dataset.py
git commit -m "feat: add DECamExposureDataset for exposure-grouped multi-scale data"
```

---

### Task 5: Multi-scale embedding generation

**Files:**
- Modify: `src/decam_qa/embeddings.py`
- Create: `tests/test_multiscale_embeddings.py`

- [ ] **Step 1: Write test_multiscale_embeddings.py**

```python
"""Tests for multi-scale embedding generation."""
import numpy as np
import h5py
import pytest
from unittest.mock import patch, MagicMock
from decam_qa.embeddings import generate_exposure_multiscale_embeddings


class TestMultiscaleEmbeddings:
    def test_creates_exposure_group_in_hdf5(self, tmp_path):
        model = MagicMock()
        model.forward.return_value = MagicMock()
        model.forward.return_value.numpy.return_value = np.zeros((1, 768), dtype=np.float32)

        dataset = MagicMock()
        dataset.__len__.return_value = 2
        mock_item = {
            "expnum": 1,
            "image_filename": "exp.fits.fz",
            "filter": 1,
            "reason_bitmask": 0,
            "global_stamp": np.zeros((1, 224, 224), dtype=np.float32),
            "selected_ccds": [
                {"ccdnum": 25, "image_hdu": 1, "selection_score": 3.5},
                {"ccdnum": 26, "image_hdu": 2, "selection_score": 2.1},
            ],
            "selected_images": [
                np.zeros((2046, 4094), dtype=np.float32),
                np.zeros((2046, 4094), dtype=np.float32),
            ],
            "num_readable_ccds": 60,
        }
        mock_item_2 = dict(mock_item)
        mock_item_2["expnum"] = 2
        dataset.__getitem__.side_effect = [mock_item, mock_item_2]

        with patch("torch.utils.data.DataLoader"):
            generate_exposure_multiscale_embeddings(
                dataset, model, "cpu", str(tmp_path),
                batch_size=1, num_workers=0, top_k=8)

        embeds_dir = tmp_path / "embeds_out"
        h5_files = list(embeds_dir.glob("*.h5"))
        assert len(h5_files) == 1
        with h5py.File(h5_files[0], 'r') as f:
            assert "exposures" in f
            assert "exp_1" in f["exposures"]
            assert "exp_2" in f["exposures"]
            g1 = f["exposures"]["exp_1"]
            assert "global" in g1
            assert "local_000" in g1
            assert "local_001" in g1
            assert "metadata" in g1

    def test_resume_skips_completed(self, tmp_path):
        embeds_dir = tmp_path / "embeds_out"
        embeds_dir.mkdir()
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'w') as f:
            g = f.create_group("exposures")
            eg = g.create_group("exp_1")
            eg.create_dataset("global", data=np.zeros(768))
            eg.create_dataset("local_000", data=np.zeros(768))
            eg.create_dataset("local_001", data=np.zeros(768))
            import json
            eg.create_dataset("metadata", data=json.dumps({
                "num_selected_views": 2, "filter": 1}))

        model = MagicMock()
        model.forward.return_value = MagicMock()
        model.forward.return_value.numpy.return_value = np.zeros((1, 768), dtype=np.float32)

        dataset = MagicMock()
        dataset.__len__.return_value = 2

        # exp_1 is complete (global + 2 locals + metadata), so skip it.
        # exp_2 should be generated.
        with patch("torch.utils.data.DataLoader"):
            generate_exposure_multiscale_embeddings(
                dataset, model, "cpu", str(tmp_path),
                batch_size=1, num_workers=0, top_k=8, resume=True)
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'r') as f:
            assert "exp_1" in f["exposures"]

    def test_resume_regenerates_partial_group(self, tmp_path):
        """Partial group (global but missing locals) gets regenerated."""
        import json
        embeds_dir = tmp_path / "embeds_out"
        embeds_dir.mkdir()
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'w') as f:
            g = f.create_group("exposures")
            eg = g.create_group("exp_1")
            eg.create_dataset("global", data=np.zeros(768))
            # Only 1 local, but metadata says 2 selected
            eg.create_dataset("local_000", data=np.zeros(768))
            eg.create_dataset("metadata", data=json.dumps({
                "num_selected_views": 2, "filter": 1}))

        model = MagicMock()
        model.forward.return_value = MagicMock()
        model.forward.return_value.numpy.return_value = np.zeros((1, 768), dtype=np.float32)

        dataset = MagicMock()
        dataset.__len__.return_value = 1
        mock_item = {
            "expnum": 1, "image_filename": "exp.fits.fz", "filter": 1,
            "reason_bitmask": 0,
            "global_stamp": np.zeros((1, 224, 224), dtype=np.float32),
            "selected_ccds": [
                {"ccdnum": 25, "image_hdu": 1, "selection_score": 3.5},
                {"ccdnum": 26, "image_hdu": 2, "selection_score": 2.1},
            ],
            "selected_images": [
                np.zeros((2046, 4094)), np.zeros((2046, 4094))],
            "num_readable_ccds": 60,
        }
        dataset.__getitem__.return_value = mock_item

        # Partial group should be deleted and regenerated
        with patch("torch.utils.data.DataLoader"):
            generate_exposure_multiscale_embeddings(
                dataset, model, "cpu", str(tmp_path),
                batch_size=1, num_workers=0, top_k=8, resume=True)
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'r') as f:
            eg = f["exposures"]["exp_1"]
            assert "local_001" in eg  # now complete
```

- [ ] **Step 2: Implement generate_exposure_multiscale_embeddings**

Add to `src/decam_qa/embeddings.py`:
```python
def generate_exposure_multiscale_embeddings(
    dataset, model, device, output_dir,
    batch_size=1, num_workers=2, top_k=8,
    crop_size=None, resume=False, overwrite=False,
):
    """Generate multi-scale embeddings for exposure-grouped data.

    For each exposure: embed the global stamp and top-K local views.
    Store one HDF5 group per exposure with global, local_N, and metadata.

    Parameters
    ----------
    dataset : DECamExposureDataset
    model : torch.nn.Module
    device : str
    output_dir : str
    batch_size : int
    num_workers : int
    top_k : int
    crop_size : tuple or None
        If None, resizes local crops to (2352, 1176).
    resume : bool
        Skip exposure groups that are already complete (global + expected number
        of locals + metadata). Partially-written groups (e.g., global present
        but missing locals after an interrupted run) are deleted and regenerated.
    overwrite : bool
        If True and resume=False, overwrite existing.
    """
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_multiscale_embeddings.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/embeddings.py tests/test_multiscale_embeddings.py
git commit -m "feat: add multi-scale embedding generation with HDF5 exposure groups"
```

---

### Task 6: Embedding aggregation + classifier

**Files:**
- Modify: `src/decam_qa/classifier.py` (add aggregation + logistic regression)
- Create: `tests/test_multiscale_classifier.py`

- [ ] **Step 1: Write test_multiscale_classifier.py**

```python
"""Tests for multi-scale embedding aggregation and classification."""
import numpy as np
import pytest
from decam_qa.classifier import (
    aggregate_exposure_embeddings,
    train_logistic_binary,
    predict_binary,
    train_multilabel_reason,
    predict_multilabel,
)


class TestAggregateExposureEmbeddings:
    def test_fixed_length_output(self):
        rng = np.random.default_rng(42)
        global_emb = rng.normal(0, 1, 768).astype(np.float32)
        local_embs = [rng.normal(0, 1, 768).astype(np.float32) for _ in range(8)]
        scores = [3.5, 2.1, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0]

        features = aggregate_exposure_embeddings(global_emb, local_embs, scores, k=8)
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        # Expected: global(768) + mean(768) + max(768) + std(768) + scores(8)
        # = 768*4 + 8 = 3080
        expected_dim = 768 * 4 + 8
        assert len(features) == expected_dim

    def test_zero_locals_fills_zeros(self):
        rng = np.random.default_rng(42)
        global_emb = rng.normal(0, 1, 768).astype(np.float32)

        features = aggregate_exposure_embeddings(global_emb, [], [], k=8)
        # local aggregates should be zeros
        local_start = 768  # after global
        assert np.all(features[local_start:local_start + 768 * 3] == 0)

    def test_fewer_than_k_locals(self):
        rng = np.random.default_rng(42)
        global_emb = rng.normal(0, 1, 768).astype(np.float32)
        local_embs = [rng.normal(0, 1, 768).astype(np.float32) for _ in range(3)]
        scores = [1.0, 2.0, 3.0]

        features = aggregate_exposure_embeddings(global_emb, local_embs, scores, k=8)
        expected_dim = 768 * 4 + 8
        assert len(features) == expected_dim


class TestBinaryClassifier:
    def test_train_on_synthetic_data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 200))
        y = np.zeros(n, dtype=int)
        y[n//2:] = 1  # 50 good, 50 bad

        model = train_logistic_binary(X, y, class_balanced=True, random_state=42)
        probs = predict_binary(model, X)
        assert len(probs) == n
        assert np.all((probs >= 0) & (probs <= 1))

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 200))
        y = np.zeros(100, dtype=int)
        y[50:] = 1

        m1 = train_logistic_binary(X, y, random_state=42)
        m2 = train_logistic_binary(X, y, random_state=42)
        p1 = predict_binary(m1, X)
        p2 = predict_binary(m2, X)
        np.testing.assert_array_equal(p1, p2)


class TestMultiReasonClassifier:
    def test_train_on_synthetic_multilabel(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 200))
        # 3 reason bits: bit 0 (Saturated), bit 1 (Clouds), bit 2 (PSF)
        pattern = [0, 1, 2, 4, 0, 1, 1|2, 2|4]
        y_multilabel = np.array(pattern * (n // len(pattern)), dtype=int)[:n]

        model = train_multilabel_reason(X, y_multilabel, n_reasons=15,
                                         class_balanced=True, random_state=42)
        probs = predict_multilabel(model, X)
        assert probs.shape == (n, 15)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_multilabel_deterministic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 200))
        y = np.array([0, 1, 2, 0, 1] * 10, dtype=int)

        m1 = train_multilabel_reason(X, y, n_reasons=15, random_state=42)
        m2 = train_multilabel_reason(X, y, n_reasons=15, random_state=42)
        p1 = predict_multilabel(m1, X)
        p2 = predict_multilabel(m2, X)
        np.testing.assert_array_almost_equal(p1, p2)
```

- [ ] **Step 2: Implement aggregation and classifiers**

Add to `src/decam_qa/classifier.py`:
- `aggregate_exposure_embeddings(global_emb, local_embs, scores, k=8)` — returns fixed-length feature vector
- `train_logistic_binary(X, y, class_balanced=True, random_state=42)` — returns fitted `LogisticRegression`
- `predict_binary(model, X)` — returns probability of "bad" class
- `train_multilabel_reason(X, y_bitmask, n_reasons=15, class_balanced=True, random_state=42)` — fits one `OneVsRestClassifier(LogisticRegression)` per reason bit. Converts bitmask labels `y_bitmask` into a multi-label binary matrix of shape `(n_samples, n_reasons)`.
- `predict_multilabel(model, X)` — returns per-reason probabilities, shape `(n_samples, n_reasons)`

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_multiscale_classifier.py -v
```
Expected: 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/classifier.py tests/test_multiscale_classifier.py
git commit -m "feat: add multi-scale aggregation, binary + multilabel logistic regression"
```

---

### Task 7: HDF5 reader for multi-scale embeddings

**Files:**
- Modify: `src/decam_qa/io.py`
- Create: `tests/test_multiscale_io.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for multi-scale HDF5 I/O."""
import numpy as np
import h5py
import pytest
from decam_qa.io import read_exposure_embeddings


class TestReadExposureEmbeddings:
    def test_reads_global_and_locals(self, tmp_path):
        fpath = tmp_path / "0_worker_embeds.h5"  # must match glob pattern
        rng = np.random.default_rng(42)
        with h5py.File(fpath, 'w') as f:
            grp = f.create_group("exposures")
            for eid in [1, 2]:
                eg = grp.create_group(f"exp_{eid}")
                eg.create_dataset("global", data=rng.normal(0, 1, 768).astype(np.float32))
                for i in range(3):
                    eg.create_dataset(f"local_{i:03d}", data=rng.normal(0, 1, 768).astype(np.float32))
                import json
                eg.create_dataset("metadata", data=json.dumps({"filter": 1}))

        result = read_exposure_embeddings(str(tmp_path))  # directory, merges all workers
        assert len(result) == 2
        for expnum, data in result.items():
            assert "global" in data
            assert len(data["locals"]) > 0
            assert "metadata" in data

    def test_merges_across_workers(self, tmp_path):
        """Two worker files with different exposures should be merged."""
        rng = np.random.default_rng(42)
        for w in range(2):
            fpath = tmp_path / f"{w}_worker_embeds.h5"
            with h5py.File(fpath, 'w') as f:
                grp = f.create_group("exposures")
                eid = w + 1  # worker 0 has exp_1, worker 1 has exp_2
                eg = grp.create_group(f"exp_{eid}")
                eg.create_dataset("global", data=rng.normal(0, 1, 768).astype(np.float32))
                eg.create_dataset("local_000", data=rng.normal(0, 1, 768).astype(np.float32))
                import json
                eg.create_dataset("metadata", data=json.dumps({"filter": 1}))

        result = read_exposure_embeddings(str(tmp_path))
        assert len(result) == 2
        assert 1 in result and 2 in result
```

- [ ] **Step 2: Implement read_exposure_embeddings**

Add to `src/decam_qa/io.py`:
```python
def read_exposure_embeddings(h5_dir_or_path):
    """Read multi-scale exposure embeddings from HDF5 directory or file.

    Accepts either a directory (globs ``*.h5``, merges across all
    workers — matching read_embeddings behavior) or a single file path.

    Returns dict: {expnum: {"global": array, "locals": [arrays], "metadata": dict}}
    """
```

- [ ] **Step 3: Commit**

```bash
git add src/decam_qa/io.py tests/test_multiscale_io.py
git commit -m "feat: add read_exposure_embeddings for multi-scale HDF5 format"
```

---

### Task 8: Multi-scale config + CLI integration

**Files:**
- Create: `configs/exposure_multiscale.yaml`
- Modify: `src/decam_qa/config.py`
- Modify: `src/decam_qa/cli.py`

- [ ] **Step 1: Add "exposure_multiscale" as a config stage**

In `config.py`, add `_REQUIRED_KEYS["exposure_multiscale"]` and defaults.

- [ ] **Step 2: Create configs/exposure_multiscale.yaml**

Per the spec design doc (Section 4.8).

- [ ] **Step 3: Extend the CLI `embed` subcommand**

Add `--output-dir` to the embed subcommand parser in `cli.py`:
```python
embed_parser.add_argument("--output-dir", default=None,
                          help="Override scratch_dir from config")
```

Update `main()` in `cli.py` so that when the config has `representation: exposure_multiscale`,
the `embed` subcommand dispatches to the multi-scale pipeline instead of the CCD-only one:
```python
elif args.command == "embed":
    from decam_qa.config import load_config
    config = load_config(args.config, "embed")

    # Check representation to decide which pipeline to use
    representation = config.get("representation", "ccd")

    if representation == "exposure_multiscale":
        from decam_qa.dataset import DECamExposureDataset
        from decam_qa.embeddings import (
            create_model, generate_exposure_multiscale_embeddings,
            convert_patch_embed_to_single_channel,
        )
        scratch = args.output_dir or config.get("scratch_dir", "./output")
        image_roots = config.get("image_roots", {})
        imdir = image_roots.get(args.dr, ".")

        ds = DECamExposureDataset(
            args.dset, imdir,
            binsize=config.get("focalplane_stamp", {}).get("binsize", 120),
            top_k=config.get("local_views", {}).get("top_k", 8),
        )
        model = create_model(config["model"]["size"], config["model"]["use_register"])
        if config["model"].get("single_channel", True):
            convert_patch_embed_to_single_channel(model)

        generate_exposure_multiscale_embeddings(
            ds, model, device="cuda", output_dir=scratch,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"].get("num_workers", 4),
            top_k=config.get("local_views", {}).get("top_k", 8),
            crop_size=config.get("local_views", {}).get("crop_size"),
            resume=args.cont,
        )
    else:
        # original CCD-only path
        from decam_qa.pipeline import ParallelEvaluator
        evaluator = ParallelEvaluator(config, args.dset, args.dr, resume=args.cont)
        evaluator.run()
    print("Embeddings generated successfully.")
```

- [ ] **Step 4: Verify CLI help includes --output-dir**

```bash
python -m decam_qa.cli embed --help
```

- [ ] **Step 5: Commit**

```bash
git add configs/exposure_multiscale.yaml src/decam_qa/config.py src/decam_qa/cli.py
git commit -m "feat: add exposure_multiscale config, --output-dir CLI arg, multi-scale dispatch"
```

---

### Task 9: Evaluation and ablation notebook/script

**Files:**
- Create: `nb/multiscale_eval.py` (jupytext notebook)
- Create: `src/decam_qa/evaluation.py`

- [ ] **Step 1: Write evaluation.py**

Implement:
- `compute_binary_metrics(y_true, y_prob)` — precision-recall, avg precision, recall @ FPR
- `compute_reason_metrics(y_true_multilabel, y_prob_multilabel)` — per-reason AP
- `exposure_grouped_cross_validate(X, y, groups, n_splits=5)` — GroupKFold + metrics

- [ ] **Step 2: Create notebook**

Jupytext notebook calling the evaluation functions on multi-scale embeddings + baseline.

- [ ] **Step 3: Commit**

```bash
git add src/decam_qa/evaluation.py nb/multiscale_eval.py
git commit -m "feat: add evaluation metrics and ablation notebook for multi-scale"
```

---

### Task 10: Integration and verification

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass (existing 80 + new ~40).

- [ ] **Step 2: Smoke test on a small dataset (GPU node)**

```bash
python -m decam_qa.cli embed \
  --config configs/exposure_multiscale.yaml \
  --dset data/samples/test_supervised_ooi_dataset.csv \
  --dr dr10 \
  --output-dir /pscratch/sd/b/brookluo/decam-exposure/multiscale
```

- [ ] **Step 3: Verify HDF5 output**

```python
from decam_qa.io import read_exposure_embeddings
# Reader accepts a directory (merges across all worker files):
result = read_exposure_embeddings("/pscratch/sd/b/brookluo/decam-exposure/multiscale/embeds_out")
print(f"Read {len(result)} exposures")
for expnum, data in result.items():
    print(f"  exp {expnum}: global {data['global'].shape}, {len(data['locals'])} locals")
```

- [ ] **Step 4: Commit**

```bash
git add src/decam_qa/ configs/ tests/ nb/ README.md && \
git diff --cached --stat && \
git commit -m "chore: integration verification — all tests pass, CLI embeds work"
```

---

## Summary: Implementation order

| Task | Module | Tests | Depends on |
|------|--------|-------|------------|
| Pre-fixes | Fix read_image + embeddings bugs | — | — |
| 1 | Single-channel DINO | 2 tests | embeddings.py |
| 2 | Focal-plane stamp | 5 tests | focalplane.py (new) |
| 3 | Anomaly scoring + top-K (no cancellation, k-bounded) | 9 tests | selection.py (new) |
| 4 | Exposure dataset (DataFrame-safe) | 8 tests | Task 2, 3 |
| 5 | Multi-scale embeddings (complete-group resume) | 3 tests | Task 1, 4 |
| 6 | Aggregation + binary + multilabel classifiers | 7 tests | — |
| 7 | HDF5 reader (directory input, multi-worker merge) | 2 tests | Task 5 |
| 8 | Config + CLI (--output-dir, representation dispatch) | — | Task 5, 6, 7 |
| 9 | Evaluation + notebook | — | Task 6, 7 |
| 10 | Integration verification | — | All above |

**Total: ~36 new tests + existing 80 = ~116 tests**
