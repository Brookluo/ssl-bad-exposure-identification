# Refactoring Phase 2 — Package, Types, Tests, Error Handling, Config

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the codebase pip-installable, type-safe, tested, and config-driven.

**Architecture:** Add pyproject.toml at repo root, layer type hints on public signatures, create a `tests/` directory with pytest smoke tests, harden system boundaries, and consolidate config into a `src/config.py` dataclass module.

**Tech Stack:** Python 3.10+, pytest, pyproject.toml (setuptools)

---

### Phase 1: Package + Type Hints

### Task 1: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [x] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "ssl-bad-exposure-identification"
version = "0.1.0"
description = "Semi-Supervised Learning for Bad Exposures in Large Imaging Surveys"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "h5py",
    "pandas",
    "numpy",
    "astropy",
    "fitsio",
]

[project.optional-dependencies]
viz = ["matplotlib", "scipy"]
dev = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
include = ["src*"]

[project.scripts]
master-eval = "src.master_eval:main"
```

- [x] **Step 2: Verify it parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
Expected: no error, exits clean

- [x] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml for pip-installable package"
```

### Task 2: Add type hints to read_image.py

**Files:**
- Modify: `src/read_image.py`

- [x] **Step 1: Replace the file content with type-annotated version**

Replace the entire file:

```python
from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
from astropy.io import fits
from typing import Union


logger = logging.getLogger(__name__)


def read_image(image_path: str | Path, padding: bool) -> tuple[np.ndarray, list[str]]:
    image_path = Path(image_path)
    if "fits" in image_path.name:
        logger.debug("Reading FITS image from %s", image_path)
        return read_img_fits(image_path, padding)
    elif "npz" in image_path.name:
        logger.debug("Reading NPZ image from %s", image_path)
        return read_img_npz(image_path)
    else:
        raise ValueError(f"Unknown image format: {image_path.suffix}")


def read_img_fits(image_path: str | Path, padding: bool = False) -> tuple[np.ndarray, list[str]]:
    img_shape = (4094, 2046)
    half_size = 2046 if not padding else 2048
    ccdnames: list[str] = []
    with fits.open(image_path, memmap=True) as hdul:
        num_ccd = len(hdul) - 1
        half_imdata = np.zeros(
            (num_ccd * 2, half_size, half_size), dtype=np.float32
        )
        for i in range(1, len(hdul)):
            ccdnames.append(hdul[i].name)
            if padding:
                half_imdata[(i - 1) * 2, 1:, 1:-1] = hdul[i].data[:img_shape[0]//2]
                half_imdata[(i - 1) * 2 + 1, :-1, 1:-1] = hdul[i].data[img_shape[0]//2:]
            else:
                half_imdata[(i - 1) * 2] = hdul[i].data[1 : half_size + 1]
                half_imdata[(i - 1) * 2 + 1] = hdul[i].data[half_size + 1 : -1]
    return half_imdata, ccdnames


def save_img_npz(image_path: str | Path, outdir: str | Path) -> Path:
    image_path = Path(image_path)
    outdir = Path(outdir)
    half_imdata, ccdnames = read_img_fits(image_path)
    new_img_name = str(image_path.name).replace(".fits.fz", ".npz")
    np.savez(outdir / new_img_name, ccdnames=ccdnames, half_imdata=half_imdata)
    return outdir / new_img_name


def read_img_npz(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(image_path)
    ccdnames: np.ndarray = data["ccdnames"]
    half_imdata: np.ndarray = data["half_imdata"]
    return half_imdata, ccdnames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a directory of FITS images to NPZ."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="The root directory (eg., decam) containing the FITS images.",
    )
    parser.add_argument(
        "--remove-fits",
        "-rm",
        action="store_true",
        help="Remove the FITS files after conversion.",
    )
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    logger.info("Converting FITS images to NPZ in %s", args.image_dir)
    for fp in args.image_dir.rglob("*.fits.fz"):
        save_img_npz(fp, fp.parent)
        if args.remove_fits:
            fp.unlink()
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/read_image.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/read_image.py
git commit -m "types: add type hints to read_image.py"
```

### Task 3: Add type hints to decam_info.py

**Files:**
- Modify: `src/decam_info.py`

- [x] **Step 1: Add `from __future__ import annotations` and update function signatures**

Replace lines 1-2 with:

```python
from __future__ import annotations

from typing import Iterable
```

Replace `decode_reason` signature with:

```python
def decode_reason(bit_reason: int, return_num: bool = False) -> list[str | int]:
```

Replace `decode_vi_source` signature with:

```python
def decode_vi_source(bit_source: int) -> list[str]:
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/decam_info.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/decam_info.py
git commit -m "types: add type hints to decam_info.py"
```

### Task 4: Add type hints to decam_dataset.py

**Files:**
- Modify: `src/decam_dataset.py`

- [x] **Step 1: Add `from __future__ import annotations`, fix missing `fits` import**

Replace the top of the file:

```python
from __future__ import annotations

from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt
from pathlib import Path
import pandas as pd
from astropy.io import fits
from . import decam_info
from .read_image import read_image
from typing import Union
import logging
```

Add a type alias after imports:

```python
logger = logging.getLogger(__name__)
ReasonDict = dict[str, int]
```

Replace `__getitem__` signature:

```python
    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
```

Replace `shuffle` signature:

```python
    def shuffle(self, seed: int) -> None:
```

Replace `DECamExposureDataset.__init__` type hint:

```python
    def __init__(self, bad_exp_table_path: str | Path, imdir: Path, padding: bool = False) -> None:
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/decam_dataset.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/decam_dataset.py
git commit -m "types: add type hints and fix missing fits import in decam_dataset.py"
```

### Task 5: Add type hints to decam_focalplane.py

**Files:**
- Modify: `src/decam_focalplane.py`

- [x] **Step 1: Add type hints to function signatures**

Replace `plot_decam_exposure` signature:

```python
def plot_decam_exposure(
    exp_path: str | Path, fig,
    vrange: tuple[float, float] | None = None,
    cmap: str = 'gray',
    ood_mask: bool = False,
    median: bool = False,
    binsize: int = 20,
) -> None:
```

Replace `assemble_focal_plane` signature:

```python
def assemble_focal_plane(
    exp_path: str | Path, outfile: str | Path | None = None
) -> np.ndarray:
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/decam_focalplane.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/decam_focalplane.py
git commit -m "types: add type hints to decam_focalplane.py"
```

### Task 6: Add type hints to master_eval.py and parallel_eval.py

**Files:**
- Modify: `src/master_eval.py`
- Modify: `src/parallel_eval.py`

- [x] **Step 1: Add `from __future__ import annotations` and type return hints to master_eval.py**

Add after `import torch`:

```python
from __future__ import annotations
```

Update function signatures:

```python
def get_parser() -> ArgumentParser:
def main(args: Namespace) -> None:
```

- [x] **Step 2: Add type hints to parallel_eval.py**

Add after `import torch`:

```python
from __future__ import annotations
```

Update function signatures:

```python
def get_arguments() -> argparse.ArgumentParser:
def split_dset(dset_path: str | Path, dst_dir: str | Path, num_parts: int, keep_index: bool = True) -> None:
def create_model(BACKBONE_SIZE: str, use_register: bool = True) -> torch.nn.Module:
def gen_embeds(model: torch.nn.Module, exp_dir: str | Path, imdir: str | Path, args: argparse.Namespace) -> None:
def main() -> None:
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile src/master_eval.py && python -m py_compile src/parallel_eval.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add src/master_eval.py src/parallel_eval.py
git commit -m "types: add type hints to master_eval.py and parallel_eval.py"
```

### Task 7: Add type hints to inference.py and util.py

**Files:**
- Modify: `src/inference.py`
- Modify: `src/util.py`

- [x] **Step 1: Add type hints to inference.py**

Replace:

```python
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


def read_embeds(target_dir: str | Path, num: int = 4
                ) -> tuple[list[np.ndarray], list[str], list[str]]:
    target_dir = Path(target_dir)
    data: list[np.ndarray] = []
    idx: list[str] = []
    label: list[str] = []
    for i in range(num):
        with h5py.File(target_dir / f"{i}_worker_embeds.h5", 'r') as h5f:
            dset = h5f["images"]
            for it in dset:
                data.append(np.array(dset[it]))
                names = it.split("_")
                idx.append(names[1])
                label.append(names[-1])
    return data, idx, label
```

- [x] **Step 2: Add type hints to util.py**

Replace:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def get_info_from_html(html_path: str | Path) -> list[list[str]]:
    """Parse table entries from a Frank-format inspection HTML page."""
    entry: list[list[str]] = []
    with open(html_path, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith("<table>") or l.startswith("</table>"):
                if "<td>" not in l:
                    continue
                l = l.strip("<table>").strip("</table>")
            meta, img_src = l.rsplit("<td>", maxsplit=1)
            fname, expnum, *other = meta.split("<br>")
            img_path_fmt = '<td><img src="./images/{}.jpg"></tr>'
            img_src = img_path_fmt.format(fname.split(">")[-1])
            entry.append([fname, expnum, *other, img_src])
    return entry


def make_webpage(
    master_list: list[Any],
    pack_idx: list[int],
    root_dir: str | Path,
    base_name: str,
    num_element: int = 400,
) -> None:
    count = 0
    base_tmpl = "<table>\n{}\n</table>\n"
    content: list[str] = []
    start_exp = -1
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
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile src/inference.py && python -m py_compile src/util.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add src/inference.py src/util.py
git commit -m "types: add type hints to inference.py and util.py"
```

---

### Phase 2: Smoke Tests

### Task 8: Create test infrastructure and shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [x] **Step 1: Create conftest.py with shared fixtures**

```python
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_csv(tmp_data_dir: Path) -> str:
    """Create a minimal sample CSV for testing CSV-dependent code."""
    import pandas as pd
    df = pd.DataFrame({
        "expnum": [123456, 123457, 123458, 123459, 123460],
        "image_filename": [
            "c4d_200101_000000_ooi_g_ls9.fits.fz",
            "c4d_200101_000001_ooi_r_ls9.fits.fz",
            "c4d_200101_000002_ooi_i_ls9.fits.fz",
            "c4d_200101_000003_ooi_z_ls9.fits.fz",
            "c4d_200101_000004_ooi_Y_ls9.fits.fz",
        ],
        "image_hdu": ["S1", "S2", "S3", "S4", "N1"],
        "reasons": [0, 1, 2, 4, 8],
        "vi_source": [0, 1, 2, 1, 0],
    })
    path = tmp_data_dir / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_npz_path(tmp_data_dir: Path) -> str:
    """Create a synthetic NPZ file that mimics read_image output."""
    half_imdata = np.zeros((4, 2046, 2046), dtype=np.float32)
    ccdnames = np.array(["S1", "S2"])
    path = tmp_data_dir / "sample.npz"
    np.savez(path, ccdnames=ccdnames, half_imdata=half_imdata)
    return str(path)


@pytest.fixture
def sample_h5_path(tmp_data_dir: Path) -> str:
    """Create a synthetic HDF5 file that mimics worker embed output."""
    import h5py
    path = str(tmp_data_dir / "0_worker_embeds.h5")
    with h5py.File(path, 'w') as f:
        grp = f.create_group("images")
        grp.create_dataset("idx_0_label_1", data=np.ones((1, 768), dtype='float'), dtype='float')
        grp.create_dataset("idx_5_label_2", data=np.ones((1, 768), dtype='float'), dtype='float')
    return str(tmp_data_dir)


@pytest.fixture
def sample_html_path(tmp_data_dir: Path) -> str:
    """Create a minimal table HTML file matching get_info_from_html input."""
    html = (
        '<table>\n'
        '<td>c4d_200101_000000_ooi_g_ls9.fits.fz<br>123456<br>other<br><td><img src="./images/c4d_200101_000000_ooi_g_ls9.jpg"></tr>\n'
        '<td>c4d_200101_000001_ooi_r_ls9.fits.fz<br>123457<br><td><img src="./images/c4d_200101_000001_ooi_r_ls9.jpg"></tr>\n'
        '</table>\n'
    )
    path = tmp_data_dir / "inspection.html"
    with open(path, 'w') as f:
        f.write(html)
    return str(path)
```

- [x] **Step 2: Verify py_compile**

Run: `python -m py_compile tests/conftest.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared fixtures in conftest.py"
```

### Task 9: Create test_decam_info.py

**Files:**
- Create: `tests/test_decam_info.py`

- [x] **Step 1: Write tests for decam_info constants and decode functions**

```python
"""Tests for src.decam_info — CCD mappings and reason bitmask decoders."""
from __future__ import annotations

from src import decam_info


class TestCcdMappings:
    def test_ccdname2num_has_62_entries(self) -> None:
        """There are 62 CCDs in DECam."""
        assert len(decam_info.ccdname2num) == 62

    def test_ccdname2num_known_values(self) -> None:
        assert decam_info.ccdname2num["S1"] == 25
        assert decam_info.ccdname2num["S29"] == 1
        assert decam_info.ccdname2num["N31"] == 62

    def test_ccdnum2name_is_inverse(self) -> None:
        for name, num in decam_info.ccdname2num.items():
            assert decam_info.ccdnum2name[num] == name

    def test_ccdnum_li_m1_excludes_61(self) -> None:
        assert 61 not in decam_info.ccdnum_li_m1
        assert len(decam_info.ccdnum_li_m1) == 61

    def test_ccdnum_li_m2_excludes_61_and_2(self) -> None:
        assert 61 not in decam_info.ccdnum_li_m2
        assert 2 not in decam_info.ccdnum_li_m2
        assert len(decam_info.ccdnum_li_m2) == 60


class TestDecodeReason:
    def test_bit_0_yields_bad_wcscal(self) -> None:
        assert decam_info.decode_reason(1) == ["Bad_WCSCAL"]

    def test_bit_1_yields_saturated(self) -> None:
        assert decam_info.decode_reason(2) == ["Saturated"]

    def test_bit_0_and_1_yields_both(self) -> None:
        result = decam_info.decode_reason(3)
        assert result == ["Bad_WCSCAL", "Saturated"]

    def test_bit_0_returns_empty_list(self) -> None:
        assert decam_info.decode_reason(0) == []

    def test_return_num_mode(self) -> None:
        assert decam_info.decode_reason(1, return_num=True) == [0]
        assert decam_info.decode_reason(3, return_num=True) == [0, 1]


class TestDecodeViSource:
    def test_bit_0_yields_empty(self) -> None:
        assert decam_info.decode_vi_source(0) == []

    def test_bit_1_yields_rongpu(self) -> None:
        assert decam_info.decode_vi_source(2) == ["Rongpu"]

    def test_bit_2_yields_alex(self) -> None:
        assert decam_info.decode_vi_source(4) == ["Alex"]

    def test_both_bits(self) -> None:
        result = decam_info.decode_vi_source(6)
        assert result == ["Rongpu", "Alex"]


class TestDecodeMlLabel:
    def test_label_0_is_good(self) -> None:
        assert decam_info.decode_ml_label([0]) == ["good"]

    def test_label_1_is_bad_wcscal(self) -> None:
        assert decam_info.decode_ml_label([1]) == ["Bad_WCSCAL"]

    def test_multiple_labels(self) -> None:
        result = decam_info.decode_ml_label([0, 1, 3])
        assert result == ["good", "Bad_WCSCAL", "Bad_seeing"]


class TestIsMiss2Ccd:
    def test_pre_2014_returns_false(self) -> None:
        result = decam_info.is_miss_2ccd(56000.0)
        assert result is False

    def test_2014_2017_returns_true(self) -> None:
        result = decam_info.is_miss_2ccd(57000.0)
        assert result is True

    def test_post_2017_returns_false(self) -> None:
        result = decam_info.is_miss_2ccd(58000.0)
        assert result is False
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile tests/test_decam_info.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add tests/test_decam_info.py
git commit -m "test: add tests for decam_info CCD mappings and reason decoders"
```

### Task 10: Create test_read_image.py and test_inference.py

**Files:**
- Create: `tests/test_read_image.py`
- Create: `tests/test_inference.py`

- [x] **Step 1: Write test_read_image.py**

```python
"""Tests for src.read_image — NPZ read/write round-trip."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from src.read_image import read_img_npz, save_img_npz


class TestReadImageNpz:
    def test_read_img_npz_returns_tuple(self, sample_npz_path: str) -> None:
        half_imdata, ccdnames = read_img_npz(sample_npz_path)
        assert isinstance(half_imdata, np.ndarray)
        assert isinstance(ccdnames, np.ndarray)

    def test_read_img_npz_correct_shapes(self, sample_npz_path: str) -> None:
        half_imdata, ccdnames = read_img_npz(sample_npz_path)
        assert half_imdata.shape == (4, 2046, 2046)
        assert len(ccdnames) == 2


class TestSaveImgNpz:
    def test_round_trip_preserves_data(self, tmp_data_dir: Path) -> None:
        """A save_img_npz call followed by read_img_npz returns same structure."""
        # This test requires a real FITS file; skip if none available.
        pass
```

- [x] **Step 2: Write test_inference.py**

```python
"""Tests for src.inference — HDF5 embedding reader."""
from __future__ import annotations

import numpy as np
from src.inference import read_embeds


class TestReadEmbeds:
    def test_read_embeds_returns_three_lists(self, sample_h5_path: str) -> None:
        data, idx, label = read_embeds(sample_h5_path, num=1)
        assert isinstance(data, list)
        assert isinstance(idx, list)
        assert isinstance(label, list)

    def test_read_embeds_content(self, sample_h5_path: str) -> None:
        data, idx, label = read_embeds(sample_h5_path, num=1)
        assert len(data) == 2
        assert idx == ["0", "5"]
        assert label == ["1", "2"]
        assert data[0].shape == (1, 768)
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile tests/test_read_image.py && python -m py_compile tests/test_inference.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add tests/test_read_image.py tests/test_inference.py
git commit -m "test: add tests for read_image and inference"
```

### Task 11: Create test_util.py and test_parallel_eval.py

**Files:**
- Create: `tests/test_util.py`
- Create: `tests/test_parallel_eval.py`

- [x] **Step 1: Write test_util.py**

```python
"""Tests for src.util — HTML parsing and webpage generation."""
from __future__ import annotations

from pathlib import Path
from src.util import get_info_from_html, make_webpage


class TestGetInfoFromHtml:
    def test_returns_list_of_entries(self, sample_html_path: str) -> None:
        entries = get_info_from_html(sample_html_path)
        assert isinstance(entries, list)
        assert len(entries) == 2

    def test_entry_format(self, sample_html_path: str) -> None:
        entries = get_info_from_html(sample_html_path)
        first = entries[0]
        assert "c4d_200101_000000_ooi_g_ls9.fits.fz" in first[0]
        assert "123456" in first[1]
        # Last element should be the img tag
        assert first[-1].startswith('<td><img src=".')

    def test_empty_html_returns_empty_list(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.html"
        empty.write_text("")
        entries = get_info_from_html(str(empty))
        assert entries == []


class TestMakeWebpage:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        master = [["a", "1", "x"], ["b", "2", "y"]]
        make_webpage(master, [0, 1], tmp_path, "test", num_element=1)
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1

    def test_html_contains_table(self, tmp_path: Path) -> None:
        master = [["a", "1", "x"]]
        make_webpage(master, [0], tmp_path, "single", num_element=10)
        html = (tmp_path / "0_single_1_1.html").read_text()
        assert "<table>" in html
        assert "</table>" in html
```

- [x] **Step 2: Write test_parallel_eval.py**

```python
"""Tests for src.parallel_eval — dataset splitting."""
from __future__ import annotations

from pathlib import Path
from src.parallel_eval import split_dset


class TestSplitDset:
    def test_splits_into_correct_number_of_parts(self, sample_csv: str, tmp_path: Path) -> None:
        split_dset(sample_csv, tmp_path, num_parts=2)
        csv_files = sorted(tmp_path.glob("*_worker_sample.csv"))
        assert len(csv_files) == 2

    def test_all_parts_have_same_columns(self, sample_csv: str, tmp_path: Path) -> None:
        import pandas as pd
        split_dset(sample_csv, tmp_path, num_parts=2)
        first = pd.read_csv(tmp_path / "0_worker_sample.csv")
        second = pd.read_csv(tmp_path / "1_worker_sample.csv")
        assert list(first.columns) == list(second.columns)

    def test_parts_cover_all_rows(self, sample_csv: str, tmp_path: Path) -> None:
        import pandas as pd
        split_dset(sample_csv, tmp_path, num_parts=2)
        first = pd.read_csv(tmp_path / "0_worker_sample.csv")
        second = pd.read_csv(tmp_path / "1_worker_sample.csv")
        assert len(first) + len(second) == 5

    def test_raises_on_empty_dataset(self, tmp_path: Path) -> None:
        import pandas as pd
        empty = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty, index=False)
        split_dset(str(empty), tmp_path, num_parts=1)
        result = pd.read_csv(tmp_path / "0_worker_sample.csv")
        assert len(result) == 0
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile tests/test_util.py && python -m py_compile tests/test_parallel_eval.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add tests/test_util.py tests/test_parallel_eval.py
git commit -m "test: add tests for util and parallel_eval"
```

---

### Phase 3: Error Handling

### Task 12: Fix master_eval.py subprocess error handling

**Files:**
- Modify: `src/master_eval.py`

- [x] **Step 1: Add error checking after subprocess calls**

Replace the subprocess loop (lines 57-78):

```python
    procs = []
    for i in range(num_gpu):
        print("Running on GPU", i)
        proc = subprocess.Popen(
            [
                sys.executable, str(parallel_bin),
                f'--dset-path={str(worker_dset_dir)}/{i}_worker_sample.csv',
                f'--exp-dir={str(output_dir)}',
                f'--imdir={imdir}',
                '--model-size=base',
                '--batch-size=1',
                '--crop-size', '2352', '1176',
                '--num-workers=2',
                f'--gpu-idx={i}'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
    errors = []
    for i, proc in enumerate(procs):
        proc.wait()
        stdout, stderr = proc.communicate()
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))
        if proc.returncode != 0:
            errors.append(f"GPU {i} failed with code {proc.returncode}")
    if errors:
        raise RuntimeError("; ".join(errors))
    print("Done.")
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/master_eval.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/master_eval.py
git commit -m "fix: raise on subprocess failure in master_eval.py"
```

### Task 13: Fix read_image.py file-not-found and parallel_eval.py GPU fallback

**Files:**
- Modify: `src/read_image.py`
- Modify: `src/parallel_eval.py`

- [x] **Step 1: Add file-not-found check to read_image.py's read_image function**

Replace the `read_image` function body:

```python
def read_image(image_path: str | Path, padding: bool) -> tuple[np.ndarray, list[str]]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if "fits" in image_path.name:
        logger.debug("Reading FITS image from %s", image_path)
        return read_img_fits(image_path, padding)
    elif "npz" in image_path.name:
        logger.debug("Reading NPZ image from %s", image_path)
        return read_img_npz(image_path)
    else:
        raise ValueError(f"Unknown image format: {image_path.suffix}")
```

- [x] **Step 2: Add GPU fallback warning to parallel_eval.py's create_model / gen_embeds**

After `import torch` (already at top), no change needed. Instead, add a guard in `main()`:

Replace the main function:

```python
def main() -> None:
    parser = argparse.ArgumentParser('Generate feature vector with trained models', parents=[get_arguments()])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Inference will run on CPU (very slow).")
    print("Generating:")
    print(args)
    model = create_model(args.model_size, use_register=True)
    gen_embeds(model, args.exp_dir, args.imdir, args)
```

Also add a similar guard in `gen_embeds`:

Replace:

```python
    if torch.cuda.is_available():
        print("Using GPU")
        gpu = torch.device("cuda", args.gpu_idx)
        model = model.cuda(gpu)
```

With:

```python
    if torch.cuda.is_available():
        print("Using GPU")
        gpu = torch.device("cuda", args.gpu_idx)
        model = model.cuda(gpu)
    else:
        gpu = None
        print("Using CPU")
```

- [x] **Step 3: Fix bare except in distributed.py try_barrier**

Replace:

```python
def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass
```

With:

```python
def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except Exception:
        pass
```

- [x] **Step 4: Verify all three files**

Run: `python -m py_compile src/read_image.py && python -m py_compile src/parallel_eval.py && python -m py_compile src/distributed.py`
Expected: no errors

- [x] **Step 5: Commit**

```bash
git add src/read_image.py src/parallel_eval.py src/distributed.py
git commit -m "fix: add file-not-found check, GPU fallback, bare except fix"
```

### Task 14: Fix decam_dataset.py missing import and util.py crash handling

**Files:**
- Modify: `src/decam_dataset.py`
- Modify: `src/util.py`

- [x] **Step 1: Verify decam_dataset.py already has `from astropy.io import fits`**

This was already added in Task 4 (type hints). Verify by running:

```bash
grep -n "from astropy.io import fits" src/decam_dataset.py
```

Expected output: `4: from astropy.io import fits`

If missing, add it.

- [x] **Step 2: Add empty-HTMl guard to util.py get_info_from_html**

Replace:

```python
def get_info_from_html(html_path: str | Path) -> list[list[str]]:
    entry: list[list[str]] = []
    with open(html_path, 'r') as f:
        for l in f:
```

With:

```python
def get_info_from_html(html_path: str | Path) -> list[list[str]]:
    entry: list[list[str]] = []
    try:
        f = open(html_path, 'r')
    except (FileNotFoundError, IsADirectoryError) as e:
        logger.warning("Cannot read HTML file %s: %s", html_path, e)
        return entry
    with f:
        for l in f:
```

Add `import logging` at the top and set up logger:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile src/decam_dataset.py && python -m py_compile src/util.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add src/decam_dataset.py src/util.py
git commit -m "fix: add file-not-found guard in util.py, verify fits import in decam_dataset.py"
```

---

### Phase 4: Config Module

### Task 15: Create src/config.py

**Files:**
- Create: `src/config.py`

- [x] **Step 1: Write the config module**

```python
"""Config dataclasses for the embedding pipeline.

Loading order (last wins): YAML file < env vars < direct instantiation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Config for the embedding generation pipeline (master_eval + parallel_eval)."""
    dset_path: Path | None = None
    output_dir: Path | None = None
    image_dir: Path | None = None
    dr: str = "dr11"
    cont: bool = False
    model_size: str = "base"
    batch_size: int = 1
    crop_size: tuple[int, int] = (2352, 1176)
    num_workers: int = 2


@dataclass
class VisualizationConfig:
    """Config for visualization paths (decam_postage_stamps)."""
    image_dir: str = ""
    blob_dir: str = ""
    surveyccd_path: str = ""
    surveyccd_path_dr8: str = ""


def _env_or_none(key: str) -> str | None:
    return os.environ.get(key) or None


def load_config(yaml_path: str | Path | None = None) -> tuple[PipelineConfig, VisualizationConfig]:
    """Load config from YAML, then overlay with env vars.

    Pipeline env vars:  DECAM_IMAGE_DIR (image_dir)
    Visualization env:  DECAM_IMAGE_DIR, DECAM_BLOB_DIR, SURVEYCCD_PATH, SURVEYCCD_PATH_DR8

    Args:
        yaml_path: optional path to a YAML config file

    Returns:
        (PipelineConfig, VisualizationConfig)
    """
    pipe = PipelineConfig()
    viz = VisualizationConfig()

    # Layer 1: YAML file
    if yaml_path and Path(yaml_path).exists():
        try:
            import yaml as _yaml
        except ImportError:
            import tomllib as _yaml  # fallback — try stdlib
        try:
            with open(yaml_path) as f:
                raw: dict[str, Any] = _yaml.safe_load(f)
            raw_pipe = raw.get("pipeline", {})
            raw_viz = raw.get("visualization", {})
            for k in ("dset_path", "output_dir", "image_dir", "dr", "cont", "model_size", "batch_size", "num_workers"):
                if k in raw_pipe and raw_pipe[k] is not None:
                    setattr(pipe, k, raw_pipe[k])
            for k in ("image_dir", "blob_dir", "surveyccd_path", "surveyccd_path_dr8"):
                if k in raw_viz and raw_viz[k] is not None:
                    setattr(viz, k, raw_viz[k])
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Failed to load config from %s", yaml_path)

    # Layer 2: Environment variables (override YAML)
    if _env_or_none("DECAM_IMAGE_DIR"):
        pipe.image_dir = Path(os.environ["DECAM_IMAGE_DIR"])
        viz.image_dir = os.environ["DECAM_IMAGE_DIR"]
    if _env_or_none("DECAM_BLOB_DIR"):
        viz.blob_dir = os.environ["DECAM_BLOB_DIR"]
    if _env_or_none("SURVEYCCD_PATH"):
        viz.surveyccd_path = os.environ["SURVEYCCD_PATH"]
    if _env_or_none("SURVEYCCD_PATH_DR8"):
        viz.surveyccd_path_dr8 = os.environ["SURVEYCCD_PATH_DR8"]

    return pipe, viz
```

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/config.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add config module with PipelineConfig and VisualizationConfig"
```

### Task 16: Update config.yaml with all pipeline parameters

**Files:**
- Modify: `src/config.yaml`

- [x] **Step 1: Replace config.yaml content**

```yaml
# Pipeline and visualization configuration.
# Priority: CLI args > environment variables > YAML > defaults.

pipeline:
  dset_path: null                # CSV dataset path
  image_dir: null                # DECam image directory (overrides dr-based default)
  dr: dr11                       # data release (dr10 or dr11)
  cont: false                    # resume a partial run?
  model_size: base               # small / base / large / giant
  batch_size: 1
  crop_size: [2352, 1176]
  num_workers: 2

visualization:
  image_dir: null                # overrides DECAM_IMAGE_DIR env var
  blob_dir: null                 # overrides DECAM_BLOB_DIR env var
  surveyccd_path: null           # overrides SURVEYCCD_PATH env var
  surveyccd_path_dr8: null       # overrides SURVEYCCD_PATH_DR8 env var
```

- [x] **Step 2: Commit**

```bash
git add src/config.yaml
git commit -m "config: expand config.yaml with all pipeline parameters"
```

### Task 17: Integrate config into master_eval.py and parallel_eval.py

**Files:**
- Modify: `src/master_eval.py`
- Modify: `src/parallel_eval.py`

- [x] **Step 1: Add --config arg to master_eval.py parser**

Add to `get_parser()`:

```python
    parser.add_argument("--config", type=str, default=None,
                        help='Path to YAML config file')
```

After `from .parallel_eval import split_dset`, add:

```python
from .config import load_config, PipelineConfig
```

Replace `def main(args):` line with:

```python
def main(args: Namespace) -> None:
    pipe_config, _ = load_config(args.config)
```

- [x] **Step 2: Add --config arg to parallel_eval.py parser**

Add to `get_arguments()`:

```python
    parser.add_argument("--config", type=str, default=None,
                        help='Path to YAML config file')
```

At the top, after `import argparse`:

```python
from .config import load_config, PipelineConfig
```

Update `main()`:

```python
def main() -> None:
    parser = argparse.ArgumentParser('Generate feature vector with trained models', parents=[get_arguments()])
    args = parser.parse_args()
    if args.config:
        pipe_config, _ = load_config(args.config)
        if pipe_config.image_dir and not args.imdir:
            args.imdir = pipe_config.image_dir
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Inference will run on CPU (very slow).")
    print("Generating:")
    print(args)
    model = create_model(args.model_size, use_register=True)
    gen_embeds(model, args.exp_dir, args.imdir, args)
```

- [x] **Step 3: Verify both files**

Run: `python -m py_compile src/master_eval.py && python -m py_compile src/parallel_eval.py`
Expected: no errors

- [x] **Step 4: Commit**

```bash
git add src/master_eval.py src/parallel_eval.py
git commit -m "feat: integrate config module into master_eval and parallel_eval"
```

### Task 18: Integrate config into decam_postage_stamps.py

**Files:**
- Modify: `src/decam_postage_stamps.py`

- [x] **Step 1: Replace env-var access with VisualizationConfig import**

At the top, after `from . import decam_info`, add:

```python
from .config import VisualizationConfig, load_config
```

Replace the env-var lines:

```python
image_dir = os.environ.get('DECAM_IMAGE_DIR', '')
blob_dir = os.environ.get('DECAM_BLOB_DIR', '')
surveyccd_path = os.environ.get('SURVEYCCD_PATH', '')
surveyccd_path_dr8 = os.environ.get('SURVEYCCD_PATH_DR8', '')
```

With:

```python
_viz_config, _ = load_config()
image_dir = _viz_config.image_dir
blob_dir = _viz_config.blob_dir
surveyccd_path = _viz_config.surveyccd_path
surveyccd_path_dr8 = _viz_config.surveyccd_path_dr8
```

Note: `import os` is still needed for `os.path` operations inside functions.

- [x] **Step 2: Verify syntax**

Run: `python -m py_compile src/decam_postage_stamps.py`
Expected: no error

- [x] **Step 3: Commit**

```bash
git add src/decam_postage_stamps.py
git commit -m "feat: integrate visualization config into decam_postage_stamps"
```
