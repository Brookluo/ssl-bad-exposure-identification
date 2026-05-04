"""Tests for src.read_image — NPZ read/write round-trip."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from src.read_image import read_img_npz


class TestReadImageNpz:
    def test_read_img_npz_returns_tuple(self, sample_npz_path: str) -> None:
        half_imdata, ccdnames = read_img_npz(sample_npz_path)
        assert isinstance(half_imdata, np.ndarray)
        assert isinstance(ccdnames, np.ndarray)

    def test_read_img_npz_correct_shapes(self, sample_npz_path: str) -> None:
        half_imdata, ccdnames = read_img_npz(sample_npz_path)
        assert half_imdata.shape == (4, 2046, 2046)
        assert len(ccdnames) == 2
