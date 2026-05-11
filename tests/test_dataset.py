"""Tests for decam_qa.dataset — DECamImageDataset indexing, label encoding, shuffle."""
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from decam_qa.dataset import DECamImageDataset


@pytest.fixture
def sample_dataset(sample_csv, tmp_fits_image):
    return DECamImageDataset(
        dataset_path=str(sample_csv),
        image_dir=str(tmp_fits_image.parent),
        reason_dict=None,
    )


class TestDatasetBasics:
    def test_dataset_len(self, sample_dataset):
        assert len(sample_dataset) == 4

    def test_dataset_getitem_returns_tuple(self, sample_dataset):
        item = sample_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_dataset_image_shape(self, sample_dataset):
        img, _ = sample_dataset[0]
        assert img.ndim == 3
        assert img.shape[0] == 1

    def test_dataset_image_dtype(self, sample_dataset):
        img, _ = sample_dataset[0]
        assert img.dtype.byteorder in ("=", "<", ">")


class TestLabels:
    def test_dataset_label_good(self, sample_dataset):
        _, label = sample_dataset[0]
        assert label == 0

    def test_dataset_label_single_bad(self, sample_dataset):
        _, label = sample_dataset[1]
        assert label == 1

    def test_dataset_label_deterministic(self, sample_dataset):
        labels = [sample_dataset[i][1] for i in range(4)]
        labels2 = [sample_dataset[i][1] for i in range(4)]
        assert labels == labels2


class TestShuffle:
    def test_dataset_shuffle_changes_order(self, sample_dataset):
        before = [sample_dataset[i][1] for i in range(len(sample_dataset))]
        sample_dataset.shuffle(seed=1)
        after = [sample_dataset[i][1] for i in range(len(sample_dataset))]
        assert before != after

    def test_dataset_shuffle_seed_reproducible(self, sample_csv, tmp_fits_image):
        ds1 = DECamImageDataset(str(sample_csv), str(tmp_fits_image.parent))
        ds2 = DECamImageDataset(str(sample_csv), str(tmp_fits_image.parent))
        ds1.shuffle(seed=42)
        ds2.shuffle(seed=42)
        assert [ds1[i][1] for i in range(len(ds1))] == [ds2[i][1] for i in range(len(ds2))]


class TestErrors:
    def test_dataset_missing_image_file(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame({
            "image_filename": ["nonexistent.fits.fz"],
            "expnum": [1], "ccdnum": [25], "image_hdu": [0],
            "filter": [1], "reasons": [0], "vi_source": [0],
        })
        df.to_csv(csv_path, index=False)
        ds = DECamImageDataset(str(csv_path), "/tmp")
        with pytest.raises(FileNotFoundError):
            ds[0]
