"""Tests for decam_qa.io — HDF5 embed I/O and FITS image reading."""
import numpy as np
import h5py
import pytest
from pathlib import Path
from decam_qa.io import read_embeddings, read_fits_image, write_embeddings


class TestHDF5Roundtrip:
    def test_write_read_embeddings_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        embeds = [rng.normal(0, 1, (1, 768)).astype(np.float32) for _ in range(8)]
        indices = list(range(8))
        labels = [0, 1, 2, 3, 0, 1, 2, 3]

        out_dir = tmp_path / "write_test"
        written = write_embeddings(embeds, indices, labels, out_dir, worker_id=0)

        data, idx, lab = read_embeddings(out_dir)
        assert len(data) == 8
        assert idx == [str(i) for i in indices]
        assert lab == [str(l) for l in labels]
        for d1, d2 in zip(embeds, data):
            np.testing.assert_array_equal(d1, d2)

    def test_read_embeddings_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            read_embeddings(empty_dir)

    def test_read_embeddings_multi_worker(self, temp_h5_dir):
        data, idx, label = read_embeddings(temp_h5_dir)
        assert len(data) == 20
        assert len(idx) == 20
        assert len(label) == 20

    def test_hdf5_key_format(self, temp_h5_dir):
        with h5py.File(temp_h5_dir / "0_worker_embeds.h5", 'r') as h5f:
            for key in h5f["images"].keys():
                parts = key.split("_")
                assert len(parts) == 4, f"Unexpected key format: {key}"
                assert parts[0] == "idx"
                int(parts[1])
                assert parts[2] == "label"
                int(parts[3])

    def test_write_embeddings_creates_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "new_dir"
        embeds = [np.zeros((1, 768), dtype=np.float32)]
        written = write_embeddings(embeds, [0], [0], out_dir, worker_id=0)
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.h5"))) == 1

    def test_write_embeddings_append(self, tmp_path):
        out_dir = tmp_path / "append_test"
        embeds1 = [np.ones((1, 768), dtype=np.float32)]
        embeds2 = [np.full((1, 768), 2.0, dtype=np.float32)]
        write_embeddings(embeds1, [0], [0], out_dir, worker_id=0)
        write_embeddings(embeds2, [1], [1], out_dir, worker_id=0)
        data, _, _ = read_embeddings(out_dir)
        assert len(data) == 2

    def test_write_embeddings_no_duplicate_keys(self, tmp_path):
        out_dir = tmp_path / "nodup_test"
        embeds1 = [np.ones((1, 768), dtype=np.float32)]
        embeds2 = [np.full((1, 768), 2.0, dtype=np.float32)]
        write_embeddings(embeds1, [0], [0], out_dir, worker_id=0)
        write_embeddings(embeds2, [0], [0], out_dir, worker_id=0)
        data, _, _ = read_embeddings(out_dir)
        assert len(data) == 1
        np.testing.assert_array_equal(data[0], np.ones((1, 768), dtype=np.float32))


class TestFITSReading:
    def test_read_fits_image_valid(self, tmp_fits_image):
        img = read_fits_image(str(tmp_fits_image))
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    def test_read_fits_image_with_hdu(self, tmp_fits_image):
        img = read_fits_image(str(tmp_fits_image), hdu=1)
        assert img.ndim == 2

    def test_read_fits_image_bad_hdu(self, tmp_fits_image):
        with pytest.raises(IndexError):
            read_fits_image(str(tmp_fits_image), hdu=99)

    def test_read_fits_image_missing_file(self):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            read_fits_image("/nonexistent/file.fits.fz")

    def test_read_fits_image_byte_order(self, tmp_fits_image):
        img = read_fits_image(str(tmp_fits_image))
        assert img.dtype.byteorder in ("=", "<")
