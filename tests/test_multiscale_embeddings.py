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
        mock_item = {
            "expnum": 2,
            "global_stamp": np.zeros((1, 224, 224), dtype=np.float32),
            "selected_ccds": [],
            "selected_images": [],
            "filter": 1,
        }
        dataset.__getitem__.side_effect = [
            {"expnum": 1, "selected_ccds": [{}, {}]},
            mock_item,
        ]

        with patch("torch.utils.data.DataLoader"):
            generate_exposure_multiscale_embeddings(
                dataset, model, "cpu", str(tmp_path),
                batch_size=1, num_workers=0, top_k=8, resume=True)
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'r') as f:
            assert "exp_1" in f["exposures"]

    def test_resume_regenerates_partial_group(self, tmp_path):
        import json
        embeds_dir = tmp_path / "embeds_out"
        embeds_dir.mkdir()
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'w') as f:
            g = f.create_group("exposures")
            eg = g.create_group("exp_1")
            eg.create_dataset("global", data=np.zeros(768))
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

        with patch("torch.utils.data.DataLoader"):
            generate_exposure_multiscale_embeddings(
                dataset, model, "cpu", str(tmp_path),
                batch_size=1, num_workers=0, top_k=8, resume=True)
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'r') as f:
            eg = f["exposures"]["exp_1"]
            assert "local_001" in eg
