"""Tests for decam_qa.pipeline — ParallelEvaluator orchestration."""
import pytest
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from unittest.mock import patch, MagicMock
from decam_qa.pipeline import _split_dset, _get_completed_indices, ParallelEvaluator


class TestSplitDataset:
    def test_split_dset_even(self, tmp_path):
        csv_path = tmp_path / "input.csv"
        df = pd.DataFrame({"a": range(8)})
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "tmp"
        _split_dset(str(csv_path), out_dir, 2, keep_index=True)
        files = sorted(out_dir.glob("*_worker_sample.csv"))
        assert len(files) == 2
        d1 = pd.read_csv(files[0])
        d2 = pd.read_csv(files[1])
        assert len(d1) == 4
        assert len(d2) == 4
        assert "original_df_idx" in d1.columns


class TestCompletedIndices:
    def test_get_completed_indices(self, tmp_path):
        embeds_dir = tmp_path / "embeds_out"
        embeds_dir.mkdir()
        with h5py.File(embeds_dir / "0_worker_embeds.h5", 'w') as f:
            grp = f.create_group("images")
            grp.create_dataset("idx_0_label_0", data=np.zeros((1, 768)))
            grp.create_dataset("idx_2_label_1", data=np.zeros((1, 768)))
        done = _get_completed_indices(str(embeds_dir))
        assert done == {0, 2}

    def test_get_completed_indices_empty_dir(self, tmp_path):
        embeds_dir = tmp_path / "embeds_out"
        embeds_dir.mkdir()
        done = _get_completed_indices(str(embeds_dir))
        assert done == set()


class TestParallelEvaluator:
    def test_creates_dirs_on_init(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2, 3, 4]}).to_csv(csv_path, index=False)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc
            with patch("torch.cuda.device_count", return_value=1):
                evaluator = ParallelEvaluator(
                    {"scratch_dir": str(tmp_path), "data": {"num_workers": 0}},
                    str(csv_path), "dr10", resume=False)
                try:
                    evaluator.run()
                except Exception:
                    pass
        assert (tmp_path / "tmp").exists()
        assert (tmp_path / "embeds_out").exists()

    def test_worker_failure_raises(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2, 3, 4]}).to_csv(csv_path, index=False)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"mock error")
            mock_popen.return_value = mock_proc
            with patch("torch.cuda.device_count", return_value=1):
                evaluator = ParallelEvaluator(
                    {"scratch_dir": str(tmp_path), "data": {"num_workers": 0}},
                    str(csv_path), "dr10", resume=False)
                with pytest.raises(RuntimeError):
                    evaluator.run()
