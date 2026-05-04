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

    def test_empty_dataset_handled(self, tmp_path: Path) -> None:
        import pandas as pd
        empty = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty, index=False)
        split_dset(str(empty), tmp_path, num_parts=1)
        result = pd.read_csv(tmp_path / "0_worker_sample.csv")
        assert len(result) == 0
