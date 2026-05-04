"""Tests for src.inference — HDF5 embedding reader."""
from __future__ import annotations

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
