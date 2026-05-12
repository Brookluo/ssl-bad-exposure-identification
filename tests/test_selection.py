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
