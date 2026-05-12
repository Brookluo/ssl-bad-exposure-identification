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
        expected_dim = 768 * 4 + 8
        assert len(features) == expected_dim

    def test_zero_locals_fills_zeros(self):
        rng = np.random.default_rng(42)
        global_emb = rng.normal(0, 1, 768).astype(np.float32)

        features = aggregate_exposure_embeddings(global_emb, [], [], k=8)
        local_start = 768
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
        y[n//2:] = 1

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

    def test_predict_binary_all_good_single_class(self):
        """When model.classes_ == [0] (all good), predict_binary returns P(bad)."""
        from unittest.mock import Mock

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 200))

        # Simulate a model that only saw class 0: predict_proba returns P(good)
        p_good = rng.uniform(0.8, 1.0, size=20)
        model = Mock()
        model.classes_ = np.array([0])
        model.predict_proba.return_value = p_good.reshape(-1, 1)

        probs = predict_binary(model, X)
        assert probs.shape == (20,)
        assert np.all((probs >= 0) & (probs <= 1))
        # P(bad) = 1 - P(good), should be small since p_good was 0.8–1.0
        np.testing.assert_allclose(probs, 1 - p_good)


class TestMultiReasonClassifier:
    def test_train_on_synthetic_multilabel(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 200))
        pattern = [0, 1, 2, 4, 0, 1, 1|2, 2|4]
        y_multilabel = np.resize(pattern, n)

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
