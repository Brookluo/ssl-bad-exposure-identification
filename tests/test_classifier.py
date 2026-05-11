"""Tests for decam_qa.classifier — kNN pipeline building, training, prediction."""
import numpy as np
import joblib
import pytest
from sklearn.pipeline import Pipeline
from decam_qa.classifier import build_pipeline, train, predict


class TestBuildPipeline:
    def test_build_pipeline_returns_sklearn_pipeline(self):
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_build_pipeline_steps_order(self):
        pipe = build_pipeline()
        step_names = [s[0] for s in pipe.steps]
        assert step_names[:4] == ["scaler", "preprocessor", "selector", "classifier"]


class TestTrain:
    def test_train_on_synthetic_data(self, small_embeddings, small_labels):
        pipe = build_pipeline()
        result = train(pipe, small_embeddings, small_labels, test_size=0.3, random_state=42, n_jobs=1)
        score = result.score(small_embeddings, small_labels)
        assert score > 0.3

    def test_train_reproducible(self, small_embeddings, small_labels):
        pipe1 = build_pipeline()
        pipe2 = build_pipeline()
        r1 = train(pipe1, small_embeddings, small_labels, test_size=0.3, random_state=42, n_jobs=1)
        r2 = train(pipe2, small_embeddings, small_labels, test_size=0.3, random_state=42, n_jobs=1)
        assert r1.named_steps["classifier"].n_neighbors == r2.named_steps["classifier"].n_neighbors

    def test_train_with_halving_random_search(self, small_embeddings, small_labels):
        pipe = build_pipeline()
        search_params = {"classifier__n_neighbors": [1, 3, 5], "classifier__p": [1, 2]}
        result = train(pipe, small_embeddings, small_labels,
                       search_params=search_params, test_size=0.3, random_state=42, n_jobs=1)
        assert isinstance(result, Pipeline)


class TestPredict:
    def test_predict_returns_labels_and_probs(self, small_embeddings, small_labels):
        pipe = build_pipeline()
        fitted = train(pipe, small_embeddings, small_labels, test_size=0.3, random_state=42, n_jobs=1)
        labels, probs = predict(fitted, small_embeddings)
        assert len(labels) == len(small_embeddings)
        assert probs.shape == (len(small_embeddings),)

    def test_predict_on_untrained_raises(self, small_embeddings):
        pipe = build_pipeline()
        with pytest.raises(Exception):
            predict(pipe, small_embeddings)

    def test_pipeline_serialization_roundtrip(self, small_embeddings, small_labels, tmp_path):
        pipe = build_pipeline()
        fitted = train(pipe, small_embeddings, small_labels, test_size=0.3, random_state=42, n_jobs=1)
        labels1, _ = predict(fitted, small_embeddings)
        pkl_path = tmp_path / "pipe.pkl"
        joblib.dump(fitted, pkl_path)
        loaded = joblib.load(pkl_path)
        labels2, _ = predict(loaded, small_embeddings)
        np.testing.assert_array_equal(labels1, labels2)
