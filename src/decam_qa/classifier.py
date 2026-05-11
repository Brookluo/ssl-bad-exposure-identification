"""kNN pipeline: build, train, and predict on DINOv2 embeddings."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV


def build_pipeline():
    """Build a scikit-learn pipeline with StandardScaler, PCA, VarianceThreshold, and kNN.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted pipeline ready for hyperparameter search and training.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("preprocessor", PCA()),
        ("selector", VarianceThreshold()),
        ("classifier", KNeighborsClassifier()),
    ])


def train(pipeline, X, y, search_params=None, test_size=0.3, random_state=42, n_jobs=1):
    """Train the pipeline on DINOv2 embeddings with optional hyperparameter search.

    Splits data into train/test sets, then fits the pipeline. When ``search_params`` is
    provided, wraps the pipeline in ``HalvingRandomSearchCV`` and returns the best estimator.

    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline (or best estimator from search).
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        if search_params is not None:
            search = HalvingRandomSearchCV(
                pipeline, search_params, cv=3,
                random_state=random_state, n_jobs=n_jobs,
                factor=2,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
        else:
            best = pipeline.fit(X_train, y_train)

        return best
    except Exception:
        raise RuntimeError("Training failed") from None


def predict(pipeline, X):
    """Predict class labels and confidence scores for DINOv2 embeddings.

    Returns: tuple[ndarray, ndarray]:
        - Predicted integer class labels, shape (n_samples,).
        - Maximum predicted probability per sample, shape (n_samples,).
    """
    y_pred_prob = pipeline.predict_proba(X)
    classes = pipeline.classes_
    labels = np.array([classes[i] for i in np.argmax(y_pred_prob, axis=1)], dtype=int)
    probs = np.max(y_pred_prob, axis=1)
    return labels, probs
