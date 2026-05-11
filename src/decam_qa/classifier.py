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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("preprocessor", PCA()),
        ("selector", VarianceThreshold()),
        ("classifier", KNeighborsClassifier()),
    ])


def train(pipeline, X, y, search_params=None, test_size=0.3, random_state=42, n_jobs=1):
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
    y_pred_prob = pipeline.predict_proba(X)
    classes = pipeline.classes_
    labels = np.array([classes[i] for i in np.argmax(y_pred_prob, axis=1)], dtype=int)
    probs = np.max(y_pred_prob, axis=1)
    return labels, probs
