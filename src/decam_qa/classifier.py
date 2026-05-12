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


def aggregate_exposure_embeddings(global_emb, local_embs, scores, k=8):
    """Aggregate global + local embeddings into a fixed-length feature vector.

    Concatenates: global embedding, mean/max/std of local embeddings,
    and selection scores (zero-padded to length k).

    Parameters
    ----------
    global_emb : np.ndarray
        Global focal-plane stamp embedding, shape (D,).
    local_embs : list of np.ndarray
        List of top-K local CCD embeddings, each shape (D,).
    scores : list of float
        Selection scores corresponding to each local embedding.
    k : int
        Expected maximum number of local views (for zero-padding).

    Returns
    -------
    np.ndarray
        Fixed-length feature vector, shape (D*4 + k,).
    """
    D = len(global_emb)
    features = [global_emb]

    if len(local_embs) > 0:
        local_stack = np.stack(local_embs, axis=0)
        features.append(np.mean(local_stack, axis=0))
        features.append(np.max(local_stack, axis=0))
        features.append(np.std(local_stack, axis=0, ddof=1))
    else:
        features.append(np.zeros(D, dtype=np.float32))
        features.append(np.zeros(D, dtype=np.float32))
        features.append(np.zeros(D, dtype=np.float32))

    padded_scores = np.zeros(k, dtype=np.float32)
    padded_scores[:len(scores)] = scores[:k]
    features.append(padded_scores)

    return np.concatenate(features)


def train_logistic_binary(X, y, class_balanced=True, random_state=42):
    """Train a binary logistic regression classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Binary labels (0 = good, 1 = bad).
    class_balanced : bool
        If True, use class_weight='balanced'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted model.
    """
    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if class_balanced else None
    model = LogisticRegression(
        class_weight=cw, random_state=random_state,
        max_iter=2000, solver="lbfgs",
    )
    model.fit(X, y)
    return model


def predict_binary(model, X):
    """Predict probability of 'bad' class for binary classifier.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted binary classifier.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Probability of class 1 (bad), shape (n_samples,).
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return 1 - proba[:, 0] if model.classes_[0] == 0 else proba[:, 0]
    return proba[:, 1]


def train_multilabel_reason(X, y_bitmask, n_reasons=15,
                            class_balanced=True, random_state=42):
    """Train multilabel reason classifiers via OneVsRest.

    Converts integer bitmask labels into a binary matrix and fits one
    LogisticRegression per reason bit.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y_bitmask : np.ndarray
        Integer bitmask labels per sample.
    n_reasons : int
        Number of reason bits (default 15).
    class_balanced : bool
        If True, use class_weight='balanced'.
    random_state : int
        Random seed.

    Returns
    -------
    sklearn.multiclass.OneVsRestClassifier
        Fitted multilabel model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    Y = np.zeros((len(y_bitmask), n_reasons), dtype=int)
    for i, val in enumerate(y_bitmask):
        for bit in range(n_reasons):
            if val & (1 << bit):
                Y[i, bit] = 1

    cw = "balanced" if class_balanced else None
    base = LogisticRegression(
        class_weight=cw, random_state=random_state,
        max_iter=2000, solver="lbfgs",
    )
    model = OneVsRestClassifier(base)
    model.fit(X, Y)
    return model


def predict_multilabel(model, X):
    """Predict per-reason probabilities.

    Parameters
    ----------
    model : sklearn.multiclass.OneVsRestClassifier
        Fitted multilabel model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Per-reason probabilities, shape (n_samples, n_reasons).
    """
    return model.predict_proba(X)
