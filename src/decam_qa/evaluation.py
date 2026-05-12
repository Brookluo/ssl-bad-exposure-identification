"""Evaluation metrics for exposure-level classification."""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold


def compute_binary_metrics(y_true, y_prob):
    """Compute binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 = good, 1 = bad).
    y_prob : np.ndarray
        Predicted probabilities of class 1.

    Returns
    -------
    dict
        Keys: avg_precision, precision_at_recall_90, recall_at_fpr_10,
        roc_auc, thresholds.
    """
    from sklearn.metrics import roc_auc_score
    avg_prec = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    target_rec = 0.9
    idx = np.argmin(np.abs(rec - target_rec))
    prec_at_rec90 = prec[idx]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    target_fpr = 0.1
    idx_fpr = np.argmin(np.abs(fpr - target_fpr))
    recall_at_fpr10 = tpr[idx_fpr]

    return {
        "avg_precision": float(avg_prec),
        "roc_auc": float(roc_auc),
        "precision_at_recall_90": float(prec_at_rec90),
        "recall_at_fpr_10": float(recall_at_fpr10),
    }


def compute_reason_metrics(y_true_bitmask, y_prob_multilabel, n_reasons=15):
    """Compute per-reason average precision.

    Parameters
    ----------
    y_true_bitmask : np.ndarray
        Integer bitmask labels per sample.
    y_prob_multilabel : np.ndarray
        Per-reason probabilities, shape (n_samples, n_reasons).
    n_reasons : int
        Number of reason bits.

    Returns
    -------
    dict
        {reason_index: average_precision} for reason bits present in data.
    """
    from decam_qa.info import reason_li
    metrics = {}
    Y = np.zeros((len(y_true_bitmask), n_reasons), dtype=int)
    for i, val in enumerate(y_true_bitmask):
        for bit in range(n_reasons):
            if val & (1 << bit):
                Y[i, bit] = 1

    for bit in range(n_reasons):
        if Y[:, bit].sum() == 0:
            continue
        ap = average_precision_score(Y[:, bit], y_prob_multilabel[:, bit])
        reason_name = reason_li[bit] if bit < len(reason_li) else f"reason_{bit}"
        metrics[reason_name] = float(ap)

    return metrics


def exposure_grouped_cross_validate(X, y, groups, n_splits=5, random_state=42):
    """Run GroupKFold cross-validation and return per-fold metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Binary labels (0/1).
    groups : np.ndarray
        Group identifiers (e.g., expnum) so all CCDs from same exposure
        stay in same fold.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed for logistic regression.

    Returns
    -------
    list of dict
        Per-fold metrics.
    """
    from sklearn.linear_model import LogisticRegression
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_train_pos = y_train.sum()
        n_test_pos = y_test.sum()
        cw = "balanced" if n_train_pos > 0 else None
        model = LogisticRegression(
            class_weight=cw, random_state=random_state,
            max_iter=2000, solver="lbfgs",
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        if y_prob.shape[1] == 1:
            y_prob_bad = y_prob[:, 0] if model.classes_[0] == 1 else 1 - y_prob[:, 0]
        else:
            y_prob_bad = y_prob[:, 1]

        metrics = compute_binary_metrics(y_test, y_prob_bad)
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        metrics["n_train_pos"] = int(n_train_pos)
        metrics["n_test_pos"] = int(n_test_pos)
        fold_metrics.append(metrics)

    return fold_metrics
