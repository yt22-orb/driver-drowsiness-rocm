"""Binary metrics with label zero treated as the drowsy positive class."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import DROWSY_LABEL, NON_DROWSY_LABEL


def select_fbeta_threshold(
    labels: np.ndarray, drowsy_scores: np.ndarray, beta: float = 2.0
) -> tuple[float, float]:
    positive = (np.asarray(labels) == DROWSY_LABEL).astype(np.int64)
    precision, recall, thresholds = precision_recall_curve(positive, drowsy_scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    beta_squared = beta**2
    denominator = beta_squared * precision[:-1] + recall[:-1]
    scores = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def classification_metrics(
    labels: np.ndarray,
    drowsy_scores: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    drowsy_scores = np.asarray(drowsy_scores, dtype=np.float64)
    predictions = np.where(drowsy_scores >= threshold, DROWSY_LABEL, NON_DROWSY_LABEL)
    positive_labels = (labels == DROWSY_LABEL).astype(np.int64)
    positive_predictions = (predictions == DROWSY_LABEL).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[DROWSY_LABEL, NON_DROWSY_LABEL])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "drowsy_precision": float(
            precision_score(positive_labels, positive_predictions, zero_division=0)
        ),
        "drowsy_recall": float(
            recall_score(positive_labels, positive_predictions, zero_division=0)
        ),
        "drowsy_fbeta": float(
            fbeta_score(positive_labels, positive_predictions, beta=beta, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(positive_labels, drowsy_scores)),
        "confusion_matrix_label_order_0_1": matrix.tolist(),
        "sample_count": int(labels.size),
        "drowsy_count": int((labels == DROWSY_LABEL).sum()),
        "non_drowsy_count": int((labels == NON_DROWSY_LABEL).sum()),
    }
