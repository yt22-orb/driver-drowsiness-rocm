import numpy as np

from drowsiness.metrics import classification_metrics, select_fbeta_threshold


def test_drowsy_label_zero_is_treated_as_positive() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.95, 0.8, 0.2, 0.1])
    metrics = classification_metrics(labels, scores, threshold=0.5, beta=2.0)
    assert metrics["accuracy"] == 1.0
    assert metrics["drowsy_recall"] == 1.0
    assert metrics["matthews_correlation_coefficient"] == 1.0
    assert metrics["non_drowsy_recall_specificity"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["confusion_matrix_label_order_0_1"] == [[2, 0], [0, 2]]


def test_extended_metrics_match_known_confusion_matrix() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.1, 0.05])
    metrics = classification_metrics(labels, scores, threshold=0.5, beta=2.0)
    assert metrics["confusion_matrix_label_order_0_1"] == [[3, 1], [1, 3]]
    assert metrics["accuracy"] == 0.75
    assert metrics["drowsy_precision"] == 0.75
    assert metrics["drowsy_recall"] == 0.75
    assert metrics["drowsy_f1"] == 0.75
    assert metrics["non_drowsy_precision"] == 0.75
    assert metrics["non_drowsy_recall_specificity"] == 0.75
    assert metrics["non_drowsy_f1"] == 0.75
    assert metrics["matthews_correlation_coefficient"] == 0.5


def test_threshold_selection_returns_valid_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.6, 0.4, 0.5, 0.2, 0.1])
    threshold, score = select_fbeta_threshold(labels, scores, beta=2.0)
    assert 0 <= threshold <= 1
    assert 0 <= score <= 1
