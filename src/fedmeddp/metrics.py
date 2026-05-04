from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        metrics.update(
            {
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "positive_recall": float(sensitivity),
                "negative_recall": float(specificity),
            }
        )
    else:
        metrics.update(
            {
                "sensitivity": metrics["recall"],
                "specificity": 0.0,
                "positive_recall": metrics["recall"],
                "negative_recall": 0.0,
            }
        )

    try:
        if num_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except ValueError:
        metrics["auc"] = 0.0

    metrics.update(compute_confidence_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score))
    return metrics


def compute_confidence_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: np.ndarray,
) -> dict[str, float]:
    if len(y_true) == 0 or y_score.size == 0:
        return {
            "confidence_accuracy_top30": 0.0,
            "confidence_accuracy_top50": 0.0,
            "confidence_accuracy_ge90": 0.0,
            "confidence_coverage_ge90": 0.0,
        }

    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    confidence = np.max(y_score, axis=1)
    order = np.argsort(-confidence)

    result: dict[str, float] = {}
    for coverage in (0.3, 0.5):
        keep = max(1, int(np.ceil(len(order) * coverage)))
        selected = order[:keep]
        result[f"confidence_accuracy_top{int(coverage * 100)}"] = float(
            accuracy_score(true_array[selected], pred_array[selected])
        )

    high_confidence_mask = confidence >= 0.9
    result["confidence_coverage_ge90"] = float(high_confidence_mask.mean())
    result["confidence_accuracy_ge90"] = (
        float(accuracy_score(true_array[high_confidence_mask], pred_array[high_confidence_mask]))
        if high_confidence_mask.any()
        else 0.0
    )
    return result
