from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedmeddp.metrics import compute_classification_metrics


def resolve_experiment_dir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / value
    if candidate.exists():
        return candidate
    return ROOT / "outputs" / value


def load_prediction_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"未找到预测概率文件：{path}")
    payload = np.load(path, allow_pickle=True)
    targets = np.asarray(payload["targets"], dtype=np.int64)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"预测概率维度错误：{path}")
    return targets, scores


def average_predictions(
    experiment_dirs: list[Path],
    split: str,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    targets_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    for experiment_dir in experiment_dirs:
        targets, scores = load_prediction_file(experiment_dir / f"{split}_predictions_best.npz")
        targets_list.append(targets)
        scores_list.append(scores)

    reference_targets = targets_list[0]
    for experiment_dir, targets in zip(experiment_dirs[1:], targets_list[1:], strict=False):
        if targets.shape != reference_targets.shape or not np.array_equal(targets, reference_targets):
            raise ValueError(
                f"{experiment_dir} 的 {split} targets 与第一个实验不一致，不能做简单概率平均。"
            )

    stacked_scores = np.stack(scores_list, axis=0)
    if weights is None:
        averaged_scores = np.mean(stacked_scores, axis=0)
    else:
        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / weight_array.sum()
        averaged_scores = np.tensordot(weight_array, stacked_scores, axes=(0, 0))
    return reference_targets, averaged_scores


def metrics_at_threshold(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores[:, 1] >= threshold).astype(int)
    return compute_classification_metrics(
        y_true=targets.tolist(),
        y_pred=predictions.tolist(),
        y_score=scores,
        num_classes=2,
    )


def select_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    min_sensitivity: float,
) -> tuple[float, dict[str, float], bool]:
    thresholds = np.linspace(threshold_min, threshold_max, max(int(threshold_steps), 2))
    metric_key = metric_name.lower()
    best_threshold = 0.5
    best_metrics = metrics_at_threshold(targets, scores, best_threshold)
    best_candidate: tuple[float, float, float] | None = None
    constraint_satisfied = False

    for threshold in thresholds:
        metrics = metrics_at_threshold(targets, scores, float(threshold))
        if float(metrics["sensitivity"]) < min_sensitivity:
            continue
        score = float(metrics.get(metric_key, metrics["accuracy"]))
        candidate = (score, float(metrics["specificity"]), float(threshold))
        if best_candidate is None or candidate > best_candidate:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_candidate = candidate
            constraint_satisfied = True

    if constraint_satisfied:
        return best_threshold, best_metrics, True

    best_candidate = None
    for threshold in thresholds:
        metrics = metrics_at_threshold(targets, scores, float(threshold))
        score = float(metrics.get(metric_key, metrics["accuracy"]))
        candidate = (score, float(metrics["specificity"]), float(threshold))
        if best_candidate is None or candidate > best_candidate:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_candidate = candidate
    return best_threshold, best_metrics, False


def main() -> None:
    parser = argparse.ArgumentParser(description="对多个实验的 best-val 预测概率做集成评估。")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="实验名或输出目录，例如 ham10000_accuracy90_seed2026。",
    )
    parser.add_argument(
        "--output-name",
        default="ham10000_accuracy90_ensemble",
        help="集成评估输出目录名，默认 ham10000_accuracy90_ensemble。",
    )
    parser.add_argument("--threshold-metric", default="accuracy", help="验证集阈值选择指标。")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=181)
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.58,
        help="验证集阈值搜索时要求的最低 Sensitivity，默认 0.58。",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="可选的模型集成权重，数量必须与 --experiments 一致；默认等权平均。",
    )
    args = parser.parse_args()

    experiment_dirs = [resolve_experiment_dir(value) for value in args.experiments]
    missing = [str(path) for path in experiment_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("以下实验目录不存在：" + ", ".join(missing))
    if args.weights is not None:
        if len(args.weights) != len(experiment_dirs):
            raise ValueError("--weights 的数量必须与 --experiments 的数量一致。")
        if any(weight < 0 for weight in args.weights):
            raise ValueError("--weights 不允许包含负数。")
        if sum(args.weights) <= 0:
            raise ValueError("--weights 的总和必须大于 0。")
        ensemble_weights = [float(weight) / float(sum(args.weights)) for weight in args.weights]
    else:
        ensemble_weights = None

    val_targets, val_scores = average_predictions(experiment_dirs, split="val", weights=ensemble_weights)
    test_targets, test_scores = average_predictions(experiment_dirs, split="test", weights=ensemble_weights)
    threshold, val_metrics, constraint_satisfied = select_threshold(
        targets=val_targets,
        scores=val_scores,
        metric_name=args.threshold_metric,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        min_sensitivity=args.min_sensitivity,
    )
    test_metrics = metrics_at_threshold(test_targets, test_scores, threshold)

    output_dir = ROOT / "outputs" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "val_predictions_ensemble.npz",
        targets=val_targets,
        scores=val_scores.astype(np.float32),
        decision_threshold=np.asarray([threshold], dtype=np.float32),
    )
    np.savez_compressed(
        output_dir / "test_predictions_ensemble.npz",
        targets=test_targets,
        scores=test_scores.astype(np.float32),
        decision_threshold=np.asarray([threshold], dtype=np.float32),
    )

    summary = {
        "experiment_name": args.output_name,
        "mechanism": "ensemble",
        "privacy_enabled": False,
        "privacy_accountant": "not_applicable",
        "privacy_accountant_note": "预测概率集成不计算隐私预算。",
        "seed": ",".join(path.name.rsplit("_seed", 1)[-1] for path in experiment_dirs if "_seed" in path.name),
        "classes": ["benign", "malignant"],
        "best_val_accuracy": float(val_metrics["accuracy"]),
        "best_val_round": 1,
        "ensemble_members": [path.name for path in experiment_dirs],
        "ensemble_weights": ensemble_weights
        if ensemble_weights is not None
        else [1.0 / len(experiment_dirs)] * len(experiment_dirs),
        "threshold_metric": args.threshold_metric,
        "threshold_tuning": True,
        "threshold_tie_breaker": "validation_specificity_then_higher_threshold",
        "decision_threshold": threshold,
        "decision_threshold_at_best_val": threshold,
        "min_sensitivity": float(args.min_sensitivity),
        "sensitivity_constraint_satisfied": constraint_satisfied,
        "test_accuracy_at_best_val": float(test_metrics["accuracy"]),
        "test_f1_at_best_val": float(test_metrics["f1"]),
        "test_auc_at_best_val": float(test_metrics["auc"]),
        "test_balanced_accuracy_at_best_val": float(test_metrics["balanced_accuracy"]),
        "test_sensitivity_at_best_val": float(test_metrics["sensitivity"]),
        "test_specificity_at_best_val": float(test_metrics["specificity"]),
        "val_metrics": {key: float(value) for key, value in val_metrics.items()},
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "best_test_accuracy": float(test_metrics["accuracy"]),
        "max_test_accuracy_observed": float(test_metrics["accuracy"]),
        "test_selection_rule": "ensemble_threshold_selected_on_validation_predictions",
        "final_test_auc": float(test_metrics["auc"]),
        "final_test_f1": float(test_metrics["f1"]),
        "final_test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "final_test_sensitivity": float(test_metrics["sensitivity"]),
        "final_test_specificity": float(test_metrics["specificity"]),
        "total_communication_mb": 0.0,
        "avg_round_communication_mb": 0.0,
        "model_size_mb": 0.0,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    history_record = {
        "round": 1,
        "decision_threshold": threshold,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_auc": val_metrics["auc"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_sensitivity": val_metrics["sensitivity"],
        "val_specificity": val_metrics["specificity"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "round_time_sec": 0.0,
        "round_communication_mb": 0.0,
        "cumulative_communication_mb": 0.0,
    }
    pd.DataFrame([history_record]).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame([history_record]).to_json(output_dir / "history.json", orient="records", indent=2)

    print(f"集成评估完成：{output_dir}")
    print(f"阈值：{threshold:.4f}")
    print(
        "测试集："
        f"Accuracy={test_metrics['accuracy']:.4f}, "
        f"AUC={test_metrics['auc']:.4f}, "
        f"Sensitivity={test_metrics['sensitivity']:.4f}, "
        f"Specificity={test_metrics['specificity']:.4f}"
    )


if __name__ == "__main__":
    main()
