import argparse
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BEST_ACCURACY = 0.9068
REFERENCE_MARGIN = 0.001
DEFAULT_REFERENCE_TARGET = REFERENCE_BEST_ACCURACY + REFERENCE_MARGIN
SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
MULTI_SEED_NUMERIC_KEYS = [
    "best_val_accuracy",
    "test_accuracy_at_best_val",
    "test_auc_at_best_val",
    "test_f1_at_best_val",
    "test_balanced_accuracy_at_best_val",
    "test_sensitivity_at_best_val",
    "test_specificity_at_best_val",
    "test_confidence_accuracy_top30_at_best_val",
    "test_confidence_accuracy_top50_at_best_val",
    "test_confidence_accuracy_ge90_at_best_val",
    "best_test_accuracy",
    "max_test_accuracy_observed",
    "final_test_auc",
    "final_test_f1",
    "final_test_balanced_accuracy",
    "rounds_to_target_accuracy",
    "final_mean_epsilon",
    "final_cumulative_epsilon",
    "avg_round_time_sec",
    "total_communication_mb",
    "avg_round_communication_mb",
]


def base_experiment_name(name: str) -> str:
    match = SEED_SUFFIX_RE.match(name)
    return match.group("base") if match else name


def seed_from_experiment_name(name: str) -> int | None:
    match = SEED_SUFFIX_RE.match(name)
    return int(match.group("seed")) if match else None


def resolve_rounds_to_target(history: pd.DataFrame, target_accuracy: float) -> int | None:
    if "val_accuracy" not in history.columns:
        return None
    reached = history[history["val_accuracy"] >= target_accuracy]
    if reached.empty:
        return None
    return int(reached.iloc[0]["round"])


def to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def select_best_validation_row(history: pd.DataFrame) -> pd.Series:
    sortable = history.copy()
    for column in ("val_accuracy", "val_auc", "val_f1"):
        if column not in sortable.columns:
            sortable[column] = 0.0
    return sortable.sort_values(
        by=["val_accuracy", "val_auc", "val_f1", "round"],
        ascending=[False, False, False, True],
    ).iloc[0]


def row_float_or_none(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    return to_float_or_none(row.get(key))


def cumulative_epsilon_series(history: pd.DataFrame) -> pd.Series:
    if "cumulative_epsilon" in history.columns:
        return pd.to_numeric(history["cumulative_epsilon"], errors="coerce")
    if "mean_epsilon" not in history.columns:
        return pd.Series([float("nan")] * len(history))
    return pd.to_numeric(history["mean_epsilon"], errors="coerce").fillna(0.0).cumsum()


def build_record(
    experiment_dir: Path,
    target_accuracy: float,
    reference_accuracy: float = DEFAULT_REFERENCE_TARGET,
) -> dict[str, object] | None:
    summary_path = experiment_dir / "summary.json"
    history_path = experiment_dir / "history.csv"
    if not summary_path.exists() or not history_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    history = pd.read_csv(history_path)
    if history.empty:
        return None

    last_row = history.iloc[-1]
    best_val_row = select_best_validation_row(history)
    rounds_to_target = resolve_rounds_to_target(history, target_accuracy)
    test_accuracy_at_best_val = float(best_val_row.get("test_accuracy", 0.0))
    test_auc_at_best_val = float(best_val_row.get("test_auc", 0.0))
    cumulative_eps = cumulative_epsilon_series(history)
    accuracy_gap_to_reference = max(float(reference_accuracy) - test_accuracy_at_best_val, 0.0)
    return {
        "experiment_name": experiment_dir.name,
        "base_experiment_name": base_experiment_name(experiment_dir.name),
        "mechanism": summary.get("mechanism", "unknown"),
        "privacy_enabled": summary.get("privacy_enabled", False),
        "privacy_accountant": summary.get("privacy_accountant", ""),
        "privacy_accountant_note": summary.get("privacy_accountant_note", ""),
        "seed": summary.get("seed", ""),
        "best_val_accuracy": summary.get("best_val_accuracy", 0.0),
        "best_val_round": int(best_val_row.get("round", 0)),
        "test_accuracy_at_best_val": test_accuracy_at_best_val,
        "test_f1_at_best_val": row_float_or_none(best_val_row, "test_f1"),
        "test_auc_at_best_val": test_auc_at_best_val,
        "test_balanced_accuracy_at_best_val": row_float_or_none(best_val_row, "test_balanced_accuracy"),
        "test_sensitivity_at_best_val": row_float_or_none(best_val_row, "test_sensitivity"),
        "test_specificity_at_best_val": row_float_or_none(best_val_row, "test_specificity"),
        "test_confidence_accuracy_top30_at_best_val": row_float_or_none(
            best_val_row, "test_confidence_accuracy_top30"
        ),
        "test_confidence_accuracy_top50_at_best_val": row_float_or_none(
            best_val_row, "test_confidence_accuracy_top50"
        ),
        "test_confidence_accuracy_ge90_at_best_val": row_float_or_none(
            best_val_row, "test_confidence_accuracy_ge90"
        ),
        "test_confidence_coverage_ge90_at_best_val": row_float_or_none(
            best_val_row, "test_confidence_coverage_ge90"
        ),
        "best_test_accuracy": test_accuracy_at_best_val,
        "max_test_accuracy_observed": float(history.get("test_accuracy", pd.Series([0.0])).max()),
        "reference_best_accuracy": float(reference_accuracy),
        "meets_reference_best_accuracy": test_accuracy_at_best_val >= float(reference_accuracy),
        "accuracy_gap_to_reference": accuracy_gap_to_reference,
        "test_selection_rule": "validation_best_accuracy_then_auc_then_f1",
        "threshold_tuning": summary.get("threshold_tuning", "decision_threshold" in history.columns),
        "threshold_metric": summary.get("threshold_metric", ""),
        "decision_threshold_at_best_val": row_float_or_none(best_val_row, "decision_threshold"),
        "final_test_auc": summary.get("final_test_auc", 0.0),
        "final_test_f1": row_float_or_none(last_row, "test_f1"),
        "final_test_balanced_accuracy": row_float_or_none(last_row, "test_balanced_accuracy"),
        "final_test_sensitivity": row_float_or_none(last_row, "test_sensitivity"),
        "final_test_specificity": row_float_or_none(last_row, "test_specificity"),
        "rounds": int(len(history)),
        "target_val_accuracy": float(target_accuracy),
        "rounds_to_target_accuracy": rounds_to_target,
        "final_mean_epsilon": to_float_or_none(summary.get("final_mean_epsilon", last_row.get("mean_epsilon"))),
        "final_cumulative_epsilon": to_float_or_none(
            summary.get("final_cumulative_epsilon", cumulative_eps.dropna().iloc[-1] if not cumulative_eps.dropna().empty else None)
        ),
        "max_mean_epsilon": to_float_or_none(history.get("mean_epsilon", pd.Series([float("nan")])).max()),
        "max_cumulative_epsilon": to_float_or_none(cumulative_eps.max()),
        "avg_round_time_sec": float(history.get("round_time_sec", pd.Series([0.0])).mean()),
        "total_communication_mb": float(
            summary.get(
                "total_communication_mb",
                history.get("cumulative_communication_mb", pd.Series([0.0])).iloc[-1],
            )
        ),
        "avg_round_communication_mb": float(
            summary.get(
                "avg_round_communication_mb",
                history.get("round_communication_mb", pd.Series([0.0])).mean(),
            )
        ),
        "model_size_mb": float(summary.get("model_size_mb", 0.0)),
        "classes": ",".join(summary.get("classes", [])),
        "config": summary.get("config", ""),
    }


def aggregate_seed_records(base_name: str, records: list[dict[str, object]], reference_accuracy: float) -> dict[str, object]:
    frame = pd.DataFrame(records)
    seeds = [str(record.get("seed", "")) for record in records if record.get("seed", "") != ""]
    representative = max(records, key=lambda row: float(row.get("test_accuracy_at_best_val", 0.0) or 0.0))
    aggregate = dict(representative)
    aggregate["experiment_name"] = base_name
    aggregate["base_experiment_name"] = base_name
    aggregate["seed"] = ",".join(seeds)
    aggregate["seed_count"] = len(records)
    aggregate["multi_seed"] = True

    for key in MULTI_SEED_NUMERIC_KEYS:
        if key not in frame.columns:
            continue
        values = pd.to_numeric(frame[key], errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate[key] = mean_value
        aggregate[f"{key}_mean"] = mean_value
        aggregate[f"{key}_std"] = std_value

    aggregate["reference_best_accuracy"] = float(reference_accuracy)
    aggregate["meets_reference_best_accuracy"] = float(aggregate.get("test_accuracy_at_best_val", 0.0)) >= float(reference_accuracy)
    aggregate["accuracy_gap_to_reference"] = max(
        float(reference_accuracy) - float(aggregate.get("test_accuracy_at_best_val", 0.0)),
        0.0,
    )
    return aggregate


def collapse_multi_seed_records(records: list[dict[str, object]], reference_accuracy: float) -> list[dict[str, object]]:
    by_base: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_base.setdefault(str(record.get("base_experiment_name", record["experiment_name"])), []).append(record)

    collapsed: list[dict[str, object]] = []
    for base_name, group in by_base.items():
        seed_records = [
            record for record in group
            if seed_from_experiment_name(str(record["experiment_name"])) is not None
        ]
        seed_values = {record.get("seed") for record in seed_records if record.get("seed", "") != ""}
        if len(seed_values) >= 2:
            collapsed.append(aggregate_seed_records(base_name, seed_records, reference_accuracy=reference_accuracy))
            continue
        for record in group:
            record.setdefault("seed_count", 1)
            record.setdefault("multi_seed", False)
            collapsed.append(record)
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总正式实验结果。")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(ROOT / "outputs"),
        help="实验输出根目录。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ham10000_",
        help="只汇总指定前缀的实验目录，留空则汇总全部。",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="experiment_summary",
        help="汇总结果文件名，不含扩展名。",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        help="用于统计达到目标准确率所需轮数的验证准确率阈值，默认 0.80。",
    )
    parser.add_argument(
        "--reference-accuracy",
        type=float,
        default=DEFAULT_REFERENCE_TARGET,
        help="用于衡量是否超过参考文献最佳水平的测试准确率阈值，默认 0.9068 + 0.001。",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    records = []
    for experiment_dir in sorted(outputs_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if args.prefix and not experiment_dir.name.startswith(args.prefix):
            continue
        record = build_record(
            experiment_dir,
            target_accuracy=args.target_accuracy,
            reference_accuracy=args.reference_accuracy,
        )
        if record is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError("未找到可汇总的实验结果，请先运行正式实验。")

    records = collapse_multi_seed_records(records, reference_accuracy=args.reference_accuracy)

    frame = pd.DataFrame(records).sort_values(
        by=["test_accuracy_at_best_val", "test_auc_at_best_val"],
        ascending=False,
    )
    csv_path = outputs_dir / f"{args.output_name}.csv"
    json_path = outputs_dir / f"{args.output_name}.json"
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"实验汇总表已生成：{csv_path}")
    print(f"实验汇总 JSON 已生成：{json_path}")


if __name__ == "__main__":
    main()
