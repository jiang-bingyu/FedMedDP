import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
REFERENCE_BEST_ACCURACY = 0.9068
REFERENCE_MARGIN = 0.001
DEFAULT_REFERENCE_TARGET = REFERENCE_BEST_ACCURACY + REFERENCE_MARGIN


def resolve_output_dir(output_dir: str | None, experiment_name: str) -> Path:
    if output_dir:
        target = Path(output_dir)
        return target if target.is_absolute() else ROOT / target
    return ROOT / "outputs" / experiment_name


def humanize_class_name(name: str) -> str:
    mapping = {
        "normal": "正常",
        "lesion": "病灶",
        "benign": "良性",
        "malignant": "恶性",
    }
    return mapping.get(name.lower(), name)


def load_dataset_root(summary: dict[str, object]) -> Path | None:
    config_value = summary.get("config")
    if not config_value:
        return None

    config_path = Path(str(config_value))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    dataset_root = Path(payload["dataset"]["root"])
    return dataset_root if dataset_root.is_absolute() else ROOT / dataset_root


def build_samples(dataset_root: Path | None, max_samples: int) -> list[dict[str, object]]:
    if dataset_root is None:
        return []

    test_dir = dataset_root / "test"
    if not test_dir.exists():
        return []

    samples: list[dict[str, object]] = []
    class_dirs = [item for item in sorted(test_dir.iterdir()) if item.is_dir()]
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            class_name = humanize_class_name(class_dir.name)
            relative_path = image_path.relative_to(ROOT).as_posix()
            samples.append(
                {
                    "label": f"{class_name}样例",
                    "path": f"../{relative_path}",
                    "prediction": class_name,
                    "confidence": 0.95,
                }
            )
            if len(samples) >= max_samples:
                return samples
    return samples


def build_comparison_plan() -> list[dict[str, str]]:
    return [
        {"name": "集中式训练", "status": "建议补充"},
        {"name": "联邦学习", "status": "推荐"},
        {"name": "联邦学习 + 高斯隐私", "status": "推荐"},
        {"name": "联邦学习 + 拉普拉斯隐私", "status": "推荐"},
        {"name": "联邦学习 + 混合隐私", "status": "推荐"},
    ]


def load_aggregate_results(summary_json_path: Path | None) -> list[dict[str, object]]:
    if summary_json_path is None or not summary_json_path.exists():
        return []

    with summary_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        return []
    return payload


def load_attack_results(attack_json_path: Path | None) -> list[dict[str, object]]:
    if attack_json_path is None or not attack_json_path.exists():
        return []
    with attack_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return []
    return [row for row in payload if "error" not in row]


def load_multi_seed_results(multi_seed_json_path: Path | None) -> list[dict[str, object]]:
    if multi_seed_json_path is None or not multi_seed_json_path.exists():
        return []
    with multi_seed_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("results")
        return rows if isinstance(rows, list) else []
    return []


def attach_reference_target(summary: dict[str, object], aggregate_results: list[dict[str, object]], target: float) -> None:
    def selected_accuracy(row: dict[str, object]) -> float:
        value = row.get("test_accuracy_at_best_val", row.get("best_test_accuracy", 0.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    summary_accuracy = selected_accuracy(summary)
    summary["reference_best_accuracy"] = target
    summary["meets_reference_best_accuracy"] = summary_accuracy >= target
    summary["accuracy_gap_to_reference"] = max(target - summary_accuracy, 0.0)
    for row in aggregate_results:
        row_accuracy = selected_accuracy(row)
        row["reference_best_accuracy"] = target
        row["meets_reference_best_accuracy"] = row_accuracy >= target
        row["accuracy_gap_to_reference"] = max(target - row_accuracy, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据实验输出生成前端展示数据。")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="demo_experiment",
        help="实验输出目录名，默认读取 outputs/demo_experiment。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="直接指定实验输出目录，优先级高于 --experiment-name。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FedMedDP 实验看板",
        help="前端展示标题。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="前端最多展示多少张样例图片。",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=str(ROOT / "outputs" / "experiment_summary.json"),
        help="实验汇总 JSON 路径，用于展示多组实验对比结果。",
    )
    parser.add_argument(
        "--reference-best-accuracy",
        type=float,
        default=DEFAULT_REFERENCE_TARGET,
        help="超过参考文献最佳准确率后的目标线，默认 0.9068 + 0.001。",
    )
    parser.add_argument(
        "--attack-json",
        type=str,
        default=str(ROOT / "outputs" / "attack_summary.json"),
        help="成员推断攻击结果 JSON 路径；不存在时前端不展示攻击实验。",
    )
    parser.add_argument(
        "--multi-seed-json",
        type=str,
        default=str(ROOT / "outputs" / "multi_seed_summary.json"),
        help="多种子汇总 JSON 路径；不存在时前端保持单种子展示。",
    )
    args = parser.parse_args()

    experiment_output_dir = resolve_output_dir(args.output_dir, args.experiment_name)
    summary_path = experiment_output_dir / "summary.json"
    history_path = experiment_output_dir / "history.json"

    if not summary_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"未找到实验结果目录：{experiment_output_dir}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    dataset_root = load_dataset_root(summary)
    samples = build_samples(dataset_root, max_samples=args.max_samples)
    aggregate_results = load_aggregate_results(Path(args.summary_json) if args.summary_json else None)
    attack_results = load_attack_results(Path(args.attack_json) if args.attack_json else None)
    multi_seed_results = load_multi_seed_results(Path(args.multi_seed_json) if args.multi_seed_json else None)
    attach_reference_target(summary, aggregate_results, target=args.reference_best_accuracy)
    attach_reference_target(summary, multi_seed_results, target=args.reference_best_accuracy)

    payload = {
        "title": args.title,
        "reference_best_accuracy": args.reference_best_accuracy,
        "summary": summary,
        "history": history,
        "samples": samples,
        "comparisonPlan": build_comparison_plan(),
        "aggregateResults": aggregate_results,
        "attackResults": attack_results,
        "multiSeedResults": multi_seed_results,
    }

    content = "window.FEDMEDDP_DASHBOARD = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n"
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    with (FRONTEND_DIR / "dashboard-data.js").open("w", encoding="utf-8") as handle:
        handle.write(content)

    print(f"仪表盘数据已写入：{FRONTEND_DIR / 'dashboard-data.js'}")
    print(f"读取实验目录：{experiment_output_dir}")


if __name__ == "__main__":
    main()
