import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_experiment.py"

GROUPS = {
    "main": [
        "configs/ham10000_centralized.yaml",
        "configs/ham10000_nodp.yaml",
        "configs/ham10000_gaussian.yaml",
        "configs/ham10000_laplace.yaml",
        "configs/ham10000_hybrid.yaml",
        "configs/ham10000_hybrid_adaptive.yaml",
    ],
    "noise_ablation": [
        "configs/ham10000_hybrid_noise_low.yaml",
        "configs/ham10000_hybrid.yaml",
        "configs/ham10000_hybrid_noise_high.yaml",
    ],
    "noniid_ablation": [
        "configs/ham10000_hybrid_alpha_10.yaml",
        "configs/ham10000_hybrid_alpha_06.yaml",
        "configs/ham10000_hybrid.yaml",
    ],
    "highacc": [
        "configs/ham10000_highacc_centralized.yaml",
        "configs/ham10000_highacc_convnext.yaml",
        "configs/ham10000_literature_target.yaml",
    ],
    "accuracy90": [
        "configs/ham10000_accuracy90.yaml",
        "configs/ham10000_accuracy90_efficientnet_b4.yaml",
    ],
}


def resolve_group(group: str) -> list[Path]:
    if group == "all":
        ordered: list[str] = []
        for key in ("main", "noise_ablation", "noniid_ablation", "highacc", "accuracy90"):
            ordered.extend(GROUPS[key])
        seen: set[str] = set()
        unique = []
        for item in ordered:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return [ROOT / item for item in unique]
    return [ROOT / item for item in GROUPS[group]]


def read_experiment_name(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["output"]["experiment_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行正式毕设实验。")
    parser.add_argument(
        "--group",
        choices=["main", "noise_ablation", "noniid_ablation", "highacc", "accuracy90", "all"],
        default="main",
        help="要运行的实验组。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果对应实验结果已存在，则跳过该实验。",
    )
    args = parser.parse_args()

    config_paths = resolve_group(args.group)
    print(f"准备运行实验组：{args.group}")
    print(f"实验数量：{len(config_paths)}")

    for index, config_path in enumerate(config_paths, start=1):
        experiment_name = read_experiment_name(config_path)
        summary_path = ROOT / "outputs" / experiment_name / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[{index}/{len(config_paths)}] 跳过：{experiment_name}（已有结果）")
            continue

        print(f"[{index}/{len(config_paths)}] 开始运行：{experiment_name}")
        subprocess.run(
            [sys.executable, str(RUN_SCRIPT), "--config", str(config_path)],
            check=True,
        )

    print("批量实验运行完成。")


if __name__ == "__main__":
    main()
