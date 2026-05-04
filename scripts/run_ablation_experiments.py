import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_experiment.py"
COLLECT_SCRIPT = ROOT / "scripts" / "collect_experiment_results.py"


def write_variant(base_config: dict, name: str, updates: dict) -> Path:
    payload = deepcopy(base_config)
    payload["output"]["experiment_name"] = name
    for section, values in updates.items():
        payload.setdefault(section, {})
        payload[section].update(values)

    target_dir = ROOT / "configs" / "ablations"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{name}.yaml"
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return target_path


def build_variants(base_config: dict) -> list[Path]:
    variants: list[Path] = []
    for value in [0.1, 0.5, 1.0, 5.0, 10.0]:
        suffix = str(value).replace(".", "_")
        variants.append(
            write_variant(
                base_config,
                name=f"ham10000_ablation_alpha_{suffix}",
                updates={"dataset": {"dirichlet_alpha": value}},
            )
        )
    for value in [3, 5, 10, 20]:
        variants.append(
            write_variant(
                base_config,
                name=f"ham10000_ablation_clients_{value}",
                updates={"dataset": {"num_clients": value}},
            )
        )
    for value in [1.0, 3.0, 5.0, 10.0]:
        suffix = str(value).replace(".", "_")
        variants.append(
            write_variant(
                base_config,
                name=f"ham10000_ablation_clip_{suffix}",
                updates={"privacy": {"clip_norm": value}},
            )
        )
    return variants


def experiment_name(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return str(payload["output"]["experiment_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并运行 Non-IID、客户端数量和裁剪阈值消融实验。")
    parser.add_argument("--base-config", type=str, default=str(ROOT / "configs" / "ham10000_hybrid.yaml"))
    parser.add_argument("--generate-only", action="store_true", help="只生成配置，不运行训练。")
    parser.add_argument("--skip-existing", action="store_true", help="已有 summary.json 时跳过训练。")
    parser.add_argument("--collect", action="store_true", help="运行结束后刷新 experiment_summary。")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    with base_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    config_paths = build_variants(base_config)
    print(f"已生成消融配置：{len(config_paths)} 个")

    if args.generate_only:
        for config_path in config_paths:
            print(config_path)
        return

    for index, config_path in enumerate(config_paths, start=1):
        name = experiment_name(config_path)
        summary_path = ROOT / "outputs" / name / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[{index}/{len(config_paths)}] 跳过：{name}")
            continue
        print(f"[{index}/{len(config_paths)}] 运行：{name}")
        subprocess.run([sys.executable, str(RUN_SCRIPT), "--config", str(config_path)], check=True)

    if args.collect:
        subprocess.run([sys.executable, str(COLLECT_SCRIPT)], check=True)


if __name__ == "__main__":
    main()
