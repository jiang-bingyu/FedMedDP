from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

from collect_experiment_results import aggregate_seed_records, build_record


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_experiment.py"
GENERATED_CONFIG_DIR = ROOT / "configs" / "generated" / "multi_seed"


def project_relative(path: Path) -> Path:
    return path.resolve().relative_to(ROOT.resolve())


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return payload


def write_seed_config(
    base_payload: dict[str, object],
    base_name: str,
    seed: int,
    config_dir: Path,
    rounds: int | None,
) -> Path:
    payload = deepcopy(base_payload)
    output_payload = dict(payload.get("output") or {})
    output_payload["experiment_name"] = f"{base_name}_seed{seed}"
    payload["output"] = output_payload
    payload["seed"] = int(seed)
    if rounds is not None:
        federated_payload = dict(payload.get("federated") or {})
        federated_payload["rounds"] = int(rounds)
        payload["federated"] = federated_payload

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{base_name}_seed{seed}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return config_path


def run_seed_config(config_path: Path, dry_run: bool) -> None:
    config_arg = project_relative(config_path).as_posix()
    command = [sys.executable, str(RUN_SCRIPT), "--config", config_arg]
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return
    subprocess.run(command, check=True, cwd=ROOT)


def collect_seed_records(
    base_name: str,
    seeds: list[int],
    base_seed: int | None,
    reuse_base_seed: bool,
    target_accuracy: float,
    reference_accuracy: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in seeds:
        seed_dir = ROOT / "outputs" / f"{base_name}_seed{seed}"
        base_dir = ROOT / "outputs" / base_name
        experiment_dir = seed_dir
        if (
            reuse_base_seed
            and
            base_seed == seed
            and not (seed_dir / "summary.json").exists()
            and (base_dir / "summary.json").exists()
            and (base_dir / "history.csv").exists()
        ):
            experiment_dir = base_dir
        record = build_record(
            experiment_dir,
            target_accuracy=target_accuracy,
            reference_accuracy=reference_accuracy,
        )
        if record is not None:
            records.append(record)
        else:
            print(f"未找到 seed={seed} 的可汇总结果，已跳过。")
    return records


def write_multi_seed_summary(base_name: str, records: list[dict[str, object]], reference_accuracy: float) -> None:
    if not records:
        raise FileNotFoundError("未找到可汇总的多种子实验结果。")

    output_dir = ROOT / "outputs"
    aggregate = aggregate_seed_records(base_name, records, reference_accuracy=reference_accuracy)
    payload = [aggregate]

    json_path = output_dir / "multi_seed_summary.json"
    csv_path = output_dir / "multi_seed_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(payload).to_csv(csv_path, index=False)

    print(f"多种子汇总 JSON 已生成：{json_path}")
    print(f"多种子汇总 CSV 已生成：{csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行同一配置的多随机种子实验并汇总 mean/std。")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "ham10000_literature_target.yaml"),
        help="基础配置文件路径。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[2026, 2027, 2028],
        help="随机种子列表，默认 2026 2027 2028。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果某个 seed 的 summary.json 已存在，则跳过训练。",
    )
    parser.add_argument(
        "--reuse-base-seed",
        action="store_true",
        help="复用基础实验目录作为基础配置 seed 的结果；默认不复用，默认会为每个 seed 生成独立输出目录。",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="只汇总已有多种子结果，不启动训练。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的训练命令，不启动训练。",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许只汇总部分 seed；默认要求请求的 seed 都有结果，防止误生成不完整的 mean/std。",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="覆盖基础配置中的 federated.rounds；例如 --rounds 60 可避免多种子实验重复跑 160 轮。",
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
        default=0.9078,
        help="参考文献超越目标，默认 0.9078。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    payload = load_yaml(config_path)
    output_payload = payload.get("output") or {}
    if not isinstance(output_payload, dict) or not output_payload.get("experiment_name"):
        raise ValueError("基础配置必须包含 output.experiment_name。")
    base_name = str(output_payload["experiment_name"])
    seeds = [int(seed) for seed in args.seeds]
    base_seed = int(payload.get("seed")) if payload.get("seed") is not None else None

    generated_config_dir = GENERATED_CONFIG_DIR
    for seed in seeds:
        experiment_name = f"{base_name}_seed{seed}"
        summary_path = ROOT / "outputs" / experiment_name / "summary.json"
        base_summary_path = ROOT / "outputs" / base_name / "summary.json"
        seed_config_path = generated_config_dir / f"{base_name}_seed{seed}.yaml"
        if not args.dry_run and not args.collect_only:
            seed_config_path = write_seed_config(
                payload,
                base_name=base_name,
                seed=seed,
                config_dir=generated_config_dir,
                rounds=args.rounds,
            )
        if args.collect_only:
            continue
        if args.skip_existing and args.reuse_base_seed and seed == base_seed and base_summary_path.exists():
            print(f"跳过 {experiment_name}：复用已有基础实验 {base_name}")
            continue
        if args.skip_existing and summary_path.exists():
            print(f"跳过 {experiment_name}：已有 summary.json")
            continue
        run_seed_config(seed_config_path, dry_run=args.dry_run)

    if not args.dry_run:
        records = collect_seed_records(
            base_name=base_name,
            seeds=seeds,
            base_seed=base_seed,
            reuse_base_seed=args.reuse_base_seed,
            target_accuracy=args.target_accuracy,
            reference_accuracy=args.reference_accuracy,
        )
        if len(records) < len(seeds):
            message = f"请求 {len(seeds)} 个种子，只汇总到 {len(records)} 个结果。"
            if not args.allow_partial:
                raise FileNotFoundError(message + "请先补齐缺失 seed，或显式添加 --allow-partial。")
            print("警告：" + message)
        write_multi_seed_summary(
            base_name=base_name,
            records=records,
            reference_accuracy=args.reference_accuracy,
        )


if __name__ == "__main__":
    main()
