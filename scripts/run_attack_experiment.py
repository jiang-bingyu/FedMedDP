import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedmeddp.client import collect_predictions, prepare_model_for_runtime
from fedmeddp.config import load_config
from fedmeddp.data import build_datasets, build_transforms
from fedmeddp.model import build_model


def resolve_config_path(experiment_name: str, output_dir: Path) -> Path:
    summary_path = output_dir / experiment_name / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        configured = Path(str(summary.get("config", "")))
        if configured.exists():
            return configured
    fallback = ROOT / "configs" / f"{experiment_name}.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"找不到 {experiment_name} 对应配置文件。")


def patch_eval_transform(dataset, transform) -> None:
    if hasattr(dataset, "dataset"):
        patch_eval_transform(dataset.dataset, transform)
        return
    if hasattr(dataset, "transform"):
        dataset.transform = transform


def prediction_features(targets: list[int], scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-8, 1.0)
    sorted_scores = np.sort(clipped, axis=1)
    max_confidence = sorted_scores[:, -1]
    margin = sorted_scores[:, -1] - sorted_scores[:, -2] if clipped.shape[1] > 1 else max_confidence
    true_confidence = clipped[np.arange(len(targets)), np.asarray(targets, dtype=np.int64)]
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    loss = -np.log(true_confidence)
    return np.column_stack([max_confidence, margin, true_confidence, entropy, loss])


def evaluate_attack(experiment_name: str, outputs_dir: Path) -> dict[str, object]:
    experiment_dir = outputs_dir / experiment_name
    model_path = experiment_dir / "final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"缺少模型文件：{model_path}")

    cfg = load_config(resolve_config_path(experiment_name, outputs_dir))
    cfg.model.pretrained = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_datasets(cfg.dataset, seed=cfg.seed)
    eval_transform = build_transforms(cfg.dataset, train=False)
    patch_eval_transform(bundle.train_dataset, eval_transform)

    loader_kwargs = {
        "batch_size": cfg.dataset.batch_size,
        "shuffle": False,
        "num_workers": cfg.dataset.num_workers,
        "pin_memory": cfg.dataset.pin_memory,
    }
    if cfg.dataset.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.dataset.persistent_workers
        loader_kwargs["prefetch_factor"] = max(int(cfg.dataset.prefetch_factor), 2)
    train_loader = DataLoader(bundle.train_dataset, **loader_kwargs)
    test_loader = DataLoader(bundle.test_dataset, **loader_kwargs)

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=max(cfg.model.num_classes, len(bundle.class_names)),
        dropout=cfg.model.dropout,
        pretrained=False,
    )
    model = prepare_model_for_runtime(model, cfg=cfg, device=device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    _, member_targets, member_scores = collect_predictions(model=model, loader=train_loader, device=device, cfg=cfg)
    _, nonmember_targets, nonmember_scores = collect_predictions(model=model, loader=test_loader, device=device, cfg=cfg)
    member_features = prediction_features(member_targets, member_scores)
    nonmember_features = prediction_features(nonmember_targets, nonmember_scores)

    sample_count = min(len(member_features), len(nonmember_features))
    if sample_count < 20:
        raise ValueError(f"{experiment_name} 可用于攻击评估的样本过少。")
    rng = np.random.default_rng(cfg.seed)
    member_indices = rng.choice(len(member_features), size=sample_count, replace=False)
    nonmember_indices = rng.choice(len(nonmember_features), size=sample_count, replace=False)
    features = np.vstack([member_features[member_indices], nonmember_features[nonmember_indices]])
    labels = np.concatenate([np.ones(sample_count, dtype=np.int64), np.zeros(sample_count, dtype=np.int64)])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.35,
        random_state=cfg.seed,
        stratify=labels,
    )
    attack_model = GradientBoostingClassifier(random_state=cfg.seed)
    attack_model.fit(x_train, y_train)
    probabilities = attack_model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    return {
        "experiment_name": experiment_name,
        "mechanism": cfg.privacy.mechanism if cfg.privacy.enabled else "none",
        "privacy_enabled": bool(cfg.privacy.enabled),
        "attack_accuracy": float(accuracy_score(y_test, predictions)),
        "attack_auc": float(roc_auc_score(y_test, probabilities)),
        "member_samples": int(sample_count),
        "attack_interpretation": "越接近 0.50 表示成员推断攻击越接近随机猜测，隐私防护越强。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行轻量级 membership inference attack 评估。")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ham10000_nodp", "ham10000_gaussian", "ham10000_hybrid", "ham10000_hybrid_adaptive"],
        help="要评估的实验名称。",
    )
    parser.add_argument("--outputs-dir", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--output-name", type=str, default="attack_summary")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    results = []
    for experiment_name in args.experiments:
        print(f"正在评估成员推断攻击：{experiment_name}")
        try:
            results.append(evaluate_attack(experiment_name=experiment_name, outputs_dir=outputs_dir))
        except Exception as exc:
            results.append({"experiment_name": experiment_name, "error": str(exc)})
            print(f"警告：{experiment_name} 攻击评估失败：{exc}")

    json_path = outputs_dir / f"{args.output_name}.json"
    csv_path = outputs_dir / f"{args.output_name}.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"警告：CSV 写入失败，仅保留 JSON：{exc}")
    print(f"攻击实验结果已写入：{json_path}")


if __name__ == "__main__":
    main()
