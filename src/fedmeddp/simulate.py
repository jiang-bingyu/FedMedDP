from __future__ import annotations

import json
from pathlib import Path
import random
import time

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch

from .client import (
    collect_predictions,
    evaluate_predictions,
    prepare_model_for_runtime,
    run_client_update,
    tune_binary_threshold_from_predictions,
)
from .config import ExperimentConfig, load_config
from .data import build_client_loaders, build_datasets
from .model import build_model
from .privacy import resolve_privacy_accountant
from .server import (
    aggregate_client_updates,
    mean_epsilon,
    mean_hybrid_alpha,
    mean_train_loss,
    mean_update_norm,
    sample_clients,
)
from .state import clone_state_dict


def configure_matplotlib_fonts() -> None:
    candidate_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    available_candidates = [font_name for font_name in candidate_fonts if font_name in installed_fonts]

    if available_candidates:
        matplotlib.rcParams["font.sans-serif"] = available_candidates + ["DejaVu Sans"]
    else:
        # 保留一组常见中文字体候选，便于系统安装字体后直接生效。
        matplotlib.rcParams["font.sans-serif"] = candidate_fonts + ["DejaVu Sans"]
        print(
            "警告：当前环境未检测到常见中文字体，训练曲线中的中文标题可能显示为方框。"
            "如仍异常，请安装 Noto Sans CJK 或 WenQuanYi Zen Hei 字体。"
        )

    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def state_dict_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in state_dict.values():
        total += tensor.numel() * tensor.element_size()
    return total


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    if cfg.runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.runtime.device)


def configure_torch_runtime(cfg: ExperimentConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cfg.runtime.cudnn_benchmark)
        if cfg.runtime.cudnn_benchmark:
            torch.backends.cudnn.deterministic = False


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def last_finite_value(values: pd.Series) -> float | None:
    cleaned = finite_series(values).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def cumulative_epsilon_from_history(history: pd.DataFrame) -> pd.Series:
    if "cumulative_epsilon" in history.columns:
        return finite_series(history["cumulative_epsilon"])
    if "mean_epsilon" not in history.columns:
        return pd.Series([float("nan")] * len(history))
    return finite_series(history["mean_epsilon"]).fillna(0.0).cumsum()


def scale_axis_series(values: pd.Series) -> tuple[pd.Series, str]:
    cleaned = finite_series(values)
    max_value = cleaned.dropna().max() if not cleaned.dropna().empty else 0.0
    if max_value >= 1000:
        return cleaned / 1000.0, "×10³"
    return cleaned, ""


def select_best_validation_row(history: pd.DataFrame) -> pd.Series:
    """Select a reporting checkpoint without peeking at test metrics."""
    ranking_columns = ["val_accuracy", "val_auc", "val_f1"]
    sortable = history.copy()
    for column in ranking_columns:
        if column not in sortable.columns:
            sortable[column] = 0.0
    return sortable.sort_values(
        by=["val_accuracy", "val_auc", "val_f1", "round"],
        ascending=[False, False, False, True],
    ).iloc[0]


def early_stopping_score(record: dict[str, float | int | str], metric_name: str) -> float:
    metric_key = metric_name.strip().lower()
    if metric_key in {"", "accuracy"}:
        metric_key = "val_accuracy"
    value = record.get(metric_key)
    if value is None:
        raise KeyError(f"early_stopping_metric 不存在：{metric_name}")
    return float(value)


def save_curves(
    history: pd.DataFrame,
    output_dir: Path,
    privacy_enabled: bool,
    mechanism: str,
    privacy_accountant: str,
    privacy_accountant_note: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis in axes:
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)

    axes[0].plot(history["round"], history["val_accuracy"], label="val_acc", linewidth=2.0)
    axes[0].plot(history["round"], history["test_accuracy"], label="test_acc", linewidth=2.0)
    axes[0].set_title("准确率")
    axes[0].set_xlabel("通信轮次")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["round"], history["val_loss"], label="val_loss", linewidth=2.0)
    axes[1].plot(history["round"], history["train_loss"], label="train_loss", linewidth=2.0)
    axes[1].set_title("损失")
    axes[1].set_xlabel("通信轮次")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    update_series = finite_series(history["mean_update_norm"])
    axes[2].plot(history["round"], update_series, label="update_norm", color="#b86a4a", linewidth=2.0)
    axes[2].set_xlabel("通信轮次")
    axes[2].set_ylabel("更新范数")

    mechanism_lower = mechanism.lower()
    if not privacy_enabled:
        axes[2].set_title("模型更新范数")
        axes[2].legend()
        axes[2].text(
            0.02,
            0.92,
            "NoDP：不展示 epsilon",
            transform=axes[2].transAxes,
            fontsize=10,
            color="#55636a",
            va="top",
        )
    elif privacy_accountant == "not_available":
        axes[2].set_title("更新范数")
        axes[2].legend()
        axes[2].text(
            0.02,
            0.92,
            privacy_accountant_note,
            transform=axes[2].transAxes,
            fontsize=10,
            color="#55636a",
            va="top",
            wrap=True,
        )
    else:
        epsilon_series, epsilon_unit = scale_axis_series(cumulative_epsilon_from_history(history))
        epsilon_axis = axes[2].twinx()
        epsilon_axis.grid(False)
        epsilon_axis.plot(
            history["round"],
            epsilon_series,
            label="cumulative_epsilon",
            color="#2f7a5e",
            linestyle="--",
            linewidth=2.0,
        )
        epsilon_axis.set_ylabel(f"累计 epsilon {epsilon_unit}".strip())
        epsilon_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
        title = "累计隐私预算 / 更新"
        if mechanism_lower == "hybrid":
            title = "累计隐私预算 / 更新（Gaussian 部分近似）"
        axes[2].set_title(title)
        lines_left, labels_left = axes[2].get_legend_handles_labels()
        lines_right, labels_right = epsilon_axis.get_legend_handles_labels()
        axes[2].legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "curves.png", dpi=200)
    plt.close(fig)


def run_simulation(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg)
    configure_torch_runtime(cfg, device)
    privacy_accountant, privacy_accountant_note = resolve_privacy_accountant(
        enabled=cfg.privacy.enabled,
        mechanism=cfg.privacy.mechanism,
        noise_multiplier=cfg.privacy.noise_multiplier,
    )
    bundle = build_datasets(cfg.dataset, seed=cfg.seed)
    client_loaders, val_loader, test_loader = build_client_loaders(
        bundle=bundle,
        cfg=cfg.dataset,
        seed=cfg.seed,
    )

    num_classes = max(cfg.model.num_classes, len(bundle.class_names))

    def model_builder():
        return build_model(
            backbone=cfg.model.backbone,
            num_classes=num_classes,
            dropout=cfg.model.dropout,
            pretrained=cfg.model.pretrained,
        )

    global_model = prepare_model_for_runtime(model_builder(), cfg=cfg, device=device)
    global_state = clone_state_dict(global_model.state_dict())
    model_size_bytes = state_dict_size_bytes(global_state)

    history: list[dict[str, float | int | str]] = []
    best_checkpoint: dict[str, object] | None = None
    best_checkpoint_key: tuple[float, float, float, int] | None = None
    early_stop_best_score = float("-inf")
    early_stop_best_round = 0
    early_stop_wait = 0
    stopped_early = False
    stop_round: int | None = None

    print(f"正在运行实验：{cfg.output.experiment_name}")
    print(f"运行设备：{device}")
    print(f"类别列表：{bundle.class_names}")
    print(f"客户端数量：{len(client_loaders)}")

    for round_index in range(1, cfg.federated.rounds + 1):
        round_start = time.time()
        selected_clients = sample_clients(
            num_clients=len(client_loaders),
            fraction=cfg.federated.client_fraction,
            seed=cfg.seed,
            round_index=round_index,
        )

        updates = []
        for client_id in selected_clients:
            client_update = run_client_update(
                client_id=client_id,
                global_state=global_state,
                model_builder=model_builder,
                loader=client_loaders[client_id],
                cfg=cfg,
                device=device,
                round_index=round_index,
            )
            updates.append(client_update)

        global_state = aggregate_client_updates(global_state, updates)
        global_model.load_state_dict(global_state)

        decision_threshold = 0.5
        val_loss, val_targets, val_scores = collect_predictions(
            model=global_model,
            loader=val_loader,
            device=device,
            cfg=cfg,
        )
        if cfg.evaluation.tune_threshold and num_classes == 2:
            decision_threshold = tune_binary_threshold_from_predictions(
                targets=val_targets,
                scores=val_scores,
                metric_name=cfg.evaluation.threshold_metric,
                threshold_min=cfg.evaluation.threshold_min,
                threshold_max=cfg.evaluation.threshold_max,
                threshold_steps=cfg.evaluation.threshold_steps,
            )
        val_metrics = evaluate_predictions(
            loss=val_loss,
            targets=val_targets,
            scores=val_scores,
            num_classes=num_classes,
            threshold=decision_threshold if num_classes == 2 else None,
        )
        test_loss, test_targets, test_scores = collect_predictions(
            model=global_model,
            loader=test_loader,
            device=device,
            cfg=cfg,
        )
        test_metrics = evaluate_predictions(
            loss=test_loss,
            targets=test_targets,
            scores=test_scores,
            num_classes=num_classes,
            threshold=decision_threshold if num_classes == 2 else None,
        )
        round_time = time.time() - round_start
        round_communication_bytes = len(selected_clients) * model_size_bytes * 2
        cumulative_communication_bytes = (
            history[-1]["cumulative_communication_bytes"] if history else 0
        ) + round_communication_bytes

        round_mean_epsilon = mean_epsilon(updates)
        previous_cumulative_epsilon = (
            history[-1]["cumulative_epsilon"] if history and pd.notna(history[-1]["cumulative_epsilon"]) else 0.0
        )
        cumulative_epsilon = (
            previous_cumulative_epsilon + round_mean_epsilon
            if pd.notna(round_mean_epsilon)
            else float("nan")
        )

        record = {
            "round": round_index,
            "selected_clients": ",".join(str(cid) for cid in selected_clients),
            "selected_client_count": len(selected_clients),
            "decision_threshold": decision_threshold,
            "train_loss": mean_train_loss(updates),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_confidence_accuracy_top30": val_metrics.get("confidence_accuracy_top30", 0.0),
            "val_confidence_accuracy_top50": val_metrics.get("confidence_accuracy_top50", 0.0),
            "val_confidence_accuracy_ge90": val_metrics.get("confidence_accuracy_ge90", 0.0),
            "val_confidence_coverage_ge90": val_metrics.get("confidence_coverage_ge90", 0.0),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_confidence_accuracy_top30": test_metrics.get("confidence_accuracy_top30", 0.0),
            "test_confidence_accuracy_top50": test_metrics.get("confidence_accuracy_top50", 0.0),
            "test_confidence_accuracy_ge90": test_metrics.get("confidence_accuracy_ge90", 0.0),
            "test_confidence_coverage_ge90": test_metrics.get("confidence_coverage_ge90", 0.0),
            "mean_epsilon": round_mean_epsilon,
            "cumulative_epsilon": cumulative_epsilon,
            "mean_hybrid_alpha": mean_hybrid_alpha(updates),
            "mean_update_norm": mean_update_norm(updates),
            "round_time_sec": round_time,
            "round_communication_bytes": round_communication_bytes,
            "round_communication_mb": round_communication_bytes / (1024 ** 2),
            "cumulative_communication_bytes": cumulative_communication_bytes,
            "cumulative_communication_mb": cumulative_communication_bytes / (1024 ** 2),
        }
        history.append(record)
        current_checkpoint_key = (
            float(record["val_accuracy"]),
            float(record["val_auc"]),
            float(record["val_f1"]),
            -round_index,
        )
        if best_checkpoint_key is None or current_checkpoint_key > best_checkpoint_key:
            best_checkpoint_key = current_checkpoint_key
            best_checkpoint = {
                "round": round_index,
                "decision_threshold": float(decision_threshold),
                "state": clone_state_dict(global_state),
                "val_targets": np.asarray(val_targets, dtype=np.int64),
                "val_scores": val_scores.astype(np.float32, copy=True),
                "test_targets": np.asarray(test_targets, dtype=np.int64),
                "test_scores": test_scores.astype(np.float32, copy=True),
            }
        epsilon_text = (
            f"{record['mean_epsilon']:.4f}"
            if pd.notna(record["mean_epsilon"])
            else "N/A"
        )

        print(
            f"第 {round_index:02d} 轮 | "
            f"验证准确率={record['val_accuracy']:.4f} | "
            f"测试准确率={record['test_accuracy']:.4f} | "
            f"round_epsilon={epsilon_text} | "
            f"耗时={record['round_time_sec']:.2f}s"
        )

        if cfg.federated.early_stopping:
            score = early_stopping_score(record, cfg.federated.early_stopping_metric)
            min_delta = max(float(cfg.federated.early_stopping_min_delta), 0.0)
            if score > early_stop_best_score + min_delta:
                early_stop_best_score = score
                early_stop_best_round = round_index
                early_stop_wait = 0
            else:
                early_stop_wait += 1
            patience = max(int(cfg.federated.early_stopping_patience), 1)
            if early_stop_wait >= patience:
                stopped_early = True
                stop_round = round_index
                print(
                    "Early stopping 触发："
                    f"{cfg.federated.early_stopping_metric} 连续 {patience} 轮未提升，"
                    f"最佳轮次={early_stop_best_round}。"
                )
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    history_df.to_json(output_dir / "history.json", orient="records", indent=2)
    torch.save(global_state, output_dir / "final_model.pt")
    best_model_path = output_dir / "best_model.pt"
    val_predictions_path = output_dir / "val_predictions_best.npz"
    test_predictions_path = output_dir / "test_predictions_best.npz"
    if best_checkpoint is not None:
        torch.save(best_checkpoint["state"], best_model_path)
        np.savez_compressed(
            val_predictions_path,
            targets=best_checkpoint["val_targets"],
            scores=best_checkpoint["val_scores"],
            round=np.asarray([best_checkpoint["round"]], dtype=np.int64),
            decision_threshold=np.asarray([best_checkpoint["decision_threshold"]], dtype=np.float32),
            classes=np.asarray(bundle.class_names),
        )
        np.savez_compressed(
            test_predictions_path,
            targets=best_checkpoint["test_targets"],
            scores=best_checkpoint["test_scores"],
            round=np.asarray([best_checkpoint["round"]], dtype=np.int64),
            decision_threshold=np.asarray([best_checkpoint["decision_threshold"]], dtype=np.float32),
            classes=np.asarray(bundle.class_names),
        )
    save_curves(
        history=history_df,
        output_dir=output_dir,
        privacy_enabled=bool(cfg.privacy.enabled),
        mechanism=cfg.privacy.mechanism if cfg.privacy.enabled else "none",
        privacy_accountant=privacy_accountant,
        privacy_accountant_note=privacy_accountant_note,
    )
    best_val_row = select_best_validation_row(history_df)

    summary = {
        "experiment_name": cfg.output.experiment_name,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "classes": bundle.class_names,
        "rounds": int(len(history_df)),
        "configured_rounds": int(cfg.federated.rounds),
        "early_stopping": bool(cfg.federated.early_stopping),
        "early_stopping_metric": cfg.federated.early_stopping_metric,
        "early_stopping_patience": int(cfg.federated.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.federated.early_stopping_min_delta),
        "stopped_early": bool(stopped_early),
        "stop_round": int(stop_round) if stop_round is not None else None,
        "batch_size": int(cfg.dataset.batch_size),
        "image_size": int(cfg.dataset.image_size),
        "backbone": cfg.model.backbone,
        "learning_rate": float(cfg.federated.lr),
        "optimizer": cfg.federated.optimizer,
        "lr_scheduler": cfg.federated.lr_scheduler,
        "min_lr": float(cfg.federated.min_lr),
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": float(model_size_bytes / (1024 ** 2)),
        "best_val_accuracy": float(history_df["val_accuracy"].max()),
        "best_val_round": int(best_val_row["round"]),
        "best_model_path": str(best_model_path),
        "val_predictions_best_path": str(val_predictions_path),
        "test_predictions_best_path": str(test_predictions_path),
        "test_accuracy_at_best_val": float(best_val_row["test_accuracy"]),
        "test_f1_at_best_val": float(best_val_row["test_f1"]),
        "test_auc_at_best_val": float(best_val_row["test_auc"]),
        "test_balanced_accuracy_at_best_val": float(best_val_row["test_balanced_accuracy"]),
        "test_sensitivity_at_best_val": float(best_val_row["test_sensitivity"]),
        "test_specificity_at_best_val": float(best_val_row["test_specificity"]),
        "test_confidence_accuracy_top30_at_best_val": float(best_val_row["test_confidence_accuracy_top30"]),
        "test_confidence_accuracy_top50_at_best_val": float(best_val_row["test_confidence_accuracy_top50"]),
        "test_confidence_accuracy_ge90_at_best_val": float(best_val_row["test_confidence_accuracy_ge90"]),
        "test_confidence_coverage_ge90_at_best_val": float(best_val_row["test_confidence_coverage_ge90"]),
        "best_test_accuracy": float(best_val_row["test_accuracy"]),
        "max_test_accuracy_observed": float(history_df["test_accuracy"].max()),
        "test_selection_rule": "validation_best_accuracy_then_auc_then_f1",
        "threshold_tuning": bool(cfg.evaluation.tune_threshold),
        "threshold_metric": cfg.evaluation.threshold_metric,
        "decision_threshold_at_best_val": float(best_val_row["decision_threshold"]),
        "final_test_auc": float(history_df["test_auc"].iloc[-1]),
        "final_test_f1": float(history_df["test_f1"].iloc[-1]),
        "final_test_balanced_accuracy": float(history_df["test_balanced_accuracy"].iloc[-1]),
        "final_test_sensitivity": float(history_df["test_sensitivity"].iloc[-1]),
        "final_test_specificity": float(history_df["test_specificity"].iloc[-1]),
        "privacy_enabled": bool(cfg.privacy.enabled),
        "mechanism": cfg.privacy.mechanism if cfg.privacy.enabled else "none",
        "privacy_accountant": privacy_accountant,
        "privacy_accountant_note": privacy_accountant_note,
        "adaptive_hybrid": bool(cfg.privacy.adaptive_hybrid),
        "hybrid_gradient_adaptive": bool(cfg.privacy.hybrid_gradient_adaptive),
        "hybrid_gradient_sensitivity_scale": float(cfg.privacy.hybrid_gradient_sensitivity_scale),
        "final_mean_hybrid_alpha": last_finite_value(history_df["mean_hybrid_alpha"]),
        "final_mean_epsilon": last_finite_value(history_df["mean_epsilon"]),
        "final_cumulative_epsilon": last_finite_value(cumulative_epsilon_from_history(history_df)),
        "avg_round_time_sec": float(history_df["round_time_sec"].mean()),
        "total_communication_bytes": int(history_df["round_communication_bytes"].sum()),
        "total_communication_mb": float(history_df["round_communication_mb"].sum()),
        "avg_round_communication_mb": float(history_df["round_communication_mb"].mean()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return output_dir
