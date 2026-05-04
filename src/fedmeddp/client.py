from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .config import ExperimentConfig
from .metrics import compute_classification_metrics
from .privacy import approximate_epsilon, compute_gradient_adaptive_alpha, privatize_update
from .state import StateDict, state_dict_l2_norm, subtract_floating_state_dict


@dataclass
class ClientUpdate:
    client_id: int
    num_examples: int
    train_loss: float
    delta: StateDict
    raw_update_norm: float
    epsilon: float | None
    effective_hybrid_alpha: float | None


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    name = cfg.federated.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.federated.lr,
            weight_decay=cfg.federated.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.federated.lr,
            weight_decay=cfg.federated.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.federated.lr,
            weight_decay=cfg.federated.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.federated.optimizer}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    total_steps: int,
):
    name = cfg.federated.lr_scheduler.lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(total_steps), 1),
            eta_min=max(float(cfg.federated.min_lr), 0.0),
        )
    raise ValueError(f"Unsupported lr_scheduler: {cfg.federated.lr_scheduler}")


def cuda_fast_path_enabled(cfg: ExperimentConfig, device: torch.device) -> bool:
    return device.type == "cuda"


def autocast_context(cfg: ExperimentConfig, device: torch.device):
    if cfg.runtime.amp and cuda_fast_path_enabled(cfg, device):
        return torch.amp.autocast("cuda")
    return nullcontext()


def move_images_to_device(
    images: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    images = images.to(device, non_blocking=cuda_fast_path_enabled(cfg, device))
    if cfg.runtime.channels_last and cuda_fast_path_enabled(cfg, device):
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def move_targets_to_device(
    targets: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    return targets.to(device, non_blocking=cuda_fast_path_enabled(cfg, device))


def tta_batches(images: torch.Tensor, cfg: ExperimentConfig) -> list[torch.Tensor]:
    if not cfg.evaluation.tta:
        return [images]
    return [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
    ]


def prepare_model_for_runtime(
    model: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    model = model.to(device)
    if cfg.runtime.channels_last and cuda_fast_path_enabled(cfg, device):
        model = model.to(memory_format=torch.channels_last)
    return model


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probabilities = torch.softmax(logits, dim=1)
        true_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        focal_weight = (1.0 - true_probabilities) ** self.gamma
        return (focal_weight * ce_loss).mean()


def extract_labels(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_labels = extract_labels(dataset.dataset)
        return [int(parent_labels[index]) for index in dataset.indices]

    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    raise TypeError("Unsupported dataset type for label extraction.")


def build_train_criterion(
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    label_smoothing = max(float(cfg.federated.label_smoothing), 0.0)
    class_weights = None

    if cfg.federated.use_class_weights:
        labels = extract_labels(loader.dataset)
        num_classes = max(cfg.model.num_classes, max(labels) + 1 if labels else 2)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_name = cfg.federated.loss.lower()
    if loss_name == "focal":
        return FocalCrossEntropyLoss(
            gamma=max(float(cfg.federated.focal_gamma), 0.0),
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss: {cfg.federated.loss}")


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    criterion = build_train_criterion(loader=loader, cfg=cfg, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        total_steps=max(int(cfg.federated.local_epochs), 1) * max(len(loader), 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.runtime.amp and cuda_fast_path_enabled(cfg, device))
    model.train()

    running_loss = 0.0
    total_samples = 0

    for _ in range(cfg.federated.local_epochs):
        for images, targets in loader:
            images = move_images_to_device(images, cfg=cfg, device=device)
            targets = move_targets_to_device(targets, cfg=cfg, device=device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scale_after_step = scaler.get_scale()
                if not scaler.is_enabled() or scale_after_step >= scale_before_step:
                    scheduler.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(total_samples, 1)


def resolve_effective_hybrid_alpha(
    cfg: ExperimentConfig,
    train_loss: float,
    round_index: int,
    gradient_alpha: float | None = None,
) -> float:
    base_alpha = float(np.clip(cfg.privacy.hybrid_alpha, 0.0, 1.0))
    if not cfg.privacy.adaptive_hybrid:
        return base_alpha

    total_rounds = max(int(cfg.federated.rounds), 1)
    progress = (round_index - 1) / max(total_rounds - 1, 1)
    global_factor = 1.0 - float(np.clip(cfg.privacy.hybrid_global_decay, 0.0, 1.0)) * progress
    loss_factor = 1.0 / (1.0 + max(float(train_loss), 0.0))
    min_weight = float(np.clip(cfg.privacy.hybrid_min_gaussian_weight, 0.0, 1.0))
    alpha = base_alpha * global_factor * loss_factor
    if gradient_alpha is not None:
        alpha = min(alpha, gradient_alpha)
    return float(np.clip(alpha, min_weight, base_alpha))


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
) -> tuple[float, list[int], np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_scores: list[np.ndarray] = []

    for images, targets in loader:
        images = move_images_to_device(images, cfg=cfg, device=device)
        targets = move_targets_to_device(targets, cfg=cfg, device=device)

        probability_sum = None
        first_logits = None
        augmented_batches = tta_batches(images, cfg)
        for augmented_images in augmented_batches:
            with autocast_context(cfg, device):
                logits = model(augmented_images)
            if first_logits is None:
                first_logits = logits
            probabilities = torch.softmax(logits.float(), dim=1)
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        assert first_logits is not None and probability_sum is not None
        loss = criterion(first_logits, targets)
        probabilities = probability_sum / len(augmented_batches)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_scores.append(probabilities.cpu().numpy())

    if total_samples == 0:
        return 0.0, [], np.empty((0, 0), dtype=np.float32)

    return total_loss / total_samples, all_targets, np.concatenate(all_scores, axis=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    cfg: ExperimentConfig,
    threshold: float | None = None,
) -> dict[str, float]:
    loss, all_targets, stacked_scores = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
    )

    if not all_targets:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    if num_classes == 2 and threshold is not None:
        all_predictions = (stacked_scores[:, 1] >= threshold).astype(int).tolist()
    else:
        all_predictions = np.argmax(stacked_scores, axis=1).tolist()

    metrics = compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_predictions,
        y_score=stacked_scores,
        num_classes=num_classes,
    )
    metrics["loss"] = loss
    return metrics


def evaluate_predictions(
    loss: float,
    targets: list[int],
    scores: np.ndarray,
    num_classes: int,
    threshold: float | None = None,
) -> dict[str, float]:
    if not targets:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    if num_classes == 2 and threshold is not None:
        predictions = (scores[:, 1] >= threshold).astype(int).tolist()
    else:
        predictions = np.argmax(scores, axis=1).tolist()
    metrics = compute_classification_metrics(
        y_true=targets,
        y_pred=predictions,
        y_score=scores,
        num_classes=num_classes,
    )
    metrics["loss"] = loss
    return metrics


@torch.no_grad()
def tune_binary_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    cfg: ExperimentConfig,
) -> float:
    _, targets, scores = collect_predictions(model=model, loader=loader, device=device, cfg=cfg)
    if not targets or scores.shape[1] < 2:
        return 0.5
    return tune_binary_threshold_from_predictions(
        targets=targets,
        scores=scores,
        metric_name=metric_name,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_steps=threshold_steps,
    )


def tune_binary_threshold_from_predictions(
    targets: list[int],
    scores: np.ndarray,
    metric_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> float:
    thresholds = np.linspace(threshold_min, threshold_max, max(int(threshold_steps), 2))
    best_threshold = 0.5
    best_score = -1.0
    metric_key = metric_name.lower()
    for threshold in thresholds:
        predictions = (scores[:, 1] >= threshold).astype(int).tolist()
        metrics = compute_classification_metrics(
            y_true=targets,
            y_pred=predictions,
            y_score=scores,
            num_classes=2,
        )
        score = float(metrics.get(metric_key, metrics["f1"]))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def run_client_update(
    client_id: int,
    global_state: StateDict,
    model_builder: Callable[[], nn.Module],
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    round_index: int,
) -> ClientUpdate:
    model = prepare_model_for_runtime(model_builder(), cfg=cfg, device=device)
    model.load_state_dict(global_state)

    train_loss = train_local_model(model=model, loader=loader, cfg=cfg, device=device)
    local_state = model.state_dict()
    delta = subtract_floating_state_dict(local_state, global_state)
    raw_update_norm = state_dict_l2_norm(delta)

    epsilon: float | None = None
    effective_hybrid_alpha: float | None = None
    if cfg.privacy.enabled:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed + client_id * 100 + round_index)
        hybrid_alpha = cfg.privacy.hybrid_alpha
        if cfg.privacy.mechanism.lower() == "hybrid":
            gradient_alpha = None
            if cfg.privacy.hybrid_gradient_adaptive:
                gradient_alpha, _ = compute_gradient_adaptive_alpha(
                    global_state=global_state,
                    local_state=local_state,
                    base_alpha=cfg.privacy.hybrid_alpha,
                    min_alpha=cfg.privacy.hybrid_min_gaussian_weight,
                    sensitivity_scale=cfg.privacy.hybrid_gradient_sensitivity_scale,
                )
            hybrid_alpha = resolve_effective_hybrid_alpha(
                cfg=cfg,
                train_loss=train_loss,
                round_index=round_index,
                gradient_alpha=gradient_alpha,
            )
            effective_hybrid_alpha = hybrid_alpha
        delta, raw_update_norm = privatize_update(
            delta=delta,
            mechanism=cfg.privacy.mechanism,
            clip_norm=cfg.privacy.clip_norm,
            noise_multiplier=cfg.privacy.noise_multiplier,
            laplace_scale=cfg.privacy.laplace_scale,
            noise_scale_mode=cfg.privacy.noise_scale_mode,
            hybrid_alpha=hybrid_alpha,
            generator=generator,
        )
        mechanism_lower = cfg.privacy.mechanism.lower()
        if mechanism_lower == "gaussian":
            sample_rate = min(cfg.dataset.batch_size / max(len(loader.dataset), 1), 1.0)
            steps = cfg.federated.local_epochs * len(loader)
            epsilon = approximate_epsilon(
                steps=steps,
                sample_rate=sample_rate,
                noise_multiplier=cfg.privacy.noise_multiplier,
                delta=cfg.privacy.delta,
            )
        elif mechanism_lower == "hybrid" and cfg.privacy.noise_multiplier > 0:
            sample_rate = min(cfg.dataset.batch_size / max(len(loader.dataset), 1), 1.0)
            steps = cfg.federated.local_epochs * len(loader)
            epsilon = approximate_epsilon(
                steps=steps,
                sample_rate=sample_rate,
                noise_multiplier=cfg.privacy.noise_multiplier,
                delta=cfg.privacy.delta,
            )

    return ClientUpdate(
        client_id=client_id,
        num_examples=len(loader.dataset),
        train_loss=train_loss,
        delta=delta,
        raw_update_norm=raw_update_norm,
        epsilon=epsilon,
        effective_hybrid_alpha=effective_hybrid_alpha,
    )
