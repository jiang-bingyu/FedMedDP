from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    root: str
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 4
    num_clients: int = 5
    dirichlet_alpha: float = 0.5
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_samples_per_client: int = 8
    weighted_sampling: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    augmentation_strength: str = "standard"
    train_crop_scale: float = 0.85
    random_erasing: float = 0.0


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v2"
    num_classes: int = 2
    dropout: float = 0.2
    pretrained: bool = True


@dataclass
class FederatedConfig:
    rounds: int = 10
    client_fraction: float = 0.6
    local_epochs: int = 1
    optimizer: str = "adam"
    lr: float = 5e-4
    weight_decay: float = 1e-4
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    loss: str = "focal"
    focal_gamma: float = 2.0
    lr_scheduler: str = "none"
    min_lr: float = 1e-6


@dataclass
class EvaluationConfig:
    tune_threshold: bool = True
    threshold_metric: str = "f1"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 91
    tta: bool = False


@dataclass
class PrivacyConfig:
    enabled: bool = True
    mechanism: str = "gaussian"
    clip_norm: float = 1.0
    noise_multiplier: float = 0.1
    laplace_scale: float = 0.02
    noise_scale_mode: str = "vector"
    hybrid_alpha: float = 0.7
    adaptive_hybrid: bool = False
    hybrid_global_decay: float = 0.8
    hybrid_min_gaussian_weight: float = 0.1
    hybrid_gradient_adaptive: bool = True
    hybrid_gradient_sensitivity_scale: float = 0.02
    delta: float = 1e-5


@dataclass
class OutputConfig:
    dir: str = "outputs"
    experiment_name: str = "default_experiment"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    amp: bool = True
    channels_last: bool = True
    cudnn_benchmark: bool = True


@dataclass
class ExperimentConfig:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    federated: FederatedConfig
    privacy: PrivacyConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    runtime: RuntimeConfig

    @property
    def output_dir(self) -> Path:
        return Path(self.output.dir) / self.output.experiment_name


def _build_section(section_cls: type[Any], data: dict[str, Any] | None) -> Any:
    payload = data or {}
    return section_cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return ExperimentConfig(
        seed=raw["seed"],
        dataset=_build_section(DatasetConfig, raw.get("dataset")),
        model=_build_section(ModelConfig, raw.get("model")),
        federated=_build_section(FederatedConfig, raw.get("federated")),
        privacy=_build_section(PrivacyConfig, raw.get("privacy")),
        evaluation=_build_section(EvaluationConfig, raw.get("evaluation")),
        output=_build_section(OutputConfig, raw.get("output")),
        runtime=_build_section(RuntimeConfig, raw.get("runtime")),
    )
