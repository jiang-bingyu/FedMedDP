from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .client import autocast_context, prepare_model_for_runtime, tta_batches
from .config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    OutputConfig,
    PrivacyConfig,
    RuntimeConfig,
)
from .data import build_transforms
from .model import build_model


CLASS_NAMES = ("benign", "malignant")
CLASS_LABELS_ZH = {
    "benign": "倾向良性",
    "malignant": "疑似恶性",
}
DEMO_WARNING = "仅用于毕业设计答辩演示，不能替代医生诊断。"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: str
    backbone: str
    image_size: int
    dropout: float
    weight: float


@dataclass(frozen=True)
class ModelPreset:
    mode: str
    display_name: str
    threshold: float
    specs: tuple[ModelSpec, ...]


@dataclass
class LoadedMember:
    spec: ModelSpec
    cfg: ExperimentConfig
    transform: Any
    model: torch.nn.Module


ENSEMBLE_PRESET = ModelPreset(
    mode="ensemble",
    display_name="weighted_ensemble",
    threshold=0.74,
    specs=(
        ModelSpec(
            name="ham10000_accuracy90_seed2026",
            model_path="outputs/ham10000_accuracy90_seed2026/best_model.pt",
            backbone="convnext_small",
            image_size=448,
            dropout=0.10,
            weight=0.25,
        ),
        ModelSpec(
            name="ham10000_accuracy90_seed2028",
            model_path="outputs/ham10000_accuracy90_seed2028/best_model.pt",
            backbone="convnext_small",
            image_size=448,
            dropout=0.10,
            weight=0.10,
        ),
        ModelSpec(
            name="ham10000_accuracy90_seed2030",
            model_path="outputs/ham10000_accuracy90_seed2030/best_model.pt",
            backbone="convnext_small",
            image_size=448,
            dropout=0.10,
            weight=0.30,
        ),
        ModelSpec(
            name="ham10000_accuracy90_efficientnet_b4",
            model_path="outputs/ham10000_accuracy90_efficientnet_b4/best_model.pt",
            backbone="efficientnet_b4",
            image_size=380,
            dropout=0.18,
            weight=0.30,
        ),
        ModelSpec(
            name="ham10000_accuracy90_seed2027",
            model_path="outputs/ham10000_accuracy90_seed2027/best_model.pt",
            backbone="convnext_small",
            image_size=448,
            dropout=0.10,
            weight=0.05,
        ),
    ),
)

SINGLE_PRESET = ModelPreset(
    mode="single",
    display_name="single_seed2030",
    threshold=0.875,
    specs=(
        ModelSpec(
            name="ham10000_accuracy90_seed2030",
            model_path="outputs/ham10000_accuracy90_seed2030/best_model.pt",
            backbone="convnext_small",
            image_size=448,
            dropout=0.10,
            weight=1.0,
        ),
    ),
)

PRESETS = {
    ENSEMBLE_PRESET.mode: ENSEMBLE_PRESET,
    SINGLE_PRESET.mode: SINGLE_PRESET,
}


class DemoSkinPredictor:
    def __init__(
        self,
        mode: str = "ensemble",
        device: str = "auto",
        root: str | Path | None = None,
    ) -> None:
        if mode not in PRESETS:
            raise ValueError(f"Unsupported inference mode: {mode}")
        self.preset = PRESETS[mode]
        self.requested_device = device
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.device = self._resolve_device(device)
        self._members: list[LoadedMember] = []
        self._load_lock = Lock()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized = device.lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(normalized)

    @property
    def loaded(self) -> bool:
        return bool(self._members)

    def missing_model_paths(self) -> list[str]:
        missing = []
        for spec in self.preset.specs:
            path = self.root / spec.model_path
            if not path.exists():
                missing.append(str(path))
        return missing

    def status(self) -> dict[str, Any]:
        missing = self.missing_model_paths()
        return {
            "mode": self.preset.mode,
            "model_mode": self.preset.display_name,
            "device": str(self.device),
            "loaded": self.loaded,
            "available": not missing,
            "missing_models": missing,
            "threshold": self.preset.threshold,
            "members": [spec.name for spec in self.preset.specs],
        }

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return

            missing = self.missing_model_paths()
            if missing:
                raise FileNotFoundError("缺少模型权重：" + "；".join(missing))

            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            members = []
            for spec in self.preset.specs:
                cfg = self._build_config(spec)
                model = build_model(
                    backbone=spec.backbone,
                    num_classes=len(CLASS_NAMES),
                    dropout=spec.dropout,
                    pretrained=False,
                )
                model = prepare_model_for_runtime(model, cfg=cfg, device=self.device)
                state = torch.load(self.root / spec.model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state)
                model.eval()
                members.append(
                    LoadedMember(
                        spec=spec,
                        cfg=cfg,
                        transform=build_transforms(cfg.dataset, train=False),
                        model=model,
                    )
                )
            self._members = members

    def predict_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        self.load()
        image = self._open_image(image_bytes)
        probabilities = self._predict_image(image)
        benign_probability = float(probabilities[0])
        malignant_probability = float(probabilities[1])
        prediction = "malignant" if malignant_probability >= self.preset.threshold else "benign"
        confidence = malignant_probability if prediction == "malignant" else benign_probability

        return {
            "prediction": prediction,
            "label_zh": CLASS_LABELS_ZH[prediction],
            "confidence": confidence,
            "probabilities": {
                "benign": benign_probability,
                "malignant": malignant_probability,
            },
            "threshold": self.preset.threshold,
            "model_mode": self.preset.display_name,
            "members": [member.spec.name for member in self._members],
            "warning": DEMO_WARNING,
        }

    @staticmethod
    def _open_image(image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("无法读取图片，请上传 JPG、PNG 或 WEBP 图片。") from exc

    def _predict_image(self, image: Image.Image) -> np.ndarray:
        weighted_probability = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        total_weight = 0.0

        with torch.inference_mode():
            for member in self._members:
                tensor = member.transform(image).unsqueeze(0).to(self.device)
                if member.cfg.runtime.channels_last and self.device.type == "cuda":
                    tensor = tensor.contiguous(memory_format=torch.channels_last)

                augmented_batches = tta_batches(tensor, member.cfg)
                probability_sum = None
                for augmented in augmented_batches:
                    with autocast_context(member.cfg, self.device):
                        logits = member.model(augmented)
                    probabilities = torch.softmax(logits.float(), dim=1)
                    probability_sum = probabilities if probability_sum is None else probability_sum + probabilities

                assert probability_sum is not None
                member_probability = (probability_sum / len(augmented_batches)).cpu().numpy()[0]
                weighted_probability += float(member.spec.weight) * member_probability
                total_weight += float(member.spec.weight)

        if total_weight <= 0:
            raise RuntimeError("模型权重配置无效。")

        normalized = weighted_probability / total_weight
        return normalized.astype(np.float64)

    def _build_config(self, spec: ModelSpec) -> ExperimentConfig:
        return ExperimentConfig(
            seed=2026,
            dataset=DatasetConfig(
                root=str(self.root / "data" / "ham10000_binary"),
                image_size=spec.image_size,
                batch_size=1,
                num_workers=0,
                num_clients=1,
                dirichlet_alpha=1.0,
                val_ratio=0.1,
                test_ratio=0.1,
                min_samples_per_client=1,
                weighted_sampling=False,
                pin_memory=self.device.type == "cuda",
                persistent_workers=False,
                prefetch_factor=2,
                augmentation_strength="accuracy90",
                train_crop_scale=0.86,
                random_erasing=0.0,
            ),
            model=ModelConfig(
                backbone=spec.backbone,
                num_classes=len(CLASS_NAMES),
                dropout=spec.dropout,
                pretrained=False,
            ),
            federated=FederatedConfig(),
            privacy=PrivacyConfig(enabled=False, mechanism="none", noise_multiplier=0.0),
            evaluation=EvaluationConfig(
                tune_threshold=True,
                threshold_metric="accuracy",
                threshold_min=0.05,
                threshold_max=0.95,
                threshold_steps=181,
                tta=True,
            ),
            output=OutputConfig(dir=str(self.root / "outputs"), experiment_name=spec.name),
            runtime=RuntimeConfig(
                device=str(self.device),
                amp=self.device.type == "cuda",
                channels_last=True,
                cudnn_benchmark=True,
            ),
        )
