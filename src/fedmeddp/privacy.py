from __future__ import annotations

from collections import OrderedDict
import math

import torch

from .state import StateDict, state_dict_l2_norm


def clip_state_dict(delta: StateDict, clip_norm: float) -> tuple[StateDict, float]:
    current_norm = state_dict_l2_norm(delta)
    if current_norm <= clip_norm or current_norm == 0.0:
        return delta, current_norm

    scale = clip_norm / (current_norm + 1e-12)
    clipped = OrderedDict((name, tensor * scale) for name, tensor in delta.items())
    return clipped, current_norm


def compute_gradient_adaptive_alpha(
    global_state: StateDict,
    local_state: StateDict,
    base_alpha: float,
    min_alpha: float = 0.1,
    sensitivity_scale: float = 0.02,
) -> tuple[float, float]:
    """Adjust Gaussian weight by update sensitivity.

    The hybrid mechanism uses alpha as the Gaussian-noise weight. A larger local
    update norm indicates a more sensitive client update, so the Gaussian weight
    is reduced and the Laplace component is increased.
    """
    delta: StateDict = OrderedDict()
    global_floating: StateDict = OrderedDict()
    for name, tensor in global_state.items():
        if not tensor.is_floating_point() or name not in local_state:
            continue
        local_tensor = local_state[name]
        if not local_tensor.is_floating_point():
            continue
        global_tensor = tensor.detach().cpu()
        delta[name] = local_tensor.detach().cpu() - global_tensor
        global_floating[name] = global_tensor
    update_norm = state_dict_l2_norm(delta)
    global_norm = state_dict_l2_norm(global_floating)
    relative_norm = update_norm / (global_norm + 1e-12)
    scale = max(float(sensitivity_scale), 1e-8)
    sensitivity = relative_norm / (relative_norm + scale)
    adjusted_alpha = float(base_alpha) * (1.0 - 0.5 * sensitivity)
    return float(max(min_alpha, min(adjusted_alpha, float(base_alpha)))), float(sensitivity)


def _gaussian_noise_like(
    reference: torch.Tensor,
    std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    ) * std


def _laplace_noise_like(
    reference: torch.Tensor,
    scale: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    uniform = torch.rand(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    ) - 0.5
    eps = torch.finfo(reference.dtype).eps
    return -scale * torch.sign(uniform) * torch.log1p(-2 * torch.abs(uniform).clamp(max=0.5 - eps))


def privatize_update(
    delta: StateDict,
    mechanism: str,
    clip_norm: float,
    noise_multiplier: float,
    laplace_scale: float,
    noise_scale_mode: str,
    hybrid_alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[StateDict, float]:
    clipped, raw_norm = clip_state_dict(delta, clip_norm)
    privatized: StateDict = OrderedDict()

    scale_denominator = 1.0
    if noise_scale_mode.lower() == "vector":
        total_parameters = sum(tensor.numel() for tensor in clipped.values() if tensor.is_floating_point())
        scale_denominator = math.sqrt(max(total_parameters, 1))
    elif noise_scale_mode.lower() != "per_parameter":
        raise ValueError(f"Unsupported noise_scale_mode: {noise_scale_mode}")

    gaussian_std = clip_norm * noise_multiplier / scale_denominator
    laplace_noise_scale = clip_norm * laplace_scale / scale_denominator

    for name, tensor in clipped.items():
        mechanism_lower = mechanism.lower()
        if mechanism_lower == "gaussian":
            noise = _gaussian_noise_like(tensor, gaussian_std, generator)
        elif mechanism_lower == "laplace":
            noise = _laplace_noise_like(tensor, laplace_noise_scale, generator)
        elif mechanism_lower == "hybrid":
            gaussian_noise = _gaussian_noise_like(tensor, gaussian_std, generator)
            laplace_noise = _laplace_noise_like(tensor, laplace_noise_scale, generator)
            noise = hybrid_alpha * gaussian_noise + (1.0 - hybrid_alpha) * laplace_noise
        else:
            raise ValueError(f"Unsupported privacy mechanism: {mechanism}")
        privatized[name] = tensor + noise

    return privatized, raw_norm


def approximate_epsilon(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
) -> float:
    if steps <= 0 or sample_rate <= 0 or noise_multiplier <= 0:
        return 0.0

    epsilon = (
        sample_rate
        * math.sqrt(2.0 * steps * math.log(1.0 / max(delta, 1e-12)))
        / noise_multiplier
    )
    return float(epsilon)


def resolve_privacy_accountant(
    enabled: bool,
    mechanism: str,
    noise_multiplier: float,
) -> tuple[str, str]:
    if not enabled:
        return "not_applicable", "无隐私机制，不计算 epsilon。"

    mechanism_lower = mechanism.lower()
    if mechanism_lower == "gaussian":
        return "gaussian_approx", "当前展示的是 Gaussian 机制的近似 epsilon。"
    if mechanism_lower == "laplace":
        return "not_available", "当前版本未实现 Laplace 机制的严格 epsilon 会计。"
    if mechanism_lower == "hybrid":
        if noise_multiplier > 0:
            return "hybrid_gaussian_component_approx", "当前展示的是 Hybrid 中 Gaussian 部分的近似 epsilon。"
        return "not_available", "当前版本未实现 Hybrid 机制的完整 epsilon 会计。"
    return "unknown", "未知隐私机制，无法解释 epsilon。"
