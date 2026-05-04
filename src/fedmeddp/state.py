from __future__ import annotations

from collections import OrderedDict

import torch


StateDict = OrderedDict[str, torch.Tensor]


def clone_state_dict(state_dict: StateDict) -> StateDict:
    return OrderedDict((name, tensor.detach().clone().cpu()) for name, tensor in state_dict.items())


def subtract_floating_state_dict(new_state: StateDict, old_state: StateDict) -> StateDict:
    delta: StateDict = OrderedDict()
    for name, tensor in new_state.items():
        if torch.is_floating_point(tensor):
            delta[name] = tensor.detach().cpu() - old_state[name].detach().cpu()
    return delta


def add_floating_delta(base_state: StateDict, delta: StateDict) -> StateDict:
    updated = clone_state_dict(base_state)
    for name, tensor in delta.items():
        updated[name] = updated[name] + tensor
    return updated


def scale_state_dict(state_dict: StateDict, scale: float) -> StateDict:
    return OrderedDict((name, tensor * scale) for name, tensor in state_dict.items())


def sum_state_dicts(state_dicts: list[StateDict]) -> StateDict:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    result: StateDict = OrderedDict(
        (name, torch.zeros_like(tensor)) for name, tensor in state_dicts[0].items()
    )
    for current in state_dicts:
        for name, tensor in current.items():
            result[name] += tensor
    return result


def state_dict_l2_norm(state_dict: StateDict) -> float:
    total = 0.0
    for tensor in state_dict.values():
        total += float(torch.sum(tensor.float() ** 2).item())
    return total ** 0.5
