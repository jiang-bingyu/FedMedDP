from __future__ import annotations

import math
import random

from .client import ClientUpdate
from .state import StateDict, add_floating_delta, scale_state_dict, sum_state_dicts


def sample_clients(num_clients: int, fraction: float, seed: int, round_index: int) -> list[int]:
    select_count = max(1, math.ceil(num_clients * fraction))
    rng = random.Random(seed + round_index)
    return sorted(rng.sample(list(range(num_clients)), select_count))


def aggregate_client_updates(global_state: StateDict, updates: list[ClientUpdate]) -> StateDict:
    if not updates:
        return global_state

    total_examples = sum(update.num_examples for update in updates)
    weighted_deltas = []
    for update in updates:
        weight = update.num_examples / total_examples
        weighted_deltas.append(scale_state_dict(update.delta, weight))

    aggregated_delta = sum_state_dicts(weighted_deltas)
    return add_floating_delta(global_state, aggregated_delta)


def mean_update_norm(updates: list[ClientUpdate]) -> float:
    if not updates:
        return 0.0
    return float(sum(item.raw_update_norm for item in updates) / len(updates))


def mean_epsilon(updates: list[ClientUpdate]) -> float:
    if not updates:
        return float("nan")
    values = [item.epsilon for item in updates if item.epsilon is not None]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def mean_hybrid_alpha(updates: list[ClientUpdate]) -> float:
    if not updates:
        return float("nan")
    values = [item.effective_hybrid_alpha for item in updates if item.effective_hybrid_alpha is not None]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def mean_train_loss(updates: list[ClientUpdate]) -> float:
    if not updates:
        return 0.0
    return float(sum(item.train_loss for item in updates) / len(updates))
