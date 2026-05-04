from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import DatasetConfig


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_labels: list[int]
    class_names: list[str]


def build_transforms(cfg: DatasetConfig, train: bool) -> transforms.Compose:
    image_size = int(cfg.image_size)
    strength = cfg.augmentation_strength.lower()
    if train:
        if strength in {"strong", "highacc"}:
            ops = [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(max(min(float(cfg.train_crop_scale), 0.98), 0.5), 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.025),
            ]
            if hasattr(transforms, "RandAugment"):
                ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
        else:
            ops = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
            ]
    else:
        ops = [transforms.Resize((image_size, image_size))]
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if train and cfg.random_erasing > 0:
        ops.append(
            transforms.RandomErasing(
                p=max(min(float(cfg.random_erasing), 0.5), 0.0),
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    return transforms.Compose(ops)


def _split_indices(
    dataset_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)

    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size - test_size

    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size : train_size + val_size].tolist()
    test_idx = indices[train_size + val_size :].tolist()
    return train_idx, val_idx, test_idx


def build_datasets(cfg: DatasetConfig, seed: int) -> DatasetBundle:
    root = Path(cfg.root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    train_transform = build_transforms(cfg, train=True)
    eval_transform = build_transforms(cfg, train=False)

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        return DatasetBundle(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_labels=list(train_dataset.targets),
            class_names=list(train_dataset.classes),
        )

    full_train = datasets.ImageFolder(root, transform=train_transform)
    full_eval = datasets.ImageFolder(root, transform=eval_transform)
    train_idx, val_idx, test_idx = _split_indices(
        dataset_size=len(full_train),
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=seed,
    )

    return DatasetBundle(
        train_dataset=Subset(full_train, train_idx),
        val_dataset=Subset(full_eval, val_idx),
        test_dataset=Subset(full_eval, test_idx),
        train_labels=[full_train.targets[index] for index in train_idx],
        class_names=list(full_train.classes),
    )


def partition_indices(
    labels: list[int],
    num_clients: int,
    alpha: float,
    min_samples_per_client: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels_array = np.array(labels)
    indices = np.arange(len(labels_array))

    if alpha <= 0:
        rng.shuffle(indices)
        return [chunk.tolist() for chunk in np.array_split(indices, num_clients)]

    unique_classes = np.unique(labels_array)

    for _ in range(200):
        client_indices = [[] for _ in range(num_clients)]
        for class_id in unique_classes:
            class_indices = indices[labels_array == class_id]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            chunks = np.split(class_indices, split_points)
            for client_id, chunk in enumerate(chunks):
                client_indices[client_id].extend(chunk.tolist())

        if min(len(chunk) for chunk in client_indices) >= min_samples_per_client:
            return [sorted(chunk) for chunk in client_indices]

    raise RuntimeError("Failed to generate client partitions satisfying the minimum sample constraint.")


def build_client_loaders(
    bundle: DatasetBundle,
    cfg: DatasetConfig,
    seed: int,
) -> tuple[list[DataLoader], DataLoader, DataLoader]:
    client_partitions = partition_indices(
        labels=bundle.train_labels,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        min_samples_per_client=cfg.min_samples_per_client,
        seed=seed,
    )

    client_loaders = []
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.persistent_workers
        loader_kwargs["prefetch_factor"] = max(int(cfg.prefetch_factor), 2)

    for client_id, indices in enumerate(client_partitions):
        subset = Subset(bundle.train_dataset, indices)
        generator = torch.Generator().manual_seed(seed + client_id)
        sampler = None
        shuffle = True
        if cfg.weighted_sampling:
            labels = [bundle.train_labels[index] for index in indices]
            counts = np.bincount(labels, minlength=max(labels) + 1 if labels else 2).astype(np.float32)
            counts[counts == 0] = 1.0
            weights = [float(len(labels) / counts[label]) for label in labels]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        client_loaders.append(
            DataLoader(
                subset,
                batch_size=cfg.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                generator=generator,
                **loader_kwargs,
            )
        )

    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 10_000),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 20_000),
        **loader_kwargs,
    )

    return client_loaders, val_loader, test_loader
