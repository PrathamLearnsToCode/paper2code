"""
Denoising Diffusion Probabilistic Models — Data Loading

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

§4 — "We set T = 1000... on CIFAR10 (32×32)... We used random horizontal
flips during training; we tried training both with and without dropout..."

Data preprocessing:
  - Normalize images to [-1, 1] range (not [0,1])
    [FROM_OFFICIAL_CODE] — Paper does not state normalization range explicitly,
    but the official code normalizes to [-1,1] which is consistent with
    the Gaussian output distribution N(x_0; μ_θ, σ²I).
  - Random horizontal flips (§4)
  - No other augmentation mentioned
"""

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_cifar10_transforms(image_size: int = 32) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms for CIFAR-10.

    §4 — "random horizontal flips during training"
    [FROM_OFFICIAL_CODE] — normalize to [-1, 1]

    Args:
        image_size: Target image size (CIFAR-10 is natively 32×32)

    Returns:
        (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # §4 — explicit mention
        transforms.ToTensor(),                    # [0, 255] -> [0, 1]
        transforms.Normalize(                     # [0, 1] -> [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test dataloaders.

    §4 — CIFAR-10 unconditional generation at 32×32
    §4 — "batch size 128" [FROM_OFFICIAL_CODE — paper does not state batch size]

    Args:
        data_dir: Root directory for dataset download/cache
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image resolution (32 for CIFAR-10)

    Returns:
        (train_loader, test_loader)
    """
    train_transform, test_transform = get_cifar10_transforms(image_size)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # [ASSUMPTION] drop incomplete last batch for stable training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
