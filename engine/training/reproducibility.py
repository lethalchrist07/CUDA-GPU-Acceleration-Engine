"""
Phase 10 Stage 2 Milestone 5: Reproducibility and Comparative Fairness Engine
Provides deterministic, bitwise-reproducible seeding for DataLoaders, stochastic
augmentations, and model parameter initializations across execution backends:
- CPU_NATIVE
- GPU_NATIVE
- GPU_CUSTOM
"""
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_epoch_seed(seed: int, epoch: int = 0) -> None:
    """
    Set deterministic pseudorandom seeds across Python, NumPy, and PyTorch (CPU + CUDA)
    for a specific training epoch.
    """
    epoch_seed = (seed * 10007 + epoch) & 0x7FFFFFFF
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)


def reset_dataloader_seed(loader: DataLoader, seed: int, epoch: int = 0) -> None:
    """
    Reset DataLoader generator to ensure identical batch ordering and stochastic
    augmentation decisions on each epoch, preventing generator advancement divergence
    between sequential backend experiments.
    """
    epoch_seed = (seed * 10007 + epoch) & 0x7FFFFFFF
    if hasattr(loader, "generator") and loader.generator is not None:
        loader.generator.manual_seed(epoch_seed)
    set_epoch_seed(seed, epoch)


def reproducible_worker_init_fn(worker_id: int) -> None:
    """
    Seed function for multi-process DataLoader workers to guarantee reproducibility.
    """
    worker_seed = (torch.initial_seed() + worker_id) & 0x7FFFFFFF
    random.seed(worker_seed)
    np.random.seed(worker_seed)
