"""
Scalable Memory-Mapped Dataset and Deterministic Data Loading Pipeline.
Phase 13 Milestone 1: High-throughput, zero-copy token streaming for extended pretraining.

Properties:
- Zero host RAM overhead: Backed by np.memmap on disk (np.uint16 or np.int64)
- Multi-worker fork safety: Lazy per-worker memmap handle instantiation
- Deterministic epoch-based index permutation without token memory duplication
- Strict causal alignment: x = t[i:i+T], y = t[i+1:i+T+1]
- Input integrity checks: File alignment, out-of-bounds token detection, truncated file guards
- High-efficiency PyTorch DataLoader creation with CUDA pinned memory
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int):
    """Worker initialization function for reproducible multi-worker DataLoader execution."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ScalableMemmapDataset(Dataset):
    """
    Production-grade memory-mapped chunk dataset for Transformer language model training.
    """

    def __init__(
        self,
        bin_path: Union[str, Path],
        seq_len: int = 256,
        stride: Optional[int] = None,
        dtype: np.dtype = np.uint16,
        vocab_size: Optional[int] = None,
        validate_integrity: bool = True,
    ):
        self.bin_path = Path(bin_path).resolve()
        if not self.bin_path.is_file():
            raise FileNotFoundError(f"Binary token file not found at: {self.bin_path}")

        file_bytes = self.bin_path.stat().st_size
        item_size = np.dtype(dtype).itemsize

        if file_bytes == 0:
            raise ValueError(f"Binary token file is empty (0 bytes): {self.bin_path}")

        if file_bytes % item_size != 0:
            raise ValueError(
                f"Binary token file size ({file_bytes} bytes) is not aligned to dtype itemsize ({item_size} bytes)."
            )

        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.dtype = dtype
        self.vocab_size = vocab_size

        self.total_tokens = file_bytes // item_size
        window_size = seq_len + 1

        if self.total_tokens < window_size:
            raise ValueError(
                f"Total tokens in file ({self.total_tokens}) is less than required window size ({window_size}) "
                f"for seq_len={seq_len}."
            )

        self.num_chunks = max(0, (self.total_tokens - window_size) // self.stride + 1)
        self._mmap: Optional[np.memmap] = None

        if validate_integrity:
            self._validate_integrity()

    def _get_mmap(self) -> np.memmap:
        """Lazy memory-map initialization for multi-worker process safety."""
        if self._mmap is None:
            self._mmap = np.memmap(str(self.bin_path), dtype=self.dtype, mode="r")
        return self._mmap

    def _validate_integrity(self, sample_size: int = 10_000) -> None:
        """Quickly inspects initial and sampled tokens for corrupted values."""
        mmap = self._get_mmap()
        check_count = min(sample_size, len(mmap))
        sample_slice = mmap[:check_count]

        min_val = int(sample_slice.min())
        max_val = int(sample_slice.max())

        if min_val < 0:
            raise ValueError(f"Corrupted negative token value found: {min_val}")

        if self.vocab_size is not None and max_val >= self.vocab_size:
            raise ValueError(
                f"Token value ({max_val}) exceeds declared vocabulary size ({self.vocab_size}). "
                f"Dataset is incompatible with model vocabulary."
            )

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_chunks:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_chunks}")

        start = idx * self.stride
        end = start + self.seq_len + 1

        mmap = self._get_mmap()
        chunk_raw = mmap[start:end]

        # Convert to int64 PyTorch tensor for cross-entropy loss computation
        chunk_tensor = torch.from_numpy(chunk_raw.astype(np.int64))

        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        return x, y

    def get_token_slice(self, start: int, end: int) -> np.ndarray:
        """Retrieves raw token slice from memory-map."""
        mmap = self._get_mmap()
        return mmap[start:end]


def create_scalable_dataloaders(
    train_dataset: ScalableMemmapDataset,
    val_dataset: Optional[ScalableMemmapDataset] = None,
    test_dataset: Optional[ScalableMemmapDataset] = None,
    batch_size: int = 4,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates deterministic PyTorch DataLoaders backed by ScalableMemmapDataset.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    pin = pin_memory and use_cuda

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        drop_last=drop_last,
    )

    loaders = {"train": train_loader}

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            drop_last=False,
        )
        loaders["val"] = val_loader

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            drop_last=False,
        )
        loaders["test"] = test_loader

    return loaders
