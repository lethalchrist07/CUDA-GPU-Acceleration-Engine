"""
Dataset and DataLoader Pipeline for Autoregressive Transformer Training.

Provides:
- TextChunkDataset: Maps token sequences into (x, y) pairs where y is shifted by 1.
- Deterministic train/validation dataset splitting.
- Corrupted sample validation and out-of-range token detection.
- Reproducible DataLoader creation with CUDA pinned memory support.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from engine.transformer.tokenizer import BaseTokenizer, CharTokenizer, ByteTokenizer


class TextChunkDataset(Dataset):
    """
    Dataset that yields input sequences x and shifted target sequences y.
    
    Given token sequence [t_0, t_1, t_2, ..., t_T],
    x = [t_0, t_1, ..., t_{T-1}]
    y = [t_1, t_2, ..., t_T]
    
    Both x and y have shape (seq_len,).
    """

    def __init__(
        self,
        tokens: Union[List[int], torch.Tensor],
        seq_len: int,
        stride: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        """
        Args:
            tokens: Sequence of token IDs
            seq_len: Context length for training (T)
            stride: Step size between consecutive chunks (defaults to seq_len for non-overlapping)
            vocab_size: Optional upper bound on token values for corruption validation
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        if isinstance(tokens, list):
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        elif isinstance(tokens, torch.Tensor):
            tokens_tensor = tokens.to(dtype=torch.long)
        else:
            raise TypeError(f"Expected tokens to be list or torch.Tensor, got {type(tokens)}")

        if tokens_tensor.numel() <= seq_len:
            raise ValueError(
                f"Total tokens ({tokens_tensor.numel()}) must be strictly greater than seq_len ({seq_len}) "
                f"to form at least one (x, y) pair of length {seq_len}."
            )

        # Integrity check: Validate token range
        if vocab_size is not None:
            min_val = tokens_tensor.min().item()
            max_val = tokens_tensor.max().item()
            if min_val < 0 or max_val >= vocab_size:
                raise ValueError(
                    f"Token values out of range [0, {vocab_size - 1}]: min={min_val}, max={max_val}"
                )

        self.tokens = tokens_tensor
        self.seq_len = seq_len
        self.stride = stride if stride is not None and stride > 0 else seq_len

        # Calculate number of chunks
        # Need window of size seq_len + 1
        window_size = seq_len + 1
        num_tokens = self.tokens.numel()
        self.num_chunks = max(0, (num_tokens - window_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_chunks:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_chunks}")

        start = idx * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_train_val_datasets(
    tokens: Union[List[int], torch.Tensor],
    seq_len: int,
    val_ratio: float = 0.1,
    stride: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> Tuple[TextChunkDataset, TextChunkDataset]:
    """
    Splits a continuous token stream deterministically into train and validation datasets.
    Validation tokens are taken from the end of the stream to prevent train/val leakage.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    if isinstance(tokens, list):
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    else:
        tokens_tensor = tokens.to(dtype=torch.long)

    total_tokens = tokens_tensor.numel()
    val_tokens_count = int(total_tokens * val_ratio)
    train_tokens_count = total_tokens - val_tokens_count

    # Enforce minimum lengths
    min_tokens = seq_len + 2
    if train_tokens_count < min_tokens or val_tokens_count < min_tokens:
        raise ValueError(
            f"Dataset too small to split into train/val with seq_len={seq_len}. "
            f"Total tokens={total_tokens}, train={train_tokens_count}, val={val_tokens_count}, required per split >= {min_tokens}"
        )

    train_tokens = tokens_tensor[:train_tokens_count]
    val_tokens = tokens_tensor[train_tokens_count:]

    train_ds = TextChunkDataset(train_tokens, seq_len=seq_len, stride=stride, vocab_size=vocab_size)
    val_ds = TextChunkDataset(val_tokens, seq_len=seq_len, stride=stride, vocab_size=vocab_size)
    return train_ds, val_ds


def create_dataloaders(
    train_dataset: TextChunkDataset,
    val_dataset: TextChunkDataset,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates deterministic PyTorch DataLoaders for training and evaluation.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
