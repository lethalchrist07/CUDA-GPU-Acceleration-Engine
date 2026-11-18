"""
CIFAR-10 Dataset Management Module for CUDA GPU Acceleration Engine (Phase 9).
Provides dataset downloading, cryptographic MD5 integrity verification,
train/val/test partitioning, standard normalization, and PyTorch DataLoaders.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

CIFAR10_URLS = [
    "https://huggingface.co/datasets/Wenbinwang/cifar10-data/resolve/main/cifar-10-python.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def compute_md5(filepath: Union[str, Path]) -> str:
    """Compute MD5 checksum of a file in streaming chunks."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate_synthetic_cifar10_directory(target_dir: Union[str, Path], samples_per_batch: int = 1000) -> Path:
    """
    Generate synthetic CIFAR-10 batch files for rapid offline testing and pilot verification.
    Matches the exact binary schema of the real CIFAR-10 dataset.
    """
    path = Path(target_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    for b in range(1, 6):
        batch_path = path / f"data_batch_{b}"
        if not batch_path.is_file():
            data = rng.randint(0, 256, size=(samples_per_batch, 3072), dtype=np.uint8)
            labels = rng.randint(0, 10, size=(samples_per_batch,)).tolist()
            with open(batch_path, "wb") as f:
                pickle.dump({b"data": data, b"labels": labels}, f)

    test_path = path / "test_batch"
    if not test_path.is_file():
        data = rng.randint(0, 256, size=(samples_per_batch, 3072), dtype=np.uint8)
        labels = rng.randint(0, 10, size=(samples_per_batch,)).tolist()
        with open(test_path, "wb") as f:
            pickle.dump({b"data": data, b"labels": labels}, f)

    meta_path = path / "batches.meta"
    if not meta_path.is_file():
        with open(meta_path, "wb") as f:
            pickle.dump({b"label_names": [c.encode("utf-8") for c in CIFAR10_CLASSES]}, f)

    logger.info(f"Generated synthetic CIFAR-10 schema at {path}")
    return path


def download_and_verify_cifar10(
    root_dir: Union[str, Path] = "data",
    allow_synthetic_fallback: bool = False,
    force_synthetic: bool = False,
) -> Path:
    """
    Download CIFAR-10 python tarball, verify MD5 integrity, and unpack archive.
    If force_synthetic is True, or if allow_synthetic_fallback is True and download fails,
    generates synthetic batches with the exact schema.
    Returns path to extracted directory.
    """
    root_path = Path(root_dir).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    extracted_dir = root_path / CIFAR10_DIR
    archive_path = root_path / CIFAR10_ARCHIVE

    if force_synthetic:
        return generate_synthetic_cifar10_directory(extracted_dir)

    # If already extracted and verified
    test_batch_file = extracted_dir / "test_batch"
    if extracted_dir.is_dir() and test_batch_file.is_file():
        logger.info(f"CIFAR-10 dataset already extracted at: {extracted_dir}")
        return extracted_dir

    # If archive exists, verify MD5
    archive_valid = False
    if archive_path.is_file():
        logger.info(f"Checking existing archive integrity: {archive_path}")
        existing_md5 = compute_md5(archive_path)
        if existing_md5 == CIFAR10_MD5:
            archive_valid = True
            logger.info("Existing archive MD5 verified successfully.")
        else:
            logger.warning(f"Archive MD5 mismatch ({existing_md5} != {CIFAR10_MD5}), re-downloading...")
            archive_path.unlink(missing_ok=True)

    if not archive_valid:
        downloaded = False
        last_error = None
        for url in CIFAR10_URLS:
            try:
                logger.info(f"Downloading CIFAR-10 from {url}...")
                with requests.get(url, stream=True, timeout=20, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(archive_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                downloaded = True
                break
            except Exception as exc:
                last_error = exc
                logger.warning(f"Failed to download from {url}: {exc}")

        if not downloaded:
            if allow_synthetic_fallback:
                logger.warning(
                    f"Download failed ({last_error}). Falling back to synthetic CIFAR-10 for offline operation."
                )
                return generate_synthetic_cifar10_directory(extracted_dir)
            raise RuntimeError(f"Could not download CIFAR-10 dataset from any source: {last_error}")

        # Verify MD5
        downloaded_md5 = compute_md5(archive_path)
        if downloaded_md5 != CIFAR10_MD5:
            if allow_synthetic_fallback:
                logger.warning("Checksum mismatch; falling back to synthetic CIFAR-10.")
                return generate_synthetic_cifar10_directory(extracted_dir)
            raise ValueError(f"Downloaded CIFAR-10 checksum mismatch: expected {CIFAR10_MD5}, got {downloaded_md5}")
        logger.info(f"Downloaded archive verified: MD5={downloaded_md5}")

    # Extract tarball
    logger.info(f"Extracting CIFAR-10 archive to {root_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root_path)

    if not test_batch_file.is_file():
        if allow_synthetic_fallback:
            return generate_synthetic_cifar10_directory(extracted_dir)
        raise FileNotFoundError(f"Extraction failed: {test_batch_file} not found.")

    logger.info(f"CIFAR-10 successfully unpacked at {extracted_dir}")
    return extracted_dir


class CIFAR10Dataset(Dataset):
    """
    In-memory PyTorch Dataset for CIFAR-10 with tensor transformations and augmentations.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        normalize: bool = True,
    ):
        """
        images: uint8 array of shape (N, 3, 32, 32)
        labels: int64 array of shape (N,)
        """
        assert images.shape[0] == labels.shape[0], "Images and labels count mismatch"
        self.images = images
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.normalize = normalize

        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def _apply_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random horizontal flip and random crop with reflection padding 4."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])

        # Random crop with 4px padding
        pad = 4
        padded = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode="reflect")
        crop_y = torch.randint(0, 2 * pad + 1, (1,)).item()
        crop_x = torch.randint(0, 2 * pad + 1, (1,)).item()
        return padded[:, crop_y : crop_y + 32, crop_x : crop_x + 32]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = self.images[idx]
        img_tensor = torch.from_numpy(img_np).float() / 255.0  # Scale to [0, 1]

        if self.augment:
            img_tensor = self._apply_augmentation(img_tensor)

        if self.normalize:
            img_tensor = (img_tensor - self.mean) / self.std

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor


def load_cifar10_raw(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Read raw batches from unpacked CIFAR-10 directory.
    Returns: (train_images, train_labels, test_images, test_labels, class_names)
    Images are shaped (N, 3, 32, 32) as uint8.
    """
    train_images_list = []
    train_labels_list = []

    for batch_id in range(1, 6):
        batch_file = data_dir / f"data_batch_{batch_id}"
        with open(batch_file, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
            data = entry[b"data"]  # (10000, 3072)
            labels = entry[b"labels"]
            train_images_list.append(data)
            train_labels_list.extend(labels)

    train_data = np.concatenate(train_images_list, axis=0)  # (50000, 3072)
    train_images = train_data.reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels_list, dtype=np.int64)

    test_file = data_dir / "test_batch"
    with open(test_file, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
        test_data = entry[b"data"]
        test_labels = np.array(entry[b"labels"], dtype=np.int64)
    test_images = test_data.reshape(-1, 3, 32, 32)

    meta_file = data_dir / "batches.meta"
    class_names = CIFAR10_CLASSES
    if meta_file.is_file():
        try:
            with open(meta_file, "rb") as f:
                meta = pickle.load(f, encoding="bytes")
                class_names = [n.decode("utf-8") for n in meta[b"label_names"]]
        except Exception as exc:
            logger.warning(f"Could not parse batches.meta: {exc}, using default class names.")

    return train_images, train_labels, test_images, test_labels, class_names


def create_cifar10_dataloaders(
    data_dir: Union[str, Path] = "data",
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    allow_synthetic_fallback: bool = False,
    force_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Prepare CIFAR-10 training, validation, and isolated test DataLoaders.
    Handles automatic download and MD5 verification.
    """
    extracted_path = download_and_verify_cifar10(
        root_dir=data_dir,
        allow_synthetic_fallback=allow_synthetic_fallback,
        force_synthetic=force_synthetic,
    )
    train_imgs, train_lbls, test_imgs, test_lbls, class_names = load_cifar10_raw(extracted_path)

    # Deterministic split into train and validation sets
    total_train = len(train_imgs)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_train)

    val_size = int(total_train * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    sub_train_imgs = train_imgs[train_indices]
    sub_train_lbls = train_lbls[train_indices]
    sub_val_imgs = train_imgs[val_indices]
    sub_val_lbls = train_lbls[val_indices]

    # Handle sample limits for fast pilots or tests
    if max_train_samples is not None and max_train_samples < len(sub_train_imgs):
        sub_train_imgs = sub_train_imgs[:max_train_samples]
        sub_train_lbls = sub_train_lbls[:max_train_samples]

    if max_val_samples is not None and max_val_samples < len(sub_val_imgs):
        sub_val_imgs = sub_val_imgs[:max_val_samples]
        sub_val_lbls = sub_val_lbls[:max_val_samples]

    if max_test_samples is not None and max_test_samples < len(test_imgs):
        test_imgs = test_imgs[:max_test_samples]
        test_lbls = test_lbls[:max_test_samples]

    train_dataset = CIFAR10Dataset(sub_train_imgs, sub_train_lbls, augment=augment, normalize=True)
    val_dataset = CIFAR10Dataset(sub_val_imgs, sub_val_lbls, augment=False, normalize=True)
    test_dataset = CIFAR10Dataset(test_imgs, test_lbls, augment=False, normalize=True)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"CIFAR-10 DataLoaders built: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}, BatchSize={batch_size}"
    )

    return train_loader, val_loader, test_loader, class_names
