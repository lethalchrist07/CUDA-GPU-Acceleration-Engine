from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticClassificationDataset(Dataset):
    """
    Lightweight synthetic dataset for reproducible model testing,
    verification, and performance benchmarking without external downloads.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 64,
        num_classes: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, input_dim, generator=generator)

        # Generate target classes based on random linear projections with noise
        weights = torch.randn(input_dim, num_classes, generator=generator)
        logits = self.features @ weights
        self.labels = torch.argmax(logits, dim=1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_synthetic_dataloaders(
    num_samples: int = 1000,
    input_dim: int = 64,
    num_classes: int = 10,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Helper function to create training and validation DataLoaders.
    """
    dataset = SyntheticClassificationDataset(
        num_samples=num_samples,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=seed,
    )
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
