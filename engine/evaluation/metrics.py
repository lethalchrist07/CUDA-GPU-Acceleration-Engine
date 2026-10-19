from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate categorical classification accuracy."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return float(correct / total) if total > 0 else 0.0


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Validate numerical parity between two tensors (e.g. CPU vs GPU outputs).
    Moves tensors to CPU for comparison.
    """
    c1 = t1.detach().cpu().to(torch.float32)
    c2 = t2.detach().cpu().to(torch.float32)

    diff = torch.abs(c1 - c2)
    max_err = float(diff.max().item())
    rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
    mismatches = int((diff > tolerance).sum().item())
    total_elements = c1.numel()
    is_close = mismatches == 0

    return {
        "max_abs_diff": max_err,
        "rmse": rmse,
        "tolerance": tolerance,
        "mismatched_elements": mismatches,
        "total_elements": total_elements,
        "is_close": is_close,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model loss and accuracy across a dataset."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "samples": total_samples,
    }
