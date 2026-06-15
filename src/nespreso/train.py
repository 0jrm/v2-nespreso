"""Training loop extracted from monolith with opt-in TensorBoard logging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on the provided data with CUDA support."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
                batch_indices = batch[2] if len(batch) > 2 else None
            else:
                inputs, labels = batch
                batch_indices = None

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels, batch_indices)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    summary_writer: SummaryWriter | None = None,
    trajectory: list[dict[str, float]] | None = None,
) -> nn.Module:
    """
    Train the model with early stopping and CUDA support.

    When ``summary_writer`` is provided (opt-in via config), logs train/val loss
    per epoch. Default is off so numerics and behavior match the monolith.
    """
    model.to(device)
    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
                batch_indices = batch[2] if len(batch) > 2 else None
            else:
                inputs, labels = batch
                batch_indices = None

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, batch_indices)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)

        if trajectory is not None:
            trajectory.append(
                {
                    "epoch": float(epoch),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

        if summary_writer is not None:
            summary_writer.add_scalar("loss/train", avg_train_loss, epoch)
            summary_writer.add_scalar("loss/val", avg_val_loss, epoch)

        if epoch == 0 or epoch % 10 == 9:
            print(
                f"Epoch [{(epoch + 1):4.0f}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Patience left: "
                f"{(100 * (patience - no_improve_count) / patience):3.0f}% | Best: {best_val_loss:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at Epoch {epoch + 1}")
                break

    model.load_state_dict(best_weights)
    return model
