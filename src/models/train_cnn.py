"""Training and evaluation loop for PVPatchCNN."""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_metrics


def train_cnn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, lr: float, device: str,
              patience: int | None = None) -> dict:
    """Train for `epochs`, or until val loss has failed to improve for
    `patience` consecutive epochs.

    With `patience` set, `epochs` becomes a ceiling rather than a fixed budget,
    and the model is left holding its best-val-loss weights instead of whatever
    the final epoch produced. A fixed budget has to be right for every training
    set size at once, which it cannot be: the same 20 epochs that overfit the
    full sample leave a 10% subset badly undertrained.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        history["train_loss"].append(running_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        if patience is None:
            continue
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_cnn(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.to(device)
    model.eval()
    all_y_true, all_y_pred, all_y_proba = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (proba >= 0.5).astype(int)
            all_y_true.append(y.numpy())
            all_y_pred.append(pred)
            all_y_proba.append(proba)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)
    return compute_metrics(y_true, y_pred, y_proba)
