"""Training and evaluation loop for PVPatchCNN."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_metrics


def train_cnn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, lr: float, device: str) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}

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
        history["val_loss"].append(val_loss / len(val_loader.dataset))

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
