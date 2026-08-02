import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.patch_dataset import PVPatchDataset
from src.models.cnn import PVPatchCNN
from src.models.train_cnn import train_cnn, evaluate_cnn


def _separable_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n_pv, n_nonpv = n // 2, n - n // 2
    pv_patches = rng.normal(0.8, 0.05, (n_pv, 6, 21, 21)).astype(np.float32)
    nonpv_patches = rng.normal(0.2, 0.05, (n_nonpv, 6, 21, 21)).astype(np.float32)
    patches = np.concatenate([pv_patches, nonpv_patches])
    labels = np.array(["pv"] * n_pv + ["non_pv"] * n_nonpv)
    return PVPatchDataset(patches, labels)


def _unlearnable_dataset(n=120, seed=0):
    """Labels independent of the pixels, so val loss can only get worse once
    the model starts memorising the training set."""
    rng = np.random.default_rng(seed)
    patches = rng.normal(0.5, 0.1, (n, 6, 21, 21)).astype(np.float32)
    labels = rng.choice(["pv", "non_pv"], size=n)
    return PVPatchDataset(patches, labels)


def _val_loss(model, loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            total += criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
    return total / len(loader.dataset)


def test_train_cnn_runs_every_epoch_when_no_patience_is_given():
    train_loader = DataLoader(_separable_dataset(seed=0), batch_size=16, shuffle=True)
    val_loader = DataLoader(_separable_dataset(seed=1), batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21)

    history = train_cnn(model, train_loader, val_loader, epochs=4, lr=1e-3, device="cpu")

    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4


def test_train_cnn_stops_early_when_val_loss_stops_improving():
    torch.manual_seed(0)
    train_loader = DataLoader(_unlearnable_dataset(seed=0), batch_size=16, shuffle=True)
    val_loader = DataLoader(_unlearnable_dataset(seed=1), batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21)

    history = train_cnn(model, train_loader, val_loader, epochs=40, lr=1e-3,
                        device="cpu", patience=3)

    assert len(history["val_loss"]) < 40


def test_train_cnn_leaves_the_best_weights_in_the_model_not_the_last_ones():
    torch.manual_seed(0)
    train_loader = DataLoader(_unlearnable_dataset(seed=0), batch_size=16, shuffle=True)
    val_loader = DataLoader(_unlearnable_dataset(seed=1), batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21)

    history = train_cnn(model, train_loader, val_loader, epochs=40, lr=1e-3,
                        device="cpu", patience=3)

    assert _val_loss(model, val_loader) == pytest.approx(min(history["val_loss"]),
                                                         abs=1e-5)


def test_train_cnn_reduces_loss_on_separable_synthetic_data():
    train_ds = _separable_dataset(seed=0)
    val_ds = _separable_dataset(seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    history = train_cnn(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu")
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_evaluate_cnn_returns_metrics_dict_with_expected_keys():
    val_ds = _separable_dataset(seed=1)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    metrics = evaluate_cnn(model, val_loader, device="cpu")
    assert set(metrics.keys()) == {
        "overall_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"
    }


def test_train_cnn_achieves_high_accuracy_on_trivially_separable_data():
    train_ds = _separable_dataset(seed=0)
    val_ds = _separable_dataset(seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    train_cnn(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu")
    metrics = evaluate_cnn(model, val_loader, device="cpu")
    assert metrics["overall_accuracy"] > 0.9
