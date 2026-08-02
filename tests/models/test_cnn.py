import torch
from src.models.cnn import PVPatchCNN


def test_forward_pass_output_shape():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    x = torch.rand(8, 6, 21, 21)
    logits = model(x)
    assert logits.shape == (8, 2)


def test_forward_pass_works_with_different_input_size():
    model = PVPatchCNN(in_channels=6, input_size=33, n_classes=2)
    x = torch.rand(4, 6, 33, 33)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_model_has_trainable_parameters():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0


def test_forward_pass_gradients_flow():
    model = PVPatchCNN(in_channels=6, input_size=21, n_classes=2)
    x = torch.rand(2, 6, 21, 21, requires_grad=False)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
