import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import make_dataset
from src.metrics import dice, iou
from src.unet_numpy import UNetNumpy, bce_dice_loss_and_grad


def test_dice_iou_known_values():
    a = np.zeros((4, 4)); a[:2, :] = 1
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0
    b = np.zeros((4, 4)); b[2:, :] = 1
    assert dice(a, b) == 0.0 and iou(a, b) == 0.0
    assert dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0  # both empty


def test_synthetic_pairs_align():
    X, Y = make_dataset(n=6, seed=1)
    assert X.shape == Y.shape == (6, 1, 32, 32)
    # lesion pixels are darker on average than background, by construction
    for i in range(6):
        assert X[i, 0][Y[i, 0] == 1].mean() < X[i, 0][Y[i, 0] == 0].mean()


def test_forward_backward_shapes_and_learning_step():
    model = UNetNumpy(seed=0)
    X, Y = make_dataset(n=4, seed=2)
    logits = model.forward(X)
    assert logits.shape == Y.shape
    loss0, grad = bce_dice_loss_and_grad(logits, Y)
    assert np.isfinite(loss0) and grad.shape == logits.shape
    model.backward(grad)
    model.step(lr=3e-3)
    loss1, _ = bce_dice_loss_and_grad(model.forward(X), Y)
    assert np.isfinite(loss1)  # training machinery runs end to end


def test_predict_proba_range():
    model = UNetNumpy(seed=0)
    X, _ = make_dataset(n=2, seed=3)
    p = model.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
