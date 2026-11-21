"""Segmentation metrics on binary masks (NumPy only)."""
import numpy as np


def _bin(a, thr=0.5):
    return np.asarray(a) >= thr


def dice(pred, true, thr=0.5):
    p, t = _bin(pred, thr).ravel(), _bin(true, thr).ravel()
    inter = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def iou(pred, true, thr=0.5):
    p, t = _bin(pred, thr).ravel(), _bin(true, thr).ravel()
    union = np.logical_or(p, t).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, t).sum() / union)


def pixel_metrics(pred, true, thr=0.5):
    p, t = _bin(pred, thr).ravel(), _bin(true, thr).ravel()
    tp = int(np.logical_and(p, t).sum()); fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum()); tn = int(np.logical_and(~p, ~t).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0}
