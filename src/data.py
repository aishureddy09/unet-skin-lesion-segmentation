"""SYNTHETIC dermoscopic-like images with paired lesion masks.

This is NOT ISIC data and contains no real patient images. Each sample is a
smooth skin-tone background with vignetting and noise, plus one darker
elliptical 'lesion' with an irregular border; the mask is that ellipse.
Ground truth is exact by construction, which makes it a valid test of the
segmentation *pipeline* — and an invalid basis for any clinical claim.
"""
import numpy as np


def make_dataset(n=120, size=32, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    Y = np.zeros((n, 1, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        skin = rng.uniform(0.55, 0.8)
        img = skin + 0.05 * np.sin(xx / size * np.pi) + rng.normal(0, 0.03, (size, size))
        cx, cy = rng.uniform(11, 21, 2)
        rx, ry = rng.uniform(5, 10, 2)
        angle = rng.uniform(0, np.pi)
        xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        wobble = 1 + 0.15 * np.sin(3 * np.arctan2(yr, xr) + rng.uniform(0, 6))
        mask = ((xr / (rx * wobble)) ** 2 + (yr / (ry * wobble)) ** 2) <= 1
        img[mask] -= rng.uniform(0.25, 0.45)          # lesion is darker
        img[mask] += rng.normal(0, 0.04, mask.sum())  # lesion texture
        # a few dark 'hair' lines, a classic dermoscopy confounder
        for _ in range(rng.integers(0, 3)):
            r = rng.integers(0, size)
            img[max(0, r - 0):r + 1, :] -= 0.12
        X[i, 0] = np.clip(img, 0, 1)
        Y[i, 0] = mask.astype(np.float32)
    return X, Y
