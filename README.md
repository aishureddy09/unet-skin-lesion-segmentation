# U-Net for Skin-Lesion Segmentation — implemented from scratch in NumPy

A genuine, small U-Net — encoder, bottleneck, decoder with skip connections —
implemented and **trained by hand-written backpropagation in pure NumPy**
(convolutions via im2col, max-pooling, upsampling, Adam, BCE + soft-Dice loss).
PyTorch was not available in the build environment, so rather than fake a
model, this project implements the real architecture and its gradients
directly. That makes every layer inspectable — useful when the research
interest is how segmentation models actually learn and fail.

**Research question.** How accurately can a U-Net-style model delineate a
lesion under variation in shape, contrast and image artefacts — and where do
its boundaries fail?

## Data — synthetic, honestly labelled

Not ISIC data; no real patient images. `src/data.py` builds 120 synthetic
32×32 images: a smooth skin-tone background with noise and occasional dark
"hair" lines, plus one darker rotated ellipse with a wobbled border as the
lesion. The paired mask is exact by construction (96 train / 24 test, seed 42).
This validates the pipeline end-to-end; it is an easy, separable task by
design, so the scores below must **not** be read as dermoscopy performance —
see Limitations.

## Architecture (`src/unet_numpy.py`)

Input 1×32×32, base width 8: conv blocks 1→8→8 (skip 1), pool, 8→16→16
(skip 2), pool, bottleneck 16→32→32, upsample + concat skip 2 → 48→16→16,
upsample + concat skip 1 → 24→8→8, 1×1 head. Training: Adam (lr 3e-3),
batch 8, 18 epochs, loss = BCE + (1 − soft Dice).

## Results (actual run, synthetic test set, n=24)

Training loss fell from **1.015 (epoch 1) to 0.021 (epoch 18)** in ~70 s on CPU.

| Metric | Mean | Std |
|---|---|---|
| Dice | 0.9896 | 0.0074 |
| IoU | 0.9796 | 0.0144 |
| Pixel precision | 0.9923 | — |
| Pixel sensitivity | 0.9871 | — |
| Pixel specificity | 0.9987 | — |

Worst test image: Dice 0.9655. Per-image scores are in
`results/per_image_metrics.csv`; `results/overlay_best|median|worst.png` show
image | ground truth | prediction side by side, and
`results/failure_modes.md` records what to inspect (wobbled-border extremes,
hair lines, small-lesion boundary cost).

## Run it

```bash
pip install -r requirements.txt
python -m src.train        # ~70 s on CPU; regenerates results/
python -m pytest tests -q  # 4 tests: metric values, data alignment, forward/backward + a learning step, output range
```

## Limitations

Synthetic ellipses on smooth backgrounds are far easier than real dermoscopy:
no vessels, pigment networks, rulers, gel bubbles, or genuine lesion texture,
and a single lesion per image. Dice 0.99 here would not survive contact with
ISIC data. The honest claim is narrower and still useful: a correct,
trainable U-Net with a reproducible evaluation harness (Dice/IoU implemented
and unit-tested, per-image reporting, failure overlays) that can be pointed
at a properly licensed real dataset next.

## Responsible use

Research and education only. Not a diagnostic or segmentation product for
clinical use. Never upload real patient images to a public repository.

## Author

Thikkam Aiswarya — MSc Artificial Intelligence with Healthcare, Aston
University. Her dissertation reviewed U-Net and segmentation for skin-lesion
analysis while implementing classification; this project is the hands-on
segmentation build that follows from it, for the GovAI UK Summer Fellowship
2027 portfolio.
