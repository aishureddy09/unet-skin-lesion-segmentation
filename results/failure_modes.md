# Failure modes (synthetic test set)

Worst test image: #0 with Dice 0.9894 (sensitivity 0.9894, precision 0.9894). See `overlay_worst.png` (image | ground truth | prediction).

Observed patterns to check on the overlays: (1) irregular/wobbled borders are under-segmented at the extremes of the wobble; (2) dark synthetic 'hair' lines crossing the background can be absorbed into the predicted mask; (3) small lesions have proportionally larger boundary error, so a single misplaced ring of pixels costs more Dice. None of this is clinical evidence — the data are synthetic shapes, not dermoscopy.
