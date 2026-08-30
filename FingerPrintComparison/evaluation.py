"""
evaluation.py
-------------
Benchmarking utilities that work directly on user-uploaded fingerprint
images — no hardcoded/synthetic test pattern, and no synthetic degradation
step, involved.

Methodology
-----------
Real fingerprint images (from a phone photo, a scanner, a public dataset,
etc.) never come with pixel-level ridge/valley ground truth, so a Confusion
Matrix or Accuracy/Precision/Recall/F1 can't be computed against them
directly. To still get rigorous, quantitative metrics FROM the user's own
image, this module derives its reference standard straight from the upload:

- A pseudo ground-truth ridge/valley mask is derived directly from the
  uploaded image using a binarisation pipeline that is NOT one of the four
  candidate algorithms (CLAHE + adaptive thresholding), so no algorithm is
  unfairly favoured.
- Every candidate algorithm enhances the SAME uploaded image as-is — this
  tests each algorithm standalone, on real data, with no synthetic
  re-degradation step in between.
- Each algorithm's output is scored against that pseudo ground truth
  (Confusion Matrix / Accuracy / Precision / Recall / F1), plus a
  no-reference Ridge Clarity Score (there is no separate clean reference to
  compute PSNR/SSIM against, since nothing here is synthetically degraded).
"""

import numpy as np
import cv2

from algorithms import normalize_image, block_orientation


def derive_pseudo_ground_truth_mask(img):
    """
    Neutral ridge/valley binarisation used as a stand-in reference standard.
    Deliberately independent of all four candidate algorithms (uses CLAHE +
    adaptive thresholding, not orientation/Gabor/STFT/bilateral filtering) so
    that deriving the "ground truth" doesn't secretly favour any of them.
    """
    img_u8 = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(img_u8)
    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=6,
    )
    return (mask > 0).astype(np.uint8)


def ridge_clarity_score(img):
    """
    No-reference clarity score: the mean local ridge-orientation coherence,
    in [0,1]. Higher = more consistent, well-defined ridge flow. This is the
    sole image-quality signal reported, since every algorithm is benchmarked
    standalone on the uploaded image with no synthetic clean reference to
    compute PSNR/SSIM against.
    """
    norm = normalize_image(img)
    _, coherence = block_orientation(norm, block_size=16)
    return float(np.clip(np.mean(coherence), 0, 1))


def load_and_resize(gray_array, max_dim=300):
    """Resize a grayscale numpy array so its longer side is <= max_dim,
    keeping aspect ratio, for consistent processing speed."""
    h, w = gray_array.shape
    scale = max_dim / max(h, w)
    if scale < 1.0:
        gray_array = cv2.resize(
            gray_array, (max(int(w * scale), 1), max(int(h * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
    return gray_array
