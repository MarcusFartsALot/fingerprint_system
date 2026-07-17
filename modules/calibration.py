"""
Shared Core Requirement: Image Calibration
-------------------------------------------
"Perform spatial scaling and rectification to maintain consistency between
pixel dimensions and physical measurements."

For fingerprints, the natural physical measurement is the ridge-to-ridge
period. Average human ridge period is fairly stable (roughly 0.40-0.55 mm,
i.e. ~450-550 dpi on a live scanner). This module:

  1. Estimates the local ridge frequency field from the orientation field
     using the sinusoidal x-signature method (Hong, Wan & Jain, 1998).
  2. Derives an *effective* scanning resolution (DPI) from the measured
     ridge period.
  3. Rescales the image to a standard reference DPI (default 500, the
     common AFIS standard) so that all downstream measurements —
     minutiae spacing, matching distances, singular point coordinates —
     are comparable across images acquired at different resolutions.

This directly reuses the orientation field already produced by Module A,
so no extra heavy computation is required.
"""

from dataclasses import dataclass

import cv2
import numpy as np

REFERENCE_RIDGE_PERIOD_MM = 0.475   # average adult ridge-to-ridge spacing
REFERENCE_DPI = 500                  # standard AFIS / NIST reference resolution
MM_PER_INCH = 25.4


@dataclass
class CalibrationResult:
    mean_ridge_period_px: float
    estimated_dpi: float
    scale_factor: float
    calibrated_image: np.ndarray


def _block_ridge_period(image_block: np.ndarray, theta: float, block_size: int) -> float:
    """
    Sinusoidal x-signature method: rotate the block so ridges run
    vertically, sum columns to build a 1-D wave, then measure the average
    distance between consecutive peaks. Returns np.nan if no reliable
    period can be found (e.g. flat/background block).
    """
    h, w = image_block.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(theta) + 90, 1.0)
    rotated = cv2.warpAffine(image_block, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    signature = rotated.sum(axis=0).astype(np.float64)
    signature -= signature.mean()

    if signature.std() < 1e-6:
        return np.nan

    # Find peaks: a point higher than both neighbours and above zero-crossing
    peaks = []
    for i in range(1, len(signature) - 1):
        if signature[i] > signature[i - 1] and signature[i] >= signature[i + 1] and signature[i] > 0:
            peaks.append(i)

    if len(peaks) < 2:
        return np.nan

    diffs = np.diff(peaks)
    diffs = diffs[(diffs > 2) & (diffs < block_size * 2)]
    if len(diffs) == 0:
        return np.nan

    return float(np.mean(diffs))


def estimate_ridge_frequency_field(image: np.ndarray, theta_field: np.ndarray,
                                    block_size: int = 24) -> np.ndarray:
    """
    Computes a block-wise ridge frequency map (cycles/pixel) across the
    whole image, using the orientation field to align each block before
    measuring the sinusoidal period.
    """
    h, w = image.shape
    freq_map = np.full((h, w), np.nan)
    img_float = image.astype(np.float64)

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_float[y:y + block_size, x:x + block_size]
            theta = theta_field[y + block_size // 2, x + block_size // 2]
            period_px = _block_ridge_period(block, theta, block_size)
            if not np.isnan(period_px) and period_px > 0:
                freq_map[y:y + block_size, x:x + block_size] = 1.0 / period_px

    return freq_map


def calibrate_image(image: np.ndarray, theta_field: np.ndarray, block_size: int = 24,
                     reference_dpi: int = REFERENCE_DPI) -> CalibrationResult:
    """
    Estimates the effective DPI of the input fingerprint image from its
    measured ridge period, then rescales it to ``reference_dpi`` so that
    minutiae/singular-point coordinates and matching distances are
    consistent across images of differing native resolution.
    """
    freq_map = estimate_ridge_frequency_field(image, theta_field, block_size)
    valid = freq_map[~np.isnan(freq_map)]

    if valid.size == 0:
        # No reliable ridge signal found (e.g. blank/very noisy image):
        # fall back to identity calibration rather than failing outright.
        mean_period_px = float(block_size) / 2.0
    else:
        mean_freq = float(np.mean(valid))
        mean_period_px = 1.0 / mean_freq if mean_freq > 0 else block_size / 2.0

    px_per_mm = mean_period_px / REFERENCE_RIDGE_PERIOD_MM
    estimated_dpi = px_per_mm * MM_PER_INCH
    estimated_dpi = float(np.clip(estimated_dpi, 150, 2000))  # sanity bound

    scale_factor = reference_dpi / estimated_dpi
    new_w = max(1, int(round(image.shape[1] * scale_factor)))
    new_h = max(1, int(round(image.shape[0] * scale_factor)))
    calibrated = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA)

    return CalibrationResult(
        mean_ridge_period_px=mean_period_px,
        estimated_dpi=estimated_dpi,
        scale_factor=scale_factor,
        calibrated_image=calibrated,
    )
