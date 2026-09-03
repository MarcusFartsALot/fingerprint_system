"""Lightweight, explainable diagnostics for one attendance fingerprint scan.

These measurements describe how closely the final binary ridge map agrees
with an independently thresholded reference derived from the same capture.
They are processing diagnostics, not biometric-identification accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def derive_reference_ridges(image: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Return a boolean ridge map using a pipeline independent of the final output."""

    source = np.clip(image, 0, 255).astype(np.uint8)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(source)
    smoothed = cv2.GaussianBlur(contrast, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        6,
    )
    return (thresholded == 0) & (region_mask > 0)


def analyse_scan(
    prepared: np.ndarray,
    enhanced: np.ndarray,
    region_mask: np.ndarray,
    histogram_bins: int = 32,
) -> dict[str, object]:
    """Calculate ridge-map agreement and ROI intensity distributions."""

    arrays = (prepared, enhanced, region_mask)
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Scan analytics expects four two-dimensional images.")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Scan analytics images must use the same dimensions.")

    valid = region_mask > 0
    if not np.any(valid):
        raise ValueError("Scan analytics requires a non-empty fingerprint region.")

    reference_ridges = derive_reference_ridges(prepared, region_mask)
    output_ridges = (enhanced == 0) & valid

    reference = reference_ridges[valid]
    predicted = output_ridges[valid]
    true_ridge = int(np.count_nonzero(reference & predicted))
    missed_ridge = int(np.count_nonzero(reference & ~predicted))
    false_ridge = int(np.count_nonzero(~reference & predicted))
    true_valley = int(np.count_nonzero(~reference & ~predicted))

    precision = _safe_ratio(true_ridge, true_ridge + false_ridge)
    recall = _safe_ratio(true_ridge, true_ridge + missed_ridge)
    f1 = _safe_ratio(2 * true_ridge, 2 * true_ridge + false_ridge + missed_ridge)
    agreement = _safe_ratio(true_ridge + true_valley, int(valid.sum()))

    edges = np.linspace(0, 256, histogram_bins + 1)
    centres = ((edges[:-1] + edges[1:]) / 2.0).round().astype(int)
    prepared_hist = np.histogram(prepared[valid], bins=edges)[0].astype(np.int64)
    enhanced_hist = np.histogram(enhanced[valid], bins=edges)[0].astype(np.int64)

    return {
        "confusion_matrix": np.array(
            [[true_valley, false_ridge], [missed_ridge, true_ridge]],
            dtype=np.int64,
        ),
        "metrics": {
            "Accuracy": agreement,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        },
        "histogram_centres": centres,
        "prepared_histogram": prepared_hist,
        "enhanced_histogram": enhanced_hist,
        "roi_pixels": int(valid.sum()),
    }
