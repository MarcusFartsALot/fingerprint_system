"""
synthetic.py
------------
OPTIONAL demo-only fallback: a mathematically generated ridge pattern, used
only when the user has no real fingerprint image on hand and wants to
explore the Algorithm Comparison tab. This is NOT the primary evaluation
path — see evaluation.py, which benchmarks directly against real uploaded
fingerprint images. Any results produced from this synthetic pattern are
clearly labelled "demo" in the UI and should not be quoted as your real
benchmark numbers in the report.
"""

import numpy as np

from evaluation import degrade_image  # shared, generic degradation logic

__all__ = ["generate_synthetic_fingerprint", "degrade_image", "make_demo_sample"]


def generate_synthetic_fingerprint(size=256, seed=None, whorl_strength=None):
    """
    Generate a synthetic whorl/loop-like ridge pattern using a swirling
    sinusoidal phase field in polar coordinates, plus the exact ground
    truth ridge (1) / valley (0) binary mask used to derive the pattern.
    DEMO USE ONLY — see evaluation.py for the real-image benchmarking path.
    """
    rng = np.random.default_rng(seed)
    if whorl_strength is None:
        whorl_strength = rng.uniform(0.4, 1.3)

    cx = size / 2 + rng.uniform(-25, 25)
    cy = size / 2 + rng.uniform(-25, 25)
    y, x = np.mgrid[0:size, 0:size]
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    wavelength = rng.uniform(9, 13)
    swirl = whorl_strength * theta
    phase = 2 * np.pi * r / wavelength + swirl
    pattern = 0.5 + 0.5 * np.cos(phase)

    mask = np.exp(-((r / (size * 0.62)) ** 4))
    img = pattern * mask + (1 - mask) * 0.5

    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    gt_mask = (pattern > 0.5).astype(np.uint8)
    return img_uint8, gt_mask


def make_demo_sample(size=256, noise_std=15, blur_sigma=1.6, occlusion=True, seed=None):
    """Convenience wrapper: returns (clean, gt_mask, degraded) for one demo sample."""
    clean, gt_mask = generate_synthetic_fingerprint(size=size, seed=seed)
    degraded = degrade_image(clean, noise_std=noise_std, blur_sigma=blur_sigma,
                              occlusion=occlusion, seed=seed)
    return clean, gt_mask, degraded
