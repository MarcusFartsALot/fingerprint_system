"""
Module B: Feature Detection — Singular Points (Core & Delta)
---------------------------------------------------------------
Author (team): Member B

Implements the Poincare Index method (Kawagoe & Tojo, 1984) directly on
the smoothed orientation field produced by Module A. This is a classical,
fully deterministic image-processing algorithm (no training data needed),
and its output (singular-point count/type) is the direct input to the
rule-based pattern classifier in Module C.

Poincare index at a point p is computed by summing the signed angular
differences of the orientation field around a small closed loop of
neighbouring blocks:

    PI(p) = sum_i  delta(i)

where delta(i) is the orientation difference between adjacent blocks on
the loop, adjusted to lie in (-pi/2, pi/2] to respect the pi-periodicity
of ridge orientation (a ridge orientation of 10 degrees is the same
physical direction as 190 degrees).

Classification of the index:
    PI ~= +pi   -> core point
    PI ~= -pi   -> delta point
    PI ~= 0     -> no singularity
"""

from dataclasses import dataclass
from typing import List

import numpy as np

CORE_THRESHOLD = 0.4     # fraction of pi tolerance around +pi
DELTA_THRESHOLD = 0.4    # fraction of pi tolerance around -pi


@dataclass
class SingularPoint:
    x: int
    y: int
    kind: str  # "core" or "delta"
    poincare_index: float


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference between two pi-periodic orientation angles."""
    diff = a - b
    while diff > np.pi / 2:
        diff -= np.pi
    while diff <= -np.pi / 2:
        diff += np.pi
    return diff


def _poincare_index(theta_field: np.ndarray, y: int, x: int, radius: int = 1) -> float:
    """
    Sums orientation differences around the 8 neighbours of (y, x),
    traversed clockwise, forming a closed loop.
    """
    offsets = [(-radius, -radius), (-radius, 0), (-radius, radius),
               (0, radius), (radius, radius), (radius, 0),
               (radius, -radius), (0, -radius), (-radius, -radius)]

    h, w = theta_field.shape
    angles = []
    for dy, dx in offsets:
        yy = np.clip(y + dy, 0, h - 1)
        xx = np.clip(x + dx, 0, w - 1)
        angles.append(theta_field[yy, xx])

    total = 0.0
    for i in range(len(angles) - 1):
        total += _angle_diff(angles[i + 1], angles[i])
    return total


def detect_singular_points(theta_field: np.ndarray, block_size: int = 16,
                            margin_ratio: float = 0.08) -> List[SingularPoint]:
    """
    Scans the orientation field on a block-size grid and flags cores and
    deltas wherever the Poincare index crosses the +/-pi thresholds.

    ``margin_ratio`` excludes a border strip (singular points detected at
    the very edge of a fingerprint image are almost always artefacts of
    the finger boundary, not true cores/deltas).
    """
    h, w = theta_field.shape
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    candidates: List[SingularPoint] = []
    for y in range(margin_y, h - margin_y, block_size):
        for x in range(margin_x, w - margin_x, block_size):
            pi_val = _poincare_index(theta_field, y, x, radius=block_size // 2 or 1)

            if pi_val >= np.pi * CORE_THRESHOLD and pi_val <= np.pi * (2 - CORE_THRESHOLD):
                candidates.append(SingularPoint(x, y, "core", pi_val))
            elif pi_val <= -np.pi * DELTA_THRESHOLD and pi_val >= -np.pi * (2 - DELTA_THRESHOLD):
                candidates.append(SingularPoint(x, y, "delta", pi_val))

    return _merge_nearby(candidates, block_size)


def _merge_nearby(points: List[SingularPoint], min_dist: int) -> List[SingularPoint]:
    """Collapses clusters of adjacent detections (from neighbouring grid
    cells all crossing the threshold near the same true singularity) into
    a single representative point per cluster."""
    if not points:
        return []

    merged: List[SingularPoint] = []
    used = [False] * len(points)

    for i, p in enumerate(points):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j] or points[j].kind != p.kind:
                continue
            if abs(points[j].x - p.x) < min_dist * 2 and abs(points[j].y - p.y) < min_dist * 2:
                cluster.append(points[j])
                used[j] = True

        avg_x = int(np.mean([c.x for c in cluster]))
        avg_y = int(np.mean([c.y for c in cluster]))
        avg_pi = float(np.mean([c.poincare_index for c in cluster]))
        merged.append(SingularPoint(avg_x, avg_y, p.kind, avg_pi))

    return merged
