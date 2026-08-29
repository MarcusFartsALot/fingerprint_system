"""Classical fingerprint enhancement and matching utilities.

The production enhancement engine preserves phone-camera ridge evidence with
clipped local contrast enhancement, edge-preserving bilateral denoising and a
mild unsharp mask. Adaptive binarisation, thinning and ridge repair then create
the stored template. A texture-based foreground guard keeps phone-photo
backgrounds out of the biometric template.
Attendance reference matching is a separate downstream concern.

No neural network or remote service is used.  All biometric processing stays
on the computer running the Streamlit application.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


CANVAS_SIZE = 360
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PIPELINE_SCHEMA_VERSION = "phone-ridge-preserving-v10"


@dataclass
class PipelineResult:
    """Images, feature maps and quality measurements from one capture."""

    original: np.ndarray
    prepared: np.ndarray
    region_mask: np.ndarray
    contrast_enhanced: np.ndarray
    denoised: np.ndarray
    detail_enhanced: np.ndarray
    binary: np.ndarray
    thinned: np.ndarray
    enhanced: np.ndarray
    skeleton: np.ndarray
    quality: dict[str, float | int]
    stages: list[str]
    processing_ms: float


@dataclass
class MatchEvidence:
    """Interpretable component scores for a fingerprint comparison."""

    similarity: float
    reference_score: float
    orb_score: float
    structural_score: float
    spectral_score: float
    reference_matches: int
    reference_inliers: int
    good_matches: int
    geometric_inliers: int
    keypoints_query: int
    keypoints_template: int
    used_canonical_reference: bool


def decode_image(source: bytes | bytearray | str | Path | BinaryIO) -> np.ndarray:
    """Decode an upload as RGB so colour can assist phone-photo segmentation."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported fingerprint format: {path.suffix}")
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:  # Pillow provides format-specific exception types.
        raise ValueError("The uploaded file is not a readable fingerprint image.") from exc

    if array.size == 0 or min(array.shape[:2]) < 40:
        raise ValueError("Fingerprint images must be at least 40 x 40 pixels.")
    return array


def prepare_canvas(image: np.ndarray, size: int = CANVAS_SIZE) -> np.ndarray:
    """Resize without distortion and pad with the estimated background tone."""

    image = _to_uint8(image)
    if image.shape == (size, size):
        return image
    height, width = image.shape
    scale = min((size - 24) / max(height, 1), (size - 24) / max(width, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    border = np.concatenate(
        [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
    )
    background = int(np.median(border))
    canvas = np.full((size, size), background, dtype=np.uint8)
    y0 = (size - new_height) // 2
    x0 = (size - new_width) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def _prepare_mask_canvas(mask: np.ndarray, size: int = CANVAS_SIZE) -> np.ndarray:
    """Apply the same aspect-preserving canvas transform to a foreground mask."""

    mask = (_to_uint8(mask) > 0).astype(np.uint8) * 255
    height, width = mask.shape
    scale = min((size - 24) / max(height, 1), (size - 24) / max(width, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_height) // 2
    x0 = (size - new_width) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def extract_fingertip_roi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop the central skin-coloured fingertip from a phone photograph.

    Fingerprint matching must be invariant to how much background the camera
    captured.  Skin colour proposes the foreground, connected-component and
    centre constraints select the finger, and a row-width guard removes the
    hand where it widens below the fingertip.
    """

    rgb = _to_uint8(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb, None
    rgb = rgb[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, cr, cb = cv2.split(ycrcb)
    _, saturation, value = cv2.split(hsv)

    # Broad thresholds cover light and dark skin while excluding neutral walls,
    # windows and most furniture. Geometry performs the final selection.
    skin_likelihood = (
        (cr >= 128)
        & (cr <= 190)
        & (cb >= 68)
        & (cb <= 142)
        & (saturation >= 18)
        & (value >= 38)
    )
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    centre_prior = (
        ((xx - width / 2.0) / (width * 0.47)) ** 2
        + ((yy - height * 0.51) / (height * 0.62)) ** 2
        <= 1.0
    )
    # A centre seed is reliable for the app's capture instructions. GrabCut then
    # follows the actual fingertip boundary even when a brown wall or bright
    # window has similar intensity to skin.
    grab_mask = np.full(gray.shape, cv2.GC_BGD, dtype=np.uint8)
    grab_mask[centre_prior] = cv2.GC_PR_BGD
    grab_mask[centre_prior & skin_likelihood] = cv2.GC_PR_FGD
    definite_finger = (
        ((xx - width / 2.0) / (width * 0.085)) ** 2
        + ((yy - height * 0.50) / (height * 0.14)) ** 2
        <= 1.0
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed_values = lab[definite_finger & skin_likelihood]
    if seed_values.size:
        centre_colour = np.median(seed_values, axis=0)
        colour_distance = (
            ((lab[:, :, 0] - centre_colour[0]) / 58.0) ** 2
            + ((lab[:, :, 1] - centre_colour[1]) / 22.0) ** 2
            + ((lab[:, :, 2] - centre_colour[2]) / 22.0) ** 2
        )
        skin_likelihood &= colour_distance <= 1.45
        grab_mask[centre_prior & ~skin_likelihood] = cv2.GC_PR_BGD
        grab_mask[centre_prior & skin_likelihood] = cv2.GC_PR_FGD
    grab_mask[definite_finger] = cv2.GC_FGD
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            grab_mask,
            None,
            np.zeros((1, 65), dtype=np.float64),
            np.zeros((1, 65), dtype=np.float64),
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        skin = (
            np.isin(grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
            & (skin_likelihood | definite_finger)
        ).astype(np.uint8)
    except cv2.error:
        skin = (centre_prior & skin_likelihood).astype(np.uint8)
    skin[~centre_prior] = 0
    kernel_size = max(9, (min(height, width) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    skin = cv2.morphologyEx(
        skin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(skin, connectivity=8)
    best_label = 0
    best_score = 0.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(gray.size * 0.025):
            continue
        cx, cy = centroids[label]
        distance = np.hypot((cx - width / 2.0) / width, (cy - height * 0.48) / height)
        score = area * float(np.clip(1.35 - 2.2 * distance, 0.05, 1.0))
        if labels[height // 2, width // 2] == label:
            score *= 1.5
        if score > best_score:
            best_label, best_score = label, score

    if best_label == 0:
        return gray, None
    component = (labels == best_label).astype(np.uint8)
    rows = np.flatnonzero(np.any(component > 0, axis=1))
    if rows.size < max(40, int(height * 0.18)):
        return gray, None

    top, bottom = int(rows[0]), int(rows[-1])
    row_widths = np.count_nonzero(component, axis=1)
    reference_start = top + int((bottom - top) * 0.18)
    reference_end = top + int((bottom - top) * 0.62)
    positive_widths = row_widths[reference_start : reference_end + 1]
    positive_widths = positive_widths[positive_widths > 0]
    typical_width = float(np.median(positive_widths)) if positive_widths.size else float(row_widths.max())
    search_start = top + int((bottom - top) * 0.58)
    too_wide = row_widths > typical_width * 1.42
    too_narrow = (row_widths > 0) & (row_widths < typical_width * 0.58)
    wide_run = np.convolve(too_wide.astype(np.uint8), np.ones(9, dtype=np.uint8), mode="same")
    narrow_run = np.convolve(too_narrow.astype(np.uint8), np.ones(9, dtype=np.uint8), mode="same")
    end_candidates = np.flatnonzero(
        ((wide_run >= 7) | (narrow_run >= 7)) & (np.arange(height) >= search_start)
    )
    if end_candidates.size:
        bottom = max(search_start, int(end_candidates[0]) - 3)

    selected = component[top : bottom + 1]
    columns = np.flatnonzero(np.any(selected > 0, axis=0))
    if columns.size < max(30, int(width * 0.12)):
        return gray, None
    left, right = int(columns[0]), int(columns[-1])
    pad_x = max(4, int((right - left + 1) * 0.035))
    pad_y = max(4, int((bottom - top + 1) * 0.025))
    left, right = max(0, left - pad_x), min(width - 1, right + pad_x)
    top, bottom = max(0, top - pad_y), min(height - 1, bottom + pad_y)

    roi_gray = gray[top : bottom + 1, left : right + 1]
    roi_mask = component[top : bottom + 1, left : right + 1] * 255
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    return roi_gray, roi_mask


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()
    array = array.astype(np.float64)
    minimum, maximum = float(array.min()), float(array.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip((array - minimum) * 255.0 / (maximum - minimum), 0, 255).astype(
        np.uint8
    )


def _normalise(image: np.ndarray, target_mean: float = 128.0, target_var: float = 900.0) -> np.ndarray:
    """Mean/variance normalisation used before local spectral analysis."""

    image_f = image.astype(np.float64)
    mean = float(image_f.mean())
    variance = float(image_f.var())
    normalised = target_mean + np.sign(image_f - mean) * np.sqrt(
        target_var * (image_f - mean) ** 2 / (variance + 1e-8)
    )
    return np.clip(normalised, 0, 255)


def clahe_contrast_enhancement(
    image: np.ndarray,
    region_mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Increase local ridge contrast without amplifying every pore indefinitely."""

    enhanced = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    ).apply(_to_uint8(image))
    enhanced[region_mask == 0] = 255
    return enhanced


def bilateral_ridge_denoising(
    image: np.ndarray,
    region_mask: np.ndarray,
    diameter: int = 5,
    sigma_colour: float = 12.0,
    sigma_space: float = 3.0,
) -> np.ndarray:
    """Suppress camera noise while retaining narrow ridge boundaries."""

    filtered = cv2.bilateralFilter(
        _to_uint8(image),
        diameter,
        sigma_colour,
        sigma_space,
    )
    inside = region_mask > 0
    filtered[~inside] = 255
    return filtered


def mild_unsharp_ridge_enhancement(
    image: np.ndarray,
    region_mask: np.ndarray,
    sigma: float = 0.8,
    amount: float = 0.75,
) -> np.ndarray:
    """Strengthen existing ridge edges without synthesising periodic lines."""

    image_f = _to_uint8(image).astype(np.float32)
    blurred = cv2.GaussianBlur(image_f, (0, 0), sigma)
    sharpened = image_f + amount * (image_f - blurred)
    enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
    enhanced[region_mask == 0] = 255
    return enhanced


def ridge_coherence(image: np.ndarray, region_mask: np.ndarray) -> float:
    """Measure how consistently local gradients follow ridge-like directions."""

    image_f = _to_uint8(image).astype(np.float32)
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 3.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 3.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 3.0)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-6)
    inside = region_mask > 0
    if not np.any(inside):
        return 0.0
    return float(np.clip(np.median(coherence[inside]), 0.0, 1.0))


def generate_region_mask(
    image: np.ndarray,
) -> np.ndarray:
    """Locate the connected fingerprint texture and reject borders/background.

    Local ridge texture creates the initial foreground. Large closing operations
    join individual ridges into one fingertip-shaped component, and a centre
    prior prevents a textured background object from becoming the fingerprint.
    """

    image_f = image.astype(np.float32)
    local_mean = cv2.boxFilter(image_f, cv2.CV_32F, (17, 17))
    local_square_mean = cv2.boxFilter(image_f * image_f, cv2.CV_32F, (17, 17))
    local_deviation = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    deviation_threshold = max(4.0, float(np.percentile(local_deviation, 48)))
    candidate = (local_deviation >= deviation_threshold).astype(np.uint8)

    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    centre_prior = (
        ((x - width / 2.0) / (width * 0.46)) ** 2
        + ((y - height / 2.0) / (height * 0.54)) ** 2
        <= 1.0
    )
    candidate[~centre_prior] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        fallback = np.zeros_like(image, dtype=np.uint8)
        cv2.ellipse(fallback, (width // 2, height // 2), (width * 3 // 10, height * 9 // 20), 0, 0, 360, 255, -1)
        return fallback

    best_label = 0
    best_score = 0.0
    best_area = 0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        box_area = max(component_width * component_height, 1)
        density = area / box_area
        component_cx = x + component_width / 2.0
        component_cy = y + component_height / 2.0
        centre_distance = np.hypot(
            (component_cx - width / 2.0) / width,
            (component_cy - height / 2.0) / height,
        )
        centre_weight = float(np.clip(1.25 - 2.0 * centre_distance, 0.08, 1.0))
        score = area * min(density / 0.28, 1.0) ** 2 * centre_weight
        if score > best_score:
            best_label, best_score, best_area = label, score, area

    component = (labels == best_label).astype(np.uint8)
    if best_label == 0 or best_area < int(image.size * 0.025):
        fallback = np.zeros_like(image, dtype=np.uint8)
        cv2.ellipse(fallback, (width // 2, height // 2), (width * 3 // 10, height * 9 // 20), 0, 0, 360, 255, -1)
        return fallback

    component = cv2.morphologyEx(
        component,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    component = cv2.dilate(
        component,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    dilated_count, dilated_labels, dilated_stats, _ = cv2.connectedComponentsWithStats(component, 8)
    if dilated_count > 1:
        dilated_label = int(np.argmax(dilated_stats[1:, cv2.CC_STAT_AREA])) + 1
        component = (dilated_labels == dilated_label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(component)
    if contours:
        fingerprint_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            mask,
            [cv2.convexHull(fingerprint_contour)],
            -1,
            1,
            thickness=cv2.FILLED,
        )
    return mask.astype(np.uint8) * 255


def refine_ridge_region(image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
    """Keep the central skin region that contains coherent fingerprint ridges.

    Colour segmentation finds the finger, but similarly coloured furniture or
    skin below the fingerprint pad can remain connected. Fingerprint ridges have
    a locally consistent doubled-angle gradient field, so coherence and texture
    separate the ridge-bearing pad before any enhancement filter is executed.
    """

    image_f = image.astype(np.float32)
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 4.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 4.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 4.0)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-6)

    local_mean = cv2.boxFilter(image_f, cv2.CV_32F, (17, 17))
    local_square_mean = cv2.boxFilter(image_f * image_f, cv2.CV_32F, (17, 17))
    local_deviation = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    inside = initial_mask > 0
    candidate = (inside & (coherence >= 0.24) & (local_deviation >= 3.0)).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    height, width = image.shape
    best_label = 0
    best_score = 0.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(image.size * 0.018):
            continue
        cx, cy = centroids[label]
        distance = np.hypot((cx - width / 2.0) / width, (cy - height / 2.0) / height)
        score = area * float(np.clip(1.35 - 2.0 * distance, 0.08, 1.0))
        if labels[height // 2, width // 2] == label:
            score *= 1.35
        if score > best_score:
            best_label, best_score = label, score

    if best_label == 0:
        return initial_mask.copy()
    component = (labels == best_label).astype(np.uint8)
    component = cv2.dilate(
        component,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return initial_mask.copy()
    refined = np.zeros_like(component)
    cv2.drawContours(refined, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, cv2.FILLED)
    refined &= inside.astype(np.uint8)
    coverage = float(np.mean(refined > 0))
    if coverage < 0.07:
        return initial_mask.copy()
    return refined * 255


def stft_ridge_enhancement(
    image: np.ndarray,
    region_mask: np.ndarray,
    block_size: int = 32,
    stride: int = 16,
) -> tuple[np.ndarray, float]:
    """Enhance locally periodic ridges with overlapping STFT filters.

    Each window estimates its dominant ridge frequency and orientation from the
    local Fourier spectrum.  A symmetric contextual band-pass retains energy
    around that ridge wave-vector, while overlap-add avoids block boundaries.
    The returned strength is the mean confidence of accepted STFT windows.
    """

    if block_size % 2 or stride <= 0 or stride > block_size:
        raise ValueError("STFT block size must be even and stride must overlap the block.")

    image_f = image.astype(np.float32)
    mask = region_mask > 0
    window_1d = np.hanning(block_size).astype(np.float32)
    window = np.sqrt(np.outer(window_1d, window_1d)).astype(np.float32)
    frequencies = np.fft.fftfreq(block_size).astype(np.float32)
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    radius = np.sqrt(fx * fx + fy * fy)
    ridge_band = (radius >= 1.0 / 18.0) & (radius <= 1.0 / 3.5)
    positive_half = ridge_band & ((fy > 0) | ((fy == 0) & (fx > 0)))

    accumulated = np.zeros_like(image_f)
    weights = np.zeros_like(image_f)
    strengths: list[float] = []
    height, width = image.shape

    for top in range(0, height - block_size + 1, stride):
        for left in range(0, width - block_size + 1, stride):
            block_mask = mask[top : top + block_size, left : left + block_size]
            if float(np.mean(block_mask)) < 0.45:
                continue
            block = image_f[top : top + block_size, left : left + block_size]
            centred = (block - float(np.mean(block[block_mask]))) * window
            spectrum = np.fft.fft2(centred)
            power = np.abs(spectrum) ** 2
            band_power = power[positive_half]
            if band_power.size == 0 or float(band_power.sum()) < 1e-6:
                continue

            peak_flat = int(np.argmax(np.where(positive_half, power, -1.0)))
            peak_y, peak_x = np.unravel_index(peak_flat, power.shape)
            dominant_frequency = float(radius[peak_y, peak_x])
            if not (1.0 / 18.0 <= dominant_frequency <= 1.0 / 3.5):
                continue

            unit_x = float(fx[peak_y, peak_x] / dominant_frequency)
            unit_y = float(fy[peak_y, peak_x] / dominant_frequency)
            projected = fx * unit_x + fy * unit_y
            perpendicular = np.abs(-fx * unit_y + fy * unit_x)
            radial_sigma = max(0.035, dominant_frequency * 0.32)
            angular_sigma = max(0.025, dominant_frequency * 0.24)
            contextual = np.exp(
                -((np.abs(projected) - dominant_frequency) ** 2) / (2.0 * radial_sigma**2)
                -perpendicular**2 / (2.0 * angular_sigma**2)
            )
            contextual *= ridge_band

            peak_power = float(power[peak_y, peak_x])
            concentration = peak_power / (float(band_power.mean()) + 1e-6)
            strength = float(np.clip((concentration - 1.0) / 13.0, 0.0, 1.0))
            strengths.append(strength)
            root_weight = np.power(power / (power.max() + 1e-6), 0.18)
            filtered = np.fft.ifft2(spectrum * contextual * (0.35 + 0.65 * root_weight)).real

            accumulated[top : top + block_size, left : left + block_size] += filtered.astype(np.float32) * window
            weights[top : top + block_size, left : left + block_size] += window * window

    response = np.divide(accumulated, weights, out=np.zeros_like(accumulated), where=weights > 1e-5)
    valid = mask & (weights > 1e-5)
    enhanced = np.full(image.shape, 255, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(response[valid], (2.0, 98.0))
        if high - low > 1e-6:
            scaled = np.clip((response - low) * 255.0 / (high - low), 0, 255)
            enhanced[valid] = scaled[valid].astype(np.uint8)
    return enhanced, float(np.mean(strengths)) if strengths else 0.0


def adaptive_local_mean_binarisation(
    image: np.ndarray,
    region_mask: np.ndarray,
    window_size: int = 13,
) -> np.ndarray:
    """Binarise each pixel against the paper's 13 x 13 local intensity mean."""

    local_mean = cv2.boxFilter(
        image.astype(np.float32),
        cv2.CV_32F,
        (window_size, window_size),
    )
    inside = region_mask > 0
    binary = np.full(image.shape, 255, dtype=np.uint8)
    binary[inside & (image.astype(np.float32) <= local_mean - 1.0)] = 0
    return binary


def morphological_thinning(binary: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Thin black binary ridges to one pixel, as required before ridge repair."""

    thinned_pixels = skeletonize((binary == 0) & (region_mask > 0))
    thinned = np.full(binary.shape, 255, dtype=np.uint8)
    thinned[thinned_pixels] = 0
    return thinned


def binary_ridge_postprocess(
    thinned: np.ndarray,
    region_mask: np.ndarray,
    minimum_length: int = 10,
) -> np.ndarray:
    """Remove short false ridges and close small gaps in valid ridge lines."""

    ridge_pixels = ((thinned == 0) & (region_mask > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ridge_pixels, 8)
    cleaned = np.zeros_like(ridge_pixels)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_length:
            cleaned[labels == label] = 1

    closed = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    )
    repaired = skeletonize(closed > 0)
    enhanced = np.full(thinned.shape, 255, dtype=np.uint8)
    enhanced[repaired] = 0
    return enhanced


def _quality_metrics(
    prepared: np.ndarray,
    contrast_enhanced: np.ndarray,
    denoised: np.ndarray,
    detail_enhanced: np.ndarray,
    enhanced: np.ndarray,
    region_mask: np.ndarray,
    coherence: float,
) -> dict[str, float | int]:
    """Summarise contrast, noise reduction and binary ridge continuity."""

    inside = region_mask > 0
    if not np.any(inside):
        inside = np.ones_like(region_mask, dtype=bool)
    raw_contrast = float(np.std(prepared[inside]))
    equalised_contrast = float(np.std(contrast_enhanced[inside]))
    contrast = float(np.clip(equalised_contrast / 74.0, 0, 1))
    high_frequency_before = np.abs(
        cv2.Laplacian(contrast_enhanced.astype(np.float32), cv2.CV_32F)
    )
    high_frequency_after = np.abs(
        cv2.Laplacian(denoised.astype(np.float32), cv2.CV_32F)
    )
    before_energy = float(np.mean(high_frequency_before[inside]))
    after_energy = float(np.mean(high_frequency_after[inside]))
    noise_reduction = float(
        np.clip(1.0 - after_energy / (before_energy + 1e-8), 0, 1)
    )
    ridge_pixels = ((enhanced == 0) & inside).astype(np.uint8)
    ridge_density = float(np.mean(ridge_pixels[inside]))
    ridge_balance = float(np.clip(1.0 - abs(ridge_density - 0.20) / 0.20, 0, 1))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ridge_pixels, 8)
    ridge_count = int(np.count_nonzero(ridge_pixels))
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    connectivity = float(np.clip(largest / max(ridge_count, 1), 0, 1))
    clarity = float(np.clip(0.55 * contrast + 0.45 * ridge_balance, 0, 1))
    exposure = float(np.mean((prepared[inside] > 8) & (prepared[inside] < 247)))
    laplacian_variance = float(cv2.Laplacian(prepared, cv2.CV_32F)[inside].var())
    sharpness = float(np.clip(np.log1p(laplacian_variance) / np.log(900.0), 0, 1))
    mask_coverage = float(np.mean(inside))
    contrast_gain = float(np.clip((equalised_contrast + 1) / (raw_contrast + 1), 0, 2) / 2)
    neighbour_count = cv2.filter2D(ridge_pixels, cv2.CV_16S, np.ones((3, 3), dtype=np.int16))
    neighbour_count = neighbour_count - ridge_pixels.astype(np.int16)
    safe_mask = cv2.erode(region_mask, np.ones((9, 9), dtype=np.uint8)) > 0
    ending_markers = ((ridge_pixels > 0) & (neighbour_count == 1) & safe_mask).astype(np.uint8)
    branch_markers = ((ridge_pixels > 0) & (neighbour_count >= 3) & safe_mask).astype(np.uint8)
    ending_components = max(cv2.connectedComponents(ending_markers, connectivity=8)[0] - 1, 0)
    branch_components = max(
        cv2.connectedComponents(
            cv2.dilate(branch_markers, np.ones((3, 3), dtype=np.uint8)),
            connectivity=8,
        )[0]
        - 1,
        0,
    )
    minutiae_count = int(ending_components + branch_components)
    overall = float(
        np.clip(
            0.20 * clarity
            + 0.15 * contrast
            + 0.12 * noise_reduction
            + 0.14 * ridge_balance
            + 0.09 * connectivity
            + 0.16 * coherence
            + 0.09 * sharpness
            + 0.03 * exposure
            + 0.02 * contrast_gain,
            0,
            1,
        )
    )
    return {
        "overall": overall,
        "clarity": clarity,
        "contrast": contrast,
        "noise_reduction": noise_reduction,
        "connectivity": connectivity,
        "ridge_density": ridge_density,
        "mask_coverage": mask_coverage,
        "ridge_coherence": coherence,
        "sharpness": sharpness,
        "minutiae_count": minutiae_count,
    }


def process_fingerprint(source: np.ndarray | bytes | bytearray | str | Path | BinaryIO) -> PipelineResult:
    """Run the ridge-preserving phone fingerprint enhancement pipeline."""

    started = perf_counter()
    original = source if isinstance(source, np.ndarray) else decode_image(source)
    if original.ndim == 3:
        working, colour_mask = extract_fingertip_roi(original)
    else:
        working, colour_mask = _to_uint8(original), None
    original = _to_uint8(original)
    # A fixed canvas is an input-shape requirement for database comparison,
    # not a visible enhancement algorithm.
    prepared = prepare_canvas(working)

    if colour_mask is not None:
        region_mask = _prepare_mask_canvas(colour_mask)
        region_mask = cv2.morphologyEx(
            region_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
    else:
        region_mask = generate_region_mask(prepared)
    region_mask = refine_ridge_region(prepared, region_mask)
    prepared[region_mask == 0] = 255
    contrast_enhanced = clahe_contrast_enhancement(prepared, region_mask)
    denoised = bilateral_ridge_denoising(contrast_enhanced, region_mask)
    detail_enhanced = mild_unsharp_ridge_enhancement(denoised, region_mask)
    coherence = ridge_coherence(detail_enhanced, region_mask)
    binary = adaptive_local_mean_binarisation(detail_enhanced, region_mask)
    thinned = morphological_thinning(binary, region_mask)
    enhanced = binary_ridge_postprocess(thinned, region_mask)
    quality = _quality_metrics(
        prepared,
        contrast_enhanced,
        denoised,
        detail_enhanced,
        enhanced,
        region_mask,
        coherence,
    )
    stages = [
        "CLAHE local contrast enhancement (clip limit 2.0)",
        "Bilateral edge-preserving denoising (5 x 5)",
        "Mild unsharp ridge enhancement (sigma 0.8)",
        "Adaptive local-mean binarization (13 x 13)",
        "Morphological thinning",
        "Binary ridge post-processing",
    ]

    return PipelineResult(
        original=original,
        prepared=prepared,
        region_mask=region_mask,
        contrast_enhanced=contrast_enhanced,
        denoised=denoised,
        detail_enhanced=detail_enhanced,
        binary=binary,
        thinned=thinned,
        enhanced=enhanced,
        skeleton=enhanced.copy(),
        quality=quality,
        stages=stages,
        processing_ms=(perf_counter() - started) * 1000.0,
    )


def _orb_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.15, nlevels=8, edgeThreshold=15, fastThreshold=7)
    return orb.detectAndCompute(image, None)


def _sift_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    sift = cv2.SIFT_create(
        nfeatures=2500,
        nOctaveLayers=5,
        contrastThreshold=0.008,
        edgeThreshold=16,
        sigma=1.1,
    )
    return sift.detectAndCompute(image, None)


def _matching_view(image: np.ndarray) -> np.ndarray:
    """Create a locally contrast-balanced view for reference feature matching."""

    prepared = prepare_canvas(image)
    normalised = _to_uint8(_normalise(prepared))
    lightly_smoothed = cv2.GaussianBlur(normalised, (0, 0), 0.6)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightly_smoothed)


def _geometric_features(
    query: np.ndarray,
    template: np.ndarray,
    method: str,
) -> tuple[float, int, int, int, int, np.ndarray | None]:
    """Score descriptor correspondences that agree on one plausible transform."""

    if method == "sift":
        query_keypoints, query_descriptors = _sift_features(query)
        template_keypoints, template_descriptors = _sift_features(template)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        # Ridge descriptors are less distinctive than natural-image features;
        # mutual matching plus RANSAC provides the stronger rejection guard.
        ratio_limit = 0.90
    else:
        query_keypoints, query_descriptors = _orb_features(query)
        template_keypoints, template_descriptors = _orb_features(template)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio_limit = 0.76

    if query_descriptors is None or template_descriptors is None:
        return 0.0, 0, 0, len(query_keypoints), len(template_keypoints), None

    pairs = matcher.knnMatch(query_descriptors, template_descriptors, k=2)
    forward = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio_limit * pair[1].distance
    ]
    reverse_pairs = matcher.knnMatch(template_descriptors, query_descriptors, k=2)
    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse_pairs
        if len(pair) == 2 and pair[0].distance < ratio_limit * pair[1].distance
    }
    good = [match for match in forward if reverse.get(match.trainIdx) == match.queryIdx]
    good_matches = len(good)
    if good_matches < 6:
        return (
            float(np.clip(good_matches / 32.0, 0, 0.16)),
            good_matches,
            0,
            len(query_keypoints),
            len(template_keypoints),
            None,
        )

    query_points = np.float32([query_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    template_points = np.float32([template_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        query_points,
        template_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=7.0 if method == "sift" else 4.5,
        maxIters=3500,
        confidence=0.997,
        refineIters=15,
    )
    if matrix is None or inlier_mask is None:
        return 0.0, good_matches, 0, len(query_keypoints), len(template_keypoints), None

    # Reject descriptor coincidences that imply an implausible scanner pose.
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    rotation = abs(float(np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0]))))
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    if not (0.55 <= scale <= 1.65 and rotation <= 38.0 and translation <= CANVAS_SIZE * 0.62):
        return 0.0, good_matches, 0, len(query_keypoints), len(template_keypoints), None

    inlier_flags = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_flags.sum())
    inlier_ratio = inliers / max(good_matches, 1)
    inlier_template_points = template_points.reshape(-1, 2)[inlier_flags]
    if len(inlier_template_points) >= 3:
        coverage = float(cv2.contourArea(cv2.convexHull(inlier_template_points))) / (CANVAS_SIZE**2)
    else:
        coverage = 0.0

    if method == "sift":
        # A dozen spatially consistent ridge features is meaningful for a
        # phone photograph even when repeated ridge texture lowers the raw
        # descriptor inlier ratio.
        score = float(
            np.clip(
                0.50 * min(inliers / 16.0, 1.0)
                + 0.20 * min(inlier_ratio / 0.28, 1.0)
                + 0.18 * min(coverage / 0.10, 1.0)
                + 0.12 * min(good_matches / 64.0, 1.0),
                0,
                1,
            )
        )
    else:
        score = float(
            np.clip(
                0.43 * min(inliers / 30.0, 1.0)
                + 0.27 * min(inlier_ratio / 0.48, 1.0)
                + 0.18 * min(coverage / 0.13, 1.0)
                + 0.12 * min(good_matches / 48.0, 1.0),
                0,
                1,
            )
        )
    return score, good_matches, inliers, len(query_keypoints), len(template_keypoints), matrix


def _phase_alignment_score(query: np.ndarray, template: np.ndarray) -> float:
    """Return the best phase-correlation response across modest rotations."""

    size = query.shape[0]
    centre = (size / 2.0, size / 2.0)
    template_f = cv2.GaussianBlur(template.astype(np.float32), (0, 0), 1.0)
    window = cv2.createHanningWindow((size, size), cv2.CV_32F)
    best = 0.0
    for degrees in (-12, -8, -4, 0, 4, 8, 12):
        matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
        rotated = cv2.warpAffine(
            query, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        ).astype(np.float32)
        _, response = cv2.phaseCorrelate(template_f, rotated, window)
        best = max(best, float(response))
    return float(np.clip(best, 0, 1))


def _spectral_similarity(query: np.ndarray, template: np.ndarray) -> float:
    """Compare low-frequency log spectra using cosine similarity."""

    def descriptor(image: np.ndarray) -> np.ndarray:
        small = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(small - small.mean()))))
        centre = magnitude[24:72, 24:72]
        vector = centre.ravel()
        return (vector - vector.mean()) / (vector.std() + 1e-8)

    first, second = descriptor(query), descriptor(template)
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-8))
    return float(np.clip((cosine + 1.0) / 2.0, 0, 1))


def _matching_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smoothed orientation field used only as matching evidence."""

    image_f = image.astype(np.float32)
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 5.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 5.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 5.0)
    orientation = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy + 1e-8)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-8)
    return orientation, np.clip(coherence, 0, 1)


def _aligned_structural_score(
    query: np.ndarray, template: np.ndarray, transform: np.ndarray | None
) -> float:
    """Compare aligned ridge texture and doubled-angle orientation fields."""

    if transform is None:
        return _phase_alignment_score(query, template)

    aligned = cv2.warpAffine(
        query,
        transform,
        (CANVAS_SIZE, CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    query_orientation, query_coherence = _matching_orientation(aligned)
    template_orientation, template_coherence = _matching_orientation(template)
    valid = (query_coherence > 0.12) & (template_coherence > 0.12)
    if np.any(valid):
        orientation = float(
            np.mean((np.cos(2.0 * (query_orientation[valid] - template_orientation[valid])) + 1.0) / 2.0)
        )
    else:
        orientation = 0.0

    def gradient(image: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    first, second = gradient(aligned), gradient(template)
    texture_mask = valid & ((first > np.percentile(first, 38)) | (second > np.percentile(second, 38)))
    if np.count_nonzero(texture_mask) > 100:
        a, b = first[texture_mask], second[texture_mask]
        correlation = float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std() + 1e-8))
        correlation = float(np.clip(correlation, 0, 1))
    else:
        correlation = 0.0
    phase = _phase_alignment_score(aligned, template)
    return float(np.clip(0.46 * correlation + 0.39 * orientation + 0.15 * phase, 0, 1))


def compare_fingerprints(
    query: PipelineResult,
    template_reference: np.ndarray | None,
    template_enhanced: np.ndarray | None = None,
) -> MatchEvidence:
    """Compare a query to the canonical enrolment capture and its enhancement.

    SIFT-RANSAC geometry is evaluated against the normalised uploaded reference
    because it is more tolerant of translation, rotation, scale and modest skin
    deformation between captures.  ORB on the enhanced pair remains secondary
    evidence.  Legacy records without a reference still use their enhancement.
    """

    if template_reference is None and template_enhanced is None:
        raise ValueError("A reference capture or enhanced template is required for matching.")
    used_reference = template_reference is not None
    reference = prepare_canvas(template_reference if template_reference is not None else template_enhanced)
    enhanced_template = prepare_canvas(template_enhanced if template_enhanced is not None else reference)

    query_reference_view = _matching_view(query.prepared if used_reference else query.enhanced)
    template_reference_view = _matching_view(reference)
    (
        reference_score,
        reference_matches,
        reference_inliers,
        keypoints_query,
        keypoints_template,
        transform,
    ) = _geometric_features(query_reference_view, template_reference_view, "sift")

    orb_score, good_matches, geometric_inliers, _, _, _ = _geometric_features(
        _matching_view(query.enhanced), _matching_view(enhanced_template), "orb"
    )
    structural = _aligned_structural_score(query_reference_view, template_reference_view, transform)
    spectral = _spectral_similarity(query.enhanced, enhanced_template)
    similarity = float(
        np.clip(
            0.54 * reference_score + 0.18 * orb_score + 0.20 * structural + 0.08 * spectral,
            0,
            1,
        )
    )
    # Global orientation and spectra can look similar across different fingers.
    # Require local geometric support before allowing those signals to dominate.
    strongest_inlier_count = max(reference_inliers, geometric_inliers)
    if strongest_inlier_count < 6:
        similarity *= 0.66
    elif strongest_inlier_count < 10:
        similarity *= 0.84

    return MatchEvidence(
        similarity=similarity,
        reference_score=reference_score,
        orb_score=orb_score,
        structural_score=structural,
        spectral_score=spectral,
        reference_matches=reference_matches,
        reference_inliers=reference_inliers,
        good_matches=good_matches,
        geometric_inliers=geometric_inliers,
        keypoints_query=keypoints_query,
        keypoints_template=keypoints_template,
        used_canonical_reference=used_reference,
    )


def generate_demo_fingerprint(seed: int, size: int = 340) -> np.ndarray:
    """Generate a deterministic loop/whorl sample for an explicitly labelled demo cohort."""

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx = size * 0.5 + rng.uniform(-28, 28)
    cy = size * 0.51 + rng.uniform(-20, 20)
    dx, dy = x - cx, y - cy
    radius = np.sqrt((dx * (1 + rng.uniform(-0.08, 0.08))) ** 2 + dy**2)
    angle = np.arctan2(dy, dx)
    wavelength = rng.uniform(9.2, 12.8)
    swirl = rng.uniform(0.35, 1.5) * angle + rng.uniform(-0.002, 0.002) * dx * dy
    ridges = 0.5 + 0.5 * np.cos(2 * np.pi * radius / wavelength + swirl)
    mask = np.exp(-((radius / (size * 0.46)) ** 5))
    image = 242 - 185 * ridges * mask
    image += rng.normal(0, 3.0, image.shape)
    image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0.55)
    return np.clip(image, 0, 255).astype(np.uint8)
