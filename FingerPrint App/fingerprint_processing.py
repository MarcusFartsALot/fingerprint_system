"""Classical fingerprint enhancement and matching utilities.

The production enhancement engine follows Greenberg et al.'s filtering and
binarisation approach: 11 x 11 local histogram equalisation, 3 x 3 adaptive
Wiener filtering, 13 x 13 local-mean binarisation, thinning and binary ridge
post-processing. A variance-based foreground guard is an explicit system
extension for rejecting scanner frames before the paper's filtering stages.
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
from skimage.filters import rank
from skimage.morphology import skeletonize


CANVAS_SIZE = 360
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PIPELINE_SCHEMA_VERSION = "greenberg-filtering-v4"


@dataclass
class PipelineResult:
    """Images, feature maps and quality measurements from one capture."""

    original: np.ndarray
    prepared: np.ndarray
    region_mask: np.ndarray
    local_equalised: np.ndarray
    wiener_filtered: np.ndarray
    binary: np.ndarray
    thinned: np.ndarray
    enhanced: np.ndarray
    skeleton: np.ndarray
    quality: dict[str, float]
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
    """Decode an uploaded image or path as an 8-bit greyscale array."""

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
        image = Image.open(BytesIO(data)).convert("L")
        array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:  # Pillow provides format-specific exception types.
        raise ValueError("The uploaded file is not a readable fingerprint image.") from exc

    if array.size == 0 or min(array.shape) < 40:
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


def local_histogram_equalisation(
    image: np.ndarray,
    region_mask: np.ndarray,
    window_size: int = 11,
) -> np.ndarray:
    """Expand local contrast with the paper's 11 x 11 histogram window."""

    footprint = np.ones((window_size, window_size), dtype=np.uint8)
    equalised = rank.equalize(image, footprint=footprint)
    equalised[region_mask == 0] = 255
    return equalised.astype(np.uint8)


def adaptive_wiener_filter(
    image: np.ndarray,
    region_mask: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Apply the paper's pixel-wise adaptive 3 x 3 Wiener noise reduction."""

    image_f = image.astype(np.float32)
    local_mean = cv2.boxFilter(image_f, cv2.CV_32F, (window_size, window_size))
    local_square_mean = cv2.boxFilter(
        image_f * image_f,
        cv2.CV_32F,
        (window_size, window_size),
    )
    local_variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    inside = region_mask > 0
    noise_variance = (
        float(np.median(local_variance[inside])) if np.any(inside) else float(np.median(local_variance))
    )
    gain = np.maximum(local_variance - noise_variance, 0.0) / (local_variance + 1e-6)
    filtered = local_mean + gain * (image_f - local_mean)
    filtered[~inside] = 255
    return np.clip(filtered, 0, 255).astype(np.uint8)


def generate_region_mask(
    image: np.ndarray,
) -> np.ndarray:
    """Locate the connected fingerprint texture and reject borders/background.

    A pixel-by-pixel local variance threshold creates the initial foreground,
    after which connected-component geometry rejects thin scanner/page frames.
    """

    image_f = image.astype(np.float32)
    local_mean = cv2.boxFilter(image_f, cv2.CV_32F, (17, 17))
    local_square_mean = cv2.boxFilter(image_f * image_f, cv2.CV_32F, (17, 17))
    local_deviation = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    deviation_threshold = max(10.0, float(np.percentile(local_deviation, 58)))
    candidate = (local_deviation >= deviation_threshold).astype(np.uint8)

    # Scanner/page frames occur close to the canvas border and otherwise have
    # produce deceptively strong local variance.
    margin = 55
    candidate[:margin, :] = 0
    candidate[-margin:, :] = 0
    candidate[:, :margin] = 0
    candidate[:, -margin:] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.ones_like(image, dtype=np.uint8) * 255

    best_label = 0
    best_score = 0.0
    best_area = 0
    height, width = image.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        box_area = max(component_width * component_height, 1)
        density = area / box_area
        frame_like = (
            component_width > width * 0.78
            and component_height > height * 0.78
            and density < 0.42
        )
        score = area * min(density / 0.32, 1.0) ** 2
        if frame_like or x <= margin // 2 or y <= margin // 2:
            score *= 0.04
        if score > best_score:
            best_label, best_score, best_area = label, score, area

    component = (labels == best_label).astype(np.uint8)
    if best_label == 0 or best_area < int(image.size * 0.025):
        return np.ones_like(image, dtype=np.uint8) * 255

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
    local_equalised: np.ndarray,
    wiener_filtered: np.ndarray,
    enhanced: np.ndarray,
    region_mask: np.ndarray,
) -> dict[str, float]:
    """Summarise contrast, noise reduction and binary ridge continuity."""

    inside = region_mask > 0
    if not np.any(inside):
        inside = np.ones_like(region_mask, dtype=bool)
    raw_contrast = float(np.std(prepared[inside]))
    equalised_contrast = float(np.std(local_equalised[inside]))
    contrast = float(np.clip(equalised_contrast / 74.0, 0, 1))
    high_frequency_before = np.abs(
        cv2.Laplacian(local_equalised.astype(np.float32), cv2.CV_32F)
    )
    high_frequency_after = np.abs(
        cv2.Laplacian(wiener_filtered.astype(np.float32), cv2.CV_32F)
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
    mask_coverage = float(np.mean(inside))
    contrast_gain = float(np.clip((equalised_contrast + 1) / (raw_contrast + 1), 0, 2) / 2)
    overall = float(
        np.clip(
            0.28 * clarity
            + 0.22 * contrast
            + 0.18 * noise_reduction
            + 0.14 * ridge_balance
            + 0.10 * connectivity
            + 0.05 * exposure
            + 0.03 * contrast_gain,
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
    }


def process_fingerprint(source: np.ndarray | bytes | bytearray | str | Path | BinaryIO) -> PipelineResult:
    """Run the paper-aligned filtering and binarisation pipeline."""

    started = perf_counter()
    original = source if isinstance(source, np.ndarray) else decode_image(source)
    if original.ndim == 3:
        original = cv2.cvtColor(_to_uint8(original), cv2.COLOR_BGR2GRAY)
    original = _to_uint8(original)
    # A fixed canvas is an input-shape requirement for database comparison,
    # not a visible enhancement algorithm.
    prepared = prepare_canvas(original)

    region_mask = generate_region_mask(prepared)
    local_equalised = local_histogram_equalisation(prepared, region_mask)
    wiener_filtered = adaptive_wiener_filter(local_equalised, region_mask)
    binary = adaptive_local_mean_binarisation(wiener_filtered, region_mask)
    thinned = morphological_thinning(binary, region_mask)
    enhanced = binary_ridge_postprocess(thinned, region_mask)
    quality = _quality_metrics(
        prepared,
        local_equalised,
        wiener_filtered,
        enhanced,
        region_mask,
    )
    stages = [
        "Local histogram equalization (11 x 11)",
        "Adaptive Wiener filtering (3 x 3)",
        "Adaptive local-mean binarization (13 x 13)",
        "Morphological thinning",
        "Binary ridge post-processing",
    ]

    return PipelineResult(
        original=original,
        prepared=prepared,
        region_mask=region_mask,
        local_equalised=local_equalised,
        wiener_filtered=wiener_filtered,
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
        ratio_limit = 0.80
    else:
        query_keypoints, query_descriptors = _orb_features(query)
        template_keypoints, template_descriptors = _orb_features(template)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio_limit = 0.72

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
        ransacReprojThreshold=4.5 if method == "sift" else 4.0,
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
    if not (0.68 <= scale <= 1.42 and rotation <= 28.0 and translation <= CANVAS_SIZE * 0.48):
        return 0.0, good_matches, 0, len(query_keypoints), len(template_keypoints), None

    inlier_flags = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_flags.sum())
    inlier_ratio = inliers / max(good_matches, 1)
    inlier_template_points = template_points.reshape(-1, 2)[inlier_flags]
    if len(inlier_template_points) >= 3:
        coverage = float(cv2.contourArea(cv2.convexHull(inlier_template_points))) / (CANVAS_SIZE**2)
    else:
        coverage = 0.0

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
