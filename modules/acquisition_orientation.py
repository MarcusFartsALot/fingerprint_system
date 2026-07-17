"""
Module A: Image Acquisition & Gradient-Based Ridge Orientation Estimation
--------------------------------------------------------------------------
Author (team): Member A
Original implementation supplied by teammate; refactored here into plain,
UI-independent functions so the other modules (calibration, detection,
classification, matching) can import and reuse it without depending on
Streamlit.

Pipeline stages implemented in this module:
  1. Image acquisition (decode raw bytes to a grayscale matrix)
  2. Contrast enhancement (CLAHE)
  3. Adaptive ridge binarisation (Gaussian adaptive threshold)
  4. Gradient-based ridge orientation estimation (Squared Gradient / Least
     Mean Square method, per Hong, Wan & Jain, 1998)
  5. Vector-field overlay for visual QA
"""

import cv2
import numpy as np


def load_grayscale(file_bytes: np.ndarray) -> np.ndarray:
    """
    Decode raw uploaded bytes into a single-channel grayscale matrix.

    Parameters
    ----------
    file_bytes : np.ndarray
        1-D uint8 array as produced by ``np.asarray(bytearray(file.read()))``.

    Returns
    -------
    np.ndarray
        2-D grayscale image matrix.
    """
    matrix = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if matrix is None:
        raise ValueError("Could not decode image bytes. File may be corrupted or in an unsupported format.")
    return matrix


def enhance_fingerprint_clahe(image: np.ndarray, clip_lim: float, grid_sz: int) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalisation to balance
    local intensity variances across the fingerprint surface.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(grid_sz, grid_sz))
    return clahe.apply(image)


def compute_ridge_binarisation(image: np.ndarray, blk_sz: int, c_val: int) -> np.ndarray:
    """
    Performs local Gaussian adaptive thresholding to produce sharp binary
    ridge skeletons (255 = ridge, 0 = valley, or vice versa depending on
    scanner polarity).
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    if blk_sz % 2 == 0:
        blk_sz += 1
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_sz, c_val
    )


def estimate_ridge_orientation(image: np.ndarray, w_size: int, sigma: float) -> np.ndarray:
    """
    Executes the Gradient-Based Ridge Orientation Estimation via the
    Squared Gradient Method.

    Returns
    -------
    np.ndarray
        Per-pixel smoothed orientation field (radians, range -pi/2..pi/2),
        same shape as ``image``. This field is reused directly by the
        singular-point detector (Module B) and the ridge-frequency
        calibration routine.
    """
    img_float = image.astype(np.float64)

    Gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

    Gxx = Gx ** 2
    Gyy = Gy ** 2
    Gxy = Gx * Gy

    sum_Gxx = cv2.boxFilter(Gxx, -1, (w_size, w_size), normalize=False)
    sum_Gyy = cv2.boxFilter(Gyy, -1, (w_size, w_size), normalize=False)
    sum_Gxy = cv2.boxFilter(Gxy, -1, (w_size, w_size), normalize=False)

    raw_theta = 0.5 * np.arctan2(2 * sum_Gxy, sum_Gxx - sum_Gyy)

    Phi_x = np.cos(2 * raw_theta)
    Phi_y = np.sin(2 * raw_theta)

    Phi_x_smoothed = cv2.GaussianBlur(Phi_x, (0, 0), sigma)
    Phi_y_smoothed = cv2.GaussianBlur(Phi_y, (0, 0), sigma)

    smoothed_theta = 0.5 * np.arctan2(Phi_y_smoothed, Phi_x_smoothed)
    return smoothed_theta


def generate_vector_overlay(bg_image: np.ndarray, theta_matrix: np.ndarray, w_size: int) -> np.ndarray:
    """
    Overlays calculated directional vector fields orthogonally onto the
    fingerprint skeleton for visual quality assurance.
    """
    h, w = bg_image.shape
    vis_output = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)

    for y in range(w_size // 2, h, w_size):
        for x in range(w_size // 2, w, w_size):
            angle = theta_matrix[y, x] + np.pi / 2

            length = w_size // 2
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))

            x1, y1 = x - dx, y - dy
            x2, y2 = x + dx, y + dy

            cv2.line(vis_output, (x1, y1), (x2, y2), (0, 255, 136), 1, cv2.LINE_AA)

    return vis_output


def run_acquisition_pipeline(raw_matrix: np.ndarray, clip_limit: float, tile_grid: int,
                              adaptive_block: int, adaptive_c: int,
                              block_size: int, gaussian_sigma: float) -> dict:
    """
    Convenience wrapper that runs the full Module A pipeline in one call
    and returns every intermediate result other modules need.
    """
    enhanced_contrast = enhance_fingerprint_clahe(raw_matrix, clip_limit, tile_grid)
    binary_skeleton = compute_ridge_binarisation(enhanced_contrast, adaptive_block, adaptive_c)
    theta_field = estimate_ridge_orientation(raw_matrix, block_size, gaussian_sigma)
    vector_overlay = generate_vector_overlay(binary_skeleton, theta_field, block_size)

    return {
        "raw": raw_matrix,
        "enhanced": enhanced_contrast,
        "binary": binary_skeleton,
        "theta_field": theta_field,
        "vector_overlay": vector_overlay,
    }
