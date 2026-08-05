"""
algorithms.py
-------------
Implements the four classical fingerprint-enhancement algorithms studied by
the project team (Mode A: Comparative & Enhancement Study), matching the
mathematical formulations in Section 2 (Literature Review) of the assignment
documentation.

    Algorithm 1 - Gradient-Based Ridge Orientation Estimation  (Ang Wei Ee)
    Algorithm 2 - Gabor Filtering                               (Lam Yi Ming)
    Algorithm 3 - Short-Time Fourier Transform (STFT)           (Marcus Kong Mun Chun)
    Algorithm 4 - Structure Tensor / Coherence-Enhancing        (Fong Jun Quan)

Every function takes a single-channel uint8 (or float) image and returns
an enhanced uint8 image (plus auxiliary maps such as orientation/coherence
where useful for visualisation).
"""

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def normalize_image(img, target_mean=100.0, target_var=100.0):
    """Hong et al. style mean/variance normalisation used as a shared
    pre-processing step ahead of every algorithm (contrast stretching)."""
    img = img.astype(np.float64)
    mean, var = np.mean(img), np.var(img)
    normed = target_mean + np.sign(img - mean) * np.sqrt(
        target_var * ((img - mean) ** 2) / (var + 1e-8)
    )
    return np.clip(normed, 0, 255)


def to_uint8(img):
    img = img.astype(np.float64)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def binarize_otsu(img):
    """Return a {0,1} ridge/valley mask via Otsu thresholding, used to score
    every enhancement algorithm against the synthetic ground truth."""
    u8 = to_uint8(img)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th > 0).astype(np.uint8)


def compute_gradients(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def get_oriented_kernel(angle_deg, length=7):
    """A thin horizontal line kernel rotated to `angle_deg`, approximating a
    directional averaging filter aligned with local ridge orientation."""
    kernel = np.zeros((length, length), dtype=np.float64)
    kernel[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1)
    rotated = cv2.warpAffine(kernel, M, (length, length))
    s = rotated.sum()
    return rotated / s if s > 1e-6 else kernel / max(kernel.sum(), 1e-6)


def oriented_smooth(img, theta, block_size=16, length=7):
    """Apply a directional smoothing kernel per block, aligned with the
    local ridge orientation theta (radians)."""
    h, w = img.shape
    out = np.copy(img).astype(np.float64)
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cy = min(by + block_size // 2, h - 1)
            cx = min(bx + block_size // 2, w - 1)
            angle = np.degrees(theta[cy, cx])
            kernel = get_oriented_kernel(angle, length)
            y2, x2 = min(by + block_size, h), min(bx + block_size, w)
            patch = out[by:y2, bx:x2]
            if patch.size == 0:
                continue
            out[by:y2, bx:x2] = cv2.filter2D(patch, -1, kernel)
    return out


def block_orientation(img, block_size=16, smooth_sigma=2.0):
    """
    Gradient-based block orientation estimation, exactly following the
    Vx/Vy double-angle formulation in Section 2.1 of the documentation:

        Vx(i,j) = sum( 2*Gx*Gy )
        Vy(i,j) = sum( Gx^2 - Gy^2 )
        theta(i,j) = 0.5 * atan2(Vx, Vy)

    followed by continuous-vector Gaussian smoothing to remove the
    double-angle discontinuities.
    """
    gx, gy = compute_gradients(img)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy

    # Block-sum aggregation implemented as a box filter (equivalent sum
    # over a non-overlapping W x W neighbourhood, evaluated densely).
    vx = cv2.boxFilter(2 * gxy, -1, (block_size, block_size), normalize=False)
    vy = cv2.boxFilter(gxx - gyy, -1, (block_size, block_size), normalize=False)
    energy = cv2.boxFilter(gxx + gyy, -1, (block_size, block_size), normalize=False)
    theta_raw = 0.5 * np.arctan2(vx, vy)

    # continuous vector smoothing (Section 2.1, "final smoothing phase")
    phix = np.cos(2 * theta_raw)
    phiy = np.sin(2 * theta_raw)
    phix_s = cv2.GaussianBlur(phix, (0, 0), smooth_sigma)
    phiy_s = cv2.GaussianBlur(phiy, (0, 0), smooth_sigma)
    theta = 0.5 * np.arctan2(phiy_s, phix_s)

    coherence = np.sqrt(vx ** 2 + vy ** 2) / (energy + 1e-6)
    return theta, coherence


def fft_dominant_frequency(block):
    """Estimate the dominant spatial frequency (cycles/pixel) of a small
    block via its 2D FFT magnitude spectrum, ignoring the DC term."""
    if block.shape[0] < 4 or block.shape[1] < 4:
        return 1.0 / 12.0
    f = np.fft.fftshift(np.fft.fft2(block - block.mean()))
    mag = np.abs(f)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    mag[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 0
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    dy, dx = idx[0] - cy, idx[1] - cx
    r = np.sqrt(dy ** 2 + dx ** 2)
    freq = r / max(block.shape)
    return float(np.clip(freq, 1 / 25, 1 / 3))


def estimate_frequency_map(img, block_size=16):
    h, w = img.shape
    freq_map = np.full((h, w), 1 / 11.0)
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            y2, x2 = min(by + block_size, h), min(bx + block_size, w)
            block = img[by:y2, bx:x2]
            freq_map[by:y2, bx:x2] = fft_dominant_frequency(block)
    return freq_map


def polar_spectrum_stats(P):
    """theta0, f0, energy extracted from a block power-spectrum P,
    following Chikkerur et al. (2006) as summarised in Section 2.3."""
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    Pn = P.copy()
    Pn[cy, cx] = 0
    total = Pn.sum() + 1e-8
    theta0 = 0.5 * np.angle(np.sum(Pn * np.exp(1j * 2 * theta)))
    f0 = float(np.sum(r * Pn) / total / max(h, w))
    energy = float(total)
    return theta0, f0, energy


# --------------------------------------------------------------------------- #
# Algorithm 1 - Gradient-Based Ridge Orientation Estimation (Ang Wei Ee)
# --------------------------------------------------------------------------- #

def algo1_gradient_orientation(img, block_size=16):
    """
    Estimates the ridge orientation field and uses it to drive a directional
    smoothing pass (oriented averaging along the local ridge direction),
    which suppresses noise while preserving ridge continuity.
    """
    norm = normalize_image(img)
    theta, coherence = block_orientation(norm, block_size=block_size)
    enhanced = oriented_smooth(norm, theta, block_size=block_size, length=7)
    return to_uint8(enhanced), {"orientation": theta, "coherence": coherence}


# --------------------------------------------------------------------------- #
# Algorithm 2 - Gabor Filtering (Lam Yi Ming)
# --------------------------------------------------------------------------- #

def gabor_kernel(freq, theta, sigma_x=4.0, sigma_y=4.0, ksize=15):
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * ((x_theta ** 2) / sigma_x ** 2 + (y_theta ** 2) / sigma_y ** 2))
    gb *= np.cos(2 * np.pi * freq * x_theta)
    return gb


def algo2_gabor(img, block_size=16):
    """
    Estimates local ridge orientation + frequency per block, then applies a
    matched Gabor filter bank (band-pass across the ridge, low-pass along
    the ridge) to enhance ridge/valley contrast, following Section 2.2.
    """
    norm = normalize_image(img)
    theta, coherence = block_orientation(norm, block_size=block_size)
    freq_map = estimate_frequency_map(norm, block_size=block_size)

    h, w = norm.shape
    enhanced = np.zeros_like(norm)
    pad = 7
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cy = min(by + block_size // 2, h - 1)
            cx = min(bx + block_size // 2, w - 1)
            f = max(freq_map[cy, cx], 1 / 20)
            t = theta[cy, cx]
            kernel = gabor_kernel(f, t)

            y2, x2 = min(by + block_size, h), min(bx + block_size, w)
            y0p, x0p = max(by - pad, 0), max(bx - pad, 0)
            y1p, x1p = min(y2 + pad, h), min(x2 + pad, w)
            patch = norm[y0p:y1p, x0p:x1p]
            filtered = cv2.filter2D(patch, -1, kernel)

            oy, ox = by - y0p, bx - x0p
            enhanced[by:y2, bx:x2] = filtered[oy:oy + (y2 - by), ox:ox + (x2 - bx)]

    return to_uint8(enhanced), {"orientation": theta, "frequency": freq_map, "coherence": coherence}


# --------------------------------------------------------------------------- #
# Algorithm 3 - Short-Time Fourier Transform (Marcus Kong Mun Chun)
# --------------------------------------------------------------------------- #

def algo3_stft(img, block_size=32, overlap=16, k=0.45):
    """
    Localised 2D STFT enhancement: overlapping Hann-windowed blocks are
    transformed to the frequency domain, the dominant spectral ring is
    boosted (spectral magnitude modification, Section 2.3), and blocks are
    reconstructed with overlap-add synthesis.
    """
    norm0 = normalize_image(img)

    # The Hann window is exactly zero at its own edges, so any pixel that
    # only ever falls on a block boundary (i.e. the image's outer border)
    # would get zero total weight no matter how blocks are placed. Padding
    # by half a block (reflected) moves every original pixel safely inside
    # some block's interior before overlap-add, then we crop back at the end.
    pad = block_size // 2
    norm = cv2.copyMakeBorder(norm0, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = norm.shape
    step = max(block_size - overlap, 4)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))

    # AC (ridge/valley texture) and DC (local brightness) are reconstructed
    # separately, each with a single analysis-window weighting, and combined
    # only at the very end. Applying the window a second time during
    # overlap-add (a common bug) corrupts the reconstruction with win^2
    # scaling and washes the image out.
    acc_ac = np.zeros((h, w))
    acc_mean = np.zeros((h, w))
    weight = np.zeros((h, w))
    orientation_map = np.zeros((h, w))
    freq_map = np.zeros((h, w))
    energy_map = np.zeros((h, w))

    def block_starts(dim):
        starts = list(range(0, max(dim - block_size, 0) + 1, step))
        last = max(dim - block_size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    ys = block_starts(h)
    xs = block_starts(w)

    for by in ys:
        for bx in xs:
            y2, x2 = min(by + block_size, h), min(bx + block_size, w)
            bh, bw = y2 - by, x2 - bx
            win = window[:bh, :bw]
            block = norm[by:y2, bx:x2]
            block_mean = block.mean()
            # remove DC before transforming so the (huge) zero-frequency bin
            # does not get exponentiated along with the ridge frequencies
            wblock = (block - block_mean) * win

            F = np.fft.fftshift(np.fft.fft2(wblock))
            P = np.abs(F) ** 2
            P_norm = P / (P.max() + 1e-8)          # scale-invariant boost factor
            F_enh = F * (P_norm ** (k / 2))
            block_rec_ac = np.real(np.fft.ifft2(np.fft.ifftshift(F_enh)))

            acc_ac[by:y2, bx:x2] += block_rec_ac
            acc_mean[by:y2, bx:x2] += block_mean * win
            weight[by:y2, bx:x2] += win

            theta0, f0, energy = polar_spectrum_stats(P)
            orientation_map[by:y2, bx:x2] = theta0
            freq_map[by:y2, bx:x2] = f0
            energy_map[by:y2, bx:x2] = energy

    weight[weight < 1e-6] = 1e-6
    enhanced = (acc_ac + acc_mean) / weight

    # crop back off the padding
    enhanced = enhanced[pad:pad + norm0.shape[0], pad:pad + norm0.shape[1]]
    orientation_map = orientation_map[pad:pad + norm0.shape[0], pad:pad + norm0.shape[1]]
    freq_map = freq_map[pad:pad + norm0.shape[0], pad:pad + norm0.shape[1]]
    energy_map = energy_map[pad:pad + norm0.shape[0], pad:pad + norm0.shape[1]]

    return to_uint8(enhanced), {"orientation": orientation_map, "frequency": freq_map, "energy": energy_map}


# --------------------------------------------------------------------------- #
# Algorithm 4 - Structure Tensor (Fong Jun Quan)
# --------------------------------------------------------------------------- #

def algo4_structure_tensor(img, sigma=2.0, iterations=6, step=0.35, block_size=16):
    """
    Computes the 2x2 structure tensor J, its eigen-decomposition (coherence
    C) and drives a simplified Coherence-Enhancing Diffusion: iteratively
    blends each pixel toward an orientation-aligned smoothed version,
    weighted by local coherence, following Section 2.4.
    """
    norm = normalize_image(img)
    gx, gy = compute_gradients(norm)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)

    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lambda1 = 0.5 * (Jxx + Jyy + tmp)
    lambda2 = 0.5 * (Jxx + Jyy - tmp)
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)) ** 2
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)  # ridge tangent direction (v2)

    enhanced = norm.copy()
    for _ in range(iterations):
        smoothed = oriented_smooth(enhanced, theta, block_size=block_size, length=5)
        enhanced = enhanced * (1 - step * coherence) + smoothed * (step * coherence)

    return to_uint8(enhanced), {"orientation": theta, "coherence": coherence}


ALGORITHMS = {
    "Algo 1: Gradient-Based Orientation (Ang Wei Ee)": algo1_gradient_orientation,
    "Algo 2: Gabor Filtering (Lam Yi Ming)": algo2_gabor,
    "Algo 3: STFT (Marcus Kong Mun Chun)": algo3_stft,
    "Algo 4: Structure Tensor (Fong Jun Quan)": algo4_structure_tensor,
}