# Filtering Methodology for the Fingerprint Attendance System

## Selected technique

This system implements the binarization-based fingerprint enhancement method
presented by Greenberg et al. The method was selected because it produces a
clean binary ridge map through a reproducible sequence of local contrast
enhancement, adaptive noise removal, local thresholding and morphology. It does
not require ridge-frequency estimation and is therefore straightforward to
explain, inspect and execute in a live Streamlit attendance application.

The paper also discusses a direct greyscale anisotropic method and compares it
with modified Gabor filtering. Those methods are comparison alternatives, not
stages mixed into the selected production pipeline.

## Processing sequence

### 1. Local histogram equalization

For every pixel, the grey value is remapped using the intensity distribution in
an `11 x 11` neighbourhood. This expands weak local ridge/valley contrast more
effectively than one global histogram operation when illumination or sensor
pressure varies across the fingertip.

### 2. Adaptive Wiener filtering

A pixel-wise Wiener filter uses a `3 x 3` neighbourhood to estimate the local
mean and variance. Smoother regions receive stronger noise suppression, while
high-variance ridge detail is retained. The system estimates image noise from
the median local variance inside the detected fingerprint foreground.

### 3. Adaptive local-mean binarization

Each filtered pixel is compared with the mean intensity in its `13 x 13`
neighbourhood. Pixels darker than the local mean become black ridge pixels;
other pixels become white valleys/background. A local threshold handles gradual
brightness changes better than one threshold for the full image.

### 4. Morphological thinning

The black binary ridges are skeletonized to one-pixel-wide centre lines. This
creates a consistent ridge representation for morphology and diagnostic review.

### 5. Binary ridge post-processing

Connected ridge fragments shorter than 10 pixels are treated as false ridges and
removed. A small cross-shaped morphological closing joins minor discontinuities,
after which the result is skeletonized again to preserve one-pixel ridge width.

## Explicit system extension

Before the five paper stages, a local-variance foreground guard rejects blank
background and scanner/page borders. It is an engineering safeguard for uploads,
not an extra filtering stage attributed to the paper. Resizing and padding also
standardize array dimensions for comparison without being described as image
enhancement.

## Pseudocode

```text
INPUT greyscale fingerprint image
PREPARE a fixed comparison canvas without geometric distortion
DETECT fingerprint foreground using local variance (system extension)

EQUALIZED <- local histogram equalization(INPUT, window=11 x 11)
FILTERED  <- adaptive Wiener filter(EQUALIZED, window=3 x 3)
BINARY    <- FILTERED <= local mean(FILTERED, window=13 x 13)
THINNED   <- skeletonize(BINARY ridges)
ENHANCED  <- remove ridge components shorter than 10 pixels
ENHANCED  <- close small gaps and skeletonize again

RETURN EQUALIZED, FILTERED, BINARY, THINNED, ENHANCED
```

## Separation from identity matching

Enhancement improves visibility but does not itself decide student identity. A
new attendance scan is compared with the original greyscale enrolment reference
using SIFT-RANSAC geometry. ORB, aligned structural and spectral comparisons are
secondary evidence. Acceptance requires a score above the configured threshold
and a sufficient lead over the next-best student. This separation allows two
different captures of the same finger to match without demanding pixel equality.

## Evaluation plan

Quantitative claims should be generated from a labelled dataset rather than
copied from a previous experiment. Use separate enrolment and query captures for
each finger, then report:

- true accept rate and false reject rate for genuine pairs;
- false accept rate for different-finger pairs;
- precision, recall, F1 and accuracy at the selected decision threshold;
- average processing time per image;
- image measures before and after filtering, such as local contrast, ridge
  continuity and removed high-frequency noise;
- representative successful, rejected and failure-case images.

Threshold selection must be performed on validation data and evaluated on a
held-out test set. The current UI quality percentages are diagnostic image
measures, not biometric accuracy or a forensic certification.

## Limitations and critical discussion

The method can remove genuine short ridge fragments along with noise, and local
thresholding may create broken ridges in extremely dry, wet or blurred captures.
The variance foreground guard is tuned for scanner-style images and may require
adjustment for another sensor. SIFT-based identification is tolerant of modest
rotation and displacement but is not a replacement for a certified minutiae
matcher or liveness detection. These limitations should be stated alongside the
experimental results.

## Reference

Greenberg, S., Aladjem, M., Kogan, D., & Dimitrov, I. (2000). Fingerprint image
enhancement using filtering techniques. *Proceedings of the 15th International
Conference on Pattern Recognition*. Extended journal version:
https://doi.org/10.1006/rtim.2001.0283.
