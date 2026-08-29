# Filtering Methodology for the Fingerprint Attendance System

## Selected technique

The production pipeline uses **edge-preserving bilateral filtering** as its main
denoising filter, supported by Contrast Limited Adaptive Histogram Equalization
(CLAHE) and a mild unsharp mask. This is a classical image-processing and image-
enhancement pipeline; it does not use a neural network.

STFT was tested but is not used in production. On the supplied phone photographs,
its local frequency and orientation estimates were unstable because of curved
skin, glare, creases, pores and changing ridge scale. It consequently generated
coarse periodic lines that were not present in the input. A visually strong but
synthetic ridge is unsafe biometric evidence.

## Executed processing sequence

1. **Foreground extraction.** Centre-seeded colour and boundary segmentation
   isolates the fingertip. Ridge coherence refines the mask, the crop is resized
   without distortion to a `360 x 360` canvas, and pixels outside the mask are
   whitened.
2. **CLAHE (`8 x 8` tile grid, clip limit `2.0`).** Local illumination is
   corrected while the clip limit prevents unrestricted noise amplification.
3. **Bilateral filtering (`5 x 5`, colour sigma `12`, spatial sigma `3`).** This
   is the main filter. It suppresses small camera noise while weighting across
   strong intensity changes less heavily, so ridge boundaries remain sharper
   than with ordinary averaging.
4. **Mild unsharp enhancement (sigma `0.8`, amount `0.75`).** Only edges already
   present in the photograph are strengthened. No ridge frequency is invented.
5. **Adaptive local-mean binarization (`13 x 13`).** Pixels darker than their
   local mean become black ridge candidates.
6. **Morphological thinning.** Candidate ridges are reduced to one-pixel centre
   lines.
7. **Binary ridge repair.** Short isolated components are removed and small gaps
   are closed before a final skeletonization.

## Pseudocode

```text
INPUT fingerprint photograph
ROI, MASK <- segment and crop central ridge-bearing fingertip
PREPARED <- resize ROI without distortion and whiten outside MASK
CONTRAST <- CLAHE(PREPARED, tile grid 8 x 8, clip limit 2.0)
DENOISED <- bilateral filter(CONTRAST, diameter 5, sigma colour 12, sigma space 3)
DETAIL <- unsharp mask(DENOISED, sigma 0.8, amount 0.75)
BINARY <- DETAIL <= local mean(DETAIL, 13 x 13)
THINNED <- skeletonize(BINARY ridges)
TEMPLATE <- remove short false ridges, close small gaps, skeletonize
RETURN diagnostic stages, TEMPLATE and quality measures
```

## What is stored and compared

The system does not identify a person using file metadata or EXIF. Each
enrolment stores a normalized greyscale reference plus a processed binary
template. The primary attendance comparison uses local SIFT descriptors and
RANSAC geometric consistency on the greyscale references. Enhanced ORB,
structural and spectral scores provide secondary evidence.

This local matching design gives limited tolerance to a small dot, crumb or
other obstruction: features in the unobstructed overlap can still agree
geometrically. The application must not blur a large obstruction away or
pretend to reconstruct hidden ridges. A large dirty, wet, glared or blurred area
should cause a low-quality decision and request another capture.

Enrol two or three genuinely different captures of the same finger. Reusing one
file tests duplicate-image recognition, not biometric recognition.

## Evaluation

Use separate labelled enrolment and query captures. Add controlled occlusion
tests (for example 0%, 5%, 10%, 20% and 30% of the ridge region covered) and
report true accept rate, false reject rate, false accept rate and processing
time. Select the acceptance threshold using a validation split, then report the
final result on a held-out test split.

A scanner dataset is useful for proving the enhancement algorithm, but it does
not by itself validate phone photographs. Report scanner and phone experiments
as separate capture domains.

## Capture guidance and limits

Use the `1x` or macro camera, diffuse light, a plain background and a clean dry
finger. Let the fingertip fill roughly 60-80% of the guide. Enhancement cannot
recover ridge detail that was never focused or was completely hidden.

The browser camera can acquire a fresh still and automatically trigger matching,
but neither a file upload nor a camera still proves liveness. Deployment needs
continuous-video challenge-response or other anti-replay checks, consent, access
control, encryption and a retention/deletion policy. This remains an educational
prototype.

## References

- OpenCV, *CLAHE class and histogram equalization documentation*:
  https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
- OpenCV, *Smoothing Images: Bilateral Filtering*:
  https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- Jea, T.-Y., & Govindaraju, V. (2005), *A minutia-based partial fingerprint
  recognition system*, Pattern Recognition, 38(10), 1672-1684.
  https://doi.org/10.1016/j.patcog.2005.03.016
