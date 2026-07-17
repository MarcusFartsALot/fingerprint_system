"""
Module B: Feature Detection — Minutiae (Ridge Endings & Bifurcations)
------------------------------------------------------------------------
Author (team): Member B

Two detection back-ends are provided behind one interface, ``detect_minutiae``:

  1. Classical Crossing-Number (CN) method (Palacios et al.) applied to
     a skeletonised ridge image. This is the *default* and *always
     available* detector — no dataset or training required, and it is
     the reference implementation used to sanity-check the YOLO model.

  2. Optional YOLO detector (Ultralytics). Because no public, licensed
     minutiae-bounding-box dataset was available, the intended workflow
     (see tools/bootstrap_labels.py + tools/train_yolo.py) is:

        classical CN detector on your own sample images
              -> auto-generates DRAFT YOLO labels
              -> you manually verify/correct the boxes in a labelling
                 tool (LabelImg / makesense.ai / Roboflow)
              -> tools/train_yolo.py fine-tunes a small YOLOv8 model
                 on your verified dataset
              -> drop the resulting weights at models/minutiae_yolo.pt

     If that file exists, ``detect_minutiae(..., prefer_yolo=True)`` uses
     it; otherwise it transparently falls back to the classical detector,
     so the app is always fully functional even before a model is trained.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "minutiae_yolo.pt")
YOLO_CLASS_NAMES = {0: "ridge_ending", 1: "bifurcation"}


@dataclass
class Minutia:
    x: int
    y: int
    angle: float          # local ridge orientation at the minutia, radians
    kind: str             # "ridge_ending" or "bifurcation"
    confidence: float = 1.0
    source: str = "classical"


# ---------------------------------------------------------------------
# 1. Classical crossing-number detector
# ---------------------------------------------------------------------

def _skeletonize(binary_ridges: np.ndarray) -> np.ndarray:
    """Reduces the binary ridge map to a 1-pixel-wide skeleton."""
    bool_img = (binary_ridges > 0)
    if _HAS_SKIMAGE:
        skel = skeletonize(bool_img)
        return skel.astype(np.uint8)
    # Fallback: OpenCV morphological thinning approximation
    thinned = np.zeros_like(binary_ridges)
    img = binary_ridges.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        subset = cv2.subtract(img, opened)
        thinned = cv2.bitwise_or(thinned, subset)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (thinned > 0).astype(np.uint8)


def _crossing_number(neighbourhood: np.ndarray) -> int:
    """Crossing number = 0.5 * sum |P(i) - P(i+1)| over the 8-neighbour ring."""
    ring = [neighbourhood[0, 0], neighbourhood[0, 1], neighbourhood[0, 2],
            neighbourhood[1, 2], neighbourhood[2, 2], neighbourhood[2, 1],
            neighbourhood[2, 0], neighbourhood[1, 0], neighbourhood[0, 0]]
    return int(0.5 * sum(abs(int(ring[i]) - int(ring[i + 1])) for i in range(8)))


def detect_minutiae_classical(binary_ridges: np.ndarray, theta_field: np.ndarray,
                               margin_ratio: float = 0.06) -> List[Minutia]:
    """
    Skeletonises the ridge image and scans every ridge pixel's 3x3
    neighbourhood, flagging:
        CN == 1  -> ridge ending
        CN == 3  -> bifurcation
    Border pixels are excluded (they are almost always skeleton
    artefacts from the finger/background boundary, not true minutiae).
    """
    skeleton = _skeletonize(binary_ridges)
    h, w = skeleton.shape
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    minutiae: List[Minutia] = []
    for y in range(margin_y + 1, h - margin_y - 1):
        for x in range(margin_x + 1, w - margin_x - 1):
            if skeleton[y, x] == 0:
                continue
            neighbourhood = skeleton[y - 1:y + 2, x - 1:x + 2]
            if neighbourhood.sum() < 2:
                continue
            cn = _crossing_number(neighbourhood)

            if cn == 1:
                kind = "ridge_ending"
            elif cn == 3:
                kind = "bifurcation"
            else:
                continue

            angle = float(theta_field[y, x])
            minutiae.append(Minutia(x=x, y=y, angle=angle, kind=kind, source="classical"))

    return _suppress_duplicates(minutiae, min_dist=8)


def _suppress_duplicates(minutiae: List[Minutia], min_dist: int) -> List[Minutia]:
    """Simple non-maximum suppression: merges minutiae of the same kind
    that fall within ``min_dist`` pixels of each other (skeleton spurs
    often produce several adjacent CN hits for one physical feature)."""
    kept: List[Minutia] = []
    for m in minutiae:
        duplicate = False
        for k in kept:
            if k.kind == m.kind and abs(k.x - m.x) < min_dist and abs(k.y - m.y) < min_dist:
                duplicate = True
                break
        if not duplicate:
            kept.append(m)
    return kept


# ---------------------------------------------------------------------
# 2. Optional YOLO detector (used once you have trained weights)
# ---------------------------------------------------------------------

def yolo_model_available() -> bool:
    return os.path.exists(YOLO_MODEL_PATH)


def detect_minutiae_yolo(image_bgr: np.ndarray, theta_field: np.ndarray,
                          conf: float = 0.35) -> Optional[List[Minutia]]:
    """
    Runs a trained YOLOv8 model over the image and converts each detected
    bounding box centre into a Minutia, sampling the local ridge
    orientation field for the minutia angle. Returns None if no trained
    weights are present (caller should fall back to classical).
    """
    if not yolo_model_available():
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model = YOLO(YOLO_MODEL_PATH)
    results = model.predict(image_bgr, conf=conf, verbose=False)

    minutiae: List[Minutia] = []
    h, w = theta_field.shape
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            kind = YOLO_CLASS_NAMES.get(cls_id, "ridge_ending")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
            angle = float(theta_field[cy, cx])
            confidence = float(box.conf[0])
            minutiae.append(Minutia(x=cx, y=cy, angle=angle, kind=kind,
                                     confidence=confidence, source="yolo"))
    return minutiae


# ---------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------

def detect_minutiae(binary_ridges: np.ndarray, theta_field: np.ndarray,
                     image_bgr: Optional[np.ndarray] = None,
                     prefer_yolo: bool = True) -> List[Minutia]:
    """
    Single call used by the UI/pipeline. Tries the trained YOLO model
    first (if available and requested); transparently falls back to the
    classical crossing-number detector otherwise.
    """
    if prefer_yolo and image_bgr is not None:
        yolo_result = detect_minutiae_yolo(image_bgr, theta_field)
        if yolo_result is not None:
            return yolo_result

    return detect_minutiae_classical(binary_ridges, theta_field)


def draw_minutiae_overlay(bg_image: np.ndarray, minutiae: List[Minutia]) -> np.ndarray:
    """Renders minutiae on top of an image: green circles for ridge
    endings, red squares for bifurcations, with a short orientation tick."""
    if len(bg_image.shape) == 2:
        vis = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = bg_image.copy()

    for m in minutiae:
        colour = (0, 220, 0) if m.kind == "ridge_ending" else (0, 0, 230)
        if m.kind == "ridge_ending":
            cv2.circle(vis, (m.x, m.y), 4, colour, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (m.x - 4, m.y - 4), (m.x + 4, m.y + 4), colour, 1, cv2.LINE_AA)
        tick_len = 10
        tx = int(m.x + tick_len * np.cos(m.angle))
        ty = int(m.y + tick_len * np.sin(m.angle))
        cv2.line(vis, (m.x, m.y), (tx, ty), colour, 1, cv2.LINE_AA)

    return vis
