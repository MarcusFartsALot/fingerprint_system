"""
Bootstrap YOLO Labels for Minutiae Detection
=================================================
Since no public, licensed dataset of bounding-box-labelled fingerprint
minutiae was available, this script generates a DRAFT labelled dataset
for you to manually verify and correct, rather than starting from a
blank page.

It runs the classical crossing-number detector (Module B) over every
image in an input folder and writes out:
  - a YOLO-format ``.txt`` annotation file per image (class 0 = ridge_ending,
    class 1 = bifurcation), with a small fixed-size box around each point
  - an ``_overlay.png`` preview so you can quickly visually sanity-check
    the draft boxes before opening them in a labelling tool

WORKFLOW
--------
1. Put 15-40 varied fingerprint images in ``data/dataset_bootstrap/images/``.
2. Run:  python tools/bootstrap_labels.py
3. Open the generated ``labels/`` folder in a labelling tool such as
   LabelImg, makesense.ai, or Roboflow (import as YOLO format) and:
     - delete false-positive boxes (classical CN detector over-triggers
       near scars, creases and skeleton spurs)
     - add any minutiae the classical detector missed
     - fix any ridge_ending / bifurcation class swaps
4. Split the corrected set into train/val (e.g. 80/20).
5. Run tools/train_yolo.py to fine-tune a YOLOv8n model on your corrected
   labels.

This "classical-CV-assisted labelling" approach is itself a legitimate,
citable technique (a lightweight form of pre-annotation / human-in-the-loop
labelling used widely to cut down manual annotation time) and is worth
mentioning in your methodology write-up as part of your innovation effort.
"""

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.acquisition_orientation import (enhance_fingerprint_clahe,  # noqa: E402
                                              compute_ridge_binarisation,
                                              estimate_ridge_orientation)
from modules.minutiae_detection import (detect_minutiae_classical,  # noqa: E402
                                         draw_minutiae_overlay)

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_bootstrap", "images")
LABEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_bootstrap", "labels")
OVERLAY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_bootstrap", "overlays")

BOX_SIZE_PX = 14  # fixed-size draft box around each detected minutia
CLASS_IDS = {"ridge_ending": 0, "bifurcation": 1}


def process_image(image_path: str) -> None:
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"  [skip] could not read {image_path}")
        return

    enhanced = enhance_fingerprint_clahe(raw, 3.0, 8)
    binary = compute_ridge_binarisation(enhanced, 15, 3)
    theta = estimate_ridge_orientation(raw, 16, 3.0)

    minutiae = detect_minutiae_classical(binary, theta)

    h, w = raw.shape
    stem = os.path.splitext(os.path.basename(image_path))[0]

    label_lines = []
    for m in minutiae:
        cls = CLASS_IDS[m.kind]
        cx, cy = m.x / w, m.y / h
        bw, bh = BOX_SIZE_PX / w, BOX_SIZE_PX / h
        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(os.path.join(LABEL_DIR, f"{stem}.txt"), "w") as f:
        f.write("\n".join(label_lines))

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    overlay = draw_minutiae_overlay(binary, minutiae)
    cv2.imwrite(os.path.join(OVERLAY_DIR, f"{stem}_overlay.png"), overlay)

    print(f"  [ok] {stem}: {len(minutiae)} draft minutiae labelled")


def main():
    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.bmp")) +
        glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    )

    if not images:
        print(f"No images found in {INPUT_DIR}.")
        print("Add 15-40 sample fingerprint images there first, then re-run this script.")
        return

    print(f"Found {len(images)} images. Generating draft YOLO labels...")
    for path in images:
        process_image(path)

    print(f"\nDone. Review/correct the draft boxes in: {LABEL_DIR}")
    print(f"Quick visual sanity-check overlays saved to: {OVERLAY_DIR}")
    print("Next step: manually verify in a labelling tool, then run tools/train_yolo.py")


if __name__ == "__main__":
    main()
