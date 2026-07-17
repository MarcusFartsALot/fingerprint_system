# Fingerprint Enhancement System
BMDS2133 Image Processing — Mode B (Innovative Solution Development)
Topic 3: Fingerprint Enhancement System

## 1. What this is

A complete, working Streamlit application that restores ridge flow patterns,
enhances low-quality/degraded fingerprint images, isolates ridge structures,
detects features (minutiae + singular points), classifies the fingerprint
pattern, matches/identifies against a local database, and exports PDF
reports — with batch and video ingestion for extra effort.

## 2. How the code maps to your four modules

| Member | Module | File(s) | What it does |
|---|---|---|---|
| A (teammate) | Image Acquisition & Orientation Estimation | `modules/acquisition_orientation.py` | CLAHE enhancement, adaptive binarisation, Squared-Gradient ridge orientation estimation. Refactored from the original Streamlit script into plain functions so every other module can reuse the orientation field. |
| B (you) | Feature Detection | `modules/singular_points.py`, `modules/minutiae_detection.py` | Poincare-index singular point detection (core/delta) directly from Module A's orientation field; minutiae detection via classical crossing-number method, with an optional trained-YOLO path that auto-falls-back if no weights are present. |
| C (you) | Classification & Matching | `modules/classification.py`, `modules/matching.py` | Rule-based Arch/Loop/Whorl classification from singular-point configuration (Galton-Henry system); minutiae-based matching/identification against the local database (Dice-coefficient scoring with core-aligned coordinates). |
| D (you) | User Interface | `app.py`, `modules/database.py`, `modules/report.py` | Streamlit UI (5 tabs), SQLite storage, PDF report generation. |

Also included, satisfying the **shared** "Image Calibration" requirement:
`modules/calibration.py` — estimates ridge frequency from the orientation
field and rescales images to a standard 500 DPI reference so measurements
are consistent across differently-scanned inputs.

## 3. Requirements coverage

**Shared Core Functional Requirements**
- Preprocessing (noise removal/enhancement): CLAHE + Gaussian + adaptive threshold — `acquisition_orientation.py`
- Image Calibration (spatial scaling/rectification): `calibration.py`
- Object/Feature Detection: minutiae + singular points — `minutiae_detection.py`, `singular_points.py`
- Data Analysis Dashboard: Analytics tab in `app.py`

**Extra Efforts**
- Reporting (PDF export): `modules/report.py` + "Export PDF Report" button
- Video Processing: "Video Ingestion" tab (frame extraction + per-frame analysis)
- GUI bulk ingestion: "Batch Processing" tab (multi-file upload)
- Supplemental functionality: local SQLite database + minutiae-based
  identification (1:N matching), which goes beyond plain enhancement into
  a full mini-AFIS-style prototype

## 4. Setup

```bash
pip install -r requirements.txt --break-system-packages   # or drop the flag in a venv
streamlit run app.py
```

The app works fully out of the box using the classical detectors — no
model training required to demo or submit.

## 5. Training the optional YOLO minutiae model

No public, licensed bounding-box-labelled minutiae dataset exists, so the
workflow is a classical-CV-assisted labelling pipeline:

```bash
# 1. Put 15-40 varied sample fingerprint images here:
#    data/dataset_bootstrap/images/

# 2. Auto-generate DRAFT labels using the classical detector:
python tools/bootstrap_labels.py

# 3. Manually verify/correct data/dataset_bootstrap/labels/*.txt in
#    LabelImg / makesense.ai / Roboflow (check the *_overlay.png previews
#    in data/dataset_bootstrap/overlays/ first to see what needs fixing —
#    the classical detector over-triggers on scars/creases and sometimes
#    swaps ridge_ending/bifurcation near skeleton spurs).

# 4. Split your corrected set into:
#    data/minutiae_dataset/images/{train,val}/*.png
#    data/minutiae_dataset/labels/{train,val}/*.txt

# 5. Train:
pip install ultralytics --break-system-packages
python tools/train_yolo.py --data_dir data/minutiae_dataset --epochs 100
```

The trained weights are copied automatically to `models/minutiae_yolo.pt`.
`app.py` detects this file at startup and switches to YOLO inference
(toggle available in the sidebar); if the file is absent it silently uses
the classical detector, so the app never breaks either way.

Mention this bootstrapped-labelling approach in your methodology write-up
(Part 1) — it is a legitimate, citable technique for cutting down manual
annotation effort when no labelled dataset exists, and demonstrates the
"learning new skills / complex algorithms" extra-effort criterion.

## 6. Suggested viva talking points (Programming & On-the-spot Coding, 40%)

Be ready to explain, in your own words:
- Why the Poincare index uses pi-periodic angle differences (ridge
  orientation, unlike a directed vector, is undefined mod pi, not mod 2*pi).
- Why the crossing-number method needs a skeletonised (1-pixel-wide) input.
- Why matching aligns to the core point rather than doing a full
  rotation/translation search (simplicity vs. robustness trade-off —
  a good place to acknowledge a limitation and propose future work).
- Why DPI calibration matters for matching (uncalibrated images from
  different scanners have different pixel-to-physical-distance ratios,
  which breaks the fixed distance tolerance used in matching).

## 7. Known limitations (worth stating honestly in your report)

- The matcher is a simplified point-pattern matcher, not a full
  rotation/translation-invariant minutiae matcher — good enough to
  demonstrate the concept and score reasonably on the same overall
  orientation, but not production AFIS-grade.
- Classical minutiae detection is sensitive to residual noise after
  binarisation; heavy scarring/creasing will produce spurious minutiae
  unless the YOLO model is trained on your own verified data.
- Pattern classification confidence is marked "low" whenever the
  singular-point count doesn't cleanly fit Arch/Loop/Whorl (e.g. partial
  prints) — this is intentional, to avoid silently mis-classifying
  low-quality captures.


python -m pip install --user opencv-python-headless numpy scikit-image reportlab pandas streamlit
python -m streamlit run app.py