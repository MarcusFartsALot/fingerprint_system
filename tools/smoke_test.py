"""Quick end-to-end smoke test (not part of the deliverable) — synthesises
a ridge-like test pattern and runs it through every module to catch
runtime errors before handing the project off."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.acquisition_orientation import run_acquisition_pipeline
from modules.calibration import calibrate_image
from modules.classification import classify_pattern
from modules.matching import identify
from modules.minutiae_detection import detect_minutiae, draw_minutiae_overlay
from modules.singular_points import detect_singular_points
from modules import database
from modules.report import generate_report


def synthesize_ridge_image(size=256):
    """Generates a swirling sinusoidal pattern loosely resembling a loop
    fingerprint, purely to exercise the pipeline code paths."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.5, size * 0.6
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-3
    theta = np.arctan2(dy, dx)
    swirl = theta + r * 0.05
    pattern = 128 + 100 * np.sin(swirl * 6 + r * 0.15)
    noise = np.random.normal(0, 8, (size, size))
    img = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    return img


def main():
    print("1. Synthesising test image...")
    raw = synthesize_ridge_image()

    print("2. Module A: acquisition + orientation pipeline...")
    pipeline = run_acquisition_pipeline(raw, 3.0, 8, 15, 3, 16, 3.0)
    assert pipeline["theta_field"].shape == raw.shape

    print("3. Calibration...")
    calib = calibrate_image(pipeline["raw"], pipeline["theta_field"], block_size=24)
    print(f"   estimated DPI = {calib.estimated_dpi:.1f}, scale = {calib.scale_factor:.3f}")

    print("4. Module B: singular points...")
    singular_points = detect_singular_points(pipeline["theta_field"], block_size=16)
    print(f"   found {len(singular_points)} singular points: "
          f"{[p.kind for p in singular_points]}")

    print("5. Module B: minutiae (classical)...")
    minutiae = detect_minutiae(pipeline["binary"], pipeline["theta_field"], prefer_yolo=False)
    print(f"   found {len(minutiae)} minutiae")

    print("6. Module C: classification...")
    classification = classify_pattern(singular_points)
    print(f"   pattern = {classification.pattern_type} ({classification.confidence})")

    print("7. Overlays...")
    minutiae_overlay = draw_minutiae_overlay(pipeline["binary"], minutiae)
    singular_overlay = pipeline["vector_overlay"].copy()

    print("8. Database insert + query...")
    database.init_db()
    rec_id = database.insert_record("test_subject", "T001", "data/images/test.png",
                                     classification.pattern_type, minutiae,
                                     singular_points, calib.estimated_dpi)
    print(f"   inserted record id={rec_id}")
    all_records = database.get_all_records()
    print(f"   total records now: {len(all_records)}")

    print("9. Matching / identification...")
    results = identify(minutiae, singular_points, all_records)
    print(f"   top match: {results[0].subject_name} score={results[0].score:.3f}" if results else "   no records")

    print("10. PDF report generation...")
    out_path = "/tmp/smoke_test_report.pdf"
    generate_report(out_path, "test_subject", "T001", pipeline["raw"], pipeline["enhanced"],
                     minutiae_overlay, singular_overlay, classification,
                     calib.estimated_dpi, len(minutiae), results)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    print(f"   report written to {out_path} ({os.path.getsize(out_path)} bytes)")

    database.delete_record(rec_id)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
