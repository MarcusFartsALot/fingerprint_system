"""
Extra Effort: Automated PDF Reporting
------------------------------------------
Generates a one-page-per-image findings report (input, enhanced, minutiae
and singular-point overlays, calibration metrics, classification result,
and — if requested — identification results against the local database)
using reportlab, per Anthropic's PDF-creation guidance (Platypus
flowables over raw canvas drawing for anything with mixed text+images).
"""

import io
from typing import List, Optional

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                 Table, TableStyle)

from modules.classification import ClassificationResult
from modules.matching import MatchResult


def _cv2_to_reportlab_image(image: np.ndarray, width_cm: float) -> Image:
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    bio = io.BytesIO(buf.tobytes())
    h, w = image.shape[:2]
    aspect = h / w
    return Image(bio, width=width_cm * cm, height=width_cm * cm * aspect)


def generate_report(output_path: str, subject_name: str, subject_id: str,
                     raw_image: np.ndarray, enhanced_image: np.ndarray,
                     minutiae_overlay: np.ndarray, singular_overlay: np.ndarray,
                     classification: ClassificationResult, estimated_dpi: float,
                     num_minutiae: int, match_results: Optional[List[MatchResult]] = None) -> str:
    """Builds the PDF and writes it to ``output_path``; returns that path."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph("Fingerprint Enhancement System — Analysis Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Subject: {subject_name or 'N/A'} &nbsp;&nbsp; ID: {subject_id or 'N/A'}",
        styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Image Calibration", styles["Heading2"]))
    story.append(Paragraph(f"Estimated scanning resolution: {estimated_dpi:.1f} DPI "
                            "(derived from measured ridge frequency and rescaled to a "
                            "500 DPI reference frame for consistent measurement).",
                            styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Enhancement Pipeline", styles["Heading2"]))
    img_table = Table([
        [_cv2_to_reportlab_image(raw_image, 7), _cv2_to_reportlab_image(enhanced_image, 7)],
        [Paragraph("Raw input", styles["Normal"]), Paragraph("CLAHE + adaptive binarisation", styles["Normal"])],
    ])
    img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Feature Detection", styles["Heading2"]))
    feat_table = Table([
        [_cv2_to_reportlab_image(minutiae_overlay, 7), _cv2_to_reportlab_image(singular_overlay, 7)],
        [Paragraph(f"Minutiae ({num_minutiae} detected)", styles["Normal"]),
         Paragraph("Singular points (core/delta)", styles["Normal"])],
    ])
    feat_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Pattern Classification", styles["Heading2"]))
    story.append(Paragraph(
        f"<b>Result: {classification.pattern_type}</b> "
        f"(confidence: {classification.confidence}) — "
        f"{classification.num_cores} core(s), {classification.num_deltas} delta(s).",
        styles["Normal"]))
    story.append(Paragraph(classification.explanation, styles["Normal"]))
    story.append(Spacer(1, 10))

    if match_results:
        story.append(Paragraph("5. Identification Against Local Database", styles["Heading2"]))
        rows = [["Rank", "Subject", "Score", "Matched Pairs", "Decision"]]
        for i, r in enumerate(match_results[:5], start=1):
            rows.append([str(i), r.subject_name, f"{r.score:.3f}", str(r.matched_pairs),
                         "MATCH" if r.is_match else "no match"])
        result_table = Table(rows, hAlign="LEFT")
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f2f2f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(result_table)

    doc.build(story)
    return output_path
