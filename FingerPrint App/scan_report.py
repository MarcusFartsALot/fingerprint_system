"""PDF report for the analytics generated during one attendance scan."""

from __future__ import annotations

import io
from datetime import datetime
from html import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BLUE = colors.HexColor("#1f77b4")
LIGHT_BLUE = colors.HexColor("#7fc2f2")
DARK_BLUE = colors.HexColor("#1f4e79")
GRID = colors.HexColor("#d8dee6")


def _page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(1.4 * cm, 1.05 * cm, A4[0] - 1.4 * cm, 1.05 * cm)
    canvas.setFillColor(colors.HexColor("#5f6b78"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.4 * cm, 0.72 * cm, "Attendance scan analytics")
    canvas.drawRightString(A4[0] - 1.4 * cm, 0.72 * cm, f"Page {document.page}")
    canvas.restoreState()


def _metric_chart(metrics: dict[str, float]) -> Drawing:
    drawing = Drawing(500, 250)
    left, bottom, width, height = 48, 44, 420, 170
    for step in range(6):
        value = step / 5
        y = bottom + height * value
        drawing.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=0.5))
        drawing.add(String(left - 8, y - 3, f"{value:.1f}", textAnchor="end", fontSize=7))

    names = list(metrics)
    slot = width / max(len(names), 1)
    bar_width = slot * 0.58
    for index, name in enumerate(names):
        value = max(0.0, min(float(metrics[name]), 1.0))
        x = left + index * slot + (slot - bar_width) / 2
        drawing.add(Rect(x, bottom, bar_width, height * value, fillColor=LIGHT_BLUE, strokeColor=None))
        drawing.add(String(x + bar_width / 2, bottom - 15, name, textAnchor="middle", fontSize=8))
        drawing.add(String(x + bar_width / 2, bottom + height * value + 5, f"{value:.3f}", textAnchor="middle", fontSize=8))
    drawing.add(String(250, 232, "Classification Metrics", textAnchor="middle", fontSize=12))
    return drawing


def _histogram_chart(analytics: dict[str, object]) -> Drawing:
    drawing = Drawing(500, 255)
    left, bottom, width, height = 48, 48, 420, 155
    prepared = list(analytics["prepared_histogram"])
    enhanced = list(analytics["enhanced_histogram"])
    centres = list(analytics["histogram_centres"])
    maximum = max(prepared + enhanced + [1])
    slot = width / max(len(centres), 1)
    bar_width = max(slot * 0.72, 1.0)

    for step in range(5):
        value = maximum * step / 4
        y = bottom + height * step / 4
        drawing.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=0.5))
        drawing.add(String(left - 6, y - 3, f"{int(value):,}", textAnchor="end", fontSize=6.5))

    for index, (original, binary) in enumerate(zip(prepared, enhanced)):
        x = left + index * slot
        original_height = height * float(original) / maximum
        binary_height = height * float(binary) / maximum
        drawing.add(Rect(x, bottom, bar_width, original_height, fillColor=BLUE, strokeColor=None))
        drawing.add(Rect(x, bottom + original_height, bar_width, binary_height, fillColor=LIGHT_BLUE, strokeColor=None))
        if index % 4 == 0:
            drawing.add(String(x + bar_width / 2, bottom - 13, str(centres[index]), textAnchor="middle", fontSize=6))

    drawing.add(String(250, 232, "Pixel Intensity Histogram (Original vs Enhanced)", textAnchor="middle", fontSize=12))
    drawing.add(Rect(145, 15, 8, 8, fillColor=BLUE, strokeColor=None))
    drawing.add(String(158, 16, "Original (Prepared)", fontSize=7))
    drawing.add(Rect(275, 15, 8, 8, fillColor=LIGHT_BLUE, strokeColor=None))
    drawing.add(String(288, 16, "Enhanced (Binary)", fontSize=7))
    return drawing


def build_scan_analytics_pdf(
    analytics: dict[str, object],
    configuration: dict[str, object],
    elapsed_ms: float,
) -> bytes:
    """Build a downloadable report matching the teammate benchmark format."""

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Student's Attendance Tracking Biometric System", styles["Title"]),
        Paragraph("Attendance Marking Report", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("1. Test Configuration", styles["Heading2"]),
        Paragraph(
            "<br/>".join(f"<b>{escape(str(key))}:</b> {escape(str(value))}" for key, value in configuration.items()),
            styles["Normal"],
        ),
        Spacer(1, 6),
        Paragraph(
            "The ridge/valley metrics compare the final binary ridge map with an independently "
            "thresholded reference derived from the same scan. They demonstrate filter behaviour "
            "and are not a measurement of biometric identity accuracy.",
            styles["Normal"],
        ),
        Spacer(1, 10),
        Paragraph("2. Metric Summary Table", styles["Heading2"]),
    ]

    metrics = analytics["metrics"]
    metric_table = Table(
        [
            ["Algorithm", "Accuracy", "Precision", "Recall", "F1", "Time (ms)"],
            [
                "Bilateral Edge-Preserving Denoising",
                f"{metrics['Accuracy']:.3f}",
                f"{metrics['Precision']:.3f}",
                f"{metrics['Recall']:.3f}",
                f"{metrics['F1']:.3f}",
                f"{elapsed_ms:.0f}",
            ],
        ],
        colWidths=[6.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 1.6 * cm, 2.1 * cm],
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), colors.whitesmoke),
            ]
        )
    )
    story.extend(
        [
            metric_table,
            Spacer(1, 12),
            Paragraph("3. Algorithm Performed: <b>Bilateral Edge-Preserving Denoising</b>", styles["Heading2"]),
            Paragraph(
                "OpenCV bilateral filtering reduces local camera noise while retaining the narrow "
                "intensity boundaries that represent fingerprint ridges.",
                styles["Normal"],
            ),
            PageBreak(),
            Paragraph("4. Confusion Matrix (Ridge vs Valley)", styles["Heading2"]),
        ]
    )

    matrix = analytics["confusion_matrix"]
    confusion = Table(
        [
            ["", "Pred: Valley (0)", "Pred: Ridge (1)"],
            ["Actual: Valley (0)", f"{int(matrix[0, 0]):,}", f"{int(matrix[0, 1]):,}"],
            ["Actual: Ridge (1)", f"{int(matrix[1, 0]):,}", f"{int(matrix[1, 1]):,}"],
        ],
        colWidths=[5.2 * cm, 5.2 * cm, 5.2 * cm],
    )
    confusion.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.extend(
        [
            confusion,
            Spacer(1, 16),
            Paragraph("5. Classification Metric Comparison", styles["Heading2"]),
            _metric_chart(metrics),
            PageBreak(),
            Paragraph("6. Intensity Histograms", styles["Heading2"]),
            _histogram_chart(analytics),
        ]
    )
    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buffer.getvalue()
