"""
pdf_report.py
-------------
Builds a downloadable PDF summarising the algorithm-comparison benchmark:
configuration, metric table, best-algorithm verdict, and all charts
(sample gallery, confusion matrices, metric bars, quality/clarity bars,
histograms).

The metric table is passed in pre-built (header + rows) rather than
recomputed here, so the same report builder works whether the benchmark
was run in "simulate" mode (PSNR/SSIM available) or "direct" mode
(no-reference clarity score instead) — see evaluation.py / app.py.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)


def _fig_to_rlimage(fig, max_width_cm=17):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    w_in, h_in = fig.get_size_inches()
    aspect = h_in / w_in
    width = max_width_cm * cm
    height = width * aspect
    return RLImage(buf, width=width, height=height)


def build_pdf_report(table_header, table_rows, best_algo, figures, params, methodology_note):
    """
    table_header : list[str]                 e.g. ["Algorithm","Accuracy",...]
    table_rows   : list[list[str]]            one row per algorithm
    best_algo    : str
    figures      : dict {section_key: matplotlib Figure}
                   recognised keys: 'gallery','confusion','metric_bars',
                   'quality_bars' (PSNR/SSIM OR clarity), 'histograms'
    params       : dict of benchmark configuration values to display
    methodology_note : str, explains which mode/assumptions were used
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                             leftMargin=1.4 * cm, rightMargin=1.4 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Fingerprint Enhancement System", styles["Title"]))
    story.append(Paragraph("Algorithm Comparison Report — BMDS2133 Image Processing (Mode A)",
                            styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Test Configuration", styles["Heading2"]))
    cfg_text = "<br/>".join(f"<b>{k}:</b> {v}" for k, v in params.items())
    story.append(Paragraph(cfg_text, styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(methodology_note, styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Metric Summary Table", styles["Heading2"]))
    table_data = [table_header] + table_rows
    t = Table(table_data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"3. Best Performing Algorithm: <b>{best_algo}</b>", styles["Heading2"]))
    story.append(Paragraph(
        "Ranked by average F1-score across ridge/valley pixel classification against the "
        "pseudo ground-truth mask, averaged over all benchmarked samples.", styles["Normal"]))
    story.append(PageBreak())

    sections = [
        ("4. Sample Gallery", "gallery"),
        ("5. Confusion Matrices (Ridge vs Valley)", "confusion"),
        ("6. Classification Metric Comparison", "metric_bars"),
        ("7. Quality / Clarity Metrics", "quality_bars"),
        ("8. Intensity Histograms", "histograms"),
    ]
    for title, key in sections:
        if key not in figures:
            continue
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(_fig_to_rlimage(figures[key]))
        story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf