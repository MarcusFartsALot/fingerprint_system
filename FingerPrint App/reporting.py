"""PDF export for class attendance registers."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#18231f")
MUTED = colors.HexColor("#60716a")
GREEN = colors.HexColor("#13a36f")
GREEN_DARK = colors.HexColor("#087754")
GREEN_PALE = colors.HexColor("#e9f8f1")
AMBER = colors.HexColor("#e39a21")
RED = colors.HexColor("#ce4d4d")
LINE = colors.HexColor("#d8e2dd")
PAPER = colors.HexColor("#f7faf8")


def _format_datetime(value: str | None) -> str:
    if not value:
        return "-"
    parsed = datetime.fromisoformat(str(value))
    return parsed.strftime("%d %b %Y, %H:%M")


def _footer(canvas, document) -> None:
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(LINE)
    canvas.line(14 * mm, 11 * mm, width - 14 * mm, 11 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(14 * mm, 7 * mm, "Fingerprint Attendance / locally generated report")
    canvas.drawRightString(width - 14 * mm, 7 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_attendance_pdf(session: dict[str, Any], roster: list[dict[str, Any]]) -> bytes:
    """Build a landscape A4 attendance register with summary and roster."""

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=16 * mm,
        title=f"{session['course_code']} Attendance Register",
        author="Student Fingerprint Attendance System",
    )
    base = getSampleStyleSheet()
    styles = {
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=GREEN_DARK,
            spaceAfter=3,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=INK,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=INK,
        ),
        "small_muted": ParagraphStyle(
            "SmallMuted",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
        ),
        "right": ParagraphStyle(
            "Right",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_RIGHT,
        ),
    }

    present = sum(row["attendance_status"] == "Present" for row in roster)
    late = sum(row["attendance_status"] == "Late" for row in roster)
    absent = sum(row["attendance_status"] == "Absent" for row in roster)
    attended = present + late
    rate = attended / len(roster) if roster else 0

    story = [
        Paragraph("BIOMETRIC ATTENDANCE REGISTER", styles["eyebrow"]),
        Paragraph(
            f"{session['course_code']} / {session['course_name']}",
            styles["title"],
        ),
    ]

    detail_data = [
        [
            Paragraph(f"<b>Session</b><br/>{_format_datetime(session['starts_at'])}", styles["body"]),
            Paragraph(f"<b>Venue</b><br/>{session['venue']}", styles["body"]),
            Paragraph(f"<b>Lecturer</b><br/>{session['lecturer']}", styles["body"]),
            Paragraph(f"<b>Grace period</b><br/>{session['grace_minutes']} minutes", styles["body"]),
            Paragraph(
                f"Generated<br/>{datetime.now().astimezone().strftime('%d %b %Y, %H:%M')}",
                styles["right"],
            ),
        ]
    ]
    details = Table(detail_data, colWidths=[53 * mm, 43 * mm, 48 * mm, 35 * mm, 66 * mm])
    details.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), PAPER),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([details, Spacer(1, 5 * mm)])

    metrics = [
        (str(len(roster)), "Enrolled students", PAPER),
        (str(present), "Present", GREEN_PALE),
        (str(late), "Late", colors.HexColor("#fff4df")),
        (str(absent), "Absent", colors.HexColor("#fcecec")),
        (f"{rate:.0%}", "Attendance rate", GREEN_PALE),
    ]
    metric_cells = [
        [
            Paragraph(
                f'<font size="17"><b>{value}</b></font><br/>'
                f'<font size="7.5" color="#60716a">{label}</font>',
                styles["body"],
            )
            for value, label, _ in metrics
        ]
    ]
    metric_table = Table(metric_cells, colWidths=[49 * mm] * len(metrics))
    metric_style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    for index, (_, _, colour) in enumerate(metrics):
        metric_style.append(("BACKGROUND", (index, 0), (index, 0), colour))
    metric_table.setStyle(TableStyle(metric_style))
    story.extend([metric_table, Spacer(1, 6 * mm)])

    table_data: list[list[Any]] = [
        ["#", "Student ID", "Student", "Programme", "Year / Group", "Status", "Marked at", "Match", "Quality"]
    ]
    for index, row in enumerate(roster, start=1):
        status = row["attendance_status"]
        table_data.append(
            [
                str(index),
                Paragraph(str(row["student_id"]), styles["small"]),
                Paragraph(str(row["full_name"]), styles["small"]),
                Paragraph(str(row["programme"]), styles["small_muted"]),
                f"Y{row['study_year']} / {row['tutorial_group']}",
                status,
                _format_datetime(row.get("marked_at")),
                f"{float(row['similarity']):.1%}" if row.get("similarity") is not None else "-",
                f"{float(row['capture_quality']):.1%}" if row.get("capture_quality") is not None else "-",
            ]
        )

    roster_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[9 * mm, 24 * mm, 37 * mm, 65 * mm, 26 * mm, 20 * mm, 37 * mm, 18 * mm, 18 * mm],
    )
    roster_style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (4, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index, row in enumerate(roster, start=1):
        status_colour = GREEN_DARK if row["attendance_status"] == "Present" else AMBER
        if row["attendance_status"] == "Absent":
            status_colour = RED
        roster_style.extend(
            [
                ("TEXTCOLOR", (5, row_index), (5, row_index), status_colour),
                ("FONTNAME", (5, row_index), (5, row_index), "Helvetica-Bold"),
            ]
        )
    roster_table.setStyle(TableStyle(roster_style))
    story.append(roster_table)

    story.extend(
        [
            Spacer(1, 5 * mm),
            KeepTogether(
                [
                    Paragraph("Verification method", styles["eyebrow"]),
                    Paragraph(
                        "Captures were processed locally using 11 x 11 local histogram equalization, "
                        "3 x 3 adaptive Wiener filtering, 13 x 13 local-mean binarization, "
                        "morphological thinning and binary ridge post-processing. "
                        "A variance-based foreground guard rejects scanner borders as a system extension. "
                        "Identity matching is a separate downstream function that compares canonical "
                        "enrolment references with SIFT-RANSAC, enhanced ORB, aligned structure and spectra. "
                        "A row records the decision made at scan time; it is not a forensic claim.",
                        styles["body"],
                    ),
                ]
            ),
        ]
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
