"""One-to-many fingerprint identification over locally enrolled templates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from fingerprint_processing import MatchEvidence, PipelineResult, compare_fingerprints


DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_AMBIGUITY_MARGIN = 0.035
MINIMUM_CAPTURE_QUALITY = 0.24


@dataclass
class IdentificationResult:
    matched: bool
    reason: str
    student: dict[str, Any] | None
    evidence: MatchEvidence | None
    runner_up_similarity: float
    candidates_checked: int
    template_path: str | None


def identify_student(
    query: PipelineResult,
    templates: list[dict[str, Any]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> IdentificationResult:
    """Return the strongest unambiguous student match across all templates."""

    if query.quality["overall"] < MINIMUM_CAPTURE_QUALITY:
        return IdentificationResult(
            matched=False,
            reason="Capture quality is too low for a reliable decision. Please scan the finger again.",
            student=None,
            evidence=None,
            runner_up_similarity=0.0,
            candidates_checked=0,
            template_path=None,
        )

    best_by_student: dict[str, tuple[dict[str, Any], MatchEvidence]] = {}
    checked = 0
    for template in templates:
        enhanced = cv2.imread(str(template["image_path"]), cv2.IMREAD_GRAYSCALE)
        if enhanced is None:
            continue
        reference = None
        reference_path = template.get("reference_path")
        if reference_path:
            reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        checked += 1
        evidence = compare_fingerprints(query, reference, enhanced)
        student_id = str(template["student_id"])
        previous = best_by_student.get(student_id)
        if previous is None or evidence.similarity > previous[1].similarity:
            best_by_student[student_id] = (template, evidence)

    if not best_by_student:
        return IdentificationResult(
            matched=False,
            reason="No readable enrolled fingerprint templates are available.",
            student=None,
            evidence=None,
            runner_up_similarity=0.0,
            candidates_checked=checked,
            template_path=None,
        )

    ranked = sorted(best_by_student.values(), key=lambda item: item[1].similarity, reverse=True)
    best_template, best_evidence = ranked[0]
    runner_up = ranked[1][1].similarity if len(ranked) > 1 else 0.0

    if best_evidence.similarity < threshold:
        return IdentificationResult(
            matched=False,
            reason=f"No fingerprint exceeded the {threshold:.0%} acceptance threshold.",
            student=None,
            evidence=best_evidence,
            runner_up_similarity=runner_up,
            candidates_checked=checked,
            template_path=str(best_template["image_path"]),
        )

    if len(ranked) > 1 and best_evidence.similarity - runner_up < ambiguity_margin:
        return IdentificationResult(
            matched=False,
            reason="The two strongest candidates are too close. Please recapture the fingerprint.",
            student=None,
            evidence=best_evidence,
            runner_up_similarity=runner_up,
            candidates_checked=checked,
            template_path=str(best_template["image_path"]),
        )

    student = {
        key: best_template[key]
        for key in (
            "student_id",
            "full_name",
            "programme",
            "study_year",
            "tutorial_group",
            "email",
            "student_status",
            "finger_label",
        )
        if key in best_template
    }
    return IdentificationResult(
        matched=True,
        reason="The fingerprint passed both similarity and ambiguity checks.",
        student=student,
        evidence=best_evidence,
        runner_up_similarity=runner_up,
        candidates_checked=checked,
        template_path=str(best_template["image_path"]),
    )


def attendance_status(session: dict[str, Any], marked_at: datetime | None = None) -> str:
    """Classify a verified scan as Present or Late using the session grace period."""

    marked_at = marked_at or datetime.now().astimezone()
    starts_at = datetime.fromisoformat(str(session["starts_at"]))
    if starts_at.tzinfo is None:
        starts_at = starts_at.astimezone()
    deadline = starts_at + timedelta(minutes=int(session.get("grace_minutes", 15)))
    return "Present" if marked_at <= deadline else "Late"


def save_enhanced_template(result: PipelineResult, path: str | Path) -> Path:
    """Persist a normalised enhanced template, raising if OpenCV cannot write it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result.enhanced):
        raise OSError(f"Could not store fingerprint template at {path}")
    return path


def save_reference_capture(result: PipelineResult, path: str | Path) -> Path:
    """Persist the consistently sized enrolment upload as the match reference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result.prepared):
        raise OSError(f"Could not store canonical fingerprint reference at {path}")
    return path
