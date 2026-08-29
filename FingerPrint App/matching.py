"""One-to-many fingerprint identification over locally enrolled templates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2

from fingerprint_processing import (
    PIPELINE_SCHEMA_VERSION,
    MatchEvidence,
    PipelineResult,
    compare_fingerprints,
)


DEFAULT_MATCH_THRESHOLD = 0.45
MATCHING_PIPELINE_SCHEMA_VERSION = PIPELINE_SCHEMA_VERSION
DEFAULT_AMBIGUITY_MARGIN = 0.035
MINIMUM_CAPTURE_QUALITY = 0.24
MINIMUM_RIDGE_COHERENCE = 0.18
MINIMUM_CAPTURE_SHARPNESS = 0.28
MINIMUM_REFERENCE_INLIERS = 12
MINIMUM_FOREGROUND_COVERAGE = 0.22


@dataclass
class IdentificationResult:
    matched: bool
    reason: str
    student: dict[str, Any] | None
    evidence: MatchEvidence | None
    runner_up_similarity: float
    candidates_checked: int
    template_path: str | None
    reference_path: str | None


def capture_quality_issue(result: PipelineResult) -> str | None:
    """Explain why a capture is unsafe to enrol or match, if applicable."""

    quality = result.quality
    if quality.get("overall", 0.0) < MINIMUM_CAPTURE_QUALITY:
        return "Capture quality is too low. Use brighter light, focus on the ridges and capture again."
    if quality.get("ridge_coherence", 1.0) < MINIMUM_RIDGE_COHERENCE:
        return "The ridge directions are not clear enough. Move closer, refocus and capture again."
    if quality.get("sharpness", 1.0) < MINIMUM_CAPTURE_SHARPNESS:
        return "The fingerprint is too blurred for reliable recognition. Hold the phone steady and capture again."
    coverage = quality.get("mask_coverage", 0.5)
    if coverage < MINIMUM_FOREGROUND_COVERAGE:
        return (
            "The fingerprint foreground is too small or incomplete. Avoid the 0.5x ultrawide camera; "
            "use 1x or macro focus and let the fingertip fill most of the capture guide."
        )
    if coverage > 0.94:
        return "The fingertip could not be separated from the background. Use a plain background and capture again."
    return None


def identify_student(
    query: PipelineResult,
    templates: list[dict[str, Any]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> IdentificationResult:
    """Return the strongest unambiguous student match across all templates."""

    quality_issue = capture_quality_issue(query)
    if quality_issue:
        return IdentificationResult(
            matched=False,
            reason=quality_issue,
            student=None,
            evidence=None,
            runner_up_similarity=0.0,
            candidates_checked=0,
            template_path=None,
            reference_path=None,
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
            reference_path=None,
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
            reference_path=str(best_template["reference_path"]) if best_template.get("reference_path") else None,
        )

    if best_evidence.used_canonical_reference and best_evidence.reference_inliers < MINIMUM_REFERENCE_INLIERS:
        return IdentificationResult(
            matched=False,
            reason=(
                "The score lacks enough geometrically consistent ridge features "
                "for a reliable identity decision. Please take a sharper capture."
            ),
            student=None,
            evidence=best_evidence,
            runner_up_similarity=runner_up,
            candidates_checked=checked,
            template_path=str(best_template["image_path"]),
            reference_path=str(best_template["reference_path"]) if best_template.get("reference_path") else None,
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
            reference_path=str(best_template["reference_path"]) if best_template.get("reference_path") else None,
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
        reference_path=str(best_template["reference_path"]) if best_template.get("reference_path") else None,
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
