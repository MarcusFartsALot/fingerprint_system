"""
Module C: Pattern Classification (Arch / Loop / Whorl)
----------------------------------------------------------
Author (team): Member C

Implements the classical Galton-Henry classification rule directly from
the singular points produced by Module B — no training data required,
fully deterministic and explainable (important for the "on-the-spot
coding" viva component of the rubric):

    0 cores, 0 deltas                -> Arch
    1 core,  1 delta                 -> Loop  (Left/Right via relative
                                                 core-delta position)
    2 cores, 2 deltas                -> Whorl
    anything else / ambiguous count  -> Undetermined (flagged for review)

A secondary heuristic distinguishes Left Loop vs Right Loop using the
horizontal offset between the core and its associated delta, which is
the standard manual-classification rule used by forensic examiners.
"""

from dataclasses import dataclass
from typing import List

from modules.singular_points import SingularPoint

PATTERN_ARCH = "Arch"
PATTERN_LOOP_LEFT = "Left Loop"
PATTERN_LOOP_RIGHT = "Right Loop"
PATTERN_WHORL = "Whorl"
PATTERN_UNDETERMINED = "Undetermined"


@dataclass
class ClassificationResult:
    pattern_type: str
    num_cores: int
    num_deltas: int
    confidence: str  # "high" / "medium" / "low" — based on how clean the singular-point count is
    explanation: str


def classify_pattern(singular_points: List[SingularPoint]) -> ClassificationResult:
    cores = [p for p in singular_points if p.kind == "core"]
    deltas = [p for p in singular_points if p.kind == "delta"]
    n_core, n_delta = len(cores), len(deltas)

    if n_core == 0 and n_delta == 0:
        return ClassificationResult(
            PATTERN_ARCH, n_core, n_delta, "high",
            "No core or delta detected — consistent with an arch pattern, "
            "where ridges flow continuously across the finger without "
            "forming a distinct triangular delta or looping core."
        )

    if n_core == 1 and n_delta == 1:
        core, delta = cores[0], deltas[0]
        if delta.x < core.x:
            pattern = PATTERN_LOOP_RIGHT
            side = "left of"
        else:
            pattern = PATTERN_LOOP_LEFT
            side = "right of"
        return ClassificationResult(
            pattern, n_core, n_delta, "high",
            f"Exactly one core and one delta detected, with the delta positioned "
            f"to the {side} the core — consistent with a {pattern.lower()} pattern."
        )

    if n_core >= 2 and n_delta >= 2:
        return ClassificationResult(
            PATTERN_WHORL, n_core, n_delta, "high",
            "Two or more core/delta pairs detected — consistent with a whorl "
            "pattern, where ridges form circular or spiral formations around "
            "multiple singular points."
        )

    # Ambiguous counts (e.g. 1 core / 0 delta from a partial print, or
    # 2 cores / 1 delta from a noisy image) — still report a best guess
    # but flag lower confidence so the UI can prompt for manual review.
    if n_core >= 1 and n_delta == 0:
        return ClassificationResult(
            PATTERN_LOOP_LEFT if n_core == 1 else PATTERN_WHORL,
            n_core, n_delta, "low",
            "Core(s) detected without a matching delta — likely a partial "
            "capture or low print quality; classification is a best estimate."
        )

    return ClassificationResult(
        PATTERN_UNDETERMINED, n_core, n_delta, "low",
        f"Detected {n_core} core(s) and {n_delta} delta(s), which does not "
        "cleanly match a standard arch/loop/whorl configuration. Manual "
        "review recommended."
    )
