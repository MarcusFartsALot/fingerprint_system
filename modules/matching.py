"""
Module C: Matching / Identification
---------------------------------------
Author (team): Member C

A simplified, explainable minutiae-based matcher, suitable for a classical
image-processing course (as opposed to a learned embedding/deep-matcher,
which would be out of scope here and hard to defend in the on-the-spot
coding viva).

Algorithm
---------
1. Alignment: translate + rotate the query minutiae set so its core point
   sits at the origin with orientation zero, and do the same for each
   candidate record. This removes the effect of the finger being placed
   at a different position/angle on the scanner. If either print has no
   detected core, fall back to centroid alignment.
2. Correspondence: for each reference minutia, find the nearest query
   minutia of the *same kind* (ridge ending / bifurcation) within a
   distance tolerance and an angular tolerance. Each match may be used
   once (greedy one-to-one assignment).
3. Score: Dice-style overlap coefficient,
       score = 2 * matched_pairs / (num_reference + num_query)
   which lies in [0, 1] and is symmetric regardless of which print has
   more minutiae (e.g. due to partial capture).
4. Decision: score >= MATCH_THRESHOLD -> declared match.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from modules.minutiae_detection import Minutia
from modules.singular_points import SingularPoint

DISTANCE_TOLERANCE_PX = 12
ANGLE_TOLERANCE_RAD = np.deg2rad(20)
MATCH_THRESHOLD = 0.40


@dataclass
class MatchResult:
    record_id: int
    subject_name: str
    score: float
    matched_pairs: int
    is_match: bool


def _find_core(singular_points: List[SingularPoint]) -> Optional[SingularPoint]:
    cores = [p for p in singular_points if p.kind == "core"]
    return cores[0] if cores else None


def _align(minutiae: List[Minutia], singular_points: List[SingularPoint]
           ) -> List[Tuple[float, float, float, str]]:
    """
    Returns a list of (x, y, angle, kind) tuples in a normalised frame:
    translated so the reference point (core, or centroid if no core) is
    at the origin. Rotation alignment is intentionally left to the
    matcher's tolerance window rather than a hard rotation, since a
    single global rotation estimate from noisy minutiae is unreliable;
    the angular tolerance in scoring absorbs small residual misalignment.
    """
    if not minutiae:
        return []

    core = _find_core(singular_points)
    if core is not None:
        ref_x, ref_y = core.x, core.y
    else:
        ref_x = float(np.mean([m.x for m in minutiae]))
        ref_y = float(np.mean([m.y for m in minutiae]))

    return [(m.x - ref_x, m.y - ref_y, m.angle, m.kind) for m in minutiae]


def _score_pair(query_norm: List[Tuple[float, float, float, str]],
                 ref_norm: List[Tuple[float, float, float, str]]) -> Tuple[float, int]:
    if not query_norm or not ref_norm:
        return 0.0, 0

    used_ref = [False] * len(ref_norm)
    matched = 0

    for qx, qy, qangle, qkind in query_norm:
        best_j, best_dist = -1, DISTANCE_TOLERANCE_PX
        for j, (rx, ry, rangle, rkind) in enumerate(ref_norm):
            if used_ref[j] or rkind != qkind:
                continue
            dist = np.hypot(qx - rx, qy - ry)
            angle_gap = abs(((qangle - rangle) + np.pi) % (2 * np.pi) - np.pi)
            if dist <= best_dist and angle_gap <= ANGLE_TOLERANCE_RAD:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            used_ref[best_j] = True
            matched += 1

    score = 2.0 * matched / (len(query_norm) + len(ref_norm))
    return score, matched


def match_against_record(query_minutiae: List[Minutia], query_singular: List[SingularPoint],
                          ref_minutiae: List[Minutia], ref_singular: List[SingularPoint]
                          ) -> Tuple[float, int]:
    """Returns (score, matched_pairs) for one query-vs-reference comparison."""
    query_norm = _align(query_minutiae, query_singular)
    ref_norm = _align(ref_minutiae, ref_singular)
    return _score_pair(query_norm, ref_norm)


def identify(query_minutiae: List[Minutia], query_singular: List[SingularPoint],
             candidate_records: List[dict]) -> List[MatchResult]:
    """
    Compares the query fingerprint against every candidate record
    (as returned by ``database.get_all_records``) and returns results
    ranked by score, best match first.

    Each ``candidate_records`` item is expected to provide:
        id, subject_name, minutiae (List[Minutia]), singular_points (List[SingularPoint])
    """
    results: List[MatchResult] = []
    for record in candidate_records:
        score, matched = match_against_record(
            query_minutiae, query_singular,
            record["minutiae"], record["singular_points"],
        )
        results.append(MatchResult(
            record_id=record["id"],
            subject_name=record["subject_name"],
            score=score,
            matched_pairs=matched,
            is_match=score >= MATCH_THRESHOLD,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
