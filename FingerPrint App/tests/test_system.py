"""Regression tests for the core non-UI attendance workflow."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from database import (
    add_fingerprint_templates,
    attendance_records,
    create_session,
    enrol_student,
    initialise_database,
    list_templates,
    mark_attendance,
    session_roster,
)
from fingerprint_processing import generate_demo_fingerprint, process_fingerprint
from matching import (
    DEFAULT_MATCH_THRESHOLD,
    attendance_status,
    identify_student,
    save_enhanced_template,
    save_reference_capture,
)
from reporting import build_attendance_pdf


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.first_raw = generate_demo_fingerprint(1101)
        cls.second_raw = generate_demo_fingerprint(2202)
        cls.first = process_fingerprint(cls.first_raw)
        cls.second = process_fingerprint(cls.second_raw)
        cls.first_path = save_enhanced_template(cls.first, cls.root / "first.png")
        cls.second_path = save_enhanced_template(cls.second, cls.root / "second.png")
        cls.first_reference = save_reference_capture(cls.first, cls.root / "first_reference.png")
        cls.second_reference = save_reference_capture(cls.second, cls.root / "second_reference.png")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_pipeline_returns_complete_fixed_size_outputs(self) -> None:
        self.assertEqual(self.first.prepared.shape, (360, 360))
        self.assertEqual(self.first.enhanced.shape, (360, 360))
        self.assertEqual(self.first.binary.shape, (360, 360))
        self.assertEqual(self.first.thinned.shape, (360, 360))
        self.assertEqual(self.first.skeleton.shape, (360, 360))
        self.assertEqual(self.first.local_equalised.shape, (360, 360))
        self.assertEqual(self.first.wiener_filtered.shape, (360, 360))
        self.assertEqual(self.first.region_mask.shape, (360, 360))
        self.assertTrue(set(np.unique(self.first.region_mask)).issubset({0, 255}))
        outside_mask = self.first.region_mask == 0
        if np.any(outside_mask):
            self.assertTrue(np.all(self.first.enhanced[outside_mask] == 255))
        self.assertTrue(set(np.unique(self.first.binary)).issubset({0, 255}))
        self.assertTrue(set(np.unique(self.first.thinned)).issubset({0, 255}))
        self.assertTrue(set(np.unique(self.first.skeleton)).issubset({0, 255}))
        self.assertGreater(self.first.quality["overall"], 0.20)
        self.assertEqual(
            self.first.stages,
            [
                "Local histogram equalization (11 x 11)",
                "Adaptive Wiener filtering (3 x 3)",
                "Adaptive local-mean binarization (13 x 13)",
                "Morphological thinning",
                "Binary ridge post-processing",
            ],
        )
        removed_outputs = [
            "normalised",
            "orientation_raw",
            "orientation",
            "frequency_map",
            "raw_binary",
            "initial_skeleton",
            "minutiae",
            "bounding_box",
            "calibration",
            "degradation",
        ]
        self.assertFalse(any(hasattr(self.first, name) for name in removed_outputs))

    def test_different_capture_conditions_use_the_same_classical_stages(self) -> None:
        blurred = process_fingerprint(cv2.GaussianBlur(self.first_raw, (15, 15), 4.0))
        noisy_raw = np.clip(
            self.first_raw.astype(np.float32)
            + np.random.default_rng(7).normal(0, 18, self.first_raw.shape),
            0,
            255,
        ).astype(np.uint8)
        noisy = process_fingerprint(noisy_raw)
        low_contrast_raw = np.clip(
            (self.first_raw.astype(np.float32) - float(self.first_raw.mean())) * 0.28 + 145,
            0,
            255,
        ).astype(np.uint8)
        low_contrast = process_fingerprint(low_contrast_raw)

        self.assertEqual(blurred.stages, self.first.stages)
        self.assertEqual(noisy.stages, self.first.stages)
        self.assertEqual(low_contrast.stages, self.first.stages)
        self.assertFalse(np.array_equal(blurred.local_equalised, blurred.enhanced))

    def test_genuine_query_ranks_correct_student(self) -> None:
        matrix = cv2.getRotationMatrix2D((170, 170), 2.0, 1.0)
        query = cv2.warpAffine(self.first_raw, matrix, (340, 340), borderValue=245)
        query = np.clip(
            query.astype(np.float32) + np.random.default_rng(99).normal(0, 2, query.shape),
            0,
            255,
        ).astype(np.uint8)
        result = process_fingerprint(query)
        templates = [
            {
                "student_id": "S1",
                "full_name": "Student One",
                "programme": "Software Engineering",
                "study_year": 3,
                "tutorial_group": "T1",
                "email": "",
                "student_status": "Active",
                "finger_label": "Right index",
                "image_path": str(self.first_path),
                "reference_path": str(self.first_reference),
            },
            {
                "student_id": "S2",
                "full_name": "Student Two",
                "programme": "Software Engineering",
                "study_year": 3,
                "tutorial_group": "T1",
                "email": "",
                "student_status": "Active",
                "finger_label": "Right index",
                "image_path": str(self.second_path),
                "reference_path": str(self.second_reference),
            },
        ]
        decision = identify_student(result, templates, threshold=DEFAULT_MATCH_THRESHOLD, ambiguity_margin=0.02)
        self.assertTrue(decision.matched, decision.reason)
        self.assertEqual(decision.student["student_id"], "S1")
        self.assertGreater(decision.evidence.similarity, decision.runner_up_similarity)
        self.assertTrue(decision.evidence.used_canonical_reference)

    def test_different_fingerprint_is_rejected(self) -> None:
        query = process_fingerprint(generate_demo_fingerprint(9933))
        templates = [
            {
                "student_id": "S1",
                "full_name": "Student One",
                "programme": "Software Engineering",
                "study_year": 3,
                "tutorial_group": "T1",
                "email": "",
                "student_status": "Active",
                "finger_label": "Right index",
                "image_path": str(self.first_path),
                "reference_path": str(self.first_reference),
            }
        ]
        decision = identify_student(query, templates, threshold=DEFAULT_MATCH_THRESHOLD)
        self.assertFalse(decision.matched)


class PersistenceAndReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "test.db"
        initialise_database(self.database)
        enrol_student(
            {
                "student_id": "S100",
                "full_name": "Test Student",
                "programme": "Bachelor of Software Engineering (Honours)",
                "study_year": 3,
                "tutorial_group": "T6",
                "email": "student@example.test",
            },
            [
                {
                    "finger_label": "Right index",
                    "image_path": str(self.root / "template.png"),
                    "reference_path": str(self.root / "reference.png"),
                    "quality": 0.75,
                    "clarity": 0.72,
                    "minutiae_count": 42,
                    "profile": {"primary_issue": "balanced capture"},
                }
            ],
            self.database,
        )
        self.session_id = create_session(
            {
                "course_code": "BMDS2133",
                "course_name": "Image Processing",
                "venue": "Lab 1",
                "lecturer": "Test Lecturer",
                "starts_at": datetime.now().astimezone(),
                "grace_minutes": 15,
            },
            self.database,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_duplicate_attendance_is_prevented(self) -> None:
        created, _ = mark_attendance(
            self.session_id, "S100", "Present", 0.82, 0.71, 110, self.database
        )
        duplicate, existing = mark_attendance(
            self.session_id, "S100", "Present", 0.90, 0.80, 95, self.database
        )
        self.assertTrue(created)
        self.assertFalse(duplicate)
        self.assertEqual(existing["student_id"], "S100")
        self.assertEqual(len(attendance_records(database_path=self.database)), 1)

    def test_enrolment_persists_canonical_reference_metadata(self) -> None:
        templates = list_templates(self.database)
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["reference_path"], str(self.root / "reference.png"))
        self.assertIn("balanced capture", templates[0]["profile_json"])

    def test_existing_student_can_receive_new_reference_without_data_loss(self) -> None:
        add_fingerprint_templates(
            "S100",
            [
                {
                    "finger_label": "Right index",
                    "image_path": str(self.root / "new_template.png"),
                    "reference_path": str(self.root / "new_reference.png"),
                    "quality": 0.81,
                    "clarity": 0.79,
                    "profile": {"stages": ["paper filtering"]},
                }
            ],
            self.database,
        )
        templates = list_templates(self.database)
        self.assertEqual(len(templates), 2)
        self.assertEqual(templates[-1]["reference_path"], str(self.root / "new_reference.png"))

    def test_roster_and_pdf_include_session_data(self) -> None:
        mark_attendance(self.session_id, "S100", "Present", 0.82, 0.71, 110, self.database)
        roster = session_roster(self.session_id, self.database)
        session = {
            "course_code": "BMDS2133",
            "course_name": "Image Processing",
            "venue": "Lab 1",
            "lecturer": "Test Lecturer",
            "starts_at": datetime.now().astimezone().isoformat(),
            "grace_minutes": 15,
        }
        pdf = build_attendance_pdf(session, roster)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 2_000)

    def test_attendance_status_respects_grace_period(self) -> None:
        starts = datetime.now().astimezone()
        session = {"starts_at": starts.isoformat(), "grace_minutes": 10}
        self.assertEqual(attendance_status(session, starts), "Present")
        self.assertEqual(
            attendance_status(session, starts + timedelta(minutes=11)),
            "Late",
        )

if __name__ == "__main__":
    unittest.main()
