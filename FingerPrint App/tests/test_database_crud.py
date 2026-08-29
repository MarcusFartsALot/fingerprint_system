"""Focused tests for student/session administration without image dependencies."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from database import (
    attendance_records,
    create_session,
    delete_session,
    delete_student,
    enrol_student,
    get_session,
    get_student,
    initialise_database,
    list_sessions,
    list_templates,
    mark_attendance,
    update_session,
    update_student,
)


class DatabaseAdministrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "attendance.db"
        initialise_database(self.database)
        enrol_student(
            {
                "student_id": "S100",
                "full_name": "Test Student",
                "programme": "Software Engineering",
                "study_year": 2,
                "tutorial_group": "T1",
                "email": "student@example.test",
            },
            [
                {
                    "finger_label": "Left thumb",
                    "image_path": str(self.root / "template.png"),
                    "reference_path": str(self.root / "reference.png"),
                    "quality": 0.75,
                    "clarity": 0.72,
                    "minutiae_count": 30,
                }
            ],
            self.database,
        )
        self.session_id = create_session(
            {
                "course_code": "IMG100",
                "course_name": "Image Processing",
                "venue": "Lab 1",
                "lecturer": "Lecturer",
                "starts_at": datetime.now().astimezone(),
                "grace_minutes": 15,
            },
            self.database,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_student_edit_and_delete_cascade(self) -> None:
        update_student(
            "S100",
            {
                "full_name": "Updated Student",
                "programme": "Data Science",
                "study_year": 3,
                "tutorial_group": "T8",
                "email": "updated@example.test",
                "status": "Inactive",
            },
            self.database,
        )
        self.assertEqual(get_student("S100", self.database)["full_name"], "Updated Student")

        mark_attendance(self.session_id, "S100", "Present", 0.8, 0.7, 100, self.database)
        delete_student("S100", self.database)
        self.assertIsNone(get_student("S100", self.database))
        self.assertEqual(list_templates(self.database), [])
        self.assertEqual(attendance_records(database_path=self.database), [])

    def test_session_edit_and_delete_cascade(self) -> None:
        update_session(
            self.session_id,
            {
                "course_code": "IMG200",
                "course_name": "Advanced Imaging",
                "venue": "Lab 2",
                "lecturer": "Updated Lecturer",
                "starts_at": datetime.now().astimezone(),
                "grace_minutes": 20,
                "active": False,
            },
            self.database,
        )
        self.assertEqual(get_session(self.session_id, self.database)["course_code"], "IMG200")

        mark_attendance(self.session_id, "S100", "Late", 0.8, 0.7, 100, self.database)
        delete_session(self.session_id, self.database)
        self.assertEqual(list_sessions(database_path=self.database), [])
        self.assertEqual(attendance_records(database_path=self.database), [])


if __name__ == "__main__":
    unittest.main()
